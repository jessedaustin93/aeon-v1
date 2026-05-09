"""Orchestrator for Aeon-V1 Layer 6.

The Orchestrator manages a pool of persistent AgentNodes and spawns
task-specific agents on demand. From the outside it is the single
entry-point that makes the swarm appear as one cohesive entity.

Design contract
===============
- Maintains up to config.max_thinking_agents "thinker" nodes in the pool.
- Spawns one "executor" agent per task selected for simulation.
- Spawns one "evaluator" agent per simulation awaiting feedback.
- A "monitor" agent is kept alive to watch memory growth and trigger reflections.
- A "consolidator" agent is kept alive to append consensus records for duplicates.
- Dissolved agents are removed from the pool; their records remain on disk.
- No subprocess, os.system, or execution primitive is ever called.
- vault/core/ is never written.
"""
import json
from typing import Dict, List, Optional

from .agent import AGENT_ROLES, AgentNode
from .approval_agent import AuthProvider
from .bus import get_bus
from .config import Config
from .data_write_agent import DataWriteAgent
from .memory_store import _generate_id
from .schemas import make_agent_message
from .simulate import SimulationStore
from .tasks import TaskStore
from .time_utils import utc_now_iso


class Orchestrator:
    """Manages a pool of AgentNodes and coordinates a single work cycle.

    Args:
        config: Optional Config. Defaults to Config().
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        auth_provider: Optional[AuthProvider] = None,
    ) -> None:
        self.config = config or Config()
        self._pool: Dict[str, AgentNode] = {}   # id → AgentNode (live pool only)
        self._write_agent = DataWriteAgent(self.config, auth_provider=auth_provider)
        self._ensure_dirs()
        self._load_pool()

    # ----------------------------------------------------------------- public API

    def tick(self) -> Dict:
        """Run one full orchestration cycle.

        Cycle order:
          1. Ensure monitor agent exists and run it.
          2. Ensure thinking pool is full; run each thinker once.
          3. Ensure consolidator agent exists and run it.
          4. Spawn and run executor agents for pending tasks (up to one per tick).
          5. Spawn and run evaluator for the oldest unreviewed simulation.
          6. Dissolve any agents whose status is already "dissolved" externally.
          7. Return a structured summary.

        This method is intentionally synchronous — swarm parallelism is
        achieved by calling tick() from multiple threads or processes in the
        caller, not inside this module.
        """
        summary: Dict = {
            "tick_at":    utc_now_iso(),
            "monitor":    None,
            "thinkers":   [],
            "consolidator": None,
            "executor":   None,
            "evaluator":  None,
        }

        bus = get_bus()
        now = utc_now_iso()

        # 1. Monitor
        monitor = self._ensure_monitor()
        if monitor.status == "idle":
            summary["monitor"] = bus.request(
                f"agent.run.{monitor.id}",
                make_agent_message(
                    agent_id="orchestrator",
                    action="run",
                    target=monitor.id,
                    payload={},
                    status="pending",
                    timestamp=now,
                    requires_approval=False,
                ),
            )

        # 2. Thinkers
        self._fill_thinker_pool()
        for node in self._pool_by_role("thinker"):
            if node.status == "idle":
                result = bus.request(
                    f"agent.run.{node.id}",
                    make_agent_message(
                        agent_id="orchestrator",
                        action="run",
                        target=node.id,
                        payload={},
                        status="pending",
                        timestamp=now,
                        requires_approval=False,
                    ),
                )
                summary["thinkers"].append(result)

        # 3. Consolidator
        consolidator = self._ensure_consolidator()
        if consolidator.status == "idle":
            summary["consolidator"] = bus.request(
                f"agent.run.{consolidator.id}",
                make_agent_message(
                    agent_id="orchestrator",
                    action="run",
                    target=consolidator.id,
                    payload={},
                    status="pending",
                    timestamp=now,
                    requires_approval=False,
                ),
            )

        # 4. Executor — one pending task per tick
        pending = TaskStore(self.config).list_tasks(status="pending")
        if pending:
            task = pending[0]
            executor = self.spawn("executor", tags=["auto"])
            summary["executor"] = bus.request(
                f"agent.run.{executor.id}",
                make_agent_message(
                    agent_id="orchestrator",
                    action="run",
                    target=executor.id,
                    payload={"task": task},
                    status="pending",
                    timestamp=now,
                    requires_approval=False,
                ),
            )
            executor.dissolve()
            self._remove_dissolved()

        # 5. Evaluator — oldest unreviewed simulation
        sim_store = SimulationStore(self.config)
        unreviewed = [s for s in sim_store.list_simulations() if s.get("feedback") == "unknown"]
        if unreviewed:
            evaluator = self.spawn("evaluator", tags=["auto"])
            summary["evaluator"] = bus.request(
                f"agent.run.{evaluator.id}",
                make_agent_message(
                    agent_id="orchestrator",
                    action="run",
                    target=evaluator.id,
                    payload={"simulation_id": unreviewed[0]["id"]},
                    status="pending",
                    timestamp=now,
                    requires_approval=False,
                ),
            )
            evaluator.dissolve()
            self._remove_dissolved()

        return summary

    def spawn(
        self,
        role: str,
        role_description: str = "",
        tags: Optional[List[str]] = None,
    ) -> AgentNode:
        """Create a new AgentNode, add it to the pool, and return it."""
        node = AgentNode(
            role=role,
            config=self.config,
            role_description=role_description,
            tags=tags or [],
        )
        self._pool[node.id] = node
        self._persist_manifest()
        return node

    def dissolve(self, agent_id: str) -> bool:
        """Dissolve an agent by ID. Returns True if the agent was found."""
        node = self._pool.get(agent_id)
        if node is None:
            node = AgentNode.load(agent_id, self.config)
        if node is None:
            return False
        if node.status != "dissolved":
            node.dissolve()
        self._pool.pop(agent_id, None)
        self._persist_manifest()
        return True

    def list_agents(self, role: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """Return dicts for all agents in the live pool (optionally filtered)."""
        agents = list(self._pool.values())
        if role:
            agents = [a for a in agents if a.role == role]
        if status:
            agents = [a for a in agents if a.status == status]
        return [a.to_dict() for a in agents]

    def pool_size(self) -> int:
        return len(self._pool)

    # ----------------------------------------------------------------- internals

    def _ensure_monitor(self) -> AgentNode:
        monitors = self._pool_by_role("monitor")
        if monitors:
            return monitors[0]
        return self.spawn("monitor", role_description="memory growth watchdog")

    def _ensure_consolidator(self) -> AgentNode:
        consolidators = self._pool_by_role("consolidator")
        if consolidators:
            return consolidators[0]
        return self.spawn("consolidator", role_description="duplicate memory consolidation agent")

    def _fill_thinker_pool(self) -> None:
        thinkers = self._pool_by_role("thinker")
        needed = self.config.max_thinking_agents - len(thinkers)
        for _ in range(needed):
            self.spawn("thinker", role_description="recursive reflection agent")

    def _pool_by_role(self, role: str) -> List[AgentNode]:
        return [n for n in self._pool.values() if n.role == role and n.status != "dissolved"]

    def _remove_dissolved(self) -> None:
        dissolved = [aid for aid, n in self._pool.items() if n.status == "dissolved"]
        for aid in dissolved:
            del self._pool[aid]
        if dissolved:
            self._persist_manifest()

    # ----------------------------------------------------------------- persistence

    def _ensure_dirs(self) -> None:
        (self.config.memory_path / "orchestrator").mkdir(parents=True, exist_ok=True)

    def _persist_manifest(self) -> None:
        manifest = {
            "updated_at":   utc_now_iso(),
            "pool_size":    len(self._pool),
            "agents":       [n.to_dict() for n in self._pool.values()],
        }
        path = self.config.memory_path / "orchestrator" / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_pool(self) -> None:
        """Reload live (non-dissolved) agents from the persisted manifest."""
        path = self.config.memory_path / "orchestrator" / "manifest.json"
        if not path.exists():
            return
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        for record in manifest.get("agents", []):
            if record.get("status") == "dissolved":
                continue
            node = AgentNode.load(record["id"], self.config)
            if node is not None and node.status != "dissolved":
                # Reset running state — we can't resume mid-run after restart.
                if node.status == "running":
                    node.status = "idle"
                    node._persist()
                self._pool[node.id] = node
