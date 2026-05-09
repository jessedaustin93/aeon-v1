"""Designated write agent for Aeon-V1.

DataWriteAgent is the SOLE component authorized to write to the memory
system within an agent execution context. All other agents must request
writes through the message bus — direct calls to ingest(), reflect(),
simulate_action(), evaluate_simulation(), or select_next_task() from
within an agent run are blocked by the write guard and raise
WriteAuthorizationError.

Every write request must pass through a human (or hardware-device)
approval gate before execution. The gate uses the pluggable AuthProvider
interface from approval_agent.py — swap CLIAuthProvider for a hardware
token or TOTP device by passing auth_provider= to the constructor or
Orchestrator(auth_provider=...).

Bus topics handled
------------------
    data.write.ingest       payload: {text, source}
    data.write.reflect      payload: {}
    data.write.simulate     payload: {task}
    data.write.evaluate     payload: {simulation, result_text}
    data.write.select_task  payload: {}
    data.write.consolidate  payload: {}

The agent is instantiated by the Orchestrator on startup and lives for
the duration of the orchestrator's lifetime. Call close() to cleanly
unsubscribe from all topics.
"""
from typing import Dict, Optional, Tuple

from .approval_agent import AuthProvider, CLIAuthProvider
from .bus import get_bus
from .config import Config
from .consolidate import consolidate_memories
from .decision import select_next_task
from .evaluate import evaluate_simulation
from .ingest import ingest
from .memory_store import _generate_id
from .reflect import reflect
from .security import AuditLog
from .simulate import simulate_action
from .write_guard import write_authorized_context

_AGENT_NAME = "DataWriteAgent"


class DataWriteAgent:
    """Sole authorized writer for the Aeon-V1 memory system.

    Every incoming write request is presented to a human operator (or
    hardware auth device) via AuthProvider.request_approval() before any
    data is written. Approved writes enter write_authorized_context() so
    the write guard passes. Rejected writes return immediately with a
    rejection result and are logged to the audit trail.

    Args:
        config:        Config instance shared with the Orchestrator.
        auth_provider: Approval gate implementation. Defaults to
                       CLIAuthProvider (stdin yes/no). Replace with a
                       hardware token or TOTP provider for production.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        auth_provider: Optional[AuthProvider] = None,
    ) -> None:
        self._config = config or Config()
        self._auth   = auth_provider or CLIAuthProvider()
        self._audit  = AuditLog(self._config)

        bus = get_bus()
        bus.subscribe("data.write.ingest",      self._handle_ingest)
        bus.subscribe("data.write.reflect",     self._handle_reflect)
        bus.subscribe("data.write.simulate",    self._handle_simulate)
        bus.subscribe("data.write.evaluate",    self._handle_evaluate)
        bus.subscribe("data.write.select_task", self._handle_select_task)
        bus.subscribe("data.write.consolidate", self._handle_consolidate)

    def close(self) -> None:
        """Unsubscribe from all write topics — call when the orchestrator shuts down."""
        bus = get_bus()
        bus.unsubscribe("data.write.ingest",      self._handle_ingest)
        bus.unsubscribe("data.write.reflect",     self._handle_reflect)
        bus.unsubscribe("data.write.simulate",    self._handle_simulate)
        bus.unsubscribe("data.write.evaluate",    self._handle_evaluate)
        bus.unsubscribe("data.write.select_task", self._handle_select_task)
        bus.unsubscribe("data.write.consolidate", self._handle_consolidate)

    # ------------------------------------------------------------------ approval gate

    def _approve(self, operation: str, message: Dict, context: Dict) -> Tuple[bool, str]:
        """Present the write request to the auth provider and log the decision.

        Returns (approved: bool, reason: str).
        """
        trace_id    = message.get("trace_id", _generate_id())
        requested_by = message.get("agent_id", "unknown")

        context.setdefault("trace_id",    trace_id)
        context.setdefault("proposed_by", requested_by)

        prompt = (
            f"DataWriteAgent: approve {operation!r} "
            f"requested by agent {requested_by!r}?"
        )
        approved, reason = self._auth.request_approval(prompt, context)

        self._audit.append(
            trace_id  = trace_id,
            agent     = _AGENT_NAME,
            action    = operation,
            result    = f"approved via {self._auth.provider_name()}" if approved
                        else f"rejected via {self._auth.provider_name()}: {reason}",
        )
        return approved, reason

    # ------------------------------------------------------------------ handlers

    def _handle_ingest(self, message: Dict) -> Dict:
        payload = message.get("payload", {})
        text    = payload.get("text", "")
        source  = payload.get("source", "agent")

        approved, reason = self._approve(
            "ingest", message,
            {
                "type":       "ingest",
                "content":    f"[source={source}] {text[:300]}",
                "confidence": 1.0,
            },
        )
        if not approved:
            return {"raw": None, "episodic": None, "semantic": None,
                    "rejected": True, "reason": reason}

        with write_authorized_context():
            return ingest(text=text, source=source, config=self._config)

    def _handle_reflect(self, message: Dict) -> Dict:
        approved, reason = self._approve(
            "reflect", message,
            {
                "type":       "reflect",
                "content":    "Run a reflection pass on all episodic and semantic memories.",
                "confidence": 1.0,
            },
        )
        if not approved:
            return {"reflection": None, "message": f"rejected: {reason}",
                    "tasks_created": [], "rejected": True}

        with write_authorized_context():
            return reflect(config=self._config)

    def _handle_simulate(self, message: Dict) -> Dict:
        payload = message.get("payload", {})
        task    = payload.get("task")
        if task is None:
            return {"simulation": None, "error": "no task in payload"}

        approved, reason = self._approve(
            "simulate_action", message,
            {
                "type":       "simulate",
                "content":    (
                    f"Simulate action for task: {task.get('title', task.get('id', '?'))}\n"
                    f"{task.get('description', '')[:200]}"
                ),
                "confidence": task.get("confidence", 1.0),
            },
        )
        if not approved:
            return {"simulation": None, "rejected": True, "reason": reason}

        with write_authorized_context():
            return simulate_action(task, config=self._config)

    def _handle_evaluate(self, message: Dict) -> Dict:
        payload     = message.get("payload", {})
        simulation  = payload.get("simulation")
        result_text = payload.get("result_text", "")
        if simulation is None:
            return {"error": "no simulation in payload"}

        approved, reason = self._approve(
            "evaluate_simulation", message,
            {
                "type":       "evaluate",
                "content":    (
                    f"Evaluate simulation {simulation.get('id', '?')} "
                    f"for task: {simulation.get('task_title', '?')}\n"
                    f"Result: {result_text[:200]}"
                ),
                "confidence": 1.0,
            },
        )
        if not approved:
            return {"evaluation": None, "rejected": True, "reason": reason}

        with write_authorized_context():
            return evaluate_simulation(simulation, result_text, config=self._config)

    def _handle_select_task(self, message: Dict) -> Dict:
        approved, reason = self._approve(
            "select_next_task", message,
            {
                "type":       "select_task",
                "content":    "Select and mark the highest-priority pending task as selected.",
                "confidence": 1.0,
            },
        )
        if not approved:
            return {"decision": None, "task": None,
                    "message": f"rejected: {reason}", "rejected": True}

        with write_authorized_context():
            return select_next_task(config=self._config)

    def _handle_consolidate(self, message: Dict) -> Dict:
        approved, reason = self._approve(
            "consolidate_memories", message,
            {
                "type": "consolidate",
                "content": (
                    "Find likely duplicate episodic/semantic memories and append "
                    "consensus records. Source memories are never deleted or rewritten."
                ),
                "confidence": 1.0,
            },
        )
        if not approved:
            return {"created": [], "created_count": 0,
                    "message": f"rejected: {reason}", "rejected": True}

        with write_authorized_context():
            return consolidate_memories(config=self._config)
