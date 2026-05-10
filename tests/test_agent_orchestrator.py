"""Tests for Layer 6: AgentNode and Orchestrator."""
import json
import pytest
from pathlib import Path

from aeon_v1 import AgentNode, AGENT_ROLES, Config, Orchestrator, ingest
from aeon_v1.memory_store import _vault_note_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg(tmp_path):
    c = Config(tmp_path)
    c.max_thinking_agents = 2
    c.reflection_interval = 5
    c.allow_low_value_reflections = True
    c.min_reflection_sources = 1
    c.skip_duplicate_reflections = False
    c.require_human_approval_for_simulation = False
    return c


# ---------------------------------------------------------------------------
# AgentNode — construction and validation
# ---------------------------------------------------------------------------

def test_agent_roles_are_defined():
    assert "thinker" in AGENT_ROLES
    assert "executor" in AGENT_ROLES
    assert "monitor" in AGENT_ROLES
    assert "evaluator" in AGENT_ROLES
    assert "consolidator" in AGENT_ROLES
    assert "custom" in AGENT_ROLES


def test_agent_unknown_role_raises(cfg):
    with pytest.raises(ValueError, match="Unknown agent role"):
        AgentNode(role="wizard", config=cfg)


def test_custom_role_requires_description(cfg):
    with pytest.raises(ValueError, match="role_description is required"):
        AgentNode(role="custom", config=cfg)


def test_custom_role_with_description(cfg):
    node = AgentNode(role="custom", config=cfg, role_description="summariser")
    assert node.role == "custom"
    assert node.role_description == "summariser"


def test_agent_starts_idle(cfg):
    node = AgentNode(role="thinker", config=cfg)
    assert node.status == "idle"


def test_agent_has_id(cfg):
    node = AgentNode(role="thinker", config=cfg)
    assert len(node.id) > 0


def test_agent_run_increments_count(cfg):
    node = AgentNode(role="monitor", config=cfg)
    assert node.run_count == 0
    node.run()
    assert node.run_count == 1
    node.run()
    assert node.run_count == 2


def test_agent_status_returns_to_idle_after_run(cfg):
    node = AgentNode(role="monitor", config=cfg)
    node.run()
    assert node.status == "idle"


def test_agent_dissolve_is_terminal(cfg):
    node = AgentNode(role="thinker", config=cfg)
    node.dissolve()
    assert node.status == "dissolved"
    with pytest.raises(RuntimeError):
        node.run()


def test_agent_cannot_dissolve_twice(cfg):
    node = AgentNode(role="thinker", config=cfg)
    node.dissolve()
    with pytest.raises(RuntimeError):
        node.dissolve()


# ---------------------------------------------------------------------------
# AgentNode — persistence
# ---------------------------------------------------------------------------

def test_agent_persists_json(cfg):
    node = AgentNode(role="monitor", config=cfg, tags=["test"])
    json_path = cfg.memory_path / "agents" / f"{node.id}.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["role"] == "monitor"
    assert "test" in data["tags"]


def test_agent_persists_markdown(cfg):
    node = AgentNode(role="thinker", config=cfg)
    md_path = _vault_note_path(cfg, "agents", node.id)
    assert md_path.exists()
    text = md_path.read_text()
    assert "thinker" in text


def test_agent_load_round_trip(cfg):
    node = AgentNode(role="monitor", config=cfg, tags=["round-trip"])
    node.run()
    loaded = AgentNode.load(node.id, cfg)
    assert loaded is not None
    assert loaded.id == node.id
    assert loaded.role == "monitor"
    assert loaded.run_count == 1
    assert "round-trip" in loaded.tags


def test_agent_load_returns_none_for_missing(cfg):
    assert AgentNode.load("nonexistent-id-xyz", cfg) is None


def test_json_updated_after_dissolve(cfg):
    node = AgentNode(role="thinker", config=cfg)
    node.dissolve()
    data = json.loads((cfg.memory_path / "agents" / f"{node.id}.json").read_text())
    assert data["status"] == "dissolved"


# ---------------------------------------------------------------------------
# AgentNode — role handlers
# ---------------------------------------------------------------------------

def test_monitor_role_returns_counts(cfg):
    node = AgentNode(role="monitor", config=cfg)
    result = node.run()
    assert result["role"] == "monitor"
    assert "episodic_count" in result
    assert "semantic_count" in result
    assert isinstance(result["reflection_trigger"], bool)


def test_thinker_role_returns_reflection_or_message(cfg):
    ingest(
        "I learned a critical key insight: reflection agent test memory.",
        config=cfg,
    )
    node = AgentNode(role="thinker", config=cfg)
    result = node.run()
    assert result["role"] == "thinker"
    assert "message" in result


def test_executor_role_no_tasks(cfg):
    node = AgentNode(role="executor", config=cfg)
    result = node.run()
    assert result["role"] == "executor"
    assert result["outcome"] == "no_pending_tasks"


def test_executor_role_with_task(cfg):
    ingest(
        "I learned a critical key insight: need to build the executor pipeline.",
        config=cfg,
    )
    from aeon_v1 import reflect
    reflect(cfg)

    from aeon_v1 import TaskStore
    tasks = TaskStore(cfg).list_tasks(status="pending")
    if not tasks:
        pytest.skip("No pending tasks created from reflection — skip.")

    node = AgentNode(role="executor", config=cfg)
    result = node.run(task=tasks[0])
    assert result["role"] == "executor"
    assert "simulation" in result


def test_evaluator_no_simulation(cfg):
    node = AgentNode(role="evaluator", config=cfg)
    result = node.run()
    assert result["role"] == "evaluator"
    assert result["outcome"] == "no_simulation_found"


def test_custom_role_run(cfg):
    node = AgentNode(role="custom", config=cfg, role_description="test-agent")
    result = node.run(foo="bar")
    assert result["role"] == "custom"
    assert result["description"] == "test-agent"


# ---------------------------------------------------------------------------
# Orchestrator — construction and pool management
# ---------------------------------------------------------------------------

def test_orchestrator_initialises(cfg):
    orch = Orchestrator(cfg)
    assert orch.pool_size() == 0  # fresh start, empty pool


def test_orchestrator_spawn_adds_to_pool(cfg):
    orch = Orchestrator(cfg)
    node = orch.spawn("thinker")
    assert orch.pool_size() == 1
    assert node.role == "thinker"


def test_orchestrator_dissolve_removes_from_pool(cfg):
    orch = Orchestrator(cfg)
    node = orch.spawn("thinker")
    assert orch.pool_size() == 1
    result = orch.dissolve(node.id)
    assert result is True
    assert orch.pool_size() == 0


def test_orchestrator_dissolve_missing_returns_false(cfg):
    orch = Orchestrator(cfg)
    assert orch.dissolve("nonexistent") is False


def test_orchestrator_list_agents_empty(cfg):
    orch = Orchestrator(cfg)
    assert orch.list_agents() == []


def test_orchestrator_list_agents_by_role(cfg):
    orch = Orchestrator(cfg)
    orch.spawn("thinker")
    orch.spawn("monitor")
    thinkers = orch.list_agents(role="thinker")
    assert len(thinkers) == 1
    assert thinkers[0]["role"] == "thinker"


def test_orchestrator_persists_manifest(cfg):
    orch = Orchestrator(cfg)
    orch.spawn("thinker")
    manifest_path = cfg.memory_path / "orchestrator" / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["pool_size"] == 1


def test_orchestrator_reload_from_manifest(cfg):
    orch = Orchestrator(cfg)
    node = orch.spawn("monitor")
    node_id = node.id

    orch2 = Orchestrator(cfg)
    agent_ids = [a["id"] for a in orch2.list_agents()]
    assert node_id in agent_ids


# ---------------------------------------------------------------------------
# Orchestrator — tick
# ---------------------------------------------------------------------------

def test_tick_returns_summary_keys(cfg):
    orch = Orchestrator(cfg)
    summary = orch.tick()
    assert "tick_at" in summary
    assert "monitor" in summary
    assert "thinkers" in summary
    assert "executor" in summary
    assert "consolidator" in summary
    assert "evaluator" in summary


def test_tick_creates_monitor_agent(cfg):
    orch = Orchestrator(cfg)
    orch.tick()
    monitors = orch.list_agents(role="monitor")
    assert len(monitors) == 1


def test_tick_fills_thinker_pool(cfg):
    cfg.max_thinking_agents = 2
    orch = Orchestrator(cfg)
    orch.tick()
    thinkers = orch.list_agents(role="thinker")
    assert len(thinkers) == 2


def test_tick_max_thinking_agents_respected(cfg):
    cfg.max_thinking_agents = 3
    orch = Orchestrator(cfg)
    orch.tick()
    orch.tick()  # second tick should not add more thinkers
    thinkers = orch.list_agents(role="thinker")
    assert len(thinkers) <= 3


def test_tick_no_pending_tasks_executor_none(cfg):
    orch = Orchestrator(cfg)
    summary = orch.tick()
    assert summary["executor"] is None


def test_tick_no_unreviewed_simulations_evaluator_none(cfg):
    orch = Orchestrator(cfg)
    summary = orch.tick()
    assert summary["evaluator"] is None


def test_tick_executor_runs_when_task_pending(cfg):
    ingest(
        "I learned a critical key insight: need to build the orchestrator pipeline.",
        config=cfg,
    )
    from aeon_v1 import reflect
    reflect(cfg)

    from aeon_v1 import TaskStore
    tasks = TaskStore(cfg).list_tasks(status="pending")
    if not tasks:
        pytest.skip("No pending tasks — skip executor tick test.")

    orch = Orchestrator(cfg)
    summary = orch.tick()
    assert summary["executor"] is not None
    assert summary["executor"]["role"] == "executor"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_max_thinking_agents_default():
    c = Config()
    assert c.max_thinking_agents == 10


def test_config_max_thinking_agents_settable():
    c = Config()
    c.max_thinking_agents = 3
    assert c.max_thinking_agents == 3


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------

def test_agent_node_exported():
    from aeon_v1 import AgentNode
    assert callable(AgentNode)


def test_orchestrator_exported():
    from aeon_v1 import Orchestrator
    assert callable(Orchestrator)


def test_agent_roles_exported():
    from aeon_v1 import AGENT_ROLES
    assert isinstance(AGENT_ROLES, set)
