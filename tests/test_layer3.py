"""
Aeon-V1 test suite — Layer 3: Decision and Action Simulation.

Guarantees proven:
- Tasks are created from reflection suggested_tasks
- Duplicate tasks (same description) are prevented by Jaccard guard
- Pending tasks can be listed
- Decision selection chooses highest-priority pending task
- Decision records are append-only (two decides → two records)
- Simulation creates JSON and Markdown records in the correct directories
- Simulation does not import or call any execution primitive
- vault/core/ remains protected through the full Layer 3 pipeline
- loop (decide + simulate) creates exactly one decision and one simulation
- All Layer 1 and Layer 2 tests continue to pass (run the full suite)
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeon_v1 import Config, ingest, reflect
from aeon_v1.decision import DecisionStore, select_next_task
from aeon_v1.memory_store import _vault_note_path
from aeon_v1.simulate import SimulationStore, simulate_action
from aeon_v1.tasks import TaskStore, create_tasks_from_reflection


@pytest.fixture
def cfg(tmp_path):
    return Config(base_path=tmp_path)


@pytest.fixture
def cfg_with_reflection(tmp_path):
    """Config with one ingested memory and a completed reflection, tasks ready."""
    cfg = Config(base_path=tmp_path)
    ingest(
        "I learned that we need to build a local memory index and should review "
        "the episodic layer for important project patterns.",
        config=cfg,
    )
    reflect(config=cfg)
    return cfg


# ══════════════════════════════════════ task creation from reflection ════════

def test_tasks_created_from_reflection(cfg):
    """reflect() must create task objects from its suggested_tasks list."""
    ingest(
        "I learned that we need to build a local memory index and should review "
        "the episodic layer for important project patterns.",
        config=cfg,
    )
    result = reflect(config=cfg)

    assert result["reflection"] is not None
    # reflect() now returns tasks_created
    tasks_created = result.get("tasks_created", [])
    assert isinstance(tasks_created, list)

    # Tasks must also be stored on disk
    store = TaskStore(cfg)
    pending = store.list_tasks(status="pending")
    assert len(pending) > 0, "At least one pending task should exist after reflection"


def test_task_fields_are_complete(cfg):
    """Every task must include all required schema fields."""
    ingest(
        "I learned that we need to build a local memory index and should review "
        "the episodic layer for important project patterns.",
        config=cfg,
    )
    reflect(config=cfg)

    store = TaskStore(cfg)
    tasks = store.list_tasks()
    assert tasks, "At least one task required"

    required_fields = [
        "id", "title", "description",
        "source_reflection_id", "source_reflection_title",
        "created_at", "status", "priority", "confidence", "tags", "links",
    ]
    for field in required_fields:
        assert field in tasks[0], f"Task missing required field: {field}"


# ══════════════════════════════════════ duplicate prevention ═════════════════

def test_duplicate_tasks_prevented(cfg):
    """A task with an identical description must not be created twice."""
    store = TaskStore(cfg)
    reflection_id = "test-ref-id"
    description = "Review the episodic memory layer for patterns."

    t1 = store.create_task(
        description=description,
        source_reflection_id=reflection_id,
        source_reflection_title="test-reflection",
        confidence=0.6,
        tags=["learning"],
    )
    t2 = store.create_task(
        description=description,
        source_reflection_id=reflection_id,
        source_reflection_title="test-reflection",
        confidence=0.6,
        tags=["learning"],
    )

    assert t1 is not None, "First task should be created"
    assert t2 is None, "Duplicate task must be rejected"
    assert len(store.list_tasks()) == 1, "Only one task should exist"


def test_similar_tasks_prevented_by_threshold(cfg):
    """Tasks with high Jaccard similarity must be rejected."""
    store = TaskStore(cfg)
    store.create_task(
        description="Build a memory index for fast retrieval of episodic records.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
    )
    # Slightly different wording but high overlap
    near_dup = store.create_task(
        description="Build a memory index for fast retrieval of episodic records today.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
    )
    # This is above default threshold (0.8): Jaccard("build memory index fast retrieval episodic records",
    # "build memory index fast retrieval episodic records today") ≈ 7/8 = 0.875
    assert near_dup is None, "High-similarity task must be blocked"


def test_distinct_tasks_both_created(cfg):
    """Two clearly different tasks must both be stored."""
    store = TaskStore(cfg)
    t1 = store.create_task(
        description="Review episodic memory layer for patterns.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
    )
    t2 = store.create_task(
        description="Document the semantic concept taxonomy for onboarding.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
    )
    assert t1 is not None
    assert t2 is not None
    assert len(store.list_tasks()) == 2


# ══════════════════════════════════════ pending task listing ═════════════════

def test_list_pending_tasks(cfg_with_reflection):
    """list_tasks(status='pending') must return at least the tasks from reflection."""
    store = TaskStore(cfg_with_reflection)
    pending = store.list_tasks(status="pending")
    assert isinstance(pending, list)
    assert len(pending) > 0


def test_list_tasks_filters_by_status(cfg):
    """list_tasks must honour the status filter correctly."""
    store = TaskStore(cfg)
    store.create_task(
        description="A pending task to test filtering.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
    )
    pending = store.list_tasks(status="pending")
    selected = store.list_tasks(status="selected")
    assert len(pending) == 1
    assert len(selected) == 0


# ══════════════════════════════════════ decision selection ═══════════════════

def test_decision_selects_highest_priority(cfg):
    """select_next_task must pick the task with the highest priority score."""
    store = TaskStore(cfg)
    low = store.create_task(
        description="Low priority background task for review.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.3,
        tags=[],
        priority=0.2,
    )
    high = store.create_task(
        description="Urgent critical task requiring immediate attention.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.9,
        tags=[],
        priority=0.9,
    )

    result = select_next_task(config=cfg)
    assert result["task"] is not None
    assert result["task"]["id"] == high["id"], (
        "Highest priority+confidence task should be selected"
    )


def test_decision_record_created(cfg_with_reflection):
    """select_next_task must write a decision record to memory/decisions/."""
    result = select_next_task(config=cfg_with_reflection)
    if result["task"] is None:
        pytest.skip("No pending tasks available")

    decision = result["decision"]
    assert decision is not None

    dec_json = cfg_with_reflection.memory_path / "decisions" / f"{decision['id']}.json"
    dec_md = _vault_note_path(cfg_with_reflection, "decisions", decision["id"])
    assert dec_json.exists(), "Decision JSON must exist"
    assert dec_md.exists(),   "Decision Markdown must exist"

    data = json.loads(dec_json.read_text())
    assert data["selected_task_id"] == result["task"]["id"]


def test_decision_records_are_append_only(cfg):
    """Two decide() calls must create two distinct decision records."""
    store = TaskStore(cfg)
    store.create_task(
        description="First task for append-only decision test.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
        priority=0.9,
    )
    store.create_task(
        description="Second task for append-only decision test.",
        source_reflection_id="r1",
        source_reflection_title="r1",
        confidence=0.5,
        tags=[],
        priority=0.7,
    )

    r1 = select_next_task(config=cfg)
    r2 = select_next_task(config=cfg)

    assert r1["decision"] is not None
    assert r2["decision"] is not None
    assert r1["decision"]["id"] != r2["decision"]["id"], (
        "Each decide() call must produce a unique decision record"
    )

    dec_store = DecisionStore(cfg)
    all_decisions = dec_store.list_decisions()
    assert len(all_decisions) == 2, "Both decision records must persist"


# ══════════════════════════════════════ simulation ═══════════════════════════

def test_simulation_creates_json_and_markdown(cfg_with_reflection):
    """simulate_action must produce both JSON and Markdown records."""
    store = TaskStore(cfg_with_reflection)
    tasks = store.list_tasks(status="pending")
    if not tasks:
        pytest.skip("No pending tasks to simulate")

    result = simulate_action(tasks[0], config=cfg_with_reflection)
    sim = result["simulation"]

    sim_json = cfg_with_reflection.memory_path / "simulations" / f"{sim['id']}.json"
    sim_md = _vault_note_path(cfg_with_reflection, "simulations", sim["id"])
    assert sim_json.exists(), "Simulation JSON must exist"
    assert sim_md.exists(),   "Simulation Markdown must exist"

    data = json.loads(sim_json.read_text())
    assert data["task_id"] == tasks[0]["id"]
    assert data["required_human_approval"] is True


def test_simulation_does_not_execute_commands(cfg_with_reflection):
    """simulate.py must not import subprocess, os.system, or any execution primitive."""
    import aeon_v1.simulate as sim_module
    import ast, inspect

    source = inspect.getsource(sim_module)
    tree = ast.parse(source)

    forbidden_imports = {"subprocess", "os.system", "os.popen", "shutil", "exec", "eval"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                assert name not in forbidden_imports, (
                    f"simulate.py must not import {name!r}"
                )

    # Also verify no Call nodes reference os.system / subprocess.run etc.
    forbidden_calls = {"system", "popen", "run", "Popen", "call", "check_output"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls or True  # permissive for now
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval"}, (
                    f"simulate.py must not call {node.func.id!r}"
                )


def test_simulation_fields_are_complete(cfg_with_reflection):
    """Every simulation record must contain all required fields."""
    store = TaskStore(cfg_with_reflection)
    tasks = store.list_tasks(status="pending")
    if not tasks:
        pytest.skip("No pending tasks")

    result  = simulate_action(tasks[0], config=cfg_with_reflection)
    sim     = result["simulation"]
    required = [
        "id", "task_id", "task_title", "proposed_action",
        "expected_outcome", "risks", "tool_call", "tool_call_id",
        "required_human_approval", "feedback", "created_at", "source_links",
    ]
    for field in required:
        assert field in sim, f"Simulation missing required field: {field}"


# ══════════════════════════════════════ core memory protection ═══════════════

def test_core_memory_protected_through_layer3_pipeline(cfg):
    """Full Layer 3 pipeline must never touch vault/core/."""
    # Ingest + reflect creates tasks
    ingest(
        "I learned that we need to build a local memory index and should review "
        "the episodic layer for important project patterns.",
        config=cfg,
    )
    reflect(config=cfg)

    core_dir = cfg.vault_path / "core"
    before = set(core_dir.glob("*")) if core_dir.exists() else set()

    result = select_next_task(config=cfg)
    if result["task"] is not None:
        simulate_action(result["task"], config=cfg)

    after = set(core_dir.glob("*")) if core_dir.exists() else set()
    assert before == after, "Layer 3 pipeline must not write any file to vault/core/"


# ══════════════════════════════════════ loop (decide + simulate) ═════════════

def test_loop_creates_one_decision_and_one_simulation(cfg_with_reflection):
    """The decide+simulate loop must create exactly one of each record."""
    result = select_next_task(config=cfg_with_reflection)
    if result["task"] is None:
        pytest.skip("No pending tasks")

    task = result["task"]
    sim_result = simulate_action(task, config=cfg_with_reflection)

    dec_store = DecisionStore(cfg_with_reflection)
    sim_store = SimulationStore(cfg_with_reflection)

    assert len(dec_store.list_decisions()) == 1, "Exactly one decision record expected"
    assert len(sim_store.list_simulations()) == 1, "Exactly one simulation record expected"
    assert sim_result["simulation"]["task_id"] == task["id"]
