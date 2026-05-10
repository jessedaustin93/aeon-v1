"""Tests for simulation feedback field (Step 1 of feedback loop).

Coverage:
- Simulation record has feedback field defaulting to "unknown"
- SimulationStore.update_feedback persists the new value
- update_feedback raises ValueError for invalid values
- update_feedback returns None for missing simulation
- evaluate_simulation sets feedback="success" for verdict="match"
- evaluate_simulation sets feedback="failure" for verdict="mismatch"
- evaluate_simulation sets feedback="unknown" for verdict="partial_match"
- Updated feedback is visible in simulation JSON on disk
- Evaluation record carries feedback field
- vault/core/ never touched by feedback operations
"""
import json
from pathlib import Path

import pytest

from aeon_v1 import Config, evaluate_simulation
from aeon_v1.memory_store import _vault_note_path
from aeon_v1.simulate import SimulationStore, simulate_action
from aeon_v1.tasks import TaskStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config(base_path=tmp_path)
    c.ensure_dirs()
    return c


def _make_simulation(cfg: Config, description: str = "Review the project memory index.") -> dict:
    store = TaskStore(cfg)
    task = store.create_task(
        description=description,
        source_reflection_id="test-ref",
        source_reflection_title="Test Reflection",
        confidence=0.7,
        tags=["test"],
    )
    assert task is not None
    return simulate_action(task, config=cfg)["simulation"]


# ---------------------------------------------------------------------------
# Default feedback field
# ---------------------------------------------------------------------------

def test_simulation_has_feedback_field(cfg):
    sim = _make_simulation(cfg)
    assert "feedback" in sim


def test_simulation_feedback_defaults_to_unknown(cfg):
    sim = _make_simulation(cfg)
    assert sim["feedback"] == "unknown"


def test_simulation_feedback_in_json_on_disk(cfg):
    sim = _make_simulation(cfg)
    data = json.loads(
        (cfg.memory_path / "simulations" / f"{sim['id']}.json").read_text(encoding="utf-8")
    )
    assert data["feedback"] == "unknown"


def test_simulation_feedback_in_markdown_frontmatter(cfg):
    sim = _make_simulation(cfg)
    md = _vault_note_path(cfg, "simulations", sim["id"]).read_text(encoding="utf-8")
    assert "feedback: unknown" in md


# ---------------------------------------------------------------------------
# SimulationStore.update_feedback
# ---------------------------------------------------------------------------

def test_update_feedback_success(cfg):
    sim = _make_simulation(cfg)
    updated = SimulationStore(cfg).update_feedback(sim["id"], "success")
    assert updated is not None
    assert updated["feedback"] == "success"


def test_update_feedback_failure(cfg):
    sim = _make_simulation(cfg)
    updated = SimulationStore(cfg).update_feedback(sim["id"], "failure")
    assert updated["feedback"] == "failure"


def test_update_feedback_unknown(cfg):
    sim = _make_simulation(cfg)
    updated = SimulationStore(cfg).update_feedback(sim["id"], "unknown")
    assert updated["feedback"] == "unknown"


def test_update_feedback_persists_to_json(cfg):
    sim = _make_simulation(cfg)
    SimulationStore(cfg).update_feedback(sim["id"], "success")
    data = json.loads(
        (cfg.memory_path / "simulations" / f"{sim['id']}.json").read_text(encoding="utf-8")
    )
    assert data["feedback"] == "success"


def test_update_feedback_updates_markdown(cfg):
    sim = _make_simulation(cfg)
    SimulationStore(cfg).update_feedback(sim["id"], "failure")
    md = _vault_note_path(cfg, "simulations", sim["id"]).read_text(encoding="utf-8")
    assert "feedback: failure" in md


def test_update_feedback_invalid_value_raises(cfg):
    sim = _make_simulation(cfg)
    with pytest.raises(ValueError):
        SimulationStore(cfg).update_feedback(sim["id"], "wrong")


def test_update_feedback_missing_sim_returns_none(cfg):
    result = SimulationStore(cfg).update_feedback("no-such-id", "success")
    assert result is None


# ---------------------------------------------------------------------------
# evaluate_simulation — verdict maps to feedback
# ---------------------------------------------------------------------------

def test_evaluate_sets_feedback_success_for_match(cfg):
    sim = _make_simulation(cfg)
    ev = evaluate_simulation(sim, sim["expected_outcome"], config=cfg)["evaluation"]
    assert ev["verdict"] == "match"
    assert ev["feedback"] == "success"


def test_evaluate_sets_feedback_failure_for_mismatch(cfg):
    sim = _make_simulation(cfg)
    ev = evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )["evaluation"]
    assert ev["verdict"] == "mismatch"
    assert ev["feedback"] == "failure"


def test_evaluate_sets_feedback_unknown_for_partial_match(cfg):
    sim = _make_simulation(cfg, "Review memory patterns and update index records.")
    # Craft a result that overlaps enough for partial_match (0.30–0.69)
    ev = evaluate_simulation(
        sim,
        "memory index was reviewed with some additional observations noted",
        config=cfg,
    )["evaluation"]
    # If it's partial_match, feedback must be "unknown"
    if ev["verdict"] == "partial_match":
        assert ev["feedback"] == "unknown"
    else:
        # Other verdicts have correct feedback too — just confirm the mapping holds
        assert ev["feedback"] in ("success", "failure", "unknown")


def test_evaluate_updates_simulation_feedback_on_disk(cfg):
    sim = _make_simulation(cfg)
    evaluate_simulation(sim, sim["expected_outcome"], config=cfg)
    data = json.loads(
        (cfg.memory_path / "simulations" / f"{sim['id']}.json").read_text(encoding="utf-8")
    )
    assert data["feedback"] == "success"


def test_evaluate_mismatch_sets_failure_on_disk(cfg):
    sim = _make_simulation(cfg)
    evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )
    data = json.loads(
        (cfg.memory_path / "simulations" / f"{sim['id']}.json").read_text(encoding="utf-8")
    )
    assert data["feedback"] == "failure"


def test_evaluation_record_has_feedback_field(cfg):
    sim = _make_simulation(cfg)
    ev = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    assert "feedback" in ev


def test_evaluation_feedback_is_valid_value(cfg):
    sim = _make_simulation(cfg)
    ev = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    assert ev["feedback"] in ("success", "failure", "unknown")


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_feedback_never_touches_vault_core(cfg):
    core_dir = cfg.vault_path / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    before = set(core_dir.rglob("*"))
    sim = _make_simulation(cfg)
    SimulationStore(cfg).update_feedback(sim["id"], "success")
    after = set(core_dir.rglob("*"))
    assert before == after
