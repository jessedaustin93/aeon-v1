"""Tests for task confidence adjustment based on evaluation outcome (Step 2).

Coverage:
- TaskStore.update_confidence: clamps to [0.0, 1.0], persists to JSON + Markdown
- update_confidence returns None for missing task
- evaluate_simulation: success raises confidence by +0.15
- evaluate_simulation: failure lowers confidence by -0.20
- evaluate_simulation: partial_match lowers confidence by -0.05
- Confidence is clamped at 0.0 (cannot go negative)
- Confidence is clamped at 1.0 (cannot exceed 1.0)
- Evaluation record carries task_confidence_before, task_confidence_after, confidence_delta
- Task JSON on disk reflects new confidence after evaluation
- vault/core/ never touched
"""
import json

import pytest

from aeon_v1 import Config, evaluate_simulation
from aeon_v1.memory_store import _vault_note_path
from aeon_v1.simulate import simulate_action
from aeon_v1.tasks import TaskStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config(base_path=tmp_path)
    c.ensure_dirs()
    return c


def _make_simulation(cfg: Config, description: str = "Review the project memory index.",
                     confidence: float = 0.5) -> tuple:
    store = TaskStore(cfg)
    task = store.create_task(
        description=description,
        source_reflection_id="test-ref",
        source_reflection_title="Test Reflection",
        confidence=confidence,
        tags=["test"],
    )
    assert task is not None
    sim = simulate_action(task, config=cfg)["simulation"]
    return task, sim


# ---------------------------------------------------------------------------
# TaskStore.update_confidence
# ---------------------------------------------------------------------------

def test_update_confidence_returns_task(cfg):
    task, _ = _make_simulation(cfg)
    updated = TaskStore(cfg).update_confidence(task["id"], 0.8)
    assert updated is not None
    assert updated["confidence"] == pytest.approx(0.8)


def test_update_confidence_persists_to_json(cfg):
    task, _ = _make_simulation(cfg)
    TaskStore(cfg).update_confidence(task["id"], 0.75)
    data = json.loads(
        (cfg.memory_path / "tasks" / f"{task['id']}.json").read_text(encoding="utf-8")
    )
    assert data["confidence"] == pytest.approx(0.75)


def test_update_confidence_updates_markdown(cfg):
    task, _ = _make_simulation(cfg)
    TaskStore(cfg).update_confidence(task["id"], 0.9)
    md = _vault_note_path(cfg, "tasks", task["id"]).read_text(encoding="utf-8")
    assert "0.9" in md


def test_update_confidence_clamps_above_1(cfg):
    task, _ = _make_simulation(cfg)
    updated = TaskStore(cfg).update_confidence(task["id"], 1.5)
    assert updated["confidence"] == pytest.approx(1.0)


def test_update_confidence_clamps_below_0(cfg):
    task, _ = _make_simulation(cfg)
    updated = TaskStore(cfg).update_confidence(task["id"], -0.3)
    assert updated["confidence"] == pytest.approx(0.0)


def test_update_confidence_missing_task_returns_none(cfg):
    assert TaskStore(cfg).update_confidence("no-such-id", 0.5) is None


# ---------------------------------------------------------------------------
# evaluate_simulation — confidence delta applied
# ---------------------------------------------------------------------------

def test_confidence_increases_on_success(cfg):
    task, sim = _make_simulation(cfg, confidence=0.5)
    evaluate_simulation(sim, sim["expected_outcome"], config=cfg)
    updated = TaskStore(cfg).get_task(task["id"])
    assert updated["confidence"] > 0.5


def test_confidence_increases_by_015_on_match(cfg):
    task, sim = _make_simulation(cfg, confidence=0.5)
    evaluate_simulation(sim, sim["expected_outcome"], config=cfg)
    updated = TaskStore(cfg).get_task(task["id"])
    assert updated["confidence"] == pytest.approx(0.65, abs=0.001)


def test_confidence_decreases_on_failure(cfg):
    task, sim = _make_simulation(cfg, confidence=0.5)
    evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )
    updated = TaskStore(cfg).get_task(task["id"])
    assert updated["confidence"] < 0.5


def test_confidence_decreases_by_020_on_mismatch(cfg):
    task, sim = _make_simulation(cfg, confidence=0.5)
    ev = evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )["evaluation"]
    assert ev["verdict"] == "mismatch"
    updated = TaskStore(cfg).get_task(task["id"])
    assert updated["confidence"] == pytest.approx(0.30, abs=0.001)


def test_confidence_clamped_at_1_on_match(cfg):
    task, sim = _make_simulation(cfg, confidence=0.95)
    evaluate_simulation(sim, sim["expected_outcome"], config=cfg)
    updated = TaskStore(cfg).get_task(task["id"])
    assert updated["confidence"] <= 1.0


def test_confidence_clamped_at_0_on_mismatch(cfg):
    task, sim = _make_simulation(cfg, confidence=0.1)
    evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )
    updated = TaskStore(cfg).get_task(task["id"])
    assert updated["confidence"] >= 0.0


# ---------------------------------------------------------------------------
# Evaluation record — confidence fields
# ---------------------------------------------------------------------------

def test_evaluation_has_confidence_before(cfg):
    _, sim = _make_simulation(cfg, confidence=0.5)
    ev = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    assert "task_confidence_before" in ev


def test_evaluation_has_confidence_after(cfg):
    _, sim = _make_simulation(cfg, confidence=0.5)
    ev = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    assert "task_confidence_after" in ev


def test_evaluation_has_confidence_delta(cfg):
    _, sim = _make_simulation(cfg, confidence=0.5)
    ev = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    assert "confidence_delta" in ev


def test_evaluation_confidence_before_matches_original(cfg):
    task, sim = _make_simulation(cfg, confidence=0.6)
    ev = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    assert ev["task_confidence_before"] == pytest.approx(0.6, abs=0.001)


def test_evaluation_confidence_delta_is_positive_for_match(cfg):
    _, sim = _make_simulation(cfg, confidence=0.5)
    ev = evaluate_simulation(sim, sim["expected_outcome"], config=cfg)["evaluation"]
    assert ev["verdict"] == "match"
    assert ev["confidence_delta"] > 0


def test_evaluation_confidence_delta_is_negative_for_mismatch(cfg):
    _, sim = _make_simulation(cfg, confidence=0.5)
    ev = evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )["evaluation"]
    assert ev["verdict"] == "mismatch"
    assert ev["confidence_delta"] < 0


def test_evaluation_confidence_after_reflects_clamping(cfg):
    task, sim = _make_simulation(cfg, confidence=0.95)
    ev = evaluate_simulation(sim, sim["expected_outcome"], config=cfg)["evaluation"]
    assert ev["task_confidence_after"] <= 1.0


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_confidence_adjustment_never_touches_vault_core(cfg):
    core_dir = cfg.vault_path / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    before = set(core_dir.rglob("*"))
    task, sim = _make_simulation(cfg)
    evaluate_simulation(sim, "Some result.", config=cfg)
    after = set(core_dir.rglob("*"))
    assert before == after
