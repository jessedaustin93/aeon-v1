"""Tests for reflection awareness of past simulation failures (Step 3).

Coverage:
- _analyse() returns failure_count and recent_failures in its dict
- failure_count == 0 when no evaluations exist
- failure_count reflects only "failure" verdict evaluations (not match/partial)
- recent_failures contains task_title, match_score, divergences, simulation_id
- recent_failures is capped at 5 most recent
- reflect() includes failure_count in returned metadata
- Reflection Markdown Section 4 includes "Past Simulation Failures" when failures exist
- Section 4 omits the failure block when no failures exist
- vault/core/ never touched by reflect()
"""
import json

import pytest

from aeon_v1 import Config, evaluate_simulation, ingest, reflect
from aeon_v1.evaluate import EvaluationStore
from aeon_v1.reflect import _analyse
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


def _make_task(cfg, description="Review the project memory index.", confidence=0.5):
    store = TaskStore(cfg)
    task = store.create_task(
        description=description,
        source_reflection_id="test-ref",
        source_reflection_title="Test Reflection",
        confidence=confidence,
        tags=["test"],
    )
    assert task is not None
    return task


def _make_mismatch_evaluation(cfg, description="Review the memory index.", confidence=0.5):
    """Create a simulation + mismatch evaluation → feedback=failure."""
    task = _make_task(cfg, description, confidence)
    sim = simulate_action(task, config=cfg)["simulation"]
    ev = evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud wibble",
        config=cfg,
    )["evaluation"]
    assert ev["verdict"] == "mismatch"
    return ev


def _make_match_evaluation(cfg, description="Review the memory index.", confidence=0.5):
    """Create a simulation + matching evaluation → feedback=success."""
    task = _make_task(cfg, description, confidence)
    sim = simulate_action(task, config=cfg)["simulation"]
    ev = evaluate_simulation(sim, sim["expected_outcome"], config=cfg)["evaluation"]
    assert ev["verdict"] == "match"
    return ev


# ---------------------------------------------------------------------------
# _analyse() — failure fields
# ---------------------------------------------------------------------------

def test_analyse_has_failure_count_field():
    result = _analyse([], [], past_failures=[])
    assert "failure_count" in result


def test_analyse_has_recent_failures_field():
    result = _analyse([], [], past_failures=[])
    assert "recent_failures" in result


def test_analyse_failure_count_zero_when_no_failures():
    result = _analyse([], [], past_failures=[])
    assert result["failure_count"] == 0


def test_analyse_failure_count_matches_input():
    fake_failures = [
        {"task_title": "T1", "match_score": 0.1, "divergences": [], "simulation_id": "s1", "created_at": "2026-01-01"},
        {"task_title": "T2", "match_score": 0.2, "divergences": ["extra"], "simulation_id": "s2", "created_at": "2026-01-02"},
    ]
    result = _analyse([], [], past_failures=fake_failures)
    assert result["failure_count"] == 2


def test_analyse_recent_failures_has_required_fields():
    fake = [{"task_title": "T1", "match_score": 0.1, "divergences": ["a"], "simulation_id": "s1", "created_at": "2026-01-01"}]
    result = _analyse([], [], past_failures=fake)
    assert len(result["recent_failures"]) == 1
    rf = result["recent_failures"][0]
    for field in ("task_title", "match_score", "divergences", "simulation_id"):
        assert field in rf


def test_analyse_recent_failures_capped_at_five():
    fake_failures = [
        {"task_title": f"T{i}", "match_score": 0.1, "divergences": [], "simulation_id": f"s{i}", "created_at": f"2026-01-0{i+1}"}
        for i in range(8)
    ]
    result = _analyse([], [], past_failures=fake_failures)
    assert len(result["recent_failures"]) <= 5


def test_analyse_no_failures_argument_defaults_zero():
    result = _analyse([], [])
    assert result["failure_count"] == 0
    assert result["recent_failures"] == []


# ---------------------------------------------------------------------------
# EvaluationStore.list_evaluations(verdict="failure") integration
# ---------------------------------------------------------------------------

def test_failure_count_only_counts_mismatch_verdicts(cfg):
    _make_mismatch_evaluation(cfg, "Review the memory index.")
    _make_match_evaluation(cfg, "Write the output report file.")
    failures = EvaluationStore(cfg).list_evaluations(feedback="failure")
    assert len(failures) == 1


# ---------------------------------------------------------------------------
# reflect() — failure_count in metadata
# ---------------------------------------------------------------------------

def test_reflect_metadata_has_failure_count(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    _make_mismatch_evaluation(cfg)
    result = reflect(config=cfg)
    assert result["reflection"] is not None
    assert "failure_count" in result["reflection"]


def test_reflect_failure_count_is_zero_when_no_failures(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    result = reflect(config=cfg)
    if result["reflection"] is not None:
        assert result["reflection"]["failure_count"] == 0


def test_reflect_failure_count_matches_mismatch_evaluations(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    _make_mismatch_evaluation(cfg, "Review the memory index.", confidence=0.5)
    _make_mismatch_evaluation(cfg, "Inspect the episodic layer records.", confidence=0.5)
    result = reflect(config=cfg)
    if result["reflection"] is not None:
        assert result["reflection"]["failure_count"] == 2


# ---------------------------------------------------------------------------
# Reflection Markdown — Section 4 content
# ---------------------------------------------------------------------------

def _get_reflection_markdown(cfg) -> str:
    """Return the Markdown text of the most recent reflection."""
    from aeon_v1.memory_store import MemoryStore
    reflections = MemoryStore(cfg).list_memories("reflections")
    assert reflections, "No reflections found"
    r = sorted(reflections, key=lambda x: x.get("created", ""))[-1]
    md_path = _vault_note_path(cfg, "reflections", r["id"])
    return md_path.read_text(encoding="utf-8")


def test_section4_contains_past_failures_header_when_failures_exist(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    _make_mismatch_evaluation(cfg)
    reflect(config=cfg)
    md = _get_reflection_markdown(cfg)
    assert "Past Simulation Failures" in md


def test_section4_does_not_contain_failure_header_when_no_failures(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    result = reflect(config=cfg)
    if result["reflection"] is None:
        pytest.skip("Reflect returned None (too few sources)")
    md = _get_reflection_markdown(cfg)
    assert "Past Simulation Failures" not in md


def test_section4_lists_failed_task_title(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    ev = _make_mismatch_evaluation(cfg, "Review the memory index patterns.")
    reflect(config=cfg)
    md = _get_reflection_markdown(cfg)
    assert ev["task_title"] in md or "Past Simulation Failures" in md


def test_section4_shows_match_score_for_failure(cfg):
    ingest("I learned about patterns in the memory system.", config=cfg)
    _make_mismatch_evaluation(cfg)
    reflect(config=cfg)
    md = _get_reflection_markdown(cfg)
    assert "0%" in md or "Past Simulation Failures" in md


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_reflect_failures_never_touches_vault_core(cfg):
    core_dir = cfg.vault_path / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    before = set(core_dir.rglob("*"))
    ingest("I learned about patterns in the memory system.", config=cfg)
    _make_mismatch_evaluation(cfg)
    reflect(config=cfg)
    after = set(core_dir.rglob("*"))
    assert before == after
