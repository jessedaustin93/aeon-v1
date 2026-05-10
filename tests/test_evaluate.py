"""Tests for Layer 5 — simulation evaluation and episodic memory storage.

Coverage:
- evaluate_simulation returns evaluation record + episodic memory
- Evaluation record has all required fields
- match_score: Jaccard similarity of expected vs result
- verdict: "match" / "partial_match" / "mismatch" thresholds
- divergences: words in result absent from expected
- Episodic memory always created (regardless of importance score)
- episodic_memory_id links to the stored episodic record
- EvaluationStore: JSON + Markdown written with correct content
- EvaluationStore.get / list_evaluations (with verdict and simulation_id filters)
- Markdown: Expected Outcome, Actual Result, Divergences, Episodic Memory sections
- No execution primitives in evaluate.py
- vault/core/ never touched
"""
import ast
import json
from pathlib import Path

import pytest

from aeon_v1 import Config, EvaluationStore, evaluate_simulation, register_builtin_tools
from aeon_v1.evaluate import _jaccard_score, _verdict, _divergences
from aeon_v1.memory_store import _vault_note_path
from aeon_v1.simulate import simulate_action
from aeon_v1.tasks import TaskStore
from aeon_v1.tools import ToolRegistry


SRC_DIR = Path(__file__).parent.parent / "src" / "aeon_v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config(base_path=tmp_path)
    c.ensure_dirs()
    return c


@pytest.fixture
def cfg_with_tools(cfg):
    register_builtin_tools(ToolRegistry(cfg))
    return cfg


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
# Comparison primitives
# ---------------------------------------------------------------------------

def test_jaccard_identical_text():
    assert _jaccard_score("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_jaccard_completely_different():
    score = _jaccard_score("apple banana cherry", "dog elephant frog")
    assert score == 0.0


def test_jaccard_partial_overlap():
    score = _jaccard_score("memory index updated successfully", "memory store partially updated")
    assert 0.0 < score < 1.0


def test_jaccard_empty_strings():
    assert _jaccard_score("", "some text") == 0.0
    assert _jaccard_score("some text", "") == 0.0


def test_jaccard_stopwords_filtered():
    # Stopwords should be ignored — wrapping content words in stopwords
    # must not change the Jaccard score.
    score_clean = _jaccard_score("fox dog", "fox dog cat")
    score_noisy = _jaccard_score("the fox a dog", "the fox a dog the cat")
    assert score_clean == score_noisy  # only "fox", "dog", "cat" matter in both


def test_verdict_match():
    assert _verdict(0.7)  == "match"
    assert _verdict(1.0)  == "match"
    assert _verdict(0.85) == "match"


def test_verdict_partial_match():
    assert _verdict(0.3)  == "partial_match"
    assert _verdict(0.5)  == "partial_match"
    assert _verdict(0.69) == "partial_match"


def test_verdict_mismatch():
    assert _verdict(0.0)  == "mismatch"
    assert _verdict(0.1)  == "mismatch"
    assert _verdict(0.29) == "mismatch"


def test_divergences_finds_new_words():
    divs = _divergences("task was completed", "task was completed with errors logged")
    assert "errors" in divs or "logged" in divs


def test_divergences_excludes_stopwords():
    divs = _divergences("memory updated", "memory updated and the a is")
    # "and", "the", "a", "is" are stopwords — should not appear
    for word in ("and", "the", "a", "is"):
        assert word not in divs


def test_divergences_max_five():
    expected = "one two three"
    result   = "one two three alpha beta gamma delta epsilon zeta"
    assert len(_divergences(expected, result)) <= 5


def test_divergences_empty_when_result_subset_of_expected():
    divs = _divergences("alpha beta gamma delta", "alpha beta")
    assert divs == []


# ---------------------------------------------------------------------------
# evaluate_simulation — return structure
# ---------------------------------------------------------------------------

def test_returns_evaluation_and_episodic(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "The memory index was updated.", config=cfg)
    assert "evaluation" in out
    assert "episodic"   in out


def test_evaluation_record_has_required_fields(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "The index was refreshed.", config=cfg)["evaluation"]
    required = [
        "id", "simulation_id", "task_id", "task_title",
        "expected_outcome", "actual_result",
        "match_score", "verdict", "divergences",
        "episodic_memory_id", "created_at", "source_links",
    ]
    for field in required:
        assert field in ev, f"Missing field: {field}"


def test_evaluation_simulation_id_matches(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Result text.", config=cfg)["evaluation"]
    assert ev["simulation_id"] == sim["id"]


def test_evaluation_task_id_matches(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Result text.", config=cfg)["evaluation"]
    assert ev["task_id"] == sim["task_id"]


def test_evaluation_expected_outcome_stored(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Result.", config=cfg)["evaluation"]
    assert ev["expected_outcome"] == sim["expected_outcome"]


def test_evaluation_actual_result_stored(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Observed: the cache was rebuilt.", config=cfg)["evaluation"]
    assert ev["actual_result"] == "Observed: the cache was rebuilt."


def test_match_score_is_float(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Memory index updated.", config=cfg)["evaluation"]
    assert isinstance(ev["match_score"], float)


def test_match_score_in_bounds(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Something happened.", config=cfg)["evaluation"]
    assert 0.0 <= ev["match_score"] <= 1.0


def test_verdict_is_valid_string(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Something happened.", config=cfg)["evaluation"]
    assert ev["verdict"] in ("match", "partial_match", "mismatch")


def test_verdict_match_for_similar_result(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, sim["expected_outcome"], config=cfg)["evaluation"]
    assert ev["verdict"] == "match"
    assert ev["match_score"] == 1.0


def test_verdict_mismatch_for_unrelated_result(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(
        sim,
        "Completely unrelated xyzzy plugh frobnicate quux thud",
        config=cfg,
    )["evaluation"]
    assert ev["verdict"] == "mismatch"


def test_divergences_is_list(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Novel unexpected outcome emerged here.", config=cfg)["evaluation"]
    assert isinstance(ev["divergences"], list)


def test_divergences_at_most_five(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(
        sim,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        config=cfg,
    )["evaluation"]
    assert len(ev["divergences"]) <= 5


# ---------------------------------------------------------------------------
# Episodic memory — always created
# ---------------------------------------------------------------------------

def test_episodic_memory_created(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "The result was observed.", config=cfg)
    assert out["episodic"] is not None


def test_episodic_memory_has_id(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "The result was observed.", config=cfg)
    assert out["episodic"]["id"]


def test_episodic_memory_file_on_disk(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "The result was observed.", config=cfg)
    ep_id   = out["episodic"]["id"]
    ep_path = cfg.memory_path / "episodic" / f"{ep_id}.json"
    assert ep_path.exists()


def test_episodic_memory_id_matches_evaluation_record(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "Outcome observed.", config=cfg)
    assert out["evaluation"]["episodic_memory_id"] == out["episodic"]["id"]


def test_episodic_type_is_episodic(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "Result noted.", config=cfg)
    assert out["episodic"]["type"] == "episodic"


def test_episodic_source_is_evaluation(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "Result noted.", config=cfg)
    assert out["episodic"]["source"] == "evaluation"


def test_episodic_tags_include_evaluation(cfg):
    sim = _make_simulation(cfg)
    out = evaluate_simulation(sim, "Result noted.", config=cfg)
    assert "evaluation" in out["episodic"]["tags"]


# ---------------------------------------------------------------------------
# EvaluationStore — persistence
# ---------------------------------------------------------------------------

def test_evaluation_json_written(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Result.", config=cfg)["evaluation"]
    assert (cfg.memory_path / "evaluations" / f"{ev['id']}.json").exists()


def test_evaluation_markdown_written(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Result.", config=cfg)["evaluation"]
    assert _vault_note_path(cfg, "evaluations", ev["id"]).exists()


def test_evaluation_json_round_trip(cfg):
    sim  = _make_simulation(cfg)
    ev   = evaluate_simulation(sim, "Some result text.", config=cfg)["evaluation"]
    data = json.loads(
        (cfg.memory_path / "evaluations" / f"{ev['id']}.json").read_text(encoding="utf-8")
    )
    assert data["verdict"]      == ev["verdict"]
    assert data["match_score"]  == ev["match_score"]
    assert data["actual_result"] == "Some result text."


def test_evaluation_store_get(cfg):
    sim  = _make_simulation(cfg)
    ev   = evaluate_simulation(sim, "Result.", config=cfg)["evaluation"]
    store = EvaluationStore(cfg)
    fetched = store.get(ev["id"])
    assert fetched is not None
    assert fetched["id"] == ev["id"]


def test_evaluation_store_get_unknown_returns_none(cfg):
    assert EvaluationStore(cfg).get("no-such-id") is None


def test_evaluation_store_list_all(cfg):
    sim = _make_simulation(cfg)
    evaluate_simulation(sim, "Result one.", config=cfg)
    evaluate_simulation(sim, "Result two.", config=cfg)
    records = EvaluationStore(cfg).list_evaluations()
    assert len(records) == 2


def test_evaluation_store_filter_by_verdict(cfg):
    sim = _make_simulation(cfg)
    evaluate_simulation(sim, sim["expected_outcome"], config=cfg)  # → match
    evaluate_simulation(sim, "xyzzy plugh frobnicate quux thud", config=cfg)  # → mismatch
    store   = EvaluationStore(cfg)
    matches = store.list_evaluations(verdict="match")
    assert all(r["verdict"] == "match" for r in matches)


def test_evaluation_store_filter_by_simulation_id(cfg):
    sim1 = _make_simulation(cfg, "Read the first index file.")
    sim2 = _make_simulation(cfg, "Write the second output file.")
    evaluate_simulation(sim1, "Result for sim1.", config=cfg)
    evaluate_simulation(sim2, "Result for sim2.", config=cfg)
    records = EvaluationStore(cfg).list_evaluations(simulation_id=sim1["id"])
    assert len(records) == 1
    assert records[0]["simulation_id"] == sim1["id"]


# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------

def test_markdown_has_expected_outcome_section(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    md  = _vault_note_path(cfg, "evaluations", ev["id"]).read_text(encoding="utf-8")
    assert "## Expected Outcome" in md


def test_markdown_has_actual_result_section(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    md  = _vault_note_path(cfg, "evaluations", ev["id"]).read_text(encoding="utf-8")
    assert "## Actual Result" in md
    assert "Some result." in md


def test_markdown_has_divergences_section(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    md  = _vault_note_path(cfg, "evaluations", ev["id"]).read_text(encoding="utf-8")
    assert "## Divergences" in md


def test_markdown_has_episodic_memory_section(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    md  = _vault_note_path(cfg, "evaluations", ev["id"]).read_text(encoding="utf-8")
    assert "## Episodic Memory" in md
    assert ev["episodic_memory_id"] in md


def test_markdown_frontmatter_has_required_fields(cfg):
    sim = _make_simulation(cfg)
    ev  = evaluate_simulation(sim, "Some result.", config=cfg)["evaluation"]
    md  = _vault_note_path(cfg, "evaluations", ev["id"]).read_text(encoding="utf-8")
    assert "type: evaluation"  in md
    assert "simulation_id:"    in md
    assert "verdict:"          in md
    assert "match_score:"      in md
    assert "episodic_memory_id:" in md


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_no_execution_primitives_in_evaluate():
    source = (SRC_DIR / "evaluate.py").read_text(encoding="utf-8")
    tree   = ast.parse(source)
    banned = {
        "subprocess", "os.system", "os.popen", "exec", "eval",
        "shutil", "PowerShell", "bash", "shell",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in banned, f"Banned import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in banned, f"Banned import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval"}, f"Banned call: {node.func.id}"


def test_evaluate_never_touches_vault_core(cfg):
    core_dir = cfg.vault_path / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    before = set(core_dir.rglob("*"))
    sim = _make_simulation(cfg)
    evaluate_simulation(sim, "Some result text.", config=cfg)
    after = set(core_dir.rglob("*"))
    assert before == after
