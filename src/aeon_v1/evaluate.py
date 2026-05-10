"""Simulation evaluation engine for Aeon-V1 Layer 5.

evaluate_simulation() compares a simulation's expected_outcome against an
observed result string, scores the match, and writes the evaluation as a
new episodic memory — closing the feedback loop from simulation to learning.

SAFETY GUARANTEE
================
This module NEVER imports or calls:
  subprocess, os.system, os.popen, shutil, exec, eval,
  PowerShell, bash, shell, or any network/execution primitive.

Records are written to:
  memory/evaluations/<id>.json   — structured evaluation record
  vault/evaluations/<id>.md      — Obsidian note with comparison sections

The episodic memory is written directly to memory/episodic/ + vault/episodic/
via MemoryStore, guaranteeing it is always stored regardless of importance score.
vault/core/ is never touched.
"""
import json
import re
from typing import Dict, List, Optional

from .config import Config
from .memory_store import MemoryStore, _generate_id, _vault_note_path, _wikilink
from .simulate import SimulationStore
from .tasks import TaskStore
from .time_utils import local_date_time_string, utc_now_iso

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "and", "or", "for", "with", "that", "this", "it",
    "its", "as", "by", "on", "at", "from", "will", "would", "should",
    "has", "have", "had", "not", "but", "if", "so", "do", "does", "did",
})

_MATCH_THRESHOLD    = 0.7
_MISMATCH_THRESHOLD = 0.3
_MAX_DIVERGENCES    = 5


def _word_set(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOPWORDS
    }


def _jaccard_score(a: str, b: str) -> float:
    """Stopword-filtered word-level Jaccard similarity in [0.0, 1.0]."""
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return round(len(wa & wb) / len(wa | wb), 3)


def _verdict(score: float) -> str:
    if score >= _MATCH_THRESHOLD:
        return "match"
    if score >= _MISMATCH_THRESHOLD:
        return "partial_match"
    return "mismatch"


def _divergences(expected: str, result: str) -> List[str]:
    """Words present in result but absent from expected (stopwords excluded)."""
    new_words = _word_set(result) - _word_set(expected)
    return sorted(new_words)[:_MAX_DIVERGENCES]


class EvaluationStore:
    """Persist and retrieve simulation evaluation records."""

    def __init__(self, config: Config):
        self.config = config
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        (self.config.memory_path / "evaluations").mkdir(parents=True, exist_ok=True)
        _vault_note_path(self.config, "evaluations", ".keep").parent.mkdir(parents=True, exist_ok=True)

    def store(self, evaluation: Dict) -> Dict:
        eval_id = evaluation["id"]
        (self.config.memory_path / "evaluations" / f"{eval_id}.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        self._write_markdown(evaluation)
        return evaluation

    def get(self, eval_id: str) -> Optional[Dict]:
        path = self.config.memory_path / "evaluations" / f"{eval_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_evaluations(
        self,
        verdict:       Optional[str] = None,
        simulation_id: Optional[str] = None,
        feedback:      Optional[str] = None,
    ) -> List[Dict]:
        eval_dir = self.config.memory_path / "evaluations"
        if not eval_dir.exists():
            return []
        records: List[Dict] = []
        for f in eval_dir.glob("*.json"):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if verdict       is not None and r.get("verdict")       != verdict:
                continue
            if simulation_id is not None and r.get("simulation_id") != simulation_id:
                continue
            if feedback      is not None and r.get("feedback")      != feedback:
                continue
            records.append(r)
        return sorted(records, key=lambda r: r.get("created_at", ""))

    def _write_markdown(self, ev: Dict) -> None:
        md_path = _vault_note_path(self.config, "evaluations", ev["id"])
        fm_lines = [
            "---",
            f"id: {ev['id']}",
            f"type: evaluation",
            f"simulation_id: {ev['simulation_id']}",
            f"task_id: {ev['task_id']}",
            f"verdict: {ev['verdict']}",
            f"match_score: {ev['match_score']}",
            f"episodic_memory_id: {ev['episodic_memory_id']}",
            f"created_at: {ev['created_at']}",
            "---",
        ]
        div_md = (
            "\n".join(f"- {d}" for d in ev.get("divergences", []))
            or "_None detected._"
        )
        ep_link = _wikilink(
            "episodic",
            ev["episodic_memory_id"],
            f"Evaluation — {ev['task_title']}",
        )
        sim_link  = _wikilink("simulations", ev["simulation_id"], ev["simulation_id"])
        task_link = _wikilink("tasks",       ev["task_id"],       ev["task_title"])
        local_ts  = local_date_time_string(ev["created_at"], self.config.display_timezone)

        body = (
            f"# Evaluation — {ev['task_title']}\n\n"
            f"**Created:** {local_ts}\n\n"
            f"**Verdict:** `{ev['verdict']}`  |  "
            f"**Match Score:** {ev['match_score']:.0%}\n\n"
            f"## Expected Outcome\n\n{ev['expected_outcome']}\n\n"
            f"## Actual Result\n\n{ev['actual_result']}\n\n"
            f"## Divergences\n\n{div_md}\n\n"
            f"## Episodic Memory\n\n{ep_link}\n\n"
            f"**Simulation:** {sim_link}  |  **Task:** {task_link}\n\n"
            "[[Evaluations]] | [[Simulations]] | [[Tasks]] | [[Episodic Memory]]"
        )
        md_path.write_text("\n".join(fm_lines) + "\n\n" + body, encoding="utf-8")


def evaluate_simulation(
    simulation: Dict,
    result: str,
    config: Optional[Config] = None,
) -> Dict:
    """Compare a simulation's expected_outcome against an observed result.

    Steps:
    1. Score the match between expected_outcome and result (Jaccard similarity).
    2. Write the evaluation text as a new raw memory.
    3. Promote it directly to episodic memory (always, regardless of score).
    4. Persist the structured evaluation record.

    Args:
        simulation: A simulation record as returned by simulate_action().
        result:     The observed or reported outcome to compare against.
        config:     Optional Config — defaults to Config().

    Returns:
        {"evaluation": evaluation_record, "episodic": episodic_memory_record}
    """
    from .write_guard import assert_write_authorized
    assert_write_authorized("evaluate_simulation")
    if config is None:
        config = Config()

    expected  = simulation.get("expected_outcome", "")
    task_id   = simulation.get("task_id",   "")
    task_title = simulation.get("task_title", task_id)
    sim_id    = simulation.get("id", "")

    score      = _jaccard_score(expected, result)
    verdict    = _verdict(score)
    diverges   = _divergences(expected, result)

    _FEEDBACK_MAP = {"match": "success", "mismatch": "failure", "partial_match": "unknown"}
    feedback   = _FEEDBACK_MAP[verdict]
    SimulationStore(config).update_feedback(sim_id, feedback)

    # --- Adjust task confidence based on evaluation outcome -------------------
    _CONFIDENCE_DELTA = {"success": +0.15, "failure": -0.20, "unknown": -0.05}
    task_store = TaskStore(config)
    current_task = task_store.get_task(task_id)
    confidence_before = round(current_task["confidence"], 4) if current_task else None
    if current_task is not None:
        new_conf = current_task["confidence"] + _CONFIDENCE_DELTA[feedback]
        task_store.update_confidence(task_id, new_conf)
    confidence_after = round(max(0.0, min(1.0, confidence_before + _CONFIDENCE_DELTA[feedback])), 4) if confidence_before is not None else None
    confidence_delta = round(_CONFIDENCE_DELTA[feedback], 4) if confidence_before is not None else None

    # --- Write raw memory verbatim, then promote directly to episodic ----------
    store       = MemoryStore(config)
    eval_text   = _evaluation_text(task_title, sim_id, expected, result, score, verdict)
    raw         = store.store_raw(eval_text, source="evaluation")
    episodic    = store.store_episodic(
        summary=(
            f"Simulation evaluation for '{task_title}': "
            f"verdict={verdict}, match={score:.0%}."
        ),
        raw_id=raw["id"],
        tags=["evaluation", "simulation", "result"] + simulation.get("source_links", [])[:0],
        importance=max(raw["importance"], 0.6),
        source="evaluation",
        raw_title=raw.get("title"),
    )

    # --- Build and persist the evaluation record ------------------------------
    eval_id   = _generate_id()
    now       = utc_now_iso()
    sim_link  = _wikilink("simulations", sim_id,   sim_id)
    task_link = _wikilink("tasks",       task_id,  task_title)
    ep_link   = _wikilink("episodic",    episodic["id"], f"Evaluation — {task_title}")

    evaluation: Dict = {
        "id":                eval_id,
        "simulation_id":     sim_id,
        "task_id":           task_id,
        "task_title":        task_title,
        "expected_outcome":  expected,
        "actual_result":     result,
        "match_score":       score,
        "verdict":           verdict,
        "feedback":              feedback,
        "task_confidence_before": confidence_before,
        "task_confidence_after":  confidence_after,
        "confidence_delta":       confidence_delta,
        "divergences":       diverges,
        "episodic_memory_id": episodic["id"],
        "created_at":        now,
        "source_links":      [sim_link, task_link, ep_link],
    }

    EvaluationStore(config).store(evaluation)
    return {"evaluation": evaluation, "episodic": episodic}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluation_text(
    task_title: str,
    sim_id:     str,
    expected:   str,
    result:     str,
    score:      float,
    verdict:    str,
) -> str:
    lines = [
        f"Simulation evaluation for task '{task_title}' (simulation {sim_id}).",
        f"Expected outcome: {expected}",
        f"Actual result: {result}",
        f"Match score: {score:.0%}. Verdict: {verdict}.",
    ]
    return " ".join(lines)
