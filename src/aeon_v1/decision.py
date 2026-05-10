"""Decision engine for Aeon-V1 Layer 3.

select_next_task() scores pending tasks, picks the best one, marks it
as selected, and writes an append-only decision record.

vault/core/ is never touched.
No subprocess, os.system, or execution primitive is imported or called.
"""
import json
from typing import Dict, List, Optional

from .config import Config
from .memory_store import _generate_id, _vault_note_path, _wikilink
from .tasks import TaskStore
from .time_utils import local_date_time_string, utc_now_iso


class DecisionStore:
    """Read/write decision records from memory/decisions/ and vault/decisions/."""

    def __init__(self, config: Config):
        self.config = config
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        (self.config.memory_path / "decisions").mkdir(parents=True, exist_ok=True)
        _vault_note_path(self.config, "decisions", ".keep").parent.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        selected_task: Dict,
        reason: str,
        alternatives: List[Dict],
    ) -> Dict:
        dec_id = _generate_id()
        now = utc_now_iso()
        task_link = _wikilink(
            "tasks", selected_task["id"], selected_task.get("title")
        )
        alt_titles = [
            a.get("title", a["id"])
            for a in alternatives
        ]

        decision: Dict = {
            "id": dec_id,
            "selected_task_id": selected_task["id"],
            "selected_task_title": selected_task.get("title", ""),
            "reason": reason,
            "created_at": now,
            "confidence": selected_task.get("confidence", 0.0),
            "alternatives_considered": alt_titles,
            "source_links": [task_link],
        }

        (self.config.memory_path / "decisions" / f"{dec_id}.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        self._write_markdown(decision)
        return decision

    def list_decisions(self) -> List[Dict]:
        dec_dir = self.config.memory_path / "decisions"
        if not dec_dir.exists():
            return []
        decisions = []
        for f in dec_dir.glob("*.json"):
            try:
                decisions.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return sorted(decisions, key=lambda d: d.get("created_at", ""))

    def _write_markdown(self, decision: Dict) -> None:
        md_path = _vault_note_path(self.config, "decisions", decision["id"])
        fm_lines = ["---"]
        for key, val in [
            ("id",               decision["id"]),
            ("type",             "decision"),
            ("created_at",       decision["created_at"]),
            ("selected_task_id", decision["selected_task_id"]),
            ("confidence",       decision["confidence"]),
        ]:
            fm_lines.append(f"{key}: {val}")
        fm_lines.append("source_links:")
        for lnk in decision["source_links"]:
            fm_lines.append(f"  - {lnk}")
        fm_lines.append("---")

        alts = (
            ", ".join(decision["alternatives_considered"])
            if decision["alternatives_considered"]
            else "none"
        )
        task_link = _wikilink(
            "tasks",
            decision["selected_task_id"],
            decision["selected_task_title"],
        )
        display_tz = self.config.display_timezone
        local_ts = local_date_time_string(decision["created_at"], display_tz)
        body = (
            f"# Decision — {local_ts}\n\n"
            f"**Selected Task:** {task_link}\n\n"
            f"**Reason:** {decision['reason']}\n\n"
            f"**Confidence:** {decision['confidence']}\n\n"
            f"**Alternatives Considered:** {alts}\n\n"
            "> This decision was produced by the rule-based engine. "
            "No real action was taken.\n\n"
            "[[Decisions]] | [[Tasks]] | [[Simulations]]"
        )
        md_path.write_text("\n".join(fm_lines) + "\n\n" + body, encoding="utf-8")


def _score_task(task: Dict) -> float:
    """Additive score for task selection. Higher is better."""
    return task.get("priority", 0.5) * 0.5 + task.get("confidence", 0.5) * 0.3


def select_next_task(config: Optional[Config] = None) -> Dict:
    """Select the best pending task, mark it selected, and write a decision record.

    Selection criteria (additive score):
      priority  × 0.5  +  confidence  × 0.3

    Returns:
      {"decision": decision_record, "task": selected_task}
    or
      {"decision": None, "task": None, "message": "..."}
    """
    from .write_guard import assert_write_authorized
    assert_write_authorized("select_next_task")
    if config is None:
        config = Config()

    task_store = TaskStore(config)
    decision_store = DecisionStore(config)

    pending = task_store.list_tasks(status="pending")
    if not pending:
        return {
            "decision": None,
            "task": None,
            "message": "No pending tasks to select from.",
        }

    scored = sorted(pending, key=_score_task, reverse=True)
    selected = scored[0]
    alternatives = scored[1:]

    reason = (
        f"Highest combined score (priority={selected.get('priority', 0.5):.2f}, "
        f"confidence={selected.get('confidence', 0.5):.2f}) among "
        f"{len(pending)} pending task(s)."
    )

    task_store.update_status(selected["id"], "selected")
    selected = task_store.get_task(selected["id"])

    decision = decision_store.store(
        selected_task=selected,
        reason=reason,
        alternatives=alternatives,
    )

    return {"decision": decision, "task": selected}
