"""Task storage layer for Aeon-V1 Layer 3.

Tasks are operational records derived from reflection suggested_tasks.
They are append-only at creation time; status updates are written in place
and tracked via separate decision records (see decision.py).

vault/core/ is never touched by any function in this module.
No subprocess, os.system, or execution primitive is imported or called.
"""
import json
import re
from typing import Dict, List, Optional

from .config import Config
from .memory_store import _generate_id, _make_title, _vault_note_path, _wikilink
from .time_utils import local_date_time_string, utc_now_iso


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(re.findall(r'[a-z0-9]+', a.lower()))
    words_b = set(re.findall(r'[a-z0-9]+', b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _is_near_duplicate(description: str, existing: List[Dict], threshold: float) -> bool:
    for task in existing:
        if _jaccard(description, task.get("description", "")) >= threshold:
            return True
    return False


class TaskStore:
    """Read/write tasks from memory/tasks/ and vault/tasks/."""

    def __init__(self, config: Config):
        self.config = config
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        (self.config.memory_path / "tasks").mkdir(parents=True, exist_ok=True)
        _vault_note_path(self.config, "tasks", ".keep").parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- create --

    def create_task(
        self,
        description: str,
        source_reflection_id: str,
        source_reflection_title: str,
        confidence: float,
        tags: List[str],
        priority: float = 0.5,
    ) -> Optional[Dict]:
        """Create a new pending task.

        Returns None if:
        - A near-duplicate already exists (Jaccard >= duplicate_task_similarity_threshold).
        - The pending task count has reached max_pending_tasks.
        """
        existing = self.list_tasks()
        threshold = self.config.duplicate_task_similarity_threshold
        if _is_near_duplicate(description, existing, threshold):
            return None

        pending_count = sum(1 for t in existing if t.get("status") == "pending")
        if pending_count >= self.config.max_pending_tasks:
            return None

        task_id = _generate_id()
        now = utc_now_iso()
        title = _make_title(description)
        source_link = _wikilink(
            "reflections", source_reflection_id, source_reflection_title
        )

        task: Dict = {
            "id": task_id,
            "title": title,
            "description": description,
            "source_reflection_id": source_reflection_id,
            "source_reflection_title": source_reflection_title,
            "created_at": now,
            "status": "pending",
            "priority": priority,
            "confidence": confidence,
            "tags": tags,
            "links": [source_link],
        }

        (self.config.memory_path / "tasks" / f"{task_id}.json").write_text(
            json.dumps(task, indent=2), encoding="utf-8"
        )
        self._write_markdown(task)
        return task

    # --------------------------------------------------------------- update --

    def update_confidence(self, task_id: str, new_confidence: float) -> Optional[Dict]:
        """Update task confidence in place, clamped to [0.0, 1.0].

        Returns the updated task, or None if the task was not found.
        """
        path = self.config.memory_path / "tasks" / f"{task_id}.json"
        if not path.exists():
            return None
        task = json.loads(path.read_text(encoding="utf-8"))
        task["confidence"] = round(max(0.0, min(1.0, new_confidence)), 4)
        path.write_text(json.dumps(task, indent=2), encoding="utf-8")
        self._write_markdown(task)
        return task

    def update_status(self, task_id: str, status: str) -> Optional[Dict]:
        """Update task status in place.

        History of status changes is preserved externally in decision records.
        """
        path = self.config.memory_path / "tasks" / f"{task_id}.json"
        if not path.exists():
            return None
        task = json.loads(path.read_text(encoding="utf-8"))
        task["status"] = status
        path.write_text(json.dumps(task, indent=2), encoding="utf-8")
        self._write_markdown(task)
        return task

    # ------------------------------------------------------------------ read --

    def list_tasks(self, status: Optional[str] = None) -> List[Dict]:
        task_dir = self.config.memory_path / "tasks"
        if not task_dir.exists():
            return []
        tasks = []
        for f in task_dir.glob("*.json"):
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
                if status is None or t.get("status") == status:
                    tasks.append(t)
            except Exception:
                pass
        return sorted(tasks, key=lambda t: t.get("created_at", ""))

    def get_task(self, task_id: str) -> Optional[Dict]:
        path = self.config.memory_path / "tasks" / f"{task_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------- markdown --

    def _write_markdown(self, task: Dict) -> None:
        md_path = _vault_note_path(self.config, "tasks", task["id"])
        fm_lines = ["---"]
        for key, val in [
            ("id",         task["id"]),
            ("title",      task["title"]),
            ("type",       "task"),
            ("created_at", task["created_at"]),
            ("status",     task["status"]),
            ("priority",   task["priority"]),
            ("confidence", task["confidence"]),
        ]:
            fm_lines.append(f"{key}: {val}")
        fm_lines.append("tags:")
        for t in task.get("tags", []):
            fm_lines.append(f"  - {t}")
        fm_lines.append("links:")
        for lnk in task.get("links", []):
            fm_lines.append(f"  - {lnk}")
        fm_lines.append("---")

        display_tz = self.config.display_timezone
        local_ts = local_date_time_string(task["created_at"], display_tz)
        src_link = _wikilink(
            "reflections",
            task["source_reflection_id"],
            task.get("source_reflection_title", task["source_reflection_id"]),
        )
        body = (
            f"# {task['title']}\n\n"
            f"**Created:** {local_ts}\n\n"
            f"**Description:** {task['description']}\n\n"
            f"**Status:** `{task['status']}`\n\n"
            f"**Priority:** {task['priority']}  |  "
            f"**Confidence:** {task['confidence']}\n\n"
            f"**Source Reflection:** {src_link}\n\n"
            "[[Tasks]] | [[Reflections]] | [[Decisions]]"
        )
        md_path.write_text("\n".join(fm_lines) + "\n\n" + body, encoding="utf-8")


def create_tasks_from_reflection(reflection: Dict, config: Config) -> List[Dict]:
    """Convert a reflection's suggested_tasks list into stored task objects.

    Called by reflect() immediately after the reflection record is written.
    Returns a list of newly created task dicts (may be empty if all are duplicates).
    """
    store = TaskStore(config)
    created = []

    ref_id = reflection.get("id", "")
    ref_title = reflection.get("title", ref_id)
    confidence = reflection.get("confidence", 0.5)
    tags = reflection.get("tags", [])

    for raw_task in reflection.get("suggested_tasks", []):
        task = store.create_task(
            description=raw_task,
            source_reflection_id=ref_id,
            source_reflection_title=ref_title,
            confidence=confidence,
            tags=tags,
        )
        if task is not None:
            created.append(task)

    return created
