import json
import re
import uuid
from datetime import datetime
try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # Python 3.10
    from datetime import timezone as _timezone
    UTC = _timezone.utc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .exceptions import CoreMemoryProtectedError
from .time_utils import utc_now_iso

# ---------------------------------------------------------------------------
# Importance scoring
#
# Multi-word phrases are listed before their component words so they match
# (and add weight) before the shorter word would also match.
# Swap _score_importance with an LLM confidence call when ready.
# ---------------------------------------------------------------------------

_IMPORTANCE_SIGNALS: List[Tuple[str, float]] = [
    ("i learned",       0.20),
    ("key insight",     0.20),
    ("must not forget", 0.20),
    ("i noticed",       0.20),
    ("i'm uncertain",   0.20),
    ("critical",        0.15),
    ("important",       0.15),
    ("uncertain",       0.15),
    ("i discovered",    0.15),
    ("i found",         0.15),
    ("remember",        0.12),
    ("project",         0.12),
    ("goal",            0.12),
    ("i wonder",        0.12),
    ("discovered",      0.10),
    ("realized",        0.10),
    ("surprising",      0.10),
    ("need to",         0.08),
    ("should ",         0.05),
    ("will ",           0.05),
    ("always",          0.05),
    ("never",           0.05),
]
_IMPORTANCE_BASE = 0.10

_TAG_KEYWORD_MAP: Dict[str, str] = {
    "project":    "project",
    "goal":       "goal",
    "learned":    "learning",
    "important":  "important",
    "remember":   "recall",
    "error":      "error",
    "bug":        "bug",
    "idea":       "idea",
    "question":   "question",
    "task":       "task",
    "experiment": "experiment",
    "concept":    "concept",
    "pattern":    "pattern",
    "rule":       "rule",
    "preference": "preference",
    "prefer":     "preference",
    "dashboard":  "dashboard",
    "launcher":   "launcher",
    "obsidian":   "obsidian",
    "lm studio":  "lm-studio",
    "memory":     "memory",
    "phrase":     "phrase",
    "number":     "number",
    "image":      "media",
    "picture":    "media",
    "video":      "media",
    "sound":      "audio",
    "audio":      "audio",
}

_CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("test-memory", ("test", "testing", "canary", "remember the phrase", "six digit", "6 digit", "random number")),
    ("media", ("image", "picture", "photo", "screenshot", "video", "audio", "sound")),
    ("tools", ("lm studio", "obsidian", "github", "dashboard", "launcher", "codex", "assistant", "chatgpt")),
    ("bugs", ("bug", "broken", "fix", "wrong", "error", "issue", "not working")),
    ("preferences", ("prefer", "preference", "tone", "talk like", "respond", "style")),
    ("project", ("aeon", "project", "v1", "memory system", "recursive", "repo")),
    ("tasks", ("need to", "should", "todo", "task", "plan", "next step")),
    ("learning", ("learned", "insight", "realized", "noticed", "pattern", "principle")),
]

# Maps memory type -> vault subdirectory name
_VAULT_DIR: Dict[str, str] = {
    "raw":        "raw",
    "episodic":   "episodic",
    "semantic":   "semantic",
    "reflection": "reflections",
}

_GENERATED_VAULT_TYPES = {
    "raw", "episodic", "semantic", "reflections", "consolidations", "media",
    "agents", "tasks", "decisions", "simulations", "evaluations",
}

_CATEGORY_TITLES: Dict[str, str] = {
    "test-memory": "Test Memory",
    "media": "Media",
    "tools": "Tools",
    "bugs": "Bugs",
    "preferences": "Preferences",
    "project": "Project",
    "tasks": "Tasks",
    "learning": "Learning",
    "general": "General",
}


def _generate_id() -> str:
    return str(uuid.uuid4())[:8]


def _score_importance(text: str) -> float:
    """Weighted keyword + length heuristic for importance [0.0, 1.0].

    Signals are additive. Longer entries get a small length bonus.
    Replace this function body with an LLM confidence call when ready —
    the signature and return type do not need to change.
    """
    text_lower = text.lower()
    score = _IMPORTANCE_BASE
    for phrase, weight in _IMPORTANCE_SIGNALS:
        if phrase in text_lower:
            score += weight
    # Length bonus: longer entries tend to be more substantive
    n = len(text.strip())
    if n >= 300:
        score += 0.10
    elif n >= 100:
        score += 0.05
    return round(min(score, 1.0), 3)


def _extract_tags(text: str) -> List[str]:
    text_lower = text.lower()
    tags = []
    for kw, tag in _TAG_KEYWORD_MAP.items():
        if kw in text_lower and tag not in tags:
            tags.append(tag)
    return tags


def _classify_memory(text: str, tags: Optional[List[str]] = None, source: str = "") -> str:
    text_lower = text.lower()
    tag_set = set(tags or [])
    source_lower = source.lower()
    for category, signals in _CATEGORY_RULES:
        if any(signal in text_lower or signal in tag_set or signal in source_lower for signal in signals):
            return category
    return "general"


def _make_title(text: str, max_words: int = 6) -> str:
    """Short, filesystem-safe title derived from text content.

    Uses the first N alphanumeric words, lowercased and hyphen-joined.
    The ID is the stable permanent identifier; this title is only for
    human readability.
    """
    words = re.findall(r'[a-zA-Z0-9]+', text.strip())[:max_words]
    return '-'.join(w.lower() for w in words) or "untitled"


def _wikilink(vault_subdir: str, mem_id: str, title: Optional[str] = None) -> str:
    """Build an Obsidian-compatible wikilink.

    Format: [[subdir/id|Readable Title]] when a title is available,
    falling back to [[subdir/id]] when not.
    The subdir/id path resolves to vault/{subdir}/{id}.md.
    """
    if vault_subdir in _GENERATED_VAULT_TYPES:
        vault_subdir = f"_generated/{vault_subdir}"
    path = f"{vault_subdir}/{mem_id}" if vault_subdir else mem_id
    return f"[[{path}|{title}]]" if title else f"[[{path}]]"


def _vault_note_path(config: "Config", vault_subdir: str, mem_id: str) -> Path:
    """Return the Obsidian-facing path for generated memory notes."""
    if vault_subdir in _GENERATED_VAULT_TYPES:
        path = config.vault_path / "_generated" / vault_subdir / f"{mem_id}.md"
    else:
        path = config.vault_path / vault_subdir / f"{mem_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _topic_link(category: str) -> str:
    title = _CATEGORY_TITLES.get(category, category.replace("-", " ").title())
    return _wikilink("topics", category, f"Topic: {title}")


def _ensure_topic_note(config: "Config", category: str) -> None:
    topic_dir = config.vault_path / "topics"
    topic_dir.mkdir(parents=True, exist_ok=True)
    path = topic_dir / f"{category}.md"
    if path.exists():
        return
    title = _CATEGORY_TITLES.get(category, category.replace("-", " ").title())
    body = (
        f"# Topic: {title}\n\n"
        "This topic page is a stable Obsidian cluster point for related memories.\n\n"
        "[[index]]"
    )
    _write_markdown(
        path,
        frontmatter={
            "id": category,
            "title": f"topic-{category}",
            "type": "topic",
            "category": category,
            "tags": ["topic"],
            "links": [],
        },
        body=body,
        config=config,
    )


def _guard_core_path(path: Path, config: Optional[Config]) -> None:
    """Raise CoreMemoryProtectedError if path targets vault/core/ and protection is active.

    Called by _write_markdown before every disk write.  Passing config=None skips
    the guard (only happens in tests that call _write_markdown directly without a
    config — in all production code paths a Config is always supplied).
    """
    if config is None or config.allow_core_modification:
        return
    core_dir = config.vault_path / "core"
    try:
        path.relative_to(core_dir)
        raise CoreMemoryProtectedError(
            f"Write to '{path}' is blocked: vault/core/ is protected. "
            "Set config.allow_core_modification = True to override."
        )
    except ValueError:
        pass  # path is not inside core_dir — safe to proceed


def _write_markdown(path: Path, frontmatter: Dict, body: str, config: Optional[Config] = None) -> None:
    _guard_core_path(path, config)
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    path.write_text("\n".join(lines) + "\n\n" + body, encoding="utf-8")


def _notify_memory_created(config: Config, memory_type: str) -> None:
    """Notify background maintenance without making memory writes depend on it."""
    try:
        from .background_consolidation import notify_memory_created
        notify_memory_created(config, memory_type)
    except Exception:
        pass


class MemoryStore:
    def __init__(self, config: Config):
        self.config = config
        config.ensure_dirs()

    # ------------------------------------------------------------------ raw --

    def store_raw(self, text: str, source: str = "manual") -> Dict:
        """Write a verbatim raw memory. Never called again on the same record."""
        mem_id = _generate_id()
        now = utc_now_iso()
        importance = _score_importance(text)
        tags = _extract_tags(text)
        category = _classify_memory(text, tags=tags, source=source)
        title = _make_title(text)
        _ensure_topic_note(self.config, category)

        memory = {
            "id": mem_id,
            "title": title,
            "type": "raw",
            "created": now,
            "source": source,
            "text": text,         # immutable — never rewritten after this point
            "importance": importance,
            "category": category,
            "tags": tags,
            "links": [],
        }

        (self.config.memory_path / "raw" / f"{mem_id}.json").write_text(
            json.dumps(memory, indent=2), encoding="utf-8"
        )
        _write_markdown(
            _vault_note_path(self.config, "raw", mem_id),
            frontmatter={
                "id": mem_id,
                "title": title,
                "type": "raw",
                "created": now,
                "source": source,
                "importance": importance,
                "category": category,
                "tags": tags,
                "links": [],
            },
            body=(
                f"# {title}\n\n"
                f"{text}\n\n"
                f"[[Raw Memory]] | {_topic_link(category)}"
            ),
            config=self.config,
        )
        _notify_memory_created(self.config, "raw")
        return memory

    # --------------------------------------------------------------- episodic -

    def store_episodic(
        self,
        summary: str,
        raw_id: str,
        tags: List[str],
        importance: float,
        source: str,
        raw_title: Optional[str] = None,
    ) -> Dict:
        mem_id = _generate_id()
        now = utc_now_iso()
        title = _make_title(summary)
        raw_link = _wikilink("raw", raw_id, raw_title)
        category = _classify_memory(summary, tags=tags, source=source)
        _ensure_topic_note(self.config, category)

        memory = {
            "id": mem_id,
            "title": title,
            "type": "episodic",
            "created": now,
            "source": source,
            "summary": summary,
            "raw_ref": raw_id,
            "importance": importance,
            "category": category,
            "tags": tags,
            "links": [raw_link],
        }

        (self.config.memory_path / "episodic" / f"{mem_id}.json").write_text(
            json.dumps(memory, indent=2), encoding="utf-8"
        )
        _write_markdown(
            _vault_note_path(self.config, "episodic", mem_id),
            frontmatter={
                "id": mem_id,
                "title": title,
                "type": "episodic",
                "created": now,
                "source": source,
                "importance": importance,
                "category": category,
                "tags": tags,
                "links": [raw_link],
            },
            body=(
                f"# {title}\n\n"
                f"**Summary:** {summary}\n\n"
                f"**Source:** {raw_link}\n\n"
                f"[[Episodic Memory]] | {_topic_link(category)}"
            ),
            config=self.config,
        )
        _notify_memory_created(self.config, "episodic")
        return memory

    # --------------------------------------------------------------- semantic -

    def store_semantic(
        self,
        concept: str,
        description: str,
        tags: List[str],
        importance: float,
        source: str,
    ) -> Dict:
        mem_id = _generate_id()
        now = utc_now_iso()
        title = _make_title(concept)
        category = _classify_memory(f"{concept}\n{description}", tags=tags, source=source)
        _ensure_topic_note(self.config, category)

        memory = {
            "id": mem_id,
            "title": title,
            "type": "semantic",
            "created": now,
            "source": source,
            "concept": concept,
            "description": description,
            "importance": importance,
            "category": category,
            "tags": tags,
            "links": [],
        }

        (self.config.memory_path / "semantic" / f"{mem_id}.json").write_text(
            json.dumps(memory, indent=2), encoding="utf-8"
        )
        _write_markdown(
            _vault_note_path(self.config, "semantic", mem_id),
            frontmatter={
                "id": mem_id,
                "title": title,
                "type": "semantic",
                "created": now,
                "source": source,
                "importance": importance,
                "category": category,
                "tags": tags,
                "links": [],
            },
            body=(
                f"# {concept}\n\n"
                f"**Concept:** {concept}\n\n"
                f"**Description:** {description}\n\n"
                f"[[Semantic Memory]] | {_topic_link(category)}"
            ),
            config=self.config,
        )
        _notify_memory_created(self.config, "semantic")
        return memory

    # ------------------------------------------------------------- reflection -

    def store_reflection(
        self,
        content: str,
        source_ids: List[str],
        tags: List[str],
        source_titles: Optional[Dict[str, Tuple[str, str]]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Append a new reflection note.

        Each call always creates a new file — prior reflections are never
        overwritten or modified.

        source_titles: optional {id: (vault_subdir, display_title)} mapping
                       for readable Obsidian wikilinks.
        metadata:      optional dict of Layer 2 fields (confidence, source_types,
                       suggested_tasks, etc.) spread directly into the JSON record.
        """
        mem_id = _generate_id()
        now = utc_now_iso()
        title = f"reflection-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}"

        links: List[str] = []
        for sid in source_ids:
            if source_titles and sid in source_titles:
                subdir, display = source_titles[sid]
                links.append(_wikilink(subdir, sid, display))
            else:
                links.append(f"[[{sid}]]")

        memory = {
            "id": mem_id,
            "title": title,
            "type": "reflection",
            "created": now,
            "source": "reflection_engine",
            "content": content,
            "source_ids": source_ids,
            "tags": tags,
            "links": links,
            **(metadata or {}),
        }

        (self.config.memory_path / "reflections" / f"{mem_id}.json").write_text(
            json.dumps(memory, indent=2), encoding="utf-8"
        )
        _write_markdown(
            _vault_note_path(self.config, "reflections", mem_id),
            frontmatter={
                "id": mem_id,
                "title": title,
                "type": "reflection",
                "created": now,
                "source": "reflection_engine",
                "tags": tags,
                "links": links,
            },
            body=(
                f"# {title}\n\n"
                f"{content}\n\n"
                "**Sources:** "
                + ", ".join(links)
                + "\n\n[[Reflections]] | [[Episodic Memory]] | [[Semantic Memory]]"
            ),
            config=self.config,
        )
        return memory

    # ------------------------------------------------------------------ util --

    def list_memories(self, memory_type: str) -> List[Dict]:
        mem_dir = self.config.memory_path / memory_type
        if not mem_dir.exists():
            return []
        memories = []
        for f in mem_dir.glob("*.json"):
            try:
                memories.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return sorted(memories, key=lambda m: m.get("created", ""))
