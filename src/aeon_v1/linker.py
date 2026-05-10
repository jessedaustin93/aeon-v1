import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .exceptions import CoreMemoryProtectedError
from .memory_store import _classify_memory, _ensure_topic_note, _topic_link, _vault_note_path, _wikilink

_VAULT_DIR_MAP: Dict[str, str] = {
    "raw":        "raw",
    "episodic":   "episodic",
    "semantic":   "semantic",
    "reflection": "reflections",
    "consolidation": "consolidations",
    "media":      "media",
    "core":       "core",       # present so paths resolve correctly before the guard fires
}


def link_memories(config: Optional[Config] = None, max_related: int = 8) -> Dict[str, List[str]]:
    """Add high-signal Obsidian wikilinks between related notes.

    Returns a map of memory_id -> [related_id, ...] for inspection.
    Links use [[subdir/id|Readable Title]] format when titles are available.
    """
    if config is None:
        config = Config()

    all_memories = _load_all_memories(config)
    link_map: Dict[str, List[str]] = {}

    for mem in all_memories:
        mem_id = mem.get("id")
        if not mem_id:
            continue

        related = _rank_related(mem, all_memories, max_related=max_related)

        if related:
            link_map[mem_id] = [r["id"] for r in related]
            _update_markdown_links(mem, related, config)

    return link_map


def _load_all_memories(config: Config) -> List[Dict]:
    memories = []
    for mem_type in ["raw", "episodic", "semantic", "reflections", "consolidations", "media"]:
        mem_dir = config.memory_path / mem_type
        if not mem_dir.exists():
            continue
        for f in mem_dir.glob("*.json"):
            try:
                memories.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    return memories


def _rank_related(memory: Dict, all_memories: List[Dict], max_related: int) -> List[Dict]:
    """Pick a small, useful neighbor set instead of making an all-to-all blob."""
    scored = []
    mem_id = memory.get("id")
    mem_tags = set(memory.get("tags", []))
    mem_category = _memory_category(memory)
    mem_type = memory.get("type", "")
    raw_ref = memory.get("raw_ref")
    source_ids = set(memory.get("source_ids", []))

    for other in all_memories:
        if other.get("id") == mem_id:
            continue
        score = 0.0
        other_id = other.get("id")
        other_tags = set(other.get("tags", []))
        shared_tags = mem_tags & other_tags
        if mem_category != "general" and mem_category == _memory_category(other):
            score += 4.0
        if shared_tags:
            score += min(3.0, len(shared_tags) * 1.0)
        if raw_ref and other_id == raw_ref:
            score += 6.0
        if other.get("raw_ref") == mem_id:
            score += 5.0
        if other_id in source_ids or mem_id in set(other.get("source_ids", [])):
            score += 4.0
        if mem_type and mem_type == other.get("type"):
            score -= 0.25
        if score >= 2.0:
            scored.append((score, other))

    scored.sort(
        key=lambda item: (
            item[0],
            float(item[1].get("importance", 0) or 0),
            str(item[1].get("created", "")),
        ),
        reverse=True,
    )
    return [other for _, other in scored[:max_related]]


def _update_markdown_links(memory: Dict, related_memories: List[Dict], config: Config) -> None:
    """Append or replace the ## Related Memories section in a vault note.

    Uses [[subdir/id|Title]] format when a title field is present on the
    related memory, falling back to [[subdir/id]] otherwise.

    vault/core/ files are always skipped when allow_core_modification is False —
    core memory is human-gated and must not be modified by automated link passes.
    """
    vault_dir_name = _VAULT_DIR_MAP.get(memory.get("type", "raw"), "raw")
    md_path = _vault_note_path(config, vault_dir_name, memory["id"])

    # Core memory protection: skip any file that lives inside vault/core/
    if not config.allow_core_modification:
        core_dir = config.vault_path / "core"
        try:
            md_path.relative_to(core_dir)
            return  # silently skip — vault/core/ is human-gated
        except ValueError:
            pass  # not inside core_dir — safe to proceed

    if not md_path.exists():
        return

    category = _memory_category(memory)
    if category and category != "general":
        _ensure_topic_note(config, category)

    link_lines = []
    if category and category != "general":
        link_lines.append(f"**Topic:** {_topic_link(category)}")
        link_lines.append("")
    for r in related_memories:
        rid   = r["id"]
        rtype = r.get("type", "raw")
        rdir  = _VAULT_DIR_MAP.get(rtype, rtype)
        rtitle = r.get("title")
        link_lines.append(f"- {_wikilink(rdir, rid, rtitle)}")

    link_section = "\n\n## Related Memories\n" + "\n".join(link_lines)

    content = md_path.read_text(encoding="utf-8")
    if "## Related Memories" in content:
        content = re.sub(
            r"\n\n## Related Memories\n.*",
            link_section,
            content,
            flags=re.DOTALL,
        )
    else:
        content += link_section

    md_path.write_text(content, encoding="utf-8")


def _memory_text(memory: Dict) -> str:
    parts = []
    for field in ("text", "summary", "concept", "description", "content", "title"):
        value = memory.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _memory_category(memory: Dict) -> str:
    existing = str(memory.get("category", "")).strip()
    if existing:
        return existing
    return _classify_memory(
        _memory_text(memory),
        tags=memory.get("tags", []),
        source=str(memory.get("source", "")),
    )
