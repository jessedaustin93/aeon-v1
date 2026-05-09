import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .exceptions import CoreMemoryProtectedError
from .memory_store import _wikilink

_VAULT_DIR_MAP: Dict[str, str] = {
    "raw":        "raw",
    "episodic":   "episodic",
    "semantic":   "semantic",
    "reflection": "reflections",
    "consolidation": "consolidations",
    "media":      "media",
    "core":       "core",       # present so paths resolve correctly before the guard fires
}


def link_memories(config: Optional[Config] = None) -> Dict[str, List[str]]:
    """Add Obsidian wikilinks between notes that share at least one tag.

    Returns a map of memory_id -> [related_id, ...] for inspection.
    Links use [[subdir/id|Readable Title]] format when titles are available.
    """
    if config is None:
        config = Config()

    all_memories = _load_all_memories(config)
    link_map: Dict[str, List[str]] = {}

    for mem in all_memories:
        mem_id = mem.get("id")
        mem_tags = set(mem.get("tags", []))
        if not mem_id or not mem_tags:
            continue

        related = [
            other
            for other in all_memories
            if other.get("id") != mem_id and mem_tags & set(other.get("tags", []))
        ]

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


def _update_markdown_links(memory: Dict, related_memories: List[Dict], config: Config) -> None:
    """Append or replace the ## Related Memories section in a vault note.

    Uses [[subdir/id|Title]] format when a title field is present on the
    related memory, falling back to [[subdir/id]] otherwise.

    vault/core/ files are always skipped when allow_core_modification is False —
    core memory is human-gated and must not be modified by automated link passes.
    """
    vault_dir_name = _VAULT_DIR_MAP.get(memory.get("type", "raw"), "raw")
    md_path = config.vault_path / vault_dir_name / f"{memory['id']}.md"

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

    link_lines = []
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
