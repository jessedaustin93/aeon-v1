"""Append-only memory consolidation for Aeon-V1.

Consolidation detects likely duplicate or overlapping memories and writes a new
consensus record. It never deletes, rewrites, or merges source memories.
"""
import json
import re
from typing import Dict, List, Optional, Set

from .config import Config
from .memory_store import MemoryStore, _generate_id, _make_title, _wikilink
from .time_utils import utc_now_iso

_DEFAULT_TYPES = ["episodic", "semantic"]
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "but",
    "can", "could", "for", "from", "had", "has", "have", "into", "its",
    "need", "not", "that", "the", "their", "then", "there", "this", "with",
    "would", "you", "your",
}


def consolidate_memories(
    config: Optional[Config] = None,
    memory_types: Optional[List[str]] = None,
) -> Dict:
    """Create append-only consensus records for likely duplicate memories."""
    from .write_guard import assert_write_authorized
    assert_write_authorized("consolidate_memories")

    cfg = config or Config()
    cfg.ensure_dirs()
    memory_types = memory_types or _DEFAULT_TYPES

    memories = _load_memories(cfg, memory_types)
    groups = _find_duplicate_groups(
        memories,
        threshold=cfg.consolidation_similarity_threshold,
    )
    existing = _existing_source_sets(cfg)

    created: List[Dict] = []
    skipped_existing = 0
    for group in groups:
        source_key = frozenset(m["id"] for m in group)
        if source_key in existing:
            skipped_existing += 1
            continue
        record = _store_consolidation(group, cfg)
        created.append(record)
        existing.add(source_key)
        if len(created) >= cfg.max_consolidations_per_pass:
            break

    return {
        "created": created,
        "created_count": len(created),
        "candidate_groups": len(groups),
        "skipped_existing": skipped_existing,
        "memory_types": memory_types,
    }


def _load_memories(config: Config, memory_types: List[str]) -> List[Dict]:
    store = MemoryStore(config)
    memories: List[Dict] = []
    for memory_type in memory_types:
        for memory in store.list_memories(memory_type):
            if memory.get("id") and _memory_text(memory):
                memories.append(memory)
    return memories


def _find_duplicate_groups(memories: List[Dict], threshold: float) -> List[List[Dict]]:
    if len(memories) < 2:
        return []

    parent = {m["id"]: m["id"] for m in memories}
    by_id = {m["id"]: m for m in memories}
    word_sets = {m["id"]: _word_set(_memory_text(m)) for m in memories}

    def find(mid: str) -> str:
        while parent[mid] != mid:
            parent[mid] = parent[parent[mid]]
            mid = parent[mid]
        return mid

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(memories):
        for right in memories[i + 1:]:
            score = _similarity_score(left, right, word_sets)
            if score >= threshold:
                union(left["id"], right["id"])

    grouped: Dict[str, List[Dict]] = {}
    for mid in parent:
        grouped.setdefault(find(mid), []).append(by_id[mid])

    groups = [g for g in grouped.values() if len(g) >= 2]
    groups.sort(key=lambda g: (-len(g), ",".join(sorted(m["id"] for m in g))))
    return groups


def _store_consolidation(group: List[Dict], config: Config) -> Dict:
    con_id = _generate_id()
    now = utc_now_iso()
    source_ids = [m["id"] for m in group]
    links = [_wikilink(_vault_dir(m), m["id"], m.get("title")) for m in group]
    tags = sorted({tag for m in group for tag in m.get("tags", [])})
    consensus = _consensus_text(group)
    title = _make_title(consensus, max_words=8) or f"consensus-{con_id}"

    record = {
        "id": con_id,
        "title": title,
        "type": "consolidation",
        "created": now,
        "source": "consolidation_agent",
        "content": consensus,
        "source_ids": source_ids,
        "source_types": {
            t: sum(1 for m in group if m.get("type") == t)
            for t in sorted({m.get("type", "") for m in group})
        },
        "tags": tags,
        "links": links,
    }

    (config.memory_path / "consolidations" / f"{con_id}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    body = (
        f"# {title}\n\n"
        "## Consensus\n"
        f"{consensus}\n\n"
        "## Sources\n"
        + "\n".join(f"- {link}" for link in links)
        + "\n\n## Safety\n"
        "This is an append-only consensus record. Source memories were not deleted, rewritten, or merged.\n\n"
        "[[Semantic Memory]] | [[Episodic Memory]] | [[Reflections]]"
    )
    frontmatter = [
        "---",
        f"id: {con_id}",
        f"title: {title}",
        "type: consolidation",
        f"created: {now}",
        "source: consolidation_agent",
        "tags:",
    ]
    frontmatter.extend(f"  - {tag}" for tag in tags)
    frontmatter.append("links:")
    frontmatter.extend(f"  - {link}" for link in links)
    frontmatter.append("---")
    (config.vault_path / "consolidations" / f"{con_id}.md").write_text(
        "\n".join(frontmatter) + "\n\n" + body,
        encoding="utf-8",
    )
    return record


def _existing_source_sets(config: Config) -> Set[frozenset]:
    result: Set[frozenset] = set()
    con_dir = config.memory_path / "consolidations"
    if not con_dir.exists():
        return result
    for path in con_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_ids = data.get("source_ids", [])
        if len(source_ids) >= 2:
            result.add(frozenset(source_ids))
    return result


def _memory_text(memory: Dict) -> str:
    parts = []
    for field in ("concept", "summary", "description", "content", "text", "title"):
        value = memory.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def _word_set(text: str) -> Set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOP_WORDS
    }


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _similarity_score(left: Dict, right: Dict, word_sets: Dict[str, Set[str]]) -> float:
    score = _jaccard(word_sets[left["id"]], word_sets[right["id"]])
    shared_tags = set(left.get("tags", [])) & set(right.get("tags", []))
    if shared_tags:
        score += 0.15
    return min(score, 1.0)


def _consensus_text(group: List[Dict]) -> str:
    common_words = set.intersection(*[_word_set(_memory_text(m)) for m in group])
    useful_terms = sorted(common_words)[:12]
    source_bits = [_short_text(_memory_text(m)) for m in group]

    lines = [
        f"{len(group)} memories appear to describe the same or strongly overlapping information.",
    ]
    if useful_terms:
        lines.append(f"Shared terms: {', '.join(useful_terms)}.")
    lines.append("Consensus:")
    for bit in source_bits:
        lines.append(f"- {bit}")
    return "\n".join(lines)


def _short_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _vault_dir(memory: Dict) -> str:
    mem_type = memory.get("type", "raw")
    if mem_type == "reflection":
        return "reflections"
    if mem_type == "consolidation":
        return "consolidations"
    return mem_type
