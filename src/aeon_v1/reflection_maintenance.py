"""Maintenance helpers for keeping reflection memory useful.

Reflection records are append-only, but the active reflection folder should not
be allowed to become a wall of near-identical maintenance output. This module
archives older/low-signal reflections out of the active search path while
writing one compact consolidation record that preserves the themes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import Config
from .memory_store import _generate_id, _make_title, _vault_note_path
from .time_utils import utc_now_iso


_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "but", "can", "could", "for", "from", "had", "has", "have", "into",
    "its", "memory", "memories", "not", "that", "the", "their", "then",
    "there", "this", "with", "would", "you", "your",
}


def minimize_reflections(
    config: Optional[Config] = None,
    keep_active: int = 80,
    keep_recent: int = 30,
) -> Dict:
    """Archive older reflection bloat and write a compact summary record.

    The archive move is non-destructive: JSON records go under
    memory/reflections/archive/<run-id>/ and generated Markdown mirrors go under
    vault/_generated/reflections/archive/<run-id>/. MemoryStore.list_memories()
    only reads top-level *.json, so archived reflections stop crowding active
    search/reflection context.
    """
    cfg = config or Config()
    cfg.ensure_dirs()
    reflection_dir = cfg.memory_path / "reflections"
    records = _load_reflections(reflection_dir)
    if len(records) <= keep_active:
        return {
            "archived_count": 0,
            "kept_count": len(records),
            "message": "Reflection count is already within the active limit.",
        }

    keep_active = max(1, int(keep_active))
    keep_recent = max(0, min(int(keep_recent), keep_active))

    sorted_records = sorted(records, key=lambda r: r.get("created", ""))
    recent = sorted_records[-keep_recent:] if keep_recent else []
    keep_ids: Set[str] = {r["id"] for r in recent}

    remaining = [r for r in sorted_records if r["id"] not in keep_ids]
    for record in sorted(remaining, key=_reflection_score, reverse=True):
        if len(keep_ids) >= keep_active:
            break
        signature = _signature(record)
        if signature and any(_signature(k) == signature for k in records if k["id"] in keep_ids):
            continue
        keep_ids.add(record["id"])

    if len(keep_ids) < keep_active:
        for record in sorted(remaining, key=_reflection_score, reverse=True):
            if len(keep_ids) >= keep_active:
                break
            keep_ids.add(record["id"])

    archived = [r for r in records if r["id"] not in keep_ids]
    archive_id = utc_now_iso().replace(":", "").replace("+", "Z").replace(".", "-")
    archive_dir = reflection_dir / "archive" / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    vault_archive_dir = cfg.vault_path / "_generated" / "reflections" / "archive" / archive_id
    vault_archive_dir.mkdir(parents=True, exist_ok=True)

    for record in archived:
        rid = record["id"]
        src = reflection_dir / f"{rid}.json"
        if src.exists():
            src.replace(archive_dir / src.name)
        for vault_root in (cfg.vault_path / "_generated" / "reflections", cfg.vault_path / "reflections"):
            md = vault_root / f"{rid}.md"
            if md.exists():
                md.replace(vault_archive_dir / md.name)

    summary = _store_reflection_compaction_summary(
        cfg,
        archived=archived,
        kept=[r for r in records if r["id"] in keep_ids],
        archive_dir=archive_dir,
    )
    return {
        "archived_count": len(archived),
        "kept_count": len(keep_ids),
        "archive_dir": str(archive_dir),
        "summary": summary,
    }


def _load_reflections(reflection_dir: Path) -> List[Dict]:
    records: List[Dict] = []
    if not reflection_dir.exists():
        return records
    for path in reflection_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("id"):
            records.append(data)
    return records


def _reflection_score(record: Dict) -> float:
    source_types = record.get("source_types", {}) or {}
    total_sources = max(1, sum(int(v or 0) for v in source_types.values()))
    raw_ratio = int(source_types.get("raw", 0) or 0) / total_sources
    confidence = float(record.get("confidence", 0.0) or 0.0)
    tasks = len(record.get("suggested_tasks", []) or [])
    core = len(record.get("suggested_core_updates", []) or [])
    patterns = len(record.get("detected_patterns", []) or [])
    return confidence + (1.0 - raw_ratio) * 0.5 + min(tasks, 4) * 0.05 + min(core, 3) * 0.06 + min(patterns, 4) * 0.03


def _signature(record: Dict) -> str:
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", str(record.get("content", "")).lower())
        if len(word) > 3 and word not in _STOP_WORDS
    ]
    common = [word for word, _ in Counter(words).most_common(8)]
    return " ".join(common)


def _store_reflection_compaction_summary(
    config: Config,
    archived: List[Dict],
    kept: List[Dict],
    archive_dir: Path,
) -> Dict:
    con_id = _generate_id()
    now = utc_now_iso()
    tags = sorted({tag for r in archived for tag in r.get("tags", [])})[:20]
    top_words = Counter()
    source_types = Counter()
    for record in archived:
        top_words.update(_signature(record).split())
        source_types.update(record.get("source_types", {}) or {})

    themes = [word for word, _ in top_words.most_common(12)]
    content = (
        f"Archived {len(archived)} repetitive reflection records out of active memory. "
        f"Kept {len(kept)} higher-signal or recent reflections active. "
        f"Dominant archived themes: {', '.join(themes) if themes else 'none detected'}. "
        f"Archived originals remain available at {archive_dir}."
    )
    title = _make_title(content, max_words=8) or f"reflection-compaction-{con_id}"
    record = {
        "id": con_id,
        "title": title,
        "type": "consolidation",
        "created": now,
        "source": "reflection_maintenance",
        "content": content,
        "source_ids": [r["id"] for r in archived[:200]],
        "archived_count": len(archived),
        "kept_count": len(kept),
        "archived_reflection_ids": [r["id"] for r in archived],
        "kept_reflection_ids": [r["id"] for r in kept],
        "source_types": dict(source_types),
        "tags": tags,
        "links": [],
        "archive_dir": str(archive_dir),
    }

    con_dir = config.memory_path / "consolidations"
    con_dir.mkdir(parents=True, exist_ok=True)
    (con_dir / f"{con_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    body = (
        f"# {title}\n\n"
        "## Reflection Compaction\n"
        f"{content}\n\n"
        "## Active Policy\n"
        "Older repetitive reflections were archived out of active memory. "
        "The archive is preserved for audit, but active search/reflection now sees the compact signal.\n\n"
        "[[Reflections]] | [[Semantic Memory]] | [[Episodic Memory]]"
    )
    _vault_note_path(config, "consolidations", con_id).write_text(body, encoding="utf-8")
    return record
