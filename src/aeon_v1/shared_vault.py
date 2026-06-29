"""Read-only access to the shared Master Vault.

The Master Vault is collaboration context, not an Aeon memory backend. This
module therefore exposes reads only. Runtime memories continue to use
``memory/`` and Aeon's local ``vault/``.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .config import Config


STARTUP_NOTES = (
    "AI/SharedMemory.md",
    "AI/Workflow.md",
    "AI/MemoryRules.md",
    "AI/AEON.md",
    "AI/ProjectStatus.md",
)
_IGNORED_PARTS = {".git", ".obsidian", ".trash"}
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "from", "about", "is", "are", "was", "were", "what", "when",
    "where", "why", "how", "who", "tell", "show", "me", "my", "you",
    "your", "this", "that", "these", "those", "please",
}


def is_available(config: Config) -> bool:
    path = config.master_vault_path
    return bool(config.master_vault_enabled and path and path.is_dir())


def load_startup_context(config: Config, max_chars_per_note: int = 12_000) -> str:
    """Load the stable shared operating notes in their defined order."""
    if not is_available(config):
        return ""
    sections: List[str] = []
    assert config.master_vault_path is not None
    for relative in STARTUP_NOTES:
        path = config.master_vault_path / relative
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if content:
            sections.append(f"[Master Vault: {relative}]\n{content[:max_chars_per_note]}")
    return "\n\n".join(sections)


def search_shared_vault(query: str, config: Config, limit: int = 5) -> List[Dict]:
    """Search shared Markdown without copying it into local memory."""
    if not is_available(config) or limit <= 0:
        return []
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    tokens = [
        word for word in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", query_lower)
        if word not in _STOP_WORDS
    ][:8]
    results: List[Dict] = []
    assert config.master_vault_path is not None
    for path in config.master_vault_path.rglob("*.md"):
        if any(part in _IGNORED_PARTS for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        lowered = content.lower()
        score = 10.0 if query_lower in lowered else 0.0
        matched = [token for token in tokens if token in lowered]
        if matched:
            score += len(matched) + len(matched) / max(1, len(tokens))
        if score <= 0:
            continue
        relative = path.relative_to(config.master_vault_path).as_posix()
        results.append({
            "match_type": "master-vault",
            "file": str(path),
            "memory": {
                "id": f"master-vault:{relative}",
                "type": "shared-context",
                "title": _title(content, path.stem),
                "content": _excerpt(content, query_lower, tokens),
                "source": "master-vault",
                "path": relative,
            },
            "score": score,
        })
    results.sort(key=lambda result: (-float(result["score"]), result["memory"]["path"]))
    return results[:limit]


def _title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _excerpt(content: str, query: str, tokens: List[str], limit: int = 320) -> str:
    plain = " ".join(line.strip() for line in content.splitlines() if not line.startswith("---"))
    lowered = plain.lower()
    positions = [lowered.find(query)] if query else []
    positions.extend(lowered.find(token) for token in tokens)
    positions = [position for position in positions if position >= 0]
    start = max(0, (min(positions) if positions else 0) - 80)
    excerpt = plain[start:start + limit].strip()
    if start:
        excerpt = "..." + excerpt
    if start + limit < len(plain):
        excerpt += "..."
    return excerpt
