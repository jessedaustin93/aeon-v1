import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .aging import age_weight
from .config import Config

# Fields checked during JSON memory search
_SEARCHABLE_FIELDS = ("text", "summary", "concept", "description", "content", "original_name")

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "about", "as", "into",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "should", "would", "will", "may", "might", "what",
    "when", "where", "why", "how", "who", "which", "tell", "show", "give",
    "me", "my", "your", "you", "i", "we", "it", "this", "that", "these",
    "those", "please", "remember", "memory", "memories", "anything",
}

_TYPE_PRIORITY = {
    "semantic": 6,
    "episodic": 5,
    "consolidation": 4,
    "consolidations": 4,
    "reflection": 3,
    "reflections": 3,
    "media": 2,
    "raw": 1,
}


def search(
    query: str,
    memory_types: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> List[Dict]:
    """Keyword search across JSON memory files and Markdown vault notes.

    Structured so vector/embedding search can replace the inner match logic later
    without changing the function signature or return shape.
    """
    if config is None:
        config = Config()
    if memory_types is None:
        memory_types = ["raw", "episodic", "semantic", "reflections", "consolidations", "media"]

    query_lower = query.lower().strip()
    query_tokens = _query_tokens(query_lower)
    results: List[Dict] = []
    seen_ids: set = set()

    for mem_type in memory_types:
        # Search structured JSON first (authoritative source)
        mem_dir = config.memory_path / mem_type
        if mem_dir.exists():
            for f in mem_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    score = _match_score(data, query_lower, query_tokens)
                    if score > 0:
                        results.append({
                            "match_type": mem_type,
                            "file": str(f),
                            "memory": data,
                            "score": score,
                        })
                        seen_ids.add(data.get("id", ""))
                except Exception:
                    pass

        # Also search vault Markdown (catches notes without a JSON counterpart).
        # New generated notes live under _generated; the legacy directory is
        # still searched so older local vaults keep working after an update.
        vault_dirs = [
            config.vault_path / "_generated" / mem_type,
            config.vault_path / mem_type,
        ]
        for vault_dir in vault_dirs:
            if not vault_dir.exists():
                continue
            for f in vault_dir.glob("*.md"):
                mem_id = f.stem
                if mem_id in seen_ids:
                    continue
                try:
                    content = f.read_text(encoding="utf-8")
                    score = _text_match_score(content, query_lower, query_tokens)
                    if score > 0:
                        results.append({
                            "match_type": f"vault/{mem_type}",
                            "file": str(f),
                            "memory": {"id": mem_id, "type": mem_type, "file": str(f)},
                            "score": score,
                        })
                        seen_ids.add(mem_id)
                except Exception:
                    pass

    results.sort(key=lambda r: _rank_result(r, config), reverse=True)
    return results


def _matches(data: Dict, query_lower: str) -> bool:
    return _match_score(data, query_lower, _query_tokens(query_lower)) > 0


def _query_tokens(query_lower: str) -> List[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", query_lower)
    tokens = [w for w in words if w not in _STOP_WORDS]
    return tokens[:8]


def _match_score(data: Dict, query_lower: str, query_tokens: List[str]) -> float:
    haystacks: List[str] = []
    for field in _SEARCHABLE_FIELDS:
        val = data.get(field)
        if isinstance(val, str) and val.strip():
            haystacks.append(val.lower())
    tags = data.get("tags", [])
    if isinstance(tags, list):
        haystacks.extend(str(t).lower() for t in tags)
    return max((_text_match_score(text, query_lower, query_tokens) for text in haystacks), default=0.0)


def _text_match_score(text: str, query_lower: str, query_tokens: List[str]) -> float:
    text_lower = text.lower()
    if not query_lower:
        return 0.0

    score = 0.0
    if query_lower in text_lower:
        score += 10.0

    if query_tokens:
        matched = [token for token in query_tokens if token in text_lower]
        if not matched:
            return score
        score += len(matched)
        score += len(matched) / max(1, len(query_tokens))

    return score


def _rank_result(result: Dict, config: Optional[Config] = None) -> tuple:
    memory = result.get("memory", {})
    mem_type = memory.get("type") or str(result.get("match_type", "")).split("/")[-1]
    importance = float(memory.get("importance", 0) or 0)
    created = str(
        memory.get("created")
        or memory.get("created_at")
        or memory.get("generated_at")
        or ""
    )
    imp = float(memory.get("importance", 0) or 0)
    w = age_weight(created, config, importance=imp) if config is not None else 1.0
    return (
        float(result.get("score", 0) or 0) * w,
        _TYPE_PRIORITY.get(str(mem_type), 0),
        importance,
        created,
    )
