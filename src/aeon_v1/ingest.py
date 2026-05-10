import re
from typing import Dict, Optional

from .config import Config
from .llm import score_importance
from .memory_store import MemoryStore, _extract_tags, _score_importance

# Semantic extraction: episodic -> semantic
SEMANTIC_KEYWORDS = [
    "concept", "principle", "rule", "pattern", "definition",
    "always", "never", "means", "is defined as",
]


def ingest(text: str, source: str = "manual", config: Optional[Config] = None) -> Dict:
    """Ingest raw text into the memory system.

    Always stores a raw memory (verbatim, immutable).
    Promotes to episodic if importance >= threshold,
    and further to semantic if semantic keywords are present.
    """
    from .write_guard import assert_write_authorized
    assert_write_authorized("ingest")
    if config is None:
        config = Config()

    store = MemoryStore(config)

    # Raw memory is always stored exactly as received — never modified after this call.
    raw = store.store_raw(text, source=source)
    result: Dict = {"raw": raw, "episodic": None, "semantic": None}

    # Rule-based score from raw memory; LLM refinement applied below if enabled.
    importance = raw["importance"]
    tags = raw["tags"]

    # LLM importance scoring — runs on the background/search model (Mistral when
    # using LM Studio) so it doesn't block the chat response path.
    llm_score = score_importance(text, config)
    if llm_score is not None:
        importance = llm_score

    if importance >= config.importance_threshold:
        summary = _make_summary(text)
        episodic = store.store_episodic(
            summary=summary,
            raw_id=raw["id"],
            tags=tags,
            importance=importance,
            source=source,
            raw_title=raw.get("title"),   # enables readable wikilink in episodic note
        )
        result["episodic"] = episodic

        concept = _extract_concept(text)
        text_lower = text.lower()
        if concept and any(kw in text_lower for kw in SEMANTIC_KEYWORDS):
            description = _make_semantic_description(text, concept)
            semantic = store.store_semantic(
                concept=concept,
                description=description,
                tags=tags,
                importance=importance,
                source=source,
            )
            result["semantic"] = semantic

    return result


def _make_summary(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return text[:150]
    first = lines[0]
    return first[:147] + "..." if len(first) > 150 else first


def _extract_concept(text: str) -> Optional[str]:
    """Naive concept extraction: quoted terms or Title Case phrases."""
    quoted = re.search(r'"([^"]{3,60})"', text)
    if quoted:
        return quoted.group(1)
    title_case = re.search(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', text)
    if title_case:
        return title_case.group(1)
    return None


def _make_semantic_description(text: str, concept: str) -> str:
    idx = text.lower().find(concept.lower())
    if idx == -1:
        return text[:200]
    start = max(0, idx - 20)
    end = min(len(text), idx + 200)
    return text[start:end].strip()
