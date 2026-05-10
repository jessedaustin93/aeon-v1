"""Local embedding support via LM Studio's /v1/embeddings endpoint.

Uses whatever embedding model is currently loaded in LM Studio (e.g. Qwen3-Embedding-0.6B).
No new Python dependencies — same urllib approach as the rest of the LM Studio calls.
Falls back gracefully (returns None) when the server is unreachable or no embedding
model is loaded, so callers can degrade to Jaccard without crashing.
"""
import json
import math
import time
import urllib.request
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

# Per-process cache keyed by (model, text_prefix) so the same memory text is
# not re-embedded within a single consolidation pass.
_embedding_cache: Dict[Tuple[str, str], List[float]] = {}
_failure_cooldown_until: Dict[Tuple[str, str], float] = {}
_CACHE_MAX = 2000


def get_embedding(text: str, config: "Config") -> Optional[List[float]]:
    """Return the embedding vector for text, or None on any failure.

    Uses config.embedding_model if set, otherwise lets LM Studio choose
    whatever model is currently loaded.
    """
    if not getattr(config, "embedding_enabled", True):
        return None
    if not text or not text.strip():
        return None

    base_url = getattr(config, "llm_base_url", "http://localhost:1234/v1")
    model = str(getattr(config, "embedding_model", "") or "")
    endpoint_key = (base_url.rstrip("/"), model)
    if time.monotonic() < _failure_cooldown_until.get(endpoint_key, 0.0):
        return None

    cache_key = (model, text[:400])
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    url = f"{base_url.rstrip('/')}/embeddings"
    payload_dict: Dict = {"input": text[:2000]}
    if model:
        payload_dict["model"] = model
    payload = json.dumps(payload_dict).encode("utf-8")

    timeout = int(getattr(config, "embedding_timeout_seconds", 10) or 10)
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vector = data["data"][0]["embedding"]
        if not isinstance(vector, list) or not vector:
            return None
        # Trim cache before inserting
        if len(_embedding_cache) >= _CACHE_MAX:
            _embedding_cache.clear()
        _embedding_cache[cache_key] = vector
        _failure_cooldown_until.pop(endpoint_key, None)
        return vector
    except Exception:
        cooldown = int(getattr(config, "embedding_failure_cooldown_seconds", 300) or 0)
        if cooldown > 0:
            _failure_cooldown_until[endpoint_key] = time.monotonic() + cooldown
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors, pure Python — no numpy needed."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (mag_a * mag_b)))


def clear_cache() -> None:
    """Clear the in-process embedding cache (useful in tests)."""
    _embedding_cache.clear()
    _failure_cooldown_until.clear()
