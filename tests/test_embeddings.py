"""Tests for embedding-based consolidation."""
import json
import math
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from aeon_v1.config import Config
from aeon_v1.embeddings import clear_cache, cosine_similarity, get_embedding


def _cfg(**kwargs):
    cfg = Config()
    cfg.embedding_enabled = True
    cfg.embedding_model = ""
    cfg.embedding_timeout_seconds = 10
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _fake_embedding_response(vector: List[float]) -> bytes:
    return json.dumps({
        "data": [{"embedding": vector, "index": 0}],
        "model": "qwen3-embedding-0.6b",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }).encode("utf-8")


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        v = [0.1, 0.5, -0.3]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_give_minus_one(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, [-1.0, 0.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_give_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_empty_vectors_give_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_give_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vector_gives_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_similar_vectors_high_score(self):
        a = [0.9, 0.1, 0.05]
        b = [0.85, 0.12, 0.06]
        sim = cosine_similarity(a, b)
        assert sim > 0.99

    def test_dissimilar_vectors_low_score(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_result_clamped_to_minus_one_to_one(self):
        # Floating-point arithmetic can push dot/mag slightly over 1.0
        a = [1.0] * 100
        sim = cosine_similarity(a, a)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------

class TestGetEmbedding:
    def setup_method(self):
        clear_cache()

    def test_returns_none_when_disabled(self):
        cfg = _cfg(embedding_enabled=False)
        assert get_embedding("hello world", cfg) is None

    def test_returns_none_for_empty_text(self):
        cfg = _cfg()
        assert get_embedding("", cfg) is None
        assert get_embedding("   ", cfg) is None

    def test_returns_vector_on_success(self):
        vec = [0.1, 0.2, 0.3]
        cfg = _cfg()

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return _fake_embedding_response(vec)

        with patch("aeon_v1.embeddings.urllib.request.urlopen", return_value=FakeResp()):
            result = get_embedding("kernel panic debugging", cfg)

        assert result == vec

    def test_returns_none_on_connection_error(self):
        cfg = _cfg()
        with patch("aeon_v1.embeddings.urllib.request.urlopen", side_effect=OSError("refused")):
            result = get_embedding("some text", cfg)
        assert result is None

    def test_connection_error_opens_short_circuit(self):
        cfg = _cfg(embedding_failure_cooldown_seconds=300)
        with patch("aeon_v1.embeddings.urllib.request.urlopen", side_effect=OSError("refused")) as mock_urlopen:
            assert get_embedding("first text", cfg) is None
            assert get_embedding("second text", cfg) is None

        assert mock_urlopen.call_count == 1

    def test_returns_none_on_bad_response_shape(self):
        cfg = _cfg()

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return b'{"data": []}'

        with patch("aeon_v1.embeddings.urllib.request.urlopen", return_value=FakeResp()):
            result = get_embedding("some text", cfg)
        assert result is None

    def test_caches_result(self):
        vec = [0.5, 0.5]
        cfg = _cfg()
        call_count = 0

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self):
                nonlocal call_count
                call_count += 1
                return _fake_embedding_response(vec)

        with patch("aeon_v1.embeddings.urllib.request.urlopen", return_value=FakeResp()):
            r1 = get_embedding("hello", cfg)
            r2 = get_embedding("hello", cfg)

        assert r1 == r2
        assert call_count == 1  # second call served from cache

    def test_model_name_sent_when_configured(self):
        vec = [0.1]
        cfg = _cfg(embedding_model="qwen3-embedding-0.6b")
        captured = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return _fake_embedding_response(vec)

        def fake_urlopen(req, timeout=None):
            import json as _json
            captured["payload"] = _json.loads(req.data.decode())
            return FakeResp()

        with patch("aeon_v1.embeddings.urllib.request.urlopen", side_effect=fake_urlopen):
            get_embedding("test text", cfg)

        assert captured["payload"].get("model") == "qwen3-embedding-0.6b"

    def test_model_name_omitted_when_empty(self):
        vec = [0.1]
        cfg = _cfg(embedding_model="")
        captured = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return _fake_embedding_response(vec)

        def fake_urlopen(req, timeout=None):
            import json as _json
            captured["payload"] = _json.loads(req.data.decode())
            return FakeResp()

        with patch("aeon_v1.embeddings.urllib.request.urlopen", side_effect=fake_urlopen):
            get_embedding("test text", cfg)

        assert "model" not in captured["payload"]


# ---------------------------------------------------------------------------
# Consolidation integration — embedding path
# ---------------------------------------------------------------------------

class TestConsolidationEmbeddingPath:
    def test_embedding_consolidates_semantic_duplicates(self, tmp_path):
        """Memories with high cosine similarity but low Jaccard should consolidate."""
        from aeon_v1.consolidate import consolidate_memories
        from aeon_v1.embeddings import clear_cache
        clear_cache()

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.embedding_enabled = True
        cfg.embedding_similarity_threshold = 0.85
        cfg.consolidation_similarity_threshold = 0.72
        cfg.max_consolidations_per_pass = 5

        # Two episodic memories that are semantically similar but use different words
        m1 = {
            "id": "aaa00001", "type": "episodic",
            "summary": "System experienced a kernel panic during boot sequence",
            "importance": 0.7, "tags": ["bugs"], "created": "2026-05-01T00:00:00",
        }
        m2 = {
            "id": "bbb00002", "type": "episodic",
            "summary": "Machine crashed with fatal OS error on startup",
            "importance": 0.7, "tags": ["bugs"], "created": "2026-05-02T00:00:00",
        }
        mem_dir = tmp_path / "memory" / "episodic"
        (mem_dir / "aaa00001.json").write_text(json.dumps(m1), encoding="utf-8")
        (mem_dir / "bbb00002.json").write_text(json.dumps(m2), encoding="utf-8")

        # Provide high-similarity embeddings (cosine ≈ 0.99)
        vec_a = [1.0, 0.0, 0.1]
        vec_b = [0.99, 0.01, 0.1]

        def fake_embed(text, config):
            if "kernel" in text or "boot" in text:
                return vec_a
            if "crashed" in text or "startup" in text:
                return vec_b
            return None

        with patch("aeon_v1.consolidate.get_embedding", side_effect=fake_embed):
            result = consolidate_memories(config=cfg)

        assert result["created_count"] >= 1, "Expected at least one consolidation"

    def test_jaccard_fallback_when_embeddings_unavailable(self, tmp_path):
        """When get_embedding returns None, Jaccard still drives consolidation."""
        from aeon_v1.consolidate import consolidate_memories
        from aeon_v1.embeddings import clear_cache
        clear_cache()

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.embedding_enabled = True
        cfg.embedding_similarity_threshold = 0.85
        cfg.consolidation_similarity_threshold = 0.5  # lower so Jaccard fires
        cfg.max_consolidations_per_pass = 5

        # Nearly identical text — Jaccard will be high
        m1 = {
            "id": "ccc00001", "type": "episodic",
            "summary": "user prefers dark mode interface settings always",
            "importance": 0.6, "tags": [], "created": "2026-05-01T00:00:00",
        }
        m2 = {
            "id": "ddd00002", "type": "episodic",
            "summary": "user prefers dark mode interface settings always enabled",
            "importance": 0.6, "tags": [], "created": "2026-05-02T00:00:00",
        }
        mem_dir = tmp_path / "memory" / "episodic"
        (mem_dir / "ccc00001.json").write_text(json.dumps(m1), encoding="utf-8")
        (mem_dir / "ddd00002.json").write_text(json.dumps(m2), encoding="utf-8")

        with patch("aeon_v1.consolidate.get_embedding", return_value=None):
            result = consolidate_memories(config=cfg)

        assert result["created_count"] >= 1

    def test_no_consolidation_below_embedding_threshold(self, tmp_path):
        """Memories with low cosine similarity should not consolidate."""
        from aeon_v1.consolidate import consolidate_memories
        from aeon_v1.embeddings import clear_cache
        clear_cache()

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.embedding_enabled = True
        cfg.embedding_similarity_threshold = 0.85
        cfg.consolidation_similarity_threshold = 0.72
        cfg.max_consolidations_per_pass = 5

        m1 = {
            "id": "eee00001", "type": "episodic",
            "summary": "user enjoys hiking in the mountains on weekends",
            "importance": 0.5, "tags": [], "created": "2026-05-01T00:00:00",
        }
        m2 = {
            "id": "fff00002", "type": "episodic",
            "summary": "memory consolidation pipeline uses Jaccard similarity algorithm",
            "importance": 0.5, "tags": [], "created": "2026-05-02T00:00:00",
        }
        mem_dir = tmp_path / "memory" / "episodic"
        (mem_dir / "eee00001.json").write_text(json.dumps(m1), encoding="utf-8")
        (mem_dir / "fff00002.json").write_text(json.dumps(m2), encoding="utf-8")

        # Very different embeddings
        with patch("aeon_v1.consolidate.get_embedding", side_effect=lambda t, c:
                   [1.0, 0.0] if "hiking" in t else [0.0, 1.0]):
            result = consolidate_memories(config=cfg)

        assert result["created_count"] == 0

    def test_embedding_disabled_uses_jaccard_only(self, tmp_path):
        """With embedding_enabled=False, get_embedding is never called."""
        from aeon_v1.consolidate import consolidate_memories
        from aeon_v1.embeddings import clear_cache
        clear_cache()

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.embedding_enabled = False
        cfg.consolidation_similarity_threshold = 0.5
        cfg.max_consolidations_per_pass = 5

        m1 = {
            "id": "ggg00001", "type": "episodic",
            "summary": "user always prefers dark mode settings enabled forever",
            "importance": 0.6, "tags": [], "created": "2026-05-01T00:00:00",
        }
        m2 = {
            "id": "hhh00002", "type": "episodic",
            "summary": "user always prefers dark mode settings enabled always",
            "importance": 0.6, "tags": [], "created": "2026-05-02T00:00:00",
        }
        mem_dir = tmp_path / "memory" / "episodic"
        (mem_dir / "ggg00001.json").write_text(json.dumps(m1), encoding="utf-8")
        (mem_dir / "hhh00002.json").write_text(json.dumps(m2), encoding="utf-8")

        with patch("aeon_v1.consolidate.get_embedding") as mock_emb:
            result = consolidate_memories(config=cfg)

        mock_emb.assert_not_called()
        assert result["created_count"] >= 1
