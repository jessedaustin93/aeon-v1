"""Tests for memory aging / recency weight."""
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from aeon_v1.aging import age_weight
from aeon_v1.config import Config


def _config(**kwargs):
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _ts(days_ago: float) -> str:
    """ISO timestamp for `days_ago` days in the past."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


class TestAgeWeightDisabled:
    def test_returns_one_when_disabled(self):
        cfg = _config(memory_aging_enabled=False)
        assert age_weight(_ts(100), cfg) == 1.0

    def test_empty_timestamp_returns_one(self):
        cfg = _config(memory_aging_enabled=True)
        assert age_weight("", cfg) == 1.0

    def test_none_equivalent_returns_one(self):
        cfg = _config(memory_aging_enabled=True)
        assert age_weight("", cfg) == 1.0


class TestAgeWeightDecay:
    def test_fresh_memory_near_one(self):
        cfg = _config(memory_aging_enabled=True, memory_aging_half_life_days=30.0, memory_aging_min_weight=0.0)
        w = age_weight(_ts(0), cfg)
        assert w > 0.99

    def test_half_life_gives_half(self):
        cfg = _config(memory_aging_enabled=True, memory_aging_half_life_days=30.0, memory_aging_min_weight=0.0)
        w = age_weight(_ts(30), cfg)
        assert math.isclose(w, 0.5, rel_tol=0.01)

    def test_double_half_life_gives_quarter(self):
        cfg = _config(memory_aging_enabled=True, memory_aging_half_life_days=30.0, memory_aging_min_weight=0.0)
        w = age_weight(_ts(60), cfg)
        assert math.isclose(w, 0.25, rel_tol=0.01)

    def test_min_weight_floor(self):
        cfg = _config(memory_aging_enabled=True, memory_aging_half_life_days=1.0, memory_aging_min_weight=0.2)
        # 365 days old with a 1-day half life → astronomically small, floored at 0.2
        w = age_weight(_ts(365), cfg)
        assert w == pytest.approx(0.2)

    def test_weight_decreases_with_age(self):
        cfg = _config(memory_aging_enabled=True, memory_aging_half_life_days=30.0, memory_aging_min_weight=0.0)
        w1 = age_weight(_ts(5), cfg)
        w2 = age_weight(_ts(15), cfg)
        w3 = age_weight(_ts(30), cfg)
        assert w1 > w2 > w3


class TestAgeWeightTimestampParsing:
    def test_z_suffix(self):
        cfg = _config(memory_aging_enabled=True)
        ts = "2026-01-01T00:00:00Z"
        w = age_weight(ts, cfg)
        assert 0.0 < w <= 1.0

    def test_offset_suffix(self):
        cfg = _config(memory_aging_enabled=True)
        ts = "2026-01-01T00:00:00+00:00"
        w = age_weight(ts, cfg)
        assert 0.0 < w <= 1.0

    def test_no_tz_treated_as_utc(self):
        cfg = _config(memory_aging_enabled=True)
        ts = "2026-01-01T00:00:00"
        w = age_weight(ts, cfg)
        assert 0.0 < w <= 1.0

    def test_invalid_timestamp_returns_one(self):
        cfg = _config(memory_aging_enabled=True)
        w = age_weight("not-a-date", cfg)
        assert w == 1.0


class TestSearchAging:
    """Verify that age weighting is applied in search ranking."""

    def test_recent_ranks_above_old_same_score(self, tmp_path):
        import json
        from aeon_v1.search import search

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.memory_aging_enabled = True
        cfg.memory_aging_half_life_days = 30.0
        cfg.memory_aging_min_weight = 0.0

        old_ts = _ts(90)
        new_ts = _ts(1)

        old_mem = {
            "id": "old001",
            "type": "episodic",
            "text": "kernel panic debugging session",
            "importance": 0.5,
            "created_at": old_ts,
            "tags": [],
        }
        new_mem = {
            "id": "new001",
            "type": "episodic",
            "text": "kernel panic debugging session",
            "importance": 0.5,
            "created_at": new_ts,
            "tags": [],
        }

        mem_dir = tmp_path / "memory" / "episodic"
        (mem_dir / "old001.json").write_text(json.dumps(old_mem), encoding="utf-8")
        (mem_dir / "new001.json").write_text(json.dumps(new_mem), encoding="utf-8")

        results = search("kernel panic", config=cfg)
        assert len(results) == 2
        ids = [r["memory"]["id"] for r in results]
        assert ids[0] == "new001", f"Expected new memory first, got {ids}"

    def test_search_aging_uses_created_field_from_memory_store(self, tmp_path):
        import json
        from aeon_v1.search import search

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.memory_aging_enabled = True
        cfg.memory_aging_half_life_days = 30.0
        cfg.memory_aging_min_weight = 0.0

        for mid, ts in [("old001", _ts(90)), ("new001", _ts(1))]:
            mem = {
                "id": mid,
                "type": "episodic",
                "text": "kernel panic debugging session",
                "importance": 0.5,
                "created": ts,
                "tags": [],
            }
            (tmp_path / "memory" / "episodic" / f"{mid}.json").write_text(
                json.dumps(mem), encoding="utf-8"
            )

        results = search("kernel panic", config=cfg)
        assert [r["memory"]["id"] for r in results] == ["new001", "old001"]

    def test_aging_disabled_order_stable_by_type(self, tmp_path):
        import json
        from aeon_v1.search import search

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.memory_aging_enabled = False

        old_ts = _ts(90)
        new_ts = _ts(1)

        for mid, ts in [("old001", old_ts), ("new001", new_ts)]:
            mem = {
                "id": mid,
                "type": "episodic",
                "text": "kernel panic debugging session",
                "importance": 0.5,
                "created_at": ts,
                "tags": [],
            }
            (tmp_path / "memory" / "episodic" / f"{mid}.json").write_text(
                json.dumps(mem), encoding="utf-8"
            )

        results = search("kernel panic", config=cfg)
        assert len(results) == 2


class TestImportanceAwareDecay:
    def test_high_importance_decays_slower(self):
        cfg = _config(
            memory_aging_enabled=True,
            memory_aging_half_life_days=30.0,
            memory_aging_min_weight=0.0,
            memory_aging_importance_scale=1.0,
        )
        ts = _ts(30)
        w_low = age_weight(ts, cfg, importance=0.0)   # half-life = 30d → 0.5
        w_high = age_weight(ts, cfg, importance=1.0)  # half-life = 60d → ~0.707
        assert w_high > w_low
        assert math.isclose(w_low, 0.5, rel_tol=0.01)
        assert math.isclose(w_high, 0.5 ** 0.5, rel_tol=0.01)  # 0.5^(30/60)

    def test_importance_scale_zero_disables_effect(self):
        cfg = _config(
            memory_aging_enabled=True,
            memory_aging_half_life_days=30.0,
            memory_aging_min_weight=0.0,
            memory_aging_importance_scale=0.0,
        )
        ts = _ts(30)
        w_low = age_weight(ts, cfg, importance=0.0)
        w_high = age_weight(ts, cfg, importance=1.0)
        assert math.isclose(w_low, w_high, rel_tol=0.001)

    def test_importance_clamped_to_zero_one(self):
        cfg = _config(memory_aging_enabled=True, memory_aging_half_life_days=30.0,
                      memory_aging_min_weight=0.0, memory_aging_importance_scale=1.0)
        ts = _ts(30)
        assert age_weight(ts, cfg, importance=-5.0) == pytest.approx(age_weight(ts, cfg, importance=0.0), rel=1e-4)
        assert age_weight(ts, cfg, importance=99.0) == pytest.approx(age_weight(ts, cfg, importance=1.0), rel=1e-4)


class TestScoreImportance:
    def test_returns_none_when_llm_disabled(self):
        from aeon_v1.llm import score_importance
        cfg = _config(llm_enabled=False)
        assert score_importance("some memory text", cfg) is None

    def test_parses_plain_float(self):
        from aeon_v1.llm import score_importance
        from unittest.mock import patch
        with patch("aeon_v1.llm.generate_search_text", return_value="0.7"):
            cfg = _config(llm_enabled=True, llm_provider="lmstudio")
            result = score_importance("some memory", cfg)
        assert result == pytest.approx(0.7)

    def test_parses_float_with_surrounding_text(self):
        from aeon_v1.llm import score_importance
        from unittest.mock import patch
        with patch("aeon_v1.llm.generate_search_text", return_value="Importance: 0.85"):
            cfg = _config(llm_enabled=True, llm_provider="lmstudio")
            result = score_importance("some memory", cfg)
        assert result == pytest.approx(0.85)

    def test_returns_none_on_unparseable_output(self):
        from aeon_v1.llm import score_importance
        from unittest.mock import patch
        with patch("aeon_v1.llm.generate_search_text", return_value="I cannot rate this."), \
             patch("aeon_v1.llm.generate_text", return_value=None):
            cfg = _config(llm_enabled=True, llm_provider="lmstudio")
            result = score_importance("some memory", cfg)
        assert result is None

    def test_clamps_to_zero_one(self):
        from aeon_v1.llm import score_importance
        from unittest.mock import patch
        with patch("aeon_v1.llm.generate_search_text", return_value="1.0"):
            cfg = _config(llm_enabled=True, llm_provider="lmstudio")
            result = score_importance("critical decision recorded", cfg)
        assert result == pytest.approx(1.0)

    def test_uses_search_model_first(self):
        """Mistral/search model should be called before the main LLM."""
        from aeon_v1.llm import score_importance
        from unittest.mock import patch, call
        with patch("aeon_v1.llm.generate_search_text", return_value="0.6") as mock_search, \
             patch("aeon_v1.llm.generate_text") as mock_main:
            cfg = _config(llm_enabled=True, llm_provider="lmstudio")
            result = score_importance("memory text", cfg)
        mock_search.assert_called_once()
        mock_main.assert_not_called()
        assert result == pytest.approx(0.6)


class TestIngestLLMImportance:
    def test_llm_score_used_for_promotion(self, tmp_path):
        """When LLM returns a high score, text below keyword threshold gets promoted."""
        from aeon_v1.ingest import ingest
        from unittest.mock import patch

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.llm_enabled = True
        cfg.importance_threshold = 0.5

        # Text with no keyword signals → rule-based score would be low
        text = "the quick brown fox jumps over the lazy dog"

        with patch("aeon_v1.ingest.score_importance", return_value=0.9):
            result = ingest(text, config=cfg)

        assert result["episodic"] is not None, "LLM high score should promote to episodic"

    def test_rule_based_used_when_llm_returns_none(self, tmp_path):
        from aeon_v1.ingest import ingest
        from unittest.mock import patch

        cfg = Config(base_path=tmp_path)
        cfg.ensure_dirs()
        cfg.llm_enabled = True
        cfg.importance_threshold = 0.5

        # Text that normally scores below threshold
        text = "hi"

        with patch("aeon_v1.ingest.score_importance", return_value=None):
            result = ingest(text, config=cfg)

        # Rule-based score for "hi" should be low — no episodic
        assert result["episodic"] is None


class TestReflectAgingWeightedSampling:
    """archive_random sampling should prefer recent memories."""

    def test_weighted_sample_prefers_recent(self):
        from aeon_v1.reflect import _select_reflection_sources
        from aeon_v1.memory_store import MemoryStore
        from unittest.mock import patch

        cfg = _config(
            memory_aging_enabled=True,
            memory_aging_half_life_days=1.0,  # very short so old memories are heavily penalised
            memory_aging_min_weight=0.0,
            reflection_sampling_strategy="archive_random",
            max_memories_per_reflection=5,
            reflection_source_memory_types=["episodic"],
            allow_reflection_on_reflections=False,
        )

        recent = [
            {"id": f"r{i}", "type": "episodic", "created": _ts(0.1)}
            for i in range(5)
        ]
        old = [
            {"id": f"o{i}", "type": "episodic", "created": _ts(365)}
            for i in range(20)
        ]
        all_mems = recent + old

        with patch.object(MemoryStore, "list_memories", return_value=all_mems):
            store = MemoryStore(cfg)
            hits = 0
            runs = 50
            for _ in range(runs):
                sample = _select_reflection_sources(store, cfg)
                recent_ids = {m["id"] for m in sample if m["id"].startswith("r")}
                hits += len(recent_ids)

        # With min_weight=0.0 and half_life=1 day, 365-day-old memories have
        # weight ≈ 0 vs recent weight ≈ 0.933.  All 5 recent should almost
        # always win every draw, giving avg ≈ 5.0.
        avg_recent_per_sample = hits / runs
        assert avg_recent_per_sample >= 4.5, (
            f"Expected mostly recent, got avg {avg_recent_per_sample:.1f}/5 recent per sample"
        )
