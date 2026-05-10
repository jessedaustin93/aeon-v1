"""Tests for conversation arc tracking (conversation.py)."""
import pytest
from pathlib import Path
from unittest.mock import patch

from aeon_v1.conversation import ConversationTracker, classify_intent, _build_rule_based_arc_text
from aeon_v1.config import Config


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

def test_classify_task_intent():
    assert classify_intent("I need to fix the memory leak in ingest.") == "task"


def test_classify_reflective_intent():
    assert classify_intent("I wonder if we should rethink the whole flow.") == "reflective"


def test_classify_questioning_intent():
    assert classify_intent("How do we handle duplicate memories?") == "questioning"


def test_classify_question_mark_signal():
    # bare "?" should push toward questioning
    result = classify_intent("This is fine?")
    assert result == "questioning"


def test_classify_confirmatory():
    assert classify_intent("Yes, that's right, exactly.") == "confirmatory"


def test_classify_general_fallback():
    assert classify_intent("hello") == "general"


# ---------------------------------------------------------------------------
# ConversationTracker — basic
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    cfg = Config(base_path=tmp_path)
    cfg.ensure_dirs()
    cfg.conversation_arc_min_turns = 2
    cfg.conversation_arc_min_shifts = 0
    return cfg


def test_add_turn_returns_classification(config):
    tracker = ConversationTracker(config=config)
    result = tracker.add_turn("user", "I need to fix the ingest bug.")
    assert result["intent"] == "task"
    assert result["turn_index"] == 0
    assert result["speaker"] == "user"


def test_no_shift_on_first_two_turns(config):
    tracker = ConversationTracker(config=config)
    r0 = tracker.add_turn("user", "I need to fix the ingest bug.")
    r1 = tracker.add_turn("aeon", "I need to update the pipeline.")
    assert not r0["shift_detected"]
    assert not r1["shift_detected"]


def test_shift_detected_on_intent_change(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "I need to fix the ingest bug.")
    tracker.add_turn("aeon", "I need to update the pipeline.")
    r2 = tracker.add_turn("user", "I wonder if we should rethink this whole approach.")
    assert r2["shift_detected"]


def test_no_shift_when_intent_stays_same(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "I need to fix the ingest bug.")
    tracker.add_turn("aeon", "I need to update the pipeline step.")
    r2 = tracker.add_turn("user", "I need to also fix the reflect module.")
    assert not r2["shift_detected"]


# ---------------------------------------------------------------------------
# ConversationTracker — get_arc
# ---------------------------------------------------------------------------

def test_get_arc_shape(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "I need to fix the ingest bug.")
    tracker.add_turn("aeon", "Done.")
    arc = tracker.get_arc()
    assert arc["turn_count"] == 2
    assert "dominant_intent" in arc
    assert "shifts" in arc
    assert "session_id" in arc
    assert "summary" in arc


def test_get_arc_non_destructive(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "I need to fix the ingest bug.")
    arc1 = tracker.get_arc()
    tracker.add_turn("aeon", "I wonder if the design needs rethinking.")
    arc2 = tracker.get_arc()
    assert arc2["turn_count"] == 2
    assert arc1["turn_count"] == 1


# ---------------------------------------------------------------------------
# ConversationTracker — close
# ---------------------------------------------------------------------------

def test_close_without_store(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "I need to fix the ingest bug.")
    tracker.add_turn("aeon", "Done.")
    result = tracker.close(store=False)
    assert result["ingest_result"] is None
    assert result["arc"]["turn_count"] == 2


def test_close_stores_arc_as_episodic(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "I need to fix the ingest bug.")
    tracker.add_turn("aeon", "I wonder if the design needs rethinking.")
    tracker.add_turn("user", "How do we approach that?")
    result = tracker.close(store=True)
    assert result["ingest_result"] is not None
    assert result["ingest_result"]["raw"] is not None


def test_close_skips_storage_below_min_turns(config):
    config.conversation_arc_min_turns = 10
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "Hi.")
    tracker.add_turn("aeon", "Hello.")
    result = tracker.close(store=True)
    assert result["ingest_result"] is None


def test_close_raises_if_called_twice(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "hi")
    tracker.add_turn("aeon", "hello")
    tracker.close(store=False)
    with pytest.raises(RuntimeError):
        tracker.close(store=False)


def test_add_turn_raises_after_close(config):
    tracker = ConversationTracker(config=config)
    tracker.add_turn("user", "hi")
    tracker.add_turn("aeon", "hello")
    tracker.close(store=False)
    with pytest.raises(RuntimeError):
        tracker.add_turn("user", "another turn")


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def test_rule_based_arc_text_contains_session_id():
    arc = {
        "session_id": "abc123",
        "turn_count": 3,
        "dominant_intent": "task",
        "shifts": [
            {"from_intent": "task", "to_intent": "reflective",
             "at_turn": 2, "speaker": "user", "snippet": "I wonder...", "noteworthy": True}
        ],
        "summary": "",
    }
    text = _build_rule_based_arc_text(arc)
    assert "abc123" in text
    assert "task" in text
    assert "reflective" in text
    assert "I noticed" in text


def test_rule_based_arc_text_no_shifts():
    arc = {
        "session_id": "xyz",
        "turn_count": 2,
        "dominant_intent": "questioning",
        "shifts": [],
        "summary": "",
    }
    text = _build_rule_based_arc_text(arc)
    assert "0 intent shift" in text


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = Config()
    assert hasattr(cfg, "conversation_arc_min_turns")
    assert hasattr(cfg, "conversation_arc_min_shifts")
    assert cfg.conversation_arc_min_turns == 3
    assert cfg.conversation_arc_min_shifts == 0


# ---------------------------------------------------------------------------
# Public API re-export
# ---------------------------------------------------------------------------

def test_public_exports():
    import aeon_v1
    assert hasattr(aeon_v1, "ConversationTracker")
    assert hasattr(aeon_v1, "classify_intent")
