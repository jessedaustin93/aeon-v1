"""Tests for the Aeon terminal chat interface."""
from pathlib import Path

from aeon_v1.chat_cli import (
    ChatTurn,
    build_chat_prompt,
    compact,
    diversify_results,
    fallback_response,
    format_history,
    format_memories,
    memory_preview,
    parse_args,
    retrieve_context,
)
from aeon_v1 import Config, ingest


def test_compact_keeps_short_text():
    assert compact("hello world", 20) == "hello world"


def test_compact_truncates_long_text():
    assert compact("one two three four", 10) == "one two..."


def test_memory_preview_prefers_summary():
    preview = memory_preview({"summary": "short summary", "content": "long content"})

    assert preview == "short summary"


def test_format_memories_includes_id_type_and_preview():
    text = format_memories([
        {
            "match_type": "semantic",
            "memory": {"id": "abc123", "description": "A useful concept"},
        }
    ])

    assert "abc123" in text
    assert "semantic" in text
    assert "A useful concept" in text


def test_format_history_includes_recent_turns():
    text = format_history([ChatTurn(user="hi", assistant="hello")])

    assert "User: hi" in text
    assert "Aeon: hello" in text


def test_build_chat_prompt_contains_user_memory_and_safety_contract():
    prompt = build_chat_prompt(
        user_text="What matters?",
        memories=[{"match_type": "episodic", "memory": {"id": "m1", "summary": "Important goal"}}],
        history=[ChatTurn(user="Earlier", assistant="Earlier reply")],
    )

    assert "What matters?" in prompt
    assert "Important goal" in prompt
    assert "Earlier reply" in prompt
    assert "Do not claim actions were executed" in prompt


def test_fallback_response_mentions_memory_when_available():
    response = fallback_response(
        "hello",
        [{"memory": {"summary": "stored memory"}}],
    )

    assert response.startswith("[local]")
    assert "stored memory" in response


def test_fallback_response_handles_no_memory():
    response = fallback_response("hello", [])

    assert response.startswith("[local]")
    assert "I stored that" in response


def test_parse_args_resolves_base_path_and_transcript(tmp_path):
    options = parse_args(["--base-path", str(tmp_path), "--reflect-every", "3"])

    assert options.base_path == tmp_path.resolve()
    assert options.reflect_every == 3
    assert options.transcript_path == tmp_path.resolve() / "memory/chat/transcript.jsonl"


def test_parse_args_can_disable_transcript(tmp_path):
    options = parse_args(["--base-path", str(tmp_path), "--transcript", "off"])

    assert options.transcript_path is None


def test_retrieve_context_includes_raw_memories(tmp_path):
    cfg = Config(tmp_path)
    ingest("The special dashboard recall number is 123456.", config=cfg)

    results = retrieve_context("123456", cfg, limit=5)

    assert any(r["match_type"] == "raw" for r in results)


def test_retrieve_context_finds_natural_language_topic(tmp_path):
    cfg = Config(tmp_path)
    ingest("Aeon is a local recursive memory system with a dashboard.", config=cfg)
    ingest("The special dashboard recall number is 123456.", config=cfg)

    results = retrieve_context("what do you remember about Aeon", cfg, limit=5)

    combined = " ".join(
        str(r.get("memory", {}).get("text", ""))
        + str(r.get("memory", {}).get("summary", ""))
        for r in results
    )
    assert "recursive memory system" in combined


def test_diversify_results_keeps_multiple_memory_layers():
    results = [
        {"match_type": "raw", "memory": {"id": "raw1", "type": "raw", "text": "raw"}},
        {"match_type": "raw", "memory": {"id": "raw2", "type": "raw", "text": "raw"}},
        {"match_type": "semantic", "memory": {"id": "sem1", "type": "semantic", "description": "sem"}},
        {"match_type": "episodic", "memory": {"id": "ep1", "type": "episodic", "summary": "ep"}},
    ]

    selected = diversify_results(results, limit=3)
    selected_types = [r["memory"]["type"] for r in selected]

    assert selected_types == ["semantic", "episodic", "raw"]
