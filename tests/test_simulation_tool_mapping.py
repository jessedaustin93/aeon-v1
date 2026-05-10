"""Tests for simulate.py — tool call mapping.

Coverage:
- tool_call field is always present in simulation records (None or dict)
- file_read matched from read-signal task descriptions
- file_write matched from write-signal task descriptions
- command_preview matched from run/execute-signal task descriptions
- No match when no builtin tools are registered
- No match when description has no recognised signals
- Highest-scoring tool wins on ambiguous descriptions
- tool_call.requires_human_review is always True
- tool_call.matched_by is "keyword"
- Path extraction: quoted path, bare extension path
- Command extraction: backtick, after action verb
- Disabled tools are excluded from matching
- Markdown includes Tool Call section
- JSON stores tool_call field
"""
import json
from pathlib import Path

import pytest

from aeon_v1 import Config, register_builtin_tools, FILE_READ, FILE_WRITE, COMMAND_PREVIEW
from aeon_v1.memory_store import _vault_note_path
from aeon_v1.simulate import (
    SimulationStore,
    _extract_command,
    _extract_path,
    _match_tool_call,
    simulate_action,
)
from aeon_v1.tasks import TaskStore
from aeon_v1.tools import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config(base_path=tmp_path)
    c.ensure_dirs()
    return c


@pytest.fixture
def cfg_with_tools(cfg):
    """Config with all three builtin tools registered."""
    registry = ToolRegistry(cfg)
    register_builtin_tools(registry)
    return cfg


def _make_task(cfg: Config, description: str, title: str = "Test Task") -> dict:
    store = TaskStore(cfg)
    task = store.create_task(
        description=description,
        source_reflection_id="test-ref",
        source_reflection_title="Test Reflection",
        confidence=0.7,
        tags=["test"],
        priority=0.5,
    )
    assert task is not None
    return task


# ---------------------------------------------------------------------------
# tool_call always present in simulation record
# ---------------------------------------------------------------------------

def test_tool_call_key_always_present_when_no_tools_registered(cfg):
    task = _make_task(cfg, "Review the project memory index.")
    result = simulate_action(task, config=cfg)
    assert "tool_call" in result["simulation"]


def test_tool_call_is_none_when_no_tools_registered(cfg):
    task = _make_task(cfg, "Read the config.json file.")
    result = simulate_action(task, config=cfg)
    assert result["simulation"]["tool_call"] is None


def test_tool_call_key_always_present_with_tools_registered(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Review the project memory index.")
    result = simulate_action(task, config=cfg_with_tools)
    assert "tool_call" in result["simulation"]


# ---------------------------------------------------------------------------
# Tool selection — file_read
# ---------------------------------------------------------------------------

def test_file_read_matched_by_read_signal(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Read the project config.json file.")
    result = _match_tool_call(task["description"], cfg_with_tools)
    assert result is not None
    assert result["tool"] == "file_read"


def test_file_read_matched_by_inspect_signal(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Inspect the memory store to find patterns.")
    result = _match_tool_call(task["description"], cfg_with_tools)
    assert result is not None
    assert result["tool"] == "file_read"


def test_file_read_matched_by_load_signal(cfg_with_tools):
    result = _match_tool_call("Load the semantic index from disk.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "file_read"


# ---------------------------------------------------------------------------
# Tool selection — file_write
# ---------------------------------------------------------------------------

def test_file_write_matched_by_write_signal(cfg_with_tools):
    result = _match_tool_call("Write the summary output to results.txt.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "file_write"


def test_file_write_matched_by_save_signal(cfg_with_tools):
    result = _match_tool_call("Save the updated index to disk.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "file_write"


def test_file_write_matched_by_generate_signal(cfg_with_tools):
    result = _match_tool_call("Generate a report and append to the log.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "file_write"


# ---------------------------------------------------------------------------
# Tool selection — command_preview
# ---------------------------------------------------------------------------

def test_command_preview_matched_by_run_signal(cfg_with_tools):
    result = _match_tool_call("Run the test suite to verify changes.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "command_preview"


def test_command_preview_matched_by_execute_signal(cfg_with_tools):
    result = _match_tool_call("Execute the migration script.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "command_preview"


def test_command_preview_matched_by_shell_signal(cfg_with_tools):
    result = _match_tool_call("Review the shell environment setup.", cfg_with_tools)
    assert result is not None
    assert result["tool"] == "command_preview"


# ---------------------------------------------------------------------------
# No match cases
# ---------------------------------------------------------------------------

def test_no_match_for_unrecognised_description(cfg_with_tools):
    result = _match_tool_call("Review project patterns and document findings.", cfg_with_tools)
    assert result is None


def test_no_match_when_tools_not_registered(cfg):
    result = _match_tool_call("Read the config.json file.", cfg)
    assert result is None


def test_no_match_when_tool_is_disabled(cfg):
    registry = ToolRegistry(cfg)
    registry.register(ToolDefinition(
        name="file_read",
        description="Read a file.",
        tags=["file"],
        layer=5,
        enabled=False,
    ))
    result = _match_tool_call("Read the config.json file.", cfg)
    assert result is None


# ---------------------------------------------------------------------------
# tool_call structure
# ---------------------------------------------------------------------------

def test_tool_call_approval_required(cfg_with_tools):
    result = _match_tool_call("Read the config.json file.", cfg_with_tools)
    assert result["approval_required"] is True


def test_tool_call_matched_by_keyword(cfg_with_tools):
    result = _match_tool_call("Read the config.json file.", cfg_with_tools)
    assert result["matched_by"] == "keyword"


def test_tool_call_arguments_is_dict(cfg_with_tools):
    result = _match_tool_call("Read the config.json file.", cfg_with_tools)
    assert isinstance(result["arguments"], dict)


def test_approval_required_propagated_from_tool_definition(cfg_with_tools):
    result = _match_tool_call("Read the config.json file.", cfg_with_tools)
    assert "approval_required" in result
    assert result["approval_required"] is True


def test_approval_required_false_when_tool_definition_says_false(cfg):
    registry = ToolRegistry(cfg)
    registry.register(ToolDefinition(
        name="file_read",
        description="Read a file.",
        tags=["file"],
        layer=5,
        approval_required=False,
    ))
    result = _match_tool_call("Read the config.json file.", cfg)
    assert result is not None
    assert result["approval_required"] is False


# ---------------------------------------------------------------------------
# Argument extraction — _extract_path
# ---------------------------------------------------------------------------

def test_extract_path_from_quoted_string():
    assert _extract_path('Read "memory/store.json" for details.') == "memory/store.json"


def test_extract_path_from_single_quoted_string():
    assert _extract_path("Load 'data/output.csv' into memory.") == "data/output.csv"


def test_extract_path_bare_extension():
    result = _extract_path("Inspect config.json for the index path.")
    assert result == "config.json"


def test_extract_path_returns_none_when_no_path():
    assert _extract_path("Review the project memory and document findings.") is None


def test_file_read_arguments_include_extracted_path(cfg_with_tools):
    result = _match_tool_call('Read "vault/index.md" to check links.', cfg_with_tools)
    assert result["tool"] == "file_read"
    assert result["arguments"].get("path") == "vault/index.md"


def test_file_write_arguments_include_extracted_path(cfg_with_tools):
    result = _match_tool_call('Write the report to "output/summary.txt".', cfg_with_tools)
    assert result["tool"] == "file_write"
    assert result["arguments"].get("path") == "output/summary.txt"


# ---------------------------------------------------------------------------
# Argument extraction — _extract_command
# ---------------------------------------------------------------------------

def test_extract_command_from_backticks():
    assert _extract_command("Run `pytest tests/ -v` to verify.") == "pytest tests/ -v"


def test_extract_command_after_run_keyword():
    result = _extract_command("Execute the migration script.")
    assert result is not None and len(result) > 0


def test_extract_command_returns_none_when_absent():
    assert _extract_command("Review the project patterns carefully.") is None


def test_command_preview_arguments_include_extracted_command(cfg_with_tools):
    result = _match_tool_call("Run `pytest tests/` to validate changes.", cfg_with_tools)
    assert result["tool"] == "command_preview"
    assert result["arguments"].get("command") == "pytest tests/"


# ---------------------------------------------------------------------------
# Simulation record — JSON and Markdown
# ---------------------------------------------------------------------------

def test_tool_call_stored_in_json(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Read the memory index.")
    result = simulate_action(task, config=cfg_with_tools)
    sim = result["simulation"]
    json_path = cfg_with_tools.memory_path / "simulations" / f"{sim['id']}.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "tool_call" in data


def test_tool_call_json_null_when_no_match(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Document the episodic memory taxonomy.")
    result = simulate_action(task, config=cfg_with_tools)
    sim = result["simulation"]
    json_path = cfg_with_tools.memory_path / "simulations" / f"{sim['id']}.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["tool_call"] is None


def test_tool_call_section_in_markdown(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Read the memory index file.")
    result = simulate_action(task, config=cfg_with_tools)
    sim = result["simulation"]
    md_path = _vault_note_path(cfg_with_tools, "simulations", sim["id"])
    md = md_path.read_text(encoding="utf-8")
    assert "**Tool Call:**" in md


def test_tool_call_markdown_shows_no_match_message(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Document the taxonomy structure thoroughly.")
    result = simulate_action(task, config=cfg_with_tools)
    sim = result["simulation"]
    md_path = _vault_note_path(cfg_with_tools, "simulations", sim["id"])
    md = md_path.read_text(encoding="utf-8")
    assert "No matching tool" in md


def test_tool_call_markdown_shows_json_block_on_match(cfg_with_tools):
    task = _make_task(cfg_with_tools, "Read the config.json file.")
    result = simulate_action(task, config=cfg_with_tools)
    sim = result["simulation"]
    md_path = _vault_note_path(cfg_with_tools, "simulations", sim["id"])
    md = md_path.read_text(encoding="utf-8")
    assert "```json" in md
    assert "file_read" in md
