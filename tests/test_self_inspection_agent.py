"""Tests for Aeon's read-only self-inspection path."""

from aeon_v1 import Config
from aeon_v1.self_inspection_agent import SelfInspectionAgent


def _write_self_files(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "README.md").write_text(
        """# Aeon-V1

Aeon-V1 is a local-first recursive AI memory and learning system.

## Current Capabilities

| Area | Status | Main modules |
|---|---|---|
| Chat-style terminal interface | Implemented | `chat_cli.py` |
| Raw memory ingestion | Implemented | `ingest.py`, `memory_store.py` |
| Future teleportation | Planned | `future.py` |
""",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "architecture.md").write_text("# Architecture\n", encoding="utf-8")
    (tmp_path / "docs" / "memory_model.md").write_text("# Memory\n", encoding="utf-8")
    (tmp_path / "scripts" / "aeon_chat.py").write_text("print('chat')\n", encoding="utf-8")


def test_self_inspection_detects_self_questions(tmp_path):
    _write_self_files(tmp_path)
    agent = SelfInspectionAgent(Config(tmp_path))

    assert agent.is_self_query("what can you do?")
    assert agent.is_self_query("how do you work?")
    assert not agent.is_self_query("what do you remember about github?")


def test_self_inspection_reads_repo_files(tmp_path):
    _write_self_files(tmp_path)
    agent = SelfInspectionAgent(Config(tmp_path))

    result = agent.handle_chat_query_with_ids("what are you?")

    assert result is not None
    assert "I checked my own repo files" in result["reply"]
    assert "Chat-style terminal interface" in result["reply"]
    assert "Raw memory ingestion" in result["reply"]
    assert "Future teleportation" not in result["reply"]
    assert "self:README.md" in result["memory_ids"]
