"""Tests for the local launcher, dashboard config, and always-on runner."""
import json
import threading
import time
from unittest.mock import patch

from aeon_v1.config import Config
from aeon_v1.dashboard import DashboardController
from aeon_v1.launcher_config import load_launcher_config
from aeon_v1.runner import run_forever
from aeon_v1.runtime import read_json, runner_status_path, runner_stop_path


def test_launcher_config_merges_local_overrides(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "launcher_config.json").write_text(
        json.dumps({
            "dashboard": {"port": 9999},
            "lm_studio": {"base_url": "http://localhost:1111/v1"},
        }),
        encoding="utf-8",
    )

    config = load_launcher_config(tmp_path)

    assert config["dashboard"]["host"] == "127.0.0.1"
    assert config["dashboard"]["port"] == 9999
    assert config["lm_studio"]["base_url"] == "http://localhost:1111/v1"
    assert config["obsidian"]["vault_path"] == "vault"


def test_runner_writes_status_and_stops_cleanly(tmp_path):
    cfg = Config(tmp_path)
    cfg.enable_background_consolidation = False

    thread = threading.Thread(
        target=run_forever,
        kwargs={"config": cfg, "poll_seconds": 0.1, "link_every_passes": 0},
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 3
    while time.time() < deadline and not runner_status_path(cfg).exists():
        time.sleep(0.05)

    status = read_json(runner_status_path(cfg))
    assert status["state"] == "running"
    assert status["component"] == "runner"

    runner_stop_path(cfg).write_text("stop", encoding="utf-8")
    thread.join(timeout=3)

    status = read_json(runner_status_path(cfg))
    assert not thread.is_alive()
    assert status["state"] == "stopped"


def test_dashboard_status_reports_local_components(tmp_path):
    cfg = Config(tmp_path)
    cfg.ensure_dirs()
    (tmp_path / "vault" / ".obsidian").mkdir(parents=True)

    controller = DashboardController(tmp_path)
    status = controller.status()

    assert status["dashboard"]["state"] == "running"
    assert status["obsidian"]["vault_exists"] is True
    assert status["obsidian"]["obsidian_config_exists"] is True
    assert "raw" in status["memory_counts"]
    assert status["lm_studio"]["base_url"] == "http://localhost:1234/v1"


def test_dashboard_chat_returns_reply_and_stores_memory(tmp_path):
    controller = DashboardController(tmp_path)

    result = controller.chat("Remember this important dashboard chat memory.")

    assert result["ok"] is True
    assert result["reply"]
    assert result["memory_ids"]

    status = controller.status()
    assert status["memory_counts"]["raw"] >= 2


def test_dashboard_search_query_answers_from_memory_agent(tmp_path):
    controller = DashboardController(tmp_path)
    controller.chat("Aeon is a local-first dashboard memory project.")
    before = controller.status()["memory_counts"]["raw"]

    result = controller.chat("what do you remember about Aeon")

    assert result["ok"] is True
    assert "local-first dashboard memory project" in result["reply"]
    assert result["llm_used"] is False
    assert controller.status()["memory_counts"]["raw"] == before


def test_dashboard_self_query_answers_from_repo_without_ingesting(tmp_path):
    (tmp_path / "README.md").write_text(
        """# Aeon-V1

Aeon-V1 is a local-first recursive AI memory and learning system.

## Current Capabilities

| Area | Status | Main modules |
|---|---|---|
| Local dashboard | Implemented | `dashboard.py` |
""",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text("# Architecture\n", encoding="utf-8")
    controller = DashboardController(tmp_path)
    before = controller.status()["memory_counts"]["raw"]

    result = controller.chat("what can you do?")

    assert result["ok"] is True
    assert result["llm_used"] is False
    assert "I checked my own repo files" in result["reply"]
    assert "Local dashboard" in result["reply"]
    assert any(mid.startswith("self:") for mid in result["memory_ids"])
    assert controller.status()["memory_counts"]["raw"] == before


def test_dashboard_image_upload_returns_media_record(tmp_path):
    controller = DashboardController(tmp_path)
    payload = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    with patch("aeon_v1.media.generate_image_description", return_value="A dashboard image."):
        result = controller.upload_image("dashboard.png", payload)

    assert result["ok"] is True
    assert result["media"]["description"] == "A dashboard image."
    assert controller.status()["memory_counts"]["media"] == 1


def test_dashboard_chat_accepts_image_attachment(tmp_path):
    controller = DashboardController(tmp_path)
    payload = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    with patch("aeon_v1.media.generate_image_description", return_value="A small image attached in chat."):
        result = controller.chat(
            "What is in this image?",
            image_filename="chat-image.png",
            image_data_url=payload,
        )

    assert result["ok"] is True
    assert result["reply"]
    assert result["media"]["original_name"] == "chat-image.png"
    assert result["media"]["description"] == "A small image attached in chat."
    assert controller.status()["memory_counts"]["media"] == 1


def test_dashboard_restores_recent_transcript_history(tmp_path):
    cfg = Config(tmp_path)
    path = cfg.memory_path / "chat" / "dashboard_transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "at": "2026-05-09T00:00:00+00:00",
            "user": "remember this",
            "assistant": "stored",
            "memory_ids": ["abc"],
            "llm_used": True,
        }) + "\n",
        encoding="utf-8",
    )

    controller = DashboardController(tmp_path)

    assert controller._chat_history
    assert controller._chat_history[-1].user == "remember this"
