"""Tests for the governed outbound Agent Mesh music dispatch."""
import json

from aeon_v1.config import Config
from aeon_v1.mesh_dispatch import (
    MeshClient,
    MeshDispatchError,
    build_music_command,
    manage_music,
)
from aeon_v1.security import AuditLog


def _configured(cfg: Config) -> Config:
    cfg.mesh_hub_url = "http://hub.test:8787"
    cfg.mesh_token = "test-token"
    return cfg


class RecordingTransport:
    """Fake HTTP transport that records calls and returns canned hub responses."""

    def __init__(self, thread_id=7, approval_id="appr-1"):
        self.thread_id = thread_id
        self.approval_id = approval_id
        self.calls = []

    def __call__(self, method, url, body, headers, timeout):
        self.calls.append({
            "method": method,
            "url": url,
            "body": json.loads(body) if body else None,
            "headers": headers,
        })
        if "/api/quick/thread" in url:
            return json.dumps({"thread_id": self.thread_id}).encode()
        if url.endswith("/api/approvals"):
            return json.dumps({"id": self.approval_id, "status": "pending"}).encode()
        raise AssertionError(f"unexpected url {url}")


def test_build_music_command_is_structured_argv():
    cmd = build_music_command("  grab the new\n Sleep Token album  ")
    assert cmd == ["aeon-music", "apply-proposal", "grab the new Sleep Token album"]


def test_unaccepted_proposal_is_not_dispatched(tmp_path):
    cfg = _configured(Config(tmp_path))
    transport = RecordingTransport()
    client = MeshClient(cfg, http_request=transport)

    result = manage_music("grab the new Sleep Token album", accepted=False, config=cfg, client=client)

    assert result["status"] == "pending_acceptance"
    assert result["ok"] is False
    assert transport.calls == []  # nothing left the machine
    audit = AuditLog(cfg).read_all()
    assert audit[-1]["action"] == "propose"
    assert audit[-1]["result"] == "pending_acceptance"


def test_empty_proposal_errors(tmp_path):
    cfg = _configured(Config(tmp_path))
    result = manage_music("   ", accepted=True, config=cfg, client=MeshClient(cfg, http_request=RecordingTransport()))
    assert result["status"] == "error"
    assert result["reason"] == "empty_proposal"


def test_accepted_but_hub_unconfigured_only_prepares(tmp_path):
    cfg = Config(tmp_path)  # no mesh_hub_url / token
    result = manage_music("dedupe my music library", accepted=True, config=cfg)
    assert result["status"] == "prepared"
    assert result["ok"] is False
    assert result["task"]["agent_id"] == cfg.mesh_music_agent
    assert result["task"]["command"][0] == "aeon-music"
    audit = AuditLog(cfg).read_all()
    assert audit[-1]["result"] == "hub_not_configured"


def test_accepted_and_configured_dispatches_pending_approval(tmp_path):
    cfg = _configured(Config(tmp_path))
    transport = RecordingTransport(thread_id=42, approval_id="appr-99")
    client = MeshClient(cfg, http_request=transport)

    result = manage_music("grab the new Sleep Token album in FLAC", accepted=True, config=cfg, client=client)

    assert result["ok"] is True
    assert result["status"] == "dispatched"
    assert result["approval_id"] == "appr-99"
    assert result["agent_id"] == "music@t3610"

    # Resolved the thread, then created the approval against it.
    assert transport.calls[0]["method"] == "GET"
    assert "music%40t3610" in transport.calls[0]["url"] or "music@t3610" in transport.calls[0]["url"]
    post = transport.calls[1]
    assert post["method"] == "POST" and post["url"].endswith("/api/approvals")
    assert post["headers"]["Authorization"] == "Bearer test-token"
    assert post["body"]["thread_id"] == 42
    assert post["body"]["agent_id"] == "music@t3610"
    assert post["body"]["command"][0] == "aeon-music"
    assert post["body"]["ttl_seconds"] == cfg.mesh_task_ttl_seconds

    audit = AuditLog(cfg).read_all()
    assert audit[-1]["action"] == "dispatch"
    assert audit[-1]["result"] == "approval_created:appr-99"


def test_dispatch_error_is_audited(tmp_path):
    cfg = _configured(Config(tmp_path))

    def boom(method, url, body, headers, timeout):
        raise MeshDispatchError("hub unreachable")

    result = manage_music("retag these tracks with beets", accepted=True, config=cfg,
                          client=MeshClient(cfg, http_request=boom))
    assert result["status"] == "error"
    assert "unreachable" in result["reason"]
    audit = AuditLog(cfg).read_all()
    assert audit[-1]["action"] == "dispatch"
    assert audit[-1]["result"].startswith("error:")


def test_do_music_command_dispatches_through_manage_music(tmp_path, capsys):
    from aeon_v1.chat_cli import ChatOptions, TerminalChatApp

    app = TerminalChatApp(Config(tmp_path), ChatOptions(base_path=tmp_path))

    app.do_music("")
    assert "Usage:" in capsys.readouterr().out

    # Hub unconfigured -> the command accepts the proposal but only prepares it.
    app.do_music("grab the new Sleep Token album in FLAC")
    out = capsys.readouterr().out
    assert "prepared but not dispatched" in out
    audit = AuditLog(app.config).read_all()
    assert audit[-1]["agent"] == "manage_music"
