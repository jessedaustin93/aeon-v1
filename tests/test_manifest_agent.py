"""Tests for the ManifestAgent — drift detection and governed tool additions."""
import json
import pytest
from pathlib import Path

from aeon_v1 import (
    ApprovalAgent,
    AuditLog,
    AuthProvider,
    Config,
    DriftReport,
    ManifestAgent,
    ToolAdditionStore,
    ValidationAgent,
    WriteAgent,
    create_proposal,
)
from aeon_v1.manifest_agent import _normalise, _validate_tool_entry, MANIFEST_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg(tmp_path):
    c = Config(tmp_path)
    c.ensure_dirs()
    return c


class MockAuthProvider(AuthProvider):
    def __init__(self, approve=True, reason="mock"):
        self._approve = approve
        self._reason  = reason
    def request_approval(self, prompt, context):
        return self._approve, self._reason
    def provider_name(self):
        return "mock"


def _write_manifest(cfg, extra_tools=""):
    """Write a minimal tools_manifest.md for testing."""
    docs_dir = cfg.base_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "# Test Manifest\n\n"
        "## 1. Core\n\n"
        "### Python 3.10+\n- Purpose: runtime\n\n"
        "### pytest\n- Purpose: testing\n\n"
        "### tzdata\n- Purpose: timezone data\n\n"
        f"{extra_tools}"
    )
    (docs_dir / "tools_manifest.md").write_text(content, encoding="utf-8")


def _write_py_file(cfg, content, name="test_module.py"):
    """Write a Python file under src/ for import scanning."""
    src = cfg.base_path / "src" / "mypackage"
    src.mkdir(parents=True, exist_ok=True)
    (src / name).write_text(content, encoding="utf-8")


# ===========================================================================
# DriftReport
# ===========================================================================

class TestDriftReport:
    def test_no_drift(self):
        r = DriftReport([], [], ["samplepkg"], "2024-01-01T00:00:00+00:00")
        assert not r.has_drift
        assert r.matched == ["samplepkg"]

    def test_has_drift_when_code_missing(self):
        r = DriftReport(["requests"], [], [], "2024-01-01T00:00:00+00:00")
        assert r.has_drift

    def test_has_drift_when_manifest_stale(self):
        r = DriftReport([], ["old_lib"], [], "2024-01-01T00:00:00+00:00")
        assert r.has_drift

    def test_summary_no_drift(self):
        r = DriftReport([], [], ["samplepkg"], "2024-01-01T00:00:00+00:00")
        assert "No drift" in r.summary()

    def test_summary_shows_missing(self):
        r = DriftReport(["requests"], [], [], "2024-01-01T00:00:00+00:00")
        assert "requests" in r.summary()
        assert "NOT in manifest" in r.summary()

    def test_summary_shows_stale(self):
        r = DriftReport([], ["old_lib"], [], "2024-01-01T00:00:00+00:00")
        assert "old_lib" in r.summary()

    def test_to_dict_keys(self):
        r = DriftReport(["a"], ["b"], ["c"], "ts")
        d = r.to_dict()
        assert set(d.keys()) == {
            "in_code_not_manifest", "in_manifest_not_code",
            "matched", "scanned_at", "has_drift",
        }

    def test_results_are_sorted(self):
        r = DriftReport(["z_pkg", "a_pkg"], ["m_pkg", "b_pkg"], [], "ts")
        assert r.in_code_not_manifest == ["a_pkg", "z_pkg"]
        assert r.in_manifest_not_code == ["b_pkg", "m_pkg"]


# ===========================================================================
# scan_manifest
# ===========================================================================

class TestScanManifest:
    def test_extracts_tool_names(self, cfg):
        _write_manifest(cfg)
        agent = ManifestAgent(cfg)
        names = agent.scan_manifest()
        assert "Python 3.10+" in names
        assert "pytest" in names
        assert "tzdata" in names

    def test_empty_when_no_manifest(self, cfg):
        agent = ManifestAgent(cfg)
        assert agent.scan_manifest() == set()

    def test_does_not_pick_up_h2_headings(self, cfg):
        _write_manifest(cfg)
        agent = ManifestAgent(cfg)
        names = agent.scan_manifest()
        assert "Core" not in names
        assert "Test Manifest" not in names

    def test_custom_tool_name(self, cfg):
        _write_manifest(cfg, extra_tools="### MyCustomTool\n- Purpose: test\n\n")
        agent = ManifestAgent(cfg)
        assert "MyCustomTool" in agent.scan_manifest()


# ===========================================================================
# scan_imports
# ===========================================================================

class TestScanImports:
    def test_finds_third_party_import(self, cfg):
        _write_py_file(cfg, "import samplepkg\n")
        agent = ManifestAgent(cfg)
        imports = agent.scan_imports()
        assert "samplepkg" in imports

    def test_excludes_stdlib(self, cfg):
        _write_py_file(cfg, "import os\nimport json\nimport pathlib\n")
        agent = ManifestAgent(cfg)
        imports = agent.scan_imports()
        assert "os" not in imports
        assert "json" not in imports
        assert "pathlib" not in imports

    def test_excludes_own_package(self, cfg):
        _write_py_file(cfg, "from aeon_v1 import Config\n")
        agent = ManifestAgent(cfg)
        imports = agent.scan_imports()
        assert "aeon_v1" not in imports

    def test_from_import(self, cfg):
        _write_py_file(cfg, "from samplepkg import Client\n")
        agent = ManifestAgent(cfg)
        imports = agent.scan_imports()
        assert "samplepkg" in imports

    def test_no_src_dir_returns_empty(self, cfg):
        agent = ManifestAgent(cfg)
        assert agent.scan_imports() == set()

    def test_multiple_files(self, cfg):
        _write_py_file(cfg, "import samplepkg\n", "a.py")
        _write_py_file(cfg, "import httpx\n", "b.py")
        agent = ManifestAgent(cfg)
        imports = agent.scan_imports()
        assert "samplepkg" in imports
        assert "httpx" in imports

    def test_syntax_error_file_skipped(self, cfg):
        src = cfg.base_path / "src" / "mypackage"
        src.mkdir(parents=True, exist_ok=True)
        (src / "bad.py").write_text("def broken(\n", encoding="utf-8")
        agent = ManifestAgent(cfg)
        imports = agent.scan_imports()  # should not raise
        assert isinstance(imports, set)


# ===========================================================================
# scan_requirements
# ===========================================================================

class TestScanRequirements:
    def test_reads_requirements_txt(self, cfg):
        (cfg.base_path / "requirements.txt").write_text(
            "pytest>=7.0\ntzdata>=2024.1\n", encoding="utf-8"
        )
        agent = ManifestAgent(cfg)
        reqs = agent.scan_requirements()
        assert "pytest" in reqs
        assert "tzdata" in reqs

    def test_skips_comments(self, cfg):
        (cfg.base_path / "requirements.txt").write_text(
            "# this is a comment\nsamplepkgX\n", encoding="utf-8"
        )
        agent = ManifestAgent(cfg)
        reqs = agent.scan_requirements()
        assert "samplepkgx" in reqs  # normalised

    def test_normalises_hyphens(self, cfg):
        (cfg.base_path / "requirements.txt").write_text(
            "my-lib>=1.0\n", encoding="utf-8"
        )
        agent = ManifestAgent(cfg)
        reqs = agent.scan_requirements()
        assert "my_lib" in reqs

    def test_no_files_returns_empty(self, cfg):
        agent = ManifestAgent(cfg)
        assert agent.scan_requirements() == set()


# ===========================================================================
# check_drift
# ===========================================================================

class TestCheckDrift:
    def test_no_drift_when_all_matched(self, cfg):
        _write_manifest(cfg, "### samplepkg\n- Purpose: test package\n\n")
        _write_py_file(cfg, "import samplepkg\n")
        (cfg.base_path / "requirements.txt").write_text("samplepkg\n", encoding="utf-8")
        agent  = ManifestAgent(cfg)
        report = agent.check_drift()
        assert "samplepkg" in report.matched
        assert "samplepkg" not in report.in_code_not_manifest

    def test_detects_import_missing_from_manifest(self, cfg):
        _write_manifest(cfg)  # no 'httpx' in manifest
        _write_py_file(cfg, "import httpx\n")
        report = ManifestAgent(cfg).check_drift()
        assert "httpx" in report.in_code_not_manifest

    def test_detects_stale_manifest_entry(self, cfg):
        _write_manifest(cfg, "### OldLibrary\n- Purpose: gone\n\n")
        # OldLibrary not imported anywhere
        report = ManifestAgent(cfg).check_drift()
        normalised = [_normalise(n) for n in report.in_manifest_not_code]
        assert "oldlibrary" in normalised

    def test_audit_entry_written(self, cfg):
        _write_manifest(cfg)
        ManifestAgent(cfg).check_drift()
        entries = AuditLog(cfg).read_all()
        assert any(e["action"] == "check_drift" for e in entries)

    def test_run_monitor_returns_keys(self, cfg):
        _write_manifest(cfg)
        result = ManifestAgent(cfg).run_monitor()
        assert "drift_report" in result
        assert "pending_additions" in result
        assert "approved_additions" in result
        assert "summary" in result


# ===========================================================================
# Helpers
# ===========================================================================

class TestHelpers:
    def test_normalise_lowercase(self):
        assert _normalise("SamplePkg") == "samplepkg"

    def test_normalise_hyphens(self):
        assert _normalise("my-lib") == "my_lib"

    def test_validate_tool_entry_valid(self):
        errors = _validate_tool_entry(
            "Redis", "Message bus", "Planned", "https://redis.io/", "Fast."
        )
        assert errors == []

    def test_validate_tool_entry_missing_name(self):
        errors = _validate_tool_entry("", "purpose", "Planned", "link", "notes")
        assert any("name" in e for e in errors)

    def test_validate_tool_entry_bad_importance(self):
        errors = _validate_tool_entry("Tool", "purpose", "Critical", "link", "notes")
        assert any("importance" in e for e in errors)

    def test_validate_tool_entry_missing_link(self):
        errors = _validate_tool_entry("Tool", "purpose", "Optional", "", "notes")
        assert any("link" in e for e in errors)


# ===========================================================================
# ToolAdditionStore
# ===========================================================================

class TestToolAdditionStore:
    def test_empty_on_init(self, cfg):
        store = ToolAdditionStore(cfg)
        assert store.list_approved() == []
        assert store.count() == 0

    def test_get_missing_returns_none(self, cfg):
        store = ToolAdditionStore(cfg)
        assert store.get("nonexistent") is None

    def test_reads_manually_written_record(self, cfg):
        additions_dir = cfg.memory_path / "tool_additions"
        additions_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "test-001", "status": "approved",
            "tool_data": {"name": "Redis"},
            "proposed_by": "test", "approved_by": "mock",
            "approved_at": "2024-01-01T00:00:00+00:00",
            "trace_id": "test-001",
        }
        (additions_dir / "test-001.json").write_text(json.dumps(record), encoding="utf-8")
        store = ToolAdditionStore(cfg)
        assert store.count() == 1
        found = store.get("test-001")
        assert found["tool_data"]["name"] == "Redis"


# ===========================================================================
# propose_tool_addition — governed pipeline
# ===========================================================================

class TestProposeToolAddition:
    def test_approved_addition_committed(self, cfg):
        agent  = ManifestAgent(cfg)
        result = agent.propose_tool_addition(
            name="Redis",
            purpose="Inter-agent message bus",
            importance="Planned",
            link="https://redis.io/",
            notes="Replace filesystem polling for multi-process swarm.",
            auth_provider=MockAuthProvider(approve=True),
        )
        assert result["ok"], result["reason"]
        assert result["decision"] == "committed"
        assert result["memory_id"]

    def test_approved_addition_stored_in_tool_additions(self, cfg):
        agent = ManifestAgent(cfg)
        agent.propose_tool_addition(
            name="ZeroMQ",
            purpose="Low-latency messaging",
            importance="Planned",
            link="https://zeromq.org/",
            notes="Alternative to Redis for local IPC.",
            auth_provider=MockAuthProvider(approve=True),
        )
        additions = agent.list_approved_additions()
        names = [a["tool_data"]["name"] for a in additions]
        assert "ZeroMQ" in names

    def test_rejected_addition_not_stored(self, cfg):
        agent  = ManifestAgent(cfg)
        result = agent.propose_tool_addition(
            name="EvilLib",
            purpose="Bad tool",
            importance="Optional",
            link="https://evil.example/",
            notes="Should be rejected.",
            auth_provider=MockAuthProvider(approve=False, reason="policy"),
        )
        assert not result["ok"]
        assert result["decision"] == "rejected"
        additions = agent.list_approved_additions()
        assert not any(a["tool_data"].get("name") == "EvilLib" for a in additions)

    def test_invalid_importance_rejected_pre_stage(self, cfg):
        agent  = ManifestAgent(cfg)
        result = agent.propose_tool_addition(
            name="Tool",
            purpose="something",
            importance="Super-Critical",   # invalid
            link="https://example.com/",
            notes="notes",
            auth_provider=MockAuthProvider(approve=True),
        )
        assert not result["ok"]
        assert result["decision"] == "rejected_pre_stage"

    def test_empty_name_rejected_pre_stage(self, cfg):
        agent  = ManifestAgent(cfg)
        result = agent.propose_tool_addition(
            name="",
            purpose="something",
            importance="Planned",
            link="https://example.com/",
            notes="notes",
            auth_provider=MockAuthProvider(approve=True),
        )
        assert not result["ok"]

    def test_full_audit_trail_for_addition(self, cfg):
        agent  = ManifestAgent(cfg)
        result = agent.propose_tool_addition(
            name="FastAPI",
            purpose="Read-only dashboard",
            importance="Planned",
            link="https://fastapi.tiangolo.com/",
            notes="Future web UI.",
            auth_provider=MockAuthProvider(approve=True),
        )
        pid     = result["proposal_id"]
        entries = AuditLog(cfg).read_by_trace(pid)
        actions = {e["action"] for e in entries}
        assert "propose"  in actions
        assert "validate" in actions
        assert "approve"  in actions
        assert "commit"   in actions

    def test_list_pending_before_approval(self, cfg):
        """Proposals in staging but not yet approved appear in pending list."""
        # Stage directly without going through the full pipeline.
        proposal = create_proposal(
            "ManifestAgent",
            json.dumps({"name": "PendingTool", "purpose": "test"}),
            "tool_addition",
            1.0,
            cfg,
        )
        agent   = ManifestAgent(cfg)
        pending = agent.list_pending_additions()
        ids = [p["trace_id"] for p in pending]
        assert proposal["trace_id"] in ids

    def test_list_pending_excludes_committed(self, cfg):
        agent  = ManifestAgent(cfg)
        result = agent.propose_tool_addition(
            name="CommittedTool",
            purpose="already done",
            importance="Optional",
            link="https://example.com/",
            notes="Done.",
            auth_provider=MockAuthProvider(approve=True),
        )
        pending = agent.list_pending_additions()
        assert not any(
            p.get("trace_id") == result["proposal_id"] for p in pending
        )

    def test_tool_addition_type_accepted_by_schema(self):
        from aeon_v1 import validate_staging_proposal
        from aeon_v1.time_utils import utc_now_iso
        proposal = {
            "trace_id":    "t1",
            "proposed_by": "agent",
            "content":     "{}",
            "type":        "tool_addition",
            "confidence":  1.0,
            "timestamp":   utc_now_iso(),
            "status":      "pending",
        }
        ok, reason = validate_staging_proposal(proposal)
        assert ok, reason


# ===========================================================================
# Export checks
# ===========================================================================

class TestExports:
    def test_manifest_agent_exported(self):
        from aeon_v1 import ManifestAgent
        assert callable(ManifestAgent)

    def test_drift_report_exported(self):
        from aeon_v1 import DriftReport
        assert callable(DriftReport)

    def test_tool_addition_store_exported(self):
        from aeon_v1 import ToolAdditionStore
        assert callable(ToolAdditionStore)
