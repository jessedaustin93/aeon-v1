"""Manifest Monitor Agent for Aeon-V1.

Two responsibilities:

1. DRIFT DETECTION — reads docs/tools_manifest.md and the actual codebase
   (imports, requirements.txt, pyproject.toml) and reports tools that appear
   in one but not the other.  Read-only; never modifies any file.

2. GOVERNED TOOL ADDITIONS — maintains a pending list of proposed tools.
   Any write to this list MUST pass through the Layer 7 pipeline:
       propose_tool_addition()
         → create_proposal()        (staging)
         → ValidationAgent          (schema + content check)
         → ApprovalAgent            (human gate — no auto-approve)
         → WriteAgent               (commits to memory/tool_additions/)
   The ManifestAgent can only PROPOSE.  It cannot commit directly.

SAFETY CONTRACT
===============
- No subprocess, os.system, exec, eval, or network primitive is used.
- vault/core/ is never written.
- All writes go through Layer 7.
- Source code and docs are accessed read-only for analysis only.
"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

from .approval_agent import ApprovalAgent, AuthProvider
from .config import Config
from .security import AuditLog, PathGuard
from .time_utils import utc_now_iso
from .write_agent import WriteAgent, create_proposal
from .schemas import validate_staging_proposal

# ---------------------------------------------------------------------------
# Known stdlib / built-in module names to exclude from drift reports.
# Only third-party packages should appear in the manifest.
# ---------------------------------------------------------------------------
_STDLIB: FrozenSet[str] = frozenset({
    "abc", "ast", "builtins", "collections", "contextlib", "copy",
    "dataclasses", "datetime", "enum", "functools", "hashlib", "inspect",
    "io", "itertools", "json", "logging", "math", "operator", "os",
    "pathlib", "pickle", "re", "shutil", "signal", "socket", "string",
    "subprocess", "sys", "tempfile", "threading", "time", "traceback",
    "types", "typing", "unittest", "urllib", "uuid", "warnings",
    "zoneinfo", "_pytest", "pytest",   # pytest is a dev tool, in manifest separately
})

_OWN_PACKAGE = "aeon_v1"

# Manifest tool name patterns: lines starting with "### " are tool headings
_MANIFEST_HEADING = re.compile(r"^###\s+(.+)$", re.MULTILINE)

# Required tool entry fields (used when proposing an addition)
_TOOL_ENTRY_FIELDS = frozenset({"name", "purpose", "importance", "link", "notes"})
_VALID_IMPORTANCE  = frozenset({"Required", "Optional", "Experimental", "Planned"})

MANIFEST_PATH = Path("docs/tools_manifest.md")


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------

class DriftReport:
    """Comparison result: manifest vs actual codebase dependencies."""

    def __init__(
        self,
        in_code_not_manifest: List[str],
        in_manifest_not_code: List[str],
        matched: List[str],
        scanned_at: str,
    ) -> None:
        self.in_code_not_manifest = sorted(in_code_not_manifest)
        self.in_manifest_not_code = sorted(in_manifest_not_code)
        self.matched              = sorted(matched)
        self.scanned_at           = scanned_at
        self.has_drift            = bool(in_code_not_manifest or in_manifest_not_code)

    def to_dict(self) -> Dict:
        return {
            "in_code_not_manifest": self.in_code_not_manifest,
            "in_manifest_not_code": self.in_manifest_not_code,
            "matched":              self.matched,
            "scanned_at":           self.scanned_at,
            "has_drift":            self.has_drift,
        }

    def summary(self) -> str:
        lines = [f"Manifest drift report — {self.scanned_at}"]
        if not self.has_drift:
            lines.append("  No drift detected. Manifest matches codebase.")
        else:
            if self.in_code_not_manifest:
                lines.append("  In code, NOT in manifest (should add):")
                for t in self.in_code_not_manifest:
                    lines.append(f"    + {t}")
            if self.in_manifest_not_code:
                lines.append("  In manifest, NOT found in code (may be stale):")
                for t in self.in_manifest_not_code:
                    lines.append(f"    - {t}")
        if self.matched:
            lines.append(f"  Matched: {', '.join(self.matched)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToolAdditionStore  — READ-ONLY view of committed additions
# ---------------------------------------------------------------------------

class ToolAdditionStore:
    """Read-only access to approved tool additions in memory/tool_additions/.

    Writes only happen via the Layer 7 pipeline (WriteAgent → _commit_tool_addition).
    """

    def __init__(self, config: Config) -> None:
        self._dir   = config.memory_path / "tool_additions"
        self._guard = PathGuard(config)
        self._dir.mkdir(parents=True, exist_ok=True)

    def list_approved(self) -> List[Dict]:
        """Return all approved tool additions."""
        additions = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") == "approved":
                    additions.append(data)
            except Exception:
                pass
        return additions

    def get(self, addition_id: str) -> Optional[Dict]:
        path = self._dir / f"{addition_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def count(self) -> int:
        return len(self.list_approved())


# ---------------------------------------------------------------------------
# ManifestAgent
# ---------------------------------------------------------------------------

class ManifestAgent:
    """Monitors tools_manifest.md vs actual codebase; governs pending additions.

    Usage:

        agent = ManifestAgent(config)

        # Check for drift
        report = agent.check_drift()
        print(report.summary())

        # Propose a new tool (goes through full Layer 7 pipeline)
        result = agent.propose_tool_addition(
            name="Redis",
            purpose="Inter-agent message bus for multi-process swarm",
            importance="Planned",
            link="https://redis.io/",
            notes="Replace filesystem polling when orchestrator spans processes.",
            auth_provider=my_auth,   # omit to use CLI approval
        )

        # View what's been approved
        additions = agent.list_approved_additions()
    """

    def __init__(self, config: Config) -> None:
        self.config    = config
        self._audit    = AuditLog(config)
        self._store    = ToolAdditionStore(config)
        self._manifest = config.base_path / MANIFEST_PATH

    # ---------------------------------------------------------------- monitoring

    def scan_manifest(self) -> Set[str]:
        """Parse the tools manifest and return the set of tool/library names listed.

        Extracts every '### Name' heading.  Returns empty set if file missing.
        """
        if not self._manifest.exists():
            return set()
        text  = self._manifest.read_text(encoding="utf-8")
        names = {m.group(1).strip() for m in _MANIFEST_HEADING.finditer(text)}
        return names

    def scan_imports(self) -> Set[str]:
        """Walk all .py files under src/ and extract third-party import roots.

        Excludes stdlib modules and the aeon_v1 package itself.
        Returns only top-level package names (e.g. 'samplepkg' not 'samplepkg.types').
        """
        src_dir = self.config.base_path / "src"
        if not src_dir.exists():
            return set()

        found: Set[str] = set()
        for py_file in src_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root not in _STDLIB and root != _OWN_PACKAGE:
                            found.add(root)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root = node.module.split(".")[0]
                        if root not in _STDLIB and root != _OWN_PACKAGE and root != "":
                            found.add(root)
        return found

    def scan_requirements(self) -> Set[str]:
        """Parse requirements.txt and pyproject.toml for listed dependencies.

        Returns normalised package names (lowercased, hyphens→underscores).
        """
        deps: Set[str] = set()
        base = self.config.base_path

        req_file = base / "requirements.txt"
        if req_file.exists():
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    name = re.split(r"[>=<!;\[]", line)[0].strip()
                    if name:
                        deps.add(_normalise(name))

        pyproject = base / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text(encoding="utf-8")
            for match in re.finditer(r'"([A-Za-z0-9_\-]+)(?:[>=<!][^"]*)?"', text):
                candidate = match.group(1)
                if candidate not in {"setuptools"}:
                    deps.add(_normalise(candidate))

        return deps

    def check_drift(self) -> DriftReport:
        """Compare manifest names against code imports + requirements.

        Returns a DriftReport describing what is mismatched.
        Only third-party packages are compared; stdlib is excluded.
        """
        manifest_names = {_normalise(n) for n in self.scan_manifest()}
        code_imports   = {_normalise(n) for n in self.scan_imports()}
        requirements   = {_normalise(n) for n in self.scan_requirements()}

        # "Used" = any third-party package found in imports OR requirements
        used = code_imports | requirements

        in_code_not_manifest = sorted(used - manifest_names)
        in_manifest_not_code = sorted(manifest_names - used)
        matched              = sorted(used & manifest_names)

        report = DriftReport(
            in_code_not_manifest=in_code_not_manifest,
            in_manifest_not_code=in_manifest_not_code,
            matched=matched,
            scanned_at=utc_now_iso(),
        )
        self._audit.append(
            "manifest-scan", "ManifestAgent", "check_drift",
            f"drift={report.has_drift}; "
            f"missing_from_manifest={len(in_code_not_manifest)}; "
            f"stale_in_manifest={len(in_manifest_not_code)}",
        )
        return report

    def run_monitor(self) -> Dict:
        """Full monitoring cycle: drift check + additions summary."""
        drift   = self.check_drift()
        pending = self._count_pending()
        approved = self._store.count()
        return {
            "drift_report":      drift.to_dict(),
            "pending_additions": pending,
            "approved_additions": approved,
            "summary":           drift.summary(),
        }

    # ---------------------------------------------------------------- governed additions

    def propose_tool_addition(
        self,
        name: str,
        purpose: str,
        importance: str,
        link: str,
        notes: str,
        auth_provider: Optional[AuthProvider] = None,
    ) -> Dict:
        """Propose a new tool addition through the full Layer 7 pipeline.

        The proposal is staged, validated, presented for human approval,
        then committed to memory/tool_additions/ if approved.

        Args:
            name:          Tool name as it will appear in the manifest.
            purpose:       What it does in THIS system.
            importance:    'Required' | 'Optional' | 'Experimental' | 'Planned'
            link:          Official download / purchase URL.
            notes:         Why this choice matters for Aeon-V1.
            auth_provider: Override the default CLIAuthProvider for approval.

        Returns a dict with keys: ok, proposal_id, decision, reason, memory_id.
        """
        # Validate inputs before touching Layer 7.
        errors = _validate_tool_entry(name, purpose, importance, link, notes)
        if errors:
            return {"ok": False, "reason": f"Invalid tool entry: {'; '.join(errors)}",
                    "proposal_id": None, "decision": "rejected_pre_stage", "memory_id": None}

        tool_data = {
            "name":       name,
            "purpose":    purpose,
            "importance": importance,
            "link":       link,
            "notes":      notes,
        }

        # Step 1 — Stage (Layer 7)
        proposal = create_proposal(
            proposed_by="ManifestAgent",
            content=json.dumps(tool_data),
            memory_type="tool_addition",
            confidence=1.0,
            config=self.config,
        )
        pid = proposal["trace_id"]

        # Step 2 — Validate (Layer 7)
        from .security import ValidationAgent
        v_result = ValidationAgent(self.config).validate_proposal(pid)
        if not v_result["ok"]:
            return {"ok": False, "proposal_id": pid,
                    "decision": "rejected_validation", "reason": v_result["reason"],
                    "memory_id": None}

        # Step 3 — Human approval (Layer 7) — NO auto-approve path.
        a_result = ApprovalAgent(self.config, auth_provider).approve_proposal(pid)
        if not a_result["ok"]:
            return {"ok": False, "proposal_id": pid,
                    "decision": a_result["decision"], "reason": a_result["reason"],
                    "memory_id": None}

        # Step 4 — Commit (Layer 7)
        c_result = WriteAgent(self.config).commit_proposal(pid)
        return {
            "ok":          c_result["ok"],
            "proposal_id": pid,
            "decision":    "committed" if c_result["ok"] else "commit_failed",
            "reason":      c_result["reason"],
            "memory_id":   c_result.get("memory_id"),
        }

    def list_pending_additions(self) -> List[Dict]:
        """Return tool addition proposals currently in staging (not yet approved)."""
        staging = self.config.memory_path / "staging"
        pending = []
        for f in sorted(staging.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if (data.get("type") == "tool_addition"
                        and data.get("status") not in ("committed", "rejected")):
                    pending.append(data)
            except Exception:
                pass
        return pending

    def list_approved_additions(self) -> List[Dict]:
        """Return all approved (committed) tool additions."""
        return self._store.list_approved()

    # ---------------------------------------------------------------- internals

    def _count_pending(self) -> int:
        return len(self.list_pending_additions())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase and replace hyphens with underscores for package name comparison."""
    return name.lower().replace("-", "_")


def _validate_tool_entry(
    name: str,
    purpose: str,
    importance: str,
    link: str,
    notes: str,
) -> List[str]:
    errors = []
    if not name or not name.strip():
        errors.append("name is required")
    if not purpose or not purpose.strip():
        errors.append("purpose is required")
    if importance not in _VALID_IMPORTANCE:
        errors.append(f"importance must be one of {sorted(_VALID_IMPORTANCE)}")
    if not link or not link.strip():
        errors.append("link is required")
    if not notes or not notes.strip():
        errors.append("notes is required")
    return errors
