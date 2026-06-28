"""Outbound Agent Mesh dispatch for governed music actions.

Aeon's music role only *plans* (see ``MUSIC_SYSTEM_PROMPT`` in ``chat_cli``). This
module is the one-way bridge that turns an explicitly accepted music proposal into
an audited, human-gated task on the Agent Mesh hub, targeted at the T3610 music
station.

It never executes anything locally and never auto-approves. The hub records the
task as a *pending approval* that a human must decide before any station claims
and runs it, so the same "no auto-approval" guarantee that governs Aeon's Layer 7
write pipeline also holds across the mesh boundary. Every outcome is written to the
append-only audit log.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import Config
from .security import AuditLog

# A pluggable HTTP transport so tests can exercise the dispatch logic without a
# live hub: (method, url, body_or_None, headers, timeout) -> response bytes.
HttpRequest = Callable[[str, str, Optional[bytes], Dict[str, str], float], bytes]


class MeshDispatchError(RuntimeError):
    """Raised when the Agent Mesh hub cannot be reached or rejects a request."""


@dataclass
class MeshTask:
    """A concrete task to hand to a mesh station as a pending approval."""

    command: List[str]
    reason: str
    agent_id: str
    cwd: Optional[str] = None
    ttl_seconds: int = 900


def _default_http_request(
    method: str,
    url: str,
    body: Optional[bytes],
    headers: Dict[str, str],
    timeout: float,
) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network shape
        detail = exc.read()[:200]
        raise MeshDispatchError(
            f"hub {method} {url} -> HTTP {exc.code}: {detail!r}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network shape
        raise MeshDispatchError(f"hub {method} {url} unreachable: {exc.reason}") from exc


def build_music_command(proposal: str) -> List[str]:
    """Represent an accepted proposal as an explicit argv for the T3610 station.

    Kept as an argv list (not a shell string) so the hub stores a structured,
    auditable command. The human-approved plan text rides along verbatim.
    """
    text = " ".join(proposal.split())
    return ["aeon-music", "apply-proposal", text]


class MeshClient:
    """Minimal authenticated client for the Agent Mesh hub approval API."""

    def __init__(self, config: Config, *, http_request: Optional[HttpRequest] = None) -> None:
        self.config = config
        self._request = http_request or _default_http_request

    @property
    def configured(self) -> bool:
        return bool(self.config.mesh_hub_url and self.config.mesh_token)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.mesh_token}",
            "Content-Type": "application/json",
        }

    def ensure_thread(self, agent_id: str) -> int:
        """Resolve (creating if needed) the World thread that holds the station's
        prompts, so the approval can be attached to a real thread."""
        query = urllib.parse.urlencode({"agent_id": agent_id})
        url = f"{self.config.mesh_hub_url}/api/quick/thread?{query}"
        raw = self._request("GET", url, None, self._headers(), self.config.mesh_timeout_seconds)
        return int(json.loads(raw)["thread_id"])

    def create_approval(self, task: MeshTask) -> Dict:
        """Create a pending, audited approval (the task) on the hub."""
        thread_id = self.ensure_thread(task.agent_id)
        body = {
            "thread_id": thread_id,
            "agent_id": task.agent_id,
            "command": task.command,
            "cwd": task.cwd,
            "reason": task.reason,
            "ttl_seconds": task.ttl_seconds,
        }
        url = f"{self.config.mesh_hub_url}/api/approvals"
        raw = self._request(
            "POST",
            url,
            json.dumps(body).encode("utf-8"),
            self._headers(),
            self.config.mesh_timeout_seconds,
        )
        return json.loads(raw)

    def approve_approval(self, approval_id: str, *, resolver: str = "aeon-auto") -> Dict:
        """Approve an approval on the operator's behalf (streamlined flow).

        Carries the separate approval credential the hub requires for /decision, so
        a music request the operator already made does not also need a manual click.
        """
        headers = dict(self._headers())
        headers["X-Agent-Mesh-Approval"] = self.config.mesh_approval_token
        url = f"{self.config.mesh_hub_url}/api/approvals/{approval_id}/decision"
        raw = self._request(
            "POST",
            url,
            json.dumps({"decision": "approved", "resolver": resolver}).encode("utf-8"),
            headers,
            self.config.mesh_timeout_seconds,
        )
        return json.loads(raw)


def manage_music(
    proposal: str,
    *,
    accepted: bool,
    config: Config,
    client: Optional[MeshClient] = None,
    audit: Optional[AuditLog] = None,
    trace_id: Optional[str] = None,
) -> Dict:
    """Convert an accepted music proposal into an audited Agent Mesh task.

    The action is governed at three layers:

    * It refuses to do anything until ``accepted`` is explicitly True -- planning a
      proposal is not the same as accepting it.
    * When the hub is not configured it only *prepares* the task and returns it,
      so nothing leaves the machine by accident.
    * When dispatched, the hub stores the task as a pending approval that a human
      must decide before a station executes it (no auto-approval).

    Returns a structured result dict and writes an audit entry for every branch.
    """
    proposal = (proposal or "").strip()
    audit = audit or AuditLog(config)
    trace_id = trace_id or f"music-{uuid.uuid4().hex[:8]}"

    if not proposal:
        audit.append(trace_id, "manage_music", "dispatch", "empty_proposal")
        return {
            "ok": False,
            "status": "error",
            "trace_id": trace_id,
            "reason": "empty_proposal",
        }

    if not accepted:
        audit.append(trace_id, "manage_music", "propose", "pending_acceptance")
        return {
            "ok": False,
            "status": "pending_acceptance",
            "trace_id": trace_id,
            "proposal": proposal,
            "detail": (
                "Aeon planned this music action. It becomes an Agent Mesh task only "
                "after you explicitly accept it."
            ),
        }

    task = MeshTask(
        command=build_music_command(proposal),
        reason=proposal[:4000],
        agent_id=config.mesh_music_agent,
        cwd=config.mesh_music_cwd or None,
        ttl_seconds=config.mesh_task_ttl_seconds,
    )

    client = client or MeshClient(config)
    if not client.configured:
        audit.append(trace_id, "manage_music", "prepare", "hub_not_configured")
        return {
            "ok": False,
            "status": "prepared",
            "trace_id": trace_id,
            "task": {
                "agent_id": task.agent_id,
                "command": task.command,
                "reason": task.reason,
                "cwd": task.cwd,
                "ttl_seconds": task.ttl_seconds,
            },
            "detail": (
                "Agent Mesh hub is not configured (set AEON_V1_MESH_HUB_URL and "
                "AEON_V1_MESH_TOKEN). Task prepared but not dispatched."
            ),
        }

    try:
        approval = client.create_approval(task)
    except MeshDispatchError as exc:
        audit.append(trace_id, "manage_music", "dispatch", f"error: {exc}")
        return {
            "ok": False,
            "status": "error",
            "trace_id": trace_id,
            "reason": str(exc),
        }

    approval_id = approval.get("id")
    audit.append(trace_id, "manage_music", "dispatch", f"approval_created:{approval_id}")

    # Streamlined path: the operator's request is the approval. Opt-in, and only
    # when an approval token is present; execution stays hard-allowlisted + audited.
    if config.mesh_auto_approve and config.mesh_approval_token:
        try:
            client.approve_approval(approval_id, resolver="aeon-auto")
        except MeshDispatchError as exc:
            audit.append(trace_id, "manage_music", "auto_approve", f"error: {exc}")
            return {
                "ok": True,
                "status": "dispatched",
                "trace_id": trace_id,
                "approval_id": approval_id,
                "agent_id": task.agent_id,
                "approval": approval,
                "detail": f"Dispatched, but auto-approve failed ({exc}); approve it in Agent Mesh.",
            }
        audit.append(trace_id, "manage_music", "auto_approve", f"approved:{approval_id}")
        return {
            "ok": True,
            "status": "approved",
            "trace_id": trace_id,
            "approval_id": approval_id,
            "agent_id": task.agent_id,
            "approval": approval,
            "detail": (
                "Dispatched and auto-approved. The T3610 music station will pick it "
                "up and run it shortly."
            ),
        }

    return {
        "ok": True,
        "status": "dispatched",
        "trace_id": trace_id,
        "approval_id": approval_id,
        "agent_id": task.agent_id,
        "approval": approval,
        "detail": (
            "Created a pending, audited Agent Mesh task on T3610. It will not run "
            "until you approve it in Agent Mesh."
        ),
    }
