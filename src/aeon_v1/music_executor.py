"""`aeon-music-executor` -- claims approved music tasks and runs the adapter.

Runs on T3610 as the `music@t3610` station. Each poll it asks the Agent Mesh hub
for approvals that a human has already *approved* and that are addressed to this
station, validates the command against a tight allowlist, claims one (atomic on
the hub), runs the narrow `aeon-music` adapter, and posts the result back.

It only ever acts on approvals a human approved through the hub's `/decision`
gate, and it refuses any command that is not `aeon-music apply-proposal` -- a
mismatched approval is claimed and completed with a rejection so it never lingers
as an unclaimable task.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config

HttpRequest = Callable[[str, str, Optional[bytes], Dict[str, str], float], bytes]
Runner = Callable[[List[str]], Tuple[int, str]]

# The only command shape this station will execute.
ALLOWED_SUBCOMMANDS = {"apply-proposal"}


class ExecutorError(RuntimeError):
    pass


def _default_http_request(method, url, body, headers, timeout):
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network shape
        raise ExecutorError(f"hub {method} {url} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network shape
        raise ExecutorError(f"hub {method} {url} unreachable: {exc.reason}") from exc


class MusicExecutor:
    def __init__(
        self,
        config: Config,
        *,
        http_request: Optional[HttpRequest] = None,
        runner: Optional[Runner] = None,
    ) -> None:
        self.config = config
        self._request = http_request or _default_http_request
        self._runner = runner or self._default_runner

    @property
    def configured(self) -> bool:
        return bool(self.config.mesh_hub_url and self.config.mesh_token)

    @property
    def agent_id(self) -> str:
        return self.config.mesh_music_agent

    def _default_runner(self, command: List[str]) -> Tuple[int, str]:
        from .music_cli import apply_proposal

        if len(command) >= 3 and command[1] == "apply-proposal":
            return apply_proposal(command[2], config=self.config)
        return 2, "unsupported aeon-music command"

    def _hub(self, method: str, path: str, body: Optional[Dict] = None):
        headers = {
            "Authorization": f"Bearer {self.config.mesh_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps(body).encode("utf-8") if body is not None else None
        raw = self._request(method, f"{self.config.mesh_hub_url}{path}", data, headers,
                            self.config.mesh_timeout_seconds)
        return json.loads(raw or "null")

    @staticmethod
    def _command_ok(command) -> bool:
        return (
            isinstance(command, list)
            and len(command) >= 2
            and command[0] == "aeon-music"
            and command[1] in ALLOWED_SUBCOMMANDS
        )

    def poll_once(self) -> List[Dict]:
        """Process all approved approvals addressed to this station. Returns a
        list of per-approval outcome dicts."""
        if not self.configured:
            raise ExecutorError("mesh hub not configured (AEON_V1_MESH_HUB_URL/TOKEN)")
        approved = self._hub("GET", "/api/approvals?status=approved") or []
        outcomes = []
        for approval in approved:
            if approval.get("agent_id") != self.agent_id:
                continue
            outcomes.append(self._process(approval))
        return outcomes

    def _process(self, approval: Dict) -> Dict:
        approval_id = approval.get("id")
        command = approval.get("command")
        # Claim first (atomic); if another worker took it, skip quietly.
        try:
            self._hub("POST", f"/api/approvals/{approval_id}/claim", {"agent_id": self.agent_id})
        except ExecutorError as exc:
            return {"id": approval_id, "claimed": False, "detail": str(exc)}

        if not self._command_ok(command):
            self._hub("POST", f"/api/approvals/{approval_id}/result",
                      {"agent_id": self.agent_id, "exit_code": 2,
                       "result": f"rejected: command not allowlisted ({command!r})"})
            return {"id": approval_id, "claimed": True, "exit_code": 2, "rejected": True}

        try:
            exit_code, summary = self._runner(command)
        except Exception as exc:  # keep the station alive; report the failure
            exit_code, summary = 1, f"executor error: {exc}"
        self._hub("POST", f"/api/approvals/{approval_id}/result",
                  {"agent_id": self.agent_id, "exit_code": int(exit_code),
                   "result": summary[:100_000]})
        return {"id": approval_id, "claimed": True, "exit_code": int(exit_code), "result": summary}

    def run_forever(self, interval: Optional[int] = None) -> None:  # pragma: no cover - loop
        interval = interval or self.config.music_executor_interval
        print(f"aeon-music-executor: {self.agent_id} -> {self.config.mesh_hub_url} every {interval}s", flush=True)
        while True:
            try:
                outcomes = self.poll_once()
                if outcomes:
                    print(f"processed {len(outcomes)} approval(s): {outcomes}", flush=True)
            except Exception as exc:
                print(f"poll error: {exc}", flush=True)
            time.sleep(interval)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aeon-music-executor")
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit.")
    parser.add_argument("--interval", type=int, default=None, help="Seconds between polls.")
    args = parser.parse_args(argv)

    config = Config()
    executor = MusicExecutor(config)
    if not executor.configured:
        print("mesh hub not configured (set AEON_V1_MESH_HUB_URL and AEON_V1_MESH_TOKEN)")
        return 1
    if args.once:
        print(json.dumps(executor.poll_once()))
        return 0
    executor.run_forever(args.interval)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
