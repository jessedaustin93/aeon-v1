"""`aeon-music` -- the narrow action adapter run by an approved Agent Mesh task.

This is the *executor side* of the governed music pipeline. `manage_music`
(dispatch side) turns an accepted proposal into a human-approved Agent Mesh
approval whose command is ``aeon-music apply-proposal "<text>"``. The music
executor claims that approval and runs this CLI.

Safety is enforced two ways:

* A hard allowlist in ``LidarrClient._request`` -- only album/artist *reads* and
  the ``AlbumSearch`` / ``ArtistSearch`` trigger commands are permitted. The
  adapter physically cannot delete, rename, change settings, or add download
  clients/indexers, regardless of how the calling code evolves.
* ``apply-proposal`` never *adds* anything. It looks up candidate releases and,
  for albums already monitored in the library, triggers a search (which uses the
  configured slskd/torrent download clients). Anything not already in the library
  is reported back for a human to add -- no silent library growth.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config

HttpRequest = Callable[[str, str, Optional[bytes], Dict[str, str], float], bytes]

# Commands the adapter may trigger via Lidarr's /command endpoint.
ALLOWED_COMMANDS = {"AlbumSearch", "ArtistSearch"}


class MusicActionError(RuntimeError):
    """Raised when the adapter cannot complete a Lidarr action."""


def _default_http_request(method, url, body, headers, timeout):
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network shape
        raise MusicActionError(f"lidarr {method} {url} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network shape
        raise MusicActionError(f"lidarr {method} {url} unreachable: {exc.reason}") from exc


class LidarrClient:
    """Allowlisted Lidarr API client. Reads + search triggers only."""

    def __init__(self, config: Config, *, http_request: Optional[HttpRequest] = None) -> None:
        self.config = config
        self._request = http_request or _default_http_request

    def _api(self, method: str, path: str, *, params: Optional[Dict] = None, body: Optional[Dict] = None):
        # --- hard allowlist: the only thing standing between a proposal and the
        # media stack. Keep it strict. ---
        allowed = (
            method == "GET" and (
                path.startswith("/api/v1/album") or path.startswith("/api/v1/artist")
            )
        ) or (
            method == "POST" and path == "/api/v1/command"
            and (body or {}).get("name") in ALLOWED_COMMANDS
        )
        if not allowed:
            raise PermissionError(f"disallowed Lidarr call: {method} {path} {(body or {}).get('name','')}")
        if not self.config.lidarr_api_key:
            raise MusicActionError("AEON_V1_LIDARR_API_KEY is not set")
        url = f"{self.config.lidarr_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = {"X-Api-Key": self.config.lidarr_api_key, "Content-Type": "application/json"}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        raw = self._request(method, url, data, headers, self.config.lidarr_timeout_seconds)
        return json.loads(raw or "null")

    def lookup_albums(self, term: str) -> List[Dict]:
        return self._api("GET", "/api/v1/album/lookup", params={"term": term}) or []

    def trigger_album_search(self, album_id: int) -> Dict:
        return self._api("POST", "/api/v1/command", body={"name": "AlbumSearch", "albumIds": [album_id]})


def _clean_term(text: str) -> str:
    return " ".join((text or "").split())


def apply_proposal(text: str, *, config: Config, client: Optional[LidarrClient] = None, limit: int = 5) -> Tuple[int, str]:
    """Look up an accepted proposal and trigger searches for matching library albums.

    Returns ``(exit_code, summary)``. exit_code 0 = handled (search queued or
    candidates reported); non-zero = error.
    """
    term = _clean_term(text)
    if not term:
        return 2, "empty proposal"
    client = client or LidarrClient(config)
    try:
        results = client.lookup_albums(term)
    except (MusicActionError, PermissionError) as exc:
        return 3, f"lookup failed: {exc}"

    # An album already in the library carries a positive id; lookup-only matches
    # report id 0. We only act on what the operator already monitors.
    in_library = [a for a in results if isinstance(a, dict) and (a.get("id") or 0) > 0]
    queued = []
    for album in in_library[:limit]:
        try:
            client.trigger_album_search(int(album["id"]))
            title = album.get("title", "?")
            artist = (album.get("artist") or {}).get("artistName", "?")
            queued.append(f"{artist} - {title}")
        except (MusicActionError, PermissionError) as exc:
            return 4, f"search trigger failed: {exc}"

    if queued:
        return 0, "Queued Lidarr search for: " + "; ".join(queued)
    candidates = [
        f"{(a.get('artist') or {}).get('artistName','?')} - {a.get('title','?')}"
        for a in results[:limit] if isinstance(a, dict)
    ]
    if candidates:
        return 0, "Not in library; add one of these first: " + "; ".join(candidates)
    return 0, f"No Lidarr matches for {term!r}."


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aeon-music", description="Narrow music action adapter.")
    sub = parser.add_subparsers(dest="action", required=True)
    ap = sub.add_parser("apply-proposal", help="Act on an accepted music proposal.")
    ap.add_argument("text", help="The accepted proposal text.")
    args = parser.parse_args(argv)

    config = Config()
    if args.action == "apply-proposal":
        code, summary = apply_proposal(args.text, config=config)
        print(summary)
        return code
    parser.error(f"unknown action {args.action}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
