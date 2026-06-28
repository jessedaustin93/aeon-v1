"""`aeon-music` -- the narrow action adapter run by an approved Agent Mesh task.

This is the *executor side* of the governed music pipeline. `manage_music`
(dispatch side) turns an accepted proposal into a human-approved Agent Mesh
approval whose command is ``aeon-music apply-proposal "<text>"``. The music
executor claims that approval and runs this CLI.

Safety is enforced two ways:

* A hard allowlist in ``LidarrClient._api`` -- only album/artist/profile *reads*,
  adding an album, and AlbumSearch/ArtistSearch triggers are permitted. The
  adapter physically cannot delete, rename, or change settings, however the
  calling code evolves.
* Adding new music is gated behind ``music_allow_add`` (off by default). When off,
  ``apply-proposal`` only triggers a search for albums already in the library and
  reports anything else for a human to add. When on, it may add the matched album
  (monitored) + search just it -- it never mass-monitors a discography.
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
    """Allowlisted Lidarr API client. Reads + add-album + search triggers only."""

    def __init__(self, config: Config, *, http_request: Optional[HttpRequest] = None) -> None:
        self.config = config
        self._request = http_request or _default_http_request

    def _api(self, method: str, path: str, *, params: Optional[Dict] = None, body: Optional[Dict] = None):
        # --- hard allowlist: the only thing between a proposal and the media stack.
        # Reads, adding an album, and search triggers. No deletes, no settings. ---
        reads = ("/api/v1/album", "/api/v1/artist", "/api/v1/rootfolder",
                 "/api/v1/qualityprofile", "/api/v1/metadataprofile")
        allowed = (
            (method == "GET" and path.startswith(reads))
            or (method == "POST" and path == "/api/v1/album")
            or (method == "POST" and path == "/api/v1/command"
                and (body or {}).get("name") in ALLOWED_COMMANDS)
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

    def rootfolders(self) -> List[Dict]:
        return self._api("GET", "/api/v1/rootfolder") or []

    def quality_profiles(self) -> List[Dict]:
        return self._api("GET", "/api/v1/qualityprofile") or []

    def metadata_profiles(self) -> List[Dict]:
        return self._api("GET", "/api/v1/metadataprofile") or []

    def add_album(self, album: Dict) -> Dict:
        return self._api("POST", "/api/v1/album", body=album)


def _clean_term(text: str) -> str:
    return " ".join((text or "").split())


def _pick(items: List[Dict], name: str, key: str) -> Optional[Dict]:
    """Pick the named item (case-insensitive) or the first one."""
    if not items:
        return None
    if name:
        for item in items:
            if str(item.get(key, "")).lower() == name.lower():
                return item
    return items[0]


def _label(album: Dict) -> str:
    artist = (album.get("artist") or {}).get("artistName", "?")
    return f"{artist} - {album.get('title', '?')}"


def _add_new_album(results: List[Dict], *, config: Config, client: LidarrClient) -> Tuple[int, str]:
    # Prefer a proper "Album" over covers/singles that album/lookup also returns.
    pick = next((a for a in results if a.get("albumType") == "Album"), results[0])
    root = _pick(client.rootfolders(), config.lidarr_root_folder, "path")
    quality = _pick(client.quality_profiles(), config.lidarr_quality_profile, "name")
    metadata = _pick(client.metadata_profiles(), config.lidarr_metadata_profile, "name")
    if not (root and quality and metadata):
        return 5, "Lidarr is missing a root folder or quality/metadata profile."

    album = dict(pick)
    album["monitored"] = True
    album["addOptions"] = {"searchForNewAlbum": True}  # search just this album
    artist = dict(album.get("artist") or {})
    artist["rootFolderPath"] = root["path"]
    artist["qualityProfileId"] = quality["id"]
    artist["metadataProfileId"] = metadata["id"]
    artist["monitored"] = True
    artist["addOptions"] = {"monitor": "none", "searchForMissingAlbums": False}
    album["artist"] = artist
    try:
        client.add_album(album)
    except (MusicActionError, PermissionError) as exc:
        return 4, f"add failed: {exc}"
    return 0, f"Added + searching: {_label(pick)}"


def apply_proposal(text: str, *, config: Config, client: Optional[LidarrClient] = None, limit: int = 5) -> Tuple[int, str]:
    """Act on an accepted proposal. Returns ``(exit_code, summary)``.

    In-library albums get a search trigger. Not-in-library albums are added +
    searched when ``music_allow_add`` is on, otherwise reported for manual add.
    """
    term = _clean_term(text)
    if not term:
        return 2, "empty proposal"
    client = client or LidarrClient(config)
    try:
        results = client.lookup_albums(term)
    except (MusicActionError, PermissionError) as exc:
        return 3, f"lookup failed: {exc}"

    in_library = [a for a in results if isinstance(a, dict) and (a.get("id") or 0) > 0]
    queued = []
    for album in in_library[:limit]:
        try:
            client.trigger_album_search(int(album["id"]))
            queued.append(_label(album))
        except (MusicActionError, PermissionError) as exc:
            return 4, f"search trigger failed: {exc}"
    if queued:
        return 0, "Queued Lidarr search for: " + "; ".join(queued)

    real = [a for a in results if isinstance(a, dict)]
    if not real:
        return 0, f"No Lidarr matches for {term!r}."
    if config.music_allow_add:
        return _add_new_album(real, config=config, client=client)
    candidates = [_label(a) for a in real[:limit]]
    return 0, "Not in library; add one of these first: " + "; ".join(candidates)


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
