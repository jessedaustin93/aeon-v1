"""Tests for the narrow aeon-music adapter and the music approval executor."""
import json

from aeon_v1.config import Config
from aeon_v1 import music_cli
from aeon_v1.music_cli import LidarrClient, apply_proposal
from aeon_v1.music_executor import MusicExecutor, ExecutorError


# --------------------------------------------------------------------------- #
# aeon-music adapter
# --------------------------------------------------------------------------- #

class FakeLidarr:
    """Fake Lidarr transport: canned lookup, records command POSTs."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.commands = []

    def __call__(self, method, url, body, headers, timeout):
        if method == "GET" and "/album/lookup" in url:
            return json.dumps(self.lookup).encode()
        if method == "POST" and url.endswith("/api/v1/command"):
            self.commands.append(json.loads(body))
            return b"{}"
        raise AssertionError(f"unexpected {method} {url}")


def _cfg(tmp_path):
    c = Config(tmp_path)
    c.lidarr_api_key = "k"
    c.lidarr_url = "http://lidarr.test:8686"
    return c


def test_apply_proposal_triggers_search_for_in_library_album(tmp_path):
    cfg = _cfg(tmp_path)
    transport = FakeLidarr([
        {"id": 0, "title": "Some Demo", "artist": {"artistName": "Nobody"}},
        {"id": 42, "title": "Even In Arcadia", "artist": {"artistName": "Sleep Token"}},
    ])
    client = LidarrClient(cfg, http_request=transport)
    code, summary = apply_proposal("Sleep Token new album", config=cfg, client=client)
    assert code == 0
    assert "Queued Lidarr search" in summary and "Sleep Token - Even In Arcadia" in summary
    assert transport.commands == [{"name": "AlbumSearch", "albumIds": [42]}]


def test_apply_proposal_reports_when_not_in_library(tmp_path):
    cfg = _cfg(tmp_path)
    transport = FakeLidarr([
        {"id": 0, "title": "Brand New Thing", "artist": {"artistName": "Some Artist"}},
    ])
    code, summary = apply_proposal("brand new thing", config=cfg, client=LidarrClient(cfg, http_request=transport))
    assert code == 0
    assert "Not in library" in summary
    assert transport.commands == []  # nothing triggered


def test_apply_proposal_empty_is_rejected(tmp_path):
    code, summary = apply_proposal("   ", config=_cfg(tmp_path))
    assert code == 2 and "empty" in summary


def test_apply_proposal_without_api_key_errors(tmp_path):
    cfg = Config(tmp_path)  # no lidarr_api_key
    code, summary = apply_proposal("anything", config=cfg)
    assert code == 3 and "API_KEY" in summary


def test_lidarr_client_hard_allowlist_blocks_mutations(tmp_path):
    cfg = _cfg(tmp_path)
    calls = []
    client = LidarrClient(cfg, http_request=lambda *a: calls.append(a) or b"{}")
    # A delete must never reach the transport.
    import pytest
    with pytest.raises(PermissionError):
        client._api("DELETE", "/api/v1/album/42")
    # A non-allowlisted command name is blocked too.
    with pytest.raises(PermissionError):
        client._api("POST", "/api/v1/command", body={"name": "DeleteAlbum"})
    assert calls == []


# --------------------------------------------------------------------------- #
# music executor
# --------------------------------------------------------------------------- #

def _mesh_cfg(tmp_path):
    c = Config(tmp_path)
    c.mesh_hub_url = "http://hub.test:8787"
    c.mesh_token = "tok"
    return c  # mesh_music_agent defaults to music@t3610


class FakeHub:
    def __init__(self, approved):
        self.approved = approved
        self.claims = []
        self.results = []

    def __call__(self, method, url, body, headers, timeout):
        assert headers["Authorization"] == "Bearer tok"
        if method == "GET" and "/api/approvals?status=approved" in url:
            return json.dumps(self.approved).encode()
        if method == "POST" and url.endswith("/claim"):
            self.claims.append(url)
            return b"{}"
        if method == "POST" and url.endswith("/result"):
            self.results.append(json.loads(body))
            return b"{}"
        raise AssertionError(f"unexpected {method} {url}")


def test_executor_claims_runs_and_reports_music_task(tmp_path):
    cfg = _mesh_cfg(tmp_path)
    hub = FakeHub([
        {"id": "a1", "agent_id": "music@t3610", "command": ["aeon-music", "apply-proposal", "Sleep Token"]},
        {"id": "a2", "agent_id": "claude@t5810", "command": ["something", "else"]},  # ignored
    ])
    ran = []
    executor = MusicExecutor(cfg, http_request=hub, runner=lambda cmd: (ran.append(cmd) or (0, "queued search")))
    outcomes = executor.poll_once()

    assert len(outcomes) == 1 and outcomes[0]["id"] == "a1" and outcomes[0]["exit_code"] == 0
    assert ran == [["aeon-music", "apply-proposal", "Sleep Token"]]
    assert hub.claims == ["http://hub.test:8787/api/approvals/a1/claim"]
    assert hub.results == [{"agent_id": "music@t3610", "exit_code": 0, "result": "queued search"}]


def test_executor_rejects_non_allowlisted_command(tmp_path):
    cfg = _mesh_cfg(tmp_path)
    hub = FakeHub([
        {"id": "b1", "agent_id": "music@t3610", "command": ["rm", "-rf", "/"]},
    ])
    executor = MusicExecutor(cfg, http_request=hub, runner=lambda cmd: (0, "should not run"))
    outcomes = executor.poll_once()
    assert outcomes[0]["rejected"] is True
    assert hub.claims  # it was claimed so it does not linger
    assert hub.results[0]["exit_code"] == 2 and "not allowlisted" in hub.results[0]["result"]


def test_executor_requires_hub_config(tmp_path):
    executor = MusicExecutor(Config(tmp_path))  # no hub url/token
    import pytest
    with pytest.raises(ExecutorError):
        executor.poll_once()


class FakeLidarrAdd:
    """Fake Lidarr with profiles + add-album, records the add payload."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.added = []
        self.commands = []

    def __call__(self, method, url, body, headers, timeout):
        if method == "GET" and "/album/lookup" in url:
            return json.dumps(self.lookup).encode()
        if method == "GET" and url.endswith("/rootfolder"):
            return json.dumps([{"id": 1, "path": "/music"}]).encode()
        if method == "GET" and url.endswith("/qualityprofile"):
            return json.dumps([{"id": 1, "name": "Any"}, {"id": 2, "name": "Lossless"}]).encode()
        if method == "GET" and url.endswith("/metadataprofile"):
            return json.dumps([{"id": 1, "name": "Standard"}]).encode()
        if method == "POST" and url.endswith("/api/v1/album"):
            self.added.append(json.loads(body))
            return b"{}"
        if method == "POST" and url.endswith("/api/v1/command"):
            self.commands.append(json.loads(body))
            return b"{}"
        raise AssertionError(f"unexpected {method} {url}")


def test_apply_proposal_adds_new_album_when_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.music_allow_add = True
    cfg.lidarr_quality_profile = "Lossless"
    transport = FakeLidarrAdd([
        {"id": 0, "albumType": "Single", "title": "Some Single", "artist": {"artistName": "X", "foreignArtistId": "x"}},
        {"id": 0, "albumType": "Album", "title": "Even In Arcadia",
         "artist": {"artistName": "Sleep Token", "foreignArtistId": "st"}},
    ])
    code, summary = apply_proposal("Sleep Token new album", config=cfg, client=LidarrClient(cfg, http_request=transport))
    assert code == 0 and "Added + searching" in summary and "Sleep Token - Even In Arcadia" in summary
    assert len(transport.added) == 1
    added = transport.added[0]
    # picked the Album (not the Single), monitored + searched, profiles resolved
    assert added["title"] == "Even In Arcadia"
    assert added["monitored"] is True and added["addOptions"] == {"searchForNewAlbum": True}
    assert added["artist"]["rootFolderPath"] == "/music"
    assert added["artist"]["qualityProfileId"] == 2  # Lossless
    assert added["artist"]["metadataProfileId"] == 1
    assert added["artist"]["addOptions"] == {"monitor": "none", "searchForMissingAlbums": False}


def test_apply_proposal_does_not_add_when_disallowed(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.music_allow_add = False  # default
    transport = FakeLidarrAdd([
        {"id": 0, "albumType": "Album", "title": "New Thing", "artist": {"artistName": "Y"}},
    ])
    code, summary = apply_proposal("new thing", config=cfg, client=LidarrClient(cfg, http_request=transport))
    assert code == 0 and "Not in library" in summary
    assert transport.added == []  # nothing added
