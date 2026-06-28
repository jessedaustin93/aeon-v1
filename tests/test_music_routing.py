from aeon_v1.chat_cli import is_music_management_request


def test_routes_explicit_song_pipeline_requests():
    assert is_music_management_request("grab the new Sleep Token album in FLAC")
    assert is_music_management_request("dedupe my music library")
    assert is_music_management_request("retag these tracks with beets")
    assert is_music_management_request("plan adding this album to my library")


def test_does_not_route_casual_music_conversation():
    assert not is_music_management_request("what music do I like?")
    assert not is_music_management_request("tell me about Sleep Token")
    assert not is_music_management_request("download the latest Ubuntu image")
