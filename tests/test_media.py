"""Tests for append-only media ingestion."""
import base64
import json
from unittest.mock import patch

from aeon_v1 import Config, ingest_image_bytes, ingest_image_data_url, search
from aeon_v1.memory_store import _vault_note_path

_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_image_ingestion_stores_file_json_and_vault_note(tmp_path):
    cfg = Config(tmp_path)
    cfg.llm_enabled = True
    cfg.llm_provider = "lmstudio"
    cfg.llm_vision_model = "vendor/vision-model"

    with patch("aeon_v1.media.generate_image_description", return_value="A tiny test image."):
        result = ingest_image_bytes(_PNG_1X1, "tiny.png", config=cfg)

    record = result["media"]
    assert record["type"] == "media"
    assert record["media_type"] == "image"
    assert record["analysis_status"] == "complete"
    assert record["analysis_model"] == "vendor/vision-model"
    assert (cfg.memory_path / "media" / f"{record['id']}.json").exists()
    assert _vault_note_path(cfg, "media", record["id"]).exists()
    assert (cfg.base_path / record["source_path"]).exists()

    data = json.loads((cfg.memory_path / "media" / f"{record['id']}.json").read_text())
    assert data["description"] == "A tiny test image."


def test_image_ingestion_is_searchable_by_description(tmp_path):
    cfg = Config(tmp_path)
    with patch("aeon_v1.media.generate_image_description", return_value="Screenshot of a dashboard with memory counts."):
        ingest_image_bytes(_PNG_1X1, "dashboard.png", config=cfg)

    results = search("memory counts", config=cfg)

    assert any(r["match_type"] == "media" for r in results)


def test_image_data_url_invalid_payload_returns_error(tmp_path):
    cfg = Config(tmp_path)

    result = ingest_image_data_url("data:image/png;base64,not-valid", "bad.png", config=cfg)

    assert result["media"] is None
    assert result["error"]
