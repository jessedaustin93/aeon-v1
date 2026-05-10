"""Media ingestion for Aeon-V1.

Media files are stored append-only, then optional perception providers add a
description. The original file is kept; analysis can be rerun later with a
different local model.
"""
import base64
import json
import re
from pathlib import Path
from typing import Dict, Optional

from .config import Config
from .llm import generate_image_description, resolve_lmstudio_vision_model
from .memory_store import _extract_tags, _generate_id, _make_title, _vault_note_path, _write_markdown
from .time_utils import utc_now_iso

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def ingest_image_file(
    path: Path,
    source: str = "dashboard-upload",
    config: Optional[Config] = None,
    prompt: Optional[str] = None,
) -> Dict:
    """Store an image and append a media memory with optional vision analysis."""
    cfg = config or Config()
    cfg.ensure_dirs()
    source_path = Path(path)
    if source_path.suffix.lower() not in _IMAGE_EXTENSIONS:
        return {"media": None, "error": f"unsupported image extension: {source_path.suffix}"}
    if not source_path.exists():
        return {"media": None, "error": "image file does not exist"}

    media_id = _generate_id()
    stored_path = cfg.memory_path / "media" / "uploads" / f"{media_id}{source_path.suffix.lower()}"
    stored_path.write_bytes(source_path.read_bytes())
    return _store_image_memory(
        media_id=media_id,
        stored_path=stored_path,
        original_name=source_path.name,
        source=source,
        config=cfg,
        prompt=prompt,
    )


def ingest_image_bytes(
    data: bytes,
    filename: str,
    source: str = "dashboard-upload",
    config: Optional[Config] = None,
    prompt: Optional[str] = None,
) -> Dict:
    """Store uploaded image bytes and append a media memory."""
    cfg = config or Config()
    cfg.ensure_dirs()
    suffix = Path(filename).suffix.lower() or ".png"
    if suffix not in _IMAGE_EXTENSIONS:
        return {"media": None, "error": f"unsupported image extension: {suffix}"}

    media_id = _generate_id()
    safe_name = _safe_filename(filename)
    stored_path = cfg.memory_path / "media" / "uploads" / f"{media_id}{suffix}"
    stored_path.write_bytes(data)
    return _store_image_memory(
        media_id=media_id,
        stored_path=stored_path,
        original_name=safe_name,
        source=source,
        config=cfg,
        prompt=prompt,
    )


def ingest_image_data_url(
    data_url: str,
    filename: str,
    source: str = "dashboard-upload",
    config: Optional[Config] = None,
    prompt: Optional[str] = None,
) -> Dict:
    """Accept a browser data URL and ingest it as an image."""
    try:
        _, _, payload = data_url.partition(",")
        data = base64.b64decode(payload, validate=True)
    except Exception:
        return {"media": None, "error": "invalid image data"}
    return ingest_image_bytes(data, filename, source=source, config=config, prompt=prompt)


def _store_image_memory(
    media_id: str,
    stored_path: Path,
    original_name: str,
    source: str,
    config: Config,
    prompt: Optional[str],
) -> Dict:
    now = utc_now_iso()
    analysis_prompt = prompt or (
        "Describe this image for a local memory system. Include visible objects, "
        "scene context, any readable text, and why it may be useful to remember. "
        "Be factual and concise."
    )
    analysis_model = resolve_lmstudio_vision_model(config)
    description = generate_image_description(stored_path, analysis_prompt, config=config)
    analysis_status = "complete" if description else "unavailable"
    description_text = description or (
        "Image stored, but no vision analysis was available. Load a vision-capable "
        "LM Studio model and reprocess later."
    )
    tags = sorted(set(["media", "image"] + _extract_tags(description_text)))
    title = _make_title(description_text if description else original_name)
    rel_path = stored_path.relative_to(config.base_path).as_posix()

    record = {
        "id": media_id,
        "title": title,
        "type": "media",
        "media_type": "image",
        "created": now,
        "source": source,
        "source_path": rel_path,
        "original_name": original_name,
        "description": description_text,
        "analysis_status": analysis_status,
        "analysis_model": analysis_model,
        "tags": tags,
        "links": [],
    }

    (config.memory_path / "media" / f"{media_id}.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        _vault_note_path(config, "media", media_id),
        frontmatter={
            "id": media_id,
            "title": title,
            "type": "media",
            "media_type": "image",
            "created": now,
            "source": source,
            "analysis_status": analysis_status,
            "tags": tags,
            "links": [],
        },
        body=(
            f"# {title}\n\n"
            f"**Original file:** {original_name}\n\n"
            f"**Stored file:** `{rel_path}`\n\n"
            f"**Analysis status:** {analysis_status}\n\n"
            "## Description\n"
            f"{description_text}\n\n"
            "[[Raw Memory]] | [[Semantic Memory]] | [[Core Memory]]"
        ),
        config=config,
    )
    return {"media": record, "error": None}


def _safe_filename(filename: str) -> str:
    name = Path(filename).name or "upload.png"
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)
