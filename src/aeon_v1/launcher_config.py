"""Portable local launcher configuration."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_LAUNCHER_CONFIG: Dict[str, Any] = {
    "dashboard": {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": True,
    },
    "runner": {
        "poll_seconds": 5,
        "link_every_passes": 60,
        "reflect_every_passes": 720,
        "consolidate_every_passes": 120,
    },
    "lm_studio": {
        "enabled": True,
        "base_url": "http://localhost:1234/v1",
        "command": "",
    },
    "obsidian": {
        "enabled": True,
        "command": "",
        "vault_name": "",
        "vault_path": "vault",
    },
}


def load_launcher_config(base_path: Path, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load local launcher settings without requiring machine-specific repo data."""
    config = json.loads(json.dumps(DEFAULT_LAUNCHER_CONFIG))
    path = config_path or base_path / "local" / "launcher_config.json"
    if not path.exists():
        return config
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return config
    _deep_update(config, loaded)
    return config


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
