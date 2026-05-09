"""Runtime status helpers for Aeon local processes."""
import json
import os
from pathlib import Path
from typing import Dict

from .config import Config
from .time_utils import utc_now_iso


def runtime_path(config: Config) -> Path:
    path = config.memory_path / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runner_status_path(config: Config) -> Path:
    return runtime_path(config) / "runner_status.json"


def launcher_status_path(config: Config) -> Path:
    return runtime_path(config) / "launcher_status.json"


def runner_stop_path(config: Config) -> Path:
    return runtime_path(config) / "stop_runner"


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def memory_counts(config: Config) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for memory_type in ("raw", "episodic", "semantic", "reflections", "consolidations", "media"):
        path = config.memory_path / memory_type
        if not path.exists():
            counts[memory_type] = 0
            continue
        counts[memory_type] = len([
            p for p in path.glob("*.json")
            if p.name != "trigger_state.json"
        ])
    return counts


def base_status(config: Config, component: str, state: str, **extra: object) -> Dict:
    return {
        "component": component,
        "state": state,
        "updated_at": utc_now_iso(),
        **extra,
    }
