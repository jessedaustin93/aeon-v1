"""Count-driven background consolidation trigger.

The trigger is intentionally based on memory growth instead of wall-clock time:
after N new source memories are written, a daemon thread runs one consolidation
pass. The counter is persisted locally so restarts do not lose progress.
"""
import json
import threading
from pathlib import Path
from typing import Dict

from .config import Config
from .consolidate import consolidate_memories
from .time_utils import utc_now_iso

_LOCK = threading.RLock()
_RUNNING: set[str] = set()


def notify_memory_created(config: Config, memory_type: str) -> None:
    """Start a background consolidation pass when enough new memories exist."""
    if not getattr(config, "enable_background_consolidation", False):
        return
    if memory_type not in getattr(config, "consolidation_trigger_memory_types", []):
        return
    check_memory_growth(config)


def check_memory_growth(config: Config) -> bool:
    """Check persisted memory counts and launch consolidation if due.

    Returns True when a consolidation thread was started. A long-running runner
    can call this periodically so memories written by any process still advance
    the count-driven maintenance loop.
    """
    if not getattr(config, "enable_background_consolidation", False):
        return False

    interval = int(getattr(config, "consolidation_trigger_interval", 0) or 0)
    if interval <= 0:
        return False

    config.ensure_dirs()
    state = _load_state(config)
    current_total = _source_memory_count(config)
    last_checked = int(state.get("last_checked_total", 0) or 0)

    if current_total - last_checked < interval:
        return False

    base_key = str(config.base_path.resolve())
    with _LOCK:
        if base_key in _RUNNING:
            return False
        _RUNNING.add(base_key)

    thread = threading.Thread(
        target=_run_consolidation_pass,
        args=(config, current_total, base_key),
        name="aeon-memory-consolidator",
        daemon=True,
    )
    thread.start()
    return True


def _run_consolidation_pass(config: Config, checked_total: int, base_key: str) -> None:
    result: Dict = {}
    error = None
    try:
        result = consolidate_memories(config=config)
    except Exception as exc:
        error = repr(exc)
    finally:
        state = {
            "last_checked_at": utc_now_iso(),
            "last_checked_total": checked_total,
            "last_result": {
                "created_count": result.get("created_count", 0),
                "candidate_groups": result.get("candidate_groups", 0),
                "skipped_existing": result.get("skipped_existing", 0),
                "error": error,
            },
        }
        _state_path(config).parent.mkdir(parents=True, exist_ok=True)
        _state_path(config).write_text(json.dumps(state, indent=2), encoding="utf-8")
        with _LOCK:
            _RUNNING.discard(base_key)


def _source_memory_count(config: Config) -> int:
    total = 0
    for memory_type in getattr(config, "consolidation_trigger_memory_types", []):
        path = config.memory_path / memory_type
        if path.exists():
            total += len(list(path.glob("*.json")))
    return total


def _load_state(config: Config) -> Dict:
    path = _state_path(config)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _state_path(config: Config) -> Path:
    return config.memory_path / "consolidations" / "trigger_state.json"
