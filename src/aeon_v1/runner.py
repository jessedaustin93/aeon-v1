"""Always-on Aeon runner.

The runner keeps lightweight local maintenance alive after the launcher starts
it. It exits cleanly when `memory/runtime/stop_runner` is created.

Maintenance passes each poll tick:
  1. Fresh-memory reflection: if new memories arrived since the last pass,
     reflect on just those memories after a small debounce window/batch.
  2. Chunk reflection: on every `reflect_every_passes`, advance the sequential
     cursor through the whole memory archive one chunk at a time.
  3. Count-driven consolidation: check_memory_growth fires whenever enough new
     memories have accumulated (config.consolidation_trigger_interval).
  4. Timed consolidation: every `consolidate_every_passes`, run a consolidation
     pass regardless of memory growth — keeps duplicate-detection current.
  5. Linking: every `link_every_passes`, run link_memories to build cross-refs.
"""
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

from .background_consolidation import check_memory_growth
from .config import Config
from .consolidate import consolidate_memories
from .linker import link_memories
from .reflect import reflect
from .runtime import (
    base_status,
    memory_counts,
    runner_status_path,
    runner_stop_path,
    write_json,
)
from .time_utils import utc_now_iso

# Source memory types tracked for fresh-memory detection.
_FRESH_TYPES = ("raw", "episodic", "semantic")


def _source_count(cfg: Config) -> int:
    counts = memory_counts(cfg)
    return sum(counts.get(t, 0) for t in _FRESH_TYPES)


def _fresh_reflection_due(cfg: Config, current_count: int, last_count: int, last_at: float) -> bool:
    new_count = current_count - last_count
    if new_count <= 0:
        return False
    min_new = max(1, int(getattr(cfg, "fresh_reflection_min_new_memories", 3) or 3))
    min_seconds = max(0, int(getattr(cfg, "fresh_reflection_min_seconds", 60) or 0))
    return new_count >= min_new or (time.monotonic() - last_at) >= min_seconds


def run_forever(
    config: Optional[Config] = None,
    poll_seconds: float = 5.0,
    link_every_passes: int = 60,
    reflect_every_passes: int = 0,
    consolidate_every_passes: int = 120,
) -> Dict:
    """Run maintenance until the local stop file is created."""
    cfg = config or Config()
    cfg.ensure_dirs()
    # CLI args take precedence over Config defaults; 0 means use Config value.
    if reflect_every_passes > 0:
        cfg.archive_reflection_every_passes = reflect_every_passes

    stop_path = runner_stop_path(cfg)
    if stop_path.exists():
        stop_path.unlink()

    passes = 0
    started_at = utc_now_iso()

    # Fresh-memory tracking: remember count and timestamp of last fresh pass.
    last_fresh_count = _source_count(cfg)
    last_fresh_at = utc_now_iso()
    last_fresh_check = time.monotonic()

    last_link = None
    last_reflection = None
    last_consolidation = None
    last_error = None
    fresh_reflections = 0

    while not stop_path.exists():
        passes += 1
        consolidation_started = False
        try:
            # --- 1. Fresh-memory reflection (immediate, on every pass) ----------
            current_count = _source_count(cfg)
            if _fresh_reflection_due(cfg, current_count, last_fresh_count, last_fresh_check):
                result = reflect(config=cfg, since_timestamp=last_fresh_at, force=True)
                if result.get("reflection"):
                    fresh_reflections += 1
                    last_reflection = utc_now_iso()
                last_fresh_at = utc_now_iso()
                last_fresh_count = current_count
                last_fresh_check = time.monotonic()

            # --- 2. Chunk reflection (scheduled, advances cursor) ---------------
            archive_reflection_every = int(
                getattr(cfg, "archive_reflection_every_passes", 0) or 0
            )
            if archive_reflection_every > 0 and passes % archive_reflection_every == 0:
                result = reflect(config=cfg)
                if result.get("reflection"):
                    last_reflection = utc_now_iso()

            # --- 3. Count-driven consolidation ----------------------------------
            consolidation_started = check_memory_growth(cfg)

            # --- 4. Timed consolidation -----------------------------------------
            if consolidate_every_passes > 0 and passes % consolidate_every_passes == 0:
                consolidate_memories(config=cfg)
                last_consolidation = utc_now_iso()

            # --- 5. Linking -----------------------------------------------------
            if link_every_passes > 0 and passes % link_every_passes == 0:
                link_memories(config=cfg)
                last_link = utc_now_iso()

            last_error = None
        except Exception as exc:
            last_error = repr(exc)

        write_json(
            runner_status_path(cfg),
            base_status(
                cfg,
                "runner",
                "running",
                pid=os.getpid(),
                started_at=started_at,
                heartbeat_at=utc_now_iso(),
                passes=passes,
                poll_seconds=poll_seconds,
                link_every_passes=link_every_passes,
                reflect_every_passes=reflect_every_passes,
                consolidate_every_passes=consolidate_every_passes,
                last_link_at=last_link,
                last_reflection_at=last_reflection,
                last_consolidation_at=last_consolidation,
                fresh_reflections=fresh_reflections,
                consolidation_started=consolidation_started,
                memory_counts=memory_counts(cfg),
                last_error=last_error,
            ),
        )
        time.sleep(max(0.1, poll_seconds))

    stopped = base_status(
        cfg,
        "runner",
        "stopped",
        pid=os.getpid(),
        started_at=started_at,
        stopped_at=utc_now_iso(),
        passes=passes,
        fresh_reflections=fresh_reflections,
        memory_counts=memory_counts(cfg),
        last_error=last_error,
    )
    write_json(runner_status_path(cfg), stopped)
    try:
        stop_path.unlink()
    except OSError:
        pass
    return stopped


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aeon background maintenance.")
    parser.add_argument("--base-path", type=Path, default=Path("."), help="Aeon repo/base path.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Maintenance poll interval.")
    parser.add_argument(
        "--link-every-passes",
        type=int,
        default=60,
        help="Run link_memories every N passes. Use 0 to disable.",
    )
    parser.add_argument(
        "--reflect-every-passes",
        type=int,
        default=0,
        help="Run reflect() every N passes. 0 uses the Config default (archive_reflection_every_passes).",
    )
    parser.add_argument(
        "--consolidate-every-passes",
        type=int,
        default=120,
        help="Run consolidate_memories() every N passes regardless of memory growth. Use 0 to disable.",
    )
    args = parser.parse_args(argv)
    run_forever(
        config=Config(args.base_path),
        poll_seconds=args.poll_seconds,
        link_every_passes=args.link_every_passes,
        reflect_every_passes=args.reflect_every_passes,
        consolidate_every_passes=args.consolidate_every_passes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
