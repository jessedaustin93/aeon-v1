"""Always-on Aeon runner.

The runner keeps lightweight local maintenance alive after the launcher starts
it. It exits cleanly when `memory/runtime/stop_runner` is created.
"""
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

from .background_consolidation import check_memory_growth
from .config import Config
from .linker import link_memories
from .runtime import (
    base_status,
    memory_counts,
    runner_status_path,
    runner_stop_path,
    write_json,
)
from .time_utils import utc_now_iso


def run_forever(
    config: Optional[Config] = None,
    poll_seconds: float = 5.0,
    link_every_passes: int = 12,
) -> Dict:
    """Run maintenance until the local stop file is created."""
    cfg = config or Config()
    cfg.ensure_dirs()
    stop_path = runner_stop_path(cfg)
    if stop_path.exists():
        stop_path.unlink()

    passes = 0
    started_at = utc_now_iso()
    last_link = None
    last_error = None

    while not stop_path.exists():
        passes += 1
        consolidation_started = False
        try:
            consolidation_started = check_memory_growth(cfg)
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
                last_link_at=last_link,
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
        default=12,
        help="Run link_memories every N passes. Use 0 to disable.",
    )
    args = parser.parse_args(argv)
    run_forever(
        config=Config(args.base_path),
        poll_seconds=args.poll_seconds,
        link_every_passes=args.link_every_passes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
