"""One-click local launcher for Aeon."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeon_v1.dashboard import run_dashboard
from aeon_v1.launcher_config import load_launcher_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the Aeon local dashboard.")
    parser.add_argument("--base-path", type=Path, default=ROOT, help="Aeon repo/base path.")
    parser.add_argument("--config", type=Path, default=None, help="Optional launcher config JSON.")
    parser.add_argument("--host", default=None, help="Dashboard host override.")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port override.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    config = load_launcher_config(args.base_path, args.config)
    dashboard = config.get("dashboard", {})
    host = args.host or dashboard.get("host", "127.0.0.1")
    port = args.port or int(dashboard.get("port", 8765))
    open_browser = bool(dashboard.get("open_browser", True)) and not args.no_browser
    run_dashboard(
        base_path=args.base_path,
        host=host,
        port=port,
        launcher_config_path=args.config,
        open_browser=open_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
