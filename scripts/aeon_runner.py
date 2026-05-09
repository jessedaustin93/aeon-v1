"""Start the Aeon always-on local runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aeon_v1.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
