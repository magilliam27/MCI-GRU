from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mci_gru.cockpit.runner import run_local_cockpit_refresh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh MCI-GRU cockpit Markdown artifacts. GitHub sync is disabled by default."
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument(
        "--github-sync",
        action="store_true",
        help="Reserved for guarded GitHub sync; disabled by default in the local runner.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.github_sync:
        raise SystemExit("GitHub sync is not implemented in the local-first runner yet.")
    result = run_local_cockpit_refresh(Path(args.repo_root).resolve(), date.fromisoformat(args.date))
    print(f"Wrote {result.register_path}")
    print(f"Wrote {result.packet_path}")
    print(f"Run color: {result.color.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
