from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cockpit.runner import run_github_cockpit_refresh, run_local_cockpit_refresh  # noqa: E402,I001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh MCI-GRU cockpit Markdown artifacts. GitHub sync is disabled by default."
    )
    parser.add_argument(
        "--date", default=date.today().isoformat(), help="Run date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--repo-root", default=".", help="Repository root. Defaults to current directory."
    )
    parser.add_argument(
        "--github-sync",
        action="store_true",
        help="Reserved for guarded GitHub sync; disabled by default in the local runner.",
    )
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--auto-decisions",
        dest="auto_decisions",
        action="store_true",
        default=True,
        help="Generate and apply deterministic disposition policy decisions (default).",
    )
    policy.add_argument(
        "--no-auto-decisions",
        dest="auto_decisions",
        action="store_false",
        help="Disable generated decisions and reproduce the legacy refresh behavior.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_date = date.fromisoformat(args.date)
    policy_options = {"auto_decisions_enabled": args.auto_decisions}
    if args.github_sync:
        result = run_github_cockpit_refresh(repo_root, run_date, **policy_options)
    else:
        result = run_local_cockpit_refresh(repo_root, run_date, **policy_options)
    print(f"Wrote {result.register_path}")
    print(f"Wrote {result.packet_path}")
    print(f"Run color: {result.color.value}")
    if result.github is not None:
        print(f"GitHub PR: {result.github.pr_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
