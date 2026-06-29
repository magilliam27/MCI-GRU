from __future__ import annotations

import argparse
from pathlib import Path

from mci_gru.evaluation.trial_ledger import build_trial_record, write_trial_ledger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trial ledger from existing run folders.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status", default="UNKNOWN")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = [
        build_trial_record(
            Path(run_dir),
            trial_id=Path(run_dir).name,
            family_id=args.family_id,
            status=args.status,
        )
        for run_dir in args.run_dir
    ]
    paths = write_trial_ledger(records, args.output_dir, force=args.force)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
