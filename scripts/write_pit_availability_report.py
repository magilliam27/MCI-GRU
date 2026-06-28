from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mci_gru.data.pit_audit import build_pit_availability_report
from mci_gru.evaluation.artifacts import write_json_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a report-only PIT availability audit.")
    parser.add_argument("--market-data", required=True)
    parser.add_argument("--pit-universe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calendar", default=None)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--stale-after-days", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_pit_availability_report(
        pd.read_csv(args.market_data),
        pd.read_csv(args.pit_universe),
        min_price=args.min_price,
        min_dollar_volume=args.min_dollar_volume,
        stale_after_days=args.stale_after_days,
        calendar=_load_calendar(args.calendar) if args.calendar else None,
    )
    path = write_json_artifact(Path(args.output), report, force=args.force)
    print(f"pit_availability_report: {path}")


def _load_calendar(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    if "dt" in frame.columns:
        return frame["dt"].astype(str).tolist()
    return frame.iloc[:, 0].astype(str).tolist()


if __name__ == "__main__":
    main()
