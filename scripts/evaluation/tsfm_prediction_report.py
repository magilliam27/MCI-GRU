#!/usr/bin/env python
"""Generate a TSFM-style report from saved MCI-GRU prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from mci_gru.evaluation import write_tsfm_prediction_report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baselines = _parse_baselines(args.baseline)
    result = write_tsfm_prediction_report(
        predictions_dir=args.predictions_dir,
        market_data_path=args.market_data,
        output_dir=args.output_dir,
        label_t=args.label_t,
        baseline_prediction_paths=baselines,
        top_k_values=args.top_k,
    )
    paths = result["paths"]
    print(f"Aligned rows: {paths['aligned_csv']}")
    print(f"JSON report:  {paths['json']}")
    print(f"Markdown:     {paths['markdown']}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved MCI-GRU predictions with TSFM-style forecast metrics."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing saved averaged_predictions/*.csv files.",
    )
    parser.add_argument(
        "--market-data",
        type=Path,
        required=True,
        help="Market CSV with dt, kdcode, and close columns used to derive realized returns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for tsfm_prediction_report.json, .md, and aligned CSV outputs.",
    )
    parser.add_argument(
        "--label-t",
        type=int,
        default=5,
        help="Forward label horizon matching the saved prediction run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="*",
        default=[10, 20, 50, 100],
        help="Top-k portfolio diagnostics to include.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional external baseline prediction CSV or directory. Repeat for many baselines.",
    )
    return parser.parse_args(argv)


def _parse_baselines(values: list[str]) -> dict[str, Path]:
    baselines: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Baseline must be NAME=PATH, got: {value}")
        name, path = value.split("=", maxsplit=1)
        name = name.strip()
        if not name:
            raise ValueError(f"Baseline name is empty in: {value}")
        baselines[name] = Path(path.strip())
    return baselines


if __name__ == "__main__":
    raise SystemExit(main())
