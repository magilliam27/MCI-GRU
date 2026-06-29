from __future__ import annotations

import argparse

import pandas as pd

from mci_gru.evaluation.capacity import compute_capacity_replay, write_capacity_replay
from mci_gru.evaluation.prediction_report import load_prediction_input


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved predictions through cost-aware capacity diagnostics."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--market-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aum", type=float, action="append", required=True)
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--adv-lookback-days", type=int, default=20)
    parser.add_argument("--max-adv-participation", type=float, default=0.10)
    parser.add_argument("--spread-bps", type=float, action="append", default=None)
    parser.add_argument("--slippage-bps", type=float, action="append", default=None)
    parser.add_argument("--min-rank-drop", type=int, action="append", default=None)
    parser.add_argument("--max-lagged-volatility", type=float, action="append", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank_drop_values: list[int | None] = args.min_rank_drop if args.min_rank_drop else [None]
    report = compute_capacity_replay(
        load_prediction_input(args.predictions),
        pd.read_csv(args.market_data),
        aum_values=args.aum,
        top_k_values=args.top_k or [10],
        adv_lookback_days=args.adv_lookback_days,
        max_adv_participation=args.max_adv_participation,
        spread_bps_values=args.spread_bps or [0.0],
        slippage_bps_values=args.slippage_bps or [0.0],
        rank_drop_values=rank_drop_values,
        max_lagged_volatility_values=args.max_lagged_volatility,
    )
    paths = write_capacity_replay(report, args.output_dir, force=args.force)
    print(f"capacity_replay_json: {paths['json']}")
    print(f"capacity_replay_csv: {paths['csv']}")


if __name__ == "__main__":
    main()
