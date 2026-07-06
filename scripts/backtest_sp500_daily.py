"""Thin CLI wrapper for daily-only backtest (holding_period=1).

The engine body lives in ``mci_gru/evaluation/backtest_engine.py`` (WS-N).
This CLI exposes the historical daily subset of flags (20 shared flags from
the pre-merge fork) and delegates to the engine with ``holding_period=1``,
default ``rebalance_style``, and MLflow tracking disabled.

Python importers (e.g. ``tests/backtest_sp500_daily.py``) may import public
symbols re-exported from the engine module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mci_gru.evaluation.backtest_engine import *  # noqa: F403,E402
from mci_gru.evaluation.backtest_engine import main as _engine_main  # noqa: E402


def _build_daily_parser() -> argparse.ArgumentParser:
    """Argparse surface matching the pre-merge daily CLI (20 shared flags)."""
    parser = argparse.ArgumentParser(
        description="Evaluate MCI-GRU model predictions per paper methodology"
    )

    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing prediction CSV files",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/raw/market/sp500_data.csv",
        help="Path to stock data CSV file",
    )
    parser.add_argument(
        "--pit_universe_csv",
        type=str,
        default=None,
        help="Optional PIT kdcode/valid_from/valid_to CSV for candidate and benchmark filtering",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top stocks to select (default: 10)"
    )
    parser.add_argument(
        "--test_start",
        type=str,
        default="2025-01-01",
        help="Test period start date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--test_end",
        type=str,
        default="2025-12-31",
        help="Test period end date (default: 2025-12-31)",
    )
    parser.add_argument(
        "--label_t", type=int, default=5, help="Forward return period in days (default: 5)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for results CSV")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plot")
    parser.add_argument(
        "--multi_model",
        type=str,
        default=None,
        help="Base path for multi-model evaluation (10 runs)",
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=10,
        help="Number of model runs for multi-model evaluation (default: 10)",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=1,
        help="Number of strategies/configurations tested for multiple testing "
        "adjustment (default: 1 = no adjustment). If you tried 50 "
        "hyperparameter combinations, set --num_tests 50",
    )
    parser.add_argument(
        "--adjustment_method",
        type=str,
        default="bhy",
        choices=["bhy", "bonferroni", "holm"],
        help="Multiple testing adjustment method (default: bhy). "
        "bhy = Benjamini-Hochberg-Yekutieli (recommended for trading), "
        "bonferroni = most stringent (FWER), "
        "holm = step-down FWER method",
    )
    parser.add_argument(
        "--transaction_costs",
        action="store_true",
        help="Enable transaction cost modeling (bid-ask spread + slippage)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=10.0,
        help="Bid-ask spread in basis points (default: 10 bps = 0.10%% round-trip). "
        "For S&P 500 large-caps, typical spreads are 5-15 bps.",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=5.0,
        help="Slippage in basis points per trade (default: 5 bps = 0.05%%). "
        "Represents execution price deviation for market orders.",
    )
    parser.add_argument(
        "--auto_save",
        action="store_true",
        help="Automatically save all outputs in organized structure",
    )
    parser.add_argument(
        "--backtest_suffix",
        type=str,
        default="",
        help='Suffix for backtest directory (e.g., "_with_costs" or "_tc")',
    )
    parser.add_argument(
        "--enable_rank_drop_gate",
        action="store_true",
        help="Enable rank-drop sell gate: only exit held stocks whose prediction rank "
        "worsened by at least --min_rank_drop vs the previous prediction day",
    )
    parser.add_argument(
        "--min_rank_drop",
        type=int,
        default=10,
        help="Minimum rank worsening (current_rank - prev_rank) required to exit a held stock (default: 10)",
    )
    return parser


def _namespace_to_engine_argv(args: argparse.Namespace) -> list[str]:
    """Translate daily CLI namespace to engine argv (omits engine-only flags)."""
    argv: list[str] = [
        "--predictions_dir",
        args.predictions_dir,
        "--data_file",
        args.data_file,
        "--top_k",
        str(args.top_k),
        "--test_start",
        args.test_start,
        "--test_end",
        args.test_end,
        "--label_t",
        str(args.label_t),
        "--num_models",
        str(args.num_models),
        "--num_tests",
        str(args.num_tests),
        "--adjustment_method",
        args.adjustment_method,
        "--spread",
        str(args.spread),
        "--slippage",
        str(args.slippage),
        "--backtest_suffix",
        args.backtest_suffix,
        "--min_rank_drop",
        str(args.min_rank_drop),
    ]
    if args.pit_universe_csv is not None:
        argv.extend(["--pit_universe_csv", args.pit_universe_csv])
    if args.output is not None:
        argv.extend(["--output", args.output])
    if args.multi_model is not None:
        argv.extend(["--multi_model", args.multi_model])
    if args.plot:
        argv.append("--plot")
    if args.transaction_costs:
        argv.append("--transaction_costs")
    if args.auto_save:
        argv.append("--auto_save")
    if args.enable_rank_drop_gate:
        argv.append("--enable_rank_drop_gate")
    return argv


def main(argv: list[str] | None = None) -> None:
    """Run daily backtest via engine with holding_period=1 and MLflow off."""
    daily_argv = sys.argv[1:] if argv is None else argv
    args = _build_daily_parser().parse_args(daily_argv)
    _engine_main(_namespace_to_engine_argv(args))


if __name__ == "__main__":
    main()
