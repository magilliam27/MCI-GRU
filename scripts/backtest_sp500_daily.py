"""Thin CLI wrapper for the replay's daily open-to-open variant.

The engine body lives in ``mci_gru/evaluation/backtest_engine.py`` (WS-N).
This CLI exposes the historical daily subset of flags (plus ``--run_dir`` and
``--pit_export_cutoff`` from #339, ``--trial_ledger`` from #340) and delegates to the
engine with ``--strategy open_to_open --holding_period 1`` (buy at open D+1, sell at
open D+2: the paper's hold, a labelled variant since #340; the default replay trades
the training label close to close) and MLflow off, auto-link included. Inputs have no
defaults: ``--run_dir`` supplies them from the run, or every one is named explicitly,
exactly as in the engine CLI. ``--multi_model`` is gone: it averaged member metrics
rather than scoring the ensemble the run reports.

Python importers may import public symbols re-exported from the engine module.
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
        "--run_dir",
        type=str,
        default=None,
        help="One window's run_experiment.py output; supplies and checks every input",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="Directory containing prediction CSV files (required without --run_dir)",
    )
    parser.add_argument("--data_file", type=str, default=None, help="Stock data CSV")
    parser.add_argument(
        "--pit_universe_csv",
        type=str,
        default=None,
        help="PIT kdcode/valid_from/valid_to CSV for candidate and benchmark membership",
    )
    parser.add_argument(
        "--pit_export_cutoff",
        type=str,
        default=None,
        help="Membership end for PIT rows with a blank valid_to",
    )
    parser.add_argument("--top_k", type=int, default=None, help="Number of top stocks to select")
    parser.add_argument("--test_start", type=str, default=None, help="Test period start date")
    parser.add_argument("--test_end", type=str, default=None, help="Test period end date")
    parser.add_argument("--label_t", type=int, default=None, help="Forward return period in days")
    parser.add_argument("--output", type=str, default=None, help="Output path for results CSV")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plot")
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
        default="holm",
        choices=["holm", "bonferroni", "bhy"],
        help="Multiple testing adjustment method (default: holm). holm and bonferroni "
        "need only --num_tests; bhy adjusts a family of p-values and needs --trial_ledger.",
    )
    parser.add_argument(
        "--trial_ledger",
        type=str,
        default=None,
        help="CSV with one row per OTHER trial in the family and its p_value (for bhy)",
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
    argv: list[str] = []
    for flag in (
        "run_dir",
        "predictions_dir",
        "data_file",
        "pit_universe_csv",
        "pit_export_cutoff",
        "top_k",
        "test_start",
        "test_end",
        "label_t",
    ):
        value = getattr(args, flag)
        if value is not None:
            argv.extend([f"--{flag}", str(value)])
    argv += [
        "--strategy",
        "open_to_open",
        "--holding_period",
        "1",
        "--disable_mlflow_autolink",
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
    if args.output is not None:
        argv.extend(["--output", args.output])
    if args.trial_ledger is not None:
        argv.extend(["--trial_ledger", args.trial_ledger])
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
    """Run the open-to-open daily variant via the engine, MLflow off."""
    daily_argv = sys.argv[1:] if argv is None else argv
    args = _build_daily_parser().parse_args(daily_argv)
    _engine_main(_namespace_to_engine_argv(args))


if __name__ == "__main__":
    main()
