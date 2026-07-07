"""Re-run saved PIT masked-panel predictions through cost-aware daily backtests.

This script is intentionally orchestration-only. It never calls training code;
it locates existing yearly ``averaged_predictions`` folders and invokes
``scripts/backtest_sp500_daily.py`` with transaction costs and the rank-drop gate
enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PIT_WINDOWS: dict[int, dict[str, str]] = {
    2022: {
        "experiment_name": "pit_true_rolling_2022",
        "test_start": "2022-01-22",
        "test_end": "2022-12-31",
    },
    2023: {
        "experiment_name": "pit_true_rolling_2023",
        "test_start": "2023-01-22",
        "test_end": "2023-12-31",
    },
    2024: {
        "experiment_name": "pit_true_rolling_2024",
        "test_start": "2024-01-22",
        "test_end": "2024-12-31",
    },
    2025: {
        "experiment_name": "pit_true_rolling_2025",
        "test_start": "2025-01-22",
        "test_end": "2025-12-31",
    },
}

DEFAULT_MARKET_CSV = Path("data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv")
DEFAULT_PIT_UNIVERSE_CSV = Path(
    "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
)
DEFAULT_BACKTEST_SUFFIX = "_pit_daily_tc_rank_gate"
COMPARISON_METRICS = [
    "total_return",
    "excess_return",
    "ARR",
    "ASR",
    "MDD",
    "avg_daily_turnover",
    "num_trading_days",
]


@dataclass(frozen=True)
class BacktestJob:
    year: int
    run_dir: Path
    predictions_dir: Path
    backtest_dir: Path
    test_start: str
    test_end: str


@dataclass(frozen=True)
class BacktestOptions:
    repo_dir: Path
    python_executable: str
    backtest_script: Path
    data_file: Path
    pit_universe_csv: Path
    backtest_suffix: str = DEFAULT_BACKTEST_SUFFIX
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    min_rank_drop: int = 30
    top_k: int = 10
    label_t: int = 5
    num_tests: int = 1
    adjustment_method: str = "bhy"


def build_backtest_command(job: BacktestJob, options: BacktestOptions) -> list[str]:
    """Build the exact daily-backtest command for a saved yearly prediction folder."""
    return [
        options.python_executable,
        "-X",
        "utf8",
        str(options.backtest_script),
        "--predictions_dir",
        str(job.predictions_dir),
        "--data_file",
        str(options.data_file),
        "--pit_universe_csv",
        str(options.pit_universe_csv),
        "--test_start",
        job.test_start,
        "--test_end",
        job.test_end,
        "--top_k",
        str(options.top_k),
        "--label_t",
        str(options.label_t),
        "--num_tests",
        str(options.num_tests),
        "--adjustment_method",
        options.adjustment_method,
        "--auto_save",
        "--backtest_suffix",
        options.backtest_suffix,
        "--transaction_costs",
        "--spread",
        _format_number(options.spread_bps),
        "--slippage",
        _format_number(options.slippage_bps),
        "--enable_rank_drop_gate",
        "--min_rank_drop",
        str(options.min_rank_drop),
    ]


def resolve_year_jobs(
    run_root: Path,
    training_results_csv: Path | None,
    years: list[int],
    backtest_suffix: str,
) -> list[BacktestJob]:
    """Resolve yearly saved prediction folders from training_results.csv or run-root layout."""
    run_root = run_root.expanduser().resolve()
    rows_by_year = _training_rows_by_year(training_results_csv)
    jobs: list[BacktestJob] = []

    for year in years:
        if year not in PIT_WINDOWS:
            raise ValueError(f"No default PIT window is known for year {year}.")

        window = PIT_WINDOWS[year]
        row = rows_by_year.get(year, {})
        predictions_dir = _resolve_predictions_dir(row, run_root, window["experiment_name"])
        run_dir = predictions_dir.parent
        jobs.append(
            BacktestJob(
                year=year,
                run_dir=run_dir,
                predictions_dir=predictions_dir,
                backtest_dir=run_dir / f"backtest{backtest_suffix}",
                test_start=str(row.get("test_start") or window["test_start"]),
                test_end=str(row.get("test_end") or window["test_end"]),
            )
        )

    return jobs


def write_summary_outputs(
    output_dir: Path,
    cost_rows: list[dict[str, Any]],
    reviewed_rows: list[dict[str, Any]] | None,
    command_lines: list[str],
) -> dict[str, Path]:
    """Write yearly, side-by-side, combined CSVs and a Markdown reproducibility note."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_rows = reviewed_rows or []

    yearly_csv = output_dir / "cost_rank_gate_yearly_backtest_results.csv"
    side_by_side_csv = output_dir / "cost_rank_gate_vs_reviewed_side_by_side.csv"
    combined_summary_csv = output_dir / "cost_rank_gate_2022_2025_summary.csv"
    summary_md = output_dir / "cost_rank_gate_reproducibility.md"

    cost_df = pd.DataFrame(cost_rows)
    reviewed_df = pd.DataFrame(reviewed_rows)

    cost_df.to_csv(yearly_csv, index=False)

    side_by_side_df = _build_side_by_side_df(cost_df, reviewed_df)
    side_by_side_df.to_csv(side_by_side_csv, index=False)

    combined_df = _build_combined_summary_df(cost_df, reviewed_df)
    combined_df.to_csv(combined_summary_csv, index=False)

    summary_md.write_text(
        _build_summary_markdown(cost_df, side_by_side_df, combined_df, command_lines),
        encoding="utf-8",
    )

    return {
        "yearly_csv": yearly_csv,
        "side_by_side_csv": side_by_side_csv,
        "combined_summary_csv": combined_summary_csv,
        "summary_md": summary_md,
    }


def run_backtest_job(
    job: BacktestJob,
    options: BacktestOptions,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one yearly command and collect metrics from its emitted backtest_results.csv."""
    cmd = build_backtest_command(job, options)
    command_line = command_to_text(cmd)
    logs_dir = output_dir / "logs" / str(job.year)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "backtest_stdout.log"
    stderr_path = logs_dir / "backtest_stderr.log"

    row: dict[str, Any] = {
        "year": job.year,
        "status": "DRY_RUN" if dry_run else "PENDING",
        "returncode": np.nan if dry_run else None,
        "run_dir": str(job.run_dir),
        "predictions_dir": str(job.predictions_dir),
        "backtest_dir": str(job.backtest_dir),
        "test_start": job.test_start,
        "test_end": job.test_end,
        "command": command_line,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "scenario.transaction_costs_enabled": True,
        "scenario.spread_bps": options.spread_bps,
        "scenario.slippage_bps": options.slippage_bps,
        "scenario.rank_gate_enabled": True,
        "scenario.min_rank_drop": options.min_rank_drop,
    }

    if dry_run:
        stdout_path.write_text(command_line + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return row

    _validate_job_inputs(job, options)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=options.repo_dir,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")

    row["status"] = "OK" if proc.returncode == 0 else "FAILED"
    row["returncode"] = proc.returncode

    result_csv = job.backtest_dir / "backtest_results.csv"
    if result_csv.exists():
        result_df = pd.read_csv(result_csv)
        if len(result_df) > 0:
            row.update({f"backtest.{k}": v for k, v in result_df.iloc[0].to_dict().items()})
            yearly_copy_dir = output_dir / "yearly" / str(job.year)
            yearly_copy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_csv, yearly_copy_dir / "backtest_results.csv")

    return row


def command_to_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def read_reviewed_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


def resolve_input_path(
    value: str | None,
    repo_dir: Path,
    default_relative: Path,
    manifest_value: str | None = None,
) -> Path:
    """Resolve a user path, manifest path, or repo-relative default."""
    candidates = [value, manifest_value, str(default_relative)]
    for candidate in candidates:
        if not candidate:
            continue
        path = _path_from_value(candidate, repo_dir)
        if path.exists():
            return path.resolve()

        remapped = _remap_data_path_to_repo(path, repo_dir)
        if remapped.exists():
            return remapped.resolve()

    return (repo_dir / default_relative).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_dir = args.repo_dir.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    manifest = _read_json(run_root / "pit_masked_panel_manifest.json")

    data_file = resolve_input_path(
        args.data_file,
        repo_dir,
        DEFAULT_MARKET_CSV,
        manifest.get("market_csv"),
    )
    pit_universe_csv = resolve_input_path(
        args.pit_universe_csv,
        repo_dir,
        DEFAULT_PIT_UNIVERSE_CSV,
        manifest.get("pit_universe_csv"),
    )
    training_results_csv = (
        args.training_results_csv.expanduser().resolve()
        if args.training_results_csv
        else run_root / "summaries" / "training_results.csv"
    )
    reviewed_backtest_csv = (
        args.reviewed_backtest_csv.expanduser().resolve()
        if args.reviewed_backtest_csv
        else run_root / "summaries" / "backtest_results.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "summaries" / "pit_saved_prediction_cost_rank_gate"
    )

    options = BacktestOptions(
        repo_dir=repo_dir,
        python_executable=args.python_executable,
        backtest_script=args.backtest_script.expanduser().resolve()
        if args.backtest_script
        else repo_dir / "scripts" / "backtest_sp500_daily.py",
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
        backtest_suffix=args.backtest_suffix,
        spread_bps=args.spread,
        slippage_bps=args.slippage,
        min_rank_drop=args.min_rank_drop,
        top_k=args.top_k,
        label_t=args.label_t,
        num_tests=args.num_tests,
        adjustment_method=args.adjustment_method,
    )

    jobs = resolve_year_jobs(
        run_root=run_root,
        training_results_csv=training_results_csv,
        years=args.years,
        backtest_suffix=args.backtest_suffix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    command_lines = []
    for job in jobs:
        cmd = build_backtest_command(job, options)
        command_lines.append(command_to_text(cmd))
        print(f"\n[{job.year}] {command_lines[-1]}")
        row = run_backtest_job(job, options, output_dir, dry_run=args.dry_run)
        rows.append(row)
        if row["status"] == "FAILED" and args.fail_fast:
            break

    paths = write_summary_outputs(
        output_dir=output_dir,
        cost_rows=rows,
        reviewed_rows=read_reviewed_rows(reviewed_backtest_csv),
        command_lines=command_lines,
    )

    print("\nSaved summary outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")

    failed = [row for row in rows if row.get("status") == "FAILED"]
    return 1 if failed else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run saved PIT masked-panel yearly averaged_predictions through "
            "scripts/backtest_sp500_daily.py with transaction costs and rank-drop gate."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Saved run root, e.g. mounted Drive path ending in 20260514_043539.",
    )
    parser.add_argument(
        "--training-results-csv",
        type=Path,
        default=None,
        help="Optional training_results.csv. Defaults to RUN_ROOT/summaries/training_results.csv.",
    )
    parser.add_argument(
        "--reviewed-backtest-csv",
        type=Path,
        default=None,
        help="Optional existing no-cost/no-gate summary CSV for side-by-side comparison.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--backtest-script", type=Path, default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--pit-universe-csv", default=None)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-t", type=int, default=5)
    parser.add_argument("--spread", type=float, default=10.0, help="Bid-ask spread in bps.")
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps.")
    parser.add_argument(
        "--min-rank-drop",
        type=int,
        default=30,
        help="Normal rank-drop gate threshold used by paper-trade portfolio logic.",
    )
    parser.add_argument("--num-tests", type=int, default=1)
    parser.add_argument(
        "--adjustment-method",
        choices=["bhy", "bonferroni", "holm"],
        default="bhy",
    )
    parser.add_argument("--backtest-suffix", default=DEFAULT_BACKTEST_SUFFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _training_rows_by_year(training_results_csv: Path | None) -> dict[int, dict[str, Any]]:
    if training_results_csv is None or not training_results_csv.exists():
        return {}

    df = pd.read_csv(training_results_csv)
    rows: dict[int, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        year_value = row.get("year", row.get("test_year"))
        if pd.isna(year_value):
            continue
        status = str(row.get("status", "OK")).upper()
        if status != "OK":
            continue
        rows[int(year_value)] = row
    return rows


def _resolve_predictions_dir(row: dict[str, Any], run_root: Path, experiment_name: str) -> Path:
    for key in ("predictions_dir", "run_dir"):
        value = row.get(key)
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        remapped = _remap_saved_run_path(str(value), run_root)
        run_dir, predictions_dir = _split_run_and_predictions_dir(remapped)
        if predictions_dir.name == "averaged_predictions":
            return predictions_dir
        return run_dir / "averaged_predictions"

    discovered = _latest_run_dir(run_root / "training_runs" / experiment_name)
    if discovered is None:
        raise FileNotFoundError(
            f"Could not resolve saved predictions for {experiment_name} under {run_root}"
        )
    return discovered / "averaged_predictions"


def _split_run_and_predictions_dir(path: Path) -> tuple[Path, Path]:
    if path.name == "averaged_predictions":
        return path.parent, path
    return path, path / "averaged_predictions"


def _remap_saved_run_path(value: str, run_root: Path) -> Path:
    original = Path(value).expanduser()
    if original.exists():
        return original.resolve()

    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return run_root

    parts = [part for part in raw.split("/") if part]
    for marker in ("training_runs", run_root.name):
        if marker in parts:
            idx = parts.index(marker)
            suffix = parts[idx:] if marker == "training_runs" else parts[idx + 1 :]
            return run_root.joinpath(*suffix)

    if not Path(raw).is_absolute():
        return run_root / raw
    return original


def _latest_run_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = sorted(path for path in base.iterdir() if path.is_dir())
    return candidates[-1] if candidates else None


def _validate_job_inputs(job: BacktestJob, options: BacktestOptions) -> None:
    if not job.predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {job.predictions_dir}")
    if not any(job.predictions_dir.glob("*.csv")):
        raise FileNotFoundError(f"No prediction CSV files found in {job.predictions_dir}")
    if not options.data_file.is_file():
        raise FileNotFoundError(f"Market data file not found: {options.data_file}")
    if not options.pit_universe_csv.is_file():
        raise FileNotFoundError(f"PIT universe CSV not found: {options.pit_universe_csv}")
    if not options.backtest_script.is_file():
        raise FileNotFoundError(f"Backtest script not found: {options.backtest_script}")


def _build_side_by_side_df(cost_df: pd.DataFrame, reviewed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cost_df.empty:
        return pd.DataFrame(
            columns=["year", "metric", "reviewed_no_cost_no_gate", "cost_rank_gate", "delta"]
        )

    reviewed_by_year = _frame_by_year(reviewed_df)
    for _, cost_row in cost_df.iterrows():
        year = int(cost_row["year"])
        reviewed_row = reviewed_by_year.get(year, {})
        for metric in COMPARISON_METRICS:
            reviewed_value = _metric_value(reviewed_row, metric)
            cost_value = _metric_value(cost_row, metric)
            rows.append(
                {
                    "year": year,
                    "metric": metric,
                    "reviewed_no_cost_no_gate": reviewed_value,
                    "cost_rank_gate": cost_value,
                    "delta": _numeric_delta(cost_value, reviewed_value),
                }
            )

    return pd.DataFrame(rows)


def _build_combined_summary_df(cost_df: pd.DataFrame, reviewed_df: pd.DataFrame) -> pd.DataFrame:
    ok_cost = cost_df[cost_df.get("status", pd.Series(dtype=str)).eq("OK")]
    reviewed = reviewed_df.copy()

    row = {
        "years": ",".join(str(int(year)) for year in sorted(cost_df.get("year", []))),
        "cost_rank_gate_compounded_total_return": _compound_metric(ok_cost, "total_return"),
        "cost_rank_gate_compounded_benchmark_return": _compound_metric(ok_cost, "benchmark_return"),
        "cost_rank_gate_compounded_excess_return": _compound_metric(ok_cost, "excess_return"),
        "cost_rank_gate_avg_ARR": _mean_metric(ok_cost, "ARR"),
        "cost_rank_gate_avg_ASR": _mean_metric(ok_cost, "ASR"),
        "cost_rank_gate_worst_MDD": _min_metric(ok_cost, "MDD"),
        "cost_rank_gate_avg_daily_turnover": _mean_metric(ok_cost, "avg_daily_turnover"),
        "total_trading_days": _sum_metric(ok_cost, "num_trading_days"),
        "transaction_costs_enabled": True,
        "rank_gate_enabled": True,
        "reviewed_compounded_total_return": _compound_metric(reviewed, "total_return"),
        "reviewed_compounded_benchmark_return": _compound_metric(reviewed, "benchmark_return"),
        "reviewed_compounded_excess_return": _compound_metric(reviewed, "excess_return"),
        "reviewed_avg_ASR": _mean_metric(reviewed, "ASR"),
        "reviewed_worst_MDD": _min_metric(reviewed, "MDD"),
    }
    return pd.DataFrame([row])


def _build_summary_markdown(
    cost_df: pd.DataFrame,
    side_by_side_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    command_lines: list[str],
) -> str:
    lines = [
        "# PIT Saved-Prediction Cost + Rank-Gate Backtests",
        "",
        "This re-evaluation reuses saved averaged predictions; it does not retrain models.",
        "Scenario: transaction costs enabled and rank-drop gate enabled.",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"```text\n{command}\n```" for command in command_lines)
    lines.extend(
        [
            "",
            "## Yearly Cost-Aware Results",
            "",
            _markdown_table(cost_df) if len(cost_df) else "No yearly rows.",
            "",
            "## Side-By-Side Metrics",
            "",
            _markdown_table(side_by_side_df)
            if len(side_by_side_df)
            else "No reviewed baseline rows were available.",
            "",
            "## Combined 2022-2025 Summary",
            "",
            _markdown_table(combined_df) if len(combined_df) else "No combined rows.",
            "",
        ]
    )
    return "\n".join(lines)


def _frame_by_year(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if df.empty or "year" not in df.columns:
        return {}
    return {int(row["year"]): row for row in df.to_dict("records") if not pd.isna(row["year"])}


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a compact GitHub-style table without pandas optional dependencies."""
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _metric_value(row: pd.Series | dict[str, Any], metric: str) -> Any:
    if isinstance(row, pd.Series):
        return row.get(f"backtest.{metric}", row.get(metric, np.nan))
    return row.get(f"backtest.{metric}", row.get(metric, np.nan))


def _numeric_delta(cost_value: Any, reviewed_value: Any) -> float:
    cost_num = pd.to_numeric(cost_value, errors="coerce")
    reviewed_num = pd.to_numeric(reviewed_value, errors="coerce")
    if pd.isna(cost_num) or pd.isna(reviewed_num):
        return np.nan
    return round(float(cost_num) - float(reviewed_num), 12)


def _metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    col = f"backtest.{metric}" if f"backtest.{metric}" in df.columns else metric
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _compound_metric(df: pd.DataFrame, metric: str) -> float:
    series = _metric_series(df, metric)
    if len(series) == 0:
        return np.nan
    return round(float(np.prod(1.0 + series.to_numpy(dtype=float)) - 1.0), 12)


def _mean_metric(df: pd.DataFrame, metric: str) -> float:
    series = _metric_series(df, metric)
    return float(series.mean()) if len(series) else np.nan


def _min_metric(df: pd.DataFrame, metric: str) -> float:
    series = _metric_series(df, metric)
    return float(series.min()) if len(series) else np.nan


def _sum_metric(df: pd.DataFrame, metric: str) -> int:
    series = _metric_series(df, metric)
    return int(series.sum()) if len(series) else 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _path_from_value(value: str, repo_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_dir / path


def _remap_data_path_to_repo(path: Path, repo_dir: Path) -> Path:
    raw = str(path).replace("\\", "/")
    marker = "/data/"
    if marker not in raw:
        return path
    suffix = raw.split(marker, maxsplit=1)[1]
    return repo_dir / "data" / suffix


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
