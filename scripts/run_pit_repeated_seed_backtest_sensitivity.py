"""Run repeated-seed PIT saved predictions through backtest sensitivity scenarios.

This is replay-only. It reads an existing repeated-seed run root, reuses saved
``averaged_predictions`` folders, and invokes ``tests/backtest_sp500_daily.py``.
It never calls training code.
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
    2022: {"test_start": "2022-01-22", "test_end": "2022-12-31"},
    2023: {"test_start": "2023-01-22", "test_end": "2023-12-31"},
    2024: {"test_start": "2024-01-22", "test_end": "2024-12-31"},
    2025: {"test_start": "2025-01-22", "test_end": "2025-12-31"},
}

DEFAULT_MARKET_CSV = Path("data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv")
DEFAULT_PIT_UNIVERSE_CSV = Path(
    "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
)
DEFAULT_BASELINE_SCENARIO_ID = "current_tc10_slip5_label5"
SENSITIVITY_METRICS = [
    "backtest.total_return",
    "backtest.benchmark_return",
    "backtest.excess_return",
    "backtest.ARR",
    "backtest.ASR",
    "backtest.MDD",
    "backtest.avg_daily_turnover",
    "backtest.num_trading_days",
]


@dataclass(frozen=True)
class SensitivityScenario:
    scenario_id: str
    description: str
    backtest_suffix: str
    spread_bps: float
    slippage_bps: float
    label_t: int
    training_label_matched: bool
    interpretation_note: str
    transaction_costs_enabled: bool = True
    rank_gate_enabled: bool = True
    min_rank_drop: int = 30
    top_k: int = 10


@dataclass(frozen=True)
class TrainingRunRow:
    name: str
    base_seed: int
    year: int
    run_dir: Path
    predictions_dir: Path
    test_start: str
    test_end: str
    status: str


@dataclass(frozen=True)
class SensitivityBacktestJob:
    training_row: TrainingRunRow
    scenario: SensitivityScenario
    backtest_dir: Path


@dataclass(frozen=True)
class BacktestSensitivityOptions:
    repo_dir: Path
    python_executable: str
    backtest_script: Path
    data_file: Path
    pit_universe_csv: Path
    output_dir: Path
    num_tests: int = 4
    adjustment_method: str = "bhy"


def default_scenarios(
    training_label_t: int = 5,
    include_label21_diagnostic: bool = True,
) -> list[SensitivityScenario]:
    """Return the standard Option A sensitivity replay matrix."""
    label = int(training_label_t)
    scenarios = [
        SensitivityScenario(
            scenario_id=f"current_tc10_slip5_label{label}",
            description="Current promoted cost/rank-gate replay: 10 bps spread plus 5 bps slippage.",
            backtest_suffix=f"_pit_daily_sensitivity_tc10_slip5_rank_gate_label{label}",
            spread_bps=10.0,
            slippage_bps=5.0,
            label_t=label,
            training_label_matched=True,
            interpretation_note=(
                "Training-matched baseline. The saved repeated-seed models were trained "
                f"with model.label_t={label}."
            ),
        ),
        SensitivityScenario(
            scenario_id=f"spread5_only_label{label}",
            description="Spread-only replay: 5 bps spread and zero slippage, rank gate unchanged.",
            backtest_suffix=f"_pit_daily_sensitivity_spread5_only_rank_gate_label{label}",
            spread_bps=5.0,
            slippage_bps=0.0,
            label_t=label,
            training_label_matched=True,
            interpretation_note=(
                "Training-matched lower-cost replay. This changes transaction-cost "
                "assumptions without changing the saved signal horizon."
            ),
        ),
    ]
    if include_label21_diagnostic:
        scenarios.append(
            SensitivityScenario(
                scenario_id="spread5_only_label21_diagnostic",
                description=(
                    "Horizon diagnostic: 5 bps spread, zero slippage, label_t=21, "
                    "rank gate unchanged."
                ),
                backtest_suffix="_pit_daily_sensitivity_spread5_only_rank_gate_label21",
                spread_bps=5.0,
                slippage_bps=0.0,
                label_t=21,
                training_label_matched=(label == 21),
                interpretation_note=(
                    "Diagnostic only unless model.label_t was 21. For the Option A run, "
                    "saved predictions were produced by 5-day trained models; this "
                    "changes prediction-vs-forward-return evaluation fields, while "
                    "daily P&L remains open-to-open in tests/backtest_sp500_daily.py."
                ),
            )
        )
    return scenarios


def resolve_training_rows(
    run_root: Path,
    training_results_csv: Path,
    years: list[int] | None,
    base_seeds: list[int] | None,
) -> list[TrainingRunRow]:
    """Read repeated-seed training rows without collapsing multiple seeds per year."""
    run_root = run_root.expanduser().resolve()
    training_results_csv = training_results_csv.expanduser().resolve()
    df = pd.read_csv(training_results_csv)
    rows: list[TrainingRunRow] = []
    year_filter = set(years or [])
    seed_filter = set(base_seeds or [])

    for raw in df.to_dict("records"):
        status = str(raw.get("status", "OK")).upper()
        if status not in {"OK", "REUSED"}:
            continue

        year = int(raw.get("year", raw.get("test_year")))
        base_seed = int(raw["base_seed"])
        if year_filter and year not in year_filter:
            continue
        if seed_filter and base_seed not in seed_filter:
            continue
        if year not in PIT_WINDOWS and (_is_blank(raw.get("test_start")) or _is_blank(raw.get("test_end"))):
            raise ValueError(f"No default PIT test window is known for year {year}.")

        run_dir, predictions_dir = _resolve_run_and_predictions_dir(raw, run_root)
        rows.append(
            TrainingRunRow(
                name=str(raw.get("name") or f"pit_seed_{base_seed}_replication_{year}"),
                base_seed=base_seed,
                year=year,
                run_dir=run_dir,
                predictions_dir=predictions_dir,
                test_start=str(raw.get("test_start") or PIT_WINDOWS[year]["test_start"]),
                test_end=str(raw.get("test_end") or PIT_WINDOWS[year]["test_end"]),
                status=status,
            )
        )

    rows.sort(key=lambda row: (row.base_seed, row.year, row.name))
    return rows


def build_sensitivity_jobs(
    training_rows: list[TrainingRunRow],
    scenarios: list[SensitivityScenario],
) -> list[SensitivityBacktestJob]:
    jobs: list[SensitivityBacktestJob] = []
    for row in training_rows:
        for scenario in scenarios:
            jobs.append(
                SensitivityBacktestJob(
                    training_row=row,
                    scenario=scenario,
                    backtest_dir=row.run_dir / f"backtest{scenario.backtest_suffix}",
                )
            )
    return jobs


def build_backtest_command(
    job: SensitivityBacktestJob,
    options: BacktestSensitivityOptions,
) -> list[str]:
    row = job.training_row
    scenario = job.scenario
    cmd = [
        options.python_executable,
        "-X",
        "utf8",
        str(options.backtest_script),
        "--predictions_dir",
        str(row.predictions_dir),
        "--data_file",
        str(options.data_file),
        "--pit_universe_csv",
        str(options.pit_universe_csv),
        "--test_start",
        row.test_start,
        "--test_end",
        row.test_end,
        "--top_k",
        str(scenario.top_k),
        "--label_t",
        str(scenario.label_t),
        "--num_tests",
        str(options.num_tests),
        "--adjustment_method",
        options.adjustment_method,
        "--auto_save",
        "--backtest_suffix",
        scenario.backtest_suffix,
    ]
    if scenario.transaction_costs_enabled:
        cmd.extend(
            [
                "--transaction_costs",
                "--spread",
                _format_number(scenario.spread_bps),
                "--slippage",
                _format_number(scenario.slippage_bps),
            ]
        )
    if scenario.rank_gate_enabled:
        cmd.extend(["--enable_rank_drop_gate", "--min_rank_drop", str(scenario.min_rank_drop)])
    return cmd


def run_sensitivity_job(
    job: SensitivityBacktestJob,
    options: BacktestSensitivityOptions,
    dry_run: bool = False,
) -> dict[str, Any]:
    cmd = build_backtest_command(job, options)
    command_line = command_to_text(cmd)
    row = job.training_row
    scenario = job.scenario
    logs_dir = options.output_dir / "logs" / scenario.scenario_id / row.name
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "backtest_stdout.log"
    stderr_path = logs_dir / "backtest_stderr.log"

    result: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "scenario.description": scenario.description,
        "scenario.spread_bps": scenario.spread_bps,
        "scenario.slippage_bps": scenario.slippage_bps,
        "scenario.label_t": scenario.label_t,
        "scenario.training_label_matched": scenario.training_label_matched,
        "scenario.transaction_costs_enabled": scenario.transaction_costs_enabled,
        "scenario.rank_gate_enabled": scenario.rank_gate_enabled,
        "scenario.min_rank_drop": scenario.min_rank_drop,
        "scenario.top_k": scenario.top_k,
        "scenario.interpretation_note": scenario.interpretation_note,
        "status": "DRY_RUN" if dry_run else "PENDING",
        "returncode": np.nan if dry_run else None,
        "name": row.name,
        "base_seed": row.base_seed,
        "year": row.year,
        "run_dir": str(row.run_dir),
        "predictions_dir": str(row.predictions_dir),
        "backtest_dir": str(job.backtest_dir),
        "test_start": row.test_start,
        "test_end": row.test_end,
        "command": command_line,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    if dry_run:
        stdout_path.write_text(command_line + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return result

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
    result["status"] = "OK" if proc.returncode == 0 else "FAILED"
    result["returncode"] = proc.returncode

    result_csv = job.backtest_dir / "backtest_results.csv"
    if result_csv.exists():
        result_df = pd.read_csv(result_csv)
        if len(result_df):
            result.update({f"backtest.{key}": value for key, value in result_df.iloc[0].to_dict().items()})
            yearly_copy_dir = options.output_dir / "yearly" / scenario.scenario_id / row.name
            yearly_copy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_csv, yearly_copy_dir / "backtest_results.csv")

    return result


def load_daily_returns(job: SensitivityBacktestJob) -> list[dict[str, Any]]:
    daily_path = job.backtest_dir / "daily_returns.csv"
    if not daily_path.exists():
        return []
    df = pd.read_csv(daily_path)
    if df.empty:
        return []
    row = job.training_row
    scenario = job.scenario
    df.insert(0, "scenario_id", scenario.scenario_id)
    df.insert(1, "base_seed", row.base_seed)
    df.insert(2, "year", row.year)
    df.insert(3, "name", row.name)
    df["scenario.label_t"] = scenario.label_t
    df["scenario.training_label_matched"] = scenario.training_label_matched
    return df.to_dict("records")


def write_sensitivity_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    daily_returns_rows: list[dict[str, Any]],
    command_lines: list[str],
    baseline_scenario_id: str = DEFAULT_BASELINE_SCENARIO_ID,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "pit_repeated_seed_backtest_sensitivity_results.csv"
    daily_returns_csv = output_dir / "pit_repeated_seed_backtest_sensitivity_daily_returns.csv"
    scenario_year_crosstab_csv = output_dir / "pit_repeated_seed_backtest_sensitivity_year_crosstab.csv"
    seed_crosstab_csv = output_dir / "pit_repeated_seed_backtest_sensitivity_seed_crosstab.csv"
    metric_deltas_csv = output_dir / "pit_repeated_seed_backtest_sensitivity_metric_deltas.csv"
    summary_md = output_dir / "pit_repeated_seed_backtest_sensitivity_summary.md"

    results_df = pd.DataFrame(rows)
    daily_df = pd.DataFrame(daily_returns_rows)
    scenario_year_df = _build_group_summary(results_df, ["scenario_id", "year"])
    seed_df = _build_group_summary(results_df, ["scenario_id", "base_seed"])
    deltas_df = _build_metric_deltas(results_df, baseline_scenario_id)

    results_df.to_csv(results_csv, index=False)
    daily_df.to_csv(daily_returns_csv, index=False)
    scenario_year_df.to_csv(scenario_year_crosstab_csv, index=False)
    seed_df.to_csv(seed_crosstab_csv, index=False)
    deltas_df.to_csv(metric_deltas_csv, index=False)
    summary_md.write_text(
        _build_summary_markdown(
            results_df=results_df,
            scenario_year_df=scenario_year_df,
            seed_df=seed_df,
            deltas_df=deltas_df,
            command_lines=command_lines,
            baseline_scenario_id=baseline_scenario_id,
        ),
        encoding="utf-8",
    )

    return {
        "results_csv": results_csv,
        "daily_returns_csv": daily_returns_csv,
        "scenario_year_crosstab_csv": scenario_year_crosstab_csv,
        "seed_crosstab_csv": seed_crosstab_csv,
        "metric_deltas_csv": metric_deltas_csv,
        "summary_md": summary_md,
    }


def command_to_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_dir = args.repo_dir.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    manifest = _read_manifest(run_root)
    training_label_t = args.training_label_t or _manifest_training_label_t(manifest) or 5
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "summaries" / "pit_repeated_seed_backtest_sensitivity"
    )

    scenarios = default_scenarios(
        training_label_t=training_label_t,
        include_label21_diagnostic=not args.skip_label21_diagnostic,
    )
    if args.scenarios:
        requested = set(args.scenarios)
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in requested]
        missing = requested - {scenario.scenario_id for scenario in scenarios}
        if missing:
            raise ValueError(f"Unknown sensitivity scenario id(s): {sorted(missing)}")

    training_results_csv = (
        args.training_results_csv.expanduser().resolve()
        if args.training_results_csv
        else run_root / "summaries" / "training_results.csv"
    )
    data_file = resolve_input_path(
        args.data_file,
        repo_dir,
        DEFAULT_MARKET_CSV,
        _manifest_value(manifest, "market_csv"),
    )
    pit_universe_csv = resolve_input_path(
        args.pit_universe_csv,
        repo_dir,
        DEFAULT_PIT_UNIVERSE_CSV,
        _manifest_value(manifest, "pit_universe_csv"),
    )
    options = BacktestSensitivityOptions(
        repo_dir=repo_dir,
        python_executable=args.python_executable,
        backtest_script=args.backtest_script.expanduser().resolve()
        if args.backtest_script
        else repo_dir / "tests" / "backtest_sp500_daily.py",
        data_file=data_file,
        pit_universe_csv=pit_universe_csv,
        output_dir=output_dir,
        num_tests=args.num_tests,
        adjustment_method=args.adjustment_method,
    )

    training_rows = resolve_training_rows(
        run_root=run_root,
        training_results_csv=training_results_csv,
        years=args.years,
        base_seeds=args.base_seeds,
    )
    jobs = build_sensitivity_jobs(training_rows, scenarios)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    command_lines: list[str] = []
    for job in jobs:
        cmd = build_backtest_command(job, options)
        command_line = command_to_text(cmd)
        command_lines.append(command_line)
        print(f"\n[{job.scenario.scenario_id}] {job.training_row.name}")
        print(command_line)
        result = run_sensitivity_job(job, options, dry_run=args.dry_run)
        rows.append(result)
        if result["status"] == "OK":
            daily_rows.extend(load_daily_returns(job))
        if result["status"] == "FAILED" and args.fail_fast:
            break

    paths = write_sensitivity_outputs(
        output_dir=output_dir,
        rows=rows,
        daily_returns_rows=daily_rows,
        command_lines=command_lines,
        baseline_scenario_id=f"current_tc10_slip5_label{training_label_t}",
    )

    print("\nSaved sensitivity outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")

    failed = [row for row in rows if row.get("status") == "FAILED"]
    return 1 if failed else 0


def resolve_input_path(
    value: str | None,
    repo_dir: Path,
    default_relative: Path,
    manifest_value: str | None = None,
) -> Path:
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay repeated-seed PIT saved predictions through cost and label-horizon "
            "backtest sensitivity scenarios."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-results-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--backtest-script", type=Path, default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--pit-universe-csv", default=None)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--base-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-label-t", type=int, default=None)
    parser.add_argument("--skip-label21-diagnostic", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--num-tests", type=int, default=4)
    parser.add_argument(
        "--adjustment-method",
        choices=["bhy", "bonferroni", "holm"],
        default="bhy",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _resolve_run_and_predictions_dir(raw: dict[str, Any], run_root: Path) -> tuple[Path, Path]:
    for key in ("predictions_dir", "run_dir"):
        value = raw.get(key)
        if _is_blank(value):
            continue
        remapped = _remap_saved_run_path(str(value), run_root)
        run_dir, predictions_dir = _split_run_and_predictions_dir(remapped)
        if predictions_dir.name == "averaged_predictions":
            return run_dir, predictions_dir
        return run_dir, run_dir / "averaged_predictions"

    name = raw.get("name")
    if _is_blank(name):
        raise ValueError("Training row has no run_dir, predictions_dir, or name.")
    candidates = sorted((run_root / "training_runs" / str(name)).glob("*"))
    candidates = [path for path in candidates if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find training run directory for {name!r} under {run_root}")
    run_dir = candidates[-1]
    return run_dir, run_dir / "averaged_predictions"


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
    if run_root.name in parts:
        idx = parts.index(run_root.name)
        return run_root.joinpath(*parts[idx + 1 :])
    if "training_runs" in parts:
        idx = parts.index("training_runs")
        return run_root.joinpath(*parts[idx:])
    if not Path(raw).is_absolute():
        return run_root / raw
    return original


def _validate_job_inputs(
    job: SensitivityBacktestJob,
    options: BacktestSensitivityOptions,
) -> None:
    if not job.training_row.predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {job.training_row.predictions_dir}")
    if not any(job.training_row.predictions_dir.glob("*.csv")):
        raise FileNotFoundError(f"No prediction CSV files found in {job.training_row.predictions_dir}")
    if not options.data_file.is_file():
        raise FileNotFoundError(f"Market data file not found: {options.data_file}")
    if not options.pit_universe_csv.is_file():
        raise FileNotFoundError(f"PIT universe CSV not found: {options.pit_universe_csv}")
    if not options.backtest_script.is_file():
        raise FileNotFoundError(f"Backtest script not found: {options.backtest_script}")


def _build_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "completed_backtests"])
    ok_df = df[df.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("OK")].copy()
    if ok_df.empty:
        return pd.DataFrame(columns=[*group_cols, "completed_backtests"])

    grouped = ok_df.groupby(group_cols, dropna=False)
    summary = grouped.size().reset_index(name="completed_backtests")
    for metric in SENSITIVITY_METRICS:
        if metric not in ok_df.columns:
            continue
        metric_summary = (
            grouped[metric]
            .apply(lambda series: pd.to_numeric(series, errors="coerce"))
            .groupby(level=group_cols)
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        metric_summary = metric_summary.rename(
            columns={
                "mean": f"{metric}.mean",
                "std": f"{metric}.std",
                "min": f"{metric}.min",
                "max": f"{metric}.max",
            }
        )
        summary = summary.merge(metric_summary, on=group_cols, how="left")
    return summary


def _build_metric_deltas(df: pd.DataFrame, baseline_scenario_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["scenario_id", "base_seed", "year", "metric", "value", "baseline_value", "delta_vs_baseline"]
        )
    ok_df = df[df.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("OK")].copy()
    if ok_df.empty:
        return pd.DataFrame(
            columns=["scenario_id", "base_seed", "year", "metric", "value", "baseline_value", "delta_vs_baseline"]
        )

    baseline = ok_df[ok_df["scenario_id"].astype(str).eq(baseline_scenario_id)]
    baseline_by_pair = {
        (int(row["base_seed"]), int(row["year"])): row
        for row in baseline.to_dict("records")
        if not pd.isna(row.get("base_seed")) and not pd.isna(row.get("year"))
    }

    rows: list[dict[str, Any]] = []
    for raw in ok_df.to_dict("records"):
        scenario_id = str(raw.get("scenario_id"))
        if scenario_id == baseline_scenario_id:
            continue
        pair = (int(raw["base_seed"]), int(raw["year"]))
        baseline_row = baseline_by_pair.get(pair)
        if baseline_row is None:
            continue
        for metric in SENSITIVITY_METRICS:
            value = pd.to_numeric(raw.get(metric), errors="coerce")
            baseline_value = pd.to_numeric(baseline_row.get(metric), errors="coerce")
            if pd.isna(value) or pd.isna(baseline_value):
                continue
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "base_seed": pair[0],
                    "year": pair[1],
                    "metric": metric,
                    "value": float(value),
                    "baseline_value": float(baseline_value),
                    "delta_vs_baseline": round(float(value - baseline_value), 12),
                }
            )
    return pd.DataFrame(rows)


def _build_summary_markdown(
    results_df: pd.DataFrame,
    scenario_year_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    command_lines: list[str],
    baseline_scenario_id: str,
) -> str:
    lines = [
        "# PIT Repeated-Seed Backtest Sensitivity",
        "",
        "This replay reuses saved averaged predictions; it does not retrain models.",
        "The default matrix compares the current 10 bps spread + 5 bps slippage setup ",
        "against a 5 bps spread-only setup with zero slippage.",
        "",
        "The label_t=21 row is a diagnostic when the saved model trained with label_t=5. ",
        "In tests/backtest_sp500_daily.py, label_t changes the prediction-vs-forward-return ",
        "evaluation fields; the portfolio P&L path remains daily open-to-open.",
        "",
        f"Baseline scenario for deltas: `{baseline_scenario_id}`.",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"```text\n{command}\n```" for command in command_lines)
    lines.extend(
        [
            "",
            "## Scenario x Year",
            "",
            _markdown_table(scenario_year_df) if not scenario_year_df.empty else "No scenario-year rows.",
            "",
            "## Scenario x Seed",
            "",
            _markdown_table(seed_df) if not seed_df.empty else "No scenario-seed rows.",
            "",
            "## Deltas vs Baseline",
            "",
            _markdown_table(deltas_df) if not deltas_df.empty else "No baseline-matched delta rows.",
            "",
            "## Raw Rows",
            "",
            _markdown_table(results_df) if not results_df.empty else "No raw rows.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame) -> str:
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


def _read_manifest(run_root: Path) -> dict[str, Any]:
    for name in ("pit_repeated_seed_replication_manifest.json", "pit_masked_panel_manifest.json"):
        manifest = _read_json(run_root / name)
        if manifest:
            return manifest
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_training_label_t(manifest: dict[str, Any]) -> int | None:
    for path in (("model_recipe", "label_t"), ("backtest", "label_t")):
        value = _nested_manifest_value(manifest, path)
        if value is not None:
            return int(value)
    return None


def _manifest_value(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    return str(value) if value is not None else None


def _nested_manifest_value(manifest: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = manifest
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


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


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
