from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_pit_saved_prediction_backtests import (
    BacktestJob,
    BacktestOptions,
    _build_combined_summary_df,
    build_backtest_command,
    resolve_pit_export_cutoff,
    resolve_year_jobs,
    write_summary_outputs,
)


def test_build_backtest_command_enables_costs_and_rank_gate() -> None:
    job = BacktestJob(
        year=2024,
        run_dir=Path("/runs/pit_true_rolling_2024/20260514_120000"),
        predictions_dir=Path("/runs/pit_true_rolling_2024/20260514_120000/averaged_predictions"),
        backtest_dir=Path("/runs/pit_true_rolling_2024/20260514_120000/backtest_tc_gate"),
        test_start="2024-01-22",
        test_end="2024-12-31",
    )
    options = BacktestOptions(
        repo_dir=Path("/repo"),
        python_executable="python",
        backtest_script=Path("/repo/scripts/backtest_sp500_daily.py"),
        data_file=Path("/data/sp500_pit_union.csv"),
        pit_universe_csv=Path("/data/sp500_pit_universe.csv"),
        backtest_suffix="_tc_gate",
        spread_bps=12.5,
        slippage_bps=2.5,
        min_rank_drop=30,
        top_k=10,
        label_t=5,
        num_tests=1,
        adjustment_method="bhy",
    )

    cmd = build_backtest_command(job, options)

    assert cmd[:3] == ["python", "-X", "utf8"]
    assert Path(cmd[3]).name == "backtest_sp500_daily.py"
    assert cmd[cmd.index("--predictions_dir") + 1] == str(job.predictions_dir)
    assert Path(cmd[cmd.index("--data_file") + 1]).name == "sp500_pit_union.csv"
    assert Path(cmd[cmd.index("--pit_universe_csv") + 1]).name == "sp500_pit_universe.csv"
    assert cmd[cmd.index("--test_start") + 1] == "2024-01-22"
    assert cmd[cmd.index("--test_end") + 1] == "2024-12-31"
    assert cmd[cmd.index("--backtest_suffix") + 1] == "_tc_gate"
    assert cmd[cmd.index("--spread") + 1] == "12.5"
    assert cmd[cmd.index("--slippage") + 1] == "2.5"
    assert cmd[cmd.index("--min_rank_drop") + 1] == "30"
    assert "--transaction_costs" in cmd
    assert "--enable_rank_drop_gate" in cmd
    assert all("run_experiment.py" not in part for part in cmd)


def test_resolve_year_jobs_remaps_stale_colab_paths_to_current_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "20260514_043539"
    run_dir = run_root / "training_runs" / "pit_true_rolling_2022" / "20260514_120000"
    predictions_dir = run_dir / "averaged_predictions"
    predictions_dir.mkdir(parents=True)

    summaries_dir = run_root / "summaries"
    summaries_dir.mkdir(parents=True)
    training_results_csv = summaries_dir / "training_results.csv"
    pd.DataFrame(
        [
            {
                "year": 2022,
                "status": "OK",
                "run_dir": (
                    "/content/MCI-GRU/pit_masked_panel_2022_2025/"
                    "20260514_043539/training_runs/pit_true_rolling_2022/20260514_120000"
                ),
                "test_start": "2022-01-22",
                "test_end": "2022-12-31",
            }
        ]
    ).to_csv(training_results_csv, index=False)

    jobs = resolve_year_jobs(
        run_root=run_root,
        training_results_csv=training_results_csv,
        years=[2022],
        backtest_suffix="_tc_gate",
    )

    assert len(jobs) == 1
    assert jobs[0].year == 2022
    assert jobs[0].run_dir == run_dir
    assert jobs[0].predictions_dir == predictions_dir
    assert jobs[0].backtest_dir == run_dir / "backtest_tc_gate"


def test_write_summary_outputs_compares_cost_aware_rows_to_reviewed_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "summaries" / "pit_saved_prediction_cost_rank_gate"
    cost_rows = [
        {
            "year": 2022,
            "status": "OK",
            "returncode": 0,
            "backtest.total_return": 0.12,
            "backtest.benchmark_return": 0.05,
            "backtest.excess_return": 0.07,
            "backtest.ARR": 0.13,
            "backtest.ASR": 1.1,
            "backtest.MDD": -0.2,
            "backtest.avg_daily_turnover": 0.35,
            "backtest.num_trading_days": 237,
            "backtest.transaction_costs_enabled": True,
            "backtest.rank_gate_enabled": True,
            "backtest.min_rank_drop": 30,
            "backtest.days_with_gate_exits": 12,
            "backtest.days_skipped_by_rank_gate": 0,
        }
    ]
    reviewed_rows = [
        {
            "year": 2022,
            "backtest.total_return": 0.18,
            "backtest.benchmark_return": 0.05,
            "backtest.excess_return": 0.13,
            "backtest.ARR": 0.19,
            "backtest.ASR": 1.4,
            "backtest.MDD": -0.18,
            "backtest.num_trading_days": 237,
            "backtest.transaction_costs_enabled": False,
            "backtest.rank_gate_enabled": False,
        }
    ]

    paths = write_summary_outputs(
        output_dir=output_dir,
        cost_rows=cost_rows,
        reviewed_rows=reviewed_rows,
        command_lines=["python scripts/run_pit_saved_prediction_backtests.py --run-root RUN"],
    )

    yearly_df = pd.read_csv(paths["yearly_csv"])
    side_by_side_df = pd.read_csv(paths["side_by_side_csv"])
    combined_df = pd.read_csv(paths["combined_summary_csv"])
    summary_md = paths["summary_md"].read_text(encoding="utf-8")

    assert yearly_df.loc[0, "backtest.transaction_costs_enabled"]
    assert yearly_df.loc[0, "backtest.rank_gate_enabled"]
    assert yearly_df.loc[0, "backtest.min_rank_drop"] == 30

    total_return = side_by_side_df[
        (side_by_side_df["year"] == 2022) & (side_by_side_df["metric"] == "total_return")
    ].iloc[0]
    assert total_return["reviewed_no_cost_no_gate"] == 0.18
    assert total_return["cost_rank_gate"] == 0.12
    assert total_return["delta"] == -0.06

    # No year carries its daily series, so nothing is combined from yearly summaries (#342).
    assert np.isnan(combined_df.loc[0, "cost_rank_gate_total_return"])
    assert combined_df.loc[0, "cost_rank_gate_combined_from"] == "MISSING_DAILY_RETURNS:2022"
    assert np.isnan(combined_df.loc[0, "reviewed_total_return"])
    assert combined_df.loc[0, "cost_rank_gate_avg_daily_turnover"] == pytest.approx(0.35)
    assert combined_df.loc[0, "total_trading_days"] == 237
    assert "transaction costs enabled" in summary_md
    assert "rank-drop gate enabled" in summary_md


def _year_with_daily_returns(tmp_path: Path, year: int, portfolio, benchmark) -> dict:
    dates = pd.bdate_range(f"{year}-01-03", periods=len(portfolio))
    path = tmp_path / f"{year}_daily_returns.csv"
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "portfolio_return": portfolio,
            "benchmark_return": benchmark,
            "excess_return": np.asarray(portfolio) - np.asarray(benchmark),
        }
    ).to_csv(path, index=False)
    values = np.cumprod(1 + np.asarray(portfolio))
    total = values[-1] - 1
    bench = np.prod(1 + np.asarray(benchmark)) - 1
    return {
        "year": year,
        "status": "OK",
        "daily_returns_csv": str(path),
        "backtest.total_return": total,
        "backtest.benchmark_return": bench,
        "backtest.excess_return": total - bench,
        "backtest.MDD": float(np.min(values / np.maximum.accumulate(values) - 1)),
        "backtest.avg_daily_turnover": 0.2 if year == 2022 else 0.4,
        "backtest.num_trading_days": len(portfolio),
    }


def test_multi_year_combine_reads_the_concatenated_daily_series(tmp_path: Path) -> None:
    """r7: a drawdown straddling the year end; yearly combine gave -0.148 against -0.274."""
    p1 = np.r_[np.full(200, 0.002), np.full(40, -0.004)]
    p2 = np.r_[np.full(40, -0.004), np.full(200, 0.0005)]
    b1, b2 = np.full(240, 0.0012), np.full(240, -0.0008)
    rows = [
        _year_with_daily_returns(tmp_path, 2022, p1, b1),
        _year_with_daily_returns(tmp_path, 2023, p2, b2),
    ]
    combined = _build_combined_summary_df(pd.DataFrame(rows), pd.DataFrame()).iloc[0]

    portfolio, benchmark = np.r_[p1, p2], np.r_[b1, b2]
    total = np.prod(1 + portfolio) - 1
    values = np.cumprod(1 + portfolio)
    true_mdd = float(np.min(values / np.maximum.accumulate(values) - 1))
    arr = (1 + total) ** (252 / portfolio.size) - 1

    assert combined["cost_rank_gate_combined_from"] == "CONCATENATED_DAILY_RETURNS"
    assert combined["cost_rank_gate_total_return"] == pytest.approx(total)
    assert combined["cost_rank_gate_excess_return"] == pytest.approx(
        total - (np.prod(1 + benchmark) - 1)
    )
    assert combined["cost_rank_gate_MDD"] == pytest.approx(true_mdd)
    assert true_mdd < min(row["backtest.MDD"] for row in rows) - 0.1  # straddles the year end
    assert combined["cost_rank_gate_ARR"] == pytest.approx(arr)
    assert combined["cost_rank_gate_ASR"] == pytest.approx(
        arr / (np.std(portfolio, ddof=1) * np.sqrt(252))
    )
    assert combined["cost_rank_gate_avg_daily_turnover"] == pytest.approx(0.3)


def test_multi_year_benchmark_covers_the_days_the_simulator_skipped(tmp_path: Path) -> None:
    """A skipped day is missing from daily_returns.csv but the benchmark still moved."""
    p1, p2 = np.full(200, 0.001), np.full(200, 0.001)
    rows = [
        _year_with_daily_returns(tmp_path, 2022, p1, np.full(200, 0.0005)),
        _year_with_daily_returns(tmp_path, 2023, p2, np.full(200, 0.0005)),
    ]
    # The engine's calendar benchmark for 2023 includes ten skipped days at -1% each.
    calendar_2023 = np.r_[np.full(200, 0.0005), np.full(10, -0.01)]
    rows[1]["backtest.benchmark_return"] = np.prod(1 + calendar_2023) - 1
    combined = _build_combined_summary_df(pd.DataFrame(rows), pd.DataFrame()).iloc[0]

    total = np.prod(1 + np.r_[p1, p2]) - 1
    bench = (1 + rows[0]["backtest.benchmark_return"]) * (1 + rows[1]["backtest.benchmark_return"])
    assert combined["cost_rank_gate_benchmark_return"] == pytest.approx(bench - 1)
    assert combined["cost_rank_gate_excess_return"] == pytest.approx(total - (bench - 1))


def test_replay_command_carries_the_pit_export_cutoff(tmp_path: Path) -> None:
    """#339's replay refuses open-ended PIT rows without the export's cutoff."""
    pit = tmp_path / "sp500_pit_x_20160101_20260513_pit_universe.csv"
    pit.write_text("kdcode,valid_from,valid_to\nAAA,2016-01-01,\n", encoding="utf-8")
    (tmp_path / "sp500_pit_x_20160101_20260513_meta.json").write_text(
        '{"end": "2026-05-13"}', encoding="utf-8"
    )
    assert resolve_pit_export_cutoff(None, None, pit) == "2026-05-13"  # the export's sidecar
    assert resolve_pit_export_cutoff(None, "2026-04-30", pit) == "2026-04-30"  # run manifest
    assert resolve_pit_export_cutoff("2026-03-31", "2026-04-30", pit) == "2026-03-31"  # flag
    assert resolve_pit_export_cutoff(None, None, tmp_path / "other.csv") is None

    job = BacktestJob(
        year=2024,
        run_dir=tmp_path,
        predictions_dir=tmp_path / "averaged_predictions",
        backtest_dir=tmp_path / "backtest",
        test_start="2024-01-22",
        test_end="2024-12-31",
    )
    options = BacktestOptions(
        repo_dir=tmp_path,
        python_executable="python",
        backtest_script=tmp_path / "backtest_sp500_daily.py",
        data_file=tmp_path / "market.csv",
        pit_universe_csv=pit,
        pit_export_cutoff="2026-05-13",
    )
    cmd = build_backtest_command(job, options)
    assert cmd[cmd.index("--pit_export_cutoff") + 1] == "2026-05-13"
    no_cutoff = build_backtest_command(
        job, BacktestOptions(**{**options.__dict__, "pit_export_cutoff": None})
    )
    assert "--pit_export_cutoff" not in no_cutoff
