from pathlib import Path

import pandas as pd

from scripts.run_pit_saved_prediction_backtests import (
    BacktestJob,
    BacktestOptions,
    build_backtest_command,
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

    assert combined_df.loc[0, "cost_rank_gate_compounded_total_return"] == 0.12
    assert combined_df.loc[0, "reviewed_compounded_total_return"] == 0.18
    assert combined_df.loc[0, "total_trading_days"] == 237
    assert "transaction costs enabled" in summary_md
    assert "rank-drop gate enabled" in summary_md
