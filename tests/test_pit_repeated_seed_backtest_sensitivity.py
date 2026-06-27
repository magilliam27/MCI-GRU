from pathlib import Path

import pandas as pd

from scripts.run_pit_repeated_seed_backtest_sensitivity import (
    BacktestSensitivityOptions,
    build_backtest_command,
    build_sensitivity_jobs,
    default_scenarios,
    resolve_training_rows,
    write_sensitivity_outputs,
)


def test_default_scenarios_include_spread_only_and_label21_diagnostic() -> None:
    scenarios = default_scenarios(training_label_t=5, include_label21_diagnostic=True)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "current_tc10_slip5_label5",
        "spread5_only_label5",
        "spread5_only_label21_diagnostic",
    ]

    spread_only = scenarios[1]
    assert spread_only.spread_bps == 5.0
    assert spread_only.slippage_bps == 0.0
    assert spread_only.label_t == 5
    assert spread_only.training_label_matched

    diagnostic = scenarios[2]
    assert diagnostic.spread_bps == 5.0
    assert diagnostic.slippage_bps == 0.0
    assert diagnostic.label_t == 21
    assert not diagnostic.training_label_matched
    assert "diagnostic" in diagnostic.interpretation_note.lower()


def test_repeated_seed_training_rows_are_not_collapsed_by_year(tmp_path: Path) -> None:
    run_root = tmp_path / "20260520_183538"
    first_run = run_root / "training_runs" / "pit_seed_314159_replication_2022" / "20260520_183617"
    second_run = run_root / "training_runs" / "pit_seed_271828_replication_2022" / "20260520_183618"
    (first_run / "averaged_predictions").mkdir(parents=True)
    (second_run / "averaged_predictions").mkdir(parents=True)

    training_results_csv = run_root / "summaries" / "training_results.csv"
    training_results_csv.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "OK",
                "name": "pit_seed_314159_replication_2022",
                "base_seed": 314159,
                "year": 2022,
                "run_dir": (
                    "/content/mci_gru_work/pit_repeated_seed_replication/20260520_183538/"
                    "training_runs/pit_seed_314159_replication_2022/20260520_183617"
                ),
                "predictions_dir": (
                    "/content/mci_gru_work/pit_repeated_seed_replication/20260520_183538/"
                    "training_runs/pit_seed_314159_replication_2022/"
                    "20260520_183617/averaged_predictions"
                ),
                "test_start": "2022-01-22",
                "test_end": "2022-12-31",
            },
            {
                "status": "REUSED",
                "name": "pit_seed_271828_replication_2022",
                "base_seed": 271828,
                "year": 2022,
                "run_dir": str(second_run),
                "predictions_dir": str(second_run / "averaged_predictions"),
                "test_start": "2022-01-22",
                "test_end": "2022-12-31",
            },
        ]
    ).to_csv(training_results_csv, index=False)

    rows = resolve_training_rows(
        run_root=run_root,
        training_results_csv=training_results_csv,
        years=[2022],
        base_seeds=None,
    )
    jobs = build_sensitivity_jobs(rows, default_scenarios(training_label_t=5)[:2])

    assert {(row.base_seed, row.year) for row in rows} == {(314159, 2022), (271828, 2022)}
    rows_by_seed = {row.base_seed: row for row in rows}
    assert rows_by_seed[314159].run_dir == first_run
    assert rows_by_seed[314159].predictions_dir == first_run / "averaged_predictions"
    assert len(jobs) == 4
    assert {job.scenario.scenario_id for job in jobs} == {
        "current_tc10_slip5_label5",
        "spread5_only_label5",
    }
    assert all(job.backtest_dir.name.startswith("backtest_pit_daily_") for job in jobs)


def test_build_backtest_command_uses_scenario_cost_and_label_settings(tmp_path: Path) -> None:
    scenario = default_scenarios(training_label_t=5)[2]
    row = resolve_training_rows(
        run_root=tmp_path,
        training_results_csv=_write_one_training_row(tmp_path),
        years=[2022],
        base_seeds=[314159],
    )[0]
    job = build_sensitivity_jobs([row], [scenario])[0]
    options = BacktestSensitivityOptions(
        repo_dir=Path("/repo"),
        python_executable="python",
        backtest_script=Path("/repo/tests/backtest_sp500_daily.py"),
        data_file=Path("/data/market.csv"),
        pit_universe_csv=Path("/data/pit.csv"),
        output_dir=tmp_path / "out",
        num_tests=4,
        adjustment_method="bhy",
    )

    cmd = build_backtest_command(job, options)

    assert cmd[cmd.index("--label_t") + 1] == "21"
    assert cmd[cmd.index("--spread") + 1] == "5"
    assert cmd[cmd.index("--slippage") + 1] == "0"
    assert cmd[cmd.index("--backtest_suffix") + 1] == scenario.backtest_suffix
    assert "--transaction_costs" in cmd
    assert "--enable_rank_drop_gate" in cmd
    assert all("run_experiment.py" not in part for part in cmd)


def test_write_sensitivity_outputs_emits_cross_tabs_and_baseline_deltas(tmp_path: Path) -> None:
    output_dir = tmp_path / "sensitivity"
    rows = [
        {
            "scenario_id": "current_tc10_slip5_label5",
            "scenario.training_label_matched": True,
            "status": "OK",
            "base_seed": 314159,
            "year": 2022,
            "backtest.total_return": 0.08,
            "backtest.excess_return": 0.03,
            "backtest.ARR": 0.09,
            "backtest.ASR": 0.8,
            "backtest.MDD": -0.2,
            "backtest.avg_daily_turnover": 0.4,
        },
        {
            "scenario_id": "spread5_only_label5",
            "scenario.training_label_matched": True,
            "status": "OK",
            "base_seed": 314159,
            "year": 2022,
            "backtest.total_return": 0.11,
            "backtest.excess_return": 0.06,
            "backtest.ARR": 0.12,
            "backtest.ASR": 1.1,
            "backtest.MDD": -0.18,
            "backtest.avg_daily_turnover": 0.4,
        },
    ]

    paths = write_sensitivity_outputs(
        output_dir=output_dir,
        rows=rows,
        daily_returns_rows=[],
        command_lines=["python tests/backtest_sp500_daily.py ..."],
        baseline_scenario_id="current_tc10_slip5_label5",
    )

    scenario_year = pd.read_csv(paths["scenario_year_crosstab_csv"])
    deltas = pd.read_csv(paths["metric_deltas_csv"])
    summary_md = paths["summary_md"].read_text(encoding="utf-8")

    spread_year = scenario_year[
        (scenario_year["scenario_id"] == "spread5_only_label5")
        & (scenario_year["year"] == 2022)
    ].iloc[0]
    assert spread_year["backtest.total_return.mean"] == 0.11

    total_return_delta = deltas[
        (deltas["scenario_id"] == "spread5_only_label5")
        & (deltas["metric"] == "backtest.total_return")
    ].iloc[0]
    assert total_return_delta["delta_vs_baseline"] == 0.03
    assert "saved averaged predictions" in summary_md
    assert "label_t=21" in summary_md


def _write_one_training_row(tmp_path: Path) -> Path:
    run_dir = tmp_path / "training_runs" / "pit_seed_314159_replication_2022" / "20260520_183617"
    (run_dir / "averaged_predictions").mkdir(parents=True)
    path = tmp_path / "summaries" / "training_results.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "OK",
                "name": "pit_seed_314159_replication_2022",
                "base_seed": 314159,
                "year": 2022,
                "run_dir": str(run_dir),
                "predictions_dir": str(run_dir / "averaged_predictions"),
                "test_start": "2022-01-22",
                "test_end": "2022-12-31",
            }
        ]
    ).to_csv(path, index=False)
    return path
