# LambdaRankIC Rank-Drop Sensitivity Replay Check

Date: 2026-07-01

Task: Item 3, replay LambdaRankIC saved predictions only at alternative
rank-drop gate thresholds for the 110-name PIT GICS top-10-per-sector universe.

## Result

No new replay rows were produced in this workspace. The replay is blocked by
missing local artifacts, not by code support.

Local blockers:

- No `averaged_predictions/` directories were found under
  `C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU`.
- The expected top-10 PIT market and universe CSVs are not present locally:
  - `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv`
  - `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv`
- The generic runner defaults are also absent:
  - `data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv`
  - `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
- The June 30 consolidation bundle in
  `C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation`
  is present, but it contains only summary JSON/CSV files. It has no saved
  prediction CSVs and cannot support a 20/50/75 threshold replay.

## Existing Threshold-30 Rows Reviewed

These are existing rows from
`docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md`,
not newly replayed rows.

| Year | Seed | Min rank drop | Net total return | ASR | MDD | Trades | Cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 161803 | 30 | 54.10% | 3.14 | -9.31% | 116 | 1.16% |
| 2023 | 271828 | 30 | 33.59% | 2.25 | -7.03% | 24 | 0.24% |
| 2024 | 161803 | 30 | 40.22% | 1.99 | -18.36% | 120 | 1.20% |
| 2024 | 271828 | 30 | 15.88% | 1.06 | -8.95% | 362 | 3.62% |
| 2025 | 161803 | 30 | 32.28% | 1.39 | -19.08% | 70 | 0.70% |
| 2025 | 271828 | 30 | 22.10% | 1.03 | -17.14% | 144 | 1.44% |

The 2024 seed `271828` row remains the key churn stress case: it paid 3.62%
cumulative transaction cost and used 362 trades while barely exceeding the
benchmark.

## Script Fit

- `tests/backtest_sp500_daily.py` exposes the gate directly:
  `--enable_rank_drop_gate --min_rank_drop N`.
- `scripts/run_pit_saved_prediction_backtests.py` is training-free and builds
  the expected cost-aware daily backtest command, but it resolves one row per
  year from `training_results.csv`. That is suitable for a single-variant yearly
  PIT run root, but not for the 28-row loss/seed matrix unless the
  `training_results.csv` is filtered first.
- For this task, direct `tests/backtest_sp500_daily.py` calls are the faithful
  seed-specific replay path.
- `scripts/run_saved_prediction_capacity_replay.py` can sweep multiple
  `--min-rank-drop` values, but it is an evidence-harness capacity replay, not
  the exact recovered daily backtest surface used in the June 30 report.

## Minimal Replay Commands

After restaging the run root and top-10 PIT CSVs locally, run seed-specific
commands like this. This does not retrain.

```powershell
$py = ".\.venv\Scripts\python.exe"
$market = "data\raw\market\sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv"
$pit = "data\raw\constituents\sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"

$jobs = @(
  @{
    year = 2024
    seed = 271828
    start = "2024-01-22"
    end = "2024-12-31"
    predictions = "<RESTAGED_RUN_ROOT>\training\lambdarank_ic\2024\seed271828\top10_lambdarank_ic_2024_seed271828\20260629_165356\averaged_predictions"
  },
  @{
    year = 2024
    seed = 161803
    start = "2024-01-22"
    end = "2024-12-31"
    predictions = "<RESTAGED_RUN_ROOT>\training\lambdarank_ic\2024\seed161803\top10_lambdarank_ic_2024_seed161803\20260629_181302\averaged_predictions"
  },
  @{
    year = 2025
    seed = 271828
    start = "2025-01-22"
    end = "2025-12-31"
    predictions = "<RESTAGED_RUN_ROOT>\training\lambdarank_ic\2025\seed271828\top10_lambdarank_ic_2025_seed271828\20260630_034205\averaged_predictions"
  },
  @{
    year = 2025
    seed = 161803
    start = "2025-01-22"
    end = "2025-12-31"
    predictions = "<RESTAGED_RUN_ROOT>\training\lambdarank_ic\2025\seed161803\top10_lambdarank_ic_2025_seed161803\20260630_045613\averaged_predictions"
  }
)

foreach ($job in $jobs) {
  foreach ($rankDrop in 20, 30, 50, 75) {
    & $py -X utf8 tests\backtest_sp500_daily.py `
      --predictions_dir $job.predictions `
      --data_file $market `
      --pit_universe_csv $pit `
      --test_start $job.start `
      --test_end $job.end `
      --top_k 10 `
      --label_t 5 `
      --num_tests 1 `
      --adjustment_method bhy `
      --auto_save `
      --backtest_suffix "_top10_tc_rankdrop$rankDrop" `
      --transaction_costs `
      --spread 10 `
      --slippage 5 `
      --enable_rank_drop_gate `
      --min_rank_drop $rankDrop
  }
}
```

If 2023 is cheap after 2024/2025, add:

```powershell
@{
  year = 2023
  seed = 271828
  start = "2023-01-22"
  end = "2023-12-31"
  predictions = "<RESTAGED_RUN_ROOT>\training\lambdarank_ic\2023\seed271828\top10_lambdarank_ic_2023_seed271828\20260629_113840\averaged_predictions"
},
@{
  year = 2023
  seed = 161803
  start = "2023-01-22"
  end = "2023-12-31"
  predictions = "<RESTAGED_RUN_ROOT>\training\lambdarank_ic\2023\seed161803\top10_lambdarank_ic_2023_seed161803\20260629_122641\averaged_predictions"
}
```

Original Colab run root recorded in the consolidation bundle:

```text
/content/drive/MyDrive/MCI-GRU-Ablations/sp500_gics_top10_loss_comparison_repeated_seeds/20260629_011839
```

## Commands Run In This Workspace

```powershell
git status --short
rg -n "LambdaRankIC|rank-drop|rank_drop|saved prediction|saved-prediction|sp500 top10|top10|271828|161803" C:\Users\magil\.codex\memories\MEMORY.md
rg -n "rank_drop|min_rank_drop|LambdaRankIC|lambdarank|saved.*prediction|prediction.*backtest|backtest.*prediction|271828|161803" .
Get-Content -Raw docs\handoffs\2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md
Get-Content -Raw scripts\run_pit_saved_prediction_backtests.py
Get-Content -Raw tests\backtest_sp500_daily.py
Get-ChildItem -Recurse -Directory -Filter averaged_predictions -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -File -Include backtest_results.csv,training_results.csv,all_years_results.csv,backtest_rows.json,pit_masked_panel_manifest.json -ErrorAction SilentlyContinue
Test-Path data\raw\market\sp500_pit_union_lseg_20150101_20260513.csv; Test-Path data\raw\constituents\sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
Get-ChildItem data\raw\market,data\raw\constituents -Force
Get-ChildItem C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation -Force
Get-Content -Raw C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation\backtest_rows_flat_current_plus_reused.csv
Get-Content -Raw scripts\run_saved_prediction_capacity_replay.py
Get-Content -Raw tests\test_pit_saved_prediction_backtests.py
```

## Recommendation

Do not promote LambdaRankIC to default on the current evidence.

Stronger hysteresis is a plausible mitigation, but not yet proven locally. The
right next gate is empirical: restage the saved predictions plus the two top-10
PIT CSVs, replay 2024 seed `271828` at `20/30/50/75`, then confirm whether a
stronger gate materially reduces trades and cost without giving up the 2023/2025
LambdaRankIC alpha. A promotion argument needs the `50` or `75` gate to tame the
2024 churn case while keeping net return and ASR competitive with pure IC and
Portfolio-IC weight50.
