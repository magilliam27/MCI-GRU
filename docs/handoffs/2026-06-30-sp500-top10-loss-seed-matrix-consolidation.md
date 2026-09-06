# S&P 500 GICS Top-10 Loss/Seed Matrix Consolidation

## Status

Started consolidation of the 110-name PIT top-10-per-GICS-sector loss comparison run tagged `20260629_011839`.

The training side is complete and locally consolidated. The transaction-cost backtest side is not yet a complete apples-to-apples matrix because the final resume pass overwrote `summaries/backtest_rows.json`.

## Drive Sources

- Run root: <https://drive.google.com/drive/folders/1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo>
- Heartbeat: <https://drive.google.com/file/d/1vymg2xDriJ6MEUPB4EM8KfHp28BD1X-J/view?usp=drivesdk>
- Summaries: <https://drive.google.com/drive/folders/18o3CdNWKndiphJQGQyCVieyWuKFkV_1E>
- Logs: <https://drive.google.com/drive/folders/1qv_Wy_G-AQLRxnkEjDApCkhTHlTn3eo3>
- Resume notebook: <https://drive.google.com/file/d/1uRIERr_OFE8HC79FV6GHhR1isZ3AjFIt/view?usp=drivesdk>

## Local Bundle

Raw and derived artifacts are under:

`artifacts/2026-06-30-sp500-top10-loss-seed-matrix-consolidation/`

Files saved locally:

- `training_rows.json`
- `backtest_rows_current.json`
- `reused_backtest_rows.json`
- `known_existing_rows.json`
- `run_summary.json`
- `training_rows_flat.csv`
- `training_summary_by_loss_year.csv`
- `backtest_rows_flat_current_plus_reused.csv`
- `backtest_summary_current_plus_reused_by_loss_year.csv`
- `coverage_matrix.csv`

## Confirmed Run State

Final `run_summary.json` contains:

- `status: OK`
- `run_tag: 20260629_011839`
- `training_rows: 28`
- `backtest_rows: 6`
- `reused_backtest_rows: 4`
- `known_existing_rows: 8`
- `data_audit.gpu_name: NVIDIA L4`
- `snapshot_min_selected: 110`
- `snapshot_max_selected: 110`
- `pit_min_scoreable_stocks: 100`

## Training Coverage

All 28 newly trained rows have `status: OK`.

| Loss | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| pure IC | seeds `161803`, `271828` | seeds `161803`, `271828` | seeds `161803`, `271828` | seeds `161803`, `271828` |
| LambdaRankIC | seeds `161803`, `271828` | seeds `161803`, `271828` | seeds `161803`, `271828` | seeds `161803`, `271828` |
| Portfolio-IC weight50 | seeds `314159`, `271828`, `161803` | seeds `314159`, `271828`, `161803` | seeds `314159`, `271828`, `161803` | seeds `314159`, `271828`, `161803` |

Known existing rows recorded by the run:

- pure IC seed `1729`, years 2022-2024 from `sp500_gics_top10_baseline_multiyear/20260623_011810`; these were marked as needing backtest replay.
- pure IC seed `1729`, year 2025 from `sp500_gics_top10_baseline/20260622_043728`; older 2018-start data family, counted per no-repeat instruction.
- LambdaRankIC seed `314159`, years 2022-2024 from `sp500_gics_top10_lambdarank_ic_full/20260626_172316`.
- LambdaRankIC seed `314159`, year 2025 from `sp500_gics_top10_lambdarank_ic_2025/20260627_012647`.

## Training Summary

These are newly trained rows only. Existing seed `314159` LambdaRankIC rows and existing pure-IC seed `1729` rows are not pooled here.

| Loss | Year | Seeds | Mean val IC | Mean val Rank IC | Test avg IC | Test avg Rank IC | Test cumulative return |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| LambdaRankIC | 2022 | `161803 271828` | 0.019629 | 0.031051 | -0.031592 | -0.047048 | -0.747488 |
| LambdaRankIC | 2023 | `161803 271828` | 0.001393 | 0.001487 | 0.057800 | 0.045862 | 0.648663 |
| LambdaRankIC | 2024 | `161803 271828` | 0.029441 | 0.030681 | 0.017613 | 0.018812 | 0.700858 |
| LambdaRankIC | 2025 | `161803 271828` | 0.015363 | 0.021246 | 0.015114 | 0.027613 | 0.826334 |
| Portfolio-IC weight50 | 2022 | `161803 271828 314159` | 0.036471 | 0.025151 | -0.059361 | -0.043129 | -0.653484 |
| Portfolio-IC weight50 | 2023 | `161803 271828 314159` | 0.002156 | 0.001592 | 0.033118 | 0.032250 | 0.486477 |
| Portfolio-IC weight50 | 2024 | `161803 271828 314159` | 0.044379 | 0.034519 | 0.043982 | 0.035120 | 0.764705 |
| Portfolio-IC weight50 | 2025 | `161803 271828 314159` | 0.039164 | 0.029217 | 0.040436 | 0.041816 | 0.908933 |
| pure IC | 2022 | `161803 271828` | 0.032563 | 0.029614 | -0.060105 | -0.046535 | -0.651350 |
| pure IC | 2023 | `161803 271828` | 0.002223 | 0.000124 | 0.040920 | 0.040852 | 0.583839 |
| pure IC | 2024 | `161803 271828` | 0.037785 | 0.031869 | 0.033394 | 0.030973 | 0.741091 |
| pure IC | 2025 | `161803 271828` | 0.032912 | 0.030523 | 0.031769 | 0.039064 | 0.954807 |

## Backtest Coverage Caveat

Current final files expose only:

| Loss | Year | Backtest seeds currently visible | Mean net total return | Mean ASR | Mean MDD |
| --- | ---: | --- | ---: | ---: | ---: |
| LambdaRankIC | 2025 | `161803 271828` | 0.271936 | 1.210772 | -0.181113 |
| Portfolio-IC weight50 | 2025 | `161803 271828 314159` | 0.223467 | 0.816599 | -0.254295 |
| pure IC | 2023 | `1729` | 0.129688 | 0.806597 | -0.128814 |
| pure IC | 2025 | `161803` | 0.268919 | 0.942542 | -0.247047 |

This is not the full backtest matrix. Drive revision history for `backtest_rows.json` (`1b3-neuNES3JRPgCbXpmdniUmVro1OPvR`) shows that the sidecar grew to `41306` bytes at `2026-06-30T00:19:39.947Z`, then restarted at `1824` bytes at `2026-06-30T03:42:05.468Z` during the pinned-resume phase, and ended at `11324` bytes at `2026-06-30T10:40:30.570Z`.

Attempted revision fetch for revision `0Bwr_sXAFm255VVdROWVMOVAzYytzWGR3bFF5KzZOZWkxTGtZPQ` failed through the connector with `GoogleDriveInvalidRequestError: No supported mimetype returned for revision`.

## Next Action

Do not treat LambdaRankIC vs pure IC vs Portfolio-IC as fully decided from the current backtest files.

Practical next step:

1. Recover the `2026-06-30T00:19:39.947Z` revision of `backtest_rows.json` through Drive UI/API if possible.
2. If revision media remains inaccessible, rebuild only the missing saved-prediction backtests from existing run folders; do not retrain.
3. Recreate `backtest_summary_by_loss_year_seed.csv` and then promote a concise research note only after the transaction-cost/rank-gate backtest matrix is complete.

## Commands Run

```powershell
rg -n "LambdaRankIC|lambdarank|sp500_top10|loss_seed_matrix|top10|110-name|110 names|Portfolio-IC|pure IC" docs scripts notebooks AGENTS.md -g "*.md" -g "*.py" -g "*.ipynb"
```

```powershell
Invoke-WebRequest -Uri <Drive signed raw URL> -OutFile artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation\<file>.json -UseBasicParsing
```

```powershell
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe - <parser>
```
