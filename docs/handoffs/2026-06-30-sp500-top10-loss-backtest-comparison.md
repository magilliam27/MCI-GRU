# 110-name PIT top-10 loss backtest comparison

Date: 2026-06-30

Scope: compare available cost-aware, rank-drop-gated daily backtests for the
110-name PIT GICS top-10-per-sector universe across pure IC, LambdaRankIC, and
Portfolio-IC weight50. This is a recovery/reporting pass only. No retraining was
run, and no GitHub or Drive files were mutated.

## Source artifacts

- Drive run root: https://drive.google.com/drive/folders/1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo
- Drive summaries folder: https://drive.google.com/drive/folders/18o3CdNWKndiphJQGQyCVieyWuKFkV_1E
- Drive logs folder: https://drive.google.com/drive/folders/1qv_Wy_G-AQLRxnkEjDApCkhTHlTn3eo3
- Heartbeat: https://drive.google.com/file/d/1vymg2xDriJ6MEUPB4EM8KfHp28BD1X-J/view
- Local consolidation bundle:
  `C:\Users\magil\.codex\worktrees\559c\MCI-GRU\artifacts\2026-06-30-sp500-top10-loss-seed-matrix-consolidation`
- `backtest_rows.json` Drive file ID: `1b3-neuNES3JRPgCbXpmdniUmVro1OPvR`

Run/data audit from `run_summary.json`:

| Field | Value |
| --- | --- |
| run_tag | `20260629_011839` |
| final status | `OK` |
| GPU | `NVIDIA L4` |
| training rows | 28 |
| final current backtest rows | 6 |
| final reused backtest rows | 4 |
| known existing rows | 8 |
| snapshot selected names | min 110, max 110 |
| PIT min scoreable stocks | 100 |
| market CSV | `/content/MCI-GRU/data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.csv` |
| PIT CSV | `/content/MCI-GRU/data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv` |

Backtest assumptions visible in the logs and current rows:

| Assumption | Value |
| --- | --- |
| portfolio | top 10 stocks |
| execution/returns | open-to-open, entry T+1 open, exit T+2 open |
| label horizon | 5 days |
| transaction costs | enabled |
| spread | 10 bps round trip |
| slippage | 5 bps per trade |
| rank-drop gate | enabled |
| min rank drop | 30 |

## Recovery status

The final `summaries/backtest_rows.json` is incomplete because it was overwritten
during the pinned resume process.

Revision history confirms:

| Revision time UTC | Size | Interpretation |
| --- | ---: | --- |
| 2026-06-29T02:12:55.347Z | 1,827 | early first row state |
| 2026-06-30T00:19:39.947Z | 41,306 | pre-overwrite larger backtest table |
| 2026-06-30T03:42:05.468Z | 1,824 | reset during pinned resume |
| 2026-06-30T10:40:30.570Z | 11,324 | final current file, 6 rows |

Attempted revision fetch for the 41,306-byte revision
`0Bwr_sXAFm255VVdROWVMOVAzYytzWGR3bFF5KzZOZWkxTGtZPQ` failed through the Drive
connector with:

```text
GoogleDriveInvalidRequestError: No supported mimetype returned for revision
```

Drive search found no standalone uploaded `backtest_metrics.json` or
`backtest_results.csv` files. Individual backtest logs do exist in the logs
folder for the newly trained rows. This report parses the required 2022 stress
year from logs and uses the final structured current-plus-reused CSV for the
visible 2025/current rows.

## Coverage

Training coverage is complete for the newly trained matrix:

| Loss | Years | Base seeds | Training rows |
| --- | --- | --- | ---: |
| pure IC | 2022-2025 | 161803, 271828 | 8 |
| LambdaRankIC | 2022-2025 | 161803, 271828 | 8 |
| Portfolio-IC weight50 | 2022-2025 | 314159, 271828, 161803 | 12 |

Backtest metric coverage available for this report:

| Loss | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| pure IC | 2 newly trained rows parsed from logs | reused seed 1729 structured; newly trained logs exist but not parsed | newly trained logs exist but not parsed | seed 161803 structured; seed 271828 log exists but not parsed |
| LambdaRankIC | 2 newly trained rows parsed from logs | newly trained logs exist but not parsed | newly trained logs exist but not parsed | seeds 161803, 271828 structured |
| Portfolio-IC weight50 | 3 newly trained rows parsed from logs | newly trained logs exist but not parsed | newly trained logs exist but not parsed | seeds 314159, 271828, 161803 structured |

Known existing rows:

| Loss | Seed | Years | Status in final consolidation |
| --- | ---: | --- | --- |
| pure IC | 1729 | 2022, 2024, 2025 | missing predictions |
| pure IC | 1729 | 2023 | replayed OK |
| LambdaRankIC | 314159 | 2022-2025 | training rows known, no visible replay rows in final current-plus-reused table |

## 2022 stress-year recovered rows

These rows were recovered from Drive backtest logs. They are cost-aware,
rank-drop-gated, top-10, open-to-open backtests.

| Loss | Seed | Net total return | Net ARR | ASR | MDD | Benchmark return | Excess return | Trades | Total cost | Avg daily turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure IC | 161803 | -0.3901 | -0.3999 | -1.0013 | -0.4142 | -0.1127 | -0.2774 | 14 | 0.0014 | 0.0029 |
| pure IC | 271828 | -0.4209 | -0.4312 | -1.0676 | -0.4421 | -0.1127 | -0.3082 | 10 | 0.0010 | 0.0020 |
| LambdaRankIC | 161803 | -0.2196 | -0.2259 | -0.8026 | -0.3213 | -0.1127 | -0.1069 | 16 | 0.0016 | 0.0033 |
| LambdaRankIC | 271828 | -0.2339 | -0.2406 | -0.8137 | -0.3018 | -0.1127 | -0.1212 | 16 | 0.0016 | 0.0033 |
| Portfolio-IC weight50 | 314159 | -0.4209 | -0.4312 | -1.0676 | -0.4421 | -0.1127 | -0.3082 | 10 | 0.0010 | 0.0020 |
| Portfolio-IC weight50 | 271828 | -0.4209 | -0.4312 | -1.0676 | -0.4421 | -0.1127 | -0.3082 | 10 | 0.0010 | 0.0020 |
| Portfolio-IC weight50 | 161803 | -0.4209 | -0.4312 | -1.0676 | -0.4421 | -0.1127 | -0.3082 | 10 | 0.0010 | 0.0020 |

2022 aggregate:

| Loss | Rows | Mean net total return | Mean ASR | Mean MDD | Mean excess return | Mean trades | Mean total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure IC | 2 | -0.4055 | -1.0345 | -0.4282 | -0.2928 | 12.0 | 0.0012 |
| LambdaRankIC | 2 | -0.2268 | -0.8082 | -0.3116 | -0.1141 | 16.0 | 0.0016 |
| Portfolio-IC weight50 | 3 | -0.4209 | -1.0676 | -0.4421 | -0.3082 | 10.0 | 0.0010 |

Interpretation: LambdaRankIC is the best of the three objectives in the 2022
stress year on loss, drawdown, and Sharpe, but it is still strongly negative in
both base seeds. Portfolio-IC weight50 does not help in 2022; all three visible
seeds land on the same net backtest path as the weaker pure-IC seed.

## Structured current-plus-reused rows

These rows come from
`backtest_rows_flat_current_plus_reused.csv` in the local consolidation bundle.

| Source | Loss | Year | Seed | Status | Net total return | Net ARR | ASR | MDD | Benchmark | Excess | Trades | Total cost | Gate-exit days |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | pure IC | 2025 | 161803 | OK | 0.268919 | 0.278866 | 0.942542 | -0.247047 | 0.157545 | 0.111373 | 20 | 0.0020 | 4 |
| current | LambdaRankIC | 2025 | 271828 | OK | 0.221025 | 0.229045 | 1.029498 | -0.171413 | 0.157545 | 0.063479 | 144 | 0.0144 | 58 |
| current | LambdaRankIC | 2025 | 161803 | OK | 0.322848 | 0.335038 | 1.392045 | -0.190813 | 0.157545 | 0.165302 | 70 | 0.0070 | 22 |
| current | Portfolio-IC weight50 | 2025 | 314159 | OK | 0.213988 | 0.221730 | 0.812889 | -0.237499 | 0.157545 | 0.056442 | 16 | 0.0016 | 3 |
| current | Portfolio-IC weight50 | 2025 | 271828 | OK | 0.158974 | 0.164593 | 0.610470 | -0.252654 | 0.157545 | 0.001428 | 24 | 0.0024 | 6 |
| current | Portfolio-IC weight50 | 2025 | 161803 | OK | 0.297439 | 0.308564 | 1.026438 | -0.272733 | 0.157545 | 0.139894 | 16 | 0.0016 | 2 |
| reused | pure IC | 2023 | 1729 | OK | 0.129688 | 0.134213 | 0.806597 | -0.128814 | 0.118499 | 0.011189 | 22 | 0.0022 | 5 |

Rows present as missing in the final reused table:

| Source | Loss | Year | Seed | Status |
| --- | --- | ---: | ---: | --- |
| reused | pure IC | 2022 | 1729 | MISSING_PREDICTIONS |
| reused | pure IC | 2024 | 1729 | MISSING_PREDICTIONS |
| reused | pure IC | 2025 | 1729 | MISSING_PREDICTIONS |

## 2025 structured comparison

The 2025 structured table is incomplete for pure IC because only seed `161803`
survived in the final current rows. LambdaRankIC has two newly trained seeds and
Portfolio-IC weight50 has three newly trained seeds.

| Loss | Rows | Seeds | Mean net total return | Mean ASR | Mean MDD | Mean trades | Mean total cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| pure IC | 1 | 161803 | 0.268919 | 0.942542 | -0.247047 | 20.0 | 0.0020 |
| LambdaRankIC | 2 | 161803, 271828 | 0.271936 | 1.210772 | -0.181113 | 107.0 | 0.0107 |
| Portfolio-IC weight50 | 3 | 314159, 271828, 161803 | 0.223467 | 0.816599 | -0.254295 | 18.7 | 0.001867 |

Interpretation: 2025 is favorable to LambdaRankIC on mean ASR and drawdown among
the visible structured rows, but that comes with much higher turnover and cost:
about 107 trades and 1.07 percent cumulative transaction cost versus roughly 20
trades/0.20 percent for the visible pure-IC row and 18.7 trades/0.187 percent
for Portfolio-IC weight50.

## Training/evaluation context

The complete training summary is not a substitute for backtest metrics, but it
does show that the newly trained matrix finished across all years:

| Loss | 2022 mean test cumulative return | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: |
| pure IC | -0.651350 | 0.583839 | 0.741091 | 0.954807 |
| LambdaRankIC | -0.747488 | 0.648663 | 0.700858 | 0.826334 |
| Portfolio-IC weight50 | -0.653484 | 0.486477 | 0.764705 | 0.908933 |

This table should be treated as model evaluation context only. The decision
surface requested here is the daily backtest, and that remains incomplete after
the `backtest_rows.json` overwrite.

## Missing backtest results

Rows still missing from a structured machine-readable backtest table:

| Loss | Year | Seeds |
| --- | ---: | --- |
| pure IC | 2023 | 161803, 271828 |
| pure IC | 2024 | 161803, 271828 |
| pure IC | 2025 | 271828 |
| LambdaRankIC | 2023 | 161803, 271828 |
| LambdaRankIC | 2024 | 161803, 271828 |
| Portfolio-IC weight50 | 2023 | 314159, 271828, 161803 |
| Portfolio-IC weight50 | 2024 | 314159, 271828, 161803 |

Also missing for known-existing reused baselines:

| Loss | Year | Seed | Reason |
| --- | ---: | ---: | --- |
| pure IC | 2022 | 1729 | missing predictions |
| pure IC | 2024 | 1729 | missing predictions |
| pure IC | 2025 | 1729 | missing predictions |
| LambdaRankIC | 2022-2025 | 314159 | known training rows, no visible replay rows in final structured table |

The missing newly trained rows likely remain recoverable from Drive backtest logs
without retraining. The better durable repair is to either recover the 41,306-byte
`backtest_rows.json` revision outside the current connector limitation or parse
all Drive backtest logs into a new CSV. A saved-prediction replay should be a
fallback only if the logs prove insufficient; it should still not retrain.

## Recommendation

Keep pure IC as the launch/default objective.

Evidence is not complete enough to promote LambdaRankIC. The recovered 2022
stress-year backtests are genuinely interesting: LambdaRankIC loses much less
than pure IC and Portfolio-IC weight50 in both visible 2022 seeds, with better
drawdown and ASR. The visible 2025 structured rows also favor LambdaRankIC on
ASR and drawdown. But the evidence is incomplete across 2023 and 2024 in the
structured table, pure IC is missing one 2025 newly trained seed, and
LambdaRankIC's visible 2025 edge comes with materially higher trades and
transaction cost.

Treat LambdaRankIC as experimental, with priority as a hybrid or second-stage
ranker candidate rather than a default replacement. Portfolio-IC weight50 remains
non-default: it has lower turnover/cost in 2025, but weaker 2025 ASR than
LambdaRankIC and no protection in the recovered 2022 stress-year rows.

