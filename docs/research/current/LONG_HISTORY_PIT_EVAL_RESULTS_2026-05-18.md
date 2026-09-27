# Long-History PIT Evaluation Results

Date: 2026-05-18
Issue: #23, "Create and evaluate long-history MCI-GRU presets"

## Provenance

These results are reconstructed from the user-provided Colab output after the
ephemeral Colab runtime deleted the local run directory before it was synced to
Google Drive. Treat this document as a saved result table and interpretation
note, not as a replacement for raw artifacts such as `training_results.csv`,
`backtest_results.csv`, checkpoint folders, logs, or equity-curve images.

Run tag reported by the notebook output: `20260518_015259`.

Evaluation setup:

- True PIT masked-panel evaluation.
- Years: 2022, 2023, 2024, 2025.
- History windows: `his_t=10`, `21`, `63`, `126`.
- Portfolio: top-10 stocks.
- Returns: open-to-open, entry at T+1 open and exit at T+2 open.
- Benchmark: equal-weighted open-to-open over the same window.
- Transaction costs: disabled.
- Rank-drop gate: disabled.
- 2025 example output reported 246 prediction dates and 244 valid trading days.

## Per-Year Backtest Results

| status | name | his_t | year | ARR | ASR | MDD | total_return | benchmark_return | excess_return |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OK | long_history_his_t_10__pit_2022 | 10 | 2022 | -0.226151 | -0.557574 | -0.305796 | -0.212651 | -0.064501 | -0.148150 |
| OK | long_history_his_t_10__pit_2023 | 10 | 2023 | 0.264787 | 0.903030 | -0.188898 | 0.244903 | 0.071219 | 0.173683 |
| OK | long_history_his_t_10__pit_2024 | 10 | 2024 | 0.500372 | 1.998540 | -0.185903 | 0.464573 | 0.117304 | 0.347269 |
| OK | long_history_his_t_10__pit_2025 | 10 | 2025 | 0.318129 | 0.943986 | -0.267008 | 0.306622 | 0.114071 | 0.192551 |
| OK | long_history_his_t_21__pit_2022 | 21 | 2022 | -0.301969 | -0.744411 | -0.378160 | -0.284833 | -0.064501 | -0.220333 |
| OK | long_history_his_t_21__pit_2023 | 21 | 2023 | 0.327430 | 1.087903 | -0.194759 | 0.302307 | 0.071219 | 0.231088 |
| OK | long_history_his_t_21__pit_2024 | 21 | 2024 | 0.411307 | 1.624153 | -0.163316 | 0.382661 | 0.117304 | 0.265356 |
| OK | long_history_his_t_21__pit_2025 | 21 | 2025 | 0.319820 | 0.966859 | -0.257500 | 0.308244 | 0.114071 | 0.194173 |
| OK | long_history_his_t_63__pit_2022 | 63 | 2022 | -0.319445 | -0.768806 | -0.395070 | -0.301546 | -0.064501 | -0.237045 |
| OK | long_history_his_t_63__pit_2023 | 63 | 2023 | 0.437239 | 1.442319 | -0.195330 | 0.402498 | 0.071219 | 0.331278 |
| OK | long_history_his_t_63__pit_2024 | 63 | 2024 | 0.465385 | 1.777970 | -0.213492 | 0.432431 | 0.117304 | 0.315127 |
| OK | long_history_his_t_63__pit_2025 | 63 | 2025 | 0.485046 | 1.450143 | -0.279652 | 0.466519 | 0.114071 | 0.352448 |
| OK | long_history_his_t_126__pit_2022 | 126 | 2022 | -0.302605 | -0.761160 | -0.407233 | -0.285442 | -0.064501 | -0.220941 |
| OK | long_history_his_t_126__pit_2023 | 126 | 2023 | 0.382280 | 1.220755 | -0.223941 | 0.352419 | 0.071219 | 0.281200 |
| OK | long_history_his_t_126__pit_2024 | 126 | 2024 | 0.537102 | 2.002917 | -0.215559 | 0.498267 | 0.117304 | 0.380963 |
| OK | long_history_his_t_126__pit_2025 | 126 | 2025 | 0.363595 | 1.020128 | -0.296904 | 0.350236 | 0.114071 | 0.236165 |

## Grouped Summary

| his_t | ARR mean | ASR mean | MDD mean | total return mean | excess return mean | ARR median | ASR median | MDD median | failure rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.214284 | 0.821995 | -0.236901 | 0.200862 | 0.141338 | 0.291458 | 0.923508 | -0.227953 | 0.000000 |
| 21 | 0.189147 | 0.733626 | -0.248434 | 0.177095 | 0.117571 | 0.323625 | 1.027381 | -0.226129 | 0.000000 |
| 63 | 0.267056 | 0.975406 | -0.270886 | 0.249976 | 0.190452 | 0.451312 | 1.446231 | -0.246572 | 0.000000 |
| 126 | 0.245093 | 0.870660 | -0.285909 | 0.228870 | 0.169347 | 0.372938 | 1.120442 | -0.260423 | 0.000000 |

## Interpretation

`his_t=63` is the best first-pass long-history candidate by mean ARR, mean ASR,
mean total return, and mean excess return across the 2022-2025 PIT masked-panel
matrix. It wins 2023 and 2025 by ASR and excess return, and it is the best
overall grouped candidate despite a weak 2022.

`his_t=126` is the second-best grouped candidate. It wins 2024, but its average
drawdown is worse than `his_t=10`, `21`, and `63`, and it gives back some of the
2025 improvement seen at `his_t=63`.

`his_t=10` remains the least-bad 2022 configuration and has the best grouped
mean and median drawdown. The result therefore does not support a blanket
"longer is always better" claim. It supports promoting `his_t=63` as the next
candidate to evaluate more deeply, while retaining the 10-day baseline as a
defensive reference.

`his_t=21` did not improve the grouped result versus the 10-day baseline.

## Issue 23 Close-Out Recommendation

The first-pass GRU-attention long-history PIT matrix is complete enough to close
issue #23 for the controlled preset-and-evaluation slice:

- Long-history presets exist for `his_t=21`, `63`, and `126`.
- `his_t=10` was included as the same-notebook baseline.
- `his_t=252` remained gated, consistent with the validation-cost guidance.
- All 16 provided PIT masked-panel rows completed with status `OK`.
- The evidence identifies `his_t=63` as the preferred first-pass candidate.

Recommended follow-up issues:

- Run `his_t=63` replication checks across additional seeds.
- Evaluate transaction-cost and rank-drop-gated variants for `his_t=63`.
- Evaluate `his_t=126` only if there is a specific reason to trade higher
  drawdown risk for its stronger 2024 behavior.
- Keep `his_t=252` and transformer temporal encoder comparisons as separate,
  explicitly budgeted follow-up work rather than blockers for issue #23.
