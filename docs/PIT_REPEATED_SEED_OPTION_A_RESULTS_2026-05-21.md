# PIT Repeated-Seed Option A Results

Date prepared: 2026-05-21

Primary Drive run:
[MCI-GRU-Ablations/pit_repeated_seed_replication/20260520_183538](https://drive.google.com/drive/folders/1ecTUflJGVILFpaZ0TKDiGrsc4W1EYhFj)

This report reviews the expanded Option A notebook run described in
`docs/handoffs/2026-05-20-pit-repeated-seed-option-a-notebook.md`.

## Executive Summary

The expanded repeated-seed run completed the intended full budget: three base
seeds, four PIT years, 12 yearly training jobs, 20 models per yearly job, and
240 trained models total. The run used the frozen default recipe, `masked_panel`
PIT universe mode, transaction-cost-aware backtests, and the rank-drop gate.

The evidence supports closing issue #31 as a PIT pipeline validity check. It
also supports closing issue #30 as a 2022 stress-regime finding: all three seeds
lost excess return in 2022, and 2022 is the worst mean-excess year.

Issue #29 should stay open under its strict closeout rule. The pooled all-seed
daily excess result is directionally strong and raw-significant, but the
BHY-adjusted p-value does not clear 0.05.

## Artifact Set

Reviewed artifacts:

- [pit_repeated_seed_replication_summary.md](https://drive.google.com/file/d/1JrJuaYDJcsBS046xb_nkUCXUirODJejE/view?usp=drivesdk)
- [pit_repeated_seed_issue_closeout_summary.csv](https://drive.google.com/file/d/1SfI64mJH_SVLMZeNcULiwDEa9II9JOZ7/view?usp=drivesdk)
- [pit_repeated_seed_seed_summary.csv](https://drive.google.com/file/d/1CJ_UvZoGRuOFgcVmUl8z9hcJOEV6CC9Z/view?usp=drivesdk)
- [pit_repeated_seed_yearly_seed_summary.csv](https://drive.google.com/file/d/1hV8RFy1rlBH2W1LV1OBMFkOW-HLPwgcu/view?usp=drivesdk)
- [pit_repeated_seed_pooled_daily_significance.csv](https://drive.google.com/file/d/18xfl61-xw4o0Qtd1727_PgNtNUyvwnQg/view?usp=drivesdk)
- [backtest_results.csv](https://drive.google.com/file/d/1jEUZrY28QRA_z54_ZF0cCxrmWIsLx3Zj/view?usp=drivesdk)
- [pit_breadth_summary.csv](https://drive.google.com/file/d/1FFXacqlKBhNjfc2WK10CYqdyK7x3oTCM/view?usp=drivesdk)
- [prediction_count_checks.csv](https://drive.google.com/file/d/18QNJQaSGeAD6p-lsrgqhzeL38MuQ8Msd/view?usp=drivesdk)
- [pit_repeated_seed_reference_comparison.csv](https://drive.google.com/file/d/1xX7VSnt41ZlRbSeXgvgP41xeKUNkYwFM/view?usp=drivesdk)
- [pit_repeated_seed_2022_monthly_diagnostics.csv](https://drive.google.com/file/d/1xTRSUycuD9BhkCnc8UqaYwLBZJFrpdtc/view?usp=drivesdk)
- [pit_repeated_seed_2022_drawdown_diagnostics.csv](https://drive.google.com/file/d/1cIp7puohB_T8vbbbI4rL_NdXYbEVl_dU/view?usp=drivesdk)
- [pit_repeated_seed_2022_holdings_diagnostics.csv](https://drive.google.com/file/d/1W1yxwjq2Dedjodj2a2KP55DDYHVVzDX_/view?usp=drivesdk)

## Run Identity

| Field | Value |
|---|---|
| Run tag | `20260520_183538` |
| Reference run tag | `20260514_043539` |
| Recipe | `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1` |
| Smoke mode | `False` |
| Base seeds | `314159`, `271828`, `161803` |
| PIT years | `2022`, `2023`, `2024`, `2025` |
| Yearly jobs | `12` |
| Models per job | `20` |
| Total models | `240` |
| Market SHA256 | `84e1f3f2b79a798246e001e17a372c8daf8bcfc658873ad3d352a99ad993840f` |
| Static regime SHA256 | `4c6071d896360b48e9c648ae3bfefe4ae783fb0d4e055b195d97dce3e2e163dc` |

## Issue Closeout Status

| Issue | Status | Evidence |
|---:|---|---|
| #31 | Supports closeout | `training_ok=True`, `backtests_ok=True`, `predictions_ok=True`, `breadth_ok=True`, `completed_seed_count=3`, `expected_jobs=12`, `expected_total_models=240`. |
| #29 | Needs more evidence | `replication_all_seeds` annualized excess is `13.31%`, IR is `0.6421`, raw p-value is `0.0198`, and the daily excess 95% CI lower bound is positive, but BHY-adjusted p-value is `0.2260`. |
| #30 | Supports closeout | 2022 completed all three seeds, all three seeds had negative excess return, mean excess was `-16.48%`, and 2022 was the worst mean-excess year. |

## Seed-Level Performance

All three seeds completed all four years. Every seed is positive across the
full 2022-2025 pooled window, but none clears the per-seed adjusted p-value
threshold.

| Base seed | Annualized excess | IR | Raw p-value | BHY-adjusted p-value | Worst yearly excess |
|---:|---:|---:|---:|---:|---:|
| `314159` | `15.60%` | `0.7414` | `0.1187` | `0.4972` | `-17.19%` |
| `271828` | `14.59%` | `0.7101` | `0.1307` | `0.4972` | `-13.85%` |
| `161803` | `9.72%` | `0.4723` | `0.3309` | `0.9196` | `-18.39%` |

The pooled all-seed row has more power and is the right row for issue #29, but
the pre-specified multiple-testing gate still fails:

| Scenario | Days | Annualized excess | IR | Newey-West t | Raw p-value | BHY-adjusted p-value | Daily CI low |
|---|---:|---:|---:|---:|---:|---:|---:|
| `replication_all_seeds` | `2829` | `13.31%` | `0.6421` | `2.3303` | `0.0198` | `0.2260` | `0.000095` |

## Yearly Cross-Seed Read

| Year | Positive seeds | Negative seeds | Mean excess | Mean ASR | Mean MDD | Worst mean-excess year |
|---:|---:|---:|---:|---:|---:|---|
| 2022 | `0` | `3` | `-16.48%` | `-0.6082` | `-33.61%` | `True` |
| 2023 | `3` | `0` | `37.74%` | `1.8356` | `-12.78%` | `False` |
| 2024 | `1` | `2` | `0.32%` | `0.5492` | `-18.75%` | `False` |
| 2025 | `3` | `0` | `32.52%` | `1.2921` | `-26.52%` | `False` |

This pattern is not a simple high-variance success story. 2023 and 2025 are
robustly positive, 2024 is effectively flat across seeds, and 2022 remains a
repeatable failure regime.

## Backtest Sensitivity Replay

I replayed the saved averaged predictions from the same Drive zip using three
cost/horizon variants. This replay does not retrain models.

Baseline replay preset:

- `top_k=10`
- `label_t=5`
- `num_tests=4`
- `adjustment_method=bhy`
- transaction costs enabled
- `spread=10`
- `slippage=5`
- rank-drop gate enabled
- `min_rank_drop=30`

Sensitivity variants:

- `spread5_only_label5`: `spread=5`, `slippage=0`, same rank gate and
  training-matched `label_t=5`.
- `spread5_only_label21_diagnostic`: `spread=5`, `slippage=0`, same rank gate,
  diagnostic `label_t=21`.

The `label_t=21` row is diagnostic for this artifact set because the Option A
models were trained with `model.label_t=5`. In `tests/backtest_sp500_daily.py`,
changing `label_t` changes prediction-vs-forward-return evaluation fields; the
daily portfolio P&L path remains open-to-open.

Mean across all 12 seed-year backtests:

| Scenario | Mean total return | Mean ARR | Mean ASR | Mean MDD | Mean transaction cost |
|---|---:|---:|---:|---:|---:|
| `current_tc10_slip5_label5` | `18.25%` | `19.88%` | `0.7621` | `-23.00%` | `4.69%` |
| `spread5_only_label5` | `22.63%` | `24.63%` | `0.9389` | `-22.57%` | `1.17%` |
| `spread5_only_label21_diagnostic` | `22.63%` | `24.63%` | `0.9389` | `-22.57%` | `1.17%` |

Mean spread-only deltas versus the promoted 10 bps spread + 5 bps slippage
baseline:

| Year | Delta total return | Delta ARR | Delta ASR | Delta MDD |
|---:|---:|---:|---:|---:|
| 2022 | `+1.10 pp` | `+1.15 pp` | `+0.0290` | `+0.79 pp` |
| 2023 | `+4.87 pp` | `+5.37 pp` | `+0.2029` | `+0.47 pp` |
| 2024 | `+7.47 pp` | `+8.02 pp` | `+0.3404` | `+0.18 pp` |
| 2025 | `+4.08 pp` | `+4.46 pp` | `+0.1347` | `+0.28 pp` |
| All seed-years | `+4.38 pp` | `+4.75 pp` | `+0.1767` | `+0.43 pp` |

The `label_t=21` diagnostic has identical P&L, ARR, ASR, MDD, turnover, and
costs to `spread5_only_label5`. It only changes the forward-return diagnostics:
mean MSE is `+0.0046` higher and mean MAE is `+0.0063` higher versus the
training-matched label-5 replay.

## 2022 Stress Diagnosis

The 2022 stress result is repeatable across the new seeds and consistent with
the earlier reference run. The replication seeds improve versus the reference
2022 excess return of `-22.06%`, but all remain meaningfully negative:

| Scenario | 2022 excess return | 2022 MDD | Drawdown bottom | Recovered by year end |
|---|---:|---:|---|---|
| Reference | `-22.06%` | `-37.88%` | `2022-12-27` | `False` |
| `314159` | `-17.19%` | `-34.35%` | `2022-10-12` | `False` |
| `271828` | `-13.85%` | `-32.86%` | `2022-10-12` | `False` |
| `161803` | `-18.39%` | `-33.63%` | `2022-12-27` | `False` |

The monthly diagnostics show a familiar shape: April is sharply negative across
all three replication seeds, July rebounds, and late-year weakness persists.
That supports treating 2022 as a regime stress case rather than a one-seed
training accident.

## PIT Integrity Checks

The PIT mechanics look clean. In `pit_breadth_summary.csv`, every train, val,
and test split reports `status=OK`, and all below-threshold counters are zero.
Test scoreable medians stay around 503 names across all years and seeds.

Prediction exports also match the expected PIT masks exactly. Across all 12
yearly seed jobs, `prediction_files_missing=0` and
`prediction_count_mismatches=0`.

## Recommendation

Close #31 as satisfied by this run's mechanics and artifact completeness.
Close #30 as a confirmed 2022 stress-regime result. Keep #29 open, but treat it
as narrowed rather than failed: the signal is positive under costs and rank
gate, and the all-seed raw significance plus positive CI are encouraging, but
the strict BHY-adjusted closeout criterion was not met.

If #29 is the remaining decision blocker, the clean next experiment is to add a
small number of additional base seeds under the exact same frozen recipe and
the same closeout table, without changing model features or backtest rules.
