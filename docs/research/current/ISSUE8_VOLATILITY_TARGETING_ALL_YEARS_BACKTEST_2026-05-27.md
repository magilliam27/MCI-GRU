# Issue #8 Volatility-Targeting All-Years Backtest

Date: 2026-05-27 UTC / 2026-05-26 ET

## Run

This report extends the issue #8 full-budget volatility-targeting test from
2022 and 2024 to the usual 2022-2025 PIT years. It replays the saved
volatility-targeted predictions through the current PIT saved-prediction daily
backtest. It does not retrain baseline models.

- 2022/2024 Colab run root:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_full/20260526_174325`
- 2023/2025 Colab run root:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_all_years/20260527_001608`
- 2022/2024 backtest output:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_full/20260526_174325/summaries/issue8_volatility_backtest/`
- 2023/2025 backtest output:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_all_years/20260527_001608/summaries/issue8_volatility_backtest_2023_2025/`
- Source predictions:
  - 2022: `training/issue8_vol_targeting_2022_seed314159/20260526_174355/averaged_predictions`
  - 2023: `training/issue8_vol_targeting_2023_seed314159/20260527_001619/averaged_predictions`
  - 2024: `training/issue8_vol_targeting_2024_seed314159/20260526_181224/averaged_predictions`
  - 2025: `training/issue8_vol_targeting_2025_seed314159/20260527_004316/averaged_predictions`
- Backtest command surface: `scripts/run_pit_saved_prediction_backtests.py`
- Scenario: `top_k=10`, `label_t=5`, 10 bps spread, 5 bps slippage,
  rank-drop gate enabled with `min_rank_drop=30`
- Volatility-targeting feature set:
  `half_lives=[20,60,90]`, target volatility `0.10`, scale clip
  `[0.25,4.0]`, interaction return window `21`

## Baseline

Primary comparison is seed-matched against the current-preset PIT repeated-seed
Option A baseline, `base_seed=314159`, using the same cost/rank-gate scenario.
This remains a single-seed read for the volatility-targeted variant.

## Results

| Year | Baseline Total | Vol Total | Delta Total | Baseline Excess | Vol Excess | Delta Excess | Baseline ASR | Vol ASR | Delta ASR | Baseline Turnover | Vol Turnover | Delta Turnover |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | -24.22% | -26.97% | -2.76% | -17.77% | -20.52% | -2.76% | -0.621 | -0.670 | -0.050 | 4.00% | 3.45% | -0.55% |
| 2023 | 60.40% | 6.33% | -54.07% | 53.28% | -0.79% | -54.07% | 2.377 | 0.337 | -2.040 | 8.85% | 8.09% | -0.77% |
| 2024 | 11.31% | 20.04% | 8.72% | -0.42% | 8.31% | 8.72% | 0.522 | 1.528 | 1.006 | 15.23% | 4.35% | -10.89% |
| 2025 | 38.79% | 50.98% | 12.19% | 31.65% | 43.84% | 12.19% | 1.292 | 1.374 | 0.082 | 6.99% | 2.80% | -4.19% |

## Vol-Targeted Backtest Rows

| Year | Total | Gross Total | Excess | ARR | ASR | AVoL | MDD | Turnover | Cost Bps/Day | Trades | Gate Exit Days |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | -26.97% | -25.78% | -20.52% | -28.62% | -0.670 | 42.68% | -42.31% | 3.45% | 0.689 | 162 | 48 |
| 2023 | 6.33% | 10.46% | -0.79% | 6.81% | 0.337 | 20.18% | -17.01% | 8.09% | 1.617 | 380 | 106 |
| 2024 | 20.04% | 22.53% | 8.31% | 21.43% | 1.528 | 14.03% | -6.66% | 4.35% | 0.869 | 206 | 74 |
| 2025 | 50.98% | 52.98% | 43.84% | 55.26% | 1.374 | 40.22% | -35.59% | 2.80% | 0.559 | 132 | 41 |

## Interpretation

Volatility targeting is not uniformly beneficial across the usual years in this
single-seed read. It helps materially in 2024 and 2025, hurts slightly in 2022,
and severely underperforms the strong 2023 seed-matched baseline.

The 2024 and 2025 improvements are consistent with the original issue #8 thesis:
the feature can reduce churn and cost drag. Turnover falls from 15.23% to 4.35%
in 2024 and from 6.99% to 2.80% in 2025. Those cost reductions come with better
net excess return in both years.

The 2023 result is the caution flag. Turnover declines only modestly, but total
return falls by 54.1 percentage points versus baseline. That suggests the
feature can dampen or distort a useful signal in a strong year, not merely save
transaction costs. The 2022 result also stays negative, so volatility targeting
does not solve the regime-stress failure on its own.

## Decision Read

This feature is worth continuing, but not yet as a default. The next promotion
gate should be repeated-seed testing and targeted diagnostics on 2023: feature
ablation by half-life, scale-clip sensitivity, and per-date/sector attribution
around the largest missed winners. The current evidence supports treating
volatility targeting as a promising risk-control signal with regime-dependent
benefits, not as a blanket improvement.
