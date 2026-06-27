# Issue #8 Volatility-Targeting Backtest Impact

Date: 2026-05-26

## Run

This report replays the full-budget issue #8 volatility-targeting predictions
through the current PIT saved-prediction daily backtest. It does not retrain
models.

- Colab run root: `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_full/20260526_174325`
- Backtest output: `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_full/20260526_174325/summaries/issue8_volatility_backtest/`
- Source predictions:
  - 2022: `training/issue8_vol_targeting_2022_seed314159/20260526_174355/averaged_predictions`
  - 2024: `training/issue8_vol_targeting_2024_seed314159/20260526_181224/averaged_predictions`
- Backtest command surface: `scripts/run_pit_saved_prediction_backtests.py`
- Scenario: `top_k=10`, `label_t=5`, 10 bps spread, 5 bps slippage,
  rank-drop gate enabled with `min_rank_drop=30`

The first terminal replay failed before backtest execution because the terminal
subprocess lacked `PYTHONPATH=/content/MCI-GRU`. The successful replay reran
the same command with that environment set.

## Baseline

Primary comparison is seed-matched against the current-preset PIT repeated-seed
Option A baseline, `base_seed=314159`, using the same cost/rank-gate scenario.
The all-seed mean is included only as context because issue #8 currently has
one full-budget seed.

## Results

| Year | Baseline Total | Vol Total | Delta Total | Baseline Excess | Vol Excess | Delta Excess | Baseline ASR | Vol ASR | Delta ASR | Baseline Turnover | Vol Turnover | Delta Turnover |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | -24.22% | -26.97% | -2.76% | -17.77% | -20.52% | -2.76% | -0.621 | -0.670 | -0.050 | 4.00% | 3.45% | -0.55% |
| 2024 | 11.31% | 20.04% | 8.72% | -0.42% | 8.31% | 8.72% | 0.522 | 1.528 | 1.006 | 15.23% | 4.35% | -10.89% |

## Vol-Targeted Backtest Rows

| Year | Total | Gross Total | Net Total | Benchmark | Excess | ARR | ASR | AVoL | MDD | Turnover | Cost Bps/Day | Trades | Gate Exit Days |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | -26.97% | -25.78% | -26.97% | -6.45% | -20.52% | -28.62% | -0.670 | 42.68% | -42.31% | 3.45% | 0.689 | 162 | 48 |
| 2024 | 20.04% | 22.53% | 20.04% | 11.73% | 8.31% | 21.43% | 1.528 | 14.03% | -6.66% | 4.35% | 0.869 | 206 | 74 |

## Baseline Context

| Year | Current Baseline Seeds | Mean Total | Mean Excess | Mean ASR | Mean Turnover |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 3 | -23.13% | -16.68% | -0.613 | 4.03% |
| 2024 | 3 | 11.75% | 0.02% | 0.536 | 18.10% |

## Interpretation

Volatility targeting did not rescue 2022 in this single-seed replay. It reduced
turnover and daily cost drag slightly, but total and excess return were about
2.8 percentage points worse than the seed-matched baseline, with a deeper
drawdown.

The 2024 effect is much more encouraging. Volatility targeting raised net
return by 8.7 percentage points versus the seed-matched baseline and turned
excess return positive. The improvement was not just lower costs: gross total
return also improved, while turnover fell from 15.23% to 4.35% and daily cost
drag dropped from 3.05 bps to 0.87 bps. That pattern is consistent with the
feature reducing churn in the difficult 2024 regime.

## Next Read

The next useful pass is a repeated-seed volatility-targeting run or a saved
prediction replay across additional cost settings. One seed is enough to show
that the feature can materially improve the 2024 churn problem, but not enough
to promote it as a new default.
