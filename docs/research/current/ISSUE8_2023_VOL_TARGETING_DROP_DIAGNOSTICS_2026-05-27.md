# Issue #8 2023 Volatility-Targeting Drop Diagnostics

## Inputs

- Baseline: `.codex_tmp/pit_option_a_extract_20260520_183538/.../pit_seed_314159_replication_2023/20260520_185710/backtest_pit_daily_sensitivity_tc10_slip5_rank_gate_label5`
- Vol-targeted: Drive run `volatility_targeting_issue8_all_years/20260527_001608/training/issue8_vol_targeting_2023_seed314159/20260527_001619/backtest_pit_daily_tc_rank_gate`
- Scenario: `top_k=10`, `label_t=5`, 10 bps spread, 5 bps slippage, `min_rank_drop=30`.
- Local diagnostic artifacts: `.codex_tmp/issue8_vol2023_diag/`

## Summary

- Net total return falls from `60.40%` to `6.33%`, a `-54.07 pp` delta.
- Gross total return falls from `67.19%` to `10.46%`, a `-56.73 pp` delta.
- Total transaction costs fall from `4.16%` to `3.80%`; lower costs help by `0.36 pp`, so costs do not explain the drop.
- Average selected-holding Jaccard is only `0.060`; the two runs share only `1.01` names/day on average.
- Baseline-only names averaged `0.29%` per daily holding bucket versus `0.07%` for vol-only replacements.

## Monthly Differential

| month | baseline net | vol net | delta net | baseline gross | vol gross | delta gross | baseline cost | vol cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-01 | 3.49% | 2.23% | -1.26 pp | 3.72% | 2.37% | -1.35 pp | 0.22% | 0.14% |
| 2023-02 | 1.44% | -1.27% | -2.71 pp | 2.03% | -1.21% | -3.24 pp | 0.58% | 0.06% |
| 2023-03 | 6.46% | -8.02% | -14.48 pp | 7.18% | -7.52% | -14.70 pp | 0.68% | 0.54% |
| 2023-04 | 2.22% | 4.77% | 2.56 pp | 2.62% | 4.92% | 2.30 pp | 0.40% | 0.14% |
| 2023-05 | 14.81% | -5.18% | -19.99 pp | 15.27% | -5.02% | -20.29 pp | 0.40% | 0.16% |
| 2023-06 | 10.16% | 7.74% | -2.42 pp | 10.68% | 7.91% | -2.77 pp | 0.48% | 0.16% |
| 2023-07 | 4.05% | 0.35% | -3.70 pp | 4.18% | 0.41% | -3.77 pp | 0.12% | 0.06% |
| 2023-08 | 2.11% | 0.02% | -2.09 pp | 2.33% | 0.47% | -1.87 pp | 0.22% | 0.44% |
| 2023-09 | -7.12% | -9.27% | -2.15 pp | -6.90% | -8.85% | -1.95 pp | 0.24% | 0.46% |
| 2023-10 | -4.89% | 2.24% | 7.12 pp | -4.54% | 2.81% | 7.35 pp | 0.36% | 0.56% |
| 2023-11 | 10.86% | 8.11% | -2.75 pp | 11.32% | 8.71% | -2.61 pp | 0.42% | 0.56% |
| 2023-12 | 6.70% | 6.31% | -0.39 pp | 6.74% | 6.87% | 0.13 pp | 0.04% | 0.52% |

The drop is concentrated in gross selection, especially May and March. October is the only major offsetting month.

## Worst Daily Net Deltas

| date | baseline return | vol return | net delta | gross delta | baseline cost | vol cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-03-13 | 11.75% | 1.81% | -9.94 pp | -9.92 pp | 0.04% | 0.06% |
| 2023-05-26 | 4.14% | -0.39% | -4.53 pp | -4.57 pp | 0.04% | 0.00% |
| 2023-03-29 | 3.44% | 0.03% | -3.40 pp | -3.42 pp | 0.02% | 0.00% |
| 2023-05-24 | 3.15% | -0.16% | -3.31 pp | -3.33 pp | 0.02% | 0.00% |
| 2023-10-11 | 1.09% | -2.09% | -3.18 pp | -3.16 pp | 0.00% | 0.02% |
| 2023-08-21 | 3.30% | 0.53% | -2.77 pp | -2.73 pp | 0.00% | 0.04% |
| 2023-12-05 | 1.77% | -0.90% | -2.67 pp | -2.63 pp | 0.00% | 0.04% |
| 2023-03-21 | 2.58% | 0.00% | -2.58 pp | -2.48 pp | 0.06% | 0.16% |
| 2023-03-14 | -4.32% | -6.87% | -2.56 pp | -2.56 pp | 0.02% | 0.02% |
| 2023-05-18 | 3.45% | 1.05% | -2.40 pp | -2.42 pp | 0.02% | 0.00% |

## Biggest Baseline-Only Winners Missed By Vol Targeting

| entry date | kdcode | return | baseline rank | baseline score |
| --- | --- | ---: | ---: | ---: |
| 2023-03-13 | FRC.N^E23 | 83.82% | 6 | -0.12358 |
| 2023-05-24 | NVDA.OQ | 27.52% | 2 | -0.12756 |
| 2023-04-27 | FRC.N^E23 | 14.61% | 5 | -0.23487 |
| 2023-02-22 | NVDA.OQ | 13.24% | 2 | -0.18532 |
| 2023-01-25 | TSLA.OQ | 12.67% | 3 | -0.29274 |
| 2023-05-15 | DISH.OQ^A24 | 11.84% | 7 | -0.16242 |
| 2023-03-29 | FRC.N^E23 | 11.22% | 1 | -0.15049 |
| 2023-08-03 | AMZN.OQ | 10.58% | 10 | 0.01113 |
| 2023-05-26 | NFLX.OQ | 10.17% | 2 | -0.13117 |
| 2023-08-14 | NVDA.OQ | 10.03% | 1 | 0.16721 |
| 2023-01-27 | TSLA.OQ | 9.57% | 14 | -0.31339 |
| 2023-11-01 | AMD.OQ | 9.50% | 9 | 0.16904 |

## Biggest Vol-Only Losers Not In Baseline

| entry date | kdcode | return | vol rank | vol score |
| --- | --- | ---: | ---: | ---: |
| 2023-05-25 | ULTA.OQ | -10.49% | 25 | -0.26213 |
| 2023-12-12 | PFE.N | -8.14% | 17 | 0.81465 |
| 2023-10-11 | PODD.OQ | -7.64% | 7 | 0.14952 |
| 2023-06-13 | UNH.N | -7.61% | 1 | -0.01654 |
| 2023-07-28 | AON.N | -7.22% | 25 | 0.24996 |
| 2023-12-04 | ALB.N | -7.13% | 3 | 0.92900 |
| 2023-08-24 | ENPH.OQ | -6.90% | 3 | 0.26747 |
| 2023-03-10 | VFC.N | -6.86% | 4 | -0.20850 |
| 2023-08-24 | SEDG.OQ | -6.36% | 19 | 0.17126 |
| 2023-03-13 | FIS.N | -6.35% | 8 | -0.25313 |
| 2023-03-15 | HAL.N | -6.34% | 9 | -0.23588 |
| 2023-03-10 | APA.OQ | -6.11% | 6 | -0.22992 |

## Diagnosis

The 2023 collapse is primarily a gross selection/ranking problem, not a turnover-cost problem. Volatility targeting lowered total costs, but the selected book changed almost completely: only about one shared holding per day.

The next experiment should isolate which volatility channel caused the ranking change. Recommended ablations:

1. Volatility level features only, without Harvey-style scaling features.
2. Volatility scaling/clipped forecast-vol features only, without extra raw volatility level features.
3. Guardrail sensitivity around the current clipped interval.
4. Matched no-volatility retrain under the same seed/year as a sanity check against ordinary seed drift.

Promotion should wait until the feature improves, or at least does not materially harm, 2023 holding overlap and gross return under the current cost-aware rank-gated backtest.
