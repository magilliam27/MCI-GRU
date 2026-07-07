# S&P 500 PIT Top-10 Loss Backtest Comparison, Recovered All Years

Date: 2026-06-30

Run root: https://drive.google.com/drive/folders/1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo
Summaries folder: https://drive.google.com/drive/folders/18o3CdNWKndiphJQGQyCVieyWuKFkV_1E
Logs folder: https://drive.google.com/drive/folders/1qv_Wy_G-AQLRxnkEjDApCkhTHlTn3eo3

## Executive Read

I recovered the missing 2023 and 2024 backtest rows from Drive log files, plus the missing 2025 pure-IC seed 271828 row. The newly trained loss matrix is now complete for the intended comparison: 28 backtest rows across 2022-2025.

Recommendation: keep pure IC as the launch/default objective. LambdaRankIC should remain experimental or be tested as a hybrid/second-stage ranker. It has the best row-average result and wins the 2022 stress year plus 2023, but it is not robust enough across years/seeds to replace pure IC: in 2024 one LambdaRankIC seed churned 362 trades, paid 3.62% cumulative cost, and barely beat the benchmark.

Portfolio-IC weight50 is also not a default replacement. It is the cleanest 2024 winner and trades much less than LambdaRankIC, but it did poorly in 2022 and lagged pure IC in 2025.

## Recovery Notes

- The visible `summaries/backtest_rows.json` was overwritten after the larger version existed. The file ID is `1b3-neuNES3JRPgCbXpmdniUmVro1OPvR`.
- The pre-overwrite revision was visible in revision history at 41,306 bytes, modified `2026-06-30T00:19:39.947Z`, revision id `0Bwr_sXAFm255VVdROWVMOVAzYytzWGR3bFF5KzZOZWkxTGtZPQ`; fetching it failed with `GoogleDriveInvalidRequestError: No supported mimetype returned for revision`.
- Drive search did not expose standalone `backtest_metrics.json` or `backtest_results.csv` files, but the uploaded `backtest_top10_*.log` files contain the full metric blocks and output paths.
- No retraining was launched. No saved-prediction replay was needed after log recovery.

## Backtest Setup Held Constant

- Universe: PIT masked-panel GICS top-10-per-sector setup, 110 selected names per snapshot in the completed run summary.
- Portfolio: top-10 stocks, open-to-open returns, entry T+1 open and exit T+2 open.
- Benchmark: equal-weight open-to-open on the full calendar in the test window.
- Costs: rank-drop gate enabled with min rank drop 30; bid-ask spread 10 bps round-trip; slippage 5 bps per trade.
- Seed interpretation: these are repeated experiment/base seeds, not the 20 ensemble member seeds.

## Coverage

| Year | pure IC | LambdaRankIC | Portfolio-IC weight50 |
|---:|---:|---:|---:|
| 2022 | 2/2 | 2/2 | 3/3 |
| 2023 | 2/2 | 2/2 | 3/3 |
| 2024 | 2/2 | 2/2 | 3/3 |
| 2025 | 2/2 | 2/2 | 3/3 |

Newly trained rows: 28/28 recovered. Known existing reused rows, including pure-IC seed 1729 and LambdaRankIC seed 314159 from older run families, are not included in the main aggregates.

## Aggregate Metrics By Loss And Year

All returns, drawdowns, and costs are shown as percentages. Cost is cumulative transaction cost over the year.

| Year | Loss | Seeds | Net total return | ARR | ASR | MDD | Excess return | Avg trades | Avg cost |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | LambdaRankIC | 2 | -22.68 | -23.33 | -0.81 | -31.15 | -11.40 | 16.0 | 0.16 |
| 2022 | Portfolio-IC weight50 | 3 | -42.09 | -43.12 | -1.07 | -44.21 | -30.82 | 10.0 | 0.10 |
| 2022 | pure IC | 2 | -40.55 | -41.55 | -1.03 | -42.82 | -29.28 | 12.0 | 0.12 |
| 2023 | LambdaRankIC | 2 | 43.85 | 45.58 | 2.70 | -8.17 | 32.00 | 70.0 | 0.70 |
| 2023 | Portfolio-IC weight50 | 3 | 28.93 | 30.01 | 1.85 | -11.87 | 17.08 | 60.7 | 0.61 |
| 2023 | pure IC | 2 | 20.82 | 21.57 | 1.38 | -9.05 | 8.97 | 48.0 | 0.48 |
| 2024 | LambdaRankIC | 2 | 28.05 | 28.84 | 1.52 | -13.66 | 13.37 | 241.0 | 2.41 |
| 2024 | Portfolio-IC weight50 | 3 | 42.41 | 43.64 | 1.77 | -20.10 | 27.73 | 24.0 | 0.24 |
| 2024 | pure IC | 2 | 40.58 | 41.75 | 1.79 | -19.81 | 25.90 | 55.0 | 0.55 |
| 2025 | LambdaRankIC | 2 | 27.19 | 28.20 | 1.21 | -18.11 | 11.44 | 107.0 | 1.07 |
| 2025 | Portfolio-IC weight50 | 3 | 22.35 | 23.16 | 0.82 | -25.43 | 6.59 | 18.7 | 0.19 |
| 2025 | pure IC | 2 | 31.38 | 32.56 | 1.14 | -23.37 | 15.62 | 23.0 | 0.23 |

## Overall Row-Average Summary

This is a row average across annual backtests and base seeds, not a stitched portfolio equity curve.

| Loss | Rows | Net total return | ARR | ASR | MDD | Excess return | Avg trades | Avg cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LambdaRankIC | 8 | 19.10 | 19.82 | 1.16 | -17.77 | 11.35 | 108.5 | 1.09 |
| Portfolio-IC weight50 | 12 | 12.90 | 13.42 | 0.84 | -25.40 | 5.14 | 28.3 | 0.28 |
| pure IC | 8 | 13.06 | 13.58 | 0.82 | -23.76 | 5.30 | 34.5 | 0.34 |

## Year Notes

2022 stress year: LambdaRankIC clearly reduced losses versus both pure IC and Portfolio-IC weight50. Mean net return was -22.68% versus -40.55% for pure IC and -42.09% for Portfolio-IC weight50. This is the strongest pro-LambdaRankIC evidence in the matrix.

2023: LambdaRankIC was the clear winner on mean net return and ASR. Its two recovered seeds returned 54.10% and 33.59% net, versus pure IC at 22.51% and 19.12%, and Portfolio-IC weight50 at 24.41%, 29.03%, and 33.34%.

2024: LambdaRankIC was not robust. Seed 161803 returned 40.22% net, but seed 271828 returned only 15.88%, had 362 trades, and paid 3.62% cumulative cost. Portfolio-IC weight50 was the cleanest 2024 result: all three seeds returned 41.20%-44.16% with only 22-26 trades.

2025: pure IC led on mean net return, 31.38%, helped by recovered seed 271828 at 35.86%. LambdaRankIC remained positive but did not beat pure IC on mean return, while Portfolio-IC weight50 lagged.

## Per-Row Backtest Metrics

| Year | Loss | Seed | Net total return | ARR | ASR | MDD | Benchmark | Excess | Trades | Cost |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | pure IC | 161803 | -39.01 | -39.99 | -1.00 | -41.42 | -11.27 | -27.74 | 14 | 0.14 |
| 2022 | pure IC | 271828 | -42.09 | -43.12 | -1.07 | -44.21 | -11.27 | -30.82 | 10 | 0.10 |
| 2022 | LambdaRankIC | 161803 | -21.96 | -22.59 | -0.80 | -32.13 | -11.27 | -10.69 | 16 | 0.16 |
| 2022 | LambdaRankIC | 271828 | -23.39 | -24.06 | -0.81 | -30.18 | -11.27 | -12.12 | 16 | 0.16 |
| 2022 | Portfolio-IC weight50 | 161803 | -42.09 | -43.12 | -1.07 | -44.21 | -11.27 | -30.82 | 10 | 0.10 |
| 2022 | Portfolio-IC weight50 | 271828 | -42.09 | -43.12 | -1.07 | -44.21 | -11.27 | -30.82 | 10 | 0.10 |
| 2022 | Portfolio-IC weight50 | 314159 | -42.09 | -43.12 | -1.07 | -44.21 | -11.27 | -30.82 | 10 | 0.10 |
| 2023 | pure IC | 161803 | 22.51 | 23.33 | 1.30 | -10.32 | 11.85 | 10.66 | 70 | 0.70 |
| 2023 | pure IC | 271828 | 19.12 | 19.80 | 1.46 | -7.78 | 11.85 | 7.27 | 26 | 0.26 |
| 2023 | LambdaRankIC | 161803 | 54.10 | 56.30 | 3.14 | -9.31 | 11.85 | 42.25 | 116 | 1.16 |
| 2023 | LambdaRankIC | 271828 | 33.59 | 34.86 | 2.25 | -7.03 | 11.85 | 21.74 | 24 | 0.24 |
| 2023 | Portfolio-IC weight50 | 161803 | 24.41 | 25.31 | 1.46 | -15.24 | 11.85 | 12.56 | 102 | 1.02 |
| 2023 | Portfolio-IC weight50 | 271828 | 29.03 | 30.11 | 2.05 | -9.39 | 11.85 | 17.18 | 34 | 0.34 |
| 2023 | Portfolio-IC weight50 | 314159 | 33.34 | 34.61 | 2.04 | -10.98 | 11.85 | 21.49 | 46 | 0.46 |
| 2024 | pure IC | 161803 | 46.84 | 48.22 | 1.97 | -21.58 | 14.68 | 32.16 | 48 | 0.48 |
| 2024 | pure IC | 271828 | 34.32 | 35.29 | 1.61 | -18.04 | 14.68 | 19.64 | 62 | 0.62 |
| 2024 | LambdaRankIC | 161803 | 40.22 | 41.38 | 1.99 | -18.36 | 14.68 | 25.54 | 120 | 1.20 |
| 2024 | LambdaRankIC | 271828 | 15.88 | 16.30 | 1.06 | -8.95 | 14.68 | 1.20 | 362 | 3.62 |
| 2024 | Portfolio-IC weight50 | 161803 | 41.20 | 42.40 | 1.72 | -20.19 | 14.68 | 26.52 | 26 | 0.26 |
| 2024 | Portfolio-IC weight50 | 271828 | 44.16 | 45.45 | 1.85 | -19.43 | 14.68 | 29.48 | 22 | 0.22 |
| 2024 | Portfolio-IC weight50 | 314159 | 41.86 | 43.07 | 1.74 | -20.69 | 14.68 | 27.18 | 24 | 0.24 |
| 2025 | pure IC | 161803 | 26.89 | 27.89 | 0.94 | -24.70 | 15.75 | 11.14 | 20 | 0.20 |
| 2025 | pure IC | 271828 | 35.86 | 37.24 | 1.33 | -22.03 | 15.75 | 20.11 | 26 | 0.26 |
| 2025 | LambdaRankIC | 161803 | 32.28 | 33.50 | 1.39 | -19.08 | 15.75 | 16.53 | 70 | 0.70 |
| 2025 | LambdaRankIC | 271828 | 22.10 | 22.90 | 1.03 | -17.14 | 15.75 | 6.35 | 144 | 1.44 |
| 2025 | Portfolio-IC weight50 | 161803 | 29.74 | 30.86 | 1.03 | -27.27 | 15.75 | 13.99 | 16 | 0.16 |
| 2025 | Portfolio-IC weight50 | 271828 | 15.90 | 16.46 | 0.61 | -25.27 | 15.75 | 0.14 | 24 | 0.24 |
| 2025 | Portfolio-IC weight50 | 314159 | 21.40 | 22.17 | 0.81 | -23.75 | 15.75 | 5.64 | 16 | 0.16 |

## Source Log IDs For Recovered Rows

2023 recovered from:

- pure IC seed 161803: `1FPfYwDxzkj09J5Xd2coszv-QbDl70aRv`
- pure IC seed 271828: `1NDsciGT98lqbRuBUwgoqL_hfTmp8Losi`
- LambdaRankIC seed 161803: `19sEm3d7AaWN7CXFGe1cw06DJMbzq-J6e`
- LambdaRankIC seed 271828: `1O22NmKTeh5nBFLIYNumLuPGBM3CpfU0N`
- Portfolio-IC weight50 seed 161803: `1ypQstAKq1HDD__x8fGNIRvrpAn52abS1`
- Portfolio-IC weight50 seed 271828: `1UxP6oY2FrbkBvhFXyl5ZryGTIQtjr5ks`
- Portfolio-IC weight50 seed 314159: `1RQ97X73UJBr1TtYGeU7GA9pzbSkUQhCb`

2024 recovered from:

- pure IC seed 161803: `1lxISNn79K6xhzgRAUH5bi8qxZ2TUwvif`
- pure IC seed 271828: `1g6TxBcnfb_JruEmgWHhY8VoVde_CIyLy`
- LambdaRankIC seed 161803: `1ZjNmFZ4_NnvtN44ZOtFnM3pai6YO6HOs`
- LambdaRankIC seed 271828: `12eWX2jYTND_UZxJXrTxndIg66P17BSi2`
- Portfolio-IC weight50 seed 161803: `16Hf4CEGOO9sz_pr3HF_6YDtFYXv7CgQM`
- Portfolio-IC weight50 seed 271828: `14BLBkxjQv7G9hdp-KMyUMOrgg-o3yrKx`
- Portfolio-IC weight50 seed 314159: `1-yGy7iaafRGlwfRpOifupiFk5F5hZeH2`

2025 additional recovery:

- pure IC seed 271828: `1JpsAYQdJGt6Z7fVtu3UZSirMsdxQp56m`

The 2022 rows and visible 2025 rows are from the prior consolidation/report bundle and current-plus-reused summary artifacts; this report keeps the main aggregate focused on newly trained 2022-2025 rows only.

## Decision Status

Evidence is now complete enough for a conservative objective decision on this matrix:

- Do not promote LambdaRankIC to default yet.
- Keep pure IC as the launch/default objective.
- Keep LambdaRankIC experimental, with follow-up focused on reducing turnover/cost sensitivity and testing it as a second-stage ranker or hybrid objective.
- Treat Portfolio-IC weight50 as a promising low-turnover variant for additional testing, especially around 2024-like regimes, but not as a default replacement.
