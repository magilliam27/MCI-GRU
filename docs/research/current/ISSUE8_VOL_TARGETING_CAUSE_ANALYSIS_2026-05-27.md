# Issue #8 Volatility Targeting Cause Analysis

Date: 2026-05-27

## Question

Why did the volatility-targeted variant produce a severe 2023 drawdown versus
baseline while helping 2024 and 2025?

This follow-up decomposes selected holdings by ex ante Harvey-style volatility
features and compares baseline-only names against vol-targeted-only
replacements across 2022-2025.

## Diagnostic Artifacts

- Local analysis directory: `.codex_tmp/issue8_vol_cause_diag/`
- Inputs:
  - Baseline daily holdings/returns from the seed-314159 PIT repeated-seed
    extraction under `.codex_tmp/pit_option_a_extract_20260520_183538/`
  - Vol-targeted daily holdings/returns downloaded from the issue #8 Drive runs
  - Market panel: `data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv`
- Generated files:
  - `holding_vol_exposure_summary_by_year.csv`
  - `base_only_minus_vol_only_deltas.csv`
  - `exclusive_holdings_with_vol_features.csv`
  - `exclusive_return_robustness.csv`
  - `top_baseline_only_winners_with_vol_features.csv`

## Feature Surface

The issue #8 variant adds nine Harvey-style inputs on top of the existing
realized-volatility feature set:

- EWM annualized volatility at half-lives `20`, `60`, and `90`
- Target-vol scale proxies at the same half-lives, clipped to `[0.25, 4.0]`
- Short/long vol-change ratio: `vol_hl20 / vol_hl90 - 1`
- Short-horizon vol-of-vol
- `ret21_lag2 * scale_hl20`

These are model inputs, not direct portfolio exposure scaling. The severe 2023
effect therefore comes from rank displacement, not from mechanically shrinking
positions.

## Cross-Year Holding Exposure

| year | category | n holdings | mean stock return | mean EWM vol 20 | mean scale 20 | low-scale frac | high-vol frac | mean ret21 lag2 | mean ret21 x scale |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | baseline only | 1084 | -0.13% | 49.02% | 0.263 | 100.00% | 65.31% | -10.54% | -2.69% |
| 2022 | vol only | 1082 | -0.17% | 56.87% | 0.269 | 98.98% | 80.59% | -8.44% | -2.21% |
| 2023 | baseline only | 2111 | 0.26% | 37.52% | 0.328 | 92.42% | 34.44% | 0.06% | 0.06% |
| 2023 | vol only | 2111 | 0.06% | 25.48% | 0.474 | 59.12% | 14.54% | -3.64% | -0.80% |
| 2024 | baseline only | 2143 | 0.06% | 35.42% | 0.353 | 86.14% | 29.58% | -7.76% | -2.34% |
| 2024 | vol only | 2143 | 0.07% | 17.48% | 0.661 | 9.33% | 3.13% | -1.10% | -0.40% |
| 2025 | baseline only | 1170 | 0.15% | 29.29% | 0.380 | 85.90% | 14.10% | 1.23% | 0.50% |
| 2025 | vol only | 1171 | 0.23% | 57.51% | 0.287 | 92.91% | 81.30% | 4.80% | 1.13% |

## Why 2023 Is Different

2023 is the only year where the names removed by the vol-targeted model had a
large positive return advantage over the replacement names.

| year | base-only minus vol-only return | base-only minus vol-only EWM vol 20 | base-only minus vol-only scale 20 | base-only minus vol-only high-vol frac | base-only minus vol-only ret21 lag2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.03 pp | -7.85 pp | -0.007 | -15.28 pp | -2.09 pp |
| 2023 | 0.20 pp | 12.05 pp | -0.146 | 19.90 pp | 3.69 pp |
| 2024 | -0.00 pp | 17.94 pp | -0.308 | 26.46 pp | -6.66 pp |
| 2025 | -0.08 pp | -28.22 pp | 0.093 | -67.20 pp | -3.57 pp |

Interpretation:

- In 2023, the baseline-only book was higher-volatility and higher-return than
  the vol-only replacements. That is the dangerous pattern.
- In 2024, the baseline-only book was also higher-volatility, but it did not
  have a return premium. Replacing it with lower-volatility names reduced churn
  and helped net results.
- In 2025, the vol-targeted book did not behave like a blanket low-volatility
  filter; it selected higher-volatility, stronger-momentum names and those names
  outperformed.

So the issue is not simply "volatility targeting hates high volatility." The
feature creates a different ranking surface. In 2023, that surface moved the
model away from the rebound winners that baseline happened to catch.

## Outlier Check

The 2023 result is amplified by a few large winners, but it is not only a single
outlier story.

| year | category | mean | median | 5-95% trimmed mean | 1% winsor mean | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | baseline only | 0.26% | 0.19% | 0.18% | 0.22% | 83.82% |
| 2023 | vol only | 0.06% | 0.07% | 0.06% | 0.06% | 15.62% |

Even after trimming the 5th and 95th percentiles, baseline-only names retain a
`0.12 pp` per-holding return advantage over the vol-only replacements.

## Examples From 2023

| entry date | kdcode | stock return | baseline rank | EWM vol 20 | scale 20 | ret21 lag2 | ret21 x scale |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-03-13 | FRC.N^E23 | 83.82% | 6 | 36.03% | 0.278 | -19.17% | -5.32% |
| 2023-05-24 | NVDA.OQ | 27.52% | 2 | 40.62% | 0.250 | 15.35% | 3.84% |
| 2023-04-27 | FRC.N^E23 | 14.61% | 5 | 197.94% | 0.250 | 27.69% | 6.92% |
| 2023-02-22 | NVDA.OQ | 13.24% | 2 | 58.80% | 0.250 | 26.62% | 6.65% |
| 2023-01-25 | TSLA.OQ | 12.67% | 3 | 74.96% | 0.250 | -10.98% | -2.74% |
| 2023-05-15 | DISH.OQ^A24 | 11.84% | 7 | 66.84% | 0.250 | -29.54% | -7.39% |
| 2023-03-29 | FRC.N^E23 | 11.22% | 1 | 257.23% | 0.250 | -89.97% | -22.49% |
| 2023-08-03 | AMZN.OQ | 10.58% | 10 | 28.56% | 0.350 | 4.52% | 1.58% |

Several of the missed 2023 winners sit at or near the `scale_hl20` floor. The
feature family therefore exposes the model to an explicit "this is very risky"
signal exactly where 2023 rewarded rebound/high-convexity exposure.

## Current Best Explanation

The most likely cause is a regime-flip interaction:

1. The 2023 model is trained through the 2022 bear/high-volatility environment.
2. The new volatility-targeting inputs make high-volatility and scaled-momentum
   state more salient to the model.
3. In 2023, many of the strongest winners were high-volatility rebound or
   high-convexity names.
4. The vol-targeted model displaced those names with lower-volatility names,
   which lowered risk/cost but also removed the upside that drove the baseline
   year.

This explains why the effect is severe in 2023 but not uniform:

- 2024: the same low-volatility displacement did not sacrifice a return premium,
  so lower churn/cost helped.
- 2025: the vol-targeted model selected higher-volatility winners, so the
  feature was not mechanically anti-volatility.
- 2022: both books were high-volatility and negative; the feature did not solve
  the broader regime-stress problem.

## Next Tests

Promotion should wait. The next experiments should be 2023-focused ablations:

1. Keep raw EWM volatility features, remove target-vol scale and interaction
   features.
2. Keep target-vol scale features, remove raw EWM volatility levels.
3. Remove only `ret21_lag2 * scale_hl20` to test whether scaled momentum is the
   rank-displacement channel.
4. Sweep `scale_clip` from `[0.25,4.0]` to less aggressive floors such as
   `[0.50,2.0]` and `[0.75,1.5]`.
5. Run a repeated-seed 2023 panel to check whether this is seed-specific or a
   stable regime-flip failure.

The immediate acceptance criterion should not be "beats 2023 baseline"; that
bar may be outlier-heavy. A safer criterion is that the ablated variant avoids a
large gross-return collapse while preserving the 2024/2025 turnover benefit.
