# PIT Masked-Panel 2022-2025 Full Run Report

Date prepared: 2026-05-16

Primary run:
[MCI-GRU-Ablations/pit_masked_panel_2022_2025/20260514_043539](https://drive.google.com/drive/folders/1p1F2NqY5C6ISBzjm7-JBkbvsE4K2E2LF)

This report reviews the full PIT masked-panel notebook run that followed
`docs/handoffs/2026-05-15-frozen-default-pit-masked-panel.md`.

## Executive Summary

The run clears the main methodological hurdle. It executed the frozen default
recipe in full mode, with global regime data enabled, across 2022, 2023, 2024,
and 2025. All four rolling yearly trainings completed, all PIT breadth checks
passed, and every expected prediction file matched the same-day PIT scoreable
mask. This is the first strong evidence that the true masked-panel workflow can
run end to end without collapsing to a survivor-only or continuous-stayer
universe.

Performance is promising but not yet a final production claim. The PIT-aware
daily backtests show strong positive excess returns in 2023, 2024, and 2025,
with a weak 2022. Compounded across the four yearly test windows, the model
returns about 91.9% versus 20.0% for the benchmark, but the current backtest
artifact reports `transaction_costs_enabled=False` and `rank_gate_enabled=False`.
Year-level significance flags are also false for all four years. Treat this as
a high-value confirmation result that deserves a cost-aware pooled follow-up,
not as a finished deployable performance proof.

## Artifact Set

Reviewed artifacts:

- [pit_masked_panel_2022_2025_summary.md](https://drive.google.com/file/d/13dADHk_OgSP5fZtRdn4olCPV09l1nhd5/view?usp=drivesdk)
- [pit_masked_panel_manifest.json](https://drive.google.com/file/d/1NQUjNLslGbz6rWhM-V6B7dYxq0gpSg8N/view?usp=drivesdk)
- [training_results.csv](https://drive.google.com/file/d/1UsMIJ_fLegj3AAIVQKtc_SlS12aBiHfV/view?usp=drivesdk)
- [pit_breadth_summary.csv](https://drive.google.com/file/d/1mL7jOb-wMjF20WNaEdKtphfv5CfZFSg-/view?usp=drivesdk)
- [prediction_count_checks.csv](https://drive.google.com/file/d/10R_g0DebCwX4UsfGOztbTO1R6uC5slAl/view?usp=drivesdk)
- [backtest_results.csv](https://drive.google.com/file/d/1goMfXTd9A5qhe3MsAI44rNNSMH3krLvj/view?usp=drivesdk)
- Local membership progression audit:
  [pit_membership_progression_audit.md](audits/pit_membership_progression_audit.md)
  and
  [pit_membership_progression_snapshot_counts.csv](audits/pit_membership_progression_snapshot_counts.csv)
- Prior context:
  [pit_universe_validation_summary.md](https://drive.google.com/file/d/1zljCmW7whhRmfw6_iV-vgAMh-R7_sqr8/view?usp=drivesdk)

The generated summary markdown still contains a stale footer saying smoke-run
backtest metrics are mechanics evidence. The manifest and CSVs contradict that
footer: this run used `smoke_mode=false`, trained 20 models per year, and used
the full frozen budget. The footer should be treated as a notebook text bug.

## Run Identity

| Field | Value |
|---|---|
| Run tag | `20260514_043539` |
| Branch | `codex/pit-universe-validation` |
| Recipe | `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1` |
| Smoke mode | `False` |
| Global regime | `True` |
| FRED key present | `True` |
| PIT mode | `masked_panel` |
| PIT min scoreable stocks | `450` |
| PIT breadth policy | `error` |
| Market file hash | `84e1f3f2b79a798246e001e17a372c8daf8bcfc658873ad3d352a99ad993840f` |

The budget matches the frozen default:

- `training.num_models=20`
- `training.num_epochs=100`
- `training.early_stopping_patience=15`
- `training.learning_rate=5e-5`
- `training.loss_type=ic`
- `training.label_type=returns`
- `training.selection_metric=val_ic`
- `model.label_t=5`
- `graph.drop_edge_p=0.1`
- current-only strict global regime features

## Training Completion

All four yearly rolling runs completed successfully.

| Test year | Status | Models trained | Elapsed minutes | Union axis |
|---:|---|---:|---:|---:|
| 2022 | OK | 20 | 19.48 | 696 |
| 2023 | OK | 20 | 18.75 | 669 |
| 2024 | OK | 20 | 17.55 | 652 |
| 2025 | OK | 20 | 24.77 | 646 |

Total reported training time was about 80.6 minutes, averaging about 20.1
minutes per yearly experiment. The union axis declines from 696 to 646 across
the rolling windows, which is consistent with date-window-specific PIT union
membership rather than a fixed survivor panel.

Mean best validation IC by year:

| Test year | Mean best val IC |
|---:|---:|
| 2022 | 0.0100 |
| 2023 | 0.0163 |
| 2024 | 0.0034 |
| 2025 | 0.0254 |

The average of those yearly means is 0.0138. The validation ICs are positive in
all four years, but they do not line up mechanically with realized backtest
strength: 2024 has the lowest mean best validation IC and the strongest
backtest, while 2022 has positive validation IC and the weakest backtest. That
argues for reading validation IC as a checkpoint-selection signal, not as a
standalone forecast of yearly portfolio performance.

## PIT Breadth And Prediction Integrity

The masked-panel checks are the cleanest part of this run.

Across all train, validation, and test splits:

- `active_count.below_threshold=0`
- `feature_ready_count.below_threshold=0`
- `loss_count.below_threshold=0`
- `scoreable_count.below_threshold=0`

Test-set scoreable breadth stayed around the intended live S&P 500 opportunity
set:

| Test year | Test dates | Scoreable min | Scoreable median | Scoreable max |
|---:|---:|---:|---:|---:|
| 2022 | 237 | 502 | 503 | 506 |
| 2023 | 237 | 501 | 503 | 504 |
| 2024 | 239 | 501 | 503 | 503 |
| 2025 | 238 | 502 | 503 | 503 |

Prediction export also matched the PIT masks exactly:

| Test year | Expected files | Matched files | Missing files | Count mismatches | Prediction rows |
|---:|---:|---:|---:|---:|---:|
| 2022 | 237 | 237 | 0 | 0 | 119,366 |
| 2023 | 237 | 237 | 0 | 0 | 119,190 |
| 2024 | 239 | 239 | 0 | 0 | 120,186 |
| 2025 | 238 | 238 | 0 | 0 | 119,704 |

Total prediction rows across the four test windows were 478,446 across 951
daily prediction files.

This is materially better than the older PIT validation run, where the
`pit_universe` control reported only 305 stocks for 2022 and showed sharply
negative PIT performance. The new masked-panel run preserves roughly 500
scoreable names per day while still enforcing point-in-time tradability.

## Backtest Results

| Test year | Total return | Benchmark return | Excess return | ARR | ASR | MDD | P-value | Significant |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2022 | -20.65% | -6.45% | -14.20% | -21.97% | -0.542 | -30.25% | 0.6009 | False |
| 2023 | 25.90% | 7.12% | 18.78% | 28.02% | 0.957 | -18.82% | 0.3552 | False |
| 2024 | 45.36% | 11.73% | 33.63% | 48.84% | 1.956 | -18.42% | 0.0579 | False |
| 2025 | 32.15% | 7.14% | 25.01% | 34.67% | 1.048 | -22.96% | 0.3106 | False |

Cross-year read:

- Three of four years have positive absolute and excess returns.
- 2022 is the clear failure year, underperforming the benchmark by 14.20%.
- 2024 is the strongest year, with 45.36% total return, 33.63% excess return,
  and ASR 1.956.
- Average annualized return across the four yearly rows is 22.39%.
- Average ASR is 0.855.
- Worst yearly max drawdown is -30.25%.
- Compounded model return over the four yearly test rows is about 91.9%;
  compounded benchmark return is about 20.0%.

The positive multi-year shape is encouraging, especially because it survives a
strict PIT masked-panel candidate set. Still, the reported `is_significant`
field is `False` in every year. The best single p-value is 0.0579 in 2024,
which is suggestive but below the repository's usual threshold for a statistical
claim.

## Interpretation Against Prior Evidence

The April 30 ablation report identified the frozen recipe shape as the strongest
deployable-looking candidate: static threshold shuffled graph, pure IC loss, raw
5-day return labels, current-only regime features, 20-model ensemble, and
moderate edge dropout. This run validates that same recipe under the more
demanding true PIT masked-panel setup.

The key update is methodological rather than just performance-related:

1. The full recipe runs with `data.pit_universe_mode=masked_panel`.
2. The run uses the LSEG PIT-union market file whose hash matches the prior PIT
   data report.
3. PIT breadth stays above 500 scoreable names on normal test days.
4. Prediction files match same-day tradable masks exactly.
5. The earlier 305-stock PIT-collapse problem is no longer present.

So the result meaningfully raises confidence that the model's signal is not
just an artifact of complete-stock filtering or a current-survivor panel. It
does not yet settle cost-aware live viability.

## Caveats

Do not overclaim this run as final production evidence yet.

- Transaction costs were disabled in the reviewed backtest artifact.
- Rank-drop gating was disabled.
- Year-by-year statistical significance is absent.
- The report does not yet include a pooled daily significance test for this new
  masked-panel run.
- There is no repeated-seed masked-panel replication yet.
- The backtest benchmark is available in the CSV, but this report did not audit
  the benchmark construction beyond the stored output fields.
- The summary markdown's stale smoke-run footer should be fixed in the notebook
  generator to avoid future confusion.

## Recommendation

Promote this run from "mechanics validation" to "successful full PIT
confirmation candidate." The model passed the hard PIT integrity checks and
produced economically interesting returns in three of four years.

Before making deployment or paper-trade changes, run a follow-up evaluation on
the same saved predictions with:

1. transaction costs enabled,
2. the normal rank-drop gate enabled,
3. pooled daily excess-return significance across 2022-2025,
4. Newey-West or block-bootstrap confidence intervals,
5. year-by-year turnover and concentration diagnostics,
6. a 2022 failure drilldown by month, sector, and market regime,
7. at least one repeated-seed PIT masked-panel replication.

If the cost-aware pooled test remains positive and the 2022 drawdown is
explainable rather than structural, this frozen recipe becomes the best current
candidate for a true PIT production-style baseline.
