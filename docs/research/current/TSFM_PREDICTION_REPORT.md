# TSFM-Style Prediction Evaluation Report

Issue #22 adds a saved-prediction report for reading statistical forecast quality next
to the IC and top-k portfolio diagnostics MCI-GRU already uses.

The report runs on existing `averaged_predictions/*.csv` files. It does not retrain
models and does not add external model dependencies.

> **Method changed, 2026-10-24 (#334).** #342 (pull request #352, `242ffb5`) moved
> this report onto the evaluation core. A report written before that date can
> differ on the points below. No saved report was re-scored for this note (#334
> decision E10).
>
> - Realized returns come from the panel's own session axis
>   (`mci_gru.evaluation.outcomes.realized_returns`), so a missing bar no longer
>   moves an endpoint the way the per-stock row shift did. A missing exit books the
>   last close in the hold.
> - The rows are the primary model's: predictions inner-joined to realized returns.
>   Baselines are left-joined onto those rows, so adding a baseline no longer
>   changes the primary's numbers. Each model is scored on its own finite rows.
>   This replaces the all-model inner join described under Alignment Rules below.
> - Top-k portfolio numbers come from the in-run summary's staggered tranche book
>   (`evaluate_run`, #336) and need the panel's closes. The older matrix-only Sharpe
>   and cumulative return treated overlapping 4-session returns as daily, and have
>   been removed.
> - `oos_r2_zero`, `direction_accuracy` and `macro_f1` still depend on the scale and
>   level of the scores. Under the IC loss those are arbitrary, so read these three
>   only for models trained to forecast returns. The #334 audit also found that
>   macro-F1 scores an empty zero class as 0, so a perfect sign forecaster reads
>   0.667; that remains as it was.

## Command

```bash
python scripts/evaluation/tsfm_prediction_report.py \
  --predictions-dir results/baseline/20260517_120000/averaged_predictions \
  --market-data data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv \
  --output-dir results/baseline/20260517_120000/tsfm_prediction_report \
  --label-t 5
```

Optional external baselines can be included as one CSV file or a directory of daily
prediction CSVs. Each baseline must use the same simple schema as MCI-GRU saved
predictions: `dt`, `kdcode`, `score`.

```bash
python scripts/evaluation/tsfm_prediction_report.py \
  --predictions-dir path/to/averaged_predictions \
  --market-data path/to/market.csv \
  --output-dir path/to/report \
  --label-t 5 \
  --baseline zero_like=path/to/baseline_predictions.csv
```

## Outputs

The output directory contains:

| File | Purpose |
|------|---------|
| `tsfm_prediction_report.json` | Machine-readable metrics and yearly decay rows |
| `tsfm_prediction_report.md` | Human-readable report template |
| `tsfm_aligned_predictions.csv` | Exact date/ticker rows used for all model comparisons |

## Metrics

- `oos_r2_zero`: out-of-sample R2 against a zero-return forecast. Values above
  zero mean the model has lower squared forecast error than predicting zero.
- `direction_accuracy`: share of rows where predicted and realized return signs
  match.
- `macro_f1`: unweighted F1 over negative, zero, and positive return signs.
- `avg_ic`, `avg_spearman_corr`, and top-k return fields: existing MCI-GRU
  ranking and portfolio diagnostics, computed on the same aligned rows.
- yearly decay: the same metrics grouped by prediction calendar year.

Read R2, direction accuracy, and macro-F1 beside IC and portfolio metrics. A model
that improves squared-error forecast quality but loses ranking IC, top-k return, or
drawdown discipline should not be treated as a better trading candidate by default.

## Alignment Rules

The comparison frame is an inner join on `dt` and `kdcode` across MCI-GRU
predictions, realized returns, and every optional baseline. That means all reported
models are scored on the same prediction dates and tradable names. Rows with
non-finite scores or realized returns are excluded before metrics are computed.

Realized returns are derived from the market CSV with the training-label convention:

```text
forward_return_label_t = close_t+label_t / close_t+1 - 1
```

Use the same `--label-t` as the saved prediction run.
