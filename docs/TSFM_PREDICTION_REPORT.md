# TSFM-Style Prediction Evaluation Report

Issue #22 adds a saved-prediction report for reading statistical forecast quality next
to the IC and top-k portfolio diagnostics MCI-GRU already uses.

The report runs on existing `averaged_predictions/*.csv` files. It does not retrain
models and does not add external model dependencies.

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
