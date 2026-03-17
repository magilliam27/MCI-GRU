# MLflow Tracking Guide

This project supports **optional local MLflow tracking** for training and backtesting.

## Philosophy

MLflow is **additive**:

- it does **not** replace the existing Hydra output folders
- it does **not** change the files used by paper-trade inference
- it does **not** move checkpoints, predictions, or run metadata

The existing local filesystem outputs remain the source of truth for downstream workflows.
MLflow adds searchable experiment history, metrics, and artifact browsing on top.

## Default behavior

MLflow is **disabled by default**.

To enable it for a training run:

```bash
python run_experiment.py tracking.enabled=true
```

## Training with MLflow

### Basic local usage

```bash
python run_experiment.py tracking.enabled=true
```

This will:

1. keep writing the normal local run directory under `results/...`
2. create/update a local MLflow store under `mlruns/`
3. log one parent training run plus child runs for each ensemble member

### Useful overrides

```bash
python run_experiment.py \
  tracking.enabled=true \
  tracking.experiment_name=mci-gru-sp500 \
  tracking.run_name=baseline-seed42
```

### Artifact options

```bash
python run_experiment.py \
  tracking.enabled=true \
  tracking.log_artifacts=true \
  tracking.log_checkpoints=true \
  tracking.log_predictions=false
```

Recommended default:

- `log_artifacts=true`
- `log_checkpoints=true`
- `log_predictions=false`

Prediction folders can contain many CSVs, so they are off by default to avoid unnecessary duplication.

## Backtesting with MLflow

If a training run was tracked, the run directory contains `mlflow_run.json`.
A backtest can auto-link itself to the tracked training experiment:

```bash
python -m tests.backtest_sp500 \
  --predictions_dir results/baseline/20260310_010203/averaged_predictions \
  --auto_save
```

When that metadata file is present, the backtest creates a **linked child run** under the original training run.

### Standalone backtest logging

```bash
python -m tests.backtest_sp500 \
  --predictions_dir path/to/averaged_predictions \
  --auto_save \
  --enable_mlflow \
  --mlflow_experiment_name standalone-backtests
```

## Launching the local UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open `http://127.0.0.1:5000`.

## What gets tracked

### Training

- Flattened run configuration
- Parent training run metadata
- Per-epoch child-run metrics: `train_loss`, `val_loss`, `best_val_loss`
- Final child-run metrics: `best_val_loss`, `final_train_loss`, `epochs_trained`
- Selected run artifacts: `config.yaml`, `run_metadata.json`, `graph_data.pt`, training logs, checkpoints (if enabled)

### Backtest

- Backtest config
- Summary metrics: `ARR`, `AVoL`, `MDD`, `ASR`, `CR`, `IR`, `MSE`, `MAE`
- Saved outputs: `backtest_metrics.json`, `summary.txt`, `equity_curve.png`, etc.

## Manual logging vs autologging

This repository uses **manual MLflow logging**, not framework autologging.
The project has a custom training loop, ensemble child runs, custom output directories,
and downstream scripts that depend on specific local files. Manual logging gives precise
control over what is logged, when, and which run it belongs to.
