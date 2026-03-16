# Output Management and Persistence Guide

This guide explains how training outputs are organized and persisted, including Google Drive for Colab workflows.

## Overview

`run_experiment.py` organizes all outputs under timestamped directories. Backtesting is a separate workflow (see `tests/backtest_sp500.py`) and is not part of the training run.

## MLflow Integration

The project also supports **optional local MLflow tracking**.
MLflow is **disabled by default** -- enable it with `tracking.enabled=true`.
It complements the filesystem outputs documented here rather than replacing them.
See `docs/MLFLOW_TRACKING.md` for setup, tracking behavior, and UI usage.

## Output Directory Structure

Training outputs go to `{output_dir}/{experiment_name}/{timestamp}/`:

```
results/
├── baseline/
│   ├── 20260202_143022/
│   │   ├── config.yaml            # Full Hydra configuration
│   │   ├── training_*.log         # Training logs
│   │   ├── run_metadata.json      # Normalization stats, feature cols, kdcode_list (for inference)
│   │   ├── graph_data.pt          # Edge index/weight (for inference)
│   │   ├── checkpoints/           # Model checkpoints
│   │   │   ├── model_0_best.pth
│   │   │   └── ...
│   │   ├── predictions_model_0/   # Per-model predictions
│   │   ├── predictions_model_1/
│   │   ├── ...
│   │   └── averaged_predictions/  # Ensemble predictions
│   └── 20260202_153045/
```

## Training Output Files

| File | Description |
|------|-------------|
| `config.yaml` | Full Hydra configuration (merged from all sources) |
| `training_{timestamp}.log` | Complete training logs |
| `run_metadata.json` | Normalization means/stds, feature_cols, kdcode_list, his_t, label_t (required for inference) |
| `graph_data.pt` | Edge index and edge weight tensors (required for inference) |
| `checkpoints/model_{i}_best.pth` | Best checkpoint for each model in the ensemble |
| `predictions_model_{i}/{date}.csv` | Per-model daily predictions |
| `averaged_predictions/{date}.csv` | Ensemble averaged predictions |

## Usage

### Local Training

```bash
# Basic usage (outputs to results/{experiment_name}/{timestamp}/)
python run_experiment.py

# Custom output directory
python run_experiment.py output_dir=/path/to/outputs

# Different experiment
python run_experiment.py +experiment=with_vix
```

### Google Colab Training

```python
from google.colab import drive
drive.mount('/content/drive')

GDRIVE_BASE = '/content/drive/MyDrive/MCI-GRU-Experiments'

# Run training - outputs saved to Google Drive
!python run_experiment.py \
    output_dir={GDRIVE_BASE} \
    experiment_name=baseline \
    training.num_epochs=100
```

### Finding Latest Run

```python
import glob

def find_latest_run(base_dir, experiment_name):
    experiment_dir = f"{base_dir}/{experiment_name}"
    run_dirs = sorted(glob.glob(f"{experiment_dir}/*/"))
    return run_dirs[-1] if run_dirs else None

latest = find_latest_run('results', 'baseline')
predictions = f"{latest}averaged_predictions"
```

## Command-Line Arguments (run_experiment.py)

```bash
# Output directory (supports Google Drive paths)
python run_experiment.py output_dir=/content/drive/MyDrive/MCI-GRU-Experiments

# Experiment name (creates subdirectory)
python run_experiment.py experiment_name=my_experiment

# Override any config parameter
python run_experiment.py model.his_t=20 training.batch_size=64

# Use experiment presets
python run_experiment.py +experiment=with_vix
python run_experiment.py +data=russell1000

# Sweep over multiple values
python run_experiment.py --multirun model.his_t=5,10,15,20
```

## Google Colab Workflow

See `colab_workflow.ipynb` for a complete example. Quick start:

```python
from google.colab import drive
drive.mount('/content/drive')
GDRIVE_BASE = '/content/drive/MyDrive/MCI-GRU-Experiments'

!python run_experiment.py output_dir={GDRIVE_BASE} experiment_name=baseline

import glob
runs = sorted(glob.glob(f"{GDRIVE_BASE}/baseline/*/"))
latest = runs[-1]
```

## Paper Trade Pipeline

`paper_trade/scripts/infer.py` loads checkpoints, `run_metadata.json`, and `graph_data.pt` from a run directory. `portfolio.py`, `track.py`, `report.py` handle portfolio decisions and reporting.

## Troubleshooting

- **No predictions:** Verify training completed and path to `averaged_predictions`; use `find_latest_run()` to locate outputs
- **Permission denied on Google Drive:** Ensure Drive is mounted and you have write access
- **Logs not appearing:** Check `{output_dir}/training_*.log` (logs go to files, not just console)

## Further Reading

- Hydra: https://hydra.cc/
- Config management: `configs/` directory
- Colab notebook: `colab_workflow.ipynb`
- Backtesting: `tests/backtest_sp500.py` (separate workflow)
