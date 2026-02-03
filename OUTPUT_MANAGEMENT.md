# Output Management and Persistence Guide

This guide explains how to use the enhanced output management features for training and backtesting with Google Drive persistence.

## Overview

The system now automatically organizes and persists all outputs including:
- Training logs
- Model checkpoints
- Predictions
- Backtest results (CSV, JSON, plots, time series)
- Configuration files

All outputs can be automatically saved to Google Drive for use in Google Colab.

## Features

### 1. Timestamped Output Directories

Hydra automatically creates timestamped directories for each run:

```
results/
├── baseline/
│   ├── 20260202_143022/          # Timestamp: YYYYMMDD_HHMMSS
│   │   ├── config.yaml            # Full configuration
│   │   ├── training_*.log         # Training logs
│   │   ├── models/                # Model checkpoints
│   │   ├── predictions/           # Per-model predictions
│   │   ├── averaged_predictions/  # Ensemble predictions
│   │   ├── backtest/              # Backtest results
│   │   └── backtest_with_costs/   # Backtest with transaction costs
│   └── 20260202_153045/           # Another run
```

### 2. Automatic Logging

All console output is automatically logged to files:

```python
# Training logs saved to: {output_dir}/training_{timestamp}.log
# Backtest logs saved to: {output_dir}/backtest/backtest_{timestamp}.log
```

### 3. Comprehensive Backtest Outputs

When using `--auto_save`, backtesting generates:

**Core Results:**
- `backtest_results.csv` - All metrics in CSV format
- `backtest_metrics.json` - Human-readable JSON
- `backtest_config.json` - Configuration used
- `summary.txt` - Text summary report

**Time Series Data:**
- `daily_returns.csv` - Daily portfolio and benchmark returns
- `cumulative_performance.csv` - Cumulative returns and drawdowns
- `monthly_performance.csv` - Monthly aggregated statistics

**Visualizations:**
- `equity_curve.png` - Portfolio vs benchmark equity curve

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

# Run training - outputs automatically saved to Google Drive
!python run_experiment.py \
    output_dir={GDRIVE_BASE} \
    experiment_name=baseline \
    training.num_epochs=100
```

### Backtesting with Auto-Save

```bash
# Automatically saves all outputs in organized structure
python evaluate_sp500.py \
    --predictions_dir results/baseline/20260202_143022/averaged_predictions \
    --auto_save

# With transaction costs (saved to separate directory)
python evaluate_sp500.py \
    --predictions_dir results/baseline/20260202_143022/averaged_predictions \
    --auto_save \
    --backtest_suffix _with_costs \
    --transaction_costs \
    --spread 10 \
    --slippage 5
```

### Finding Latest Run

```python
import glob
import os

def find_latest_run(base_dir, experiment_name):
    """Find the most recent run for an experiment."""
    experiment_dir = f"{base_dir}/{experiment_name}"
    run_dirs = sorted(glob.glob(f"{experiment_dir}/*/"))
    return run_dirs[-1] if run_dirs else None

latest = find_latest_run('results', 'baseline')
predictions = f"{latest}averaged_predictions"
```

## Command-Line Arguments

### run_experiment.py

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
```

### evaluate_sp500.py

```bash
# Required
--predictions_dir PATH          # Path to averaged_predictions folder

# Auto-save options
--auto_save                     # Enable automatic comprehensive output saving
--backtest_suffix SUFFIX        # Suffix for backtest directory (e.g., "_with_costs")

# Backtest parameters
--top_k K                       # Number of stocks in portfolio (default: 10)
--test_start DATE               # Test period start (default: 2023-01-01)
--test_end DATE                 # Test period end (default: 2023-12-31)

# Transaction costs
--transaction_costs             # Enable transaction cost modeling
--spread BPS                    # Bid-ask spread in basis points (default: 10)
--slippage BPS                  # Slippage in basis points (default: 5)

# Multiple testing adjustment
--num_tests N                   # Number of strategies tested (for Sharpe haircut)
--adjustment_method METHOD      # bhy, bonferroni, or holm

# Visualization
--plot                          # Generate equity curve plot
```

## Google Colab Workflow

See `colab_workflow.ipynb` for a complete example notebook that demonstrates:

1. **Setup:** Mount Google Drive and install dependencies
2. **Training:** Run experiments with automatic Google Drive saving
3. **Backtesting:** Evaluate with comprehensive outputs
4. **Analysis:** Load and compare multiple experiments
5. **Visualization:** Plot performance comparisons

### Quick Start

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')
GDRIVE_BASE = '/content/drive/MyDrive/MCI-GRU-Experiments'

# 2. Clone and setup
!git clone https://github.com/yourusername/MCI-GRU.git
%cd MCI-GRU
!pip install -r requirements.txt

# 3. Train
!python run_experiment.py output_dir={GDRIVE_BASE} experiment_name=baseline

# 4. Find latest run
import glob
runs = sorted(glob.glob(f"{GDRIVE_BASE}/baseline/*/"))
latest = runs[-1]

# 5. Backtest with auto-save
!python evaluate_sp500.py \
    --predictions_dir {latest}averaged_predictions \
    --auto_save \
    --plot

# 6. View results
!cat {latest}backtest/summary.txt
```

## Output Files Reference

### Training Outputs

| File | Description |
|------|-------------|
| `config.yaml` | Full Hydra configuration (merged from all sources) |
| `training_{timestamp}.log` | Complete training logs |
| `models/model_{i}_best.pt` | Best model checkpoint for each run |
| `predictions/model_{i}/{date}.csv` | Per-model daily predictions |
| `averaged_predictions/{date}.csv` | Ensemble averaged predictions |

### Backtest Outputs (with --auto_save)

| File | Description |
|------|-------------|
| `backtest_results.csv` | All metrics in CSV format |
| `backtest_metrics.json` | Metrics in JSON format |
| `backtest_config.json` | Configuration used for backtest |
| `summary.txt` | Human-readable summary report |
| `daily_returns.csv` | Daily portfolio/benchmark returns |
| `cumulative_performance.csv` | Cumulative returns and drawdowns |
| `monthly_performance.csv` | Monthly aggregated statistics |
| `equity_curve.png` | Portfolio vs benchmark plot |
| `backtest_{timestamp}.log` | Complete backtest logs |
| `model_{i}_results.json` | Individual model results (if multi-model) |

## Comparing Experiments

```python
import pandas as pd
import os

def load_experiment_results(base_dir, experiment_name):
    """Load backtest results from an experiment."""
    latest_run = find_latest_run(base_dir, experiment_name)
    if not latest_run:
        return None
    
    results_file = f"{latest_run}backtest/backtest_results.csv"
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df['experiment'] = experiment_name
        return df
    return None

# Compare multiple experiments
experiments = ['baseline', 'with_vix', 'lookback_20']
results = [load_experiment_results('results', exp) for exp in experiments]
comparison = pd.concat([r for r in results if r is not None])

print(comparison[['experiment', 'ARR', 'ASR', 'MDD', 'CR', 'IR']])
```

## Tips

1. **Use descriptive experiment names** for easy identification
2. **Enable auto_save** to get comprehensive backtest outputs
3. **Use backtest_suffix** to save multiple backtest configurations side-by-side
4. **Keep Google Drive synced** to access results from any device
5. **Use the Colab notebook** for a complete workflow example

## Troubleshooting

### "No predictions found"
- Check that the training completed successfully
- Verify the path to `averaged_predictions` directory
- Use `find_latest_run()` helper to locate outputs

### "Permission denied" on Google Drive
- Ensure Drive is mounted: `drive.mount('/content/drive')`
- Check you have write permissions to the target directory

### "Logs not appearing"
- Logs are written to files, not just console
- Check `{output_dir}/training_*.log` and `backtest/backtest_*.log`

## Examples

### Run multiple experiments with different configs

```bash
# Baseline
python run_experiment.py output_dir=results experiment_name=baseline

# With VIX
python run_experiment.py output_dir=results +experiment=with_vix

# Different lookback
python run_experiment.py output_dir=results experiment_name=lookback_20 model.his_t=20

# Then backtest all
for exp in baseline with_vix lookback_20; do
    latest=$(ls -td results/$exp/*/ | head -1)
    python evaluate_sp500.py \
        --predictions_dir ${latest}averaged_predictions \
        --auto_save
done
```

### Hyperparameter sweep

```bash
# Sweep over lookback periods
python run_experiment.py \
    --multirun \
    output_dir=results \
    experiment_name=lookback_sweep \
    model.his_t=5,10,15,20

# Each run gets its own subdirectory: results/lookback_sweep/run_0/, run_1/, etc.
```

## Integration with Existing Code

The new features are **backward compatible**. Existing scripts will work without changes:

```bash
# Old way still works
python run_experiment.py  # Outputs to results/baseline/

# New way adds features
python run_experiment.py --auto_save  # Adds logging and organized outputs
```

## Further Reading

- **Hydra Documentation:** https://hydra.cc/
- **Config Management:** See `configs/` directory structure
- **Colab Notebook:** `colab_workflow.ipynb` for complete example
