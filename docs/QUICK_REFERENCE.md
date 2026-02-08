# Quick Reference: Output Persistence

## Training

```bash
# Local
python run_experiment.py

# Google Drive (Colab)
python run_experiment.py output_dir=/content/drive/MyDrive/MCI-GRU-Experiments
```

**Outputs:** `{output_dir}/{experiment_name}/{timestamp}/`
- `config.yaml` - Full configuration
- `training_*.log` - Training logs
- `models/` - Model checkpoints
- `averaged_predictions/` - Ensemble predictions

## Backtesting

```bash
# Basic (old way)
python evaluate_sp500.py --predictions_dir path/to/predictions

# Enhanced (new way)
python evaluate_sp500.py \
    --predictions_dir path/to/predictions \
    --auto_save \
    --plot
```

**Outputs:** `{prediction_dir}/../backtest/`
- `backtest_results.csv` - Metrics
- `daily_returns.csv` - Time series
- `equity_curve.png` - Plot
- `summary.txt` - Report

## Transaction Costs Comparison

```bash
# Without costs
python evaluate_sp500.py \
    --predictions_dir path/to/predictions \
    --auto_save

# With costs (separate directory)
python evaluate_sp500.py \
    --predictions_dir path/to/predictions \
    --auto_save \
    --backtest_suffix _with_costs \
    --transaction_costs
```

**Result:** Two directories side-by-side for easy comparison
- `backtest/` - No transaction costs
- `backtest_with_costs/` - With transaction costs

## Finding Latest Run

```python
import glob

# Find latest run
runs = sorted(glob.glob("results/baseline/*/"))
latest = runs[-1]
predictions = f"{latest}averaged_predictions"
```

## Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

GDRIVE = '/content/drive/MyDrive/MCI-GRU-Experiments'

# Train
!python run_experiment.py output_dir={GDRIVE} experiment_name=baseline

# Find and backtest
import glob
latest = sorted(glob.glob(f"{GDRIVE}/baseline/*/"))[-1]
!python evaluate_sp500.py \
    --predictions_dir {latest}averaged_predictions \
    --auto_save \
    --plot
```

## Key Arguments

### run_experiment.py
- `output_dir=PATH` - Where to save (supports Google Drive)
- `experiment_name=NAME` - Experiment identifier
- `+experiment=PRESET` - Use config preset (with_vix, etc.)

### evaluate_sp500.py  
- `--predictions_dir PATH` - Path to predictions
- `--auto_save` - Save comprehensive outputs
- `--backtest_suffix SUFFIX` - Add suffix to output dir
- `--transaction_costs` - Enable TC modeling
- `--plot` - Generate equity curve

## Output Structure

```
output_dir/
└── experiment_name/
    └── YYYYMMDD_HHMMSS/           # Timestamp
        ├── config.yaml
        ├── training_*.log
        ├── models/
        ├── averaged_predictions/
        ├── backtest/
        │   ├── backtest_results.csv
        │   ├── daily_returns.csv
        │   ├── equity_curve.png
        │   └── summary.txt
        └── backtest_with_costs/
            └── ... same structure ...
```

## See Also

- `OUTPUT_MANAGEMENT.md` - Full documentation
- `colab_workflow.ipynb` - Complete Colab example
- `configs/` - Configuration files
