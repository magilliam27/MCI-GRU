# Quick Reference: Output Persistence

## Training

```bash
# Local (MLflow on by default; LSEG default data source in sp500.yaml)
python run_experiment.py

# Smoke: CSV data + short run + no MLflow
python run_experiment.py training.num_epochs=2 training.num_models=1 data.source=csv tracking.enabled=false

# Custom output directory
python run_experiment.py output_dir=/path/to/output
```

**Outputs:** `{output_dir}/{experiment_name}/{timestamp}/`
- `config.yaml` - Full configuration
- `training_*.log` - Training logs
- `run_metadata.json` - Norm stats, features, **`data_file_sha256`** (when file exists), etc.
- `graph_data.pt` - Static edge tensors for inference
- `checkpoints/` - Model checkpoints
- `averaged_predictions/` - Ensemble predictions

## Backtesting

Backtesting is done via `tests/backtest_sp500.py` and `tests/backtest_sp500_daily.py`:

```bash
# Basic backtest
python tests/backtest_sp500.py --predictions_dir path/to/averaged_predictions

# With auto-save and plot
python tests/backtest_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --auto_save \
    --plot

# With transaction costs (separate directory)
python tests/backtest_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --auto_save \
    --backtest_suffix _with_costs \
    --transaction_costs
```

**Outputs:** `{prediction_dir}/../backtest/` (or `backtest_with_costs/` when using `--backtest_suffix _with_costs`)
- `backtest_results.csv` - Metrics
- `daily_returns.csv` - Time series
- `equity_curve.png` - Plot (with `--plot`)
- `summary.txt` - Report

## Paper Trade Pipeline

The paper trading pipeline in `paper_trade/scripts/` provides inference, portfolio decisions, tracking, and reporting:

```bash
# Full nightly run (refresh data, track, infer, portfolio, report)
python paper_trade/scripts/run_nightly.py

# Individual steps
python paper_trade/scripts/infer.py
python paper_trade/scripts/portfolio.py
python paper_trade/scripts/track.py
python paper_trade/scripts/report.py
```

Pipeline order: `refresh_data` -> `track` -> `infer` -> `portfolio` -> `report`

## Finding Latest Run

```python
import glob

runs = sorted(glob.glob("results/baseline/*/"))
latest = runs[-1]
predictions = f"{latest}averaged_predictions"
```

## Key Arguments

### run_experiment.py
- `output_dir=PATH` - Where to save outputs
- `experiment_name=NAME` - Experiment identifier
- `+experiment=PRESET` - Use config preset (with_vix, etc.)
- `+data=PRESET` - Use data preset (russell1000, etc.)

## Output Structure

```
output_dir/
└── experiment_name/
    └── YYYYMMDD_HHMMSS/
        ├── config.yaml
        ├── training_*.log
        ├── checkpoints/
        ├── averaged_predictions/
        └── backtest/
            ├── backtest_results.csv
            ├── daily_returns.csv
            ├── equity_curve.png
            └── summary.txt
```

## See Also

- `OUTPUT_MANAGEMENT.md` - Full documentation
- `colab_workflow.ipynb` - Complete Colab example
- `configs/` - Configuration files
