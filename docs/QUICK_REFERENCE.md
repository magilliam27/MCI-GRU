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
- Re-score a saved run with the current evaluation code, no retraining: `python scripts/rescore_run.py --run_dir path/to/run` writes `evaluation_summary.rescored.json` and `daily_ic.rescored.csv` beside the originals and refuses inputs whose bytes differ from the run's (#337)

## Backtesting

Backtesting is done via `scripts/backtest_sp500.py` and `scripts/backtest_sp500_daily.py`.
Point `--run_dir` at one window's run output: it supplies the data file, PIT file and
export cutoff, test window, label, k and members, and refuses a file whose bytes differ
from what the run consumed (#339). Input flags are then checked overrides. Without
`--run_dir`, every input (`--predictions_dir --data_file --test_start --test_end --top_k
--label_t`) must be named; none has a default.

The default replay trades the return the model was trained on: buy at close `D+1`,
sell at close `D+label_t`, in staggered tranches (four for `label_t=5`), with
buy-and-hold weights inside a tranche, against the equal-weight PIT universe with
each issuer counted once. Price returns only. A held name with no exit close is
sold at its last close, the tranche holds cash, and the event is logged. At most
one share class per issuer is held (#340); `--issuer_map` (a kdcode,issuer CSV) adds pairs the built-in map lacks.

```bash
# Default: close D+1 to close D+5 in four tranches
python scripts/backtest_sp500.py --run_dir path/to/run --top_k 10 --auto_save --plot

# The paper's open D+1 to open D+2 hold, a labelled variant (also what
# backtest_sp500_daily.py runs)
python scripts/backtest_sp500.py --run_dir path/to/run --top_k 10 --strategy open_to_open

# With transaction costs (separate directory); costs are booked on the trade session
python scripts/backtest_sp500.py \
    --run_dir path/to/run --top_k 10 \
    --auto_save \
    --backtest_suffix _with_costs \
    --transaction_costs
```

Every metric and file reads the book's one daily series. `sharpe` is mean/sd
annualised; `ASR` is the paper's ARR/AVoL, labelled as such. MDD starts from 1.0,
IR is annualised, monthly returns compound. The multiple-testing haircut is Holm by
default (`--num_tests`); BHY needs `--trial_ledger` with every other trial's
p-value. `--multi_model` is gone: score the ensemble's averaged predictions.

**Outputs:** `{prediction_dir}/../backtest/` (or `backtest_with_costs/` when using `--backtest_suffix _with_costs`)
- `backtest_results.csv`, `backtest_metrics.json` - Metrics (with input SHA-256s)
- `backtest_config.json` - The config and the input record
- `daily_returns.csv`, `cumulative_performance.csv`, `monthly_performance.csv` - The one series
- `holdings.csv`, `holdings_summary.csv`, `trade_journal.csv`, `missing_close_events.csv`
- `equity_curve.png` - Plot (with `--plot`)
- `summary.txt` - Report

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
- `docs/NOTEBOOK_BEST_PRACTICES.md` - Colab notebook contract
- `configs/` - Configuration files
