# Configuration Guide

This guide explains all configuration options and how to use them effectively.

## Table of Contents
1. [Configuration Files Structure](#configuration-files-structure)
2. [Data Sources](#data-sources)
3. [Default Settings](#default-settings)
4. [Common Configurations](#common-configurations)
5. [Troubleshooting](#troubleshooting)

## Configuration Files Structure

```
configs/
├── config.yaml              # Main configuration
├── data/
│   ├── sp500.yaml          # S&P 500 (default: LSEG)
│   ├── csv_sp500.yaml      # S&P 500 (CSV source)
│   ├── lseg_sp500.yaml     # S&P 500 (explicit LSEG)
│   └── russell1000.yaml    # Russell 1000
├── features/
│   ├── base.yaml           # Basic features only
│   ├── with_momentum.yaml  # Default (with momentum)
│   └── full.yaml           # All features
└── experiment/
    ├── baseline.yaml       # Baseline experiment
    ├── with_vix.yaml       # With VIX features
    └── lookback_sweep.yaml # Lookback period sweep
```

## Data Sources

### LSEG (Default)

**Configuration:** `configs/data/sp500.yaml` or `configs/data/lseg_sp500.yaml`

```yaml
source: lseg
api_key: ${oc.env:LSEG_API_KEY}
```

**Setup:**
```bash
# Set environment variable
export LSEG_API_KEY="your_api_key_here"

# Or in Colab Secrets (left sidebar -> Key icon)
# Add secret: LSEG_API_KEY = your_api_key_here
```

**Usage:**
```bash
python run_experiment.py  # Uses LSEG by default
```

### CSV

**Configuration:** `configs/data/csv_sp500.yaml`

```yaml
source: csv
filename: sp500_data.csv
```

**Usage:**
```bash
# Use CSV explicitly
python run_experiment.py +data=csv_sp500

# Or override filename
python run_experiment.py data.source=csv data.filename=my_data.csv
```

## Default Settings

### Updated Defaults (After Configuration Fix)

| Setting | Old Value | New Value | Why Changed |
|---------|-----------|-----------|-------------|
| `data.source` | `csv` | `lseg` | Use LSEG by default |
| `data.filename` | `sp500_yf_download.csv` | `sp500_data.csv` | Match actual file |
| `evaluate.test_start` | `2023-01-01` | `2025-01-01` | Match training config |
| `evaluate.test_end` | `2023-12-31` | `2025-12-31` | Match training config |
| `evaluate.data_file` | `sp500_yf_download.csv` | `sp500_data.csv` | Match actual file |

### Date Ranges

**Default dates** (can be overridden):
```yaml
train_start: "2019-01-01"
train_end: "2023-12-31"
val_start: "2024-01-01"
val_end: "2024-12-31"
test_start: "2025-01-01"
test_end: "2025-12-31"
```

## Common Configurations

### 1. Basic Training (Uses All Defaults)

```bash
python run_experiment.py
```

### 2. Custom Output Directory (Google Drive)

```bash
python run_experiment.py \
    output_dir=/content/drive/MyDrive/MCI-GRU-Experiments
```

### 3. Quick Test Run

```bash
python run_experiment.py \
    experiment_name=quick_test \
    training.num_epochs=2 \
    training.num_models=1
```

### 4. Use CSV Data Source

```bash
python run_experiment.py +data=csv_sp500
```

### 5. Different Lookback Period

```bash
python run_experiment.py \
    experiment_name=lookback_20 \
    model.his_t=20
```

### 6. With VIX Features

```bash
python run_experiment.py \
    +experiment=with_vix \
    +features=full
```

### 7. Hyperparameter Sweep

```bash
python run_experiment.py \
    --multirun \
    experiment_name=lookback_sweep \
    model.his_t=5,10,15,20
```

### 8. Russell 1000 Dataset

```bash
python run_experiment.py \
    +data=russell1000 \
    experiment_name=russell1000_baseline
```

## Backtesting Configurations

### 1. Basic Backtest (Uses Defaults)

```bash
python evaluate_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --auto_save
```

### 2. With Transaction Costs

```bash
python evaluate_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --auto_save \
    --backtest_suffix _with_costs \
    --transaction_costs \
    --spread 10 \
    --slippage 5
```

### 3. Custom Date Range (Override Defaults)

```bash
python evaluate_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --test_start 2024-01-01 \
    --test_end 2024-12-31 \
    --data_file my_data.csv \
    --auto_save
```

### 4. Different Portfolio Size

```bash
python evaluate_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --top_k 20 \
    --auto_save
```

## Troubleshooting

### Issue: "Data file not found"

**Cause:** Mismatch between config and actual file

**Solutions:**
```bash
# Option 1: Override filename
python run_experiment.py data.filename=your_actual_file.csv

# Option 2: Use explicit CSV config
python run_experiment.py +data=csv_sp500

# Option 3: Check what files exist
ls -lh *.csv
```

### Issue: "No valid trading days found"

**Cause:** Date mismatch between predictions and backtest

**Solution:**
```bash
# Specify correct test dates
python evaluate_sp500.py \
    --predictions_dir path/to/predictions \
    --test_start YYYY-MM-DD \
    --test_end YYYY-MM-DD
```

### Issue: "LSEG API key not found"

**Cause:** Environment variable not set

**Solutions:**
```bash
# Option 1: Set environment variable
export LSEG_API_KEY="your_key_here"

# Option 2: Use CSV instead
python run_experiment.py +data=csv_sp500

# Option 3: In Colab, use Secrets
# Left sidebar -> Key icon -> Add LSEG_API_KEY
```

### Issue: "Output not saved to Google Drive"

**Cause:** Hydra not respecting output_dir override

**Status:** ✅ Fixed in latest version

**Verify Fix:**
```bash
# Check that run_experiment.py uses:
# - HydraConfig.get().runtime.output_dir
# - Passes output_path to train_multiple_models
```

## Validation

Before running experiments, validate your configuration:

```bash
python check_config.py
```

This will check:
- ✓ Configuration files exist
- ✓ Dates are valid
- ✓ Data source is configured
- ✓ Backtest defaults match training
- ✓ Data files exist

## Override Syntax

### Command-Line Overrides

```bash
# Single parameter
python run_experiment.py model.his_t=20

# Multiple parameters
python run_experiment.py model.his_t=20 training.batch_size=64

# Nested parameters
python run_experiment.py data.train_start=2020-01-01

# Add config group
python run_experiment.py +experiment=with_vix

# Override config group
python run_experiment.py data=russell1000
```

### Multi-Run (Sweeps)

```bash
# Sweep over single parameter
python run_experiment.py --multirun model.his_t=5,10,15,20

# Sweep over multiple parameters (cartesian product)
python run_experiment.py --multirun \
    model.his_t=5,10 \
    training.batch_size=32,64
```

## Best Practices

1. **Use validation script** before experiments
2. **Use config groups** for common patterns
3. **Document experiments** with meaningful names
4. **Use auto_save** for comprehensive backtest outputs
5. **Verify dates** align between training and backtesting
6. **Check data files** exist before training
7. **Use CSV fallback** if LSEG unavailable
8. **Test locally** before running in Colab

## Further Reading

- `QUICK_REFERENCE.md` - Quick commands
- `OUTPUT_MANAGEMENT.md` - Output persistence
- `COLAB_CELLS_UPDATED.md` - Updated Colab cells
- Hydra documentation: https://hydra.cc/
