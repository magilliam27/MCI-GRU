# Configuration Guide

This guide explains all configuration options and how to use them effectively.

## Configuration Files Structure

The configuration system is Hydra-based with typed dataclasses in `mci_gru/config.py`: `DataConfig`, `FeatureConfig`, `GraphConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`.

```
configs/
├── config.yaml              # Base configuration
├── data/
│   ├── sp500.yaml          # S&P 500 (default: LSEG)
│   ├── csv_sp500.yaml      # S&P 500 (CSV source)
│   ├── lseg_sp500.yaml     # S&P 500 (explicit LSEG)
│   ├── russell1000.yaml    # Russell 1000
│   └── ...
├── features/
│   ├── base.yaml           # Basic features only
│   ├── with_momentum.yaml  # Default (with momentum)
│   ├── with_credit.yaml    # With credit spread features
│   ├── with_regime.yaml    # With regime features
│   ├── full.yaml           # All features
│   └── ...
└── experiment/
    ├── baseline.yaml       # Baseline experiment
    ├── with_vix.yaml       # With VIX features
    ├── momentum_dynamic.yaml
    ├── lookback_sweep.yaml # Lookback period sweep
    └── ...
```

## Data Sources

### LSEG (Default)

**Configuration:** `configs/data/sp500.yaml` or `configs/data/lseg_sp500.yaml`

```yaml
source: lseg
api_key: ${oc.env:LSEG_API_KEY}
```

**Setup:** `export LSEG_API_KEY="your_api_key_here"`

**Usage:**
```bash
python run_experiment.py  # Uses LSEG by default
```

### CSV Fallback

**Configuration:** `configs/data/csv_sp500.yaml`

```yaml
source: csv
filename: sp500_data.csv
```

**Usage:**
```bash
python run_experiment.py +data=csv_sp500
```

## Default Settings

| Category | Setting | Default |
|----------|---------|---------|
| Data | source | lseg |
| Data | train | 2019-01-01 to 2023-12-31 |
| Data | val | 2024-01-01 to 2024-12-31 |
| Data | test | 2025-01-01 to 2025-12-31 |
| Model | his_t | 10 |
| Model | label_t | 5 |
| Model | gru_hidden_sizes | [32, 10] |
| Training | batch_size | 32 |
| Training | learning_rate | 5e-5 |
| Training | num_epochs | 100 |
| Training | num_models | 10 |

## Common Configurations

### Basic Training (Uses All Defaults)

```bash
python run_experiment.py
```

### Custom Output Directory

```bash
python run_experiment.py output_dir=/content/drive/MyDrive/MCI-GRU-Experiments
```

### Quick Test Run

```bash
python run_experiment.py experiment_name=quick_test training.num_epochs=2 training.num_models=1
```

### Use CSV Data Source

```bash
python run_experiment.py +data=csv_sp500
```

### Different Lookback Period

```bash
python run_experiment.py experiment_name=lookback_20 model.his_t=20
```

### With VIX Features

```bash
python run_experiment.py +experiment=with_vix +features=full
```

### Hyperparameter Sweep

```bash
python run_experiment.py --multirun experiment_name=lookback_sweep model.his_t=5,10,15,20
```

### Russell 1000 Dataset

```bash
python run_experiment.py +data=russell1000 experiment_name=russell1000_baseline
```

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
python run_experiment.py --multirun model.his_t=5,10 training.batch_size=32,64
```

## Troubleshooting

### Issue: "Data file not found"

**Cause:** Mismatch between config and actual file

**Solutions:**
```bash
python run_experiment.py data.filename=your_actual_file.csv
python run_experiment.py +data=csv_sp500
```

### Issue: "LSEG API key not found"

**Cause:** Environment variable not set

**Solutions:**
```bash
export LSEG_API_KEY="your_key_here"
python run_experiment.py +data=csv_sp500
```

### Issue: "Output not saved to Google Drive"

**Cause:** Hydra not respecting output_dir override

**Solution:** Ensure `run_experiment.py` uses `HydraConfig.get().runtime.output_dir` and passes `output_path` to `train_multiple_models`.

## Further Reading

- `QUICK_REFERENCE.md`, `OUTPUT_MANAGEMENT.md`, Hydra: https://hydra.cc/
