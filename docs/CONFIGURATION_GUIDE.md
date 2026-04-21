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

Values below reflect **`configs/config.yaml`** merged with **`configs/data/sp500.yaml`** (Hydra `defaults`). Python dataclass defaults in `mci_gru/config.py` match these where duplicated.

| Category | Setting | Default |
|----------|---------|---------|
| Data | source | `lseg` in `sp500.yaml` (use `data.source=csv` or `+data=csv_sp500` if Refinitiv is not installed) |
| Data | train | 2019-01-01 to 2023-12-31 |
| Data | val | 2024-01-08 to 2024-12-31 (gap after `train_end` **>** `label_t` days — label embargo) |
| Data | test | 2025-01-08 to 2025-12-31 (gap after `val_end` **>** `label_t` days) |
| Data | skip_embargo_check | `false` (`ExperimentConfig` raises if gaps are too small; set `true` only for legacy repro) |
| Model | his_t | 10 |
| Model | label_t | 5 |
| Model | gru_hidden_sizes | [32, 10] |
| Graph | use_multi_feature_edges | `true` (4-D edge features; `paper_faithful` preset pins `false`) |
| Training | batch_size | 32 |
| Training | learning_rate | 5e-5 |
| Training | num_epochs | 100 |
| Training | num_models | 10 |
| Training | loss_type | `combined` (MSE + IC; `ic_loss_alpha` 0.5) |
| Training | selection_metric | `val_ic` (checkpoint / early stopping; use `val_loss` to mirror loss only) |
| Training | lr_scheduler | `cosine` (linear warmup `warmup_steps` then cosine decay; `none` disables) |
| Training | use_amp | `true` on CUDA (no-op on CPU) |
| Tracking | enabled | `true` (local `./mlruns`; set `tracking.enabled=false` to disable) |

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
python run_experiment.py experiment_name=quick_test training.num_epochs=2 training.num_models=1 data.source=csv tracking.enabled=false
```

Use `data.source=csv` when LSEG / `refinitiv-data` is unavailable; disable MLflow for a quieter smoke run if desired.

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
