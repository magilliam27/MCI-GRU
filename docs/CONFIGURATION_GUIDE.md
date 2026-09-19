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
│   ├── full.yaml           # All features
│   └── ...
└── experiment/
    ├── baseline.yaml       # Baseline experiment
    ├── with_vix.yaml       # With VIX features
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

### True Rolling PIT S&P 500 Panel

Use `data.pit_universe_mode=masked_panel` when the model should score the
real-world S&P 500 opportunity set for each date instead of the old continuous
member subset. The pipeline keeps a fixed PIT union axis internally, then uses
daily `active_member`, `feature_ready`, `loss`, and `tradable` masks for
training, validation, prediction export, graph batching, and evaluation.

```yaml
data:
  use_pit_universe: true
  pit_universe_csv: data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
  pit_universe_mode: masked_panel
  pit_min_scoreable_stocks: 450
  pit_breadth_policy: error
```

Pre-membership OHLCV is allowed for lookback features because it was public at
the time. A future joiner is still excluded from loss and prediction rows until
its `valid_from` date. Legacy temporal constituent CSVs (`sp500_constituents_2016`
and similar) are not PIT-clean membership histories; use Joiner/Leaver interval
artifacts for true rolling panels.

Ready-made experiment presets:

```bash
python run_experiment.py +experiment=pit_temporal_2022
python run_experiment.py +experiment=pit_temporal_2023
python run_experiment.py +experiment=pit_temporal_2024
python run_experiment.py +experiment=pit_temporal_2025
```

## Regime Inputs

Global regime features use the live FRED/LSEG-backed loader by default. Leave
`features.regime_inputs_csv` unset or `null`; with `include_global_regime=true`,
`DataManager.load_regime_inputs()` builds the full seven-variable regime surface:
market, yield curve, oil, copper, stock-bond correlation, monetary policy, and
volatility.

Set `FRED_API_KEY` for the normal regime workflow. When `data.source=lseg`,
configured LSEG RICs can supplement live market, copper, yield, oil, or VIX
series where available.

`features.regime_inputs_csv` is deprecated. It remains only as a legacy offline
escape hatch, emits a `DeprecationWarning`, and requires `dt` plus all seven
regime variables if used. Do not set it for production training, paper-trade
inference, or notebook runs.

## Default Frozen Experiment Recipe

Production-style confirmation notebooks and PIT validation runs should use the
frozen recipe documented in
[`DEFAULT_EXPERIMENT_RECIPE.md`](DEFAULT_EXPERIMENT_RECIPE.md):
`static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`.

Core overrides:

| Category | Setting | Frozen recipe value |
|----------|---------|---------------------|
| Experiment | seed | `1729` |
| Training | num_models | `20` |
| Training | num_epochs | `100` |
| Training | early_stopping_patience | `15` |
| Training | loss_type | `ic` |
| Training | label_type | `returns` |
| Training | selection_metric | `val_ic` |
| Training | shuffle_train | `true` |
| Model | label_t | `5` |
| Graph | update_frequency_months | `0` |
| Graph | corr_lookback_days | `252` |
| Graph | top_k / top_k_metric | `0` / `corr` |
| Graph | use_multi_feature_edges | `true` |
| Graph | drop_edge_p | `0.1` |
| Graph | lead-lag / snapshot-age | disabled |
| Features | momentum | `features=with_momentum`, weekly momentum on, static 50/50 blend |
| Features | global regime | `include_global_regime=true`, `regime_strict=true`, `regime_include_subsequent_returns=false` |

`FRED_API_KEY` is required for the full recipe unless a smoke run explicitly
disables global regime features.

## Hydra Base Defaults

Values below reflect **`configs/config.yaml`** merged with **`configs/data/gics_top10_110_2016.yaml`** (Hydra `defaults`). Python dataclass defaults in `mci_gru/config.py` do **not** all match these: the data-group values below come from the YAML and override the dataclass.

| Category | Setting | Default |
|----------|---------|---------|
| Data | source | `csv` (the base default is no longer an LSEG config; use `data=lseg_sp500` for the live path) |
| Data | filename | `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.csv` (gitignored) |
| Data | train | 2016-01-04 to 2023-12-31 |
| Data | val | 2024-01-22 to 2024-12-31 (gap after `train_end` **>** `label_t` days — label embargo) |
| Data | test | 2025-01-22 to 2025-12-31 (gap after `val_end` **>** `label_t` days) |
| Data | skip_embargo_check | `false` (`ExperimentConfig` raises if gaps are too small; set `true` only for legacy repro) |
| Data | use_pit_universe | `true` — PIT filtering is required for this universe, not optional |
| Data | pit_universe_csv | `data/raw/constituents/..._pit_universe.csv` (**not in the repository**) |
| Data | pit_universe_mode | `masked_panel` |
| Data | pit_min_scoreable_stocks | `104` (measured session minimum is 108) |
| Data | pit_breadth_policy | `error` |

Because the base default sets `use_pit_universe: true` against a PIT CSV that is not committed, any run supplying its own panel must pass `data.use_pit_universe=false`. `scripts/ci_smoke.py` does this.
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
| Training | minimum_selection_rows | `1` (eligible validation dates the selection metric needs; a run whose `selection_metric` has fewer raises instead of selecting a checkpoint from an empty metric) |
| Training | lr_scheduler | `cosine` (linear warmup `warmup_steps` then cosine decay; `none` disables) |
| Training | use_amp | `true` on CUDA (no-op on CPU) |
| Tracking | enabled | `true` (local `./mlruns`; set `tracking.enabled=false` to disable) |

## Common Configurations

### Basic Training (Hydra Base Config)

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

### Long-History Presets

Issue #23 adds controlled long-history presets for testing whether more
temporal context helps the frozen production-style recipe. These presets keep
the frozen graph, feature, loss, label, selection, and ensemble semantics fixed;
`model.his_t` is the intended experimental factor.

```bash
python run_experiment.py +experiment=long_history_his_t_21
python run_experiment.py +experiment=long_history_his_t_63
python run_experiment.py +experiment=long_history_his_t_126
```

`his_t=252` is intentionally not a first-pass preset. Treat it as a gated
manual candidate after the shorter windows pass memory and runtime checks.

For a cheap mechanics smoke, use the non-PIT anchored historical snapshot universe
set rather than the base `sp500_data.csv` fallback. The 2025-style
local surface is `data=temporal_2019`, which points at
`sp500_2019_universe_data_through_2026.csv`. Override only the runtime cost and
any unavailable external inputs:

```bash
python run_experiment.py +experiment=long_history_his_t_21 data=temporal_2019 training.num_epochs=1 training.num_models=1 training.early_stopping_patience=2 tracking.enabled=false features.include_global_regime=false features.regime_strict=false
```

Do not treat non-PIT smoke metrics as model-performance evidence. Full
long-history evaluation should run the generated Colab notebook:

```bash
python scripts/gen_long_history_pit_eval_nb.py
```

Then open `notebooks/long_history_pit_eval_colab.ipynb` in Colab. The notebook
evaluates `his_t=10`, `21`, `63`, and `126` across the 2022, 2023, 2024, and
2025 true PIT masked-panel presets, with `his_t=252` behind
`INCLUDE_HIS_T_252 = False`.

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

## Linux CPU reference environment

Issue [#143](https://github.com/magilliam27/MCI-GRU/issues/143) defines a separate
experimental CPU recipe for **Ubuntu 24.04, x86_64, CPython 3.12.11**. This is
neither the repository's minimum/default interpreter nor a GPU/Colab profile.
`pyproject.toml` remains the authority for package ranges; `requirements.txt`
retains its Colab-facing ranges. Existing CI policy remains unchanged pending D3.

The recipe consists of two files, installed in this order:

| File | Role and package origins |
| --- | --- |
| `requirements-linux-cpu-bootstrap.lock` | Exact PyPI wheels for pip 25.0.1 and setuptools 80.9.0, with SHA-256 hashes |
| `requirements-linux-cpu.lock` | Exact URLs and SHA-256 hashes for the 62 dev+fred dependency artifacts; PyPI distributions except the official PyTorch CPU wheel |

The Linux recipe keeps all 61 non-torch versions from `requirements.lock` and
explicitly selects **torch 2.12.1+cpu** instead of PyPI torch 2.12.1. Its official
CPU wheel declares `setuptools<82`, satisfied by the bootstrap pin, and does not
require the Linux CUDA packages. `wheel==0.47.0` is retained in the dependency
lock; `antlr4-python3-runtime==4.9.3` is a hashed source archive built locally
using the pinned tools without build isolation. Other artifacts are wheels.
The recipe pins installed versions and downloaded inputs, not byte-identical
outputs of a source build or bitwise training results.

`requirements.lock` remains the separate Windows reference and is unchanged.
The earlier [Windows-lock-on-Linux measurement](https://github.com/magilliam27/MCI-GRU/issues/143#issuecomment-5739448133)
is negative evidence: its 62 pins installed, but eight missing Linux torch
dependencies caused dependency-check failure, 37 test-collection errors and a
smoke import failure. It must not be counted as a successful CPU rebuild.

### Rebuild in two new environments

Run from the intended clean checkout, with CPython **3.12.11** already available
as `python3.12`. The command rejects another patch version. Keep the temporary
directory as evidence; never synchronize, repair or relocate a shared venv.
The locks use exact artifact URLs and `--require-hashes`; installation disables
dependency resolution and build isolation. A missing artifact, hash mismatch or
incompatible dependency is a failure, not permission to substitute another pin.

```bash
set -euo pipefail
python3.12 -c 'import sys; assert sys.version_info[:3] == (3, 12, 11)'
export CPU_SOURCE_ROOT="$(git rev-parse --show-toplevel)"
cpu_root="$(mktemp -d)"
git rev-parse HEAD > "$cpu_root/source-sha.txt"
sha256sum requirements-linux-cpu-bootstrap.lock requirements-linux-cpu.lock \
  > "$cpu_root/recipe-files.sha256"
sha256sum "$cpu_root/recipe-files.sha256"
for rebuild in 1 2; do
  python3.12 -m venv "$cpu_root/venv-$rebuild"
  cpu_python="$cpu_root/venv-$rebuild/bin/python"
  "$cpu_python" -m pip install --no-cache-dir --no-deps --require-hashes \
    --force-reinstall -r requirements-linux-cpu-bootstrap.lock \
    --report "$cpu_root/bootstrap-$rebuild.json"
  "$cpu_python" -m pip install --no-cache-dir --no-deps --no-build-isolation \
    --require-hashes -r requirements-linux-cpu.lock \
    --report "$cpu_root/install-$rebuild.json"
  "$cpu_python" -m pip install --no-deps --no-build-isolation -e '.[dev,fred]' \
    --report "$cpu_root/editable-$rebuild.json"
  "$cpu_python" -m pip list --format=json > "$cpu_root/versions-$rebuild.json"
  "$cpu_python" -m pip check
  (cd "$cpu_root" && "$cpu_python" -c 'import os, pathlib, mci_gru, torch; assert pathlib.Path(mci_gru.__file__).resolve() == pathlib.Path(os.environ["CPU_SOURCE_ROOT"]) / "mci_gru/__init__.py"; assert torch.__version__ == "2.12.1+cpu" and torch.version.cuda is None; print(mci_gru.__file__)')
  "$cpu_python" -m pytest tests/ -v -m "not requires_data and not requires_lseg" \
    --timeout=120 --junitxml="$cpu_root/pytest-$rebuild.xml"
  "$cpu_python" scripts/ci_smoke.py
done
diff -u "$cpu_root/versions-1.json" "$cpu_root/versions-2.json"
printf 'Retained rebuild evidence: %s\n' "$cpu_root"
```

The dedicated `.github/workflows/lock-stack-linux.yml` performs the same rebuild
in two independent hosted jobs without package caching, then compares the
installed version sets, package download origins/hashes, source SHA, combined
recipe digest, exact Python version and CPU identity. Both jobs must pass
`pip check`, project build/core/dev/fred constraint checks, the existing
capability-independent suite, smoke and outside-checkout editable import check.
The combined recipe digest is SHA-256 of the ordered `sha256sum` output for the
bootstrap lock followed by the dependency lock, using repository-relative names.

Per-run artifacts retain both recipe files, interpreter and runner identity,
installation reports with package origins, complete installed versions, logs,
JUnit and failed/skipped step outcomes for 30 days. Qualification requires two
successful rebuilds and a successful comparison for the same submitted head;
failed historical attempts remain separate. Optional MLflow, real market data,
LSEG credentials and GPU coverage are outside this dev+fred CPU profile. The
synthetic smoke explicitly disables tracking and external data access.

## Further Reading

- `QUICK_REFERENCE.md`, `OUTPUT_MANAGEMENT.md`, Hydra: https://hydra.cc/
