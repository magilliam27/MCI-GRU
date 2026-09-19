# Configuration Guide

This guide explains the configuration groups, the presets that exist, and how to
override them. Feature columns and their flags are in [`FEATURES.md`](FEATURES.md); what
the model and graph parameters drive is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Configuration Files Structure

The configuration system is Hydra-based with typed dataclasses in `mci_gru/config.py`: `DataConfig`, `FeatureConfig`, `GraphConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`.

```
configs/
├── config.yaml                   # Base configuration; composes data=gics_top10_110_2016 and features=with_momentum
├── data/                         # Source, universe, PIT settings, date windows (select with data=<name>)
│   ├── gics_top10_110_2016.yaml  #   110-name GICS top-10 point-in-time universe, 2016 start (default; CSV)
│   ├── gics_top10_110.yaml       #   the same universe, 2021 start (CSV)
│   ├── csv_sp500.yaml            #   S&P 500 from a CSV, PIT off
│   ├── sp500.yaml                #   S&P 500 through the LSEG loader
│   ├── lseg_sp500.yaml           #   S&P 500 through the LSEG loader, explicit
│   ├── russell1000.yaml          #   Russell 1000 through the LSEG loader
│   ├── temporal_2016.yaml ...    #   anchored historical snapshot universes, 2016 to 2019 (mechanics only)
│   └── index_level.yaml          #   one index series; experiment_mode=index_level
├── features/                     # Feature families (select with features=<name>)
│   ├── base.yaml                 #   OHLCV only
│   ├── with_momentum.yaml        #   default
│   └── full.yaml                 #   every family
└── experiment/                   # Named experiments (add with +experiment=<name>); catalogued below
```

## Data Sources

Every data preset sets `source` (`csv` or `lseg`), the universe, the panel file, the
point-in-time settings, and the train, validation, and test windows. Select one with
`data=<name>`; `data` is already in the base defaults list, so it takes no `+` prefix.

### CSV (default)

The base configuration composes `data=gics_top10_110_2016`: a CSV panel with
`kdcode, dt, open, high, low, close, volume` columns and a point-in-time membership
table (`kdcode, valid_from, valid_to`). Both files are LSEG-derived and gitignored, so
the default runs only where they exist, and `use_pit_universe: true` is required for
that universe rather than optional. A run that supplies its own panel passes
`data.use_pit_universe=false`:

```bash
python run_experiment.py data.filename=/path/to/panel.csv data.use_pit_universe=false
```

`csv_sp500` is the generic CSV preset (`data/raw/market/sp500_data.csv`, PIT off, a 2019
training start):

```bash
python run_experiment.py data=csv_sp500
```

### LSEG

`sp500`, `lseg_sp500`, and `russell1000` set `source: lseg` and fetch constituent OHLCV
through the Refinitiv Workspace desktop application, which must be running. Nothing
selects LSEG by default.

```bash
python run_experiment.py data=lseg_sp500
```

### FRED

Credit-spread and global-regime features read FRED series and need `FRED_API_KEY` in
the environment (see `.env.example`). The frozen recipe enables global regime, so it
needs the key unless a smoke run turns that family off.

```bash
export FRED_API_KEY="your_key_here"
python run_experiment.py features=full
```

### Index-Level Mode

`data=index_level` sets `experiment_mode: index_level`: a single index series (FRED
`SP500`, or a CSV with `dt, close` named in `data.index_filename`) in place of a stock
panel, for experiments free of stock-level survivorship bias.

### Universes

| Universe | Names | Preset |
| --- | ---: | --- |
| GICS top-10 by market cap, point-in-time, 2016 start | ~110 | `data=gics_top10_110_2016` (default) |
| GICS top-10 by market cap, point-in-time, 2021 start | ~110 | `data=gics_top10_110` |
| S&P 500, LSEG | ~500 | `data=sp500`, `data=lseg_sp500` |
| S&P 500, CSV | ~500 | `data=csv_sp500` |
| Russell 1000, LSEG | ~1000 | `data=russell1000` |
| Anchored historical snapshot universes, 2016 to 2019 | S&P 500 snapshot | `data=temporal_2016` … `data=temporal_2019` |

The `temporal_*` presets are non-PIT anchored historical snapshot universes: mechanics
smokes only, never headline evidence (see Long-History Presets below).

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

## Experiment Presets

`configs/experiment/`, added with `+experiment=<name>`. Each is a `@package _global_`
overlay on the base configuration; the file's header comment records its purpose.

| Preset | What it changes |
| --- | --- |
| `baseline` | Nothing; names the run `baseline`. |
| `paper_faithful` | The paper's architecture: legacy GRU, no multi-scale encoder, no self-attention, ReLU, scalar edge weights, MSE loss, loss-based selection, no scheduler or AMP. |
| `correlation_dynamic` | Rebuilds the correlation graph every 6 months (`graph.update_frequency_months=6`). |
| `graph_zeroed` | Ablation control: every correlation edge suppressed, model unchanged. |
| `graph_thr05` | Correlation threshold 0.5 instead of 0.8. |
| `graph_topk20_static` | Per-node top-20 neighbours by signed correlation on the static cadence. |
| `graph_sector_only` | Correlation edges zeroed, sector relation on, sector map derived from the universe metadata export. |
| `long_history_his_t_21`, `_63`, `_126` | The frozen recipe with a longer temporal window; non-PIT runs are mechanics checks only. |
| `momentum_dynamic` | Cycle-aware dynamic momentum blend (`features.momentum_blend_mode=dynamic`). |
| `volatility_targeting` | `with_momentum` plus the volatility and volatility-targeting families. |
| `with_vix` | The `full` feature preset with VIX and volatility on. |
| `pit_temporal_2022` … `pit_temporal_2025` | True rolling point-in-time S&P 500 masked panels: five training years, one validation year, the named test year. |

Retired presets are readable at tag `archive/pre-cleanup-2026-09`.

## Feature Presets

`configs/features/`, selected with `features=<name>`: `base`, `with_momentum` (the
default), and `full`. Every column each family produces, its window, and its timing rule
are in [`FEATURES.md`](FEATURES.md).

## Graph Parameters

`graph.*`, `GraphConfig`. Base values from `configs/config.yaml`; the frozen recipe pins
the same values.

| Parameter | Base | Meaning |
| --- | --- | --- |
| `judge_value` | 0.8 | Correlation threshold for an edge; used when `top_k` is 0. |
| `update_frequency_months` | 0 | Months between graph rebuilds; 0 is a static graph. |
| `corr_lookback_days` | 252 | Trading days of returns behind each correlation. |
| `top_k` | 0 | Per-node top-K neighbour selection; 0 keeps the threshold rule. |
| `top_k_metric` | `corr` | Ranking metric when `top_k` is set: `corr` or `abs_corr`. |
| `use_multi_feature_edges` | true | Four-channel edge attributes instead of a scalar weight. |
| `drop_edge_p` | 0.1 | Train-time edge dropout; 0 disables. |
| `isolate_edge_dropout_rng` | false | Fork the RNG around edge dropout so the global stream is untouched. |
| `append_snapshot_age_days` | false | One extra edge column: days since the snapshot's valid-from date. |
| `use_lead_lag_features` | false | Two extra edge columns from the best lead-lag correlation (`lead_lag_days`). |
| `use_sector_relation` | false | Second GAT branch over sector edges from `sector_map_csv`. |
| `zero_edges` | false | Suppress every correlation edge (the ablation control). |

The graph's construction, edge attributes, and static, dynamic, and sector forms are in
[`ARCHITECTURE.md`](ARCHITECTURE.md), Graph.

## Model Parameters

`model.*`, `ModelConfig`. Base values from `configs/config.yaml`.

| Parameter | Base | Meaning |
| --- | --- | --- |
| `his_t` | 10 | Lookback window in trading days. |
| `label_t` | 5 | Forward-return horizon in trading days. |
| `gru_hidden_sizes` | [32, 10] | Hidden size of each GRU layer. |
| `hidden_size_gat1` | 32 | Hidden width of the cross-sectional GAT. |
| `output_gat1` | 4 | Output width of the cross-sectional GAT. |
| `gat_heads` | 4 | Attention heads in the GAT layers. |
| `hidden_size_gat2` | 32 | Hidden width of the prediction GAT. |
| `num_hidden_states` | 32 | Learned market latent state vectors per stream. |
| `cross_attn_heads` | 4 | Heads in the market-latent cross-attention. |
| `slow_kernel` | 5 | Kernel of the slow temporal path's convolution. |
| `slow_stride` | 2 | Stride of the slow temporal path's convolution. |
| `use_multi_scale` | true | Fast and slow temporal paths; false is the plain encoder. |
| `temporal_encoder` | `gru_attn` | Temporal backbone: `legacy`, `gru_attn`, or `transformer`. |
| `use_self_attention` | true | Cross-stock self-attention before the prediction GAT. |
| `use_group_type_embed` | true | Stream-type embedding inside that self-attention. |
| `use_a1_a2_cross_attention` | false | A2 queries A1's temporal sequence (`cross_a2_num_heads`, 4). |
| `use_nn_multihead_attention` | true | Library attention in the latent stage instead of the legacy implementation. |
| `use_trunk_regularisation` | true | LayerNorm and dropout (`trunk_dropout`, 0.1) on the concatenated streams. |
| `activation` | `elu` | Activation inside the GAT blocks: `elu` or `relu`. |
| `output_activation` | `none` | Final head activation: `none`, `elu`, `relu`, or `sigmoid`. Quote `"none"` in YAML. |
| `latent_init_scale` | 0.02 | Standard deviation of the latent state initialisation. |

What each parameter drives is in [`ARCHITECTURE.md`](ARCHITECTURE.md), Model
Architecture.

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
python run_experiment.py experiment_name=quick_test data.filename=/path/to/panel.csv data.use_pit_universe=false training.num_epochs=2 training.num_models=1 tracking.enabled=false
```

`data.use_pit_universe=false` is needed for any panel other than the default's, whose
membership CSV is not in the repository. Disable MLflow for a quieter smoke run if
desired.

### Use CSV Data Source

```bash
python run_experiment.py data=csv_sp500
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
python run_experiment.py +experiment=with_vix
```

The preset already selects the `full` feature set; `features` is in the base defaults
list, so it is overridden with `features=<name>`, never appended with `+`.

### Hyperparameter Sweep

```bash
python run_experiment.py --multirun experiment_name=lookback_sweep model.his_t=5,10,15,20
```

### Russell 1000 Dataset

```bash
python run_experiment.py data=russell1000 experiment_name=russell1000_baseline
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
python run_experiment.py data=csv_sp500
```

### Issue: "LSEG API key not found"

**Cause:** Environment variable not set

**Solutions:**
```bash
export LSEG_API_KEY="your_key_here"
python run_experiment.py data=csv_sp500
```

### Issue: "Output not saved to Google Drive"

**Cause:** Hydra not respecting output_dir override

**Solution:** Ensure `run_experiment.py` uses `HydraConfig.get().runtime.output_dir` and passes `output_path` to `train_multiple_models`.

## Further Reading

- `FEATURES.md`, `QUICK_REFERENCE.md`, `OUTPUT_MANAGEMENT.md`, Hydra: https://hydra.cc/
