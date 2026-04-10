# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_dynamic_graph_updates.py -v

# Run a single test class or function
python -m pytest tests/test_momentum_blend_modes.py::TestDynamicBlend -v

# Quick smoke run (2 epochs, 1 model)
python run_experiment.py training.num_epochs=2 training.num_models=1

# Full experiment with a preset
python run_experiment.py +experiment=paper_faithful

# Backtest against averaged_predictions output
python tests/backtest_sp500.py --predictions_dir results/<name>/<ts>/averaged_predictions --auto_save --plot

# Nightly paper-trade pipeline
python paper_trade/scripts/run_nightly.py [--skip-refresh]
```

## Architecture

### Data flow

```
CSV / LSEG / FRED
  → DataManager          (mci_gru/data/data_manager.py)
  → FeatureEngineer      (mci_gru/features/)
  → prepare_data()       (mci_gru/pipeline.py)   ← normalization, windowing, graph build
  → create_data_loaders  (data_manager.py)        ← CombinedDataset + combined_collate_fn
  → Trainer / train_multiple_models               ← ensemble of N independent models
  → averaged_predictions/
```

`pipeline.py` is the central orchestrator. It calls DataManager to load raw data, FeatureEngineer to add columns, then builds sliding-window tensors `(days, stocks, his_t, features)`, computes z-score normalization **from training dates only** (stored in `run_metadata.json` for reuse at inference), and calls `GraphBuilder` to produce edge tensors.

### Model (mci_gru/models/mci_gru.py)

Four parallel streams whose outputs are concatenated before the final predictor:

- **A1** — ImprovedGRU (or MultiScaleTemporalEncoder) over the `his_t`-day window → per-stock temporal embedding
- **A2** — GATBlock on the correlation graph using the *most recent day's* node features → cross-sectional embedding
- **B1/B2** — MarketLatentStateLearner: learned market-state vectors (R1, R2) enriched via multi-head cross-attention with A1/A2 → B1, B2

Concatenated `[A1, A2, B1, B2]` → optional self-attention → final GATBlock → per-stock scalar score.

### Graph (mci_gru/graph/builder.py)

`GraphBuilder` builds a Pearson-correlation graph over trailing returns (default 252 days). Pairs with `|corr| > judge_value` (default 0.8) get an edge weighted by the correlation value.

- **Static mode** (`update_frequency_months=0`, default): graph built once before training; `edge_index`/`edge_weight` passed through collate as fixed tensors.
- **Dynamic mode** (`update_frequency_months > 0`): graph updates every N months *per batch date* during training/val/test. Requires `batch_size=1` and `shuffle=False` for train. `Trainer._apply_dynamic_graph` calls `update_if_needed` → `get_current_graph` each batch; `_batched_edges` handles index shifting.

### Config system (Hydra)

`configs/config.yaml` is the base. Override groups with:

```bash
python run_experiment.py \
  data=temporal_2019 \          # configs/data/temporal_2019.yaml
  features=with_momentum \      # configs/features/with_momentum.yaml
  +experiment=paper_faithful    # configs/experiment/paper_faithful.yaml (additive)
```

Key parameters: `model.his_t`, `model.label_t`, `graph.judge_value`, `graph.update_frequency_months`, `training.loss_type` (mse/ic/combined), `training.label_type` (returns/rank), `training.num_models`.

Output lands in `results/{experiment_name}/{hydra_timestamp}/`.

### Paper trading (paper_trade/)

Uses **frozen** checkpoints from `paper_trade/Model/`. `infer.py` loads `run_metadata.json` (normalization stats, feature columns, stock list) and `graph_data.pt` (static graph) — it does **not** use `GraphBuilder` at inference time. The rank-drop gate in `portfolio.py` only sells a position if its rank drops ≥ 30 places.

## Key conventions

- **No lookahead**: normalization stats, graph edges, and label computation all use strict train-period cutoffs. Tests in `test_momentum_blend_modes.py` and `test_regime_features.py` explicitly verify this.
- **Dynamic graph requires `batch_size=1`**: enforced (with a warning) in `ExperimentConfig.__post_init__` and clamped in `create_data_loaders`.
- **`combined_collate_fn` returns a 7-tuple**: `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates)`. `batch_dates` is `None` in static mode.
- **Ensemble**: `train_multiple_models` trains `config.training.num_models` independent models; final prediction is the mean. Each model gets its own checkpoint under `checkpoints/model_{i}_best.pth`.
- **FRED_API_KEY** env var required when `features.include_global_regime=true` or credit spread features are enabled.
