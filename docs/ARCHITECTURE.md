# Architecture

> This document is the deep architectural reference for the MCI-GRU system.
> Start here when you need to understand how data flows, how the model works,
> or how components connect.

## Data Flow (end to end)

```
CSV / LSEG / FRED
 → DataManager          (mci_gru/data/data_manager.py)
 → FeatureEngineer      (mci_gru/features/registry.py)
 → prepare_data()       (mci_gru/pipeline.py)   ← normalization, windowing, graph build
 → create_data_loaders  (data_manager.py)        ← CombinedDataset + combined_collate_fn
 → Trainer / train_multiple_models               ← ensemble of N independent models
 → averaged_predictions/
```

### Step-by-step

1. **Raw data loading** — `DataManager.load_data()` reads OHLCV from CSV or LSEG API.
   Output: a pandas DataFrame with columns `kdcode, dt, open, high, low, close, volume`.

2. **Feature engineering** — `FeatureEngineer.add_features()` iterates the feature registry
   (momentum, volatility, VIX, credit spreads, regime) and appends columns.
   Each feature module lives in `mci_gru/features/` and follows a common interface.

3. **Normalization** — `pipeline.py` computes per-feature z-score stats using **training dates
   only**, then applies 3-sigma clipping + standardization across all splits.
   Stats (plus **`data_file_sha256`** / size / mtime for `data.filename` when the file exists) are persisted in `run_metadata.json` for inference reuse and data provenance.

4. **Windowing** — Sliding windows of shape `(days, stocks, his_t, features)` are constructed.
   Labels are `label_t`-day forward returns (or rank percentiles if `label_type=rank`).

5. **Graph construction** — `GraphBuilder` computes Pearson correlation over trailing returns.
   Pairs with `|corr| > judge_value` get edges (or top-K selection when `top_k > 0`).
   By default **`use_multi_feature_edges=true`**: edge attributes are `(E, 4)` `[corr, |corr|, corr², rank_pct]`; legacy `(E,)` scalars when `false`.
   Static mode builds once; dynamic mode rebuilds every N months per batch date.

6. **DataLoaders** — `create_data_loaders` wraps tensors in `CombinedDataset`.
   The `combined_collate_fn` returns a 7-tuple:
   `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates)`.
   `batch_dates` is `None` in static mode.

7. **Training** — `Trainer.train()` uses **AdamW**, optional **cosine LR schedule** with linear warmup (`TrainingConfig.lr_scheduler`, `warmup_steps`), **CUDA AMP** when `use_amp` and a GPU are available, and early stopping / checkpoint selection by **`selection_metric`** (`val_ic` or `val_loss`). Default training loss is **`combined`** (MSE + IC). `train_multiple_models` repeats training **N** times with **`set_seed(seed + model_id)`** per member, then averages predictions.

8. **Inference** — Each model produces per-stock scalar scores. The ensemble mean is the
   final prediction, saved as CSV files in `averaged_predictions/`.

9. **Evaluation trust layer** — `mci_gru/evaluation/` provides shared IC,
   Newey-West Sharpe, moving-block bootstrap CIs, top-k returns, rank-drop
   decisions, and feature-drift metrics. `run_experiment.py` writes
   `evaluation_summary.json` and train-only `feature_reference.json`.

## Model Architecture (mci_gru/models/mci_gru.py)

Four parallel streams whose outputs are concatenated before the final predictor:

```
Input: (batch, stocks, his_t, features)
         │
    ┌────┴────┐
    ▼         ▼
   A1        A2
 Temporal   Cross-sectional
 (GRU)      (GAT on graph)
    │         │
    ▼         ▼
   B1        B2
 CrossAttn  CrossAttn
 (A1 × R1)  (A2 × R2)
    │         │
    └────┬────┘
         ▼
  Z = [A1, A2, B1, B2]
         │
   SelfAttention (optional)
         │
   Final GATBlock → scalar score per stock
```

### A1: Temporal stream

- **MultiScaleTemporalEncoder** (default): parallel fast-path (full-seq GRU) and
  slow-path (Conv1d → GRU). Outputs concatenated and projected.
- **ImprovedGRU** (fallback): multi-layer GRU with AttentionResetGRUCell replacing the
  standard reset gate with scaled dot-product attention.

### A2: Cross-sectional stream

- 2-layer GATBlock over the correlation graph using the most recent day's features.
- Layer 1: multi-head GAT (concat). Layer 2: single-head GAT.

### B1/B2: Market latent state

- `MarketLatentStateLearner` maintains learned vectors R1, R2 of shape `(num_states, D)`.
- Multi-head cross-attention: B1 = CrossAttn(query=A1, kv=R1), B2 = CrossAttn(query=A2, kv=R2).

### Prediction head

- Concatenate `[A1, A2, B1, B2]` → optional cross-stock SelfAttention → final GATBlock → activation.

## Graph (mci_gru/graph/builder.py)

| Mode | Config | Behavior |
|------|--------|----------|
| Static | `update_frequency_months=0` | Built once before training. Fixed tensors. |
| Dynamic | `update_frequency_months>0` | `GraphSchedule` precomputes snapshots every N months; collate does O(log n) lookup per sample. Any batch size works. |

The graph is a Pearson-correlation adjacency: trailing `corr_lookback_days` (default 252)
returns are used. When `graph.top_k == 0`, edges connect pairs with `|corr| > judge_value` (default 0.8); when `top_k > 0`, each node keeps its top-K neighbours by `top_k_metric`.

**`GraphSchedule`** (introduced in commit `f873f84`): when `update_frequency_months > 0`,
`GraphBuilder.precompute_snapshots()` builds all graph snapshots up-front during
`prepare_data()`. Each snapshot uses only data **before** its valid-from date (no lookahead).
The `combined_collate_fn` resolves the correct snapshot per sample via `bisect`, eliminating
the previous `batch_size=1` constraint.

## Config System (Hydra)

```
configs/
├── config.yaml          ← base defaults
├── data/                ← DataConfig overrides (sp500, russell1000, temporal_2019, ...)
├── features/            ← FeatureConfig overrides (base, with_momentum, full, ...)
└── experiment/          ← full experiment presets (paper_faithful, hybrid, ...)
```

All configs map to typed dataclasses in `mci_gru/config.py`. `ExperimentConfig` is the
root, containing `DataConfig`, `FeatureConfig`, `GraphConfig`, `ModelConfig`,
`TrainingConfig`, and `TrackingConfig`. On construction, **`ExperimentConfig` validates calendar gaps** between train/val and val/test are **strictly greater than `model.label_t`** days unless `data.skip_embargo_check=true` (discouraged). **`pipeline._build_tensors`** aligns `stock_features_*` rows to label dates so embargo gaps do not desynchronize time-series tensors from graph features.

Override from CLI: `python run_experiment.py model.his_t=20 training.loss_type=ic`

## Paper Trading (paper_trade/)

Uses **frozen** checkpoints from `paper_trade/Model/`. The inference path:

1. `infer.py` loads `run_metadata.json` (norm stats, feature list, stock list)
   and `graph_data.pt` (precomputed static graph). It does **not** call `GraphBuilder`.
2. `portfolio.py` applies the rank-drop gate: only sell if rank drops ≥ 30 places.
3. `track.py` records fills and computes open-to-open returns.
4. `monitor.py` compares latest normalized inference features to the train-window
   feature reference and writes `feature_drift.json` / `feature_drift.csv`.
5. `report.py` generates daily markdown reports, including feature drift when available.
6. `run_nightly.py` orchestrates all steps in order.

## Package Layout

```
mci_gru/
├── __init__.py          ← version, public exports
├── config.py            ← ExperimentConfig and sub-configs (dataclasses)
├── pipeline.py          ← prepare_data(): load → features → normalize → window → graph
├── models/
│   ├── __init__.py      ← create_model(), StockPredictionModel
│   └── mci_gru.py       ← StockPredictionModel, GATBlock, ImprovedGRU, MarketLatentStateLearner
├── data/
│   ├── data_manager.py  ← DataManager, CombinedDataset, combined_collate_fn, create_data_loaders
│   ├── preprocessing.py ← generate_time_series_features, generate_graph_features, compute_labels
│   ├── lseg_loader.py   ← LSEG/Refinitiv API data fetching
│   ├── fred_loader.py   ← FRED API data fetching (credit, macro)
│   ├── reshape.py       ← LSEG data reshape utilities
│   ├── path_resolver.py ← project-aware path resolution
│   └── universes.py     ← stock universe definitions (SP500, R1000, MSCI)
├── features/
│   ├── registry.py      ← FeatureEngineer (orchestrates feature modules)
│   ├── base.py          ← base OHLCV features (turnover)
│   ├── momentum.py      ← MTP momentum (binary/continuous/buffered, static/dynamic blend)
│   ├── volatility.py    ← realized vol, VIX, RSI, MA features
│   ├── credit.py        ← credit spread features from FRED
│   └── regime.py        ← global regime similarity features
├── graph/
│   └── builder.py       ← GraphBuilder + GraphSchedule (Pearson correlation, static/dynamic)
├── evaluation/
│   ├── statistics.py    ← IC, Newey-West Sharpe, moving-block bootstrap CIs
│   ├── portfolio.py     ← top-k returns, turnover, rank-drop gate
│   └── drift.py         ← PSI / KS-style feature drift metrics
└── training/
    ├── trainer.py       ← Trainer, train_multiple_models, early stopping
    ├── losses.py        ← ICLoss, CombinedMSEICLoss
    └── metrics.py       ← evaluation metrics (IC, Sharpe, hit rate)
```
