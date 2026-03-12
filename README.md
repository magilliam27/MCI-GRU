# MCI-GRU: Multi-head Cross-attention and Improved GRU for Stock Prediction

An experiment framework implementing the MCI-GRU model for cross-sectional stock ranking, based on the paper *"MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU"* (Neurocomputing 2025).

The system trains an ensemble of models that learn temporal patterns (via a modified GRU), cross-sectional relationships (via graph attention on a correlation graph), and latent market states (via cross-attention with learned state vectors). It then ranks stocks by predicted forward returns for portfolio construction.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Configuration Reference](#configuration-reference)
- [Data Sources](#data-sources)
- [Training](#training)
- [Paper Trading Pipeline](#paper-trading-pipeline)
- [Testing](#testing)
- [References](#references)

---

## Architecture Overview

```
                          ┌─────────────────────────┐
                          │   Input: OHLCV + Features│
                          │   (batch, stocks, T, F)  │
                          └────────┬────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
              ▼                                         ▼
   ┌─────────────────────┐                ┌──────────────────────┐
   │  Part A: Temporal    │                │  Part B: Cross-      │
   │                      │                │  Sectional (GAT)     │
   │  MultiScaleEncoder   │                │                      │
   │  or ImprovedGRU      │                │  2-layer GATBlock    │
   │  ─────────────────   │                │  on correlation graph│
   │  Fast: full-seq GRU  │                │  ─────────────────── │
   │  Slow: Conv1d → GRU  │                │  in: (stocks, F)     │
   │  Combined: Linear    │                │  out: (stocks, D)    │
   │                      │                │                      │
   │  out: A1 (stocks, D) │                │  out: A2 (stocks, D) │
   └──────────┬───────────┘                └──────────┬───────────┘
              │                                       │
              ▼                                       ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Part C: Market Latent State Learning                       │
   │                                                              │
   │  Learned vectors R1, R2 (num_states x D)                    │
   │  Multi-head cross-attention:                                 │
   │    B1 = CrossAttn(query=A1, key/value=R1)                   │
   │    B2 = CrossAttn(query=A2, key/value=R2)                   │
   └──────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Part D: Prediction                                          │
   │                                                              │
   │  Z = concat[A1, A2, B1, B2]         (stocks, 4D)            │
   │  Z = SelfAttention(Z)               (optional, cross-stock) │
   │  score = GATBlock(Z, graph)          (stocks, 1)             │
   │  output = activation(score)          (stocks,)               │
   └──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
MCI-GRU/
├── run_experiment.py              # Hydra entry point (thin orchestrator)
├── requirements.txt               # Python dependencies
├── configs/                       # Hydra YAML configuration
│   ├── config.yaml                #   Base config
│   ├── data/                      #   Data source presets
│   ├── features/                  #   Feature set presets
│   └── experiment/                #   Experiment presets
├── mci_gru/                       # Core package
│   ├── config.py                  #   Typed dataclass configs
│   ├── pipeline.py                #   Data preparation pipeline
│   ├── models/
│   │   └── mci_gru.py             #   Model architecture
│   ├── features/
│   │   ├── registry.py            #   FeatureEngineer pipeline
│   │   ├── momentum.py            #   Momentum features (binary/continuous/buffered)
│   │   ├── volatility.py          #   Volatility, VIX, RSI, MA features
│   │   ├── credit.py              #   Credit spread features (FRED)
│   │   ├── regime.py              #   Global regime features
│   │   └── base.py                #   Base OHLCV features
│   ├── data/
│   │   ├── data_manager.py        #   Data loading and DataLoader creation
│   │   ├── preprocessing.py       #   Tensor construction and label computation
│   │   ├── reshape.py             #   LSEG data reshape utilities
│   │   ├── lseg_loader.py         #   LSEG/Refinitiv API loader
│   │   ├── fred_loader.py         #   FRED API loader (credit, macro)
│   │   ├── path_resolver.py       #   Project-aware path resolution
│   │   └── universes.py           #   Stock universe definitions
│   ├── training/
│   │   ├── trainer.py             #   Training loop, early stopping, inference
│   │   ├── losses.py              #   ICLoss, CombinedMSEICLoss
│   │   └── metrics.py             #   Evaluation metrics (IC, Sharpe, hit rate)
│   └── graph/
│       └── builder.py             #   Correlation graph construction
├── paper_trade/                   # Paper trading pipeline
│   ├── scripts/
│   │   ├── run_nightly.py         #   Orchestrator (runs all steps)
│   │   ├── refresh_data.py        #   Incremental LSEG data fetch
│   │   ├── infer.py               #   Standalone model inference
│   │   ├── portfolio.py           #   Rank-drop gate portfolio decisions
│   │   ├── track.py               #   Execution simulation + return tracking
│   │   └── report.py              #   Daily markdown/chart reports
│   ├── state/                     #   Persistent state (holdings, ranks)
│   └── Model/                     #   Frozen model checkpoints
├── tests/                         # Test suite + backtest scripts
├── scripts/                       # Utility scripts (data fetching, analysis)
├── docs/                          # Additional documentation
└── notebooks/                     # Colab notebooks
```

---

## Installation

```bash
git clone https://github.com/magilliam27/MCI-GRU.git
cd MCI-GRU
pip install -r requirements.txt
```

**Required dependencies:**
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- Hydra >= 1.3
- NumPy, Pandas, SciPy, scikit-learn, matplotlib, tqdm

**Optional dependencies:**
- `refinitiv-data` — for LSEG/Refinitiv live data (requires Workspace desktop app)
- `fredapi` — for FRED API access (credit spreads, macro data; requires `FRED_API_KEY` env var)

---

## Quick Start

```bash
# Run a baseline experiment (requires data CSV)
python run_experiment.py

# Quick test (2 epochs, 1 model)
python run_experiment.py training.num_epochs=2 training.num_models=1

# Use CSV data source explicitly
python run_experiment.py +data=csv_sp500

# Use a different experiment preset
python run_experiment.py +experiment=with_vix

# Sweep lookback periods
python run_experiment.py --multirun model.his_t=5,10,15,20
```

Outputs are saved to `results/{experiment_name}/{timestamp}/`.

---

## Model Architecture

The MCI-GRU model processes stock data through four stages:

### Part A: Temporal Feature Extraction

Two options, controlled by `model.use_multi_scale`:

**MultiScaleTemporalEncoder** (default, `use_multi_scale: true`):
- **Fast path**: Full-sequence ImprovedGRU capturing short-term patterns
- **Slow path**: Conv1d temporal aggregation followed by ImprovedGRU, capturing longer-term trends
- Outputs are concatenated and projected back to the GRU output dimension

**ImprovedGRU** (`use_multi_scale: false`):
- Multi-layer GRU where the standard reset gate is replaced with an attention mechanism
- The **AttentionResetGRUCell** computes a scaled dot-product attention score between the hidden state (query) and input (key/value), producing a gating signal that modulates how much prior state influences the candidate
- Default: 2 layers with hidden sizes [32, 10]

### Part B: Cross-Sectional Feature Extraction

A **GATBlock** (two-layer Graph Attention Network) processes the most recent day's features for all stocks simultaneously over a correlation-based graph:
- Layer 1: Multi-head GAT (`in_channels` -> `hidden * heads`, concatenated)
- Layer 2: Single-head GAT (`hidden * heads` -> `out_channels`)
- The graph is built from Pearson correlation of trailing returns; edges connect stock pairs with correlation above `judge_value` (default 0.8)

### Part C: Market Latent State Learning

**MarketLatentStateLearner** uses two sets of learned latent state vectors (R1, R2) and multi-head cross-attention:
- B1 = CrossAttention(query=A1, key/value=R1) — enriches temporal features with market state context
- B2 = CrossAttention(query=A2, key/value=R2) — enriches cross-sectional features with market state context
- Default: 32 latent states, 4 attention heads

### Part D: Prediction

The four representations are concatenated: Z = [A1, A2, B1, B2]. Then:
1. **Optional SelfAttention** (`use_self_attention: true`): cross-stock attention over Z, allowing feature groups to interact
2. **Final GATBlock**: produces a scalar score per stock
3. **Activation** (ELU or ReLU): final output

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gru_hidden_sizes` | [32, 10] | Hidden sizes for each GRU layer |
| `hidden_size_gat1` | 32 | Hidden dimension in feature extraction GAT |
| `output_gat1` | 4 | Output dimension of feature extraction GAT |
| `gat_heads` | 4 | Number of attention heads in GAT layers |
| `hidden_size_gat2` | 32 | Hidden dimension in prediction GAT |
| `num_hidden_states` | 32 | Number of learned latent market state vectors |
| `cross_attn_heads` | 4 | Attention heads in market state cross-attention |
| `his_t` | 10 | Historical lookback window (trading days) |
| `label_t` | 5 | Forward return horizon (trading days) |
| `use_multi_scale` | true | Use MultiScaleTemporalEncoder vs plain ImprovedGRU |
| `use_self_attention` | true | Apply cross-stock self-attention before prediction |
| `activation` | elu | Activation function (elu or relu) |
| `slow_kernel` | 5 | Conv1d kernel size for slow temporal path |
| `slow_stride` | 2 | Conv1d stride for slow temporal path |
| `latent_init_scale` | 0.02 | Std for latent state vector initialization |

---

## Feature Engineering

Features are applied by `FeatureEngineer`, configured via `FeatureConfig` or Hydra feature presets.

### Base Features (always included)

| Feature | Description |
|---------|-------------|
| `close` | Closing price |
| `open` | Opening price |
| `high` | Daily high |
| `low` | Daily low |
| `volume` | Trading volume |
| `turnover` | close * volume |

### Momentum Features (`include_momentum: true`)

Based on the "Momentum Turning Points" paper (Goulding, Harvey, Mazzoleni).

| Feature | Description |
|---------|-------------|
| `slow_momentum` | 252-day (12-month) trailing return |
| `fast_momentum` | 21-day (1-month) trailing return |
| `slow_signal` | Signal derived from slow momentum |
| `fast_signal` | Signal derived from fast momentum |
| `momentum_blend` | Speed-selected blend of slow and fast signals |
| `cycle_bull` | 1 if Bull state (slow+, fast+), else 0 |
| `cycle_correction` | 1 if Correction state (slow+, fast-), else 0 |
| `cycle_bear` | 1 if Bear state (slow-, fast-), else 0 |
| `weekly_momentum` | 5-day trailing return (if `include_weekly_momentum`) |
| `weekly_signal` | Signal from weekly momentum |

**Encoding variants** (`momentum_encoding`):
- `binary`: +1/-1 signals (paper default)
- `continuous`: cross-sectional z-scored momentum values
- `buffered`: no-trade zones at extremes (controlled by `buffer_low`/`buffer_high`)

**Blend modes** (`momentum_blend_mode`):
- `static`: fixed weight between slow and fast (default: `momentum_blend_fast_weight=0.5`)
- `dynamic`: paper-style Proposition 9 estimator that adapts speed based on trailing state-conditioned returns (no look-ahead)

### Volatility Features (`include_volatility: true`)

| Feature | Description |
|---------|-------------|
| `volatility_5d` | 5-day annualized realized volatility |
| `volatility_21d` | 21-day annualized realized volatility |
| `vol_ratio` | 5d / 21d ratio (rising = increasing uncertainty) |

### VIX Features (`include_vix: true`)

| Feature | Description |
|---------|-------------|
| `vix` | VIX index level |
| `vix_change` | Daily percentage change in VIX |
| `vix_regime` | 1 if VIX > 10-day MA, else 0 |

### Credit Spread Features (`include_credit_spread: true`)

Sourced from FRED API (requires `FRED_API_KEY`).

| Feature | Description |
|---------|-------------|
| `ig_spread` | Investment-grade OAS (basis points) |
| `hy_spread` | High-yield OAS (basis points) |
| `ig_spread_change` | Daily pct change in IG spread |
| `hy_spread_change` | Daily pct change in HY spread |
| `ig_spread_zscore` | 63-day rolling z-score of IG spread |
| `hy_spread_zscore` | 63-day rolling z-score of HY spread |
| `credit_spread_diff` | HY - IG (risk premium proxy) |

### Global Regime Features (`include_global_regime: true`)

Scalar regime signal from macro/market data (market, yield curve, oil, copper, stock-bond correlation). No look-ahead: month T only compares to historical months.

| Feature | Description |
|---------|-------------|
| `regime_global_score` | Mean distance to all historical regime months |
| `regime_similarity_q20_mean` | Mean distance to most-similar 20% of months |
| `regime_dissimilarity_q80_mean` | Mean distance to most-dissimilar 20% |
| `regime_similarity_spread` | Dissimilarity - similarity (novelty measure) |

### Additional Features

| Flag | Features Added |
|------|---------------|
| `include_rsi` | `rsi_14`, `rsi_normalized` |
| `include_ma_features` | `dist_ma50`, `dist_ma200`, `ma_cross` |
| `include_price_features` | `daily_range`, `body_ratio`, `overnight_return`, `intraday_return` |
| `include_volume_features` | `volume_ma20`, `volume_ratio`, `dollar_volume` |

### Feature Set Presets

| Preset YAML | Features Included |
|-------------|-------------------|
| `base` | Base OHLCV only |
| `with_momentum` | Base + momentum + weekly (default) |
| `momentum_no_weekly` | Base + momentum (no weekly) |
| `with_credit` | Base + momentum + credit spreads |
| `with_regime` | Base + momentum + global regime |
| `full` | All features enabled |

---

## Configuration Reference

Configuration uses [Hydra](https://hydra.cc/) with typed dataclasses. Override any parameter from the command line.

### Config Groups

```
configs/
├── data/           # Data source and date ranges
├── features/       # Feature engineering options
└── experiment/     # Pre-built experiment configurations
```

### Experiment Presets

| Preset | Description |
|--------|-------------|
| `baseline` | Default configuration |
| `paper_faithful` | Matches paper exactly (no multi-scale, no self-attention, ReLU) |
| `hybrid` | Best of paper + code (no multi-scale, self-attention, ELU, rank labels) |
| `with_vix` | Adds VIX and volatility features |
| `with_regime` | Adds global regime features |
| `momentum_continuous` | Continuous momentum encoding |
| `momentum_buffered` | Buffered momentum with no-trade zones |
| `momentum_dynamic` | Cycle-aware dynamic speed selection |
| `momentum_intermediate` | Static 50/50 slow-fast blend |
| `correlation_dynamic` | Updates correlation graph every 6 months |
| `lookback_sweep` | For multi-run lookback period sweeps |
| `russell1000` | Uses Russell 1000 universe |

### Graph Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `judge_value` | 0.8 | Correlation threshold for edge creation |
| `update_frequency_months` | 0 | Graph update interval (0 = static) |
| `corr_lookback_days` | 252 | Trading days of return history for correlation |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Samples per training batch |
| `learning_rate` | 5e-5 | Adam optimizer learning rate |
| `num_epochs` | 100 | Maximum training epochs |
| `num_models` | 10 | Ensemble size (predictions are averaged) |
| `early_stopping_patience` | 10 | Epochs without improvement before stopping |
| `weight_decay` | 1e-3 | L2 regularization weight |
| `gradient_clip` | 1.0 | Max gradient norm (0 = no clipping) |
| `loss_type` | mse | Loss function: `mse`, `ic`, or `combined` |
| `ic_loss_alpha` | 0.5 | IC weight when `loss_type=combined` |
| `label_type` | returns | Label format: `returns` (raw) or `rank` (percentile) |

### Command-Line Override Examples

```bash
# Override individual parameters
python run_experiment.py model.his_t=20 training.batch_size=64

# Use a preset experiment
python run_experiment.py +experiment=with_vix

# Combine presets with overrides
python run_experiment.py +experiment=momentum_dynamic +data=russell1000 training.num_models=5

# Multi-run sweep
python run_experiment.py --multirun model.his_t=5,10,15,20

# Custom output location
python run_experiment.py output_dir=/content/drive/MyDrive/experiments
```

---

## Data Sources

### CSV (default fallback)

Place a CSV at `data/raw/market/sp500_data.csv` with columns: `kdcode`, `dt`, `open`, `high`, `low`, `close`, `volume`.

```bash
python run_experiment.py +data=csv_sp500
```

### LSEG/Refinitiv

Requires Refinitiv Workspace desktop app to be running.

```bash
python run_experiment.py  # Uses LSEG by default via configs/data/sp500.yaml
```

The LSEG loader fetches historical OHLCV data for index constituents in batches with retry logic.

### FRED API

Used for credit spread features (IG/HY OAS) and regime input series (yields, oil, copper). Requires `FRED_API_KEY` environment variable.

```bash
export FRED_API_KEY="your_key_here"
python run_experiment.py +features=with_credit
```

### Index-Level Mode

For survivorship-bias-free experiments using a single index series:

```bash
python run_experiment.py +data=index_level
```

Uses FRED SP500 index or a custom CSV with `dt`, `close` columns.

### Supported Universes

| Universe | Stocks | Config |
|----------|--------|--------|
| S&P 500 | ~500 | `+data=sp500` (default) |
| Russell 1000 | ~1000 | `+data=russell1000` |
| MSCI World | ~1500 | `+data=msci_world` |
| NASDAQ 100 | ~100 | Available via LSEG |

---

## Training

### How It Works

1. **Data preparation** (`mci_gru/pipeline.py`):
   - Load raw data (CSV or LSEG)
   - Apply feature engineering (momentum, volatility, etc.)
   - Per-day mean imputation for missing values
   - Compute normalization stats from training period only
   - Apply 3-sigma clipping + z-score normalization
   - Build sliding-window time-series tensors
   - Build correlation graph
   - Compute forward-return labels

2. **Multi-model training** (`mci_gru/training/trainer.py`):
   - Train `num_models` independent models (default: 10)
   - Each model uses Adam optimizer with L2 regularization
   - Early stopping on validation loss
   - Optional gradient clipping
   - Optional dynamic graph updates during training

3. **Inference and averaging**:
   - Each model produces per-stock scores
   - Final prediction = mean across all models
   - Predictions saved per-model and as averaged ensemble

### Training Outputs

Saved to `{output_dir}/{experiment_name}/{timestamp}/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Full Hydra configuration |
| `training_*.log` | Training logs |
| `run_metadata.json` | Normalization stats, feature columns, stock list |
| `graph_data.pt` | Edge index and weights (for inference) |
| `checkpoints/model_{i}_best.pth` | Best checkpoint per model |
| `predictions_model_{i}/` | Per-model daily predictions |
| `averaged_predictions/` | Ensemble predictions |

### Loss Functions

| Loss | Description |
|------|-------------|
| `mse` | Mean Squared Error (default) |
| `ic` | Negative Pearson IC (maximizes cross-sectional correlation) |
| `combined` | `(1-alpha)*MSE + alpha*(-IC)` blend |

---

## Paper Trading Pipeline

The `paper_trade/scripts/` directory implements a full paper trading system:

### Pipeline Steps (run in order)

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `refresh_data.py` | Append today's LSEG bars to master CSV |
| 2 | `track.py` | Record fills, compute open-to-open returns |
| 3 | `infer.py` | Score universe using frozen checkpoints |
| 4 | `portfolio.py` | Rank-drop gate, generate orders |
| 5 | `report.py` | Daily markdown report + equity curve |

### Running the Pipeline

```bash
# Full nightly run
python paper_trade/scripts/run_nightly.py

# Skip data refresh (data already current)
python paper_trade/scripts/run_nightly.py --skip-refresh

# Dry run (show what would execute)
python paper_trade/scripts/run_nightly.py --dry-run
```

### Standalone Inference

```bash
python paper_trade/scripts/infer.py \
    --model-dir paper_trade/Model/Seed73_trained_to_2062026 \
    --csv data/raw/market/sp500_2019_universe_data_through_2026.csv
```

### Rank-Drop Gate

The portfolio decision engine uses a rank-drop gate: held positions are only sold if their rank deteriorates by at least `--min-rank-drop` (default: 30) positions since the prior rebalance. This reduces turnover and avoids churning positions that are still reasonably ranked.

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_momentum_blend_modes.py -v
python -m pytest tests/test_regime_features.py -v
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_backtest_fairness.py` | Return calculation, simulation timing, rank-drop gate |
| `test_index_level_mode.py` | Index-level config validation, data loading |
| `test_momentum_blend_modes.py` | Static/dynamic blend, Proposition 9, no-lookahead |
| `test_output_management.py` | Output directory structure, logging, Hydra config |
| `test_regime_features.py` | Regime computation, no-lookahead, data contract |

### Backtesting

Full backtesting with transaction cost modeling, multiple testing adjustments, and multi-day holding periods:

```bash
python tests/backtest_sp500.py \
    --predictions_dir path/to/averaged_predictions \
    --auto_save \
    --plot
```

---

## References

1. **MCI-GRU Paper**: "MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU" (Neurocomputing 2025) — [arXiv:2410.20679](https://arxiv.org/abs/2410.20679)

2. **Momentum Turning Points**: Goulding, Harvey, Mazzoleni — "Momentum Turning Points" (SSRN-3489539). Basis for the momentum feature engineering (slow/fast signals, Bull/Correction/Bear cycle states, dynamic speed selection).

3. **Graph Attention Networks**: Velickovic et al. — "Graph Attention Networks" (ICLR 2018). Basis for the GATBlock cross-sectional feature extraction.
