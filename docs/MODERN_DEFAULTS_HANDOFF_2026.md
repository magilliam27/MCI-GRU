# MCI-GRU Modern Defaults Handoff

Last updated: 2026-04-29

This is a fresh-chat handoff for the MCI-GRU ablation and 2026 holdout work.

## Current Recommendation

`modern_defaults` is the leading candidate.

The 2026 holdout run was the first clean post-selection check using the new 2026 data slice. It did not fully clear the portfolio-return confidence gate, but it did produce a statistically positive ranking signal. That is enough to continue, but the next work should narrow around `modern_defaults` instead of broad graph exploration.

## Exact Modern Defaults Config

This is the effective repo default for `modern_defaults` before notebook/runtime overrides. In the ablation notebook, `modern_defaults` means no `+experiment=...` override; it uses `configs/config.yaml` with `data=sp500` and `features=with_momentum`.

```yaml
defaults:
  - data: sp500
  - features: with_momentum
  - _self_

data:
  universe: sp500
  source: lseg
  filename: data/raw/market/sp500_data.csv
  train_start: "2019-01-01"
  train_end: "2023-12-31"
  val_start: "2024-01-08"
  val_end: "2024-12-31"
  test_start: "2025-01-08"
  test_end: "2025-12-31"

features:
  base_features:
    - close
    - open
    - high
    - low
    - volume
    - turnover
  include_momentum: true
  include_weekly_momentum: true
  momentum_encoding: binary
  momentum_blend_mode: static
  momentum_blend_fast_weight: 0.5
  momentum_dynamic_correction_fast_weight: 0.15
  momentum_dynamic_rebound_fast_weight: 0.7
  momentum_dynamic_lookback_periods: 0
  momentum_dynamic_min_history: 252
  momentum_dynamic_min_state_observations: 3
  momentum_buffer_low: 0.1
  momentum_buffer_high: 0.9
  include_volatility: false
  include_vix: false
  include_credit_spread: false
  include_global_regime: false
  regime_change_months: 12
  regime_norm_months: 120
  regime_clip_z: 3.0
  regime_exclusion_months: 1
  regime_similarity_quantile: 0.2
  regime_min_history_months: 24
  regime_strict: false
  regime_lseg_market_ric: .SPX
  regime_lseg_copper_ric: .MXCOPPFE
  regime_lseg_yield_10y_ric: US10YT=RR
  regime_lseg_yield_3m_ric: US3MT=RR
  regime_lseg_oil_ric: CLc1
  regime_inputs_csv: null
  regime_enforce_lag_days: 0
  include_rsi: false
  include_ma_features: false
  include_price_features: false
  include_volume_features: false

model:
  his_t: 10
  label_t: 5
  gru_hidden_sizes: [32, 10]
  hidden_size_gat1: 32
  output_gat1: 4
  gat_heads: 4
  hidden_size_gat2: 32
  num_hidden_states: 32
  cross_attn_heads: 4
  slow_kernel: 5
  slow_stride: 2
  use_multi_scale: true
  use_self_attention: true
  activation: elu
  output_activation: "none"
  latent_init_scale: 0.02
  use_group_type_embed: true
  use_trunk_regularisation: true
  trunk_dropout: 0.1
  use_nn_multihead_attention: true
  temporal_encoder: gru_attn
  use_a1_a2_cross_attention: false
  cross_a2_num_heads: 4

graph:
  judge_value: 0.8
  update_frequency_months: 0
  corr_lookback_days: 252
  top_k: 0
  top_k_metric: corr
  use_multi_feature_edges: true
  drop_edge_p: 0.1
  append_snapshot_age_days: false
  use_lead_lag_features: false
  lead_lag_days: [1, 2, 3, 5]
  use_sector_relation: false
  sector_map_csv: null
  sector_top_k: 10

training:
  batch_size: 32
  learning_rate: 5e-5
  num_epochs: 100
  num_models: 10
  early_stopping_patience: 10
  weight_decay: 1e-3
  gradient_clip: 1.0
  loss_type: combined
  ic_loss_alpha: 0.5
  label_type: returns
  warmup_steps: 1000
  lr_scheduler: cosine
  use_amp: true
  selection_metric: val_ic
  walkforward:
    enabled: false
    window_train_years: 4
    window_val_months: 6
    test_span_months: 3
    step_months: 6
    expanding: false
    max_windows: null

evaluation:
  top_k_values: [10, 20, 50, 100]
  bootstrap_enabled: true
  bootstrap_resamples: 1000
  bootstrap_seed: 42
  ci_level: 0.95
  block_size: null
  sharpe_method: newey_west
  newey_west_lags: null

tracking:
  enabled: true
  tracking_uri: mlruns
  experiment_name: null
  run_name: null
  log_artifacts: true
  log_checkpoints: true
  log_predictions: false

experiment_name: baseline
output_dir: results
seed: 42

hydra:
  run:
    dir: ${output_dir}/${experiment_name}/${now:%Y%m%d_%H%M%S}
  sweep:
    dir: ${output_dir}/${experiment_name}
    subdir: run_${hydra.job.num}
  job:
    chdir: false
```

## Exact 2026 Holdout Modern Defaults Command Shape

The 2026 holdout notebook overrides the base defaults like this:

```bash
python -u run_experiment.py \
  features=with_momentum \
  data.source=csv \
  data.filename=sp500_2019_universe_data_through_2026.csv \
  data.train_start=2019-01-01 \
  data.train_end=2024-12-31 \
  data.val_start=2025-01-08 \
  data.val_end=2025-12-31 \
  data.test_start=2026-01-08 \
  data.test_end=2026-04-30 \
  training.num_models=20 \
  training.num_epochs=100 \
  training.early_stopping_patience=15 \
  training.batch_size=32 \
  training.learning_rate=5e-5 \
  training.walkforward.enabled=false \
  evaluation.bootstrap_resamples=1000 \
  tracking.enabled=true \
  tracking.log_predictions=false \
  seed=42 \
  experiment_name=holdout_2026_modern_defaults
```

Important: no graph override is applied for `modern_defaults`. That means:

```yaml
graph.top_k: 0
graph.top_k_metric: corr
graph.use_multi_feature_edges: true
graph.update_frequency_months: 0
graph.drop_edge_p: 0.1
```

## What We Changed

- Pulled and overwrote the local repo from newest `main`.
- Fixed the Colab GPU preflight path so notebook runs fail loudly if CUDA is not available when expected.
- Added and pushed a finalist confirmation workflow to `notebooks/ablation_evaluation_loop_colab.ipynb`.
- Added and pushed a 2026 holdout workflow at the bottom of `notebooks/ablation_evaluation_loop_colab.ipynb`.
- Updated tests in `tests/test_ablation_notebook_gpu_preflight.py` so the notebook cells are parseable and assert the full-run budgets.
- Pushed the updated notebook/tests to `main`.

## Confirmation Workflows Now In The Notebook

### Final Confirmation

Purpose: walk-forward confirmation across the existing historical sample.

Defaults:

```python
FINAL_CONFIRMATION_NUM_MODELS = 20
FINAL_CONFIRMATION_NUM_EPOCHS = 100
FINAL_CONFIRMATION_EARLY_STOPPING_PATIENCE = 15
FINAL_CONFIRMATION_BOOTSTRAP_RESAMPLES = 1000
FINAL_CONFIRMATION_MAX_WINDOWS = 5
```

Primary runs:

- `modern_defaults`: base repo config.
- `topk30_abs_graph`: `graph.top_k=30`, `graph.top_k_metric=abs_corr`.

Optional challenger:

- `topk10_abs_graph`: `graph.top_k=10`, `graph.top_k_metric=abs_corr`.

### 2026 Holdout

Purpose: clean post-selection holdout using the first 2026 data slice.

Defaults:

```python
HOLDOUT_2026_NUM_MODELS = 20
HOLDOUT_2026_NUM_EPOCHS = 100
HOLDOUT_2026_EARLY_STOPPING_PATIENCE = 15
HOLDOUT_2026_BOOTSTRAP_RESAMPLES = 1000
```

Split:

```text
Train: 2019-01-01 through 2024-12-31
Validation: 2025-01-08 through 2025-12-31
Test: 2026-01-08 through 2026-04-30
```

Primary runs:

- `modern_defaults`
- `topk30_abs_graph`

Optional challenger:

- `topk10_abs_graph`

## Latest 2026 Holdout Results

Drive run:

```text
MCI-GRU-Ablations/holdout_2026/20260429_005753
```

Decision table:

```text
holdout_2026_decision_table.csv
```

### modern_defaults

```text
status: OK
recommendation: BORDERLINE
models_trained: 20
avg_ic: 0.08267176117401785
avg_ic_ci_lower: 0.02294992646383297
avg_ic_ci_upper: 0.1376530176736598
avg_spearman_corr: 0.06670270721987877
median_spearman_corr: 0.05808384367276028
ic_ir: 0.5616463620677435
top_20_return: 0.010246569890654289
top_20_sharpe_newey_west: 2.9286369999626536
top_20_return_ci_lower: -0.0028990898393128116
top_20_return_ci_upper: 0.02376637687744536
top_50_return: 0.00721991286017538
top_50_sharpe_newey_west: 2.7806969019401744
top_50_return_ci_lower: -0.0026704345750703583
top_50_return_ci_upper: 0.016996816840877442
hit_rate: 0.5219687092568448
long_short_spread: 0.007444628747180104
```

### topk30_abs_graph

```text
status: OK
recommendation: BORDERLINE
models_trained: 20
avg_ic: 0.06470200966804907
avg_ic_ci_lower: 0.00623838137327441
avg_ic_ci_upper: 0.1165765434954337
avg_spearman_corr: 0.05703860626174089
median_spearman_corr: 0.0733163695464373
ic_ir: 0.4855972964358842
top_20_return: 0.002378740337522826
top_20_sharpe_newey_west: 1.105839011264754
top_20_return_ci_lower: -0.005997079693382913
top_20_return_ci_upper: 0.01063135712624115
top_50_return: 0.005164683762127354
top_50_sharpe_newey_west: 2.4226256817520513
top_50_return_ci_lower: -0.0035163308674041414
top_50_return_ci_upper: 0.012834919017419786
hit_rate: 0.4864406779661017
long_short_spread: 0.005093487387057394
```

## Interpretation

The 2026 result is a positive surprise compared with the earlier uncertainty. Both candidates have positive IC confidence intervals, which means the ranking signal is showing up out of sample.

`modern_defaults` is clearly stronger than `topk30_abs_graph` on this holdout:

- Higher average IC.
- Higher IC confidence lower bound.
- Higher top-20 and top-50 returns.
- Higher top-20 and top-50 Newey-West Sharpe.
- Better hit rate.

The reason this is still `BORDERLINE` rather than a full green light is that the portfolio-return confidence intervals still include zero. The ranking signal is statistically positive; the tradable portfolio edge is promising but not yet fully proven.

## Recommended Next Step

Stop broad architecture/graph exploration for now.

Next work should focus on `modern_defaults`:

1. Add or inspect turnover and cost/slippage-adjusted returns for the 2026 holdout.
2. Break 2026 results down by month to check whether one month is carrying the signal.
3. Compare sector/position concentration for top-20 and top-50 picks.
4. Continue forward/paper-trading confirmation as more 2026 data arrives.
5. Only revisit graph variants if `modern_defaults` fails the trading-realism checks.

## Useful Files

```text
notebooks/ablation_evaluation_loop_colab.ipynb
tests/test_ablation_notebook_gpu_preflight.py
configs/config.yaml
configs/data/sp500.yaml
configs/features/with_momentum.yaml
docs/ARCHITECTURE.md
```

