# Default Frozen Experiment Recipe

Last updated: 2026-05-14

Use this recipe for production-style confirmation notebooks and PIT validation
runs unless an experiment is explicitly testing one of these factors.

Recipe slug:

```text
static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1
```

## Hydra Overrides

```text
seed=1729
training.num_models=20
training.num_epochs=100
training.early_stopping_patience=15
training.learning_rate=5e-5
training.lr_scheduler=cosine
training.loss_type=ic
training.label_type=returns
training.selection_metric=val_ic
training.shuffle_train=true
model.label_t=5

graph.judge_value=0.8
graph.update_frequency_months=0
graph.corr_lookback_days=252
graph.top_k=0
graph.top_k_metric=corr
graph.use_multi_feature_edges=true
graph.append_snapshot_age_days=false
graph.use_lead_lag_features=false
graph.drop_edge_p=0.1

features=with_momentum
features.include_momentum=true
features.include_weekly_momentum=true
features.momentum_encoding=binary
features.momentum_blend_mode=static
features.momentum_blend_fast_weight=0.5
features.include_global_regime=true
features.regime_strict=true
features.regime_enforce_lag_days=0
features.regime_include_subsequent_returns=false
features.regime_change_months=12
features.regime_norm_months=120
features.regime_exclusion_months=1
features.regime_similarity_quantile=0.2
features.regime_min_history_months=24
```

## Notes

- `FRED_API_KEY` is required when `features.include_global_regime=true` and
  `features.regime_strict=true`.
- The graph is the static threshold graph, not top-K and not dynamic schedule.
- The objective is pure IC on raw 5-day return labels. Do not substitute rank
  labels for performance scoring unless the rank-label evaluation scale has
  been explicitly audited.
- Full confirmation notebooks should use a 20-model ensemble. Cheap smoke
  notebooks may lower `training.num_models`, `training.num_epochs`, bootstrap
  resamples, and patience, but should keep the recipe's feature, graph, loss,
  label, and selection semantics unless the smoke is explicitly mechanics-only.

Canonical notebook generators that already encode this recipe:

- `scripts/gen_temporal_rolling_backtest_nb.py`
- `scripts/gen_performance_proof_nb.py`
- `scripts/gen_pit_universe_validation_nb.py`
- `scripts/gen_pit_masked_panel_2022_2025_nb.py`
