# Default Frozen Experiment Recipe

Last updated: 2026-10-04

Use this recipe for production-style confirmation notebooks and PIT validation
runs unless an experiment is explicitly testing one of these factors.

> **The universe changed on 2026-08-08. Recipe-labelled evidence produced before
> and after that date is not directly comparable.**
>
> Until then this document named no data config, so it silently inherited
> whatever `configs/config.yaml` composed. When that base default moved from
> `data: sp500` to `data: gics_top10_110_2016`, the recipe's effective universe
> moved with it and this document did not change. It now pins its data config
> explicitly, so its meaning no longer depends on a mutable default.
>
> | | before 2026-08-08 | from 2026-08-08 |
> |---|---|---|
> | data config | `configs/data/sp500.yaml` (inherited) | `configs/data/gics_top10_110_2016.yaml` (pinned) |
> | source / universe | `lseg`, ~500 names | `csv`, ~110 names |
> | train start | 2019-01-01 | 2016-01-04 |
> | `use_pit_universe` / mode | `false` / `row_filter` | `true` / `masked_panel` |
>
> The pre-recipe `seed_results/` (retired to tag `archive/pre-cleanup-2026-09`) were
> produced under the inherited S&P 500 universe and are not comparable with
> recipe-labelled runs.

> **The model changed on 2026-10-04, before the first admitted run (#187).**
>
> Until then the recipe named no `model.*` key, so it inherited the legacy
> forms that `configs/config.yaml` keeps for checkpoint compatibility. It now
> pins the fixed forms, and the slug gained a suffix to say so (#273).
>
> | | before 2026-10-04 | from 2026-10-04 |
> |---|---|---|
> | `model.market_latent_mode` | `static` (inherited): the B1/B2 latents are fixed parameters that cannot see the date, issue #198 | `data_dependent` (pinned): the latents read each date's PIT-active names first |
> | `model.cross_section_block` | `legacy` (inherited): the cross-stock block replaces `z` and discards most of the variation between stocks (#197) | `residual` (pinned): the block corrects `z` as `z + Attn(LayerNorm(z))` |
> | `model.gru_attn_layer_widths` | `shared` (inherited): `gru_hidden_sizes: [32, 10]` built two GRU layers of width 10 (#131) | `per_layer` (pinned): a 32-wide layer feeding a 10-wide one |
> | slug | `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1` | `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1__latents-data__xsec-residual__gru-32-10` |
>
> All three forms hold different parameters from the legacy ones, so a checkpoint
> trained under the earlier slug does not load into this recipe's model.

Recipe slug:

```text
static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1__latents-data__xsec-residual__gru-32-10
```

## Hydra Overrides

```text
data=gics_top10_110_2016

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
model.temporal_encoder=gru_attn
model.use_multi_scale=true
model.gru_hidden_sizes=[32,10]
model.use_nn_multihead_attention=true
model.market_latent_mode=data_dependent
model.cross_section_block=residual
model.gru_attn_layer_widths=per_layer

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

- `data=gics_top10_110_2016` is pinned deliberately. The recipe must not inherit
  its universe from `configs/config.yaml`; a recipe whose data moves when a
  default moves is not frozen. `tests/test_default_experiment_recipe.py` pins
  that the selector is present and names a config that exists.
- That config sets `use_pit_universe: true` against a `pit_universe_csv` that is
  **not committed**, with `pit_min_scoreable_stocks: 104` and
  `pit_breadth_policy: error`. Confirmation runs must supply that CSV. Runs that
  bring their own panel instead must pass `data.use_pit_universe=false`, as
  `scripts/ci_smoke.py` does.
- `FRED_API_KEY` is required when `features.include_global_regime=true` and
  `features.regime_strict=true`.
- The seven `model.*` keys after `model.label_t` are pinned for the same reason
  as the data config. `market_latent_mode=data_dependent` and
  `cross_section_block=residual` are the corrected forms from #198 and #197, and `gru_attn_layer_widths=per_layer` makes
  `gru_hidden_sizes: [32, 10]` mean a 32-wide layer then a 10-wide one, the
  maintainer's decision on #131. `configs/config.yaml` keeps the legacy forms as
  its defaults so older checkpoint directories still rebuild. Data-dependent
  latents need `use_nn_multihead_attention=true` and `ModelConfig` refuses the
  combination without it, so that key is pinned too. `temporal_encoder=gru_attn`,
  `use_multi_scale=true` and `gru_hidden_sizes=[32,10]` are pinned so the slug's
  `gru-32-10` does not move if a base default does.
  `tests/test_default_experiment_recipe.py` composes this block the way
  `run_experiment.py` does and checks the model it builds.
- The graph is the static threshold graph, not top-K and not dynamic schedule.
- The objective is pure IC on raw 5-day return labels. Do not substitute rank
  labels for performance scoring unless the rank-label evaluation scale has
  been explicitly audited.
- Full confirmation notebooks should use a 20-model ensemble. Cheap smoke
  notebooks may lower `training.num_models`, `training.num_epochs`, bootstrap
  resamples, and patience, but should keep the recipe's feature, graph, loss,
  label, and selection semantics unless the smoke is explicitly mechanics-only.

Notebook generators that encode the recipe as it stood before 2026-10-04 (the
earlier slug, with the legacy model forms). They are historical and have not
been moved to this recipe:

- `scripts/gen_temporal_rolling_backtest_nb.py`
- `scripts/gen_performance_proof_nb.py`
- `scripts/gen_pit_universe_validation_nb.py`
- `scripts/gen_pit_masked_panel_2022_2025_nb.py`
- `scripts/gen_long_history_pit_eval_nb.py`
- `scripts/gen_pit_repeated_seed_replication_nb.py`
- `scripts/gen_sp500_pit_gics_top10_baseline_nb.py`

Where one of them calls itself the frozen recipe, it means the earlier slug.
