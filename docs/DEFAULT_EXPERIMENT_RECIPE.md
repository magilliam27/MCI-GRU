# Default Frozen Experiment Recipe

Last updated: 2026-10-08

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

> **The price panel changed on 2026-10-08, before the first admitted run (#187).**
>
> The LSEG subscription has ended (#232), so the owner chose to train the first
> run on EODHD prices (#281). The universe is unchanged: the same point-in-time
> membership file, byte for byte. Only the price and volume panel changes source
> (#283). The slug is unchanged, because it has never encoded the data config.
>
> | | before 2026-10-08 | from 2026-10-08 |
> |---|---|---|
> | data config | `configs/data/gics_top10_110_2016.yaml` | `configs/data/gics_top10_110_2016_eodhd.yaml` |
> | price panel | LSEG export, `..._lseg_20150101_20260731.csv` | EODHD pull of 2026-10-08, `..._eodhd_20150101_20260731.csv` |
> | input package | LSEG r1 manifest | `data/manifests/sp500_pit_gics_top10_mcap_monthly_20160104_20260731_eodhd.r1.json` |
> | names scored in 2016-01..2017-08 | at most 110 | at most 109: the pre-merger DuPont (`DD.N^I17`) has no EODHD history |
>
> Before the package was accepted, each name in the panel had its EODHD daily
> returns checked against the LSEG panel over the span the universe needs. The
> prices are split-adjusted, plus the spin-offs, share conversions and special
> dividends declared in `data/mappings/eodhd_symbols_gics_top10_110_2016.json`.
> Regular dividends are not applied, which matches the LSEG panel's basis.
> Results on the two panels are close but are not the same run.

Recipe slug:

```text
static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1__latents-data__xsec-residual__gru-32-10
```

## Hydra Overrides

```text
data=gics_top10_110_2016_eodhd
data.auxiliary_snapshot_mode=capture

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

- `data=gics_top10_110_2016_eodhd` is pinned deliberately. The recipe must not
  inherit its universe from `configs/config.yaml`; a recipe whose data moves when
  a default moves is not frozen. `tests/test_default_experiment_recipe.py` pins
  that the selector is present and names a config that exists. The base default
  stays the LSEG config `gics_top10_110_2016`, which has the same universe,
  windows and breadth floor; the two differ only in the price panel and its
  manifest.
- The pre-merger DuPont (`DD.N^I17`, a member from 2016-01-04 to 2017-08-31) has
  no EODHD history and is absent from the panel; the owner chose to leave it out
  (2026-10-08). The masked-panel pipeline drops a member with no rows from the
  stock axis, so those months score at most 109 names, still above
  `pit_min_scoreable_stocks: 104`. The manifest records the gap in
  `provenance.unknowns`.
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
- No cessation (delisting) event file is declared:
  `data.pit_cessation_events_csv` stays `null` (decided 2026-10-04, #273). The
  EODHD panel keeps the LSEG identifiers. Five of its names carry LSEG delisted
  suffixes (ATVI.OQ^J23, DOW.N^I17, HES.N^G25, PXD.N^E24, WBA.OQ^H25); DD.N^I17,
  the sixth in the universe, has no rows. A declared cessation changes only
  `eligible`, and the traded population also needs a close on the date itself
  (`tradable = eligible & feature_ready & price_observed` in `build_pit_masks`).
  So while a delisted name has no close after its last real session, the file
  would change no prediction, label, loss or metric. It would change only the
  input manifest and how the PIT eligibility report classes those sessions
  (`cessation_excluded` rather than `price_gap`).
  `tests/test_first_run_cessation.py` pins this, with a control showing that
  carried closes after delisting do change the masks.
  - **Precondition, checked on the real panel before the run:**
    `python scripts/check_delisted_tails.py`. By default it reads the panel of
    the data config this recipe selects. Exit 1 means a delisted name ends
    in repeated closes or zero volume, or in a single row that repeats the
    previous close with no volume. That name then needs a cessation row dated
    to its real last session, or the carried rows removed at source. The #223
    frozen-price rule does not catch this, because it flags only a history that
    is constant throughout. On 2026-10-04 it passed against the LSEG panel: exit
    0, all six delisted names ended cleanly (0 repeated closes, 0 zero-volume
    sessions), and with `--all` so did all 206 names. That result does not carry
    over to the EODHD panel. #281's acceptance compares returns only on LSEG
    sessions, up to 10 days after each name's last window, so a row after a
    delisting is unchecked there.
  - **On the EODHD panel the precondition is not yet met.** Run on 2026-10-08
    against the published r1 panel (sha256 matches the manifest), the check
    flags three names. Each keeps one row on its delisting day that repeats the
    previous close at zero volume, one session after its last LSEG close:
    ATVI.OQ^J23 (2023-10-13), HES.N^G25 (2025-07-18) and WBA.OQ^H25
    (2025-08-28). Each row keeps that name tradable for one session. It adds a
    0-return, zero-volume row to the features and to that date's cross-section,
    and HES and WBA fall in the 2025 test window. Labels on those rows are
    unobservable, so the loss is unchanged. The fix belongs to the price
    package (#281): drop the rows at source, or declare cessations with
    `known_from` evidence before each delisting day. PSKY.OQ, a live name, also ends in
    two zero-volume repeats, in its last rows to 2026-07-31. That is after the
    test window and its labels, so it does not touch this run.
  - A cessation dated on a name's final session and known by 20:00 New York that
    day would also drop that one session from the cross-section. Its own loss is
    unchanged, but other names' scores on that date move slightly. The likely
    case was DD and DOW at the DowDuPont merger close on 2017-08-31; on the EODHD
    panel only DOW remains. The
    2017-09-01..09-13 gap that the data config calls a pricing gap is most likely
    those two names after the merger (inferred, not checked against the CSV).
  - The last sessions before each delisting have unobservable 5-day labels,
    which are omitted and counted until #129 values terminal outcomes. A
    cessation file would not change that.
- `data.auxiliary_snapshot_mode=capture` was added on 2026-10-04, when the owner
  chose to capture the regime inputs for the first run: five FRED series and the
  S&P 500 index file named by `features.regime_market_csv` (#276). It changes no
  input value: the run reads the same inputs, and it also keeps their exact
  bytes. With no `data.auxiliary_snapshot_directory`, they go to
  `input_snapshots/` in the run's output folder. Each window's input attachment
  (#208) then binds every regime read to its snapshot, so the run's inputs verify
  as `complete`. In `source` mode they read as `incomplete`. `complete` means
  every input is declared and bound to a retained manifest; it does not check
  that the snapshot bytes are still on disk (preservation is #207). Keep that
  folder with the run, because it is the only copy of what FRED returned and of
  the index file the run read, and replay needs it.
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
