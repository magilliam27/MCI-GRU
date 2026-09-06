# Architecture

> This document is the deep architectural reference for the MCI-GRU system.
> Start here when you need to understand how data flows, how the model works,
> or how components connect.
>
> Every statement here is written from the current code, executable configuration,
> and tests. When this document disagrees with them, they win. Historical plans and
> research reports are not implementation evidence.

## Data Flow (end to end)

```
CSV / LSEG (+ FRED, VIX, credit, regime auxiliaries)
 → DataManager.load()          (mci_gru/data/data_manager.py)
 → FeatureEngineer.transform() (mci_gru/features/registry.py)
 → prepare_data()              (mci_gru/pipeline.py)   ← impute, PIT, normalize, window, graph
 → create_data_loaders()       (mci_gru/data/data_manager.py)  ← CombinedDataset + combined_collate_fn
 → train_multiple_models()     (mci_gru/training/ensemble.py)  ← N independently seeded models
 → averaged_predictions/
```

`run_experiment.py` is the composition root. It resolves Hydra config into an
`ExperimentConfig`, generates one or more walk-forward windows, and runs the
preparation → training → prediction → evaluation sequence once per window.

### Step-by-step

1. **Raw data loading** — `DataManager.load()` dispatches on `data.source`:
   `csv` reads the configured file, `lseg` fetches the universe through
   `mci_gru/data/lseg_loader.py`. Downstream stock-level code expects
   `kdcode, dt, open, high, low, close, volume` columns. VIX, credit-spread, and
   global-regime series are loaded separately by `load_vix()`,
   `load_credit_spreads()`, and `load_regime_inputs()`; FRED
   (`mci_gru/data/fred_loader.py`) is an auxiliary and index source, not a third
   stock-panel mode.

2. **Feature engineering** — `FeatureEngineer.transform()` composes feature
   functions imported from `mci_gru/features/`: base OHLCV/turnover always,
   then momentum, volatility, volatility targeting, VIX, credit, regime, RSI,
   moving-average, price, and volume features as configured.
   `get_feature_columns()` declares the resulting column list. Feature modules
   are functions called in a fixed order, not classes registered through a
   plugin interface.

3. **Imputation** — `impute_feature_nans_by_day()` (`mci_gru/data/transforms.py`)
   fills each feature from its same-day cross-sectional mean, then zero-fills
   whatever remains.

4. **PIT mode resolution** — with `data.use_pit_universe=true`, membership
   intervals are loaded from `data.pit_universe_csv`. `row_filter` (the default
   mode) drops rows outside `[valid_from, valid_to]`. `masked_panel` keeps every
   row and defers eligibility to daily masks (step 8).

5. **Normalization** — `data.normalisation=zscore` fits per-feature mean and
   standard deviation on rows with `dt <= data.train_end` only, then applies a
   3-sigma clip followed by standardization to the whole panel.
   `rank_gauss` instead fits sorted train-period values per feature and maps all
   rows through empirical ranks to Gaussian quantiles. In masked-panel mode the
   fitting source is restricted to rows inside valid PIT membership intervals
   while the fitted transform is still applied to the full panel. The z-score
   statistics, plus `data_file_sha256` / size / mtime for `data.filename` when
   the file exists, are persisted in `run_metadata.json`.

6. **Stock axis and splits** — masked-panel runs take every ticker whose
   membership interval overlaps `[train_start, test_end]`
   (`active_kdcodes_in_period()` in `mci_gru/data/pit.py`) and do not require
   complete coverage. Other runs require completeness: `filter_complete_stocks()`
   across the whole experiment calendar, or `filter_complete_stocks_per_split()`
   followed by an intersection when `data.filter_stocks_per_split=true`.
   `split_by_period()` then cuts train, validation, and test frames.

7. **Windowing and labels** — `mci_gru/data/preprocessing.py` builds:
   - time-series input of shape `(days, stocks, his_t, features)`, where each
     row's window covers the `his_t` dates strictly **before** its sample date;
   - graph node input of shape `(days, stocks, features)` taken from the sample
     date itself;
   - labels of shape `(days, stocks)` using the implemented formula
     `close[t + label_t] / close[t + 1] - 1`.

   The first `his_t` training dates are consumed as lookback, so training labels
   start at `train_dates[his_t]`. Non-masked runs fill missing labels with the
   same-day cross-sectional mean and then zero; masked-panel runs pass
   `fill_missing=False` so unobservable labels stay `NaN` and cannot enter loss
   or evaluation. `training.label_type=rank` converts same-day returns to
   cross-sectional percentiles, which uses only same-day information.

   The formula makes `model.label_t=1` degenerate: the label becomes
   `close[t+1] / close[t+1] - 1`, identically zero for every stock and date.
   Nothing in the config layer rejects that value; `scripts/ci_smoke.py` sets
   `model.label_t=2` for this reason.

8. **Masked-panel eligibility** — `mci_gru/data/pit.py` defines the mask algebra:

   ```text
   tradable = active_member & feature_ready
   loss     = tradable & label_available
   ```

   Pre-membership price history can satisfy the lookback requirement, so a new
   entrant can be feature-ready before it is tradable, but it becomes tradable
   only inside its inclusive membership interval. Rank labels are recomputed over
   the same-day `loss` set. Daily `tradable` breadth is audited against
   `data.pit_min_scoreable_stocks` under `data.pit_breadth_policy`
   (`error`, `warn`, or `off`), and the per-date counts are written into
   `run_metadata.json` as `pit_breadth`.

9. **Graph construction** — the correlation graph and the optional static sector
   graph are built last, from the engineered raw frame (before normalization and
   before PIT row filtering) restricted to the selected stock axis. The graph
   section below describes selection, edge features, and scheduling.

10. **Index-level branch** — `data.experiment_mode=index_level` dispatches to
    `prepare_data_index_level()`, which reads a configured index CSV or a
    one-day-lagged FRED S&P 500 series, assigns the single stock code `INDEX`,
    and reuses the feature and tensor path with a one-node, zero-edge graph.
    This is a separate implementation branch, not stock-level mode with a
    one-name universe.

## Dataset and Batch Contract

`CombinedDataset` keeps time-series inputs, same-day graph inputs, labels,
sample dates, and optional per-date stock masks aligned.

`combined_collate_fn` stacks the dense tensors, concatenates graph nodes across
samples, offsets each sample's edge indices by `i * num_stocks`, and returns a
**9-tuple**:

```text
(
  time_series,
  labels,
  graph_features,
  edge_index,
  edge_weight,
  n_stocks,
  batch_dates,
  edge_index_sector,
  edge_weight_sector,
)
```

The first seven entries match the historical contract. The sector entries are
`None` unless `graph.use_sector_relation=true`. `edge_weight` is `(E,)`,
`(E, 4)`, or wider when lead-lag or snapshot-age columns are enabled; collate
concatenates along dim 0. `Trainer._unpack_loader_batch` still accepts a 7-tuple
for backward compatibility.

Additional collate behavior:

- **Dynamic graphs.** When a `GraphSchedule` is supplied and samples carry dates,
  each sample resolves its own snapshot, so one batch may mix snapshots and any
  batch size works.
- **Stock masks.** When masks are present, `batch_dates` becomes a dict with
  `dates` and `stock_mask`; otherwise it stays a date list or `None`.
  Masked nodes have their time-series and graph features zeroed, and
  `filter_edges_by_stock_mask()` removes correlation and sector edges touching
  them.
- **Snapshot age.** With `graph.append_snapshot_age_days=true`, one column of
  calendar days from the active snapshot's `valid_from` to the sample date is
  appended to 2-D edge attributes at collate time.

`create_data_loaders()` shuffles training data for static graphs and keeps it
sequential for dynamic graphs unless `training.shuffle_train` overrides that.
Train and validation loaders use `training.batch_size`; the test loader uses
batch size 1 with dummy zero labels, because the real test labels are kept
separately for post-prediction evaluation.

## Model Architecture (mci_gru/models/)

The model is defined in `mci_gru/models/trunk.py` and built by
`create_model()` in `mci_gru/models/factory.py`. `mci_gru/models/mci_gru.py` is
a compatibility re-export shim for legacy imports and contains no model logic.

Four parallel streams are concatenated before the final predictor:

```
Input: time_series (B, N, T, F)        graph_features (B*N, F)
         │                                     │
    ┌────┴────┐                                │
    ▼         ▼                                ▼
   A1        (optional A2→A1 cross-attn)   correlation GAT
 Temporal                                  (+ sector GAT + fuse)
 encoder                                         │
    │                                            ▼
    │                                           A2
    ▼         ▼
   B1        B2          B1 = CrossAttn(query=A1, kv=R1)
 latent R1   latent R2   B2 = CrossAttn(query=A2, kv=R2)
    │         │
    └────┬────┘
         ▼
  Z = [A1, A2, B1, B2]
         │
   optional LayerNorm + Dropout
         │
   optional cross-stock SelfAttention
         │
   prediction GATBlock → output activation → score per stock (B, N)
```

### A1: temporal stream

`mci_gru/models/temporal.py` provides three backbones selected by
`model.temporal_encoder`:

- `legacy` — `ImprovedGRU`, built from `AttentionResetGRUCell`, which replaces
  the GRU reset gate with a scaled dot-product attention term;
- `gru_attn` — stacked `nn.GRU` plus a single post-hoc attention readout;
- `transformer` — a causal `nn.TransformerEncoder` stack.

With `model.use_multi_scale=true`, `MultiScaleTemporalEncoder` runs a fast branch
on the raw sequence and a slow branch on a `Conv1d`-downsampled sequence, then
projects the concatenation back to one representation. In transformer mode the
fast branch is the causal Transformer and the slow branch is GRU-with-attention.
The executable base config selects multi-scale `gru_attn`.

### A2: cross-sectional stream

`mci_gru/models/graph.py` implements `GATBlock`: a multi-head concatenating
`GATConv`, an activation, optional inter-layer dropout, then a single-head
`GATConv`. The primary A2 branch consumes the correlation graph over the sample
date's node features.

When `graph.use_sector_relation=true`, a second `GATBlock` consumes the sector
edges with a scalar edge dimension and a `nn.Linear` fuses the two branch
outputs. This is dual GAT plus fusion; there is no `RGATConv` and no true
multi-relation message passing in the code.

With `model.use_a1_a2_cross_attention=true`, A2 additionally queries A1's fast
temporal sequence through `nn.MultiheadAttention` before the latent stage.

### B1/B2: market latent state

`MarketLatentStateLearner` (`mci_gru/models/latent.py`) holds learned state
matrices `R1` and `R2` of shape `(num_hidden_states, D)`. A1 and A2 query them
independently. `model.use_nn_multihead_attention` switches between the legacy
eight-`Linear` implementation and `nn.MultiheadAttention`.

`model.market_latent_mode` decides what those states are:

| Mode | Behaviour |
|------|-----------|
| `static` (default) | `R1` and `R2` are plain parameters, frozen after training. Each stock's output is a function of its own vector alone, so **these streams cannot observe the date's market** despite the name (issue #198). |
| `data_dependent` | The latents first read the date's PIT-active cross-section, then every stock reads those date-conditioned latents (the Set Transformer induced-set construction). Inactive names are excluded as attention keys, not merely zeroed, so the gathered state does not drift with the width of the PIT union axis. Requires `use_nn_multihead_attention=true`; the legacy eight-`Linear` path cannot take per-date keys. |

The two modes hold different parameters, so a checkpoint belongs to the mode
that produced it. `static` remains the default and the frozen recipe is
unchanged.

### Prediction head

`[A1, A2, B1, B2]` are concatenated in that order (the order `SelfAttention`'s
stream-type embedding depends on), optionally LayerNormed and dropped out,
optionally mixed across stocks by `SelfAttention`
(`mci_gru/models/attention.py`), then passed through a final `GATBlock` to one
score per stock. `model.output_activation` selects identity, ELU, ReLU, or
sigmoid.

During training, `graph.drop_edge_p` drops correlation and sector edges through
`torch_geometric.utils.dropout_edge`. When a stock mask is supplied, masked nodes
are zeroed before and after each major stage so they neither contribute features
nor emit nonzero scores.

## Graph (mci_gru/graph/)

Graph responsibilities are split across the package:

| Module | Responsibility |
|---|---|
| `mci_gru/graph/correlation.py` | Return pivot, Pearson correlation, edge selection, edge-feature math |
| `mci_gru/graph/schedule.py` | `GraphSchedule` and O(log n) date lookup |
| `mci_gru/graph/builder.py` | `GraphBuilder` construction facade and snapshot precomputation |
| `mci_gru/graph/sector_edges.py` | Optional static same-sector relation |
| `mci_gru/graph/utils.py` | `edge_feature_dim()` width calculation |

### Correlation graph

The return panel is built from `close / prev_close - 1` (or a per-stock
`pct_change` when `prev_close` is absent), truncated to observations strictly
**before** the graph's valid-from date, limited to the last
`graph.corr_lookback_days` dates (default 252), reindexed to the stock axis, and
zero-filled before `DataFrame.corr()`.

| Mode | Configuration | Implemented selection |
|---|---|---|
| Signed threshold | `top_k == 0` | Off-diagonal directed edges where **`corr > judge_value`** |
| Positive top-K | `top_k > 0`, `top_k_metric=corr` | Each node keeps up to K most-positive valid neighbours |
| Signed top-K | `top_k > 0`, `top_k_metric=abs_corr` | Each node keeps up to K largest by absolute correlation, storing the signed value |

The threshold path compares the **signed** correlation, so the sign of
`judge_value` decides what it can reach. At the shipped `0.8` only strongly
co-moving pairs survive; at a negative threshold anti-correlated pairs survive
too. `GraphConfig` admits the whole of `[-1, 1)` — before issue #162 it required
`0 < judge_value < 1`, so no threshold could admit a negative correlation and
top-K was the only selection path that could. Not only `abs_corr`: `corr` top-K
ranks by *signed* correlation and keeps the K most-positive **available**
neighbours, so a node with fewer than K positive neighbours contributes negative
edges under either metric.

The comparison is **strict**, which is why `-1` is a floor rather than a literal
"every pair": a pair at exactly `-1.0` is dropped at `judge_value = -1`. That is
measure-zero on real return panels and reachable in a fixture, and it is pinned
by `tests/test_threshold_selection_strictness.py`. Top-K edges are directed and
need not be reciprocal.

### Edge attributes

- Scalar mode (`graph.use_multi_feature_edges=false`) returns the signed
  correlation with shape `(E,)`.
- Multi-feature mode (the base-config default) returns `(E, 4)` with columns
  `[corr, |corr|, corr², rank_pct]`. `rank_pct` is zero in threshold mode
  because it is only computed by the top-K path.
- `graph.use_lead_lag_features` appends two columns: the best lag among
  `0` and `graph.lead_lag_days` normalized by the largest candidate lag, and the
  signed correlation at that lag.
- `graph.append_snapshot_age_days` appends one column during collate.

`edge_feature_dim()` derives the final width from `GraphConfig`, and
`run_experiment.py` passes it into `create_model()`, so the model's
`edge_feature_dim` always matches the tensors the loaders emit.

### Static, dynamic, and sector graphs

| Graph | Behavior |
|---|---|
| Static correlation | `graph.update_frequency_months == 0`: built once with `valid_from = data.train_start` and reused for every batch |
| Dynamic correlation | `graph.update_frequency_months > 0`: `GraphBuilder.precompute_snapshots()` builds one snapshot per interval from `train_start` through `test_end`; `combined_collate_fn` resolves each sample by date through `GraphSchedule` |
| Static sector | Each node links to every other node sharing its sector, with scalar weight `1.0`; consumed by the separate sector GAT branch |

Every snapshot uses only observations before its own valid-from date, which is
what keeps the dynamic graph free of lookahead. It also means a meaningful first
or static graph requires price history before `train_start`.

That requirement is enforced for the dynamic path rather than assumed. A
`GraphSchedule` is built with the warm-up sessions the panel actually provides
and the first sample date, and refuses construction when the first snapshot has
fewer than `graph.corr_lookback_days` sessions behind it, or when a sample
precedes the first snapshot. `GraphSchedule.is_ready` reports whether the
contract was verified, and is `False` when the caller supplied no evidence.
The guard matters because `_daily_returns_pivot()` trims to the lookback window
only when enough sessions exist, and otherwise builds from whatever is present —
including nothing — without complaint.

`get_graph_for_date()` raises for any date earlier than the first `valid_from`
instead of clamping to it, since a snapshot built after a sample must never
serve it. Snapshot and lookup dates must be canonical `YYYY-MM-DD`: string
bisection places `2020-1-5` after `2020-07-01`, so a non-canonical date would
otherwise resolve to a future snapshot. Snapshots must also be sorted and
unique, which is the precondition `bisect` assumes. `CombinedDataset`
normalizes `sample_dates` once at the dataset boundary.

Sector buckets come from `graph.sector_map_csv` via `load_sector_map_csv()`,
which accepts either schema: a `gics_sector` column derives the mapping from a
universe metadata export, and a `sector` column reads a curated map. Deriving
from the export keeps coverage in step with the universe instead of depending on
a hand-maintained file. Any `kdcode` with no sector — including blank and
placeholder values such as `UNKNOWN` — is **isolated** with no sector edges at
all, rather than sharing a bucket that would wire every unmapped name to every
other. Coverage is logged, as is any disagreement between snapshots.

## Training and Ensembles

`mci_gru/training/losses.py` centralizes the objectives selected by
`training.loss_type`:

| `loss_type` | Objective |
|---|---|
| `mse` | Mean squared error over finite prediction/target pairs |
| `ic` | Negative mean same-day Pearson information coefficient |
| `combined` | Weighted blend of masked MSE and IC (`training.ic_loss_alpha`) |
| `portfolio_ic` | IC plus a differentiable soft top-K forward-return utility |
| `lambdarank_ic` | Deterministically capped same-day pairwise LambdaRankIC-style surrogate |

`Trainer` (`mci_gru/training/trainer.py`) uses **AdamW**, an optional per-step
linear warmup followed by cosine decay (`training.lr_scheduler`,
`training.warmup_steps`), CUDA autocast and gradient scaling when
`training.use_amp` is set and the device is CUDA, gradient clipping, and
optional first-N-batch step profiling.

Validation returns a `ValidationObservation` carrying loss, Pearson IC, Spearman
rank IC, and the number of eligible rows behind each IC. **Checkpoint selection
fails closed**: IC metrics are `None` rather than `0.0` when no rows are
eligible, and `ValidationObservation.selection_value()` raises `ValueError` when
the configured `training.selection_metric` has fewer than
`training.minimum_selection_rows` eligible rows. Early stopping and checkpointing
both use that single metric; the co-metrics recorded in `TrainingResult` come
from the selected epoch whenever they are available on it.

`mci_gru/training/ensemble.py` implements the ensemble contract.
`train_multiple_models()` builds `training.num_models` independent models; member
`model_id` is seeded with `config.seed + model_id`, writes its own checkpoint at
`checkpoints/model_<id>_best.pth` and its own dated CSVs under
`predictions_model_<id>/`, and the final prediction is the unweighted arithmetic
mean across members. Prediction CSVs have `kdcode,dt,score` rows, round scores to
five decimal places, and omit masked or non-finite names — so in masked PIT mode
a date's CSV contains only that date's tradable candidates.

## Walk-Forward Windows

`mci_gru/walkforward.py` turns one base config into a list of window configs when
`training.walkforward.enabled` is true, and returns `[base]` otherwise. Rolling
mode advances `train_start`; expanding mode holds it fixed and advances
`train_end`. Each window inserts a `label_t + 1` day gap between train and
validation and between validation and test, and any window whose dates fail
`ExperimentConfig._validate_embargo` is silently skipped; if no window survives,
generation raises.

Each generated window is a full `ExperimentConfig` that carries the base
`features`, `graph`, `model`, `training`, **`evaluation`**, and `tracking`
sections with only `data.*` dates rewritten, so per-window evaluation uses the
configured `EvaluationConfig` rather than defaults
(`tests/test_walkforward_config_propagation.py`).

`merge_walkforward_summary()` aggregates per-window training summaries and the
mean of each numeric evaluation metric across windows;
`select_training_objective_value()` returns the aggregate matching
`training.selection_metric`.

## Run Artifacts

A single-window run writes this artifact set into the Hydra output directory:

```text
output/
├── config.yaml                             # the composed Hydra config
├── training_<timestamp>.log
├── mlflow_run.json                         # when tracking is enabled
├── resolved_config.json                    # fully resolved ExperimentConfig for this window
├── run_metadata.json                       # stock list, features, z-score stats, PIT breadth, provenance
├── feature_reference.json                  # train-only normalized feature histograms
├── graph_data.pt                           # train-start correlation graph + optional sector edges
├── checkpoints/model_<id>_best.pth
├── predictions_model_<id>/<date>.csv
├── averaged_predictions/<date>.csv
├── training_summary.json
├── evaluation_summary.json
├── timing_summary.json
└── training_step_profile_model_<id>.jsonl  # when training.profile_batches > 0
```

With walk-forward enabled, the per-window artifacts move under
`walkforward/w###/`, while `config.yaml`, the training log, and `mlflow_run.json`
stay at the run root next to `walkforward_summary.json`.

`write_resolved_config()` (`mci_gru/evaluation/experiment_summary.py`) serializes
the complete resolved `ExperimentConfig` for **every** window and returns both
the file name and a SHA-256 taken from the bytes on disk;
`run_metadata.json` carries them as `resolved_config_path` and
`resolved_config_sha256`. Absolute paths are replaced with the literal marker
`<ABSOLUTE_PATH>` rather than deleted, so a reader can still distinguish a
redacted setting from an unset one. The writer refuses to overwrite an existing
artifact by default; `run_experiment.py` passes `force=True` because its sibling
artifacts all overwrite unconditionally.
`mci_gru/evaluation/run_bundle.py` accepts both `resolved_config.json` and the
legacy `resolved_config.yaml` in its `CONFIG_CANDIDATES`.

MLflow logging (`mci_gru/tracking/mlflow_manager.py`) is optional and mirrors
parameters, metrics, and selected artifacts according to `TrackingConfig`, with a
child run per ensemble member and, in walk-forward mode, a child run per window.

## Evaluation Surfaces

Evaluation has distinct trust boundaries: a metric summary is not an economic
backtest, and neither is paper trading.

### In-run prediction evaluation

`mci_gru/evaluation/metrics.py` computes regression errors, Pearson and Spearman
IC, hit rate, prediction quantiles, top-K returns, naive and Newey-West Sharpe,
and optional moving-block bootstrap intervals
(`mci_gru/evaluation/statistics.py`). The headline `sharpe_ratio` is the
Newey-West value when `label_t > 1` and the naive value otherwise.
`resolved_evaluation_kwargs()` derives defaults from `EvaluationConfig`:
Newey-West lags default to `label_t - 1` and the bootstrap block size defaults to
`max(1, label_t)`. `run_experiment.py` writes the result as
`evaluation_summary.json`. `mci_gru/training/metrics.py` is a compatibility
re-export of the evaluation module.

### Economic saved-prediction replay

`scripts/backtest_sp500.py` is a thin CLI over
`mci_gru/evaluation/backtest_engine.py`. The engine consumes dated prediction
files plus market data, models T-close scoring with T+1-open entry and
open-to-open returns, supports daily, staggered, and block rebalancing, and can
apply costs and a rank-drop gate.

### Selection-research evidence

`scripts/run_saved_prediction_selection_audit.py --research-evidence` measures the
information content of one frozen prediction set without training or capital
simulation. `mci_gru/evaluation/selection_audit.py` validates alignment and
protocol requirements before any metric is computed,
`mci_gru/evaluation/selection_nulls.py` runs a deterministic matched within-date
score null, and `mci_gru/evaluation/artifacts.py` writes one versioned five-file
bundle:

```text
protocol.json
date_evidence.csv
result.json
report.md
manifest.json
```

`validate_trial_family()` in `mci_gru/evaluation/trial_ledger.py` enforces exact,
unique, successful membership against a declared `expected_trial_ids` set, and
the research protocol requires that set whenever a complete trial ledger is
claimed. This surface measures dated stock-selection information only; it does
not model capital, orders, fills, costs, leverage, or paper trading.

### Supporting surfaces

- `mci_gru/evaluation/capacity.py` — saved-prediction capacity diagnostics across
  AUM, top-K, costs, rank-drop, and lagged ADV/volatility thresholds.
- `mci_gru/evaluation/prediction_report.py` — aligned prediction/baseline
  comparison with JSON, Markdown, and CSV output.
- `mci_gru/evaluation/run_bundle.py` — opt-in run provenance manifests and
  core-artifact validation.
- `mci_gru/evaluation/portfolio.py` — ranking, top-K selection, turnover, and the
  shared rank-drop gate.
- `mci_gru/evaluation/drift.py` — PSI / KS-style feature-drift metrics used by
  paper trading.

## Config System (Hydra)

```
configs/
├── config.yaml          ← executable base composition and defaults
├── data/                ← DataConfig overrides (sp500, russell1000, temporal_2019, ...)
├── features/            ← FeatureConfig overrides (base, with_momentum, full, ...)
└── experiment/          ← multi-section experiment presets (paper_faithful, hybrid, ...)
```

`create_config_from_dict()` in `mci_gru/config.py` is the single plain-dict
ingestion path. `ExperimentConfig` owns `DataConfig`, `FeatureConfig`,
`GraphConfig`, `ModelConfig`, `TrainingConfig` (which nests
`WalkforwardConfig`), `EvaluationConfig`, and `TrackingConfig`, plus
`experiment_name`, `output_dir`, and `seed`.

Construction validates chronological dates and requires calendar gaps
**strictly greater than `model.label_t`** between train/validation and
validation/test, unless `data.skip_embargo_check=true`, which downgrades the
failure to a warning and is discouraged. `pipeline._stock_feature_row_slice()`
maps label dates back to the correct time-series rows so embargo gaps do not
desynchronize the temporal tensors from the graph features.

Base-config defaults worth knowing (`configs/config.yaml`): `his_t=10`,
`label_t=5`, multi-scale `gru_attn` temporal encoder, static graph with
`top_k=0`, `use_multi_feature_edges=true`, `drop_edge_p=0.1`, `num_models=10`,
`loss_type=combined`, `selection_metric=val_ic`, and `minimum_selection_rows=1`.
The frozen production recipe in `docs/DEFAULT_EXPERIMENT_RECIPE.md` overrides
several of these (20 members, pure IC loss, patience 15); presets are
configuration, not architectural constants.

`create_model()` keeps legacy defaults so old partial checkpoint configs still
load. New runs receive explicit values from typed config plus the graph-derived
`edge_feature_dim`, `drop_edge_p`, and `use_sector_relation` injected by
`run_experiment.py`.

Override from the CLI:
`python run_experiment.py model.his_t=20 training.loss_type=ic`

## Paper Trading (paper_trade/)

Paper trading is frozen-checkpoint inference, not a continuation of training.

`paper_trade/scripts/infer.py` loads the frozen `config.yaml`,
`run_metadata.json`, every `model_*_best.pth` under the model directory, and
`graph_data.pt`. It reconstructs the feature engineer, optionally fetches regime
inputs through the inference date, reuses the shared imputation, z-score, and
single-date tensor helpers from `mci_gru/data/transforms.py`, runs every
checkpoint, and averages their scores. It does **not** instantiate or call
`GraphBuilder`.

`paper_trade/scripts/run_nightly.py` runs the steps in this order, because
execution tracking needs the new day's open before a new target portfolio is
formed:

1. `refresh_data.py` — append incremental LSEG bars to the master CSV.
2. `track.py` — simulate prior orders at the next open and update open-to-open
   position returns, costs, trades, and persistent fill state. Its benchmark is
   `SPY.P`.
3. `infer.py` — write dated scores plus the normalized feature matrix used for
   monitoring.
4. `portfolio.py` — rank scores, apply the rank-drop policy, and write target
   holdings, orders, and persisted rank/holding state.
5. `monitor.py` — compare normalized inference features against the frozen
   train-window `feature_reference.json` and write `feature_drift.json` /
   `feature_drift.csv`.
6. `report.py` — write Markdown and JSON reports plus equity and drawdown charts.

The default policy exits a scored holding when its rank worsens by at least 30
places (`DEFAULT_MIN_RANK_DROP` in `paper_trade/scripts/portfolio.py`, applied
through the shared gate in `mci_gru/evaluation/portfolio.py`).

`paper_trade/scripts/catchup.py` replays missed dates sequentially and
`paper_trade/scripts/compare_regime.py` compares frozen regime and no-regime
model outputs. Neither path trains a model.

## Package Layout

```
run_experiment.py                 ← training/prediction composition root
mci_gru/
├── config.py                     ← typed configuration dataclasses and validation
├── pipeline.py                   ← staged stock/index data preparation
├── walkforward.py                ← rolling / expanding window generation
├── regime_contract.py            ← regime input column contract
├── data/
│   ├── data_manager.py           ← DataManager, CombinedDataset, combined_collate_fn, create_data_loaders
│   ├── preprocessing.py          ← windows, graph-node features, labels, rank transforms
│   ├── transforms.py             ← shared imputation, z-score, single-date tensors
│   ├── pit.py                    ← membership / readiness / loss / tradable masks
│   ├── pit_audit.py              ← PIT audit helpers
│   ├── lseg_loader.py            ← LSEG/Refinitiv API access
│   ├── fred_loader.py            ← FRED auxiliary and index series
│   ├── reshape.py                ← vendor-frame reshape helpers
│   ├── path_resolver.py          ← project-aware data paths
│   └── universes.py              ← named universe definitions
├── features/
│   ├── registry.py               ← FeatureEngineer composition
│   ├── base.py                   ← base OHLCV features (turnover), price and volume features
│   ├── momentum.py               ← MTP momentum (binary/continuous/buffered, static/dynamic blend)
│   ├── volatility.py             ← realized vol, VIX, RSI, MA, volatility targeting
│   ├── credit.py                 ← credit spread features from FRED
│   └── regime.py                 ← global regime similarity features
├── graph/
│   ├── builder.py                ← GraphBuilder construction facade
│   ├── correlation.py            ← correlation and edge-selection math
│   ├── schedule.py               ← GraphSchedule
│   ├── sector_edges.py           ← static sector relation
│   └── utils.py                  ← edge_feature_dim()
├── models/
│   ├── factory.py                ← create_model()
│   ├── trunk.py                  ← StockPredictionModel
│   ├── temporal.py               ← ImprovedGRU, GRUWithAttention, CausalTransformerEncoder, MultiScaleTemporalEncoder
│   ├── graph.py                  ← GATBlock
│   ├── latent.py                 ← MarketLatentStateLearner
│   ├── attention.py              ← cross-stock SelfAttention
│   └── mci_gru.py                ← compatibility re-export shim
├── training/
│   ├── trainer.py                ← Trainer, ValidationObservation, early stopping, inference
│   ├── ensemble.py               ← train_multiple_models(), prediction averaging
│   ├── losses.py                 ← loss implementations and factory
│   └── metrics.py                ← compatibility shim to evaluation.metrics
├── evaluation/
│   ├── metrics.py                ← prediction metrics
│   ├── statistics.py             ← IC, Newey-West, bootstrap primitives
│   ├── experiment_summary.py     ← in-run evaluation policy, resolved-config provenance
│   ├── backtest_engine.py        ← economic saved-prediction replay
│   ├── portfolio.py              ← ranking, top-K, turnover, rank-drop gate
│   ├── capacity.py               ← capacity diagnostics
│   ├── selection_audit.py        ← selection-evidence protocol and decisions
│   ├── selection_nulls.py        ← deterministic matched null
│   ├── artifacts.py              ← canonical bundles and JSON artifact writer
│   ├── prediction_report.py      ← comparative prediction reports
│   ├── run_bundle.py             ← provenance manifests
│   ├── trial_ledger.py           ← cross-run trial ledger
│   └── drift.py                  ← feature drift
├── tracking/
│   └── mlflow_manager.py         ← optional MLflow integration
└── utils/
    └── seeding.py                ← set_seed()
paper_trade/
├── Model/                        ← frozen configs, metadata, and checkpoints
├── scripts/                      ← refresh, track, infer, portfolio, monitor, report, nightly
└── state/                        ← persistent holdings, ranks, fills, run manifest
                                     (dated scores, orders, monitoring, and reports are
                                      written at runtime under paper_trade/results/,
                                      which is not tracked in git)
scripts/                          ← supported CLIs for backtest, research, and reporting
tests/                            ← pytest suite (run via scripts/run_pytest_isolated.py)
```

## Current Implementation Boundaries

These are properties of the code as written, not intended guarantees:

- Correlation construction needs observations before `train_start`. CSV inputs
  can supply that buffer, but `DataManager._load_from_lseg()` requests only
  `train_start` through `test_end`, so an LSEG-sourced first or static graph has
  no pre-train observations.
- `graph_data.pt` stores only the `train_start` correlation graph and the
  optional sector edges. A dynamic `GraphSchedule` lives in memory and is not
  exported, so frozen paper-trade inference uses the static saved graph even
  after dynamic-graph training.
- `run_metadata.json` serializes z-score means and standard deviations but not a
  fitted rank-Gaussian reference, so the frozen paper-trade path is not
  self-contained for a `rank_gauss` training run.
- `EvaluationConfig.sharpe_method` is validated but is not passed by
  `resolved_evaluation_kwargs()`; headline Sharpe selection follows `label_t` as
  described above.
- Index-level mode always uses z-score normalization regardless of
  `data.normalisation`, and builds labels from the normalized frame, whereas
  stock-level labels come from the raw engineered price frame.
- `StockPredictionModel.forward()` fails open: a model built with
  `graph.use_sector_relation=true` whose sector tensors arrive as `None` skips
  sector fusion silently rather than raising. This is deliberate and documented
  in `trunk.py` — `prepare_data()` always builds the tensors, and the only path
  that hard-codes them to `None`, `prepare_data_index_level()`, is now rejected
  at config validation, so no reachable configuration reaches the fail-open.
- The correlation graph is point-in-time blind: snapshots are built over the
  full panel axis, so they connect names that had not yet entered the universe
  or had already left it. Tracked in issue #123.
