# MCI-GRU Agent Guide

Status: current-state routing layer for nontrivial work.

`AGENTS.md` is the short entrypoint: invariants, environment, and the commands
you run most. This guide is the next layer down. It exists to answer four
questions quickly, without reading the whole codebase:

1. which module owns the behaviour I am about to change;
2. which contracts sit next to that module;
3. which focused tests cover it;
4. which engineering constraints apply.

This guide is a routing layer, not a second architecture specification. It does
not describe how the model works — [../ARCHITECTURE.md](../ARCHITECTURE.md) does
that, and it is the canonical description of implemented architecture. Where
this guide and current code disagree, the code wins; see [domain.md](domain.md)
for the full source-of-truth order.

Future-state design is deliberately out of scope here. It belongs in
[target-architecture.md](target-architecture.md), which is human-led and is not
current-state authority.

## How To Use This Guide

1. Read `AGENTS.md` for the invariants and required reads.
2. Use the tables below to identify the owning module, adjacent contracts, and
   focused tests for your task.
3. Read that code before relying on any prose, including this guide's.
4. Name the contracts in scope before editing.
5. Run the smallest focused proof first, then broaden verification in
   proportion to the shared contracts you touched.
6. Report exact commands, exit status, skips, and residual risk.

Nothing in this guide is evidence that the implementation behaves a certain way.
Neither is a handoff, a historical plan under `docs/agent_references/`, or a
generated report.

## 1. Repository And Architectural Awareness

### Source map

| Source | Role | Do not use it for |
| --- | --- | --- |
| `AGENTS.md` | Short agent entrypoint: invariants, environment, commands | Detailed implementation behaviour |
| [guide.md](guide.md) (this file) | Current-state routing: owning modules, contracts, tests, constraints | Architecture specification or future-state decisions |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Canonical implemented architecture | Aspirational design or historical intent |
| `configs/config.yaml` and its config groups | Executable base values and Hydra composition | Constants that a preset may override |
| `mci_gru/config.py` | Typed configuration contract and validation | The values selected for a particular run |
| [../CONFIGURATION_GUIDE.md](../CONFIGURATION_GUIDE.md) | Hydra composition and override patterns | Proof that a documented value is still the default |
| [../DEFAULT_EXPERIMENT_RECIPE.md](../DEFAULT_EXPERIMENT_RECIPE.md) | Frozen human-approved confirmation recipe | Base defaults, or smoke-run budgets |
| [../TESTING_GUIDE.md](../TESTING_GUIDE.md) | Verification policy, ladder, and evidence taxonomy | Proof that an unexecuted test currently passes |
| `../../CONTEXT.md` | Repo-wide vocabulary | Runtime behaviour |
| [../research/README.md](../research/README.md) | Router for current versus superseded research evidence | Implementation truth |
| [../handoffs/](../handoffs/) | Operational continuity notes | Research evidence or canonical behaviour |
| [../agent_references/README.md](../agent_references/README.md) | Historical Claude/Cursor guidance and plans | Current requirements without code verification |
| [../index.md](../index.md) | Full documentation map | A substitute for reading the routed document |

### Composition roots and system boundaries

| Surface | Composition root | Owns | Boundary to preserve |
| --- | --- | --- | --- |
| Training and prediction | `run_experiment.py` | Hydra composition, data preparation, model construction, ensemble training, summaries, tracking, walk-forward windows | Does not define paper-trade execution policy |
| Panel preparation | `mci_gru/pipeline.py` | Loading, feature composition, PIT resolution, train-only normalisation, universe selection, tensors, graph artifacts | Timing and mask contracts stay explicit at each stage |
| Configuration | `configs/` plus `mci_gru/config.py` | YAML selects values; dataclasses validate them | Do not hide experiment behaviour in hard-coded branches |
| Model | `mci_gru/models/factory.py` and `mci_gru/models/trunk.py` | Model assembly and the prediction trunk | `mci_gru/models/mci_gru.py` is a compatibility re-export shim only |
| Training | `mci_gru/training/` | Objectives, optimisation, validation, checkpoint selection, ensemble averaging | Evaluation semantics live in `mci_gru/evaluation/` |
| Evaluation | `mci_gru/evaluation/` plus `scripts/` CLIs | Prediction metrics, economic replay, selection evidence, capacity, provenance | Research evidence is not automatically economic or production evidence |
| Paper trading | `paper_trade/scripts/run_nightly.py` | Frozen-checkpoint inference, portfolio decisions, tracking, monitoring, reporting | Loads a frozen `graph_data.pt`; never imports `GraphBuilder` |

Economic saved-prediction replay and paper trading are downstream systems.
Neither is part of the training loop.

### Current entry path

`run_experiment.py` imports exactly these composition seams, which is the
fastest way to confirm the live path:

```text
Hydra YAML
  -> create_config_from_dict()            mci_gru/config.py
  -> prepare_data() / prepare_data_index_level()   mci_gru/pipeline.py
  -> create_data_loaders()                mci_gru/data/data_manager.py
  -> edge_feature_dim()                   mci_gru/graph/utils.py
  -> create_model()                       mci_gru/models/factory.py
  -> train_multiple_models()              mci_gru/training/ensemble.py
  -> compute_evaluation_summary(), write_resolved_config()
                                          mci_gru/evaluation/experiment_summary.py
  -> generate_walkforward_configs(), merge_walkforward_summary()
                                          mci_gru/walkforward.py
  -> MLflowTrackingManager                mci_gru/tracking/mlflow_manager.py
```

`edge_feature_dim` is a module-level function in `mci_gru/graph/utils.py`, not a
method on `GraphConfig`. `run_experiment.py` calls it and passes the result to
`create_model` under the `model` key `edge_feature_dim`; `paper_trade/scripts/infer.py`
calls the same helper so frozen inference and training agree on edge width.

### Pipeline stages

`mci_gru/pipeline.py` is staged rather than monolithic. Prefer extending an
existing stage over widening `run_experiment.py`:

| Stage function | Produces |
| --- | --- |
| `load_raw_data` | `DataManager` plus the raw panel |
| `load_auxiliary_data` | credit / regime / index auxiliary frames |
| `engineer_features` | `PipelineFrames` |
| `resolve_pit_context` | `PitContext` |
| `fit_normalisation` | `NormFit` (train-period statistics only) |
| `select_universe` | stock axis and split dates |
| `build_tensors` | `TensorBundle` |
| `apply_pit_masks_to_tensors` | masked-panel tensors and daily masks |
| `build_correlation_graph` | `GraphArtifacts` |
| `prepare_data` / `prepare_data_index_level` | the assembled stock-level or index-level result |

### Module ownership map

| Area | Primary implementation | Adjacent contracts |
| --- | --- | --- |
| Typed config | `mci_gru/config.py` | `configs/config.yaml`, `configs/data/`, `configs/features/`, `configs/experiment/` |
| Pipeline stages | `mci_gru/pipeline.py` | data, features, graph, PIT, run-artifact export |
| Data loading | `mci_gru/data/data_manager.py`, `lseg_loader.py`, `fred_loader.py`, `path_resolver.py` | standard panel columns, auxiliary sources, `FRED_API_KEY` |
| Transforms and tensors | `mci_gru/data/transforms.py`, `preprocessing.py`, `reshape.py` | train-only fits, window construction, single-date inference |
| Universes | `mci_gru/data/universes.py` | universe CSVs and constituent exports under `scripts/data/` |
| PIT semantics | `mci_gru/data/pit.py`, `pit_audit.py` | `PITMaskSet`, `PITKnowledgeClass`, fixed union axis, daily masks |
| Feature composition | `mci_gru/features/registry.py` | `FeatureConfig`, `configs/features/`, owning feature module |
| Feature calculations | `mci_gru/features/momentum.py`, `volatility.py`, `credit.py`, `regime.py`, `base.py` | [../REGIME_DATA_CONTRACT.md](../REGIME_DATA_CONTRACT.md), `mci_gru/regime_contract.py` |
| Correlation edges | `mci_gru/graph/correlation.py` | `GraphConfig.judge_value`, `top_k`, `top_k_metric`, multi-feature and lead-lag flags |
| Graph construction and snapshots | `mci_gru/graph/builder.py` | `GraphBuilder.build_graph`, `precompute_snapshots`, `should_update` |
| Dynamic graph timing | `mci_gru/graph/schedule.py` | `GraphSchedule.get_graph_for_date`, `snapshot_valid_from_for_date` |
| Sector relation | `mci_gru/graph/sector_edges.py` | `graph.use_sector_relation`, `sector_map_csv`, extra collate slots |
| Edge width | `mci_gru/graph/utils.py` | must agree across collate, `create_model`, and frozen inference |
| Loader batching | `mci_gru/data/data_manager.py` | `CombinedDataset`, `combined_collate_fn`, the 9-tuple invariant |
| Model assembly | `mci_gru/models/factory.py`, `trunk.py` | `mci_gru/models/temporal.py`, `graph.py`, `latent.py`, `attention.py` |
| Objectives | `mci_gru/training/losses.py` | `build_training_loss`, `TrainingConfig.loss_type`, masked cross sections |
| Training lifecycle | `mci_gru/training/trainer.py` | `Trainer`, `TrainingResult`, `ValidationObservation`, selection metric |
| Ensembling | `mci_gru/training/ensemble.py` | `train_multiple_models`, per-member seeds, mean prediction |
| In-run metrics | `mci_gru/training/metrics.py`, `mci_gru/evaluation/metrics.py`, `statistics.py` | `EvaluationConfig`, bootstrap and Sharpe policy |
| Run summaries and provenance | `mci_gru/evaluation/experiment_summary.py` | `run_metadata.json`, `resolved_config.json` and its SHA-256 |
| Economic replay | `mci_gru/evaluation/backtest_engine.py`, `portfolio.py`, `scripts/backtest_sp500.py` | score / execution / return timing, costs, benchmark |
| Selection research | `mci_gru/evaluation/selection_audit.py`, `selection_nulls.py`, `trial_ledger.py`, `artifacts.py` | [../evaluation/EVIDENCE_HARNESS.md](../evaluation/EVIDENCE_HARNESS.md), `SelectionResearchProtocol` |
| Run bundles | `mci_gru/evaluation/run_bundle.py` | manifest hashes, `CONFIG_CANDIDATES`, immutability |
| Capacity and drift | `mci_gru/evaluation/capacity.py`, `drift.py` | replay inputs, feature-drift reporting |
| Walk-forward | `mci_gru/walkforward.py` | `WalkforwardConfig`, per-window `ExperimentConfig` fidelity |
| Paper-trade inference | `paper_trade/scripts/infer.py` | frozen `config.yaml`, `run_metadata.json`, `graph_data.pt`, checkpoints |
| Paper-trade decisions | `paper_trade/scripts/portfolio.py`, `track.py`, `monitor.py`, `report.py` | ranks, holdings, simulated fills, state under `paper_trade/state/` |
| Nightly orchestration | `paper_trade/scripts/run_nightly.py` | `STEPS` order: `refresh_data.py` → `track.py` → `infer.py` → `portfolio.py` → `monitor.py` → `report.py` |

### Contracts worth locating before you change anything

- `ExperimentConfig` is the typed root config; `DataConfig`, `FeatureConfig`,
  `GraphConfig`, `ModelConfig`, `TrainingConfig`, `WalkforwardConfig`,
  `EvaluationConfig`, and `TrackingConfig` are its sections.
- `PipelineFrames`, `PitContext`, `NormFit`, `TensorBundle`, and
  `GraphArtifacts` are the pipeline seams.
- `CombinedDataset` and `combined_collate_fn` own the dataset-to-trainer
  boundary, including the 9-tuple contract in `AGENTS.md`.
- `GraphSchedule` owns date-indexed dynamic graph selection.
- `StockPredictionModel` is the model trunk.
- `TrainingResult` and `ValidationObservation` carry checkpoint-selection
  outcomes; IC metrics are `None`, not `0.0`, when no rows are eligible.
- `averaged_predictions/` is the ensemble prediction surface consumed by
  backtests, selection research, and paper trading.
- `run_metadata.json`, `resolved_config.json`, `feature_reference.json`,
  `graph_data.pt`, and member checkpoints form the frozen inference inputs.

### Base defaults versus the frozen recipe

`configs/config.yaml` composes `data: sp500` and `features: with_momentum`. Its
base values are not the frozen confirmation recipe, and confusing the two is a
recurring documentation error. Verified base values include:

| Key | Base value |
| --- | --- |
| `model.his_t` / `model.label_t` | `10` / `5` |
| `graph.judge_value` | `0.8` (used only when `top_k == 0`) |
| `graph.update_frequency_months` | `0` (static graph) |
| `graph.corr_lookback_days` | `252` |
| `graph.top_k` / `graph.top_k_metric` | `0` / `corr` |
| `graph.use_multi_feature_edges` | `true` |
| `graph.drop_edge_p` | `0.1` |
| `graph.use_lead_lag_features`, `append_snapshot_age_days`, `use_sector_relation` | `false` |
| `training.num_models` | `10` |
| `training.num_epochs` | `100` |
| `training.early_stopping_patience` | `10` |
| `training.loss_type` | `combined` |
| `training.label_type` | `returns` |
| `training.selection_metric` | `val_ic` |
| `training.minimum_selection_rows` | `1` |
| `training.walkforward.enabled` | `false` |
| `evaluation.sharpe_method` | `newey_west` |
| `data.pit_universe_mode` (via `configs/data/sp500.yaml`) | `row_filter` |
| `seed` / `output_dir` | `1729` / `results` |

The frozen recipe in [../DEFAULT_EXPERIMENT_RECIPE.md](../DEFAULT_EXPERIMENT_RECIPE.md)
differs on purpose — it specifies `training.num_models=20`,
`training.early_stopping_patience=15`, and `training.loss_type=ic`. Read the
recipe document rather than restating its values from memory, and do not
describe recipe values as base defaults.

`pit_universe_mode=masked_panel` is selected by the `configs/experiment/pit_temporal_*.yaml`
presets, not by the base config. When it is selected, the masked-panel breadth
invariant in `AGENTS.md` applies.

## 2. Semantic Navigation

Start from the task concept, not from a guessed filename.

### Task to read set

| Task | Read | Adjacent contract | Focused tests |
| --- | --- | --- | --- |
| Add or change a feature | `mci_gru/features/registry.py` plus the owning feature module | `FeatureConfig`, `configs/features/` | `tests/test_feature_config_yaml.py`, `tests/test_momentum_blend_modes.py`, `tests/test_volatility_features.py`, `tests/test_volatility_targeting_features.py`, `tests/test_regime_features.py`, `tests/test_feature_drift.py` |
| Regime inputs | `mci_gru/features/regime.py`, `mci_gru/regime_contract.py` | [../REGIME_DATA_CONTRACT.md](../REGIME_DATA_CONTRACT.md) | `tests/test_regime_features.py` |
| Normalisation or windowing | `mci_gru/data/transforms.py`, `preprocessing.py`, `mci_gru/pipeline.py` | train-only fit boundary | `tests/test_transforms.py`, `tests/test_preprocessing_vectorised.py`, `tests/test_data_loading_helpers.py` |
| PIT breadth or masks | `mci_gru/data/pit.py`, `pit_audit.py`, `mci_gru/pipeline.py` | masked-panel invariant, `PITKnowledgeClass` | `tests/test_pit_masked_panel.py`, `tests/test_pit_availability_report.py` |
| Correlation edge selection | `mci_gru/graph/correlation.py`, `mci_gru/graph/builder.py` | `GraphConfig` threshold / top-K semantics | `tests/test_dynamic_graph_updates.py` |
| Dynamic graph timing | `mci_gru/graph/schedule.py`, `builder.py`, `combined_collate_fn` | strict valid-from timing | `tests/test_dynamic_graph_updates.py`, `tests/test_phase3_graph_and_walkforward.py` |
| Sector relation | `mci_gru/graph/sector_edges.py`, `data_manager.py`, `models/trunk.py` | 9-tuple sector slots | `tests/test_phase3_graph_and_walkforward.py` |
| Edge feature width | `mci_gru/graph/utils.py`, `models/graph.py`, `paper_trade/scripts/infer.py` | collate width must equal model `edge_feature_dim` | `tests/test_inference_edge_dim.py`, `tests/test_dynamic_graph_updates.py` |
| Model trunk or encoder | `mci_gru/models/factory.py`, `trunk.py`, `temporal.py`, `graph.py` | `ModelConfig` | `tests/test_mci_gru_phase2.py`, `tests/test_phase3_graph_and_walkforward.py` |
| Loss or selection metric | `mci_gru/training/losses.py`, `trainer.py` | `TrainingConfig`, fail-closed selection | `tests/test_lambdarank_ic_loss.py`, `tests/test_lambdarank_ic_trainer.py`, `tests/test_portfolio_ic_loss.py`, `tests/test_portfolio_ic_trainer.py`, `tests/test_lambdarank_ic_config.py`, `tests/test_portfolio_ic_config.py` |
| Training efficiency knobs | `mci_gru/training/trainer.py`, `mci_gru/config.py` | dataloader and AMP settings | `tests/test_training_efficiency_config.py` |
| Ensemble behaviour | `mci_gru/training/ensemble.py` | ensemble invariant | `tests/test_ensemble_averaging.py` |
| Walk-forward windows | `mci_gru/walkforward.py`, `run_experiment.py` | per-window config fidelity | `tests/test_walkforward_config_propagation.py`, `tests/test_phase3_graph_and_walkforward.py` |
| Run summary or provenance | `mci_gru/evaluation/experiment_summary.py` | `run_metadata.json`, `resolved_config.json` | `tests/test_experiment_summary.py`, `tests/test_run_bundle_manifest.py` |
| Evaluation statistics | `mci_gru/evaluation/statistics.py`, `metrics.py`, `portfolio.py` | `EvaluationConfig` | `tests/test_evaluation_statistics.py`, `tests/test_evaluation_portfolio.py`, `tests/test_prediction_report.py` |
| Economic backtest | `mci_gru/evaluation/backtest_engine.py`, `scripts/backtest_sp500.py` | timing, costs, benchmark; [../BACKTEST_FAIRNESS_AUDIT.md](../BACKTEST_FAIRNESS_AUDIT.md) as history | `tests/test_backtest_engine_golden.py`, `tests/test_backtest_fairness.py`, `tests/test_backtest_plotting.py`, `tests/test_pit_saved_prediction_backtests.py` |
| Selection research evidence | `mci_gru/evaluation/selection_audit.py`, `selection_nulls.py`, `trial_ledger.py`, `artifacts.py` | [../evaluation/EVIDENCE_HARNESS.md](../evaluation/EVIDENCE_HARNESS.md) | `tests/test_selection_research_claims.py`, `tests/test_selection_research_statistics.py`, `tests/test_selection_research_artifacts.py`, `tests/test_selection_research_integration.py`, `tests/test_selection_research_pit.py`, `tests/test_saved_prediction_selection_audit.py`, `tests/test_trial_ledger.py` |
| Capacity replay | `mci_gru/evaluation/capacity.py`, `scripts/run_saved_prediction_capacity_replay.py` | replay inputs and provenance | `tests/test_capacity_replay.py` |
| Output layout | `run_experiment.py`, `mci_gru/evaluation/` writers | [../OUTPUT_MANAGEMENT.md](../OUTPUT_MANAGEMENT.md) | `tests/test_output_management.py` |
| MLflow tracking | `mci_gru/tracking/mlflow_manager.py` | [../MLFLOW_TRACKING.md](../MLFLOW_TRACKING.md), `TrackingConfig` | `tests/test_mlflow_tracking.py` |
| Paper-trade inference | `paper_trade/scripts/infer.py` | frozen artifacts; `GraphBuilder` ban | `tests/test_paper_trade_infer.py` |
| Paper-trade monitoring | `paper_trade/scripts/monitor.py`, `report.py` | state files under `paper_trade/state/` | `tests/test_paper_trade_monitor.py` |
| Notebook generators | the owning `scripts/gen_*.py` | [../NOTEBOOK_BEST_PRACTICES.md](../NOTEBOOK_BEST_PRACTICES.md), [../workflows/COLAB_CHROME_CONTROL_GUIDE.md](../workflows/COLAB_CHROME_CONTROL_GUIDE.md) | the matching `tests/test_*_notebook*.py` contract test |
| CI smoke path | `scripts/ci_smoke.py` | smoke overrides prove wiring only | `tests/test_ci_smoke.py` |
| Docs placement | `scripts/check_docs_sot.py` | dated reports belong under `docs/research/` | `tests/test_check_docs_sot.py` |
| Retired surfaces | `tests/test_repository_retirement_guard.py` | retired paths must stay absent | `tests/test_repository_retirement_guard.py` |

### High-value symbol searches

```powershell
rg -n "class ExperimentConfig|def create_config_from_dict" mci_gru/config.py
rg -n "def prepare_data|def prepare_data_index_level" mci_gru/pipeline.py
rg -n "class CombinedDataset|def combined_collate_fn|def create_data_loaders" mci_gru/data/data_manager.py
rg -n "class GraphSchedule|def precompute_snapshots|def get_graph_for_date" mci_gru/graph
rg -n "def edge_feature_dim" mci_gru/graph/utils.py
rg -n "class StockPredictionModel|def create_model" mci_gru/models
rg -n "def build_training_loss|class Trainer|def train_multiple_models" mci_gru/training
rg -n "averaged_predictions|graph_data.pt|run_metadata.json" run_experiment.py mci_gru paper_trade
rg -n "GraphBuilder" paper_trade
```

### Change propagation

Feature change:

```text
FeatureConfig -> configs/features or experiment preset
  -> build_feature_list / FeatureEngineer
  -> owning feature module
  -> pipeline feature columns
  -> run metadata and feature_reference.json
  -> calculation + wiring + no-lookahead tests
```

Graph change:

```text
GraphConfig -> mci_gru/graph/utils.py edge_feature_dim()
  -> correlation.py / builder.py / schedule.py / sector_edges.py
  -> combined_collate_fn
  -> create_model()
  -> GAT blocks in models/graph.py and models/trunk.py
  -> paper_trade/scripts/infer.py
  -> dynamic-timing + edge-width + 9-tuple + inference tests
```

Training objective change:

```text
TrainingConfig -> build_training_loss()
  -> Trainer train/validation accounting
  -> ValidationObservation and selection metric
  -> checkpoint selection (fails closed on empty coverage)
  -> member predictions -> arithmetic-mean ensemble
  -> evaluation summary and run artifacts
```

Frozen inference change:

```text
training artifacts
  -> config.yaml + run_metadata.json + feature_reference.json
  -> graph_data.pt + member checkpoints
  -> paper_trade/scripts/infer.py
  -> portfolio.py -> monitor.py -> report.py
```

### Stop conditions

Resolve the boundary before editing when:

- a change crosses training, economic replay, and paper trading without
  distinguishing their semantics;
- a feature or graph proposal has no statement of when each input is knowable;
- a PIT change would reduce the fixed masked-panel union to complete stocks,
  continuous members, or stayers only;
- a proposed value conflicts with the frozen recipe but is being described as a
  base default;
- a historical plan or handoff is the only source for an alleged current
  behaviour;
- the task depends on gitignored outputs with no provenance record;
- a live Colab or other external operation is required but not authorised.

## 3. Engineering Constraints

### Source-of-truth order

1. current code and tests for implemented behaviour;
2. repository invariants in `AGENTS.md`;
3. canonical docs and focused data contracts;
4. current research evidence routed by [../research/README.md](../research/README.md);
5. handoffs and historical references.

See [domain.md](domain.md) for the full policy, including how to report
source-of-truth drift instead of silently rewriting prose.

### Runtime invariants

`AGENTS.md` holds the authoritative list. Changing one is an architecture
decision, not a local refactor, and it cannot be done by editing prose. The
guard surfaces are:

| Invariant | Guard surfaces |
| --- | --- |
| No lookahead in normalisation, features, graph observations, and labels | pipeline, transform, feature, and graph tests, including `test_precompute_snapshots_no_lookahead` in `tests/test_dynamic_graph_updates.py` and `test_compute_regime_monthly_features_no_lookahead_exclusion_effect` in `tests/test_regime_features.py` |
| Dynamic graph resolves by sample date through `GraphSchedule` | `tests/test_dynamic_graph_updates.py`, `tests/test_phase3_graph_and_walkforward.py` |
| `combined_collate_fn` returns the 9-tuple and concatenates edge tensors on the edge dimension | `mci_gru/data/data_manager.py`, trainer unpacking, `tests/test_phase3_graph_and_walkforward.py` |
| Ensemble prediction is the unweighted mean of independently seeded members | `tests/test_ensemble_averaging.py` |
| Paper-trade inference loads a frozen `graph_data.pt` and never imports `GraphBuilder` | `tests/test_paper_trade_infer.py`, `rg -n "GraphBuilder" paper_trade` |
| `masked_panel` keeps the fixed PIT union axis and daily masks | `tests/test_pit_masked_panel.py` |
| Retired repository surfaces stay absent | `tests/test_repository_retirement_guard.py` |

### Timing and finance constraints

- Every new feature must state when each input becomes knowable.
- Rolling, expanding, EWM, regime, graph, and forward-return logic needs a
  future-row mutation canary or an equivalent timing proof. See the
  No-Lookahead Canary section of [../TESTING_GUIDE.md](../TESTING_GUIDE.md).
- Name label construction, score date, execution date, holding period, and
  return attribution separately.
- Mechanics smokes prove wiring only. They do not support performance,
  profitability, or production-readiness claims.
- Saved-prediction selection evidence, economic backtests, and paper trading
  are separate evidence surfaces.
- Rank labels, raw return labels, prediction scores, and realised portfolio
  returns are not interchangeable scales.

### Configuration constraints

- Put runtime knobs in the owning dataclass in `mci_gru/config.py` and add
  validation for invalid values and combinations.
- Put selected values in Hydra YAML under `configs/`.
- Keep `pyproject.toml` as the dependency range source of truth; keep
  `requirements.txt` as Colab-facing ranges rather than pinning it to
  `requirements.lock`.
- Do not change [../DEFAULT_EXPERIMENT_RECIPE.md](../DEFAULT_EXPERIMENT_RECIPE.md)
  silently. A recipe change needs explicit human review and supporting evidence.

### Code constraints

- Python 3.10 or later; type hints on public functions.
- `ruff check .` and `ruff format .`, configured in `pyproject.toml`.
- Imports stay at module scope.
- Prefer the existing typed pipeline seams over widening `run_experiment.py`.
- Keep feature composition in `FeatureEngineer` and calculations in the owning
  feature module.
- Keep model construction in `mci_gru/models/factory.py`; do not add
  implementation to the `mci_gru/models/mci_gru.py` compatibility shim.
- Keep loss construction in `mci_gru/training/losses.py` and evaluation
  semantics in `mci_gru/evaluation/`.
- Preserve public and serialised contracts unless a migration is designed and
  tested.

### Test constraints

- Use the repo virtual environment, and on Windows run pytest through
  `scripts/run_pytest_isolated.py` rather than bare `pytest`. It gives each run
  a unique temp root for basetemp and cache.
- Start from a tiny deterministic synthetic regression test.
- Prefer observable behaviour over implementation detail, except for small
  architectural guards such as the paper-trade `GraphBuilder` ban.
- Mark data-, credential-, GPU-, or runtime-dependent tests with the existing
  pytest markers and add a fast companion test where practical.
- Do not move, archive, delete, or restructure tests without explicit approval.
- Regenerate `docs/TEST_REGISTRY.md` with `scripts/generate_test_registry.py`
  after adding, renaming, or removing tests; do not hand-edit it.
- Broaden from a focused proof to the non-slow suite plus ruff, then the full
  suite when shared pipeline, graph, model, or paper-trade contracts change.

```powershell
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py <focused-test> -v
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -m "not slow" -v
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -v
.\.venv\Scripts\python.exe -m ruff check .
```

### Artifact and evidence constraints

- `results/`, `outputs/`, `*.pth`, and `*.pt` are gitignored and are not source
  of truth merely because they exist locally.
- `seed_results/` holds committed experiment artifacts, not production code.
- Keep each `(year, base_seed)` distinct in frozen-prediction research and use
  `averaged_predictions/`; do not blend seeds before protocol-defined
  aggregation.
- Research evidence needs a real run, a reviewed source, or an explicit
  decision record. Synthetic fixtures stay synthetic.
- Handoffs preserve continuity but are not research evidence by default.
- Live Colab success requires visible execution plus the expected Drive-backed
  artifacts; notebook contract tests prove structure, not completion.
- Report unknown or partial evidence as unknown or partial.

### Workspace constraints

- Preserve unrelated tracked and untracked work. Inspect `git status` and the
  worktree topology before editing.
- Do not clean temporary directories, generated artifacts, branches, worktrees,
  or stashes unless the task explicitly authorises it.
- A read-only task does not authorise a commit, push, PR, GitHub mutation,
  Drive mutation, training run, or backtest.
- Treat credentials, unexpected authentication prompts, and expensive cloud
  work as stop-and-ask boundaries.

## 4. Target Architecture

Target-state architecture is not defined here. Use
[target-architecture.md](target-architecture.md) for that work, with the project
owner. Historical plans, research ideas, and agent proposals may inform the
discussion, but none of them becomes a target-state decision until it is
recorded there.

## Guide Maintenance

Update this guide when a change affects a routing contract:

- a composition root or owning module moves;
- a public or serialised contract changes;
- a canonical document is added, replaced, or demoted;
- a task category needs a different minimum read set;
- an engineering constraint becomes stable policy.

Do not copy volatile values into this guide: test counts, benchmark results,
branch names, or active worktree state. Config values are included only where
they are load-bearing for navigation, and they must be re-verified against
`configs/` and `mci_gru/config.py` when edited.
