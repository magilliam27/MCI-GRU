# AGENTS.md

> This file is the **table of contents** for any AI agent working in this repository.
> It is intentionally short (~100 lines). Deep details live in the files linked below.

## Quick Commands

```bash
python -m pytest tests/ -v                         # run all tests
python run_experiment.py training.num_epochs=2 training.num_models=1  # smoke run
python paper_trade/scripts/run_nightly.py           # nightly paper-trade pipeline
```

## Repository Map

```
AGENTS.md            ← you are here (start point for all agents)
CLAUDE.md            ← Claude-specific guidance (extends this file)
docs/
├── ARCHITECTURE.md  ← model, pipeline, graph, data flow (READ THIS FIRST)
├── CONFIGURATION_GUIDE.md
├── QUICK_REFERENCE.md
├── REGIME_DATA_CONTRACT.md
├── BACKTEST_FAIRNESS_AUDIT.md
├── OUTPUT_MANAGEMENT.md
├── MLFLOW_TRACKING.md
└── mci_gru_implementation_plan.md
configs/             ← Hydra YAML (config.yaml is the base; graph experiments under configs/experiment/)
.cursor/plans/graph_signal_upgrades_c28cf640.plan.md  ← dynamic-graph audit + roadmap (levers 1–4)
mci_gru/             ← core Python package
├── config.py        ← typed dataclass configs (ExperimentConfig)
├── pipeline.py      ← central orchestrator: load → features → normalize → window → graph
├── models/mci_gru.py   ← four-stream architecture (A1, A2, B1, B2)
├── data/            ← DataManager, preprocessing, loaders (LSEG, FRED, CSV)
├── features/        ← FeatureEngineer + registry (momentum, vol, credit, regime)
├── graph/builder.py ← Pearson-correlation graph (static or dynamic)
└── training/        ← Trainer, losses (MSE/IC/combined), metrics
paper_trade/         ← frozen-checkpoint inference + portfolio pipeline
tests/               ← pytest suite + backtest scripts
```

## Invariants — Do Not Break

1. **No lookahead**: normalization stats, graph edges, and labels use strict train-period cutoffs.
2. **Dynamic graph uses `GraphSchedule`**: precomputed snapshots indexed by date; any batch size works.
3. **`combined_collate_fn` returns a 7-tuple**: `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates)`. Here `edge_weight` is either shape `(E,)` (legacy scalar weight) or `(E, 4)` when `graph.use_multi_feature_edges=true`; collate concatenates along dim 0.
4. **Ensemble averaging**: `train_multiple_models` trains N independent models; prediction = mean.
5. **Paper-trade inference does not use `GraphBuilder`**: it loads a frozen `graph_data.pt`.

## Environment

- Python 3.10+
- See `pyproject.toml` for all dependencies (install: `pip install -e ".[dev]"`)
- `FRED_API_KEY` env var required when credit spread or regime features are enabled
- See `.env.example` for all environment variables

## How to Work in This Repo

- **Before editing**, read `docs/ARCHITECTURE.md` for the data flow and model structure.
- **Before adding features**, read `mci_gru/features/registry.py` for the plugin pattern.
- **Before changing the graph**, read `mci_gru/graph/builder.py`, `docs/ARCHITECTURE.md` (Graph section), and `.cursor/plans/graph_signal_upgrades_c28cf640.plan.md` (audit + roadmap).
- **Before touching paper_trade/**, understand that it uses frozen checkpoints — do not import `GraphBuilder`.
- **Run tests** after every change: `python -m pytest tests/ -v`
- **Config changes** go through Hydra YAML in `configs/` — see `docs/CONFIGURATION_GUIDE.md`.

## Testing

```bash
python -m pytest tests/ -v                                    # full suite
python -m pytest tests/test_dynamic_graph_updates.py -v       # single file
python -m pytest tests/ -k "test_no_lookahead" -v             # by keyword
python -m pytest tests/ -m "not slow" -v                      # skip slow tests
```

Tests verify: no-lookahead invariants, dynamic graph wiring, momentum blend modes,
regime data contracts, backtest fairness, output management, MLflow tracking.

## Correlation graph: plan vs implementation

The file `.cursor/plans/graph_signal_upgrades_c28cf640.plan.md` has two layers: (1) an **audit** that the dynamic graph is wired end-to-end (no lookahead; `GraphSchedule.get_graph_for_date` in `combined_collate_fn` when `graph.update_frequency_months > 0`; `run_experiment.py` sets `dynamic_graph` from that flag), and (2) a **roadmap** of levers 1–4. The YAML frontmatter todos there are still largely *pending* relative to that roadmap.

**Implemented today (code, not the whole roadmap)**

- **Dynamic schedule**: If `graph.update_frequency_months > 0`, `prepare_data` in `mci_gru/pipeline.py` calls `GraphBuilder.precompute_snapshots(...)` and passes `graph_schedule` into `create_data_loaders(..., dynamic_graph=True)`. Each batch resolves edges for the sample date via the schedule (see `mci_gru/data/data_manager.py` `combined_collate_fn`).
- **Lever 1a (partial)**: `GraphConfig.top_k` and `GraphConfig.top_k_metric` (`"corr"` or `"abs_corr"`). `top_k == 0` keeps the legacy global threshold `corr > judge_value`. `top_k > 0` selects per-node top-K neighbours (`mci_gru/graph/builder.py` `build_edges` / `_select_edges_topk`).
- **Lever 1c (partial vs plan)**: `GraphConfig.use_multi_feature_edges` makes `build_edges` return a **4-D** edge tensor `[corr, |corr|, corr^2, rank_pct]` instead of a scalar `(E,)`. The plan text also mentioned a 5th channel (`snapshot_age_days`) and `edge_dim=5`; the codebase uses **four** features and `run_experiment.py` sets `edge_feature_dim` to **4** or **1** when constructing the model. GAT blocks consume that width via `create_model(..., edge_feature_dim=...)`.
- **Experiments**: Use Hydra includes such as `configs/experiment/correlation_dynamic.yaml` (6-month updates) or `correlation_dynamic_topk20_pos.yaml` (top-K + multi-feature + updates) rather than relying on base `configs/config.yaml` alone (defaults: static graph, `top_k=0`, `use_multi_feature_edges=false`).

**Still roadmap / not implemented as described in that plan**

- Signed two-relation / `RGATConv` (Lever 1b), lead–lag edges (Lever 2a), graph-aware temporal encoder (Lever 3a), shorter cadence defaults (Lever 4b), rate-of-change edge feature (Lever 4c), fifth edge feature column, and the optional graph-zeroed ablation workflow called out in the plan.

**Diagnostic**

- `scripts/diagnose_dynamic_graph.py` — reproduces snapshot edge-count / Jaccard style statistics from the audit.

## Code Style

- Linting: `ruff check .` (config in `pyproject.toml`)
- Formatting: `ruff format .`
- No inline imports — keep imports at top of file
- Type hints on all public functions

## Key Gotchas

- `results/`, `outputs/`, `*.pth`, `*.pt` are gitignored — don't reference them as source of truth
- The `archive/` directory contains legacy code — do not treat as current
- `seed_results/` and `_uncertain/` are experimental artifacts, not production code
