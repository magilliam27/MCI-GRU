# AGENTS.md

> This file is the **Codex-first table of contents** for agents working in this repository.
> It is intentionally short (~100 lines). Deep details live in the files linked below.

## Quick Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest  # Windows-preferred full suite
.\.venv\Scripts\python.exe run_experiment.py training.num_epochs=2 training.num_models=1 data.source=csv tracking.enabled=false  # smoke run (CSV + no MLflow)
.\.venv\Scripts\python.exe paper_trade/scripts/run_nightly.py  # nightly paper-trade pipeline
```

## Default Experiment Recipe

For production-style confirmation notebooks and PIT validation runs, use the
frozen recipe in `docs/DEFAULT_EXPERIMENT_RECIPE.md`:
`static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`.
It means a 20-model, 100-epoch, patience-15 ensemble; pure IC loss; raw 5-day
return labels; `selection_metric=val_ic`; shuffled static threshold graph;
multi-feature edges; `drop_edge_p=0.1`; static weekly momentum; and strict
current-only global regime features. `FRED_API_KEY` is required unless a smoke
run explicitly disables global regime.

## Repository Map

```
AGENTS.md            ← you are here (start point for all agents)
docs/
├── ARCHITECTURE.md  ← model, pipeline, graph, data flow (READ THIS FIRST)
├── CONFIGURATION_GUIDE.md
├── DEFAULT_EXPERIMENT_RECIPE.md
├── TESTING_GUIDE.md
├── agents/          ← issue tracker, triage labels, source-of-truth policy
├── research/        ← current/archive research evidence lifecycle
├── QUICK_REFERENCE.md
├── REGIME_DATA_CONTRACT.md
├── BACKTEST_FAIRNESS_AUDIT.md
├── OUTPUT_MANAGEMENT.md
├── MLFLOW_TRACKING.md
└── mci_gru_implementation_plan.md
configs/             ← Hydra YAML (config.yaml is the base; graph experiments under configs/experiment/)
docs/agent_references/
├── claude/CLAUDE.md  ← Claude-specific guidance, retained for reference
└── cursor/plans/graph_signal_upgrades_c28cf640.plan.md  ← dynamic-graph audit + roadmap (levers 1–4)
mci_gru/             ← core Python package
├── config.py        ← typed dataclass configs (ExperimentConfig)
├── pipeline.py      ← central orchestrator: load → features → normalize → window → graph
├── models/mci_gru.py   ← four-stream architecture (A1, A2, B1, B2)
├── data/            ← DataManager, preprocessing, loaders (LSEG, FRED, CSV)
├── features/        ← FeatureEngineer + registry (momentum, vol, credit, regime)
├── graph/builder.py ← Pearson-correlation graph (static or dynamic)
└── training/        ← Trainer, losses (MSE/IC/combined), metrics
paper_trade/         ← frozen-checkpoint inference + portfolio pipeline
skills/              ← versioned Codex skills for GitHub review/upload
tests/               ← pytest suite + backtest scripts
```

## Invariants — Do Not Break

1. **No lookahead**: normalization stats, graph edges, and labels use strict train-period cutoffs.
2. **Dynamic graph uses `GraphSchedule`**: precomputed snapshots indexed by date; any batch size works.
3. **`combined_collate_fn` returns a 9-tuple**: `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates, edge_index_sector, edge_weight_sector)`. The first seven entries match the historical contract; the last two are `None` unless `graph.use_sector_relation=true`. `edge_weight` is `(E,)`, `(E, 4)`, or wider when lead-lag / snapshot-age columns are enabled; collate concatenates along dim 0.
4. **Ensemble averaging**: `train_multiple_models` trains N independent models; prediction = mean.
5. **Paper-trade inference does not use `GraphBuilder`**: it loads a frozen `graph_data.pt`.
6. **True PIT masked panels keep breadth**: `data.pit_universe_mode=masked_panel`
   keeps a fixed PIT union axis and carries daily stock masks; do not replace it
   with complete-stock filtering or continuous-member/stayer-only filtering.

## Environment

- Python 3.10+
- See `pyproject.toml` for all dependencies (install: `pip install -e ".[dev]"`)
- `FRED_API_KEY` env var required when credit spread or regime features are enabled
- See `.env.example` for all environment variables

## How to Work in This Repo

- **Before editing**, read `docs/ARCHITECTURE.md` for the data flow and model structure.
- **When docs disagree**, current code and the invariants in this file win; see `docs/agents/domain.md`.
- **Before starting new Codex branches/worktrees**, base them on `origin/main` unless the user explicitly says to continue an existing feature branch; use a fresh `codex/<task-name>-YYYYMMDD` branch for scoped work.
- **Before cleaning stale branches/worktrees**, follow `docs/agents/worktree-hygiene.md`; classify and report candidates first, and delete nothing without explicit user approval for the exact target.
- **For automated Colab work**, default to `chrome:control-chrome` and the runbook in `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`; use Playwright MCP only as a documented legacy fallback.
- **For Colab evidence**, notebook contract tests are not live-run proof; live Colab claims need visible Chrome/Colab execution plus Drive artifacts (heartbeat/results), per `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`.
- **Before translating finance papers into implementation work**, use `skills/research-paper-to-mci-gru/` to produce an MCI-GRU-aware brief and GitHub-ready issue drafts.
- **Before adding features**, read `mci_gru/features/registry.py` for the plugin pattern.
- **Before changing the graph**, read `mci_gru/graph/builder.py`, `docs/ARCHITECTURE.md` (Graph section), and `docs/agent_references/cursor/plans/graph_signal_upgrades_c28cf640.plan.md` (audit + roadmap).
- **Before touching paper_trade/**, understand that it uses frozen checkpoints — do not import `GraphBuilder`.
- **Run tests** after every change with the repo venv and repo-local pytest temp on Windows: `.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest`; system Python/profile temp has been unreliable here. See `docs/TESTING_GUIDE.md`.
- **Config changes** go through Hydra YAML in `configs/` — see `docs/CONFIGURATION_GUIDE.md`.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\python.exe -m pytest tests/test_dynamic_graph_updates.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\python.exe -m pytest tests/ -k "test_no_lookahead" -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\python.exe -m pytest tests/ -m "not slow" -v --basetemp .tmp_pytest\pytest
```

Tests verify: no-lookahead invariants, dynamic graph wiring, momentum blend modes,
regime data contracts, backtest fairness, output management, MLflow tracking.

## Correlation graph: plan vs implementation

The file `docs/agent_references/cursor/plans/graph_signal_upgrades_c28cf640.plan.md` has two layers: (1) an **audit** that the dynamic graph is wired end-to-end (no lookahead; `GraphSchedule.get_graph_for_date` in `combined_collate_fn` when `graph.update_frequency_months > 0`; `run_experiment.py` sets `dynamic_graph` from that flag), and (2) a **roadmap** of levers 1–4. The YAML frontmatter todos there are still largely *pending* relative to that roadmap.

**Implemented today (code, not the whole roadmap)**

- **Dynamic schedule**: If `graph.update_frequency_months > 0`, `prepare_data` in `mci_gru/pipeline.py` calls `GraphBuilder.precompute_snapshots(...)` and passes `graph_schedule` into `create_data_loaders(..., dynamic_graph=True)`. Each batch resolves edges for the sample date via the schedule (see `mci_gru/data/data_manager.py` `combined_collate_fn`).
- **Lever 1a (partial)**: `GraphConfig.top_k` and `GraphConfig.top_k_metric` (`"corr"` or `"abs_corr"`). `top_k == 0` keeps the legacy global threshold `corr > judge_value`. `top_k > 0` selects per-node top-K neighbours (`mci_gru/graph/builder.py` `build_edges` / `_select_edges_topk`).
- **Lever 1c + Phase 3**: `GraphConfig.use_multi_feature_edges` makes `build_edges` return at least **4** channels `[corr, |corr|, corr^2, rank_pct]`, optionally **+2** lead–lag columns (`use_lead_lag_features`). `append_snapshot_age_days` adds **one** column at collate time. `run_experiment._edge_feature_dim` must match the final width passed to `create_model`.
- **Experiments**: Use Hydra includes such as `configs/experiment/correlation_dynamic.yaml` (6-month updates) or `correlation_dynamic_topk20_pos.yaml` (top-K + multi-feature + updates) for dynamic-graph presets. Base `configs/config.yaml` defaults: static graph, `top_k=0`, `use_multi_feature_edges=true`.

**Still roadmap / not implemented as described in that plan**

- `RGATConv` / true multi-relation message passing (Phase 3 ships **dual GAT + fuse** for sector instead); graph-aware temporal encoder (Lever 3a), shorter cadence defaults (Lever 4b), rate-of-change edge feature (Lever 4c), and the optional graph-zeroed ablation workflow called out in the plan.

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
- Handoffs are operational continuity notes, not research evidence; use `docs/research/README.md` for current/archive evidence status.
