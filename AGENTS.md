# AGENTS.md

> This file is the **Codex-first table of contents** for agents working in this repository.
> It is intentionally short (~100 lines). Deep details live in the files linked below.

## Quick Commands

```powershell
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -v  # Windows-preferred full suite
.\.venv\Scripts\python.exe scripts/ci_smoke.py  # end-to-end smoke on a synthetic panel; what CI runs
.\.venv\Scripts\python.exe run_experiment.py data.use_pit_universe=false training.num_epochs=2 training.num_models=1 tracking.enabled=false  # smoke run against the default CSV group (needs its CSV locally); PIT masks off
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

## Agent skills

### Issue tracker

Implementation work and PRDs are tracked in GitHub Issues for
`magilliam27/MCI-GRU`. Write-oriented workflows may autonomously manage their
issue and PR lifecycle through scoped `codex/*` or `claude/*` branches, never
directly on `main`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the two category roles and five state roles from the Matt Pocock triage
vocabulary, with their exact default label strings. A fully triaged issue should
carry one of each — but the state machine's first move is `needs-triage` alone,
before a category is determined, so a state role without a category is a normal
intermediate rather than a defect. Missing canonical labels may be created as
part of the requested workflow. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository with its glossary in root `CONTEXT.md` and
decisions under `docs/adr/` when created. See `docs/agents/domain.md`.

## Repository Map

```
AGENTS.md            ← you are here (start point for all agents)
CLAUDE.md            ← Claude Code entrypoint; imports AGENTS.md
docs/
├── ARCHITECTURE.md  ← model, pipeline, graph, data flow (READ THIS FIRST)
├── CONFIGURATION_GUIDE.md
├── DEFAULT_EXPERIMENT_RECIPE.md
├── TESTING_GUIDE.md
├── agents/          ← current-state guide, target architecture, issue tracker, triage labels, source-of-truth policy
├── research/        ← current/archive research evidence lifecycle
├── QUICK_REFERENCE.md
├── REGIME_DATA_CONTRACT.md
├── OUTPUT_MANAGEMENT.md
└── MLFLOW_TRACKING.md
configs/             ← Hydra YAML (config.yaml is the base; graph experiments under configs/experiment/)
docs/research/archive/graph_signal_upgrades_plan_2026-04.md  ← the April dynamic-graph audit + roadmap (levers 1–4), archived
docs/agent_references/claude/CLAUDE.md  ← previous Claude guidance, retained until the CLAUDE.md ticket retires it
mci_gru/             ← core Python package
├── config.py        ← typed dataclass configs (ExperimentConfig)
├── pipeline.py      ← central orchestrator: load → features → normalize → window → graph
├── models/factory.py + trunk.py ← model construction and four-stream architecture (A1, A2, B1, B2)
├── data/            ← DataManager, preprocessing, loaders (LSEG, FRED, CSV)
├── features/        ← FeatureEngineer + registry (momentum, vol, credit, regime)
├── graph/           ← Pearson-correlation graph (static or dynamic): builder, correlation math, schedule, sector, edge width
└── training/        ← Trainer, losses (MSE/IC/combined), ensemble
skills/              ← versioned Codex skills for GitHub review/upload
.claude/skills/      ← repo-owned model-invoked skills (work-the-map, implement-ticket)
tests/               ← pytest suite and golden fixtures
```

## Invariants — Do Not Break

1. **No lookahead**: normalization stats, graph edges, and labels use strict train-period cutoffs.
2. **Dynamic graph uses `GraphSchedule`**: precomputed snapshots indexed by date; any batch size works.
3. **`combined_collate_fn` returns a 9-tuple**: `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates, edge_index_sector, edge_weight_sector)`. The first seven entries match the historical contract; the last two are `None` unless `graph.use_sector_relation=true`. `edge_weight` is `(E,)`, `(E, 4)`, or wider when lead-lag / snapshot-age columns are enabled; collate concatenates along dim 0.
4. **Ensemble averaging**: `train_multiple_models` trains N independent models; prediction = mean.
5. **True PIT masked panels keep breadth**: `data.pit_universe_mode=masked_panel`
   keeps a fixed PIT union axis and carries daily stock masks; do not replace it
   with complete-stock filtering or continuous-member/stayer-only filtering.

## Environment

- Python 3.10+
- **Dependency manifest:** `pyproject.toml` is source of truth for version ranges.
  - `requirements.lock` — pinned dev+fred closure (Windows-reference; regenerate with
    `pip-compile --extra=dev --extra=fred -o requirements.lock pyproject.toml`; do not
    pip-sync shared venvs).
  - `requirements.txt` — Colab-facing core ranges (kept as ranges, NOT pins, so Colab's
    preinstalled torch/CUDA satisfies them; keep manually in sync with `pyproject.toml`).
- Install for development: `pip install -e ".[dev,fred]"` (CI uses CPU torch index on ubuntu).
- `FRED_API_KEY` env var required when credit spread or regime features are enabled
- See `.env.example` for all environment variables

## How to Work in This Repo

- **Before editing**, read `docs/ARCHITECTURE.md` for the data flow and model structure.
- **For nontrivial work**, use `docs/agents/guide.md` to locate the owning
  modules, adjacent contracts, focused tests, and engineering constraints.
- **For future-state design**, work with the project owner in
  `docs/agents/target-architecture.md`; it is human-led and is not a description
  of current behaviour.
- **When docs disagree**, current code and the invariants in this file win; see `docs/agents/domain.md`.
- **`paper_trade/` was retired in 2026-09 (map #211).** It is readable at tag
  `archive/pre-cleanup-2026-09`, and `tests/test_repository_retirement_guard.py`
  forbids reintroducing it, as it does the cockpit surfaces.
- **For automated Colab work**, default to `chrome:control-chrome` and the runbook in `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`; use Playwright MCP only as a documented legacy fallback.
- **For Colab evidence**, notebook contract tests are not live-run proof; live Colab claims need visible Chrome/Colab execution plus Drive artifacts (heartbeat/results), per `docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`.
- **Before translating finance papers into implementation work**, use `skills/research-paper-to-mci-gru/` to produce an MCI-GRU-aware brief and GitHub-ready issue drafts.
- **Before adding features**, read `mci_gru/features/registry.py` for the plugin pattern.
- **Before changing the graph**, read `mci_gru/graph/builder.py` and `mci_gru/graph/correlation.py`, `docs/ARCHITECTURE.md` (Graph section), and `docs/research/archive/graph_signal_upgrades_plan_2026-04.md` (audit + roadmap, archived).
- **Run tests** after every change with the repo venv and isolated pytest launcher
  on Windows: `.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -v`.
  It creates a unique temp root for each run and routes both pytest's basetemp
  and cache into it so normal-user and sandbox identities never reuse
  identity-specific ACLs. See `docs/TESTING_GUIDE.md`.
- **Config changes** go through Hydra YAML in `configs/` — see `docs/CONFIGURATION_GUIDE.md`.

## Testing

```powershell
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -v
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/test_dynamic_graph_updates.py -v
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -k "test_no_lookahead" -v
.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -m "not slow" -v
```

Tests verify: no-lookahead invariants, dynamic graph wiring, momentum blend modes,
regime data contracts, backtest fairness, output management, MLflow tracking.

`docs/TEST_REGISTRY.md` is the auto-generated registry of all tests (with
last-run status when `test_reports/junit.xml` exists). Regenerate after adding
or renaming tests: `.\.venv\Scripts\python.exe scripts/generate_test_registry.py`.

## Correlation graph

How the graph is built, selected (threshold or top-K), given edge features, and scheduled is documented in `docs/ARCHITECTURE.md` (Graph section) and routed in `docs/agents/guide.md` (Correlation edge selection, Edge feature width, Graph change propagation).
Whether the graph earns its place is a measured question: see the Graph Specification Evidence section of `docs/research/README.md`, where the graph-zeroed control is not beaten by any specification tried.
The April 2026 roadmap those measurements replaced is archived at `docs/research/archive/graph_signal_upgrades_plan_2026-04.md`; its lever names are still used in tickets, its todos are not a plan.

## Code Style

- Linting: `ruff check .` (config in `pyproject.toml`)
- Formatting: `ruff format .`
- No inline imports — keep imports at top of file
- Type hints on all public functions

## Key Gotchas

- `results/`, `outputs/`, `*.pth`, `*.pt` are gitignored — don't reference them as source of truth
- `seed_results/`, `paper_trade/`, the handoffs, and the pre-cleanup notebooks and scripts were retired in 2026-09 (map #211) and are readable at tag `archive/pre-cleanup-2026-09`; do not treat them as source of truth
- The tracker is the continuity surface; use `docs/research/README.md` for current/archive evidence status.
