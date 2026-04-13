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
configs/             ← Hydra YAML (config.yaml is the base)
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
2. **Dynamic graph requires `batch_size=1`**: enforced in `ExperimentConfig.__post_init__`.
3. **`combined_collate_fn` returns a 7-tuple**: `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates)`.
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
- **Before changing the graph**, read `mci_gru/graph/builder.py` and the static/dynamic mode docs.
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

## Code Style

- Linting: `ruff check .` (config in `pyproject.toml`)
- Formatting: `ruff format .`
- No inline imports — keep imports at top of file
- Type hints on all public functions

## Key Gotchas

- `results/`, `outputs/`, `*.pth`, `*.pt` are gitignored — don't reference them as source of truth
- The `archive/` directory contains legacy code — do not treat as current
- `seed_results/` and `_uncertain/` are experimental artifacts, not production code
