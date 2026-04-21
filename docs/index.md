# Documentation Index

> Start with `AGENTS.md` (repo root) for the quick map.
> Start with this file to find specific deep-dive docs.

## Architecture & Design

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model, pipeline, graph, data flow — the full system map |
| [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) | Prioritized upgrade roadmap + audit notes (**§0** = post–Phase 1 status; **§9** checklist) |
| [mci_gru_implementation_plan.md](mci_gru_implementation_plan.md) | Colab-oriented paper walkthrough; repo defaults differ — see ARCHITECTURE / CONFIGURATION_GUIDE |

## Configuration & Operations

| Document | Purpose |
|----------|---------|
| [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) | Hydra config system, override patterns, preset catalog |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Cheat sheet for common commands and workflows |
| [OUTPUT_MANAGEMENT.md](OUTPUT_MANAGEMENT.md) | Output directory structure, naming, cleanup |
| [MLFLOW_TRACKING.md](MLFLOW_TRACKING.md) | MLflow experiment tracking setup and usage |

## Data Contracts & Correctness

| Document | Purpose |
|----------|---------|
| [REGIME_DATA_CONTRACT.md](REGIME_DATA_CONTRACT.md) | Regime feature inputs, outputs, no-lookahead guarantees |
| [BACKTEST_FAIRNESS_AUDIT.md](BACKTEST_FAIRNESS_AUDIT.md) | Historical fairness audit (lookahead / return timing); line refs may predate refactors |
