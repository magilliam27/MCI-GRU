---
name: phase-4-eval-mlops
overview: Phase 4 adds a shared evaluation and paper-trade trust layer: statistical confidence intervals, Newey-West Sharpe, shared portfolio helpers, train-only feature references, live feature drift monitoring, Optuna sweep scaffolding, and CI smoke coverage.
todos:
  - id: p4-eval-statistics
    content: Add mci_gru/evaluation/statistics.py with daily IC, Newey-West Sharpe, and moving-block bootstrap confidence intervals
    status: completed
  - id: p4-portfolio-helpers
    content: Add mci_gru/evaluation/portfolio.py and route paper_trade portfolio rank-drop gate through it
    status: completed
  - id: p4-eval-config-summary
    content: Add EvaluationConfig, evaluation YAML defaults, test_labels, evaluation_summary.json, and MLflow evaluation metrics
    status: completed
  - id: p4-feature-reference
    content: Save train-only feature_reference.json and feature_reference_path metadata for drift monitoring
    status: completed
  - id: p4-paper-monitor
    content: Add paper_trade/scripts/monitor.py, normalized inference feature output, nightly monitor step, and report Feature Drift section
    status: completed
  - id: p4-edge-dim-helper
    content: Move edge feature width logic to mci_gru/graph/utils.py and use it in training and inference
    status: completed
  - id: p4-sweep-ci
    content: Add phase4_optuna_sweep.yaml, scripts/ci_smoke.py, and CI smoke workflow step
    status: completed
  - id: p4-docs-tests
    content: Add Phase 4 tests and update ARCHITECTURE / ARCHITECTURE_REVIEW docs
    status: completed
isProject: false
---

# Phase 4 - Eval & MLOps Trust Layer

Target: [docs/ARCHITECTURE_REVIEW.md](../../docs/ARCHITECTURE_REVIEW.md) Phase 4.

## Goals

1. Make model comparisons less dependent on point estimates by adding bootstrap confidence intervals and Newey-West Sharpe.
2. Share low-risk portfolio decisions between research/backtest and paper-trade code.
3. Persist train-only feature references and surface live feature drift in nightly paper-trade reports.
4. Keep Phase 3 graph edge-width inference compatible without importing `GraphBuilder` in paper-trade.
5. Add a lightweight CI smoke run and Optuna sweep example.

## Invariants

- No lookahead: feature references are fit from train-period rows only.
- Paper-trade inference still loads frozen `graph_data.pt`; it does not rebuild graphs.
- `combined_collate_fn` remains a 9-tuple.
- Existing metric names remain available; corrected/Newey-West variants are additive.
