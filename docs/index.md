# Documentation Index

> Start with `AGENTS.md` in the repo root for the short Codex entrypoint.
> Use this file to choose the right deep-dive document.

## Canonical Docs

| Document | Purpose |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current model, pipeline, graph, and data-flow map. |
| [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) | Hydra config system, override patterns, and preset catalog. |
| [DEFAULT_EXPERIMENT_RECIPE.md](DEFAULT_EXPERIMENT_RECIPE.md) | Frozen production-style recipe for confirmation notebooks and PIT validation. |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | MCI-GRU testing patterns, verification ladder, and test categories. |
| [REGIME_DATA_CONTRACT.md](REGIME_DATA_CONTRACT.md) | Regime feature inputs, outputs, and no-lookahead guarantees. |
| [NOTEBOOK_BEST_PRACTICES.md](NOTEBOOK_BEST_PRACTICES.md) | Colab notebook structure, Drive safety, outputs, and review checklist. |

## Operations And Evaluation

| Document | Purpose |
| --- | --- |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Cheat sheet for common commands and workflows. |
| [workflows/COLAB_CHROME_CONTROL_GUIDE.md](workflows/COLAB_CHROME_CONTROL_GUIDE.md) | Reusable agent runbook for operating authenticated Colab notebooks through Chrome control. |
| [OUTPUT_MANAGEMENT.md](OUTPUT_MANAGEMENT.md) | Output directory structure, naming, cleanup, and persistence. |
| [MLFLOW_TRACKING.md](MLFLOW_TRACKING.md) | MLflow experiment tracking setup and usage. |
| [TSFM_PREDICTION_REPORT.md](TSFM_PREDICTION_REPORT.md) | Saved-prediction R2, sign, IC, and yearly decay report method. |
| [evaluation/EVIDENCE_HARNESS.md](evaluation/EVIDENCE_HARNESS.md) | Additive run manifests, trial ledgers, saved-prediction audits, PIT availability reports, and capacity replay. |

## Research Evidence

| Document | Purpose |
| --- | --- |
| [research/README.md](research/README.md) | Current/archive research evidence lifecycle and report status map. |
| [research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md](research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md) | Active June 21 research map for evidence-harness wave ordering and guarded opportunities. |

Keep detailed current and superseded evidence lists in
[research/README.md](research/README.md), not in this top-level index.

## Historical And Reference Material

| Document | Purpose |
| --- | --- |
| [BACKTEST_FAIRNESS_AUDIT.md](BACKTEST_FAIRNESS_AUDIT.md) | Historical fairness audit; line references may predate refactors. |
| [agent_references/README.md](agent_references/README.md) | The previous Claude guidance; the Cursor-era material is at tag `archive/pre-cleanup-2026-09`. |

## Retired Material

Handoffs, the Cursor-era plans, and the pre-cleanup notebooks and scripts were
retired in 2026-09 (map #211). They are readable at tag
`archive/pre-cleanup-2026-09`; the tracker is the continuity surface.

## Agent Docs And Skills

| Document | Purpose |
| --- | --- |
| [agents/guide.md](agents/guide.md) | Current-state routing for nontrivial work: owning modules, adjacent contracts, focused tests, and engineering constraints. |
| [agents/target-architecture.md](agents/target-architecture.md) | Human-led workspace for future-state architecture decisions; intentionally undecided and not current-state authority. |
| [agents/domain.md](agents/domain.md) | Source-of-truth hierarchy and stale-doc policy. |
| [agents/issue-tracker.md](agents/issue-tracker.md) | GitHub issue tracker policy. |
| [agents/triage-labels.md](agents/triage-labels.md) | Default triage label vocabulary. |
| [research-paper-to-mci-gru](../skills/research-paper-to-mci-gru/SKILL.md) | Translate finance papers into MCI-GRU-aware briefs and issue drafts. |
