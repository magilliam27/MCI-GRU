# Research Evidence

This directory defines the lifecycle for MCI-GRU research evidence. Existing
reports in the docs root are not moved in this first organization pass; this
file is the status map that tells agents which reports are current evidence and
which need later archive review.

## Lifecycle

- **Current Research Evidence**: a report still informing active model,
  validation, data, or experiment decisions.
- **Superseded Research Evidence**: a report whose facts remain useful history
  but whose conclusion has been replaced by newer evidence.
- **Research Archive**: the future home for superseded summary reports. Bulky
  artifacts, raw results, checkpoints, and run folders should stay in Drive or
  external artifact storage, with links or run IDs recorded in the summary.

## Current Evidence Map

Current branch-local research maps:

| Report | Status |
| --- | --- |
| `docs/research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md` | Active June 21 research map and source of truth for evidence-harness wave ordering. |

Keep these root-level reports easy to find until a follow-up pass moves or
reclassifies them:

| Report | Status |
| --- | --- |
| `docs/PIT_UNIVERSE_REPORT.md` | Current evidence for PIT universe construction and membership behavior. |
| `docs/PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16.md` | Current evidence for PIT/LSEG alias coverage. |
| `docs/PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md` | Current evidence for masked-panel full-run interpretation. |
| `docs/LONG_HISTORY_PIT_EVAL_RESULTS_2026-05-18.md` | Current evidence for long-history PIT evaluation. |
| `docs/PIT_REPEATED_SEED_OPTION_A_RESULTS_2026-05-21.md` | Current evidence for repeated-seed PIT replication, if present in the worktree. |
| `docs/TSFM_PREDICTION_REPORT.md` | Current method reference for saved-prediction evaluation reports. |

## Archive Review Queue

These reports should be reviewed before any move. Do not archive them just
because they are dated.

| Report | Review note |
| --- | --- |
| `docs/ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md` | Decide whether newer PIT evidence supersedes its recommendations. |
| `docs/FULL_FEATURE_FACTORIAL_ABLATION.md` | Decide whether it is an active experiment design or historical planning note. |
| `docs/MODERN_DEFAULTS_HANDOFF_2026.md` | Handoff-like status document; keep separate from research evidence if retained. |
| `docs/BACKTEST_FAIRNESS_AUDIT.md` | Historical fairness audit with useful caveats; canonical status should be checked against current code. |

## Handoffs

Handoffs remain under `docs/handoffs/` and are operational continuity notes.
They can be cited by a research report for provenance, but they are not research
evidence by themselves.
