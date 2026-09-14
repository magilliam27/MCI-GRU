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

Current evidence must be traceable to a real run, reviewed source, or explicit
decision record. Synthetic fixtures, example values, and files generated under
pytest or other temporary test directories are not research evidence, even when
they exercise the same report schema. Keep those outputs in tests/examples and
label them as synthetic rather than promoting them into `current/`.

## Current Evidence Map

### Program Maps And Research Queues

| Report | Status |
| --- | --- |
| `docs/research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md` | Promoted active research map for implementation planning. Use this June 21 top-university-gated scan before the broader June 19 opportunity scan when prioritizing testing, backtesting, evidence-harness, data, model, loss, or paper-trade work. |
| `docs/research/current/MCI_GRU_PROGRAM_MAP_2026-06-19.md` | Current structural companion map for MCI-GRU components and safe tweak surfaces. |
| `docs/research/current/MCI_GRU_TRUNK_ARCHITECTURE_OPPORTUNITIES_2026-09-05.md` | Current trunk-architecture map: measured information-flow diagnostics on the four-stream trunk and a ranked, pre-registered trunk-hygiene ablation (residual cross-stock block, market-state gate, capacity-matched widths). Diagnostics are mechanics-level, not performance evidence. |
| `docs/research/current/MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md` | Superseded as the main prioritization map by the June 21 top-university-gated scan; keep as broader background and source-lead history. |

### PIT, Backtest, And Volatility-Targeting Evidence

| Report | Status |
| --- | --- |
| `docs/research/current/SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23.md` | Current reduced S&P 500 PIT GICS Top-10 multiyear baseline and caveat record. |
| `docs/research/current/ISSUE8_VOLATILITY_TARGETING_BACKTEST_2026-05-26.md` | Current first-pass Issue #8 volatility-targeting backtest impact read. |
| `docs/research/current/ISSUE8_VOLATILITY_TARGETING_ALL_YEARS_BACKTEST_2026-05-27.md` | Current all-years Issue #8 volatility-targeting backtest follow-up. |
| `docs/research/current/ISSUE8_2023_VOL_TARGETING_DROP_DIAGNOSTICS_2026-05-27.md` | Current diagnostic for the 2023 volatility-targeting drop. |
| `docs/research/current/ISSUE8_VOL_TARGETING_CAUSE_ANALYSIS_2026-05-27.md` | Current cause analysis and next-test routing for Issue #8 volatility targeting. |

### Graph Specification Evidence

Evidence produced under the Wayfinder map "the correlation graph's specification is
chosen on measured evidence" (issue #157).

| Report | Status |
| --- | --- |
| `docs/research/current/GRAPH_EDGE_DENSITY_PIT_AXIS_2026-08-28.md` | Current production figures for correlation-graph edge density, node isolation, and grid-point distinctness, measured on the PIT-admissible axis across 120 monthly build dates. Supersedes, as production numbers, the union-axis density table preserved in issue #157's original body; that table is reproduced here as a control and is not itself wrong, but its five figures describe a different axis. Read before quoting any edge-count, density, or isolation figure for the correlation graph. |
| `docs/research/current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md` | Current result of the graph-specification ablation run under the ticket-164 protocol (ticket 167): five arms — graph-zeroed control, shipped threshold 0.8, threshold 0.5, top-K 20, sector-relation-only — at screen (3 × 20) and frozen-recipe confirm (20 × 100) on the PIT 110-name 2016 universe. The graph-zeroed control ranks first on the decided arbiter (test-span pooled daily IC) at both stages, no arm separates from it, and every 95% CI contains zero. Read before quoting any arm-against-arm result; read its section 7 before quoting Sharpe, rank-IC, or composite figures, which order the arms differently from the arbiter. |
| `docs/research/current/GRAPH_PAIRED_REANALYSIS_2026-09-02.md` | Current paired re-analysis of the 2026-09-01 graph-specification ablation (ticket 179, Phase 0 of the multi-year protocol proposal on issue #157): every arm scored against the graph-zeroed control on the same 238 test days with overlap-aware HAC and block-bootstrap inference and BHY across the four comparisons; Spearman, winsorised, and median sensitivity; Sharpe with bootstrap error bars; seed-paired per-model IC; the ensemble-scale audit; and the power table (`sd(Δ)` per arm and the minimum detectable effect by pooled test days). No arm separates from the control on any variant, and every 2025 gap is below its arm's four-fold minimum detectable effect. Read before sizing or designing any multi-year graph protocol; read its section 6 before quoting a detectable-effect figure, and its section 3 before quoting the per-year IC table from the ablation report, which used a different label. |

### Loss-Path Evidence

| Report | Status |
| --- | --- |
| `docs/research/current/LOSS_FUNCTION_LITERATURE_SCAN_2026-06-03.md` | Current background scan for loss-function families and experiment ordering. |
| `docs/research/current/LOSS_PATH_DECISION_2026-06-04.md` | Current June 4 conservative loss-path decision note; read with later handoffs before making LambdaRankIC readiness claims. |
| `docs/research/current/LOSS_PATH_EXPERIMENTAL_SEARCH_2026-06-04.md` | Current exploratory companion for uncertainty-adjusted ranking, distributional heads, and deferred optimizer-style losses. |

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
