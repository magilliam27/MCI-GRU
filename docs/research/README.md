# Research Evidence

This directory defines the lifecycle for MCI-GRU research evidence. Reports live
under `current/` or `archive/`; the `docs/` root holds canonical guides only, and
`scripts/check_docs_sot.py` refuses a dated report placed there. This file is the
status map that tells agents which reports are current evidence and which are
superseded.

## Lifecycle

- **Current Research Evidence**: a report still informing active model,
  validation, data, or experiment decisions.
- **Superseded Research Evidence**: a report whose facts remain useful history
  but whose conclusion has been replaced by newer evidence.
- **Research Archive**: the home for superseded summary reports, with one
  status line per report in `archive/README.md`. Bulky artifacts, raw results,
  checkpoints, and run folders stay in Drive or external artifact storage, with
  links or run IDs recorded in the summary.

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

The June 19 program map and opportunity scan are archived; `docs/agents/guide.md`
now carries the routing role, and the status lines are in `archive/README.md`.

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

### Point-In-Time Universe And Evaluation Evidence

| Report | Status |
| --- | --- |
| `docs/research/current/PIT_UNIVERSE_REPORT.md` | Current evidence for PIT universe construction and membership behavior. Its alias-coverage companion and two CSVs sit beside it. |
| `docs/research/current/PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16.md` | Current evidence for PIT/LSEG alias coverage. |
| `docs/research/current/PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md` | Current evidence for masked-panel full-run interpretation. Its membership-progression audit and CSV are under `current/audits/`. |
| `docs/research/current/LONG_HISTORY_PIT_EVAL_RESULTS_2026-05-18.md` | Current evidence for long-history PIT evaluation. |
| `docs/research/current/PIT_REPEATED_SEED_OPTION_A_RESULTS_2026-05-21.md` | Current evidence for repeated-seed PIT replication. |
| `docs/research/current/TSFM_PREDICTION_REPORT.md` | Current method reference for saved-prediction evaluation reports. |

## Archive

Superseded reports live under `archive/`. `archive/README.md` carries one status
line per report saying what replaced it, which artifacts support it, and what
still stands as background; it is the only list of archived reports.

## Drive Links

Run folders and artifacts cited by these reports live in the maintainer's Google
Drive, which is private; a link there is a permission wall for anyone else. The
tables and figures inside each report are the public record. Run-backed reports
carry their run identity (run tag and Drive folder, and where recorded the commit
and data hash) so a maintainer can locate the artifact; literature scans and
method notes cite their sources instead.

## Handoffs

Handoffs were retired in 2026-09 (map #211) and are readable at tag
`archive/pre-cleanup-2026-09`. A current report may still cite one for provenance
by its tag path; the tracker is the continuity surface.
