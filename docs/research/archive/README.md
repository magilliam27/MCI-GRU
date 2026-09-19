# Research Archive

Superseded research evidence summaries live here. Each row says what replaced
the report, where its artifacts are, and what still stands as background.

Archived summaries should state:

- what newer report or decision superseded them;
- which run IDs, Drive folders, or artifacts support the historical result;
- whether any conclusion is still useful as background.

Do not store bulky run artifacts, checkpoints, or raw outputs here. Keep those
in Drive or external artifact storage and link to them from the summary. Drive
links in these reports point at the maintainer's private artifact store; the
tables inside each report are the public record.

Code paths cited inside an archived report describe the tree as it stood when
the report was written and are not maintained.

## Archived Summaries

| Document | Status |
| --- | --- |
| `ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md` | The April 2026 ablation on the anchored 2019-snapshot universe. Its `top_k=20` finding is reconciled, and its recommendations superseded, by `current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md` (section 8), which re-ran top-K 20 under the corrected point-in-time protocol. Its regime-context observation is still cited as background. |
| `FULL_FEATURE_FACTORIAL_ABLATION.md` | A May 2026 experiment design for a full-feature factorial on the anchored universe, run through notebooks retired to tag `archive/pre-cleanup-2026-09`. Superseded as a plan by the frozen recipe in `docs/DEFAULT_EXPERIMENT_RECIPE.md` and by the point-in-time evidence under `current/`. |
| `MODERN_DEFAULTS_HANDOFF_2026.md` | A handoff-style status document from the modern-defaults work of spring 2026. Its decisions are carried by `docs/DEFAULT_EXPERIMENT_RECIPE.md` and `configs/config.yaml`; kept for the reasoning behind the defaults. |
| `BACKTEST_FAIRNESS_AUDIT.md` | The March 2026 fairness audit of the original `evaluate_sp500.py`, later refactored into `mci_gru/evaluation/backtest_engine.py`. Its timing and attribution caveats are the origin of the backtest checks in `docs/TESTING_GUIDE.md` and the guard tests under `tests/`; its line references predate the refactor. |
| `PIT_2022_2024_REPLAY_DIAGNOSTICS_2026-05-24.md` | A May 2026 memo on saved-prediction replays for 2022 to 2024 under Option A. Superseded by `current/PIT_REPEATED_SEED_OPTION_A_RESULTS_2026-05-21.md` for the repeated-seed read and by the graph-specification evidence for the 2025 test span; the 2022 diagnostic question is open on ticket #30. |
| `MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md` | The broad June 19 opportunity scan. Superseded as the prioritisation map by `current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md` on the status map's own ruling; kept for source leads. Several config presets it proposes were never created. |
| `MCI_GRU_PROGRAM_MAP_2026-06-19.md` | The June 19 structural companion map. Its routing role is now carried by `docs/agents/guide.md` and its architecture description by `docs/ARCHITECTURE.md`; kept for the component-by-component tweak-surface tables. |
| `graph_signal_upgrades_plan_2026-04.md` | The April 2026 Cursor-era dynamic-graph audit and roadmap (levers 1–4). Its audit layer is reflected in `docs/ARCHITECTURE.md`; its roadmap is superseded by the graph-specification evidence under `current/`, where the graph-zeroed control is not beaten. Kept as background for the lever names `AGENTS.md` still uses. |
