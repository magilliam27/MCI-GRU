# Research Archive

Superseded research evidence summaries live here. Each row says what replaced
the report, which artifacts support it where it had any (run tags, Drive
folders), and what still stands as background.

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
| `ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md` | The April 2026 ablation on the anchored 2019-snapshot universe. Artifacts: Drive `MCI-GRU-Ablations`, runs `dynamic_graph_methods/20260430_024339` and `training_factor_ablation/20260430_024344`, both linked from the report. Its `top_k=20` finding is reconciled, and its recommendations superseded, by `current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md` (section 8), which re-ran top-K 20 under the corrected point-in-time protocol. Its observation that regime context behaves as a secondary tuning dimension is still cited as background. |
| `FULL_FEATURE_FACTORIAL_ABLATION.md` | The April 2026 design (last updated 2026-04-29) for a 2 × 4 × 3 momentum, graph and regime factorial of 24 primary runs on the anchored universe, launched through notebooks now at tag `archive/pre-cleanup-2026-09`. A design, not a result; the runs it planned are the April ablation report's, above. Superseded as a plan by the frozen recipe in `docs/DEFAULT_EXPERIMENT_RECIPE.md` and by the point-in-time evidence under `current/`. Background value: the factor definitions and the smoke-versus-primary budget split. |
| `MODERN_DEFAULTS_HANDOFF_2026.md` | The April 2026 handoff-style status of the `modern_defaults` ablation. Artifacts: Drive `MCI-GRU-Ablations/holdout_2026/20260429_005753`, named in the report. Its decisions are carried by `docs/DEFAULT_EXPERIMENT_RECIPE.md` and `configs/config.yaml`; kept for the reasoning behind the defaults. |
| `BACKTEST_FAIRNESS_AUDIT.md` | The February 2026 fairness audit of the original `evaluate_sp500.py` (a March note records its refactor into what is now `mci_gru/evaluation/backtest_engine.py`). No run artifact; it is a code audit. Its timing and attribution caveats are the origin of the backtest checks in `docs/TESTING_GUIDE.md` and the guard tests under `tests/`; its line references predate the refactor. |
| `PIT_2022_2024_REPLAY_DIAGNOSTICS_2026-05-24.md` | A May 24 memo on Option A saved-prediction replays for 2022 to 2024, built on the May 20 repeated-seed run `20260520_183538` (Drive `MCI-GRU-Ablations/pit_repeated_seed_replication/20260520_183538/summaries/pit_2022_2024_diagnostics_20260524_colab/`), the same run `current/PIT_REPEATED_SEED_OPTION_A_RESULTS_2026-05-21.md` reports. Never promoted to current: its question, the 2022 failure, is open on ticket #30, and the 2025 graph-specification evidence made the Option A path secondary. Kept as the last per-year replay diagnostic on those runs. |
| `MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md` | The broad June 19 opportunity scan; desk research, no run. Superseded as the prioritisation map by `current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md` on the status map's own ruling; kept for source leads. Several config presets it proposes were never created. |
| `MCI_GRU_PROGRAM_MAP_2026-06-19.md` | The June 19 structural companion map; desk research, no run. Its routing role is now carried by `docs/agents/guide.md` and its architecture description by `docs/ARCHITECTURE.md`; kept for the component-by-component tweak-surface tables. |
| `graph_signal_upgrades_plan_2026-04.md` | The April 2026 Cursor-era dynamic-graph audit and roadmap (levers 1–4); no run. Its audit layer is reflected in `docs/ARCHITECTURE.md`; its roadmap is superseded by the graph-specification evidence under `current/`, where the graph-zeroed control is not beaten. The Jaccard diagnostic script it names was never in this repository. Kept as background for the lever names tickets still use. |
