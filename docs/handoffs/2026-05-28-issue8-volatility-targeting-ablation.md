# Issue 8 Volatility-Targeting Ablation Handoff

Last updated: 2026-05-29

## Resume Here

- Start with GitHub issue #8, especially the consolidated report comment:
  `https://github.com/magilliam27/MCI-GRU/issues/8#issuecomment-4569382089`.
- Current state: the Colab Stage 2 ablation sweep completed all 12 Stage 2 rows for 2024/2025, the runtime was disconnected/deleted, and issue #8 has been updated with the final readout.
- Immediate next move: plan repeated-seed validation for the narrow candidate set, not another ad hoc single-seed variant.

## Current Objective

- Decide whether the issue #8 Harvey-style volatility-targeting feature family should become a default, remain an explicit ablation/config option, or be reduced to a narrower risk-control signal.
- The current evidence says it is regime-sensitive and should not be blanket-enabled without repeated-seed confirmation.

## What Changed

- Completed the branch-backed Colab Stage 2 ablation sweep from:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/stage2_plus_scale_20260527_210208`.
- Posted a consolidated Issue #8 report covering:
  - all-years broad full VT results for 2022-2025,
  - 2023 component/guardrail ablations,
  - 2024/2025 Stage 2 ablations,
  - recommended next steps.
- Updated this handoff from the older paused-runtime state to the current completed-sweep state.

## Key Decisions

- Do not compare absolute baseline totals between the all-years broad VT read and the staged ablation read as if they are the same artifact family.
- Treat the original broad `[0.25, 4.0]` full bundle as regime-sensitive: bad in 2023, good in 2024/2025, slightly worse in 2022.
- Treat `vt_clip_0p50_2p0` as the current best compromise candidate, because it rescued 2023 and was strong in 2025, but it needs repeated-seed validation because it underperformed 2024.
- Treat `vt_scale_only` as a defensive/risk-control candidate, not the main return engine.
- Do not promote `vt_no_dynamics`; the 2023 negative-control row collapsed, suggesting dynamics were protective.

## Important Files

- `docs/research/current/ISSUE8_VOLATILITY_TARGETING_ALL_YEARS_BACKTEST_2026-05-27.md`: all-years broad full VT read for 2022-2025.
- `docs/research/current/ISSUE8_2023_VOL_TARGETING_DROP_DIAGNOSTICS_2026-05-27.md`: 2023 holding/ranking displacement diagnosis.
- `docs/research/current/ISSUE8_VOL_TARGETING_CAUSE_ANALYSIS_2026-05-27.md`: cross-year selected-holding explanation.
- `scripts/gen_volatility_targeting_ablation_nb.py`: generator for the staged Colab ablation notebook.
- `notebooks/volatility_targeting_ablation_colab.ipynb`: branch-backed Colab notebook used for the completed Stage 2 run.
- `tests/test_volatility_targeting_ablation_notebook.py`: notebook stage/variant regression checks.

## Verification

- Current turn inspected `git status --short`; main worktree is dirty with unrelated local/project work. Do not stage or absorb it casually.
- Current turn inspected the old handoff and local issue #8 research reports.
- Current turn fetched full Issue #8 comments with `gh issue view 8 --repo magilliam27/MCI-GRU --comments --json comments` after sandboxed network access failed.
- Current turn posted the consolidated issue report through the GitHub connector; returned comment id `4569382089`.
- No tests were run in this turn because the only repo edit was this handoff update and no code/config behavior changed.

## Open Risks

- All volatility-targeting ablation reads are still single-seed unless otherwise noted.
- 2022 has only the all-years broad full VT comparison; no component/guardrail ablation sweep has been run for 2022 yet.
- Stage 2 final CSV lives in Drive, not as a committed repo artifact. The issue comment and this handoff preserve the key metrics.
- The best policy may be regime-gated, but no explicit gating rule has been validated.

## Next Actions

1. Build a repeated-seed validation plan for `baseline_vol`, `vt_full_clip_0p25_4p0`, `vt_clip_0p50_2p0`, `vt_scale_only`, and `vt_ewm_only`.
2. Include 2022-2025 in the repeated-seed sweep so 2022 remains a guardrail year.
3. Reuse the 2023 diagnostic checklist for each candidate: holding overlap, monthly gross deltas, missed winners, high-vol/rebound exposure, and cost/turnover attribution.
4. Only after repeated-seed evidence, decide whether to add durable YAML presets or keep these as notebook-only ablation variants.

## Data/Experiment State

- Branch-backed notebook:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/issue8-vol-ablation-sweep/notebooks/volatility_targeting_ablation_colab.ipynb`
- Stage 2 Drive folder:
  `https://drive.google.com/drive/folders/13hb_AlipPZy0Pgod0wPf6NWuoi7m062z`
- Stage 2 results CSV:
  `https://drive.google.com/file/d/1b-Eoq-ZTRF5qZuWpBS94bvjsNwyWkGsa/view?usp=drivesdk`
- Stage 2 deltas CSV:
  `https://drive.google.com/file/d/1gs6FoD2FeK6wNVL6S-Cg3vSgZkAQfLse/view?usp=drivesdk`

All-years broad full VT, seed 314159:

| Year | Baseline total | Broad VT total | Delta total | Baseline ASR | Broad VT ASR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | -24.22% | -26.97% | -2.76 pp | -0.621 | -0.670 |
| 2023 | 60.40% | 6.33% | -54.07 pp | 2.377 | 0.337 |
| 2024 | 11.31% | 20.04% | +8.72 pp | 0.522 | 1.528 |
| 2025 | 38.79% | 50.98% | +12.19 pp | 1.292 | 1.374 |

2023 ablation highlights:

| Variant | Total return | ASR | MDD | Turnover | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_vol` | 19.47% | 0.641 | -26.29% | 14.89% | Control |
| `vt_full_clip_0p25_4p0` | 4.24% | 0.226 | -18.12% | 8.04% | Broad clip failure |
| `vt_scale_only` | 17.56% | 1.255 | -8.42% | 12.60% | Healthy defensive row |
| `vt_no_dynamics` | -10.72% | -0.374 | -28.00% | 12.34% | Negative control |
| `vt_clip_0p50_2p0` | 19.23% | 1.018 | -18.24% | 11.53% | Best 2023 rescue |
| `vt_clip_0p75_1p5` | 14.49% | 0.971 | -13.70% | 7.32% | Over-constrained return |

2024 Stage 2 final ranking by total return:

| Variant | Total return | ASR | MDD | Turnover |
| --- | ---: | ---: | ---: | ---: |
| `vt_full_clip_0p25_4p0` | 21.09% | 1.558 | -6.66% | 4.43% |
| `baseline_vol` | 20.69% | 0.884 | -21.91% | 17.81% |
| `vt_scale_only` | 17.70% | 1.810 | -6.63% | 3.76% |
| `vt_clip_0p50_2p0` | 9.30% | 0.719 | -12.43% | 0.72% |
| `vt_ewm_only` | 8.93% | 0.649 | -12.71% | 18.52% |
| `vt_no_scaled_return` | 8.63% | 0.903 | -7.18% | 3.12% |

2025 Stage 2 final ranking by total return:

| Variant | Total return | ASR | MDD | Turnover |
| --- | ---: | ---: | ---: | ---: |
| `vt_ewm_only` | 79.02% | 1.968 | -36.72% | 1.95% |
| `vt_clip_0p50_2p0` | 63.88% | 1.740 | -34.58% | 3.26% |
| `vt_full_clip_0p25_4p0` | 54.63% | 1.449 | -34.67% | 3.14% |
| `baseline_vol` | 39.13% | 1.110 | -33.02% | 8.90% |
| `vt_scale_only` | 37.15% | 1.151 | -25.44% | 7.46% |
| `vt_no_scaled_return` | 26.85% | 0.825 | -28.22% | 6.06% |

## Commands Run

- `git status --short`
- `git remote -v`
- `Get-Content docs/handoffs/2026-05-28-issue8-volatility-targeting-ablation.md`
- `Get-Content docs/research/current/ISSUE8_2023_VOL_TARGETING_DROP_DIAGNOSTICS_2026-05-27.md`
- `Get-Content docs/research/current/ISSUE8_VOLATILITY_TARGETING_ALL_YEARS_BACKTEST_2026-05-27.md`
- `rg -n "vt_clip_0p50|vt_clip_0p75|vt_scale_only|vt_no_dynamics|stage1|Stage 1|0.1923|0.1449|0.0424" docs scripts notebooks tests .codex -S`
- `gh issue view 8 --repo magilliam27/MCI-GRU --comments --json comments`

## User Preferences

- Keep Colab monitoring quiet and event-driven.
- Do not restart paid Colab runtime without a clear next experiment and explicit user go-ahead.
- Preserve unrelated dirty work in the main checkout.
- Report source-of-truth drift instead of silently following stale handoffs.

## Do Not Do

- Do not mark issue #8 as solved or close it from single-seed evidence.
- Do not default-enable volatility targeting in configs yet.
- Do not rerun broad ad hoc variants before designing the repeated-seed candidate sweep.
- Do not click through Google Drive consent or auth boundaries on the user's behalf.

## References

- Consolidated issue report: `https://github.com/magilliam27/MCI-GRU/issues/8#issuecomment-4569382089`
- Previous final Stage 2 issue update: `https://github.com/magilliam27/MCI-GRU/issues/8#issuecomment-4568981011`
- 2023 tight-clip rescue issue update: `https://github.com/magilliam27/MCI-GRU/issues/8#issuecomment-4558347988`
- 2023 Stage 1 readout issue update: `https://github.com/magilliam27/MCI-GRU/issues/8#issuecomment-4558545533`
