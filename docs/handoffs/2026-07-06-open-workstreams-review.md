# Open Workstreams Review — 2026-07-06

Post-merge project-status review, run the evening PR #57 (`fe6271a`,
"Rearchitecture phase 0") landed on `main`. Conducted read-only from
worktree `C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU` on branch
`codex/rearchitecture-phase0-20260704` (content-identical to `origin/main`,
0 ahead / 1 behind — the only missing commit is the merge commit itself).

Evidence sources: the 7 handoffs in `docs/handoffs/`, per-worktree git probes
across `C:\Users\magil\.codex\worktrees\`,
`gh pr list` / `gh issue list`, `git branch -a`, sampled Codex session records
under `C:\Users\magil\.codex\sessions\`, and `docs/research/README.md`.

---

## Open Workstreams Board

### 1. Active research (priority 1): LambdaRankIC stability

**Decision standing (consistent across all handoffs):** keep **pure IC** as the
launch/default objective; LambdaRankIC stays **experimental**; Portfolio-IC
weight50 is a promising low-turnover variant but not a default replacement.

**Evidence base** (handoffs `2026-06-30-sp500-top10-loss-backtest-comparison.md`
and `...-all-years-recovered.md`): the 28-row loss/seed matrix (2022–2025,
110-name PIT GICS top-10-per-sector) is complete after Drive-log recovery.
LambdaRankIC has the best row-average (19.10% vs 13.06% pure IC) and clearly wins
the 2022 stress year (−22.7% vs −40.6%) and 2023 — but 2024 seed `271828` is a
severe churn outlier: 15.88% net, 362 trades, 3.62% cumulative cost, 18.1x
cumulative one-way top-10 turnover (vs 6.0x for the sibling seed). Cross-seed
return gaps of 20–24 pp in 2023/2024 make it not default-ready.

**Where it stalled** (handoffs `2026-06-30-lambdarankic-rank-stability-diagnostics.md`,
`2026-07-01-lambdarankic-{cross-seed-agreement,rank-drop-sensitivity-replay,
stability-diagnostics-coordination}.md`): four no-retrain diagnostics (cost
sensitivity, rank stability, rank-drop-gate sensitivity at 20/30/50/75, cross-seed
agreement) are all **blocked locally** — no `averaged_predictions/` CSVs and no
top-10 PIT market/universe CSVs on this machine. Drive search found the older
seed-`314159` top10 prediction folders and one Portfolio-IC 2025 artifact, but
**not** the current repeated-seed LambdaRankIC `161803`/`271828` prediction
folders (they may exist only under the Drive `artifacts/local_run_root/training`
tree, unexposed to search, or be lost with the Colab VM).

**Next steps, per the coordination handoff (explicit ordering):**

1. Run a saved-prediction-only diagnostic job **in Colab/Drive** (not on the
   local PC): rank-stability + cross-seed agreement on 2024 LambdaRankIC
   `161803` vs `271828`. The diagnostic code is fully drafted in the
   rank-stability handoff.
2. Replay 2024 seed `271828` at rank-drop gates 20/30/50/75.
3. Replay the same row at cost assumptions 10+5 / 5+2.5 / 1+0.5 / 0+0 bps.
4. Only if 2024 becomes controlled, expand to 2023/2025 rows.
5. Promotion gate: LambdaRankIC can become default only if the 2024 `271828`
   instability is explained or mitigated (cost-only mitigation is insufficient —
   gross 2024 return was only 20.14%).

**Rebase note (rearchitecture impact):** every replay command in these handoffs
invokes `tests/backtest_sp500_daily.py`, which is now a deprecated shim over
`scripts/backtest_sp500_daily.py` / `mci_gru/evaluation/backtest_engine.py`.
Commands still work today, but any Colab notebook or script generated for this
diagnostic should target the new `scripts/` path directly.

**Housekeeping:** the 6 handoff files are untracked in this worktree only. They
are the sole record of this line of work — commit them.

Codex session records corroborate: the 2026-07-01 delegated thread ("plan, not
execution, for what additional LambdaRankIC runs are needed to verify base-seed
stability … before considering LambdaRankIC as default") is the planning parent
of these handoffs. A separate 2026-06-30 thread produced an AI-process
improvement plan from a best-practice PDF — its artifact is the untracked
`docs/agents/agentic-engineering-technical-spec.md` in the primary checkout
(decide: commit or discard).

### 2. Post-merge chores (priority 2 — most are quick)

| # | Chore | Evidence | State | Next action |
| --- | --- | --- | --- | --- |
| 1 | Sync primary checkout to main | Merge-plan handoff step 4; primary is on `codex/top10-lambdarank-screen-20260625` with 1 untracked doc | Not done | Park/commit the untracked spec doc, then `git checkout main; git pull` in `C:\Users\magil\MCI-GRU` |
| 2 | Commit the 6 LambdaRankIC handoffs | This worktree, untracked | Not done | Commit to main (docs-only) |
| 3 | Regenerate `notebooks/performance_proof_missing_grid_colab.ipynb` | Merge-plan follow-up 2 | Not done | Regenerate against `scripts/backtest_sp500.py`; must precede shim removal |
| 4 | File shim-removal tracking issue (~early Aug 2026) | Merge-plan follow-up 1 | Not done | Issue listing `tests/backtest_sp500*.py`, `mci_gru/models/mci_gru.py` shim, GraphBuilder wrappers, `train_multiple_models` facade; grep notebooks/docs before removal |
| 5 | Drop 3 stale stashes in this worktree | `ws_i_baseline`, `ws_j_diff`, `ws_j_diff2` — all `pipeline.py` WIP superseded by WS-I/WS-J commits | Verified present | Eyeball `git stash show -p`, then drop all three |
| 6 | Prune detached worktrees | 10 detached worktrees and probes | Accumulating | See stale list below — **preserve `559c` artifacts first** |
| 7 | Delete merged branch `codex/rearchitecture-phase0-20260704` | Merge-plan step 3 ("keep a week or two") | Keep for now | Delete local+remote ~mid-July |

### 3. Stale / close-candidates

**PRs (all drafts predate the rearchitecture; anything touching moved files needs
rebase-or-close):**

| PR | Branch | Verdict |
| --- | --- | --- |
| #50 (draft) Promote top-10 LambdaRankIC branch and research maps | `codex/top10-lambdarank-screen-20260625` (= primary checkout branch) | **Decision needed.** Docs/research-map content, last commit 06-26. If the maps are still wanted, rebase onto main and un-draft; otherwise close. Blocks chore #1 either way. |
| #49 (draft) Preserve LambdaRankIC 1024 Colab launch artifacts | `codex/lambdarankic-1024-all-years-20260625` | The pair-cap-1024 experiment was superseded by the top10 loss matrix. Archive-value only — close, keep branch as archive. |
| #46 (draft) Filter missing S&P 500 export RICs | `codex/filter-missing-sp500-rics-20260624` | Data-export helper, dormant since 06-24. Decide keep-and-rebase vs close. |
| #34 (draft) Regime CSV no-backfill regression | `codex/regime-csv-no-backfill-coverage` | Regression-test coverage, merged main on 06-26 (pre-rearchitecture). Probably still valuable — rebase onto new layout, un-draft, merge or close deliberately. |

**Branches with no PR (close/archive candidates):** `codex/Top10nLambda` (06-21,
superseded by loss-matrix line), `codex/lambdarankic-loss-optimization` (06-19),
`codex/lambdarank-ic-colab` (06-04), `codex/lambdarankic-lower-pair-screen`
(06-20), `codex/main_2`, `codex/archive-*`, `codex/integrate-local-main-20260526`,
`codex/italy-work-snapshot-20260605`, `codex/phase-4-eval-mlops`,
`codex/pit-universe-validation`, `codex/remove-seed-42-references`,
`codex/thermo-cleanup-review`, and remote-only `Economic_layer`, `lC_loss`,
`modular_correct`, `cursor/*` (all
pre-May relics). None show activity after 06-27; all predate the rearchitecture.

**Worktrees (probed 2026-07-06):**

- Removable when their chats are done (clean, detached at old main merges):
  `19ac`, `f402` (at current main), `5a20`, `cbd5`, `757f`, `d655`,
  and the ~15 empty probe dirs (`0aed`, `1608`, `2f1c`, …).
- **`559c` — do NOT remove until artifacts are copied out.** Detached, dirty=5;
  holds `artifacts/2026-06-30-sp500-top10-loss-seed-matrix-consolidation`
  (training_rows.json etc.), the consolidation bundle every LambdaRankIC handoff
  references. Copy it into the repo or Drive first.
- `81be`/`8ab5`/`8c5d` (detached at `0d6b7c4`, LambdaRankIC full-tranche
  notebook, 1–4 dirty files each) and `6cc7`/`d641` (dirty) — eyeball the dirt,
  then remove.
- Alive/named worktrees: `602a` (lambdarankic-1024, PR #49), `334e` (PR #46),
  `7969` (ruff baseline — PR #51 merged, remainder is one CI
  smoke fix commit; check if already on main, then remove), plus the named
  `evidence-harness`, `issue8-vol-ablation-sweep`, `portfolio-ic-hybrid-testing`,
  and `pr34-regime-no-backfill` dirs — fate follows their PR/branch verdicts above.

### 4. Blocked / decision-needed

| Item | Evidence | Blocker / decision | Next action |
| --- | --- | --- | --- |
| LambdaRankIC daily diagnostics data | Coordination handoff Drive search | Current repeated-seed `161803`/`271828` top10 `averaged_predictions` not exposed by Drive search; may be Colab-VM-local only | Browse Drive `artifacts/local_run_root/training` tree directly from a Colab session; if truly absent, decide: re-run the 4 affected 2024 training rows (contradicts no-retrain stance) vs proceed with seed-`314159` folders as a weaker proxy |
| Portfolio-IC weight50 | `codex/portfolio-ic-hybrid-testing` (06-01) | Promote / park / rerun | Evidence says: keep testing as low-turnover variant, esp. 2024-like regimes; not default. Park the branch or fold future tests into the LambdaRankIC comparison harness |
| Issue #8 volatility targeting | GitHub issue #8; 3 `issue8-*` branches | Parked pending user prioritization | Leave parked; pick one canonical branch when resumed |
| PIT masked-panel issue backlog | Issues #25, #29, #30, #31, #33 (needs-triage since May) | Promotion decision for frozen default recipe + follow-up validation runs | Triage pass: #33 (promote frozen recipe?) is the decision gate; others hang off it |
| Colab reliability doc | Issue #36 (ready-for-agent) | None — just unscheduled | Good first post-merge agent task |
| LSEG access | Unresolved external access | External data access | Refresh probe only when data needed |
| Agentic-engineering process spec | Untracked in primary checkout; 06-30 Codex thread (Day_1_v3 PDF plan) | Commit, iterate, or discard | Review and decide during chore #1 |

---

## Priority Order (decisive)

1. **Chores 1+2** (sync primary checkout, commit handoffs) — minutes of work
   that make main the single continuation surface.
2. **LambdaRankIC Colab/Drive diagnostic job** — the only active research; fully
   specified in the handoffs; blocked only on running it where the artifacts live.
3. **Notebook regeneration + shim-removal tracking issue** — deadline-driven
   (~early Aug 2026).
4. **Stash drop + worktree prune** — with the `559c` artifact-preservation caveat.
5. **Draft-PR triage** (#50, #49, #46, #34) and stale-branch deletion sweep.
6. **Issue backlog triage** (#33 first as the PIT-recipe decision gate).
