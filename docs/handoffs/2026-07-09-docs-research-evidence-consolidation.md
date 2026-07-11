# Docs Research Evidence Consolidation Handoff

Last updated: 2026-07-11

## Resume Here

- PR #60 (`codex/salvage-pr50-research-docs-20260707`) is the approved merge
  surface for the docs/research routing repair. After it merges, current
  `origin/main` is the canonical continuation surface; do not reopen or re-mine
  the old PR #50 branch for these documents.
- The durable payload is the June 19 program map, the broader June 19 research
  opportunity scan, and complete routing in both research README files.
- PR #62 remains a separate evidence-preservation review. Its incomplete
  loss/seed matrix and artifact policy are not part of PR #60.

## Outcome

PR #60 consolidates the docs/research work without reviving the broad, closed
PR #50 surface:

- `MCI_GRU_PROGRAM_MAP_2026-06-19.md` is retained as the structural companion
  map for the codebase.
- `MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md` is retained as broader
  source-lead history. The June 21 top-university scan remains the primary
  implementation-priority map.
- `docs/research/README.md` and `docs/research/current/README.md` explicitly
  route every Markdown report currently under `docs/research/current/`.
- The evidence lifecycle now states that synthetic fixtures, example values,
  and pytest/temp outputs are not eligible for promotion to current research.

## Merge Review Correction

The proposed
`PIT_SAVED_PREDICTION_COST_RANK_GATE_REPRODUCIBILITY.md` was removed during
merge review. It was not a real backtest artifact:

- Its only 2022 row exactly matched the synthetic dictionaries in
  `test_write_summary_outputs_compares_cost_aware_rows_to_reviewed_artifact`.
- The report's `0.12` cost-aware return, `0.18` reviewed return, and all other
  values came from that unit-test fixture.
- Its command used the placeholder run root `RUN`, and the report carried no
  real run ID or artifact provenance.

The unit test still validates report formatting, but its generated temp output
must remain test evidence, not empirical research evidence.

## Durable Decisions

- After PR #60 merges, use `origin/main` plus the two research README files as
  the canonical routing surface. A stale branch or handoff is not grounds to
  reopen this consolidation decision.
- New files under `docs/research/current/` must be added to both indexes in the
  same PR and must identify real source, run, or decision provenance.
- Synthetic test reports may be kept only as clearly labeled fixtures or
  examples outside the current-evidence set.
- PR #62 must be reviewed independently and must not be described as a complete
  loss-family comparison unless its missing backtest evidence is recovered.
- Branch and worktree cleanup remains approval-gated and outside this merge.

## Surface Disposition

| Surface | Disposition | Next action |
| --- | --- | --- |
| PR #60 / `codex/salvage-pr50-research-docs-20260707` | Canonical docs/research merge surface. | Merge after amended CI and merge-tree verification; then treat current main as canonical. |
| PR #62 / `codex/salvage-loss-seed-matrix-evidence-20260707` | Separate, incomplete evidence-preservation review. | Keep separate until artifact policy and missing-backtest caveat are accepted. |
| PR #50 / `codex/top10-lambdarank-screen-20260625` | Closed historical surface. | Do not reopen for docs/research consolidation. |
| Already merged docs/research branches | Landed history. | Park for later cleanup only with explicit approval. |

## Verification

- Pre-review `origin/main`: `f7cc972` (includes merged PR #70).
- Pre-correction PR #60 head: `724a1d8`.
- Provenance comparison:
  `tests/test_pit_saved_prediction_backtests.py` against the removed report.
- Focused generator test: `1 passed`; it recreated the synthetic comparison from
  the fixture without using real backtest data.
- `python scripts/check_docs_sot.py`, `ruff check .`, `ruff format --check .`,
  and `git diff --check` passed.
- Complete-index comparison passed: all 11 current Markdown reports are routed
  in both research README files.
- `git merge-tree --write-tree origin/main <reviewed-head>` completed without
  conflicts against current main.
- No training, backtest, Colab, data-vendor, or Drive operation is required for
  this docs-only merge.

## Remaining Risk

- The June 19 opportunity scan is intentionally a broad source-lead map. Its own
  header requires source re-verification before opening implementation tickets;
  the June 21 scan remains the preferred priority source.
- PR #62 still needs its own provenance and artifact-policy decision.
- No branch or worktree cleanup was performed.

## Next Actions

1. Merge PR #60 only after the amended head is green and conflict-free.
2. Let the next cockpit refresh consume the merged registry and README state;
   do not create another competing docs/research continuation branch.
3. Review PR #62 separately as an incomplete evidence-preservation decision.
4. Require explicit approval before deleting or removing parked branches and
   worktrees.

## Do Not Do

- Do not promote pytest temp output, synthetic fixtures, or example metrics as
  current research evidence.
- Do not reopen PR #50 to recover these docs.
- Do not merge PR #62 as an implicit part of this README repair.
- Do not run training, new backtests, Colab, data vendors, or Drive mutations
  for this consolidation.
- Do not delete or reset branches or worktrees without explicit approval.

## References

- `AGENTS.md`
- `docs/agents/domain.md`
- `docs/agents/cockpit/workstream-decisions.json`
- `docs/research/README.md`
- `docs/research/current/README.md`
- `tests/test_pit_saved_prediction_backtests.py`
- GitHub PRs #50, #60, and #62
