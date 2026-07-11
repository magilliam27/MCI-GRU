# Docs Research Evidence Consolidation Handoff

Last updated: 2026-07-09

## Resume Here

- Start with: refresh `codex/salvage-pr50-research-docs-20260707` from current `origin/main`, then update `docs/research/README.md` and `docs/research/current/README.md` so they route every current `docs/research/current/*` file.
- Current state: `origin/main` is `16bcea9`, which includes PR #61 (`docs/handoffs/2026-06-21-loss-function-research-map.md`) and PR #63. The current checkout is detached at that exact SHA.
- Immediate next move: use PR #60 as the canonical docs/research follow-up, but keep PR #62 as a separate evidence-preservation review because it carries a compact loss/seed matrix bundle and an explicit incomplete-backtest caveat.

## Current Objective

Decide the canonical continuation surface for docs and research evidence after the 2026-07-08 cockpit packet named competing surfaces. Inventory landed, parked, and still-open evidence branches without deleting, rebasing, merging, or rewriting research conclusions.

## What Changed

- Added this handoff only: `docs/handoffs/2026-07-09-docs-research-evidence-consolidation.md`.
- No branch cleanup, archive action, merge, rebase, reset, training, Colab, data-vendor, or Drive mutation was run.

## Key Decisions

- `codex/salvage-pr50-research-docs-20260707` / PR #60 is the canonical docs/research continuation surface. It is the narrow salvage of closed-unmerged PR #50's research-document portion and adds the June 19 program/opportunity maps, the saved-prediction cost/rank-gate note, and README index changes.
- `docs/research/README.md` on current `origin/main` is incomplete as a routing surface. It indexes only the June 21 top-university scan under `docs/research/current/`, while current main also contains Issue #8 volatility-targeting reports, loss-path reports, and the SP500 PIT GICS Top-10 baseline report under the same directory.
- PR #61 is already landed and should remain a handoff, not be promoted to current research evidence by default. It preserves loss-function decision rationale and says later LambdaRankIC diagnostics supersede it for default-readiness.
- `codex/salvage-loss-seed-matrix-evidence-20260707` / PR #62 should stay separate from the README repair. It preserves an incomplete loss/seed matrix with useful Drive/source pointers and compact local summaries; do not treat it as a complete loss-family comparison until the missing transaction-cost/rank-gate backtest matrix is recovered or rebuilt.
- `codex/evidence-harness-20260627`, `codex/research-paper-skill-main`, `codex/salvage-loss-function-research-map-20260707`, and `codex/salvage-paper-trade-frozen-graph-test-20260707` are already ancestors of current `origin/main`. They can be parked as branch/worktree cleanup candidates, subject to user approval.

## Branch Disposition

| Surface | Live state | Disposition | Recommended action |
| --- | --- | --- | --- |
| `codex/evidence-harness-20260627` | Local+remote branch at `8287f42`; clean worktree; ancestor of `origin/main` by live merge-base check; no GitHub PR found for that head. | Already landed. | Park for later cleanup only; no follow-up PR needed for this surface. |
| `codex/research-paper-skill-main` | Local-only branch at `1986f2e`; ancestor of `origin/main`; no attached worktree found. | Already landed. | Park for later cleanup; main already tracks `skills/research-paper-to-mci-gru/*`. |
| `codex/salvage-loss-function-research-map-20260707` | Local+remote branch at `5ebf1dc`; clean worktree; PR #61 merged 2026-07-09; ancestor of `origin/main`. | Already landed. | Keep the landed file under `docs/handoffs/`; do not list it as research evidence unless a current report cites it as provenance. |
| `codex/salvage-paper-trade-frozen-graph-test-20260707` | Local+remote branch at `592cf53`; clean worktree; PR #63 merged 2026-07-09; ancestor of `origin/main`. | Already landed, adjacent invariant work. | No docs/research action. |
| `codex/salvage-pr50-research-docs-20260707` | Local+remote branch at `914d012`; PR #60 open draft; 5 commits behind and 1 ahead of `origin/main`; clean merge-tree into current main; worktree has one untracked cockpit decision note. | Needs follow-up PR. | Refresh/rebase or merge current main, update README routing completely, then move PR #60 out of draft when the docs-only diff is reviewed. Preserve the untracked cockpit note separately. |
| `codex/salvage-loss-seed-matrix-evidence-20260707` | Local+remote branch at `53a43e8`; PR #62 open draft; 5 behind and 1 ahead; clean worktree; clean merge-tree into current main. | Needs separate evidence-preservation review. | Keep as draft until artifact policy and incomplete-backtest caveat are accepted. Consider splitting bulky `artifacts/` from the handoff/notebook generator if review wants a lighter PR. |
| `codex/salvage-lambdarankic-recovery-20260707` | Local+remote branch at `62f3ed6`; PR #65 open draft; current-base, 1 ahead; clean merge-tree. | Outside docs/research canonical routing. | Treat as LambdaRankIC recovery/safety work, not the docs/research continuation surface. Review after PR #60/#62 or in the LambdaRankIC workstream. |
| `codex/cockpit-refresh-20260708` | Local+remote branch at `be21b1d`; PR #64 open; 5 behind and 2 ahead; contains `docs/agents/cockpit/2026-07-08.md` and `docs/agents/workstreams.md`. | Snapshot only for this decision. | If keeping the cockpit packet, refresh/regenerate after PR #61 because the packet was written before that merge. Do not use it as current main state by itself. |
| `codex/top10-lambdarank-screen-20260625` / PR #50 | PR #50 closed unmerged; branch/worktree still exists and is dirty in the primary checkout. | Parked historical/LambdaRankIC surface. | Do not resurrect PR #50 for docs/research. PR #60 is the narrower salvage path for its research docs. |

## Important Files

- `docs/research/README.md`: canonical evidence lifecycle map; currently incomplete relative to files already in `docs/research/current/`.
- `docs/research/current/README.md`: branch-local index; PR #60 improves it but should be refreshed against current main.
- `docs/handoffs/2026-06-21-loss-function-research-map.md`: landed by PR #61; useful historical loss-path bridge, not current evidence by itself.
- `docs/handoffs/2026-06-30-sp500-top10-loss-seed-matrix-consolidation.md`: proposed by PR #62; preserves the loss/seed matrix caveat and Drive/source pointers.
- `docs/agents/cockpit/2026-07-08.md`: exists on PR #64, not current main; useful as a dated cockpit snapshot only.
- `docs/agents/domain.md`: source-of-truth policy says code/tests, invariants, canonical docs, current research evidence, then handoffs/historical references.

## Verification

- `git fetch --all --prune` was run with approved access and updated live refs.
- Current checkout verified as detached `HEAD` at `16bcea9`, matching `origin/main`.
- `git log --oneline -5 origin/main` shows PR #61 merged at `16bcea9`, after PR #63.
- GitHub connector confirmed PR states: PR #60 open draft, PR #61 merged, PR #62 open draft, PR #63 merged, PR #64 open, PR #65 open draft, PR #50 closed unmerged.
- `git merge-base --is-ancestor` checks classified the named branches as landed or unmerged.
- `git merge-tree --write-tree origin/main <branch>` returned clean tree hashes for PR #60, PR #62, and PR #65, so no textual merge conflicts were found in this pass.
- Cross-worktree status checks used per-command `-c safe.directory=...`; only the PR #60 worktree had relevant dirt: untracked `docs/agents/cockpit/2026-07-07-dirty-branch-decisions.md`.
- No pytest or ruff suite was run because the only change is this docs handoff.

## Open Risks

- PR #60 improves the research README, but its current branch copy still needs a final current-main pass. Decide whether the README must enumerate every `docs/research/current/*` file or intentionally separate "active research maps" from other current evidence.
- PR #62 commits compact JSON/CSV summaries under `artifacts/`; this may be acceptable as a small evidence index, but it should be reviewed against the `docs/research/README.md` policy that bulky/raw artifacts belong in Drive or external artifact storage.
- The 2026-07-08 cockpit packet is not merged to main and was generated before PR #61 landed, so it is useful context, not current truth.
- Local branch cleanup remains unperformed by design; do not delete/archive branches or worktrees without explicit user approval.

## Next Actions

1. Refresh PR #60 onto current `origin/main` and update `docs/research/README.md` plus `docs/research/current/README.md` to route all current evidence files explicitly.
2. Decide whether PR #62 should merge as-is, split handoff/notebook from `artifacts/`, or stay parked until the missing backtest rows are recovered.
3. After PR #60 lands, regenerate or refresh the cockpit packet so the Docs and research evidence row no longer points at already-landed surfaces as blockers.
4. Review PR #65 under the LambdaRankIC recovery workstream, not as part of docs/research evidence consolidation.
5. Only after the above, ask for explicit approval before deleting stale branches or removing worktrees.

## Commands Run

- `git status --short --branch`
- `git fetch --all --prune`
- `git rev-parse --abbrev-ref HEAD`
- `git rev-parse --short HEAD`
- `git rev-parse --short origin/main`
- `git log --oneline -5 origin/main`
- `git remote get-url origin`
- `git worktree list --porcelain`
- `git branch --all --format=...`
- `git branch --all --no-merged origin/main --format=...`
- `git show origin/codex/cockpit-refresh-20260708:docs/agents/cockpit/2026-07-08.md`
- `git diff --name-status origin/main...<branch>`
- `git diff --stat origin/main...<branch>`
- `git merge-tree --write-tree origin/main <branch>`
- `git -c safe.directory=<worktree> -C <worktree> status --short --branch`
- GitHub connector: recent PR list, PR #50 metadata, and PR search for `codex/evidence-harness-20260627`.

## Do Not Do

- Do not delete, archive, reset, rebase, merge, or close branches without explicit user approval.
- Do not treat handoffs as research evidence unless a current research report cites them as provenance.
- Do not treat the visible PR #62 backtest rows as a complete loss-family comparison.
- Do not use the 2026-07-08 cockpit packet as current main state without accounting for PR #61 and later main merges.

## References

- `AGENTS.md`
- `docs/agents/domain.md`
- `docs/agents/workstreams.md`
- `docs/agents/cockpit/2026-07-08.md` from `origin/codex/cockpit-refresh-20260708`
- `docs/research/README.md`
- `docs/research/current/README.md`
- `docs/handoffs/2026-07-06-rearchitecture-merge-plan.md`
- GitHub PRs #50, #60, #61, #62, #63, #64, and #65
