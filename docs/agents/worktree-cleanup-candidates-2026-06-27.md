# Worktree Cleanup Candidates - 2026-06-27

This is a read-only cleanup report. No worktrees, branches, PRs, or files were
deleted while preparing it.

Evidence collected:

- `git fetch --prune origin`
- `git worktree list --porcelain`
- `git status --short --branch --untracked-files=all`
- `git branch -vv --all`
- `git branch --merged origin/main`
- `gh pr list --repo magilliam27/MCI-GRU --state all --limit 100`

## Published Or Preserve-First Surfaces

| Surface | State | Evidence | Recommendation |
| --- | --- | --- | --- |
| `codex/cockpit-hygiene-20260626` | Clean branch, `0 1` versus `origin/main` before publish | Commit `bd4dc46` fixes cockpit GitHub-sync packet reporting and adds worktree hygiene policy. | Push and open a draft PR to `main`. |
| `codex/lambdarankic-1024-all-years-20260625` | Clean branch, no upstream before publish, `0 1` versus `origin/main` | Commit `eb6245e` preserves LambdaRankIC 1024 launch notebooks, generators, and handoff. | Push and open a draft PR or keep as a preserved branch. |

## Active Or Keep By Default

| Surface | State | Evidence | Recommendation |
| --- | --- | --- | --- |
| `C:/Users/magil/MCI-GRU` on `codex/top10-lambdarank-screen-20260625` | Dirty active checkout | Modified `AGENTS.md`; untracked research docs; untracked pytest review summary. Upstream is `origin/codex/top10-lambdarank-screen-20260625`. | Keep active. Decide separately whether to preserve or discard each dirty file. |
| `codex/filter-missing-sp500-rics-20260624` | Clean worktree at `C:/Users/magil/.codex/worktrees/334e/MCI-GRU` | Open draft PR #46. | Keep until PR #46 is merged, closed, or explicitly abandoned. |
| `codex/issue8-vol-ablation-sweep` | Clean worktree | Remote branch exists; related to Issue #8 volatility ablation work. | Keep unless the Issue #8 line is intentionally parked or superseded. |
| `codex/portfolio-ic-hybrid-testing` | Clean worktree | Remote branch exists; Portfolio-IC has been a needs-user-decision workstream. | Keep until the Portfolio-IC path is promoted, parked, or explicitly abandoned. |
| `codex/lambdarank-ic-colab` | Clean nested worktree | Remote branch exists; LambdaRankIC Colab comparison branch. | Keep until LambdaRankIC branch consolidation is decided. |
| `codex/cockpit-refresh-20260626` | Remote branch with open PR #47 | Latest cockpit refresh PR is still open. | Keep or merge/close as part of cockpit PR cleanup. |

## Stale Candidates Pending Approval

These look safe to remove only after explicit approval naming the target.

| Surface | State | Evidence | Suggested action after approval |
| --- | --- | --- | --- |
| `C:/Users/magil/.codex/worktrees/241a/MCI-GRU` | Clean detached HEAD | At `49dcf61`, same as `origin/main`; `0 0` versus `origin/main`. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/963d/MCI-GRU` | Clean detached HEAD | At `49dcf61`, same as `origin/main`; `0 0` versus `origin/main`. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/4adb/MCI-GRU` | Clean detached HEAD | At merged PR #40 commit `c02d777`; no unique commits versus current main. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/6182/MCI-GRU` | Clean detached HEAD | At merged PR #40 commit `c02d777`; no unique commits versus current main. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/6a11/MCI-GRU` | Clean detached HEAD | At merged PR #40 commit `c02d777`; no unique commits versus current main. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/75c4/MCI-GRU` | Clean detached HEAD | At merged PR #40 commit `c02d777`; no unique commits versus current main. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/7760/MCI-GRU` | Clean detached HEAD | At merged PR #40 commit `c02d777`; no unique commits versus current main. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/8717/MCI-GRU` | Clean detached HEAD | At merged PR #40 commit `c02d777`; no unique commits versus current main. | Remove worktree. |
| `C:/Users/magil/.codex/worktrees/9e8c/MCI-GRU` and branch `codex/github-pipeline-policy-20260624` | Clean branch worktree | At `49dcf61`, same as `origin/main`; upstream is `origin/main`. | Remove worktree, then delete local branch if no longer checked out. |
| Local branch `codex/cockpit-pr-reuse-fix-20260620` | Merged local branch with gone upstream | Included in `git branch --merged origin/main`; PR #39 merged. | Delete local branch. |
| Local branch `codex/deprecate-regime-csv` | Merged local branch with gone upstream | Included in `git branch -vv` as gone; PR #21 merged. | Delete local branch. |

## Blocked Or Needs Keep/Toss Decision

These contain dirty files, untracked artifacts, open PRs, or ambiguous
provenance. They should not be removed without a specific keep/toss decision.

| Surface | State | Evidence | Decision needed |
| --- | --- | --- | --- |
| `C:/Users/magil/MCI-GRU` dirty files | Active top-10 checkout | `AGENTS.md` adds the origin-main branch rule; untracked `docs/agents/agentic-engineering-process-plan.md`; untracked `MCI_GRU_PROGRAM_MAP_2026-06-19.md`; untracked `MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md`; untracked pytest review cost-rank-gate summary. | Preserve these on a branch, move them to a handoff/report branch, or discard the scratch output. |
| `C:/Users/magil/.codex/worktrees/81be/MCI-GRU` | Dirty detached HEAD | Untracked `docs/handoffs/2026-06-21-loss-function-research-map.md`; detached at `0d6b7c4`. | Preserve the handoff, or discard the worktree. |
| `C:/Users/magil/.codex/worktrees/8ab5/MCI-GRU` | Dirty detached HEAD | Untracked `docs/handoffs/2026-06-21-lambdarankic-pair-cap-investigation.md`; detached at `0d6b7c4`. | Preserve the handoff, or discard the worktree. |
| `C:/Users/magil/.codex/worktrees/8c5d/MCI-GRU` | Dirty detached HEAD | Untracked research docs and `scripts/data/export_sp500_gics_top10_mcap.py`; detached at `0d6b7c4`. | Preserve the research/export artifacts, or discard the worktree. |
| `C:/Users/magil/.codex/worktrees/d641/MCI-GRU` | Dirty detached HEAD | Modified `tests/test_training_efficiency_config.py`; detached at `5a80246`. | Inspect and preserve or discard the test change. |
| PR #41 `codex/cockpit-refresh-20260622` | Open PR | Superseded by later cockpit refreshes and current cleanup branch. | Close PR and delete branch, or keep open for a reason. |
| PR #45 `codex/cockpit-refresh-20260624` | Open PR | Superseded by later cockpit refreshes and current cleanup branch. | Close PR and delete branch, or keep open for a reason. |
| PR #47 `codex/cockpit-refresh-20260626` | Open PR | Latest cockpit refresh PR. | Merge/keep/close after comparing against `codex/cockpit-hygiene-20260626`. |
| PR #34 `codex/regime-csv-no-backfill-coverage` | Open draft PR, no local worktree in this inventory | Older draft branch remains open. | Keep as active regression coverage, or close/archive. |

## Recommended Next Decision Batch

1. Approve deletion of the stale candidates listed above, or name any to keep.
2. Choose what to do with the active top-10 dirty files.
3. Choose whether to preserve or discard dirty detached worktrees `81be`, `8ab5`,
   `8c5d`, and `d641`.
4. Choose cockpit PR cleanup for #41, #45, #47.
5. Choose whether PR #34 still belongs in the active queue.
