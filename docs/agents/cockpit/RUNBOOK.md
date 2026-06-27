# MCI-GRU Cockpit Runbook

The cockpit refresh writes the repo-local operating picture:

- `docs/agents/workstreams.md`
- `docs/agents/cockpit/YYYY-MM-DD.md`

Manual local refresh:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20
```

GitHub sync is disabled by default. Live sync requires `--github-sync`:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20 --github-sync
```

Before running live sync, confirm GitHub CLI authentication:

```bash
gh auth status
```

Live sync uses a dated branch named `codex/cockpit-refresh-YYYYMMDD`, commits
only cockpit files, pushes the branch, opens or reuses `Cockpit refresh:
YYYY-MM-DD`, creates or reuses the `MCI-GRU Cockpit` issue, and comments there
with the run color, PR link, and decision queue.

Labels are applied only when they already exist in GitHub. Missing labels are
reported as skipped actions instead of being created with near-duplicate names.
Safe labels include `cockpit-reviewed`, `ready-for-agent`, `needs-info`, and
`needs-triage` when present.

Issue closure must go through `close_issue_with_evidence`, which comments with
the closure evidence before closing the issue. Do not close ambiguous research,
experiment, or data-access issues without clear evidence such as a merged PR,
existing code/tests/docs, a linked duplicate, or an explicit user decision.

Before committing cockpit output, inspect:

```bash
git status --short
git diff -- docs/agents/workstreams.md docs/agents/cockpit/
```

Only cockpit files should be staged for a cockpit refresh.

## Daily Branch And PR Integration Review

Every cockpit refresh must include a branch-integration pass before writing the
workstream register.

Read-only evidence to collect:

```bash
git status --short
git branch --show-current
git worktree list --porcelain
git branch --all --no-merged origin/main
gh pr list --state all --json number,title,state,isDraft,headRefName,baseRefName,mergedAt,updatedAt,url
gh issue view 38 --json number,title,state,updatedAt,url,comments
```

For each durable workstream, the cockpit must name the canonical continuation
surface in `docs/agents/workstreams.md`. Do not use generic placeholders such as
"See latest branch, worktree, issue, or handoff evidence" when a branch,
worktree, PR, issue, handoff, or merged commit is known.

The canonical continuation surface should be one of:

- a GitHub issue when implementation work should start from the tracker;
- an open PR when review or merge is the next action;
- a named branch or worktree when local continuation is expected;
- `main` plus a merged PR or commit when the work is already integrated;
- a dated cockpit PR when the row is cockpit-status-only;
- `needs-user-decision` when two or more plausible continuation surfaces exist.

If the cockpit cannot identify one canonical surface, mark the row
`needs-user-decision` and put the competing surfaces in `Blocked On` or
`Next Action`.

## Stale Cockpit PR Handling

Dated cockpit PRs are audit records, but only the newest relevant cockpit PR
should remain the active review surface for the day.

During daily review:

1. List open `Cockpit refresh: YYYY-MM-DD` PRs.
2. Compare each open dated cockpit PR with newer merged cockpit PRs.
3. If a newer cockpit PR has merged and covers the same generated files, mark
   the older open PR as `stale` or `superseded` in the cockpit packet.
4. Do not close the superseded PR automatically unless the user has approved PR
   cleanup for that run.
5. Do not delete the dated branch automatically.
6. Add the proposed action to the GitHub Cockpit issue comment when live sync is
   enabled.

A superseded cockpit PR should be described like this:

```markdown
- PR #41 (`codex/cockpit-refresh-20260622`) is still open, but PR #42
  (`codex/cockpit-refresh-20260623`) has merged newer cockpit output. Proposed
  action: close PR #41 as superseded after user approval; keep the branch unless
  separately approved for deletion.
```

## Integration Branch Selection

The cockpit must keep branch integration separate from cockpit refreshes.

Rules:

- Cockpit branches may update only cockpit/status files and small cockpit policy
  docs under `docs/agents/`.
- Active research, experiment, Colab, paper-trade, or pipeline work must name
  its own canonical branch/worktree before continuation.
- If a workstream has multiple plausible branches, do not continue on an
  arbitrary one. Mark the workstream `needs-user-decision` and list the branch
  choices.
- If a branch has already been merged to `main`, the canonical continuation
  surface becomes `main` plus the merged PR/commit, unless follow-up work is
  explicitly opened elsewhere.
- If a branch/worktree is detached, local-only, or stale, say so directly in the
  `Branch / Worktree` column.

Branch integration decisions that require user approval:

- closing a superseded PR;
- deleting or pruning a branch/worktree;
- force-updating any branch;
- choosing between competing non-cockpit implementation branches;
- moving code from a local-only worktree into an integration branch.

## GitHub Hygiene

Each cockpit run should include a short GitHub hygiene section when any branch
or PR state needs attention.

Check for:

- the latest merged PR to `main`;
- open cockpit PRs that appear superseded by newer merged cockpit output;
- draft PRs older than 14 days, especially when dirty against `main`;
- pushed branches with no open PR, grouped by workstream when possible;
- active or `ready-for-agent` issues that lack a named branch, open PR, or
  explicit "not started" state.

If live sync goes sideways, inspect and recover with:

```bash
git status --short
git branch --show-current
gh pr list --head codex/cockpit-refresh-YYYYMMDD
gh issue list --search "MCI-GRU Cockpit in:title"
```
