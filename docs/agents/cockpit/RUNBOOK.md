# MCI-GRU Cockpit Runbook

The cockpit refresh writes the repo-local operating picture:

- `docs/agents/workstreams.md`
- `docs/agents/cockpit/YYYY-MM-DD.md`

It reads durable human decisions from:

- `docs/agents/cockpit/workstream-decisions.json`

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
git status --short --branch
git branch --show-current
git worktree list --porcelain
git branch --all --no-merged origin/main
git rev-list --left-right --count origin/main...HEAD
gh pr list --state all --json number,title,state,isDraft,headRefName,baseRefName,mergedAt,updatedAt,url
gh issue view 38 --json number,title,state,updatedAt,url,comments
```

On Windows/Codex worktrees, per-worktree status checks should use a local
safe-directory override rather than changing global git config:

```bash
git -c safe.directory=<worktree-path> -C <worktree-path> status --porcelain=v1 -b --untracked-files=all
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

## Workstream Decision Registry

`workstream-decisions.json` is the canonical, versioned input for reviewed
workstream and git-surface decisions. Generated files such as
`docs/agents/workstreams.md` must not be edited to preserve a decision; the next
refresh overwrites them.

Each `workstreams` entry records the chosen status, canonical continuation
surface, rationale, next action, and review date. Each `surfaces` entry records
which workstreams reviewed a normalized branch name and whether that surface is
`canonical`, `parked`, `archive`, or `stale`.

Resolution rules:

1. A registry decision wins over branch-name heuristics for every reviewed
   surface.
2. Reviewed historical surfaces remain visible as parked, stale, or archive
   candidates, but they do not reopen the workstream decision.
3. A newly matching branch that is absent from the registry reopens only the
   affected workstream as `needs-user-decision`.
4. Explicit `surfaces[*].workstreams` assignments classify branches whose names
   do not match the seeded branch terms.
5. Missing registry files make the cockpit red through the required-doc check.
   Invalid JSON, unknown fields, unknown workstreams, remote-prefixed surface
   keys, and unsupported format versions fail fast.

Surface keys must use the normalized branch name emitted by the cockpit, for
example `codex/example`, not `origin/codex/example`,
`remotes/origin/codex/example`, or `refs/heads/codex/example`.

When a decision changes, update the existing entry and `last_reviewed`. When a
new branch is reviewed, add it to `surfaces`; do not broaden branch-name terms
just to suppress a decision. Branch deletion, worktree removal, PR closure, and
force-pushes remain separately approval-gated even when a surface is marked
`archive` or `stale`.

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

## Git Tree Impact

Each daily packet must include a `Git Tree Impact` section that summarizes:

- current branch;
- snapshot timing, especially whether the packet was collected before GitHub
  sync commits/pushes;
- `origin/main...HEAD` divergence;
- count and names of branches not merged into `origin/main`, labelled as
  local, remote-only, or local+remote;
- total, detached, and dirty worktrees;
- dirty or detached worktree paths that require review.

Primary repo git commands should run with `safe.directory` set for the current
checkout. Secondary worktree status probes should set `safe.directory` for the
worktree path being inspected.

The workstream register should be live-topology-first. Seeded workstream names
may enrich matching branches/worktrees, but unmatched local branches, worktree
branches, and remote-only unmerged branches should get explicit `Git surface:`
rows so they cannot disappear from the solo-dev continuation map.

Use `green` only when no branch/worktree attention items are found. Use
`yellow` when there is normal solo-dev branch pressure, such as unmerged
branches, detached worktrees, or non-current dirty worktrees. Use `red` when
the current checkout itself is dirty before cockpit generation or required
docs are missing.

If live sync goes sideways, inspect and recover with:

```bash
git status --short
git branch --show-current
gh pr list --head codex/cockpit-refresh-YYYYMMDD
gh issue list --search "MCI-GRU Cockpit in:title"
```
