# Worktree And Branch Hygiene

This policy keeps Codex continuation surfaces discoverable without silently
deleting work that may still matter.

## Required Snapshot

Before starting a new branch, reviewing recent agent work, or proposing cleanup,
capture the current state from live Git refs:

```powershell
git fetch --prune origin
git worktree list --porcelain
git branch --all --verbose --no-abbrev
git status --short --branch --untracked-files=all
```

For each candidate branch or worktree, record:

- Worktree path and branch or detached HEAD.
- Ahead/behind count versus `origin/main`.
- Dirty tracked paths and untracked artifacts.
- Related PR, issue, handoff, notebook, or experiment artifact.
- Whether the branch is active, preserved, stale-candidate, or blocked.

## Classification

- **Active**: current user work, open PR, live experiment, dirty checkout, or
  explicit continuation handoff.
- **Preserved**: unique local commits or untracked artifacts were found and have
  been committed, pushed, or documented before cleanup is considered.
- **Stale candidate**: clean worktree, no unique commits versus `origin/main`,
  no active PR/issue/handoff, and no untracked artifacts.
- **Blocked**: detached HEAD, missing remote context, ambiguous provenance, or
  anything that cannot be classified from local evidence.

## Cleanup Rules

- Do not remove worktrees, delete branches, force-delete refs, close PRs, or
  archive threads without explicit user approval for that exact target.
- Do not treat untracked files as disposable. Preserve launch notebooks,
  generator scripts, handoffs, and result summaries before any cleanup proposal.
- Prefer `git branch -d` over `git branch -D`. Force deletion requires a
  separate explicit approval naming the branch.
- A stale-candidate report is enough for an automated pass. Actual deletion is a
  follow-up action after the user accepts the candidate list.

## Cockpit Reporting

Daily cockpit or thread-hygiene reviews should list stale candidates with enough
evidence for a human decision:

- Branch/worktree identifier.
- Clean or dirty state.
- Unique commit count versus `origin/main`.
- Open PR or issue status when available.
- Recommended action: keep, preserve, close/archive, or ask the user.
