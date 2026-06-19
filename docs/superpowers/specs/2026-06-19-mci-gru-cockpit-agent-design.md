# MCI-GRU Cockpit Agent Design

Date: 2026-06-19
Status: Approved design

## Purpose

MCI-GRU needs a daily project cockpit that turns branches, worktrees, GitHub
issues, handoffs, research notes, run artifacts, and recent agent activity into
one reviewable operating picture.

The cockpit is meant to compensate for a fast-moving solo workflow. It should
make the project easier to supervise without requiring the user to become a
full-time project manager.

## Goals

- Maintain a repo-local, GitHub-backed view of current workstreams.
- Refresh the cockpit daily and on demand.
- Use dated cockpit branches and PRs so every refresh has an audit trail.
- Let the agent perform low-risk project-management actions automatically.
- Keep strategic, expensive, or ambiguous decisions in a short user decision
  queue.
- Compress many chats and handoffs into a few durable workstreams.

## Non-Goals

- Replace GitHub Issues with an external tracker.
- Push non-cockpit code changes automatically.
- Launch Colab, training, or data-vendor jobs automatically.
- Delete branches, worktrees, artifacts, or reports automatically.
- Treat handoffs or scratch artifacts as stronger sources of truth than code,
  tests, canonical docs, or current research evidence.

## Core Surfaces

### Workstream Register

Path: `docs/agents/workstreams.md`

This is the current-state cockpit table. It is the first file an agent should
open when asking "what is active right now?"

The table has one row per durable workstream, not one row per chat.

Columns:

| Column | Meaning |
| --- | --- |
| Workstream | Durable topic, such as LambdaRankIC, Portfolio-IC, issue #8, Colab operations, regime CSV, LSEG access, docs/research, git hygiene, or daily scans. |
| Status | One of the controlled statuses below. |
| GitHub Issue / PR | Durable tracker links when present. |
| Branch / Worktree | Canonical continuation surface, including local-only or detached state when relevant. |
| Source Of Truth | The strongest current evidence source for the row. |
| Latest Artifact | Handoff, report, notebook, Drive artifact, CSV, PR, or branch evidence. |
| Last Verification | Latest meaningful command, artifact check, or explicit no-op evidence. |
| Blocked On | Missing approval, auth, runtime, data, artifact, or strategic decision. |
| Next Action | The next concrete action for an agent or user. |
| Owner | User, Codex, GitHub issue assignee, or unassigned. |
| Last Reviewed | Date of last cockpit review. |

Controlled statuses:

- `active`: branch, issue, or artifact has a clear next step.
- `blocked`: work cannot continue without missing data, auth, runtime, approval,
  or artifact evidence.
- `parked`: intentionally paused but still valid.
- `local-only`: useful local work exists but is not pushed or PR-ready.
- `ready-for-agent`: next action is clear enough for a future Codex thread.
- `needs-user-decision`: multiple valid paths exist and the user should choose.
- `done`: workstream has closure evidence.
- `archive`: preserved for provenance, not active work.
- `stale`: old surface still exists but is probably superseded or duplicated.

### Daily Review Packet

Path pattern: `docs/agents/cockpit/YYYY-MM-DD.md`

Each cockpit run writes a dated narrative packet with:

- executive summary
- run color: `green`, `yellow`, or `red`
- decision queue
- active workstreams
- blocked workstreams
- local-only work
- stale or archive candidates
- GitHub actions taken
- GitHub actions proposed or skipped
- verification notes
- evidence gaps and contradictions

Run colors:

- `green`: no material changes and no new user decisions.
- `yellow`: status changed or one user decision is needed.
- `red`: blocked work, failed sync, stale branch risk, failed GitHub action, or
  conflicting evidence.

Green days should stay quiet. Yellow and red days should surface the decision
queue clearly.

### GitHub Cockpit Issue

The project should have one pinned or easy-to-find GitHub issue named
`MCI-GRU Cockpit`.

The cockpit runner comments there after each run:

- green run: one short line with the dated PR link
- yellow or red run: PR link plus the decision queue

This issue is the user-facing review surface. The repo files remain the
agent-facing source of cockpit state.

### Dated Cockpit Branch And PR

Each run uses a dated branch:

`codex/cockpit-refresh-YYYYMMDD`

Example:

`codex/cockpit-refresh-20260619`

Each run opens or updates a dated PR:

`Cockpit refresh: YYYY-MM-DD`

The branch should contain only cockpit/status files:

- `docs/agents/workstreams.md`
- `docs/agents/cockpit/YYYY-MM-DD.md`
- small cockpit policy or runbook updates under `docs/agents/`, when needed

The cockpit must never push directly to an active research, experiment, or code
branch.

## Authority Model

The cockpit agent has guarded autonomy.

### Allowed Automatically

- Read git status, branches, worktrees, logs, docs, handoffs, research maps, and
  GitHub issues or PRs.
- Create a clear evidence-backed GitHub issue when untracked work has a concrete
  next action.
- Comment on existing GitHub issues with cockpit status updates.
- Apply low-risk labels such as `needs-info`, `blocked`, `ready-for-agent`,
  `stale`, or `cockpit-reviewed` when the evidence is clear and the labels
  exist.
- Close a GitHub issue when closure evidence is strong.
- Update cockpit-only files.
- Commit and push cockpit-only changes on the dated cockpit branch.
- Open a dated cockpit PR.
- Comment on the GitHub Cockpit issue with the daily summary and PR link.

### Strong Evidence Required For Issue Closure

The cockpit may close an issue only when at least one of these is true:

- A merged PR explicitly resolves it.
- Code, tests, or canonical docs show the requested work already exists.
- The issue is a duplicate and the canonical issue is linked.
- The user previously decided to park, close, or reject it.

Every automatic closure must leave a comment explaining the evidence.

### Requires User Approval

- Push non-cockpit code changes.
- Delete, prune, or archive branches or worktrees.
- Start Colab, training, paper-trade, data-vendor, or other expensive jobs.
- Close ambiguous research or experiment issues.
- Mark an experiment as abandoned when evidence is incomplete.
- Change canonical non-cockpit docs or code.
- Introduce a new external tracker or workflow dependency.

## Runner Algorithm

### 1. Preflight

- Record the run date.
- Record current branch, dirty state, remotes, and upstream status.
- List worktrees and branches.
- Refuse to overwrite unrelated user changes.
- Create or switch to `codex/cockpit-refresh-YYYYMMDD`.
- Confirm the branch will only stage cockpit files.

### 2. Collect Evidence

Read:

- `AGENTS.md`
- `docs/agents/*`
- existing `docs/agents/workstreams.md`, when present
- recent files under `docs/handoffs/`
- `docs/research/README.md`
- current research evidence listed from the research map
- relevant workflow docs under `docs/workflows/`
- recent git commits, branches, and worktrees
- GitHub issues and PRs
- recent memory summaries, when available to the running agent

The runner should prefer current code, tests, repo invariants, canonical docs,
current research evidence, and then handoffs, in that order.

### 3. Normalize Evidence Into Workstreams

- Group by durable topic, not by chat or artifact.
- Merge duplicate surfaces such as repeated no-op daily scans.
- Distinguish active continuation surfaces from archive/provenance surfaces.
- Mark local-only commits or branches explicitly.
- Flag contradictions such as "verified locally but not pushed" or "handoff says
  G4 required but run used T4."
- Preserve source links and evidence handles.

### 4. Decide Safe Actions

- Create issues for clear untracked work.
- Comment or label existing issues when status changes are obvious.
- Close issues only with strong closure evidence.
- Queue ambiguous decisions for the user.
- Record skipped actions and the reason they were skipped.

### 5. Write Local Artifacts

- Update `docs/agents/workstreams.md`.
- Write `docs/agents/cockpit/YYYY-MM-DD.md`.
- Include all automatic GitHub actions taken.
- Include all GitHub actions requiring approval.
- Include verification status and any failed syncs.

### 6. Commit, Push, And PR

- Stage only cockpit files.
- Commit with `Refresh cockpit status for YYYY-MM-DD`.
- Push `codex/cockpit-refresh-YYYYMMDD`.
- Open or update `Cockpit refresh: YYYY-MM-DD`.
- Comment on the GitHub Cockpit issue with the run color, PR link, and decision
  queue.

### 7. Final User Report

The runner's final chat response should include:

- what changed
- what it did automatically
- what it skipped
- what needs user decision
- exact file, PR, branch, and issue references

## Daily Cadence

The cockpit should run daily by default and manually on demand.

Daily runs are valuable because MCI-GRU is a fast-moving solo project. The
runner must still control noise:

- Green runs keep the GitHub comment short.
- Yellow and red runs show the decision queue.
- Repeated no-op evidence collapses into one row.
- Stale detached worktrees are not treated as active unless new evidence appears.
- Already-covered contract checks are not reopened without a distinct regression
  gap.

## Failure Behavior

- If GitHub auth fails, write local artifacts and record `GitHub sync skipped`.
- If the worktree is dirty, stage only cockpit files and report unrelated dirty
  files.
- If a dated branch already exists, update it only if it is a cockpit branch.
- If evidence conflicts, mark the row `needs-user-decision` or `blocked` rather
  than guessing.
- If a GitHub label is missing, report the missing label rather than inventing a
  near-duplicate.

## Initial Workstreams To Seed

The first cockpit refresh should seed rows for:

- LambdaRankIC
- Portfolio-IC
- Issue #8 volatility targeting
- Colab operations
- Regime CSV contract
- LSEG access
- Daily bug scans
- Docs and research evidence
- Git and worktree hygiene

## Acceptance Criteria

- A user can open `docs/agents/workstreams.md` and understand current project
  state in under five minutes.
- A user can open the latest `docs/agents/cockpit/YYYY-MM-DD.md` and see the
  decision queue without reading handoffs.
- A future Codex thread can resume a workstream from the cockpit table without
  re-reading every recent chat.
- The cockpit can run daily without creating noisy GitHub comments on green
  days.
- All automatic GitHub actions are explainable from cited evidence.
- No non-cockpit code or docs are pushed by the cockpit without user approval.

## Implementation Notes

The first implementation should be conservative:

- Start with a runner script or documented agent command that writes local
  artifacts.
- Add GitHub issue and PR mutation after the local artifact format is stable.
- Keep the first run human-reviewed even if the authority model allows guarded
  autonomy.
- Prefer simple Markdown over a database until the workflow proves it needs more
  structure.
