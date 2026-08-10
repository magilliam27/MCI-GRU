# Issue Tracker: GitHub

Issues and PRDs for MCI-GRU live in GitHub Issues for
`magilliam27/MCI-GRU`. GitHub issue numbers, titles, comments, assignments,
labels, relationships, and state are the durable work tracker. Local notes,
handoffs, and research reports may support an issue, but they do not replace
the tracker unless the user explicitly requests a local-only workflow.

## GitHub Operation Routing

- Use the connected GitHub app as the primary path for supported issue and pull
  request reads and writes.
- Use local Git for worktrees, branches, commits, fetches, and pushes.
- Use standalone host-routed `gh` only for connector gaps, including
  repository-label creation and advanced Wayfinder sub-issue or dependency
  APIs.
- Do not fall back to browser automation or a new GitHub login when the
  connector and the existing host-side CLI route cover the operation.

## Windows Codex Authentication

A sandboxed `gh auth status` result is not authoritative on this machine
because the sandbox cannot reliably access the Windows GitHub keyring.

Always run `gh auth status --hostname github.com` as a standalone command
through the configured host-side rule. If that check succeeds, classify GitHub
CLI authentication as healthy even if an earlier sandboxed check reported an
invalid token.

Never initiate `gh auth login`, `gh auth logout`, or token retrieval because of
a sandbox-only failure. Login and logout require a separate approval; token
retrieval is forbidden. If the host-side status check genuinely fails, report
the exact blocker instead of attempting reauthentication automatically.

## Autonomous Engineering Operations

When the user invokes a write-oriented engineering workflow, such as
implementation, TDD, triage, specification, ticket creation, Wayfinder, or
publishing, the agent is authorized to perform the normal repository and
GitHub operations required to complete that workflow without requesting
approval for each operation.

Normal in-scope operations include:

- Creating, editing, labelling, assigning, and commenting on issues.
- Creating issue hierarchies, Wayfinder maps, child tickets, and dependency
  relationships.
- Creating missing labels using the exact configured vocabulary.
- Closing eligible non-code tickets after recording their resolution.
- Creating an isolated worktree and scoped `<harness>/<task>` branch.
- Committing only the files owned by the requested work and pushing that
  confined branch.
- Creating and updating a draft pull request.
- Posting verification evidence and progress updates to the relevant issue or
  pull request.
- Marking a draft pull request ready when its acceptance criteria and
  verification requirements are satisfied.

This authority is limited to the requested workflow and repository. An
explanation, review, status report, or diagnosis remains read-only unless the
user also requests implementation or publication.

## Claim Before Branching

**A session's first tracker write is a claim, and it precedes the branch.**

This applies to *any* work that will produce a branch, not only to Wayfinder map
children. Work that arrives without a ticket gets an issue filed first, then
assigned, then branched. One issue, assigned, before the first edit.

More than one session can run against this repository at once. They share a
single object store, so each can already read the others' branches, worktrees,
and staged files — visibility has never been the problem. The tracker is the
only surface every session reads, so a claim there is what makes concurrent work
safe.

**A claim only works if it is both made and read.** Two sessions collided on
2026-07-31 and the two halves failed differently. One session had filed a map
and four children and opened a pull request — but left every ticket unassigned,
so nothing marked them as taken. The other never looked at the open issues or
the open pull requests before branching, so it would not have seen a claim even
had one been made. They then wrote the same policy change hours apart.

So both halves are required, every time:

1. **Before branching, read the open tickets and the open pull requests.** Check
   their assignees and their owned paths.
2. **Assign the ticket to yourself before any work.** An open, unassigned ticket
   reads as unclaimed no matter how much work is already under way on it.

**Declare owned paths in the ticket body at claim time.** A bounded owned-file
set is already required of every slice, but until the pull request exists it
lives only in the author's head — which is after a collision has happened.
Publishing it at claim time lets another session check before choosing its own
files. Use a single line near the top of the body:

```
**Owned paths:** path/one.md, path/two.py. Nothing else.
```

Widening later is allowed under the existing exception rule, recorded in the
commit message.

Before starting, read the open assigned tickets and their owned paths. If a path
you need is claimed, coordinate rather than proceed.

## Branch and Pull Request Policy

- Never commit or push directly to `main` or another protected branch.
- Claim the work on the tracker first; see **Claim Before Branching** above.
- Start write-oriented work from the current remote base, normally
  `origin/main`.
- Use a scoped `<harness>/<task>` branch. The prefix records which harness
  produced the work: `claude/*` is reserved for Claude Code sessions, and all
  other work continues to use `codex/*`. The prefix is provenance only. Every
  scoped branch carries identical obligations regardless of prefix.
- When the active checkout is dirty or contains unrelated work, use an
  isolated worktree instead of modifying, stashing, cleaning, or absorbing
  that work.
- Preserve unrelated tracked and untracked files.
- Run verification appropriate to the change before publication.
- Stage and commit only the files owned by the requested workflow.
- Open a draft pull request by default and link relevant issues.
- Do not force-push or rewrite shared history unless the user explicitly
  requests it.
- Merging into `main` remains a separate explicit final action.

## Closing Issues

For implementation issues, include `Closes #<number>` in the pull request so
GitHub closes the issue when the change merges. Do not close an implementation
issue merely because a draft pull request exists.

Decision, research, Wayfinder, triage, duplicate, invalid, and rejected tickets
may close directly once their resolution is recorded in a durable comment. If
a Wayfinder child is an implementation issue, follow the implementation
closure rule instead.

Agents should autonomously add the appropriate comments, labels, links,
assignments, and closure state when the invoked workflow reaches that outcome.

## Pull Requests as a Request Surface

**PRs as a request surface: no.**

External pull requests do not enter the issue-triage queue as requests. This
does not prevent agents from creating or managing pull requests as part of an
approved engineering workflow.

GitHub issues and pull requests share one number space. Resolve an ambiguous
`#<number>` as a pull request first, then fall back to an issue.

## Skill Conventions

- When a skill says to publish to the issue tracker, create a GitHub issue.
- When a skill says to fetch the relevant ticket, read the issue body, labels,
  assignees, relationships, and comments.
- Before mutating the tracker, confirm the repository and requested workflow
  scope; do not ask again for each normal in-scope operation.

## Wayfinding operations

`wayfinder/SKILL.md` refers to this section **by this name**. Do not rename it.

Wayfinder uses one map issue and linked child issues as tickets.

**No session can invoke `/mattpocock-skills:wayfinder`.** It carries
`disable-model-invocation: true`, so only a human typing its name reaches it and
no skill can chain to it. This section is therefore not a pointer to the skill —
it is the operative substitute, and it is what an agent actually follows. Where
it and `wayfinder/SKILL.md` disagree, the `SKILL.md` wins unless the divergence
is recorded here as deliberate, with its reason.

- **Map:** Maintain a single issue labelled `wayfinder:map`. Its body tracks
  `Destination`, `Notes`, `Decisions so far`, `Not yet specified` (this
  repository calls it `Fog`), and `Out of scope`.
  - `Destination` is named **first** and fixes the scope; every session orients
    to it before choosing a ticket. A map whose destination cannot be named is
    not a map — that is the test for whether charting is premature.
  - `Out of scope` is work ruled beyond the destination. It never graduates into
    a ticket, and it exists so that beyond-scope work stops reading as takeable
    frontier.
  - `Fog` is *in-scope* uncertainty and does graduate. When a fog patch becomes
    a ticket, clear it from `Fog` so it lives in exactly one place.
  - The upstream `setup-matt-pocock-skills` seed template still describes only
    `Notes` / `Decisions-so-far` / `Fog`. That template is stale against
    `wayfinder/SKILL.md`; the skill wins.
- **Plan, don't do.** A map produces decisions, not deliverables. An effort that
  needs execution tickets may override this — but the override goes in that
  map's own `Notes`, in writing. It is a per-map exception, not the default.
- **HITL or AFK.** Every child is one or the other, and the map should say
  which. `wayfinder:research` is AFK: an agent resolves it alone.
  `wayfinder:grilling` and `wayfinder:prototype` are **HITL** and resolve only
  through live exchange with the human — **an agent must never answer its own
  grilling questions.** `wayfinder:task` is either, depending on the ticket.
- **One ticket per session.** Never resolve more than one ticket in a session,
  except research tickets. This bounds how much unreviewed change lands at once.
- **Child ticket:** Prefer a native GitHub sub-issue linked to the map and label
  it with exactly one type:
  `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or
  `wayfinder:task`. If native sub-issues are unavailable, add the child to a
  task list in the map and put `Part of #<map>` at the top of the child body.
- **Dependency:** Prefer GitHub's native issue dependency relation. Host-side
  `gh api` may create the edge using the blocker's numeric database ID, not its
  issue number or node ID. If native dependencies are unavailable, put
  `Blocked by: #<number>, #<number>` at the top of the dependent ticket.
- **Frontier:** Read the map's open children in map order, exclude tickets with
  any open blocker or any assignee, and select the first remaining ticket.
- **Claim:** Assign the selected ticket to the driving developer; this is the
  session's first tracker write. See **Claim Before Branching**, which extends
  this step to work that has no ticket yet.
- **Resolve:** Record the answer or outcome in a durable ticket comment. Close
  an eligible non-code ticket, then add a concise pointer to the map's
  `Decisions so far`. Implementation tickets remain open for merge-driven
  closure through `Closes #<number>`.
  - **The comment comes first, always.** A ticket closed with no recorded
    resolution leaves an empty record, and what it delivered becomes
    reconstructable only from someone else's prose elsewhere. This has happened.
  - If a pull request deliberately omits the closing keyword, the ticket falls
    back to the direct-closure route — which means it needs the resolution
    comment. Declining one route does not exempt you from the other.

### Deliberate divergence: map state lives in the newest comments

Recorded here rather than only in `CLAUDE.md`, so that every harness reads it.

A map's body edit raises no timeline entry, notification, or email, so a body
alone reads as frozen. This repository therefore requires **every material body
update to be paired with a short companion comment** on the map.

`wayfinder/SKILL.md` treats the body as the whole map at low resolution, loaded
once per session. That model and this convention pull in opposite directions,
and the failure has now been observed in both directions:

- A body that stops tracking its comments goes stale invisibly. One map's body
  sat eleven days behind five comments and asserted a closed ticket as the
  frontier.
- A comment that is never folded back can be wrong while the body is right. On
  another map the newest frontier-bearing comment named the wrong ticket for
  eight days, and "trust the newest comment" routed readers to it.

So the convention is **both**: comment on every material body update, *and* fold
comment-recorded state changes back into the body. Neither surface is allowed to
drift from the other.
