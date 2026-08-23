---
name: implement-ticket
description: Implement a ticket in this repository end to end — claim, branch, TDD at agreed seams, mutation-check the guards, review, draft pull request. Use when building the work described by an issue or spec, fixing a bug against a ticket, or when a Wayfinder task ticket needs code written.
---

# Implement a ticket

Covers the upstream `implement` skill, which in several harnesses only a human
can start. Its own steps are thin — use `/tdd`, typecheck, run the suite once,
`/code-review`, commit — and it ends *"commit your work to the current branch"*,
which this repository forbids.

**The rules are `CLAUDE.md` (Hard rules, Windows and tooling) and `AGENTS.md`
(Testing, Code Style). Read them; do not reconstruct them from memory.** This
file adds only what they do not say.

## Before any code

**Claim, then branch — in that order.** Assign the ticket and declare
`**Owned paths:**` in its body *before the branch exists*. A declaration in the
pull request is after the fact and useless to a concurrent session deciding
whether to branch. If the work has no ticket, file one first. See
`docs/agents/issue-tracker.md` § *Claim Before Branching*.

Read `CONTEXT.md` before naming anything, so your names match the project's
vocabulary rather than inventing a parallel one.

## The loop

Use `/mattpocock-skills:tdd` — namespaced, because more than one enabled plugin
ships a similarly named skill and the wrong one produces differently shaped
evidence.

**Confirm the seams with the maintainer before writing any test.** The `tdd`
skill is explicit that no test is written at an unconfirmed seam. Say which
public boundary you intend to test and why, then wait. This is the step that
gets skipped, and it decides whether the tests are worth keeping.

**Red before green, one slice at a time.** Writing tests against frozen code
produces tests shaped to fit it. A guard written that way here captured state
the module had already changed at import, so it could never fail.

## Before publishing

**Mutation-check every load-bearing test**, and report the table on the pull
request. Two things that make that report honest, neither of which is obvious:

- **Verify the mutation actually changed behaviour before scoring it.** A no-op
  mutation reads as "survived" and sends you hunting a phantom gap — or as
  "caught" by an unrelated test.
- **A mutation set that catches everything first time is usually weak, not
  strong.** Test the *inversion* of a rule, not only its deletion. A guard here
  scored 5/5 against deletions and was later shown blind to every inversion.

**Run `/mattpocock-skills:code-review`** — two axes, Standards and Spec. The
Spec axis catches scope creep, which self-reporting in a pull request body does
not substitute for.

Verify CI on the **exact tip commit**, not a stale run for an earlier push.
Merging is a separate explicit action by the maintainer.
