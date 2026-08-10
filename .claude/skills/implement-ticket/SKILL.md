---
name: implement-ticket
description: Implement a ticket in this repository end to end — claim, branch, TDD at agreed seams, mutation-check the guards, review, draft pull request. Use when building the work described by an issue or spec, fixing a bug against a ticket, or when a Wayfinder task ticket needs code written.
---

# Implement a ticket

Covers `implement/SKILL.md`, which is user-invoked and so unreachable by an
agent, and replaces its generic steps with this repository's stricter ones. The
upstream version ends *"Commit your work to the current branch"*; here that
would violate claim-before-branching and the draft-pull-request rule.

**If the maintainer has typed `/mattpocock-skills:implement`, prefer it** and
apply the repository rules below on top.

## Before any code

**Claim first.** Assign the ticket, and declare `**Owned paths:**` in its body
*before the branch exists*. A declaration in the pull request is after the fact
and useless to a concurrent session deciding whether to branch. If the work has
no ticket, file one first.

**Then branch.** Worktree off *then-live* `origin/main` under
`C:\Users\magil\.claude\worktrees\`, named for the work. Scoped `claude/*`
branch. Never work on `main`.

**Read before editing.** `AGENTS.md` for the invariants, `docs/agents/guide.md`
to find the owning module and its adjacent contracts, and `CONTEXT.md` so your
names match the project's vocabulary.

## The loop

Use `/tdd`. It is model-invoked, so reach for it.

**Confirm the seams with the maintainer before writing any test.** The `tdd`
skill is explicit: *"No test is written at an unconfirmed seam."* Say which
public boundary you intend to test and why, and wait. This step gets skipped
constantly and it is the one that decides whether the tests are worth keeping.

**Red before green, one slice at a time.** Failing test, then only enough code
to pass it. Not all the implementation followed by all the tests — writing tests
against frozen code produces tests shaped to fit it. A guard written that way in
this repository turned out to be vacuous: it captured state that the module had
already changed at import, so it could never fail.

Run focused tests often; the full suite once at the end:

```
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ --junitxml=test_reports\junit.xml -q
```

Always the isolated launcher, never bare pytest. It also pins `PYTHONPATH` to the
repository root — an ad-hoc script does not, and the venv's editable install
resolves `mci_gru` to the protected checkout instead of your worktree.

## Before publishing

**Mutation-check every load-bearing test.** Break the behaviour, confirm the
test fails, restore, confirm it passes, and report the table on the pull
request. Three shipped defects here survived because their guarding tests were
vacuous.

Two things that make a mutation report honest:

- **Verify the mutation actually changed behaviour** before scoring it. A
  mutation that is a no-op will read as "survived" and send you hunting a
  phantom gap — or worse, as "caught" by an unrelated test.
- **Report survivors.** A pass where everything is caught first time is more
  often a weak mutation set than a strong test set.

**Run `/code-review`** — model-invoked, so reach for it. Two axes: Standards and
Spec. The Spec axis is the one that catches scope creep, which self-reporting in
a pull request body does not substitute for. Namespace it if ambiguous; two
enabled plugins ship a `code-review`.

**Regenerate `docs/TEST_REGISTRY.md`** if you added or renamed a test, or CI
lint fails. Run the suite with `--junitxml` first so statuses record.

Then `ruff check .`, `ruff format --check .`, and `scripts/check_docs_sot.py`.

## Publishing

Draft pull request, `Closes #<n>` for an implementation issue. Scan the body and
every commit message for a closing keyword near a `#`-prefixed number first —
the parser is lexical and ignores negation, and this has fired twice here.
`--body-file` at a UTF-8-without-BOM file; `--body` mangles non-ASCII.

Verify CI on the **exact tip commit**, not on a stale run for an earlier push.
Merging is a separate explicit action by the maintainer.

## Never

Push to `main`, force-push, rebase a published branch (merge `main` *into* it),
or delete a branch, ref, worktree or stash without separate exact-target
approval. `C:\Users\magil\MCI-GRU` is read-only — `git -C` reads only, and never
`git gc`, `prune`, `repack`, or `stash clear` there. Fingerprint it at session
start and end and report both.
