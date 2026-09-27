# Contributing

This is a single-maintainer research repository, and work moves through it the same
way whoever is doing it. The rules below are a digest. Each links the document that
holds the full policy, and that document wins where the two differ.

## Where work comes from

Work is tracked in the repository's GitHub issues. A map issue names a destination and
the tickets that reach it. Each ticket says whether it resolves through live exchange
with the maintainer or can be taken unattended, and carries one ticket-type label, one
category role, and one state role. The label vocabulary is in
[`docs/agents/triage-labels.md`](docs/agents/triage-labels.md); the tracker policy,
including how maps and tickets are shaped, is in
[`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md). New tickets start from
the issue templates, which carry those sections.

## Before you branch

1. Read the open issues and the open pull requests, including their assignees and their
   owned paths.
2. Claim the ticket: assign it to yourself and declare the paths the change will own in
   a line near the top of its body. Work that has no ticket gets one first.
3. Branch from the current `origin/main`, normally in an isolated worktree.

The order matters. More than one session can work here at once, and the tracker is the
only surface all of them read. A claim made after branching arrives too late to be read
by the other side of a collision.

## Branches and pull requests

- Branch names are `<harness>/<task>`. The prefix records what produced the work; every
  branch carries the same obligations.
- Nothing is pushed to `main`. Every change arrives as a draft pull request and stays a
  draft until the maintainer merges it.
- Merge `main` into a published branch; never rebase one.
- No branch, ref, stash, or worktree is deleted without exact-target approval.
- Commit only the files the ticket owns. Widening is allowed and is recorded in the
  commit message.
- Before publishing a pull request body or a commit message, scan it for a closing
  keyword next to a `#`-number. GitHub's parser reads the keyword and the reference and
  ignores the words between them.

The branch and pull request policy, and the rule that merging is a separate action, are
in [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md).

## What a pull request carries

The pull request template lays these out. Each is a section, not a checkbox.

- **What changed**, and the ticket it closes on merge.
- **Owned paths**, with any widening and why, and what was deliberately not touched.
- **A review on two axes**: the repository's standards and the ticket's spec, with each
  finding fixed, accepted with a reason, or disclosed.
- **A mutation table** for every load-bearing test: the behaviour is broken on purpose,
  the test must fail; the behaviour is restored, the test must pass. A mutation that
  changed nothing is reported as such. A green test is not evidence on its own.
- **Verification**: the interpreter, the working directory, each command, and its exit
  status.
- **No performance claim**, whenever research code is touched: which model, recipe,
  feature, graph, loss, evaluation, and data defaults are unchanged, and where any
  changed number is read from.

## Tests and docs

- Tests run through the isolated launcher on Windows, and a change that adds, renames,
  or removes a test regenerates the test registry, or CI lint fails.
  [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) has the commands, the markers, the
  verification ladder, and the no-lookahead canary every timing-sensitive feature
  carries.
- Engineering constraints (source-of-truth order, runtime invariants, timing and
  finance rules, configuration, code, test, and workspace rules) are in
  [`docs/agents/guide.md`](docs/agents/guide.md), under Engineering Constraints.
- A dated research report lives under `docs/research/current/` or
  `docs/research/archive/`, never at the `docs/` root; CI refuses one that does.
  [`docs/research/README.md`](docs/research/README.md) is the status map.
- When a document and the code disagree, the code wins.
  [`docs/agents/domain.md`](docs/agents/domain.md) gives the order and says how to
  report the drift rather than silently rewriting prose.

## Evidence

- A report marks each figure `[Verified]`, read from a run artifact, or `[Inferred]`,
  reasoning on top of one, and a run-backed report names its run tag and its artifact
  folder. The research issue template carries both requirements.
- A handoff is not evidence, a notebook contract test is not a live run, and a
  mechanics smoke proves wiring only. Research claims come from
  `docs/research/current/`.
- Run folders and artifacts live in the maintainer's private Drive, so a link there is
  a permission wall for anyone else. The tables and figures inside each report are the
  public record.
- Confirmation-scale runs are generated notebooks;
  [`docs/NOTEBOOK_BEST_PRACTICES.md`](docs/NOTEBOOK_BEST_PRACTICES.md) is the contract
  they follow.

## Retired surfaces

Files leave `main` through pull requests and stay in history. Everything removed in the
2026-09 cleanup is readable at the tag `archive/pre-cleanup-2026-09`:

```bash
git show archive/pre-cleanup-2026-09:<path>
```

A guard test keeps retired paths from drifting back, so a retired file returns only
through a ticket that says why.

## License

Contributions are accepted under the repository's [MIT license](LICENSE).
