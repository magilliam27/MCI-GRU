@AGENTS.md

# Claude Code Notes

`AGENTS.md` above is the repository entrypoint and is imported automatically.
This file carries only what is Claude-specific. It is deliberately not a second
architecture, configuration, or testing document: those have canonical homes and
duplicating them creates a competing source of truth.

## Skills

This repository runs the Matt Pocock skill set. `superpowers` is disabled here
so that every harness working this repository produces identically shaped
tickets, specs, and evidence. Under the plugin install these skills are
namespaced: `/mattpocock-skills:implement`, not `/implement`.

## Cold start

Work may arrive here from any harness, or from a previous session of this one.
Do not assume a particular predecessor and do not rely on a handoff document
being in context. Brief yourself from the tracker every time:

1. `docs/agents/issue-tracker.md` and `docs/agents/guide.md`.
2. `gh issue view 97 --repo magilliam27/MCI-GRU --comments` for the Wayfinder
   map. **The map's current state lives in its newest comments, not its body.**
   Body edits raise no timeline entry, so the body alone reads as frozen.
3. The active child ticket and its evidence comments.
4. Pick the next unstarted item, then create a worktree off then-live
   `origin/main` under `C:\Users\magil\.claude\worktrees\`, never under
   `.codex\worktrees\`.

## Hard rules

- Draft pull request only. Never push to `main`. `main` carries no branch
  protection, so this is policy rather than mechanism.
- Never delete a branch, ref, worktree, stash, or snapshot without separate
  exact-target approval. See issue #99.
- Merge `main` **into** a published branch. Never rebase one; that requires a
  force-push, which is forbidden.
- `C:\Users\magil\MCI-GRU` is read-only. Use `git -C` reads only. It stays on
  `codex/paper_trade_scrape` at `e286649` with 40 dirty entries. Fingerprint
  before and after every session and report both. A
  `warning: could not open directory '.pytest-tmp/': Permission denied` is
  expected and is not a change.
- Load-bearing tests are mutation-checked: break the behaviour, confirm the test
  fails, restore, confirm it passes, and report the table. Three shipped defects
  in this repository survived because their guarding tests were vacuous.

## Windows and tooling

- Tests run as
  `.\.venv\Scripts\python.exe scripts/run_pytest_isolated.py tests/ -v`. Use the
  isolated launcher rather than bare pytest; it avoids Windows ACL reuse
  problems.
- Any branch that adds or renames a test must regenerate `docs/TEST_REGISTRY.md`
  or CI lint fails. Run the suite with `--junitxml=test_reports\junit.xml` first
  so statuses are recorded rather than blank.
- `gh issue edit` and `gh pr create`: always `--body-file` pointing at a
  UTF-8-without-BOM file. `--body` mangles non-ASCII on Windows PowerShell and
  the corruption compounds across successive edits.
- `gh issue close` takes `--comment`, not `--body-file`. Post the comment first,
  then close.
- Retargeting a pull request's base fires an `edited` event, which is not in the
  default `pull_request` trigger set, so CI will not run. Close and reopen the
  pull request to trigger it. Never merge on "no checks reported".
- PowerShell mangles `gh --jq` filters containing `.[]`. Pipe to
  `ConvertFrom-Json` instead.
- Pytest `addopts` overrides `-q`; use `-o addopts=` when you need clean node
  identifiers.
- Piping `git archive` into `tar` corrupts the stream on Windows. Write the
  archive to a file first.
- `gh auth status --hostname github.com` must be run standalone. A sandboxed
  result is not authoritative on this machine.
