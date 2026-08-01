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
4. Claim the work on the tracker before branching — file an issue first if it
   has none, assign it, and declare its owned paths. See
   `docs/agents/issue-tracker.md`, **Claim Before Branching**. More than one
   session runs here at a time; the tracker is the only surface all of them
   read.
5. Then create a worktree off then-live `origin/main` under
   `C:\Users\magil\.claude\worktrees\`, never under `.codex\worktrees\`.
   **Name the directory after the work, not after nothing.** The `claude/*`
   branch prefix already records which harness produced it; the directory should
   record *what* it is — `mci-gru-<ticket>-<slug>`. Harness-generated names such
   as `elegant-mendeleev-12bad9` identify nothing to the next session reading
   `git worktree list`.

The standing default workspace is
`C:\Users\magil\.claude\worktrees\mci-gru-workspace`, a linked worktree detached
at `origin/main`. Start sessions there rather than in the protected checkout,
which is pinned before this file existed and therefore carries none of this
policy. Refresh it with `git checkout --detach origin/main` after a fetch. It is
a reading surface, not a work surface: scoped task worktrees still get their own
directory and branch.

Note that this harness resolves a repository root from `--git-common-dir`, which
is the protected checkout for **every** linked worktree, so a session-created
worktree lands at `C:\Users\magil\MCI-GRU\.claude\worktrees\<name>` regardless of
where the session started. That path is ignored (see `.gitignore`) so it cannot
disturb the fingerprint, and it shares the one object store, so nothing is lost
by it. It is a third worktree location the policy above does not name; treat it
as expected rather than as a violation.

## Hard rules

- Draft pull request only. Never push to `main`. `main` carries no branch
  protection, so this is policy rather than mechanism.
- Never delete a branch, ref, worktree, stash, or snapshot without separate
  exact-target approval. See issue #99.
- Merge `main` **into** a published branch. Never rebase one; that requires a
  force-push, which is forbidden.
- `C:\Users\magil\MCI-GRU` is read-only. Use `git -C` reads only. **The hard
  invariant is branch and HEAD: `codex/paper_trade_scrape` at `e286649`.** Those
  never move. If either has changed, stop and report before doing anything else.
- Fingerprint at session start and again at session end, and report both. The
  dirty count is **not** a fixed number — it moves whenever anyone exports data
  or writes an untracked artifact, and a rule that is routinely wrong stops
  being read. Compare against *your own* start value:
  - A delta inside your session is yours to explain. You caused it.
  - A delta across sessions should be explained by an open, assigned ticket
    whose owned paths cover it. Concurrent sessions are normal here; see
    `docs/agents/issue-tracker.md`, **Claim Before Branching**.
  - **A delta with no matching claim is the alarm.** Establish provenance
    before proceeding — do not adopt it as the new baseline.
  A `warning: could not open directory '.pytest-tmp/': Permission denied` is
  expected and is not a change. Count porcelain lines excluding it.
- **That directory is never removable, and "retired" never means "delete".** It
  is no longer the default working surface, but its `.git` is the only object
  store on this machine: every linked worktree resolves through it, and it holds
  roughly 68 ref entries and 6 stashes that exist on no remote — `refs/codex/*`
  backups, snapshots, and turn-diffs. `git ls-remote origin` advertises only
  `refs/heads/*` and `refs/pull/*`, so a fresh clone inherits none of them and
  does not warn. Prefer `git worktree add` over any clone.
- **Never run `git gc`, `git gc --prune=now`, `git prune`, `git repack -ad`, or
  `git stash clear` there.** Unreachable objects in that repository are
  load-bearing. `gc.pruneExpire=never` and `gc.auto=0` are set as a backstop, but
  an explicit `--prune=now` still overrides them.
- **Project settings resolve from the session's working directory and do not
  walk up; `CLAUDE.md` does.** A session started one directory below a workspace
  root quotes the correct policy while running the wrong skill set, with no
  warning. Start at the workspace root. `claude --setting-sources user` and SDK
  entrypoints load neither file regardless of directory.
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
- **GitHub's closing-keyword parser is lexical and ignores negation.** A pull
  request body or a commit message containing `does not fix #<N>` will close
  issue N. So will `this is not a fix for #<N>` and `unrelated to fixes #<N>` —
  only the keyword and the reference are read, never the words between them.
  When naming an issue a change does *not* resolve, keep the keyword away from
  the reference: write `#<N> is unaffected by this change` or `#<N> remains
  open`.
- **Documentation of that trap must not contain a live instance of it.** Write
  every example with the number symbolic — `#<N>`, never a real issue number —
  so the parser has nothing to match. This rule is here because the commit that
  first documented the trap quoted a real issue number in its own message, and
  on merge it closed that issue. That was the second occurrence; the first was
  the pull request the note had been written about. Both times the end state was
  correct by accident and the tracker recorded the wrong cause. The same applies
  to comments you post and to scripts that echo these strings: scan your own
  text for a closing keyword within a few words of a `#`-prefixed number before
  publishing it.
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
- The connected GitHub app is not usable from Claude Code. In CLI sessions no
  GitHub connector tools are registered at all; in desktop sessions the
  connector is present but unauthorized. Either case is a **connector gap**
  under `docs/agents/issue-tracker.md`'s routing rule, so host-routed `gh` is
  the compliant path here rather than an override of policy. Do not attempt to
  authorize a connector or start a new GitHub login; `gh` is already
  authenticated.
