# Rearchitecture Branch Merge Plan — 2026-07-06

**Branch:** `codex/rearchitecture-phase0-20260704` (22 commits, WS-A through WS-Q)
**Target:** `main` (GitHub: magilliam27/MCI-GRU, default branch `main`)
**Worktree:** `C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU` (primary checkout at `C:\Users\magil\MCI-GRU`, currently on `codex/top10-lambdarank-screen-20260625`)
**Status at time of writing:** branch NOT pushed; working tree clean except 6 pre-existing untracked `docs/handoffs/*.md` files (leave alone).

---

## Investigation Findings

### 1. Divergence from main
- Merge base: `63511f2` ("Upgrade cockpit git topology monitoring") — identical to local `main` tip.
- `origin/main` has moved **exactly one commit** past the branch point: `92fd561` "Surface named cockpit workstream decisions" (+39 lines across `mci_gru/cockpit/{models,render,runner}.py` and `tests/test_cockpit_runner.py`).
- **Conflict check performed:** `git merge-tree --write-tree codex/rearchitecture-phase0-20260704 origin/main` produced a clean tree with **zero conflicts**. Git's rename detection correctly routed `92fd561`'s edits into the relocated `cockpit/` package (verified: merged tree contains `cockpit/models.py +1`, `cockpit/render.py +1`, `cockpit/runner.py +21`, `tests/test_cockpit_runner.py +16` relative to the branch tip). The WS-B relocation absorbs the upstream change cleanly.
- Local `main` is 1 behind `origin/main`; local `main` is not checked out in any worktree (safe to fast-forward with `git branch -f`).

### 2. Remote / GitHub state
- Single remote `origin` → `https://github.com/magilliam27/MCI-GRU.git`.
- `gh` CLI authenticated (account magilliam27, `repo`+`workflow` scopes). Repo allows merge commits, squash, and rebase merges; `deleteBranchOnMerge` is off.
- Branch has **no upstream** — must be pushed before a PR can exist.
- Open PRs of note: cockpit-refresh PRs #52–#55 and several draft PRs sit on branches that still use the old `mci_gru/cockpit` layout (see Risks).

### 3. CI
- One workflow, `.github/workflows/ci.yml`, triggers on push/PR to main: `ruff check`, `ruff format --check` (ruff pinned 0.15.12), `python scripts/check_docs_sot.py`, then pytest (`-m "not requires_data and not requires_lseg" --timeout=120`, CPU torch on ubuntu), then `scripts/ci_smoke.py`.
- **SOT check is safe:** `check_docs_sot.py` uses `docs_dir.glob("*.md")` — non-recursive, so `docs/audits/BACKTEST_ENGINE_DIVERGENCE_2026-07-06.md` is never scanned. Both spec docs are explicitly in `ALLOWED_EXISTING_FILES`. No other dated-report offenders exist at `docs/` root.
- **Fresh-clone golden concern resolved:** the `65f4bae` fixup un-ignored the fixture CSVs; `git ls-files tests/fixtures/backtest_golden` shows 74 tracked files. CI's `actions/checkout` is effectively a fresh clone, so a green CI run on the PR is itself the fresh-clone proof.
- Recent CI runs on main are green (last three: success).

### 4. Commit history quality
All 22 commits are logically ordered (spec docs → config/test infra → relocations → module splits → transforms/pipeline decomposition → coupling/logging → backtest fork merge) and each was individually gated (ruff clean + full pytest green). Actual diff vs merge base: **223 files, +11,567 / −18,354** (larger than the ~+5k/−8k estimate; the delta is mostly WS-Q's `archive/` + `_uncertain/` deletions and the WS-N duplicate-engine deletion).

The only "fixup" is `65f4bae` (un-ignore golden CSVs), which logically belongs with `3f4fa8f`. **Recommendation: do NOT rewrite it.** It is self-documenting, was itself gated, and rebasing to squash it would rewrite the SHA of `3b531f0` and detach history from the gated states. Not worth it for one additive commit.

### 5. Risk surface for external consumers
- `mci_gru.cockpit` references: **only** the spec doc mentions the old path; `scripts/refresh_cockpit.py` and all tests import top-level `cockpit`. Clean.
- `tests/backtest_sp500*.py`: deprecated shims exist and re-export from `mci_gru.evaluation.backtest_engine` / `scripts/backtest_sp500_daily.py`, so `python tests/backtest_sp500.py` keeps working for one release cycle.
- Committed Colab notebooks were regenerated to `scripts/` paths **except one**: `notebooks/performance_proof_missing_grid_colab.ipynb` still invokes `REPO_DIR / 'tests' / 'backtest_sp500.py'`. Works today via the shim; must be regenerated before shim removal.
- `paper_trade/` docs/comments already reference the new engine path; nightly pipeline uses frozen checkpoints and does not import the moved modules by old path. `docs/workflows/` has no references to moved paths.

### 6. Stashes
Three stale stashes, all `mci_gru/pipeline.py` WIP snapshots from WS-I/WS-J (`ws_i_baseline`, `ws_j_diff`, `ws_j_diff2`). Contents are committed. Safe to drop after a quick eyeball (verification command below).

---

## DECISION: Single PR, merged with a **merge commit** (no squash, no rebase)

**Rationale:**
1. **Bisectability is the whole point of 22 gated commits.** This is a behavior-preserving rearchitecture; if a subtle regression surfaces weeks later, `git bisect` across the 22 commits pinpoints the workstream in ~5 steps. Squash collapses +11.5k/−18.4k into one unbisectable, effectively unreviewable-in-retrospect commit — the worst possible choice here.
2. **Merge commit preserves the exact tested SHAs.** Every intermediate state on the branch is a state that actually passed ruff + full pytest. GitHub's "rebase and merge" would rewrite all 22 SHAs on top of `92fd561`, producing 22 *untested* intermediate combinations. With a merge commit, only the merge commit itself is new — and PR CI validates exactly that tree (which `merge-tree` already showed is conflict-free).
3. **Single revert point.** `git revert -m 1 <merge-commit>` backs out the entire rearchitecture in one motion if needed, while individual commits remain revertable for surgical fixes.
4. **Review remains tractable** despite 223 files: review commit-by-commit in the PR UI; each commit message documents its workstream and gate.

**Alternatives rejected:** squash (destroys bisect/revert granularity); rebase-and-merge (untested rewritten SHAs, no single revert point); stacked PRs (the work is done and gated — splitting now adds days of mechanical overhead for zero additional safety, since per-commit review inside one PR gives the same granularity).

---

## Exact Command Sequence

All from the worktree `C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU` unless noted. PowerShell: use `;` not `&&`.

### Step 0 — Pre-flight (fresh evidence, ~5 min)
```powershell
cd C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU
git status                       # expect: only the 6 untracked docs/handoffs files (+ this plan file)
git fetch origin
git log --oneline codex/rearchitecture-phase0-20260704..origin/main   # expect: only 92fd561
git merge-tree --write-tree codex/rearchitecture-phase0-20260704 origin/main  # expect: tree hash, no conflict output
.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest  # expect 327 passed / 2 skipped
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```
If `origin/main` gained new commits beyond `92fd561`, re-run the `merge-tree` check; only proceed conflict-free.

Optionally commit this plan file first (it lives in `docs/handoffs/`, exempt from the SOT check which only scans `docs/` root):
```powershell
git add docs/handoffs/2026-07-06-rearchitecture-merge-plan.md
git commit -m "docs: rearchitecture merge plan handoff"
```

### Step 1 — Push and open PR
```powershell
git push -u origin codex/rearchitecture-phase0-20260704
gh pr create --base main --title "Rearchitecture phase 0: 17 workstreams (WS-A..WS-Q)" --body-file docs/handoffs/2026-07-06-rearchitecture-merge-plan.md
```
(Or write a dedicated PR body; key content: behavior-preserving, per-commit gates, artifact-diff/golden-fixture evidence, path-move table, review-commit-by-commit instruction.)

### Step 2 — Wait for CI
```powershell
gh pr checks --watch
```
CI green on the PR is the decisive gate: it is a fresh ubuntu clone, proving golden fixtures ship correctly, the SOT check passes, and ruff 0.15.12 formatting matches.

### Step 3 — Merge (merge commit, keep branch history)
```powershell
gh pr merge --merge --delete-branch=false
```
Keep the branch for a week or two as a reference, then delete.

### Step 4 — Sync local checkouts
`main` is not checked out in any worktree, so it can be force-moved from here:
```powershell
git fetch origin
git branch -f main origin/main
```
Then update the primary checkout (currently on `codex/top10-lambdarank-screen-20260625`):
```powershell
cd C:\Users\magil\MCI-GRU
git checkout main
git pull
```
(If the lambdarank branch there has uncommitted work, stash or commit it first.)

### Step 5 — Clean up stashes (after verification; destructive)
```powershell
cd C:\Users\magil\.codex\worktrees\fd6c\MCI-GRU
git stash show -p 'stash@{0}'   # ws_j_diff2  — pipeline.py WIP, superseded by WS-J commit 213251d
git stash show -p 'stash@{1}'   # ws_j_diff   — same
git stash show -p 'stash@{2}'   # ws_i_baseline — superseded by WS-I commit f34a5e6
git stash drop 'stash@{0}'; git stash drop 'stash@{0}'; git stash drop 'stash@{0}'
```
(Drop index 0 three times — indices reshuffle after each drop.)

---

## Pre-Merge Checklist
- [ ] `git fetch origin` — confirm main still only 1 commit ahead of merge base (else re-run merge-tree).
- [ ] Full Windows pytest at branch tip: 327 passed / 2 skipped.
- [ ] `ruff check` + `ruff format --check` clean locally (CI pins ruff 0.15.12).
- [ ] PR CI green — this simultaneously proves fresh-clone golden fixtures, ubuntu/CPU-torch cross-platform behavior of golden tests (fixtures generated on Windows), SOT check, and smoke run.
- [ ] Working tree has no accidental staged files beyond the plan doc; the 6 pre-existing handoff files stay untracked/local.

## Post-Merge Follow-Ups
1. **Shim removal (one release cycle, target ~early August 2026).** File a tracking issue listing: `tests/backtest_sp500.py`, `tests/backtest_sp500_daily.py`, `mci_gru/models/mci_gru.py` compat shim, GraphBuilder wrapper surface, `train_multiple_models` lazy facade. Before removal, grep notebooks/docs for old paths.
2. **Regenerate `notebooks/performance_proof_missing_grid_colab.ipynb`** — the only committed notebook still invoking `tests/backtest_sp500.py`. Works via shim today; breaks at shim removal. Do this before item 1.
3. **In-flight cockpit branches/PRs (#52–#55 refresh PRs, draft #56).** They modify `mci_gru/cockpit/*` which no longer exists on main. Rename detection may cope, but the refresh PRs are dated snapshots — recommend closing the stale ones and regenerating cockpit output from post-merge main instead of merging them.
4. **Communicate path moves** (commit to `AGENTS.md`/`docs/QUICK_REFERENCE.md` if not already): `mci_gru/cockpit` → `cockpit/`; `tests/backtest_sp500.py` → engine `mci_gru/evaluation/backtest_engine.py` + CLI `scripts/backtest_sp500.py`; `tests/backtest_sp500_daily.py` → `scripts/backtest_sp500_daily.py`.
5. **Drop the 3 stashes** (Step 5 above) and optionally prune the many stale detached-HEAD worktrees (`git worktree prune` won't remove them; use `git worktree remove <path>` per worktree once their chats are done).

## Risks (ranked)
1. **In-flight branches based on pre-merge layout** (cockpit refresh PRs, other codex worktrees): future merges of those branches will hit the relocations. Mitigation: close/regenerate stale PRs; rebase active branches promptly after this merge lands. *Likelihood: high for cockpit PRs; impact: low (docs/status content).*
2. **Cross-platform golden tests**: fixtures generated on Windows, CI runs ubuntu. Goldens use JSON with 1e-12 tolerance and pandas-parsed CSVs, so line endings/float formatting should be inert — but PR CI is the first true cross-platform run of the golden suite from a clean clone. Mitigation: it runs *before* merge; a failure blocks at Step 2, nothing lands broken.
3. **Old-path notebook** (`performance_proof_missing_grid_colab.ipynb`): silently depends on the deprecated shim. No merge-day breakage; becomes a landmine at shim removal if follow-up 2 is skipped.
4. **origin/main drift between now and merge**: currently only the clean-merging cockpit commit. Mitigation: Step 0 re-checks; window is small.
5. **None found for**: docs SOT check (non-recursive, allowlist correct), paper_trade nightly (frozen checkpoints, paths already updated), `docs/workflows/` (no stale path references).
