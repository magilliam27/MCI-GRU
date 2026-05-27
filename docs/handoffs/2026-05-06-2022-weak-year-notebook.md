# 2022 Weak-Year Notebook Handoff

Last updated: 2026-05-06

## Resume Here

- If continuing notebook fixes, start from `origin/main`, not the local `main` checkout.
- Current local state from `git status -sb`: `main...origin/main [ahead 1, behind 7]`.
- The local-only commit is `c1d86bf Preserve regime CSV lag semantics`; do not hard-reset or overwrite it.
- Remote `origin/main` has the merged notebook work through `fea613f` / PR #19.
- Fastest safe local setup for more notebook work: create a new branch from `origin/main`, then inspect `notebooks/2022_weak_year_investigation.ipynb` and `scripts/gen_2022_weak_year_investigation_nb.py`.

## Current Objective

Provide a robust Colab investigation notebook for explaining why 2022 was weak in the completed backtest proof grid. The notebook should load the real Drive artifacts, survive mixed artifact date formats, and guide follow-up diagnostics without reintroducing Drive-path guessing.

## What Changed

- PR #16 added `notebooks/2022_weak_year_investigation.ipynb` and `scripts/gen_2022_weak_year_investigation_nb.py`.
- PR #17 fixed the first setup cell to clone/pull `main` in Colab and added more Drive artifact handling, but the Drive folder guess was still wrong.
- PR #18 used the Google Drive connector to find the real artifact folder and updated the notebook/generator/tests:
  - `/content/drive/MyDrive/MCI-GRU-Ablations/performance_proof_missing_grid/20260505_030758`
- PR #19 fixed pandas mixed-format date parsing in the notebook loader and updated notebook best practices to require Google Drive skill/connector discovery before hardcoding existing Drive artifact paths.
- This handoff file was added locally after PR #19; it is not yet committed or pushed unless a later agent does that explicitly.

## Key Decisions

- Drive artifact paths must be grounded with the Google Drive skill/connector, not inferred from local `drive_outputs/`, sibling notebooks, or run-tag memory.
- The real Drive hierarchy discovered for the 2022 artifacts:
  - `MCI-GRU-Ablations` folder ID: `1KUIj06ekfNpZa1IkkcAdhHXbVZt-PYT5`
  - `performance_proof_missing_grid` folder ID: `1TAgNzfaKeULzgqGh_w-Bbj47ZE88aKvj`
  - `20260505_030758` run folder ID: `1LtDhVGxKb8aEyY7oWAuVMNL8LgRdz52K`
- The run folder contains:
  - `completed_proof_decision_table.csv`
  - `completed_pooled_daily_returns.csv`
  - `20260505_030758.zip`
- Mixed date strings in Drive CSVs are expected. The notebook uses:
  - `pd.to_datetime(values, format='mixed').dt.normalize()`
- The setup cell should fail early with searched candidates when artifacts are missing instead of allowing a later loader cell to raise a vague `FileNotFoundError`.

## Important Files

- `notebooks/2022_weak_year_investigation.ipynb` - user-facing Colab investigation notebook.
- `scripts/gen_2022_weak_year_investigation_nb.py` - source of truth; regenerate notebook from this script after edits.
- `tests/test_2022_weak_year_notebook_paths.py` - regression tests for Drive folder path, setup-cell failure behavior, and mixed date parsing.
- `docs/NOTEBOOK_BEST_PRACTICES.md` - now includes `Drive Path Discovery` guidance and a checklist item requiring Google Drive connector verification.
- `drive_outputs/weak_year_diagnostic/` - local gitignored mirror used during verification; do not treat it as Drive source of truth.

## Verification

- Current evidence pass:
  - `git status -sb` showed `main...origin/main [ahead 1, behind 7]`.
  - `git show --stat --oneline origin/main` showed `fea613f Merge pull request #19`.
  - `git show origin/main:notebooks/2022_weak_year_investigation.ipynb` confirmed `performance_proof_missing_grid`, `parse_mixed_dates`, and `format='mixed'`.
  - `git show origin/main:docs/NOTEBOOK_BEST_PRACTICES.md` confirmed `Drive Path Discovery`.
- Verification run before PR #19 merge:
  - `.venv\Scripts\python.exe -m pytest tests\test_2022_weak_year_notebook_paths.py -v` -> 3 passed.
  - `.venv\Scripts\python.exe -m ruff check scripts\gen_2022_weak_year_investigation_nb.py tests\test_2022_weak_year_notebook_paths.py` -> passed.
  - `.venv\Scripts\python.exe -m json.tool notebooks\2022_weak_year_investigation.ipynb` -> passed.
  - `.venv\Scripts\python.exe -m py_compile scripts\gen_2022_weak_year_investigation_nb.py` -> passed.
  - Direct smoke check executed setup/import/load cells against local artifacts; loaded 95,445 daily rows, `daily_raw['date']` was `datetime64[ns]`, and mixed dates normalized cleanly.
- GitHub CI for PRs #17-#19 failed on the pre-existing repo-wide `ruff check .` backlog, not on the changed notebook files.
- Full notebook execution in Colab was not run by Codex after PR #19; the user reported the date parse error before the fix.

## Open Risks

- Local `main` is intentionally divergent. Another agent must decide whether to merge/rebase `origin/main` with local commit `c1d86bf`; do not use `git reset --hard`.
- GitHub global CI remains red due existing ruff violations in unrelated files such as `mci_gru/graph/builder.py`, `mci_gru/graph/sector_edges.py`, `run_experiment.py`, other notebook generators, and tests.
- The notebook has been smoke-checked through artifact load locally, but not run end-to-end in Colab after PR #19.
- Rich ticker attribution depends on zip members being present for the selected scenario; the notebook handles missing rich artifacts by printing guidance.
- Local remote-tracking refs for merged `codex/` branches may appear stale; rely on `git ls-remote` or fresh fetch/prune before drawing conclusions.

## Next Actions

1. In Colab, rerun the first three notebook cells from GitHub `main`; confirm the load cell gets past `daily_raw['date']`.
2. If another notebook bug appears, reproduce it against the local `drive_outputs/weak_year_diagnostic` mirror when possible, then patch `scripts/gen_2022_weak_year_investigation_nb.py` and regenerate the notebook.
3. Continue the 2022 investigation in the notebook sections: top-k concentration, month clustering, graph/regime deltas, rank-drop stickiness, market context, ticker attribution.
4. If local repo work is needed, branch from `origin/main` to avoid dragging in `c1d86bf`, or deliberately reconcile local `main` first.
5. Treat the repo-wide ruff backlog as a separate cleanup task; do not mix it into notebook investigation fixes.

## References

- PR #16: `https://github.com/magilliam27/MCI-GRU/pull/16`
- PR #17: `https://github.com/magilliam27/MCI-GRU/pull/17`
- PR #18: `https://github.com/magilliam27/MCI-GRU/pull/18`
- PR #19: `https://github.com/magilliam27/MCI-GRU/pull/19`
