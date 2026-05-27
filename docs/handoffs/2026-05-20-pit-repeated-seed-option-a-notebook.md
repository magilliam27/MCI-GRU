# PIT Repeated-Seed Option A Notebook Handoff

Last updated: 2026-05-20

## Resume Here

- First inspect `notebooks/pit_repeated_seed_replication_colab.ipynb` and `scripts/gen_pit_repeated_seed_replication_nb.py`.
- The notebook has been expanded to Option A: three replication base seeds, fixed recipe, four PIT years, and a final issue-closeout summary for issues #29, #30, and #31.
- If the user has already run the expanded Colab, go straight to Google Drive:
  `MCI-GRU-Ablations/pit_repeated_seed_replication/<RUN_TAG>/summaries/`.
- The most important analysis file for the next conversation is:
  `pit_repeated_seed_issue_closeout_summary.csv`.

## Current Objective

The user wants the PIT repeated-seed notebook to produce enough evidence to close issues #29, #30, and #31. The next conversation should analyze the resulting Google Drive artifacts after the expanded notebook run finishes.

## What Changed

- `REPLICATION_BASE_SEEDS` changed from `[314159]` to `[314159, 271828, 161803]`.
- The full run budget is now:
  - 3 base seeds
  - 4 PIT test years: 2022, 2023, 2024, 2025
  - 20 models per yearly job
  - 12 yearly training jobs
  - 240 trained models total
- Added a new notebook section:
  `## 9. Cross-Seed Closeout Evidence For Issues #29-#31`.
- Added generated summary artifacts:
  - `pit_repeated_seed_seed_summary.csv`
  - `pit_repeated_seed_yearly_seed_summary.csv`
  - `pit_repeated_seed_issue_closeout_summary.csv`
- Regenerated:
  - `notebooks/pit_repeated_seed_replication_colab.ipynb`
- Updated notebook contract tests:
  - `tests/test_pit_repeated_seed_replication_notebook.py`

## Key Decisions

- Keep the frozen default recipe fixed. This is not a recipe sweep.
- Expand only the seed dimension so issue #29 and issue #30 can be interpreted as robustness questions rather than config-search outcomes.
- Issue #31 closeout status requires all expected training/backtest pairs, PIT breadth, prediction counts, and at least 3 complete seeds.
- Issue #29 closeout status uses the `replication_all_seeds` pooled significance row. It only marks `supports_closeout` when annualized excess is positive, BHY-adjusted p-value is at most `0.05`, and the daily excess CI lower bound is positive.
- Issue #30 closeout status marks `supports_closeout` only when 2022 is complete for all three seeds, all three seeds have negative 2022 excess return, and 2022 is the worst mean-excess year.
- The previous one-seed Drive run `20260520_161523` is useful background, but it is not the expanded Option A result.

## Important Files

- `notebooks/pit_repeated_seed_replication_colab.ipynb`
  - The Colab notebook the user will run.
- `scripts/gen_pit_repeated_seed_replication_nb.py`
  - Source generator for the notebook. Edit this first, then regenerate the notebook.
- `tests/test_pit_repeated_seed_replication_notebook.py`
  - Contract tests that ensure the notebook pins branch, data links, recipe, static regime inputs, artifacts, and Option A closeout logic.
- `docs/DEFAULT_EXPERIMENT_RECIPE.md`
  - Defines the frozen recipe the notebook intentionally preserves.
- `docs/ARCHITECTURE.md`
  - PIT masked-panel and pipeline invariants.

## Verification

Commands run during this iteration:

- Red test before implementation:
  - `.\.venv\Scripts\python.exe -m pytest tests\test_pit_repeated_seed_replication_notebook.py -v --basetemp=.codex_tmp\pytest_issue31_expand_red`
  - Result: failed as expected, 3 failed and 5 passed, because the notebook still had one seed and no closeout summary artifacts.
- Regenerated notebook:
  - `.\.venv\Scripts\python.exe scripts\gen_pit_repeated_seed_replication_nb.py`
  - Result: wrote `notebooks\pit_repeated_seed_replication_colab.ipynb`.
- Green notebook contract test:
  - `.\.venv\Scripts\python.exe -m pytest tests\test_pit_repeated_seed_replication_notebook.py -v --basetemp=.codex_tmp\pytest_issue31_expand_green`
  - Result: 8 passed. There was a pytest cache warning because `.pytest_cache` could not be written.
- Lint:
  - `.\.venv\Scripts\ruff.exe check scripts\gen_pit_repeated_seed_replication_nb.py tests\test_pit_repeated_seed_replication_notebook.py`
  - Result: all checks passed.
- Synthetic dry-run of the new closeout cell:
  - Executed the generated notebook closeout cell against small synthetic DataFrames.
  - Result: produced 3 issue rows and marked issues #29, #30, and #31 as `supports_closeout` under the intended synthetic pass conditions.
- Diff hygiene:
  - `git diff --check -- scripts\gen_pit_repeated_seed_replication_nb.py tests\test_pit_repeated_seed_replication_notebook.py notebooks\pit_repeated_seed_replication_colab.ipynb`
  - Result: no whitespace errors; Git printed line-ending warnings only.

Not run:

- The expanded 240-model Colab training run was not run locally.
- No full test suite was run.
- No commit or push was performed after the Option A notebook edit.

## Open Risks

- The expanded Option A notebook has not been pushed yet. If the user opens the GitHub Colab link before pushing, they may still get the previously pushed one-seed notebook.
- The new closeout logic is tested syntactically and with synthetic data, but the actual status values depend on the next full Colab run.
- The issue #29 significance threshold is intentionally strict. A directionally positive result may still show `needs_more_evidence` if corrected p-value or CI lower bound does not clear.
- The issue #30 status is also strict: mixed 2022 seed outcomes will remain `needs_more_evidence`.
- The working tree contains unrelated user/previous-agent changes. Do not stage or revert them casually.

## Next Actions

1. If the user wants to run Colab from GitHub, stage, commit, and push only:
   - `scripts/gen_pit_repeated_seed_replication_nb.py`
   - `notebooks/pit_repeated_seed_replication_colab.ipynb`
   - `tests/test_pit_repeated_seed_replication_notebook.py`
   - this handoff file if desired
2. After the user runs the expanded notebook, locate the newest Drive run under:
   `MCI-GRU-Ablations/pit_repeated_seed_replication/`.
3. Fetch and inspect these artifacts first:
   - `summaries/pit_repeated_seed_issue_closeout_summary.csv`
   - `summaries/pit_repeated_seed_seed_summary.csv`
   - `summaries/pit_repeated_seed_yearly_seed_summary.csv`
   - `summaries/pit_repeated_seed_pooled_daily_significance.csv`
   - `summaries/backtest_results.csv`
   - `summaries/pit_breadth_summary.csv`
   - `summaries/prediction_count_checks.csv`
4. Write the analysis report around the issue closeout table:
   - #31: pipeline and PIT validity
   - #29: pooled all-seed significance
   - #30: whether 2022 is a repeatable stress regime
5. If any issue says `needs_more_evidence`, explain exactly which condition failed and whether the right next move is more seeds, a 2022-specific diagnostic, or closing as not supported.

## Data/Experiment State

- Prior one-seed Drive run already analyzed:
  - Run tag: `20260520_161523`
  - Seed: `314159`
  - This run fixed the regime oil issue and completed all four years, but it is not enough by itself to close all three issues.
- Expected expanded run output path:
  - `MCI-GRU-Ablations/pit_repeated_seed_replication/<new RUN_TAG>/`
- Static regime input behavior remains unchanged:
  - Notebook draws or reuses one static FRED-backed regime CSV.
  - All training jobs receive `features.regime_inputs_csv=<relative path>`.
  - Static regime marker prevents accidentally reusing old live-FRED run dirs.

## Do Not Do

- Do not change the frozen recipe while analyzing Option A results.
- Do not interpret a one-seed run as the expanded Option A result.
- Do not stage unrelated dirty files shown by `git status`.
- Do not claim issues #29 or #30 are closed unless the issue closeout artifact and underlying summaries support that claim.

