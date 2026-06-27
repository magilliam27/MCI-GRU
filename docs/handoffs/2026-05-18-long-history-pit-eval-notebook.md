# Long-History PIT Eval Notebook Handoff

Last updated: 2026-05-18

## Resume Here

- Start by locating the newest Colab output folder for `long_history_pit_eval`, likely under `/content/drive/MyDrive/MCI-GRU-Ablations/long_history_pit_eval/<RUN_TAG>` or the local Colab mirror `/content/mci_gru_work/long_history_pit_eval/<RUN_TAG>`.
- Current objective: evaluate the completed notebook results, not re-plan the experiment. Inspect `summaries/training_results.csv`, `summaries/backtest_results.csv`, `summaries/long_history_decision_table.csv`, `summaries/grouped_his_t_summary.csv`, and `summaries/long_history_pit_eval_summary.md`.
- If the notebook did not finish or setup fails, first check whether the Colab clone is on `codex/pit-universe-validation` and whether the branch contains the notebook fixes described below.

## Current Objective

- Evaluate issue #23: whether longer MCI-GRU history windows improve the frozen production-style recipe under true PIT masked-panel evaluation.
- The result review should compare `his_t=10`, `21`, `63`, and `126` across PIT test years `2022`, `2023`, `2024`, and `2025`.

## What Changed

- Added long-history Hydra presets:
  - `configs/experiment/long_history_his_t_21.yaml`
  - `configs/experiment/long_history_his_t_63.yaml`
  - `configs/experiment/long_history_his_t_126.yaml`
- Added generated Colab notebook and generator:
  - `scripts/gen_long_history_pit_eval_nb.py`
  - `notebooks/long_history_pit_eval_colab.ipynb`
- Added regression coverage:
  - `tests/test_long_history_issue23.py`
- Updated terminology/docs:
  - `CONTEXT.md`
  - `docs/CONFIGURATION_GUIDE.md`

## Key Decisions

- Full performance evidence must use true PIT masked-panel data.
- Non-PIT anchored historical snapshot universe runs are mechanics smokes only; do not use them for performance claims.
- The full matrix is `his_t=[10, 21, 63, 126]` by `year=[2022, 2023, 2024, 2025]`.
- `his_t=252` is gated by `INCLUDE_HIS_T_252 = False`; do not include it in first-pass conclusions unless explicitly enabled later.
- The notebook pins the frozen recipe semantics: static threshold graph, `drop_edge_p=0.1`, pure IC loss, returns labels, `selection_metric=val_ic`, `shuffle_train=true`, `label_t=5`, `gru_attn`, 20 models, 100 epochs, patience 15.
- The decision score is only a sorting aid. Read Sharpe/ASR, ARR, excess return, MDD, turnover, failure rate, and per-year consistency directly.

## Important Files

- `notebooks/long_history_pit_eval_colab.ipynb`: Colab artifact the user ran.
- `scripts/gen_long_history_pit_eval_nb.py`: source of truth for regenerating the notebook.
- `tests/test_long_history_issue23.py`: guards notebook content, presets, code-cell parsing, PIT setup behavior, and docs expectations.
- `docs/DEFAULT_EXPERIMENT_RECIPE.md`: recipe semantics the notebook is meant to preserve.
- `docs/CONFIGURATION_GUIDE.md`: long-history usage and smoke guidance.
- `CONTEXT.md`: agreed domain language for Long-History Preset, Mechanics Smoke, Anchored Historical Snapshot Universe, and Colab Evaluation Notebook.

## Verification

- Passed locally after the branch/data-config fixes:
  - `.venv\Scripts\python.exe -m pytest tests/test_long_history_issue23.py -v --basetemp=.codex_tmp\pytest_issue23_long_history_branch`
  - `.venv\Scripts\ruff.exe check scripts\gen_long_history_pit_eval_nb.py tests\test_long_history_issue23.py`
- Passed locally after reuse hardening:
  - `.venv\Scripts\python.exe -m pytest tests/test_long_history_issue23.py -v --basetemp=.codex_tmp\pytest_issue23_long_history_reuse`
  - `.venv\Scripts\ruff.exe check scripts\gen_long_history_pit_eval_nb.py tests\test_long_history_issue23.py`
- Passed locally after the checkout-rerun regression was added:
  - `.venv\Scripts\python.exe -m pytest tests/test_long_history_issue23.py -v --basetemp=.codex_tmp\pytest_issue23_checkout_fix`
  - `.venv\Scripts\ruff.exe check scripts\gen_long_history_pit_eval_nb.py tests\test_long_history_issue23.py`
- The pytest runs passed with a local `.pytest_cache` permission warning; the tests themselves passed.
- The actual Colab experiment results have not been evaluated in this chat. The next chat should inspect the saved CSV/Markdown outputs before making any model-performance claims.

## Open Risks

- The branch `codex/pit-universe-validation` was pushed through commit `52beb1d` (`Harden long-history Colab run reuse`).
- After that push, an additional checkout-rerun fix was staged locally but not committed/pushed when the user paused to ask for explanation. It affects:
  - `scripts/gen_long_history_pit_eval_nb.py`
  - `notebooks/long_history_pit_eval_colab.ipynb`
  - `tests/test_long_history_issue23.py`
- That staged fix makes existing Colab clones unlink notebook-generated `pit_temporal_*.yaml` files before `git checkout -B codex/pit-universe-validation origin/codex/pit-universe-validation`. Without it, a reused Colab runtime can hit a git checkout error if prior generated YAMLs exist.
- If evaluating results from a notebook run before the branch/setup repairs, expect failed rows with errors like missing `experiment/pit_temporal_YYYY` or `DataConfig.__init__() got an unexpected keyword argument 'pit_universe_mode'`; exclude those failed setup runs from performance interpretation.

## Next Actions

1. Locate the newest successful `long_history_pit_eval` run folder in Drive and read the summary artifacts.
2. Confirm all 16 training jobs completed or identify which `his_t/year` rows failed.
3. Build a concise evaluation: per-year table, grouped `his_t` summary, failure rate, best/worst years, and whether any longer window beats the `his_t=10` baseline robustly.
4. Treat decision score as a helper only; base recommendations on ASR/Sharpe, excess return, drawdown, turnover, and cross-year stability.
5. If the setup/checkout issue still appears in Colab, either run the manual repair cell from the previous chat or commit/push the staged checkout-rerun fix before rerunning setup.

## Commands Run

- `.venv\Scripts\python.exe scripts\gen_long_history_pit_eval_nb.py`
- `.venv\Scripts\python.exe -m pytest tests/test_long_history_issue23.py -v --basetemp=.codex_tmp\pytest_issue23_checkout_fix`
- `.venv\Scripts\ruff.exe check scripts\gen_long_history_pit_eval_nb.py tests\test_long_history_issue23.py`
- `git push origin codex/pit-universe-validation` succeeded for commits through `52beb1d`.

## Data/Experiment State

- Notebook branch: `codex/pit-universe-validation`.
- Colab output base:
  - Local runtime: `/content/mci_gru_work/long_history_pit_eval/<RUN_TAG>`
  - Drive sync: `/content/drive/MyDrive/MCI-GRU-Ablations/long_history_pit_eval/<RUN_TAG>`
- Observed failed-run tag in user logs: `20260518_014212`. There may be a newer successful run tag after the repair cell; inspect the notebook output or Drive folder timestamps.
- Market CSV used by the notebook:
  - `data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv`
- PIT universe CSV:
  - `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
- The notebook writes PIT temporal presets into the Colab clone if needed:
  - `pit_temporal_2022`
  - `pit_temporal_2023`
  - `pit_temporal_2024`
  - `pit_temporal_2025`

## User Preferences

- The user wants another chat to evaluate the experiment results, not restart planning.
- Keep terminology precise: use "PIT masked-panel" for performance evidence and "Anchored Historical Snapshot Universe" only for mechanics smoke.
- The user agreed that full eval must use PIT, while smoke checks can be cheap wiring checks.

## Do Not Do

- Do not treat failed setup rows as experiment evidence.
- Do not use non-PIT smoke metrics as model-performance evidence.
- Do not compare the long-history presets against a missing baseline; `his_t=10` in the same notebook is the baseline.
- Do not rely solely on the decision score for recommendations.
- Do not rerun the full 16-job matrix unless the saved artifacts are missing or incomplete.

## References

- Pushed commits:
  - `106fced Add long-history PIT Colab evaluation`
  - `52beb1d Harden long-history Colab run reuse`
- Relevant notebook outputs to inspect:
  - `summaries/training_results.csv`
  - `summaries/backtest_results.csv`
  - `summaries/long_history_decision_table.csv`
  - `summaries/grouped_his_t_summary.csv`
  - `summaries/long_history_pit_eval_summary.md`
