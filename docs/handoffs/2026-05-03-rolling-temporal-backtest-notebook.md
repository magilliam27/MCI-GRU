# Rolling Temporal Backtest Notebook Handoff

Last updated: 2026-05-03

## Resume Here

- Pull latest `main` and open `notebooks/rolling_temporal_backtest_colab.ipynb`.
- In the notebook, paste the FRED API key into the `## 2. FRED API Key` cell (`MY_FRED_KEY = ''`), then run cells top-to-bottom in Colab with a GPU runtime.
- If the user reports a notebook runtime error, start with `scripts/gen_temporal_rolling_backtest_nb.py` and regenerate the notebook with `python scripts/gen_temporal_rolling_backtest_nb.py`.

## Current Objective

Run a fixed promising MCI-GRU recipe across prior untouched test years 2022, 2023, and 2024 using older S&P 500 universe CSVs from Google Drive. The goal is robustness evidence, not a new model-selection sweep.

## What Changed

- Added `configs/data/temporal_2016.yaml` for the 2022 holdout window.
- Added `scripts/gen_temporal_rolling_backtest_nb.py` as the reproducible source for the notebook.
- Added `notebooks/rolling_temporal_backtest_colab.ipynb`.
- Pushed three commits to `main`:
  - `a997080` - `Add rolling temporal backtest notebook`
  - `2328d43` - `Add FRED key cell to rolling backtest notebook`
  - `f4a64a2` - `Fix rolling backtest notebook model budget`

## Key Decisions

- Use rolling fixed-universe windows:
  - 2022 test: `sp500_2016_universe_data.csv`, train `2016-2020`, validate `2021`, test `2022`.
  - 2023 test: `sp500_2017_universe_data.csv`, train `2017-2021`, validate `2022`, test `2023`.
  - 2024 test: `sp500_2018_universe_data.csv`, train `2018-2022`, validate `2023`, test `2024`.
- Use the Drive data folder `/content/drive/MyDrive/MCI_GRU_shared/data`.
- Keep the model recipe fixed across years:
  - `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`
  - `training.loss_type=ic`
  - `training.label_type=returns`
  - `training.selection_metric=val_ic`
  - `model.label_t=5`
  - `graph.drop_edge_p=0.1`
  - `seed=1729`
- Set full-budget default `NUM_MODELS = 20` per `docs/NOTEBOOK_BEST_PRACTICES.md`.
- Keep `REGIME_STRICT = True`; the notebook now has a `MY_FRED_KEY` paste cell before strict regime checks.
- Avoid the previous `repo_data_path` KeyError by deriving repo-relative CSV paths in the matrix cell instead of depending on mutation from the data-check cell.

## Important Files

- `notebooks/rolling_temporal_backtest_colab.ipynb` - user-facing Colab notebook.
- `scripts/gen_temporal_rolling_backtest_nb.py` - source of truth for regenerating the notebook.
- `configs/data/temporal_2016.yaml` - new 2022 holdout data config.
- `configs/data/temporal_2017.yaml` - existing 2023 holdout config; notebook overrides `data.filename`.
- `configs/data/temporal_2018.yaml` - existing 2024 holdout config; notebook overrides `data.filename`.
- `docs/NOTEBOOK_BEST_PRACTICES.md` - confirmed final confirmation notebooks should use `training.num_models = 20`.
- `tests/backtest_sp500.py` - notebook backtest engine; uses open-to-open returns, costs, rank-drop gate, BHY haircut.

## Verification

- Ran `git status --short`; current state shows only untracked `docs/handoffs/` plus permission warnings for user/global ignore and blocked cache dirs.
- Ran `git rev-list --left-right --count HEAD...origin/main`; output `0 0`, so local `main` and `origin/main` matched after the push.
- Ran `git log --oneline -5`; latest commits include `f4a64a2`, `2328d43`, and `a997080`.
- Inspected the generated notebook and generator with `Select-String`:
  - `MY_FRED_KEY` exists.
  - `NUM_MODELS = 20` exists.
  - `repo_data_path_for` exists.
  - `/content/drive/MyDrive/MCI_GRU_shared/data` exists.
  - all three CSV names exist.
- Previously verified during implementation:
  - `python scripts\gen_temporal_rolling_backtest_nb.py`
  - `python -m py_compile scripts\gen_temporal_rolling_backtest_nb.py`
  - notebook JSON parses and all code cells AST-parse.
  - minimal local repro with a pandas stub built 3 training jobs and confirmed `training.num_models=20`.
- Not run locally:
  - full model training.
  - full backtests.
  - full pytest suite.

## Open Risks

- Actual training/backtesting must run in Colab or another GPU environment; local environment was not used for full execution.
- Strict regime runs need `FRED_API_KEY` unless `REGIME_INPUTS_CSV` is set to a valid repo-relative regime CSV.
- The notebook assumes the Google Drive folder is mounted at `/content/drive/MyDrive/MCI_GRU_shared/data`; if the user has it as a Drive shortcut/shared-drive path, the setup cell may need path adjustment.
- `docs/handoffs/` remains untracked locally. It currently contains the earlier `2026-05-02-backtest-fairness-review.md` and this handoff file.
- Full-budget defaults mean each year trains a 20-model ensemble; Colab runtime and Drive storage can be substantial.

## Next Actions

1. In Colab, pull latest `main`, open `notebooks/rolling_temporal_backtest_colab.ipynb`, paste `MY_FRED_KEY`, and run through data availability and matrix definition first.
2. Confirm the matrix displays exactly three training jobs and the overrides include `training.num_models=20`.
3. Run training and backtests; inspect `RUN_ROOT` under `MCI-GRU-Ablations/rolling_temporal_backtest/<RUN_TAG>`.
4. Review `backtest_decision_table.csv`, `rolling_temporal_backtest_summary.md`, and plots before interpreting results.
5. If runs fail, inspect the notebook's failed-run inspection cell and the per-run logs under `RUN_ROOT/logs/`.

## Data/Experiment State

- Required Drive files:
  - `sp500_2016_universe_data.csv`
  - `sp500_2017_universe_data.csv`
  - `sp500_2018_universe_data.csv`
- Output root:
  - `/content/drive/MyDrive/MCI-GRU-Ablations/rolling_temporal_backtest/<RUN_TAG>`
- Backtest scenarios:
  - `k10_spread5_slip0_rankdrop30_daily`
  - `k20_spread5_slip0_rankdrop30_daily`
- Both scenarios use daily open-to-open backtesting, 5 bps spread, 0 bps slippage, rank-drop gate `30`, BHY multiple-testing adjustment, and `label_t=5`.

## Do Not Do

- Do not pick different winning configs per year if the goal remains untouched-year validation.
- Do not treat haircutted Sharpe `0` as zero realized return; it can mean BHY adjusted p-value hit `1.0`.
- Do not use the 2019-through-2026 universe for 2022-2024 robustness unless explicitly changing the fairness design.
- Do not import `GraphBuilder` into `paper_trade/`; this notebook uses training/backtest scripts, not paper-trade inference.
