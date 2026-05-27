# PIT Universe Validation Handoff

Last updated: 2026-05-13

## Resume Here

- Start from branch `codex/pit-universe-validation`, pushed at commit `c8d56b3`.
- The immediate next user-facing step is helping run the full Colab notebook:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_universe_validation_colab.ipynb`
- The PIT CSV is in Google Drive, not GitHub:
  `https://drive.google.com/file/d/1jNTr3TlRJfPI-eenRryS-5IUs0V-AYY_/view?usp=drivesdk`
- In Colab, set:
  - `PIT_UNIVERSE_CSV = '/content/drive/MyDrive/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv'`
  - `GENERATE_PIT_UNIVERSE = False`
  - `FAST_MODE = False` for the full run.

## Current Objective

Prove whether MCI-GRU predicts stock cross-sections under stricter universe controls. The important framing decision is that the existing `sp500_2016/2017/2018` files are anchored point-in-time snapshot universes, not naive 2026-current universes.

PIT is now best treated as an audit/robustness layer, not the central thesis. The user's goal is proving stock prediction skill, not necessarily predicting the rolling S&P 500 membership.

## What Changed

- Added a Colab-ready PIT validation notebook:
  - `notebooks/pit_universe_validation_colab.ipynb`
- Added notebook generator:
  - `scripts/gen_pit_universe_validation_nb.py`
- Updated LSEG Joiner/Leaver exporter:
  - `scripts/data/export_sp500_joiner_leaver_pit.py`
  - Lazy-loads `refinitiv.data` so tests can import without the SDK.
  - Emits pipeline-ready PIT schema: `kdcode, valid_from, valid_to`.
  - Adds unsuffixed aliases for LSEG tombstone RICs like `HOLX.OQ^D26`.
  - Coalesces overlapping/adjacent intervals per `kdcode` to avoid duplicate `(kdcode, dt)` rows after merge.
- Added tests:
  - `tests/test_pit_universe_validation_notebook.py`
  - `tests/test_sp500_joiner_leaver_pit_export.py`
- Pushed code branch:
  - `origin/codex/pit-universe-validation`
- Uploaded the PIT CSV to Drive after explicit user approval. The PIT CSV is not in the pushed Git commit.

## Key Decisions

- Existing universes are better described as anchored historical snapshot universes:
  - `sp500_2016_universe_data.csv` is the 2016 S&P 500 snapshot carried forward.
  - `sp500_2017_universe_data.csv` is the 2017 snapshot carried forward.
  - `sp500_2018_universe_data.csv` is the 2018 snapshot carried forward.
- The current PIT notebook is therefore a PIT-filtered anchored-universe stress test.
- A true rolling S&P 500 PIT backtest still requires a union price panel containing all names active in the S&P 500 over each window. The PIT CSV alone can remove invalid rows, but it cannot add post-anchor joiners that are absent from the underlying anchored market CSV.
- Do not frame PIT as "we care about predicting the S&P 500." Frame it as "we are removing future membership/survivor filtering objections."
- The best current proof stack is:
  - anchored 2016 -> 2022,
  - anchored 2017 -> 2023,
  - anchored 2018 -> 2024,
  - per-split completeness,
  - PIT-filtered anchored stress test,
  - later optional union-of-PIT-members price panel.

## Important Files

- `notebooks/pit_universe_validation_colab.ipynb` - runnable Colab notebook on branch `codex/pit-universe-validation`.
- `scripts/gen_pit_universe_validation_nb.py` - source of truth for regenerating the notebook.
- `scripts/data/export_sp500_joiner_leaver_pit.py` - LSEG Joiner/Leaver PIT exporter.
- `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv` - local ignored PIT CSV, uploaded to Drive.
- `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_meta.json` - local untracked metadata from export.
- `results/pit_smoke/pit_smoke/20260512_213843` - local one-epoch PIT smoke output.

## Verification

Observed current Git state:

- Current branch: `codex/pit-universe-validation`
- HEAD: `c8d56b3 Add PIT universe validation notebook`
- Upstream: `origin/codex/pit-universe-validation`
- `git show --stat --oneline HEAD` showed 5 committed files:
  - notebook,
  - exporter update,
  - generator,
  - two tests.
- `git ls-tree -r --name-only HEAD | Select-String sp500_pit...` returned no rows, confirming the PIT CSV is not in Git.

PIT export evidence:

- LSEG export command ran successfully after installing `refinitiv-data==1.6.2` into `.venv`.
- Generated local files:
  - `sp500_pit_joiner_leaver_20160101_20260513_current_members.csv`
  - `sp500_pit_joiner_leaver_20160101_20260513_changes.csv`
  - `sp500_pit_joiner_leaver_20160101_20260513_intervals.csv`
  - `sp500_pit_joiner_leaver_20160101_20260513_snapshots.csv`
  - `sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
  - `sp500_pit_joiner_leaver_20160101_20260513_meta.json`
- Metadata:
  - `current_members`: 503
  - `change_rows`: 503
  - `joiners`: 251
  - `leavers`: 252
  - `interval_rows`: 755
  - `pit_universe_rows`: 879
  - `snapshot_dates`: 230
  - `min_members_per_snapshot`: 503
  - `max_members_per_snapshot`: 509
- PIT CSV:
  - rows: 879
  - unique `kdcode`: 870
  - `valid_from` min: 2016-01-01
  - `valid_to` max: 2026-05-13
  - sha256: `a9ef692b83a9575cc329ff6f7d56b2e84daa9571ab60e66aa6ba2d7eed0b991b`

Commands/checks that passed:

- `.venv\Scripts\python.exe -m pytest tests\test_pit_universe_validation_notebook.py tests\test_sp500_joiner_leaver_pit_export.py -v`
  - 7 passed.
  - Warning: pytest cache path permission warning only.
- `.venv\Scripts\ruff.exe check scripts\gen_pit_universe_validation_nb.py scripts\data\export_sp500_joiner_leaver_pit.py tests\test_pit_universe_validation_notebook.py tests\test_sp500_joiner_leaver_pit_export.py`
  - All checks passed.
- `.venv\Scripts\python.exe -m py_compile scripts\gen_pit_universe_validation_nb.py scripts\data\export_sp500_joiner_leaver_pit.py`
  - Passed earlier in the session.
- One-epoch PIT smoke run completed:
  - command used `data=temporal_2019`, `+data.use_pit_universe=true`, `+data.filter_stocks_per_split=true`, one model, one epoch.
  - output: `results/pit_smoke/pit_smoke/20260512_213843`
  - selected 394 usable stocks after PIT plus per-split filtering.
  - graph built with 408 edges for 394 nodes.
  - one epoch completed and averaged predictions were written.

Coverage checks:

- After adding alias/coalesce behavior, no missing S&P codes for the 2016/2017/2018/2019 fixed CSVs after PIT matching.
- `sp500_2019_universe_data_through_2026.csv` had only `SPY.P` missing, treated as an ETF artifact.
- No duplicated `(kdcode, dt)` rows after PIT merge.

Anchored-vs-rolling PIT analysis, deduping tombstone aliases to base RICs:

- 2016 anchor / 2022 test:
  - rolling PIT active avg: 503.7
  - overlap with fixed file avg: 445.0
  - active PIT names absent from fixed file avg: 58.7
  - PIT-filtered fixed unique: 453
- 2017 anchor / 2023 test:
  - rolling PIT active avg: 503.0
  - overlap avg: 459.6
  - active PIT names absent avg: 43.4
  - PIT-filtered fixed unique: 468
- 2018 anchor / 2024 test:
  - rolling PIT active avg: 503.0
  - overlap avg: 475.2
  - active PIT names absent avg: 27.8
  - PIT-filtered fixed unique: 484

## Open Risks

- The full Colab run has not yet been executed.
- The full matrix is large:
  - 3 years,
  - 3 seeds,
  - 2 variants,
  - 4 universe controls,
  - 72 training jobs,
  - 20 ensemble members and 100 epochs per full job.
- The current notebook proves a PIT-filtered anchored-universe stress test, not a complete rolling S&P 500 PIT backtest.
- A true rolling PIT test needs a union-of-members price panel generated from LSEG, not only the PIT membership CSV.
- Drive upload put the PIT CSV in Drive root. If Colab cannot find it, set `PIT_UNIVERSE_CSV` explicitly to `/content/drive/MyDrive/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`.
- There are unrelated local dirty files that were intentionally not staged or pushed:
  - `docs/REGIME_DATA_CONTRACT.md`
  - `mci_gru/data/data_manager.py`
  - `scripts/colab_regime_reconcile.py`
  - `skills/research-paper-to-mci-gru/SKILL.md`
  - `skills/research-paper-to-mci-gru/agents/openai.yaml`
  - `tests/test_regime_features.py`
  - `docs/handoffs/2026-05-06-2022-weak-year-notebook.md`
  - `docs/research-paper-evaluations/`
  - `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_meta.json`

## Next Actions

1. Help the user run the notebook in Colab from branch `codex/pit-universe-validation`.
2. In the notebook, set `PIT_UNIVERSE_CSV` explicitly to the Drive root path and `GENERATE_PIT_UNIVERSE = False`.
3. Run `FAST_MODE=True` once to confirm Drive paths/GPU/output.
4. Switch `FAST_MODE=False` for the full PIT validation matrix.
5. After results land, compare `baseline` vs `pit_plus_per_split` first, then inspect whether any performance survives across years, seeds, and `no_regime`.

## Commands Run

- `git status --short`
- `git branch --show-current`
- `git log --oneline --decorate -3`
- `git show --stat --oneline HEAD`
- `git push -u origin codex/pit-universe-validation`
- `.venv\Scripts\python.exe scripts\data\export_sp500_joiner_leaver_pit.py --start 2016-01-01 --end 2026-05-13 --index-ric .SPX --chain-ric 0#.SPX --output-dir data\raw\constituents`
- `.venv\Scripts\python.exe scripts\gen_pit_universe_validation_nb.py`
- `.venv\Scripts\python.exe -m pytest tests\test_pit_universe_validation_notebook.py tests\test_sp500_joiner_leaver_pit_export.py -v`
- `.venv\Scripts\ruff.exe check scripts\gen_pit_universe_validation_nb.py scripts\data\export_sp500_joiner_leaver_pit.py tests\test_pit_universe_validation_notebook.py tests\test_sp500_joiner_leaver_pit_export.py`
- one-epoch `run_experiment.py` PIT smoke with `data=temporal_2019`, `+data.use_pit_universe=true`, `+data.filter_stocks_per_split=true`, `training.num_epochs=1`, `training.num_models=1`.

## Data/Experiment State

- Drive PIT CSV URL:
  `https://drive.google.com/file/d/1jNTr3TlRJfPI-eenRryS-5IUs0V-AYY_/view?usp=drivesdk`
- Colab notebook URL:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_universe_validation_colab.ipynb`
- GitHub branch:
  `https://github.com/magilliam27/MCI-GRU/tree/codex/pit-universe-validation`

## User Preferences

- User wants the work framed around proving MCI-GRU predicts stocks, not necessarily that it predicts the S&P 500 as an index.
- User is rightly skeptical of over-stating PIT. Be precise:
  - anchored historical snapshots are already point-in-time at selection date,
  - PIT is a robustness/audit layer,
  - full rolling PIT requires union price data.

## Do Not Do

- Do not claim the current notebook is a pure rolling S&P 500 PIT backtest.
- Do not overwrite or revert unrelated dirty files.
- Do not push LSEG-derived data to GitHub unless the user explicitly asks again.
- Do not assume Colab has LSEG access; use the Drive PIT CSV.
