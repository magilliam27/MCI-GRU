# PIT Masked-Panel Colab Handoff

Last updated: 2026-05-14

## Resume Here

- The user reports they have now run the full
  `notebooks/pit_masked_panel_2022_2025_colab.ipynb` smoke workflow end to end.
- Treat the `USE_GLOBAL_REGIME` cell-5 issue as resolved; it was fixed and
  pushed on branch `codex/pit-universe-validation` in commit
  `7e3d0a9 Make PIT Colab regime toggle optional`.
- Immediate next move: help the user interpret the Colab smoke artifacts, then
  decide the next run budget. Ask for or inspect the Drive summary files if
  available, especially:
  - `summaries/pit_masked_panel_2022_2025_summary.md`
  - `summaries/pit_breadth_summary.csv`
  - `summaries/prediction_count_checks.csv`
  - `summaries/backtest_results.csv`

## Current Objective

Proceed from completed strict true PIT masked-panel smoke validation into the
next testing phase. The next phase should confirm the smoke artifacts, then run
a more meaningful PIT experiment budget across 2022, 2023, 2024, and 2025.

## What Changed

- Implemented true PIT masked-panel mode with fixed PIT union axis and daily
  masks.
- Added PIT temporal presets:
  - `configs/experiment/pit_temporal_2022.yaml`
  - `configs/experiment/pit_temporal_2023.yaml`
  - `configs/experiment/pit_temporal_2024.yaml`
  - `configs/experiment/pit_temporal_2025.yaml`
- Added a Colab notebook and generator:
  - `notebooks/pit_masked_panel_2022_2025_colab.ipynb`
  - `scripts/gen_pit_masked_panel_2022_2025_nb.py`
- Added PIT system report:
  - `docs/PIT_UNIVERSE_REPORT.md`
- Pushed the work to GitHub branch `codex/pit-universe-validation`.

## Key Decisions

- The notebook defaults to `SMOKE_MODE=True`: 1 epoch, 1 model, strict breadth
  checks. This is intentional to prove data/mask mechanics before long GPU runs.
- The notebook defaults to `USE_GLOBAL_REGIME=False`, so FRED is not required
  for the first PIT smoke. Users can set `USE_GLOBAL_REGIME=True` and provide
  `FRED_API_KEY` later.
- The big market CSV is not in Git. It must live in Google Drive:
  `sp500_pit_union_lseg_20150101_20260513.csv`.
- The PIT membership CSV must also be in Drive:
  `sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`.
- Cell 5 now writes PIT preset YAMLs into the Colab clone before training, so it
  can tolerate a clone that lacks those config files.
- Cell 5 now also defines safe defaults for `USE_GLOBAL_REGIME`,
  `REGIME_STRICT`, and `REGIME_ENFORCE_LAG_DAYS` if the optional regime cell was
  skipped.

## Important Files

- `notebooks/pit_masked_panel_2022_2025_colab.ipynb`: user-facing Colab notebook.
- `scripts/gen_pit_masked_panel_2022_2025_nb.py`: source generator for the notebook.
- `mci_gru/data/pit.py`: PIT interval normalization, masks, label masking, and edge filtering.
- `mci_gru/pipeline.py`: masked-panel orchestration, breadth audit, fixed union axis.
- `mci_gru/data/data_manager.py`: dataset/collate stock-mask transport while preserving the 9-tuple.
- `mci_gru/models/mci_gru.py`: inactive node masking in model/self-attention paths.
- `mci_gru/training/trainer.py`: PIT-tradable prediction export.
- `tests/test_pit_masked_panel.py`: focused regression tests for masked-panel behavior.
- `docs/PIT_UNIVERSE_REPORT.md`: durable explanation of PIT data and validation.

## Verification

Fresh checks run before pushing `bca43e1` and `7e3d0a9`:

- `.venv\Scripts\python.exe -m pytest tests\test_pit_masked_panel.py -v --basetemp .codex_tmp\pytest-tmp`
  - Result: `11 passed`.
- `.venv\Scripts\ruff.exe check ... --ignore UP035,UP008,UP006,UP045,SIM910`
  - Result: `All checks passed`.
  - Reason for ignores: `mci_gru/models/mci_gru.py` has pre-existing modernization
    warnings unrelated to PIT.
- `.venv\Scripts\python.exe -c "...parse notebook code cells..."`
  - Result: `notebook code cells parse OK: 21`.
- `.venv\Scripts\ruff.exe check scripts\gen_pit_masked_panel_2022_2025_nb.py`
  - Result: `All checks passed`.
- User-reported Colab state after this handoff was first written:
  - The full PIT masked-panel notebook completed with `SMOKE_MODE=True`.
  - Exact Colab artifact metrics were not pasted into chat before this handoff
    update, so the next agent should inspect/copy the Drive summary artifacts
    before quoting year-by-year numbers.

Git evidence:

- Branch: `codex/pit-universe-validation`.
- Latest pushed commit:
  - `7e3d0a9 Make PIT Colab regime toggle optional`.
- Main PIT implementation commit:
  - `bca43e1 Add true PIT masked-panel Colab workflow`.

## Open Risks

- The user reports the Colab smoke workflow completed, but the exact output
  tables are not in this chat yet. Do not quote 2022-2025 smoke metrics until
  the Drive summary artifacts are inspected or pasted.
- The next run budget is not decided. Options include a modest validation run
  (`num_models=3-5`, `num_epochs=15-30`) or a full research run
  (`num_models=10`, `num_epochs=100`).
- It is not yet decided whether the next PIT run should stay momentum-only
  (`USE_GLOBAL_REGIME=False`) or enable FRED-backed regime features
  (`USE_GLOBAL_REGIME=True` with `FRED_API_KEY`).
- Large CSV and `.meta.json` data artifacts are not tracked by Git; Drive paths
  must be correct in Colab.
- The working tree still contains unrelated local changes:
  - `docs/REGIME_DATA_CONTRACT.md`
  - `scripts/colab_regime_reconcile.py`
  - `tests/test_regime_features.py`
  - `skills/research-paper-to-mci-gru/*`
  - untracked handoff/research/data sidecars.
  Do not stage or revert these casually.

## Next Actions

1. Ask the user for the Drive output folder or pasted summary from the completed
   smoke run; inspect `pit_breadth_summary.csv`,
   `prediction_count_checks.csv`, and `backtest_results.csv`.
2. Confirm all four years passed strict breadth and prediction-count validation:
   `scoreable_count.below_threshold == 0` and
   `prediction_count_mismatches == 0`.
3. Decide the next budget:
   - fast validation: `SMOKE_MODE=False`, `NUM_MODELS=3-5`,
     `NUM_EPOCHS=15-30`;
   - full PIT research pass: `NUM_MODELS=10`, `NUM_EPOCHS=100`,
     normal early stopping.
4. Decide whether to keep `USE_GLOBAL_REGIME=False` for clean PIT isolation or
   set `USE_GLOBAL_REGIME=True` with `FRED_API_KEY` for the richer recipe.
5. Run the next-budget notebook pass and compare results to the smoke summary;
   do not interpret one-epoch smoke backtest metrics as performance evidence.

## Commands Run

- `git status --short --branch`
- `git log --oneline --decorate -5`
- `git show --stat --oneline HEAD`
- `git show --stat --oneline HEAD~1`
- `.venv\Scripts\python.exe scripts\gen_pit_masked_panel_2022_2025_nb.py`
- `.venv\Scripts\ruff.exe check scripts\gen_pit_masked_panel_2022_2025_nb.py`
- `.venv\Scripts\python.exe -m pytest tests\test_pit_masked_panel.py -v --basetemp .codex_tmp\pytest-tmp`
- `git add ...`
- `git commit -m "Add true PIT masked-panel Colab workflow"`
- `git push origin codex/pit-universe-validation`
- `git commit -m "Make PIT Colab regime toggle optional"`
- `git push origin codex/pit-universe-validation`

## Data/Experiment State

- New local LSEG PIT-union market file:
  `data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv`
  - 1,849,404 rows.
  - 759 resolved identifiers with rows.
  - Dates: 2015-01-02 through 2026-05-13.
  - SHA256 recorded in prior strict smoke:
    `84e1f3f2b79a798246e001e17a372c8daf8bcfc658873ad3d352a99ad993840f`.
- PIT membership file:
  `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`.
- Earlier local strict 2022 smoke passed with:
  - Train scoreable min/median/max: 503/505/507.
  - Val: 504/505/506.
  - Test: 502/503/506.
  - 237 prediction files and 0 prediction-count mismatches.
- User reports the full Colab notebook has now run in smoke mode across all
  years. Exact artifact paths/metrics still need to be inspected before being
  used as evidence.

## User Preferences

- User wants practical Colab instructions and quick unblock fixes, not abstract
  discussion.
- Be precise about what is pushed versus only local.
- Do not overclaim performance from one-epoch smokes.

## Do Not Do

- Do not push or try to commit the large market CSV.
- Do not treat the old anchored universe CSVs as true PIT-clean data.
- Do not re-enable complete-stock/stayer filtering in `masked_panel` mode.
- Do not stage unrelated regime/research-paper dirty files unless the user
  explicitly asks.

## References

- `docs/PIT_UNIVERSE_REPORT.md`
- `docs/handoffs/2026-05-13-true-pit-masked-panel.md`
- `docs/handoffs/2026-05-13-pit-universe-validation.md`
- GitHub branch: `codex/pit-universe-validation`
- Latest pushed commit: `7e3d0a9`
