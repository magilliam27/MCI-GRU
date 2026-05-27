# True PIT Masked Panel Handoff

Last updated: 2026-05-13

## Resume Here

- Start by reviewing `mci_gru/pipeline.py`, `mci_gru/data/pit.py`, and `tests/test_pit_masked_panel.py`.
- The implementation is functionally in place and tests pass. The next best action is a real data smoke run with one of the new presets, likely `+experiment=pit_temporal_2022 training.num_epochs=1 training.num_models=1 tracking.enabled=false`.
- Use `--basetemp .codex_tmp\pytest-tmp` for pytest in this workspace; the default Windows temp/cache locations produced permission warnings/errors.

## Current Objective

Implement a true rolling PIT S&P 500 panel so the model scores the real-world date-varying opportunity set, normally 450+ scoreable candidates per post-warmup date, instead of collapsing to continuous members.

The core design is a fixed PIT union `kdcode_list` plus daily masks for active membership, feature readiness, loss eligibility, and tradable candidates.

## What Changed

- Added `mci_gru/data/pit.py` with PIT interval normalization, active membership masks, feature-ready masks, label availability masks, combined mask building, breadth diagnostics, label masking, and edge filtering.
- Added `data.pit_universe_mode`, `data.pit_min_scoreable_stocks`, and `data.pit_breadth_policy` to `DataConfig`.
- Updated `prepare_data()` so `data.pit_universe_mode=masked_panel`:
  - loads Joiner/Leaver PIT intervals,
  - keeps the broad PIT union axis,
  - bypasses complete-stock filtering,
  - computes masks after tensor construction,
  - masks labels with `NaN`,
  - supports PIT rank labels over same-day valid names,
  - writes `pit_breadth` metadata,
  - errors or warns when scoreable candidate count drops below threshold.
- Updated `CombinedDataset`, `combined_collate_fn`, and `create_data_loaders()` to carry optional `stock_mask` while preserving the 9-tuple collate contract.
- Updated `StockPredictionModel` and `SelfAttention` so inactive nodes are zeroed and cannot affect graph/self-attention outputs.
- Updated training, metrics, and prediction export:
  - `MaskedMSELoss`,
  - NaN-aware IC and mean IC,
  - NaN-aware evaluation metrics,
  - `prediction_rows_for_date()`,
  - PIT-tradable filtering for per-model and averaged prediction CSVs.
- Updated both backtest scripts to optionally load a PIT universe CSV and filter prediction candidates plus equal-weight benchmark rows to PIT-active names.
- Added experiment presets:
  - `configs/experiment/pit_temporal_2022.yaml`
  - `configs/experiment/pit_temporal_2023.yaml`
  - `configs/experiment/pit_temporal_2024.yaml`
  - `configs/experiment/pit_temporal_2025.yaml`
- Updated docs:
  - `docs/ARCHITECTURE.md`
  - `docs/CONFIGURATION_GUIDE.md`
  - `AGENTS.md`

## Key Decisions

- Pre-membership OHLCV is allowed for feature lookbacks because it was public at the time; membership controls whether a name is tradable/loss-eligible.
- PIT masks use `NaN` labels rather than zero-filled labels so existing loss/evaluation paths can ignore invalid stock/date pairs.
- The collate return remains a 9-tuple. PIT metadata is stored inside slot 6 as `{"dates": ..., "stock_mask": ...}` when masks exist.
- Dynamic/static graphs can still be built on the union axis, but collate filters edges per sample so inactive nodes do not send messages.
- `pit_min_scoreable_stocks` defaults to `450` and `pit_breadth_policy` defaults to `error`; synthetic tests set the threshold to `0`.
- Legacy `row_filter` mode remains available for backward compatibility.

## Important Files

- `mci_gru/data/pit.py` - new PIT mask and edge-filter helper module.
- `mci_gru/pipeline.py` - true PIT masked-panel orchestration and breadth audit.
- `mci_gru/data/data_manager.py` - optional stock masks in dataset/collate/loaders.
- `mci_gru/models/mci_gru.py` - model/self-attention mask handling.
- `mci_gru/training/losses.py` - masked MSE and NaN-aware IC.
- `mci_gru/training/trainer.py` - masked forward calls and prediction CSV filtering.
- `mci_gru/data/preprocessing.py` - optional non-filled labels and masked rank labels.
- `mci_gru/training/metrics.py`, `mci_gru/evaluation/portfolio.py` - NaN-aware evaluation/top-k returns.
- `run_experiment.py` - passes masks into loaders/training and records PIT breadth metadata.
- `tests/test_pit_masked_panel.py` - core synthetic PIT coverage.
- `tests/backtest_sp500.py`, `tests/backtest_sp500_daily.py` - optional PIT benchmark/candidate filtering.
- `configs/experiment/pit_temporal_*.yaml` - PIT temporal presets.

## Verification

Commands run and observed passing:

- `.venv\Scripts\python.exe -m pytest tests\test_pit_masked_panel.py -v --basetemp .codex_tmp\pytest-tmp`
  - `11 passed`
- `.venv\Scripts\ruff.exe check mci_gru\data\pit.py mci_gru\data\data_manager.py mci_gru\training\losses.py mci_gru\training\trainer.py mci_gru\data\preprocessing.py mci_gru\evaluation\portfolio.py mci_gru\training\metrics.py mci_gru\pipeline.py run_experiment.py tests\test_pit_masked_panel.py tests\backtest_sp500_daily.py`
  - passed
- `.venv\Scripts\ruff.exe check tests\backtest_sp500.py --ignore E402`
  - passed
- `.venv\Scripts\python.exe -m py_compile mci_gru\data\pit.py mci_gru\data\data_manager.py mci_gru\models\mci_gru.py mci_gru\training\losses.py mci_gru\training\trainer.py mci_gru\data\preprocessing.py mci_gru\evaluation\portfolio.py mci_gru\training\metrics.py mci_gru\pipeline.py run_experiment.py tests\test_pit_masked_panel.py tests\backtest_sp500_daily.py`
  - passed
- `.venv\Scripts\python.exe -m py_compile tests\backtest_sp500.py tests\backtest_sp500_daily.py`
  - passed
- `.venv\Scripts\python.exe -m pytest tests\test_backtest_fairness.py -v --basetemp .codex_tmp\pytest-tmp`
  - `7 passed`
- `.venv\Scripts\python.exe -m pytest tests\ -m "not slow" -v --basetemp .codex_tmp\pytest-tmp`
  - `130 passed, 1 skipped`
- `git diff --check -- <touched files>`
  - passed; only line-ending warnings.

Warnings observed:

- Pytest cache path warnings from `.pytest_cache` permission denial.
- Existing `PytestReturnNotNoneWarning` in several legacy tests.
- Git warning: unable to access `C:\Users\magil/.config/git/ignore`.

## Open Risks

- No real 2022/2023/2024/2025 PIT experiment smoke run has been executed after these code changes.
- The 450+ breadth audit may fail on real CSVs if the input file lacks enough PIT-union OHLCV history or ticker mappings; that failure is intended but needs triage with data evidence.
- Dynamic graphs are mask-filtered at collate time, but graph snapshot construction itself still uses the union axis. This is acceptable for first implementation, but a future improvement could build snapshot correlations only from names active/feature-ready at snapshot date.
- Multi-day holding backtest paths in `tests/backtest_sp500.py` still only pass PIT masks through the one-day `simulate_trading_strategy()` path. The `block` and `staggered` functions were not made PIT-aware.
- GitHub issues from the original issue-draft list have not been created.
- There were unrelated pre-existing dirty files in the worktree before this implementation: regime docs/scripts/tests, research-paper skill files, handoff/research artifacts, and ignored/local data artifacts. Do not revert them casually.

## Next Actions

1. Run a one-epoch real-data smoke:
   `python run_experiment.py +experiment=pit_temporal_2022 training.num_epochs=1 training.num_models=1 tracking.enabled=false`
2. Inspect `run_metadata.json` for `pit_breadth` and confirm normal post-warmup dates have 450+ scoreable names or a clear data explanation.
3. Inspect `averaged_predictions/YYYY-MM-DD.csv` for a few dates to confirm new joiners appear only after `valid_from` and leavers disappear after `valid_to`.
4. Run the PIT-aware backtest with `--pit_universe_csv data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`.
5. Create GitHub issues from the original seven issue drafts if the workflow needs external tracking.

## Commands Run

- `git status --short`
- `git diff --stat -- <relevant touched files>`
- `Get-ChildItem docs\handoffs`
- `Test-Path docs\handoffs\2026-05-13-true-pit-masked-panel.md`
- Verification commands listed above.

## Do Not Do

- Do not restore complete-stock filtering in `masked_panel` mode; that is the failure mode that collapsed the PIT universe to continuous stayers.
- Do not treat `sp500_constituents_2016.csv`, `sp500_constituents_2017.csv`, or `sp500_constituents_2018.csv` as PIT-clean historical membership.
- Do not revert unrelated dirty work in regime docs/scripts/tests or research-paper skill files unless the user explicitly asks.

## References

- Prior handoff: `docs/handoffs/2026-05-13-pit-universe-validation.md`
- PIT universe artifact used by presets: `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
- User target: true rolling PIT S&P 500 panel with 450+ daily candidate breadth.
