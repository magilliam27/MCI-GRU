# Issue 28 Cost Rank-Gate Backtests Handoff

Last updated: 2026-05-16

## Resume Here

- Start with GitHub issue #28:
  `https://github.com/magilliam27/MCI-GRU/issues/28`.
- The implementation support is already pushed in commit `bd0d7d6` on branch
  `codex/pit-universe-validation`; the issue remains open because the actual
  saved-prediction backtest has not been run.
- Exact next action: make the full run artifact available as a local/mounted
  path, then run `scripts/run_pit_saved_prediction_backtests.py` against
  `MCI-GRU-Ablations/pit_masked_panel_2022_2025/20260514_043539`.
- If using Google Drive Desktop on Windows, the expected shape is:

```powershell
.\.venv\Scripts\python.exe scripts\run_pit_saved_prediction_backtests.py `
  --run-root "G:\My Drive\MCI-GRU-Ablations\pit_masked_panel_2022_2025\20260514_043539" `
  --data-file data\raw\market\sp500_pit_union_lseg_20150101_20260513.csv `
  --pit-universe-csv data\raw\constituents\sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv `
  --spread 10 `
  --slippage 5 `
  --min-rank-drop 30 `
  --fail-fast
```

If the Drive root is not mounted locally, download/extract the Drive run folder
or zip first; do not retrain the models.

## Current Objective

Re-evaluate the saved `20260514_043539` PIT masked-panel predictions with
transaction costs and the normal rank-drop gate enabled. The output should turn
issue #28 from "runner implemented" into "cost-aware evidence reviewed."

## What Changed

- Added `scripts/run_pit_saved_prediction_backtests.py`.
  - Reuses yearly `averaged_predictions`; it never calls training code.
  - Invokes `tests/backtest_sp500_daily.py` with `--transaction_costs`,
    `--enable_rank_drop_gate`, and `--min_rank_drop`.
  - Defaults to rank-drop threshold `30`, spread `10` bps, slippage `5` bps,
    `top_k=10`, `label_t=5`, and `adjustment_method=bhy`.
  - Resolves stale Colab `/content/...` paths from `training_results.csv` back
    under the mounted `--run-root`.
  - Writes yearly result copies and combined summaries under
    `RUN_ROOT/summaries/pit_saved_prediction_cost_rank_gate/`.
- Added `tests/test_pit_saved_prediction_backtests.py`.
- Pushed in commit `bd0d7d6 Add PIT masked-panel follow-up audits`.
- Commented on issue #28 that implementation support landed, but the issue
  remains open until the actual mounted-Drive backtest completes.

## Key Decisions

- Do not retrain. This issue is explicitly about saved prediction
  re-evaluation.
- Keep #28 open until all four yearly saved predictions have actually been
  backtested with costs and rank gate, and the summary outputs are reviewed.
- Use `--min-rank-drop 30` as the normal rank-drop gate setting because that is
  the paper-trade style threshold called out in the issue follow-up.
- Use local repo data files for market/PIT universe unless the Drive manifest
  points to an accessible equivalent.
- `--fail-fast` is recommended for the first real execution so missing Drive
  folders or malformed paths stop early.

## Important Files

- `scripts/run_pit_saved_prediction_backtests.py`: orchestration CLI to run the
  cost-aware/rank-gated saved-prediction backtests.
- `tests/test_pit_saved_prediction_backtests.py`: focused regression tests for
  command construction, stale path remapping, and summary comparison output.
- `tests/backtest_sp500_daily.py`: underlying PIT-aware daily backtest script;
  already supports `--transaction_costs`, `--spread`, `--slippage`,
  `--enable_rank_drop_gate`, and `--min_rank_drop`.
- `docs/PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md`: explains
  why the no-cost/no-gate backtest is not enough for promotion.
- Drive run folder:
  `MCI-GRU-Ablations/pit_masked_panel_2022_2025/20260514_043539`.
- Expected run-root files:
  - `pit_masked_panel_manifest.json`
  - `summaries/training_results.csv`
  - `summaries/backtest_results.csv`
  - `training_runs/pit_true_rolling_2022/<timestamp>/averaged_predictions/`
  - same layout for 2023, 2024, 2025

## Verification

Fresh checks run for this handoff:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pit_saved_prediction_backtests.py -v -o cache_dir=.codex_tmp\pytest-cache --basetemp .codex_tmp\pytest-tmp
```

Result: `3 passed`.

```powershell
.\.venv\Scripts\ruff.exe check scripts\run_pit_saved_prediction_backtests.py tests\test_pit_saved_prediction_backtests.py
```

Result: `All checks passed!`.

```powershell
.\.venv\Scripts\python.exe scripts\run_pit_saved_prediction_backtests.py --help
```

Result: exit `0`; help text shows the required `--run-root`, `--spread`,
`--slippage`, `--min-rank-drop`, `--dry-run`, and `--fail-fast` options.

Previously, before commit `bd0d7d6`, the full PIT follow-up focused suite also
passed: 12 tests across notebook summary, membership audit, saved-prediction
runner, and alias audit.

Checks not run:

- The actual mounted-Drive cost-aware/rank-gated backtest has not been run.
- No resulting `cost_rank_gate_yearly_backtest_results.csv` has been reviewed.
- Issue #28 has not been closed.

## Open Risks

- Local Drive mount may not exist at `G:\My Drive\...`; if so, the run root must
  be downloaded or mounted first.
- The Drive run root may contain Colab paths in `training_results.csv`; the
  runner is designed to remap them, but real execution is the proof.
- The backtest can be slow or generate large outputs because it replays all
  saved daily prediction files for four yearly windows.
- Cost assumptions (`spread=10`, `slippage=5`) are defaults, not a market
  microstructure audit.
- If the cost-aware results degrade materially, follow-on issue #29 pooled
  significance and #30 2022 drilldown should use the cost-aware outputs rather
  than the old no-cost/no-gate results.

## Next Actions

1. Locate or mount the saved run root for
   `MCI-GRU-Ablations/pit_masked_panel_2022_2025/20260514_043539`.
2. Run a dry-run first to verify resolved yearly prediction folders:

```powershell
.\.venv\Scripts\python.exe scripts\run_pit_saved_prediction_backtests.py `
  --run-root "<mounted-or-downloaded-run-root>" `
  --data-file data\raw\market\sp500_pit_union_lseg_20150101_20260513.csv `
  --pit-universe-csv data\raw\constituents\sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv `
  --dry-run
```

3. Run the real command with `--spread 10 --slippage 5 --min-rank-drop 30
   --fail-fast`.
4. Inspect the generated files under
   `RUN_ROOT/summaries/pit_saved_prediction_cost_rank_gate/`:
   - `cost_rank_gate_yearly_backtest_results.csv`
   - `cost_rank_gate_vs_reviewed_side_by_side.csv`
   - `cost_rank_gate_2022_2025_summary.csv`
   - `cost_rank_gate_reproducibility.md`
5. Update issue #28 with the result summary. Close #28 only if all acceptance
   criteria are met; otherwise leave it open with the blocker.

## Commands Run

- `git status --short --branch`
- `git show --name-status --oneline --no-renames bd0d7d6`
- `Get-Content -TotalCount 260 scripts\run_pit_saved_prediction_backtests.py`
- `Get-Content -Raw tests\test_pit_saved_prediction_backtests.py`
- `gh issue view 28 --repo magilliam27/MCI-GRU --comments`
- `gh issue view 28 --repo magilliam27/MCI-GRU --json number,title,state,url,body,comments`
- `.\.venv\Scripts\python.exe -m pytest tests\test_pit_saved_prediction_backtests.py -v -o cache_dir=.codex_tmp\pytest-cache --basetemp .codex_tmp\pytest-tmp`
- `.\.venv\Scripts\ruff.exe check scripts\run_pit_saved_prediction_backtests.py tests\test_pit_saved_prediction_backtests.py`
- `.\.venv\Scripts\python.exe scripts\run_pit_saved_prediction_backtests.py --help`

## Do Not Do

- Do not retrain models for issue #28.
- Do not close #28 just because the runner exists; close only after real
  cost-aware/rank-gated outputs are generated and reviewed.
- Do not stage or revert unrelated dirty files currently in the worktree:
  `docs/REGIME_DATA_CONTRACT.md`, `scripts/colab_regime_reconcile.py`,
  `skills/research-paper-to-mci-gru/*`, and `tests/test_regime_features.py`.
- Do not treat the old no-cost/no-gate backtest as promotion evidence.

## References

- Issue #28:
  `https://github.com/magilliam27/MCI-GRU/issues/28`
- Parent issue #25:
  `https://github.com/magilliam27/MCI-GRU/issues/25`
- Pushed implementation commit:
  `bd0d7d6 Add PIT masked-panel follow-up audits`
- Branch:
  `codex/pit-universe-validation`
