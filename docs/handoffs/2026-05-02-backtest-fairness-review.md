# Backtest Fairness Review Handoff

Last updated: 2026-05-02

## Resume Here

- Start by deciding whether to implement fixes or run validation experiments.
- First actionable fix: address `mci_gru/data/data_manager.py` full-period complete-stock filtering, because it is the highest-risk fairness issue.
- First validation run: freeze the May 1, 2026 selected candidate and rerun with a point-in-time or date-aware universe, then compare the same top-k/rank-drop settings.
- Re-check the latest Google Drive folder before using Drive results as current evidence: `recommended_backtests/20260501_235210`.

## Current Objective

The user wants to validate whether strong backtest returns are fair and live-edge-relevant, especially for the recent recommended backtests in Google Drive. The recent returns are promising, but the current review found material caveats around universe construction, horizon alignment, and selection/multiple-testing interpretation.

## What Changed

- Added this handoff file only.
- No source code changes were made during the review.
- No commits or pushes were made.

## Key Decisions

- Treat the latest Drive backtest as exploratory, not final proof of robustness.
- The old execution-timing concern appears materially improved: current backtests use prediction date to next trading day and evaluate portfolio and benchmark with `open_to_open_return`.
- The biggest fairness concern is not the open-to-open simulator; it is the default universe filter requiring complete data from `train_start` through `test_end`.
- The reported `haircutted_sharpe=0` does not mean zero realized return. It means BHY multiple-testing adjustment pushed adjusted p-value to `1.0`, which converts back to adjusted t-stat `0` and then Sharpe `0`.
- Additional untouched out-of-sample years can improve the haircutted Sharpe only if the frozen strategy continues to perform and those years are not used for tuning.

## Important Files

- `mci_gru/data/data_manager.py` - `filter_complete_stocks` requires complete coverage through `test_end`; main survivorship/future-completeness risk.
- `mci_gru/config.py` - `DataConfig.filter_stocks_per_split=False` and `use_pit_universe=False` by default; split embargo validation exists in `_validate_embargo`.
- `mci_gru/pipeline.py` - applies optional PIT universe filter and chooses full-period vs per-split stock filtering.
- `mci_gru/data/preprocessing.py` - `compute_labels` fills unavailable future-return labels near split ends.
- `tests/backtest_sp500.py` - current backtest implementation, multiple-testing haircut code, open-to-open simulator, rank-drop gate, and result writer.
- `tests/backtest_sp500_daily.py` - duplicate/near-duplicate backtest script; risk of divergence from `tests/backtest_sp500.py`.
- `docs/BACKTEST_FAIRNESS_AUDIT.md` - old audit documenting prior close-to-close / trade timing issue.
- `docs/ARCHITECTURE.md` - data flow, graph, training, evaluation, and no-lookahead invariants.

## Verification

- Ran `git status --short`; it returned no changed-file entries but emitted permission warnings for:
  - `C:\Users\magil/.config/git/ignore`
  - `.pytest_cache/`
  - `tmp/tmpylp40fy0/`
- Ran `git rev-list --left-right --count HEAD...origin/main`; result was `0 0` before this handoff was created.
- Attempted targeted pytest verification:
  - `python -m pytest tests/test_backtest_fairness.py tests/test_evaluation_portfolio.py tests/test_dynamic_graph_updates.py -q`
  - Failed because default Python 3.13 has no `pytest`.
  - `.venv\Scripts\python.exe -m pytest ...` collected zero tests due missing `numpy`.
  - `lseg_env\Scripts\python.exe -m pytest ...` failed because the environment points to a missing Python 3.11 executable.
- Manual evidence inspected:
  - Latest Google Drive run `recommended_backtests/20260501_235210`.
  - `backtest_comparison_interim.csv`.
  - `selected_candidates.csv`.
  - Top-10 summary, daily returns, return attribution, and trade journal artifacts.
  - Local files listed in "Important Files".

## Open Risks

- Default complete-stock filtering likely leaks future availability into universe selection unless PIT filtering or date-aware masks are enabled.
- Recent Drive run uses `label_t=5` but `holding_period=1`; the result is a daily open-to-open strategy driven by a 5-day rank signal, not a direct 5-day holding-period test.
- End-of-window label filling can affect validation/test selection metrics near split ends.
- The latest Drive artifact reports strong raw performance but `BHY adjusted p=1.0`, `haircutted_sharpe=0`, and not significant after multiple-testing adjustment.
- Test verification is currently blocked by broken or incomplete Python environments.
- It is unclear whether `num_tests=15` fully captures all model/config/manual selection attempts. If more variants were explored, the multiple-testing penalty should be larger.

## Next Actions

1. Implement or configure a point-in-time/date-aware universe validation path and rerun the May 1 selected candidate unchanged.
2. Run sensitivity backtests with `holding_period=5` using staggered and block rebalance styles for the same frozen predictions/config.
3. Change validation/test label handling so rows without a full future horizon are masked/excluded rather than mean/zero-filled.
4. Consolidate `tests/backtest_sp500.py` and `tests/backtest_sp500_daily.py` into one backtest engine outside `tests/`, with thin CLI/test wrappers.
5. Repair the local Python environment and rerun targeted tests before claiming any backtest fairness fix passes.

## Data/Experiment State

- Latest reviewed Drive folder: `recommended_backtests/20260501_235210`.
- Selected candidate name: `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`.
- Selected candidate characteristics:
  - static threshold graph
  - pure IC loss
  - raw 5-day return labels
  - selected by validation IC
  - regime current only
  - edge dropout `p=0.1`
  - seed `42`
  - `label_t=5`
  - `num_tests=15`
- Latest reviewed Drive top-10 result:
  - ARR about `77.2%`
  - ASR about `2.19`
  - BHY adjusted p-value `1.0`
  - haircutted Sharpe `0.0`
  - statistical significance: no
- Latest reviewed Drive top-20 result:
  - ARR about `65.1%`
  - ASR about `1.99`
  - BHY adjusted p-value `1.0`
  - haircutted Sharpe `0.0`
  - statistical significance: no

## Do Not Do

- Do not interpret `haircutted_sharpe=0` as zero realized return.
- Do not treat `results/`, `outputs/`, `*.pt`, or `*.pth` as source of truth without checking how they were produced.
- Do not change the selected strategy while using additional years as an untouched holdout.
- Do not import `GraphBuilder` into `paper_trade/`; paper-trade inference uses frozen `graph_data.pt`.

## References

- Google Drive latest reviewed run: `recommended_backtests/20260501_235210`
- Google Drive top-10 summary file: `k10_spread5_slip0_rankdrop30/summary.txt`
- Google Drive trade journal file: `k10_spread5_slip0_rankdrop30/trade_journal.csv`
- Local audit: `docs/BACKTEST_FAIRNESS_AUDIT.md`
- Repo guide: `AGENTS.md`
