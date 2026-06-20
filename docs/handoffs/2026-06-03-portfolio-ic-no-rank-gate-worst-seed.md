# Portfolio-IC Worst-Seed No-Rank-Gate Replay Handoff

## Resume Here

Run a quick saved-prediction backtest for the worst observed Portfolio-IC sweep seed with the rank-drop gate disabled.

Primary target from the fetched row-level results:

- Variant: `portfolio_ic_weight100`
- Year: `2025`
- Seed: `161803`
- Existing gated run: `portfolio_ic_weight100_2025_seed161803`
- Existing rank-gate metrics observed in the full-row CSV fetch:
  - `backtest.total_return = -0.1007984763325774`
  - `backtest.ARR = -0.524665607120405`
  - `backtest.ASR = -1.3828402117889516`
  - `backtest.MDD = -0.1605533005055988`
  - `backtest.num_trading_days = 36`
  - `days_with_gate_exits = 3`
  - `total_trades = 16`
  - `total_transaction_cost = 0.0016`

Important caveat: the row-level CSV fetch was large and tool output truncated, so treat this as the worst row observed in the fetched output. First action should be to parse the full row CSV programmatically and confirm whether this is the true worst row, true worst seed aggregate, or just the worst short-window slice.

## Current Objective

Determine whether the rank-drop gate materially caused or worsened the worst observed seed result by rerunning the same saved predictions with only the rank gate removed.

This is a backtest replay only. Do not train models for this check.

## What Changed

- The `.75` and `1.0` Portfolio-IC saved-prediction backtests completed after earlier Colab runner issues.
- The completed sweep report showed that higher Portfolio-IC weights reduced turnover/cost drag but did not beat the pure-IC baseline on aggregate.
- A GitHub issue was created to document Colab reliability findings: https://github.com/magilliam27/MCI-GRU/issues/36
- This handoff was added to direct the next model toward a narrow no-rank-gate diagnostic rather than another full sweep.

## Key Decisions

- Keep every backtest setting fixed except the rank gate.
- Use saved predictions only; no model training and no GPU requirement for this replay.
- Because this is CPU/pandas replay, the prior hard G4/non-T4 gate is not relevant unless the next agent unexpectedly starts training.
- Write no-rank-gate outputs to a new suffix such as `_pit_daily_tc_no_rank_gate`; do not overwrite the existing `_pit_daily_tc_rank_gate` outputs.
- Treat Drive artifacts as the durable source of truth. Scratch paths under `/content/...` in old CSV rows are not durable after Colab runtime release.

## Backtest Results We Have

Full sweep aggregate summary from the completed report:

| Variant | Total Return | ARR | ASR | MDD | Turnover | Total Cost | Cost Drag ARR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pure_ic_baseline` | 17.63% | 19.19% | 0.741 | -23.17% | 9.98% | 4.71% | 6.41% |
| `portfolio_ic_weight25` | 14.67% | 15.96% | 0.683 | -25.52% | 4.53% | 2.14% | 2.78% |
| `portfolio_ic_weight50` | 15.30% | 16.62% | 0.710 | -26.33% | 3.40% | 1.60% | 2.09% |
| `portfolio_ic_weight75` | 14.09% | 15.38% | 0.701 | -27.07% | 2.52% | 1.19% | 1.67% |
| `portfolio_ic_weight100` | 12.64% | 10.35% | 0.620 | -25.19% | 2.73% | 1.21% | 1.81% |

Useful year-level context:

- `portfolio_ic_weight100` 2022 was weak: total return about `-26.78%`, ASR about `-0.706`, MDD about `-38.25%`.
- `portfolio_ic_weight75` 2022 was also weak: total return about `-30.89%`, ASR about `-0.822`, MDD about `-40.36%`.
- `portfolio_ic_weight100` 2025 seed `161803` is the worst observed row in the fetched CSV, but it only had `36` trading days. Verify whether that is a data/prediction coverage artifact before over-interpreting the seed.

## Durable Result Artifacts

- Completed backtest heartbeat: https://drive.google.com/file/d/1B2bhh68RN6sO6rpsu75plvlYAACFBXnl/view
- Full sweep report folder: https://drive.google.com/drive/folders/13-recBHH2zIEgLKixTOsJ3XhDH9toRiS
- Markdown report: https://drive.google.com/file/d/1O01Sl-X_84oORdod0TPhTgv6JfMAmh7i/view
- Full row CSV: https://drive.google.com/file/d/1114JfSchprfqLuqkD0PqbOLRhYGqLFWd/view
- Full by-variant CSV: https://drive.google.com/file/d/1dcs0xE8mGZA2QA1yxbagO_7ur5i59Yu-/view
- Full by-year CSV: https://drive.google.com/file/d/1pVZAVMxWnEJb3hpTTbkVDlbToWJVVNr_/view
- New `.75`/`1.0` output folder: https://drive.google.com/drive/folders/1ryawwcHkuyr7kYL4yMf7I0vJZ-KR3qI5
- New `.75`/`1.0` result CSV: https://drive.google.com/file/d/1UEF5El-YrpgXy9cDoMS3stVgGxsaaaya/view

## Important Files

- `tests/backtest_sp500_daily.py` - direct single-run backtest entrypoint.
- `scripts/run_pit_saved_prediction_backtests.py` - batch saved-prediction runner; note that its current command path enabled the rank gate for the completed sweep, so the quick diagnostic is easiest as a direct `tests/backtest_sp500_daily.py` call unless you add a no-rank-gate flag.
- `docs/handoffs/2026-05-31-portfolio-ic-weight-sweep.md` - original experiment handoff.
- `docs/handoffs/2026-06-01-colab-portfolio-ic-runner-issues.md` - Colab runner failure diagnosis and operational notes.
- `docs/workflows/COLAB_PLAYWRIGHT_MCP_GUIDE.md` - current Colab control-plane guidance.
- `docs/NOTEBOOK_BEST_PRACTICES.md` - notebook lifecycle and cleanup guidance.

## Suggested No-Rank-Gate Command

Use the same data and prediction files, but omit `--enable_rank_drop_gate` and omit `--min_rank_drop`.

```bash
python -X utf8 tests/backtest_sp500_daily.py \
  --predictions_dir <RESTAGED_PREDICTIONS_DIR_FOR_portfolio_ic_weight100_2025_seed161803> \
  --data_file data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv \
  --pit_universe_csv data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv \
  --test_start 2025-01-22 \
  --test_end 2025-12-31 \
  --top_k 10 \
  --label_t 5 \
  --num_tests 1 \
  --adjustment_method bhy \
  --auto_save \
  --backtest_suffix _pit_daily_tc_no_rank_gate \
  --transaction_costs \
  --spread 10 \
  --slippage 5
```

The gated command in the CSV used:

```bash
--backtest_suffix _pit_daily_tc_rank_gate --transaction_costs --spread 10 --slippage 5 --enable_rank_drop_gate --min_rank_drop 30
```

## Verification

- `git status --short` was checked before writing this handoff. The repository is dirty with many pre-existing tracked and untracked changes, including prior docs, configs, scripts, tests, and handoffs. Do not assume those changes belong to this handoff.
- The row-level result CSV was fetched from Drive, but the command output was truncated because the CSV is large. The primary target above is based on the worst visible row, not a completed local parse.
- No backtest was run as part of this handoff.
- No tests were run for this docs-only handoff.

## Open Risks

- The phrase "worst seed" could mean worst single row, worst seed aggregate across years, or worst full-year stress case. Confirm by parsing the full row CSV first.
- The observed `2025` worst row only has `36` trading days. If that is a coverage artifact, use a full-year stress comparator instead, likely `portfolio_ic_weight100_2022_seed271828` or `portfolio_ic_weight75_2022_seed161803`.
- Old CSV paths such as `/content/portfolio_ic_api_run/.../averaged_predictions` are Colab scratch paths and may no longer exist. Restage predictions from Drive before running.
- If DriveFS is flaky in Colab, use Drive API download/listing fallback rather than repeatedly remounting Drive.

## Next Actions

1. Fetch and parse the full row CSV from Drive to confirm the actual worst row and worst seed aggregate.
2. Restage saved predictions for `portfolio_ic_weight100_2025_seed161803` from the Drive-backed training run artifacts.
3. Run the direct no-rank-gate backtest command above with a new suffix.
4. Compare gated vs no-gate metrics: total return, ARR, ASR, MDD, trading days, turnover, total cost, trades, and gate-exit days.
5. If the 2025 short-window row is not a fair diagnostic, repeat the same no-rank-gate replay on a full-year weak case such as `portfolio_ic_weight100_2022_seed271828`.
