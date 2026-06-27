# SP500 GICS Top-10 Multiyear Baseline Handoff

Last updated: 2026-06-22

## Resume Here

- Build/run a new Colab workflow that repeats the reduced PIT GICS top-10 baseline for test years 2022, 2023, and 2024.
- Start from `notebooks/sp500_pit_gics_top10_baseline_colab.ipynb` and `scripts/gen_sp500_pit_gics_top10_baseline_nb.py`, but convert the single 2025 run into a year matrix.
- Do not silently reuse the 2025 split. For each test year `Y`, shift training and validation windows with the test year.
- Before launching full Colab, decide whether the current reduced PIT selector history is sufficient for 2022/2023. The current monthly selector artifact starts at `2018-01-02`, so old-style 2022 and 2023 rolling windows may need an LSEG repull or an explicitly documented shorter train window.

## Current Objective

Measure whether the point-in-time reduced S&P 500 universe, defined as top 10 by market cap within each GICS sector at monthly PIT snapshots, behaves similarly across test years 2022, 2023, and 2024. Isolate the smaller-universe effect by preserving the frozen default recipe and masked-panel PIT mechanics.

## What Changed

- Added a reduced PIT universe/data path in the current checkout:
  - `scripts/data/export_sp500_pit_gics_top10_mcap.py`
  - `tests/test_sp500_pit_gics_top10_mcap_export.py`
  - `scripts/gen_sp500_pit_gics_top10_baseline_nb.py`
  - `notebooks/sp500_pit_gics_top10_baseline_colab.ipynb`
- Pulled and uploaded the current reduced PIT data bundle:
  - Market CSV: `sp500_pit_gics_top10_mcap_monthly_20180102_20260622_lseg_20150101_20260622.csv`
  - PIT universe CSV: `sp500_pit_gics_top10_mcap_monthly_20180102_20260622_pit_universe.csv`
  - Snapshot CSV: `sp500_pit_gics_top10_mcap_monthly_20180102_20260622_snapshots.csv`
- Completed a 2025 baseline run in Colab:
  - Drive folder: `https://drive.google.com/drive/folders/1W1Ykd-gvPXcGQjuunKjx1Upn-Dnsv_mp`
  - Run tag: `20260622_043728`
  - Status: `OK`

## Key Decisions

- Keep the frozen default recipe unchanged:
  - `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`
  - 20 models, 100 epochs, patience 15
  - pure IC loss, raw 5-day return labels, `selection_metric=val_ic`
  - static threshold graph, multi-feature edges, `drop_edge_p=0.1`
  - strict current-only global regime features; `FRED_API_KEY` required
- Keep true PIT/masked-panel mechanics:
  - `data.use_pit_universe=true`
  - `data.pit_universe_mode=masked_panel`
  - `data.pit_breadth_policy=error`
  - Use a reduced-universe-appropriate breadth floor, currently `data.pit_min_scoreable_stocks=100`
- Use the older rolling-year guidance as the intended comparator shape:
  - 2022 test: train `2016-2020`, validate `2021`, test `2022`
  - 2023 test: train `2017-2021`, validate `2022`, test `2023`
  - 2024 test: train `2018-2022`, validate `2023`, test `2024`
- The current reduced PIT selector starts at 2018. Do not claim fully matched 2022/2023 rolling-window comparability unless selector snapshots are extended earlier or the shorter train history is deliberately chosen and labeled.
- If extending selector history, use LSEG Workspace and keep the same PIT top-10 by market-cap logic. Watch GICS sector taxonomy before 2018, especially Real Estate sector availability.

## Important Files

- `docs/DEFAULT_EXPERIMENT_RECIPE.md` - frozen default recipe.
- `docs/handoffs/2026-05-03-rolling-temporal-backtest-notebook.md` - prior rolling-window guidance for 2022/2023/2024 test years.
- `scripts/data/export_sp500_pit_gics_top10_mcap.py` - reduced PIT selector and market data exporter.
- `scripts/gen_sp500_pit_gics_top10_baseline_nb.py` - current single-year Colab generator to adapt.
- `notebooks/sp500_pit_gics_top10_baseline_colab.ipynb` - current single-year notebook.
- `tests/test_sp500_pit_gics_top10_mcap_export.py` - PIT selector unit tests.

## Verification

- Local tests already passed for the reduced PIT exporter:
  - `.\.venv\Scripts\python.exe -m pytest tests\test_sp500_pit_gics_top10_mcap_export.py tests\test_sp500_gics_top10_mcap_export.py -v --basetemp .tmp_pytest\pytest -p no:cacheprovider`
  - Result observed: `6 passed`
- Ruff already passed for the new exporter/generator/test files.
- Local data audit already passed:
  - 103 snapshot dates
  - min/max selected per snapshot: 110/110
  - bad sector cells: 0
  - PIT interval rows: 473
  - PIT union: 197 names
  - market rows: 547,235
  - market date range: `2015-01-02` to `2026-06-18`
- 2025 Colab run completed on non-T4 GPU:
  - GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
  - Training: 20 models
  - Mean best validation IC: `0.0337335076`
  - Test avg IC: `0.0321129471`
  - Test avg rank IC: `0.0448513563`
  - PIT daily top-10 no-transaction-cost backtest total return: `26.59%`
  - Benchmark return: `15.75%`
  - Excess return: `10.83%`
  - ASR: `0.913`
  - Max drawdown: `-25.50%`
- Not yet run:
  - Reduced PIT 2022, 2023, or 2024 full-budget Colab runs.

## Open Risks

- The existing reduced PIT artifact may not support old-style 2022/2023 rolling-window comparability because selector snapshots begin in 2018.
- Extending the selector earlier may hit GICS taxonomy differences. If 11-sector top-10 cannot be formed before Real Estate exists as a sector, document the choice rather than forcing a fake 110-name panel.
- Colab needs a G4/L4-class runtime or better. The 2025 notebook correctly refused T4.
- `FRED_API_KEY` secret access must be granted in Colab for strict current-only regime features.
- DriveFS listing can lag; use heartbeat and summary files as truth.

## Next Actions

1. Inspect the current reduced PIT artifacts and decide whether 2022/2023 require an LSEG selector repull to preserve the intended rolling windows.
2. Adapt `scripts/gen_sp500_pit_gics_top10_baseline_nb.py` into a matrix notebook for years `[2022, 2023, 2024]`.
3. For each year, use shifted train/validation/test windows and keep all recipe/data/PIT overrides fixed except the split dates and output names.
4. Run the notebook in visible Colab on a non-T4 GPU, with Drive-backed heartbeat and per-year summaries.
5. Produce a concise comparison table against the completed 2025 reduced PIT baseline and, where appropriate, the prior full PIT masked-panel 2022-2025 report.

## Do Not Do

- Do not run a static current 110-name universe.
- Do not disable masked-panel PIT to make older years easier.
- Do not compare 2022/2023/2024 as if only `test_start` changed; train and validation windows must move too.
- Do not silently weaken the frozen default recipe or switch off strict regime features to avoid the FRED secret.
- Do not claim apples-to-apples full-universe comparison if the train history is shorter because the reduced selector starts later.
