# Reduced S&P 500 PIT GICS Top-10 Multiyear Baseline

Date prepared: 2026-06-23

Primary run:
[MCI-GRU-Ablations/sp500_gics_top10_baseline_multiyear/20260623_011810](https://drive.google.com/drive/folders/1KYDRL0npEvBuYyN8Vp-5IBas0ipAvJ3X)

Notebook:
[sp500_pit_gics_top10_baseline_multiyear_20160104_colab.ipynb](https://drive.google.com/file/d/1BeGtuY5CtIXChFJTSk_5KVbRWv6MX5A5/view)

Canonical issue:
[GitHub issue #43](https://github.com/magilliam27/MCI-GRU/issues/43)

## Executive Summary

The reduced S&P 500 point-in-time GICS top-10 universe was extended back to
2016-01-04 and run across 2022, 2023, and 2024 with the frozen default MCI-GRU
recipe. The run completed on visible Colab with a G4-class runtime and Drive
artifacts as truth.

The result is mixed. The reduced 110-active-name universe held up strongly in
2023 and 2024, but 2022 was a hard negative out-of-sample year. This means the
smaller PIT universe is promising but not uniformly robust across all requested
test years.

## Run Identity

| Field | Value |
| --- | --- |
| Run tag | `20260623_011810` |
| Runtime | `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| Recipe | `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1` |
| PIT mode | `masked_panel` |
| Loss | pure IC |
| Labels | raw 5-day returns |
| Selection metric | `val_ic` |
| Ensemble | 20 models |
| Epoch budget | 100 epochs, patience 15 |
| Backtest | PIT daily top-10, no transaction costs, no rank-drop gate |
| Volatility targeting | disabled |

## Data Audit

The selector-history blocker was resolved by extending the PIT selector from
2018-01-02 back to 2016-01-04.

| Check | Value |
| --- | ---: |
| Selector range | `2016-01-04` to `2026-06-22` |
| Monthly snapshots | 127 |
| Selected names per snapshot | 110 min / 110 max |
| PIT union names | 205 |
| PIT intervals | 541 |
| Missing market identifiers | 0 |
| Market rows | 565,871 |
| Market date range | `2015-01-02` to `2026-06-18` |

The merged repo tracks only scripts, tests, notebook source, handoff, and small
metadata JSON files. Large market, snapshot, and PIT CSV artifacts remain
gitignored and live in Drive / local data storage.

## Results

| Year | Mean Best Val IC | Test Avg IC | Test Rank IC | Top-10 No-TC Total | Benchmark | Excess | ASR | MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.0300 | -0.0583 | -0.0477 | -33.12% | -11.27% | -21.85% | -0.905 | -37.16% |
| 2023 | -0.0020 | 0.0558 | 0.0555 | 41.66% | 11.85% | 29.81% | 2.163 | -11.60% |
| 2024 | 0.0383 | 0.0385 | 0.0370 | 37.19% | 14.68% | 22.50% | 1.782 | -17.33% |

## Interpretation

The 2023 and 2024 rows support the smaller-universe hypothesis: the reduced PIT
candidate set can preserve strong out-of-sample behavior under the frozen
default recipe. The 2022 row prevents a blanket robustness claim. It is weak in
both IC and realized portfolio performance, despite a reasonable validation IC.

The next useful analysis is not another broad rerun. It is a 2022-specific
failure drilldown against the full PIT masked-panel result: month, sector,
turnover, concentration, and market-regime slices. Cost-aware and rank-gated
saved-prediction replays should also be run before making any live viability
claim.

## Caveats

- Backtests are no-transaction-cost and no-rank-gate.
- Year-level statistical significance was not promoted as a deployment claim.
- The run used a reduced 110-name active selector, not the full S&P 500 PIT
  masked-panel universe.
- The selector uses contemporaneous S&P 500 membership and top 10 by GICS
  sector market cap; do not replace it with a static current-name list.
- `data.pit_universe_mode=masked_panel` remains required so the PIT union axis
  and daily masks are preserved.

## Verification

Fresh local verification in the run thread passed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sp500_pit_gics_top10_mcap_export.py tests\test_sp500_gics_top10_mcap_export.py tests\test_sp500_pit_gics_top10_baseline_notebook.py -v --basetemp .tmp_pytest\pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m py_compile scripts\gen_sp500_pit_gics_top10_baseline_nb.py
.\.venv\Scripts\ruff.exe check scripts\data\export_sp500_gics_top10_mcap.py scripts\data\export_sp500_pit_gics_top10_mcap.py scripts\gen_sp500_pit_gics_top10_baseline_nb.py tests\test_sp500_gics_top10_mcap_export.py tests\test_sp500_pit_gics_top10_mcap_export.py tests\test_sp500_pit_gics_top10_baseline_notebook.py
```

Observed result: 10 focused tests passed, `py_compile` exited 0, and ruff
reported all checks passed.
