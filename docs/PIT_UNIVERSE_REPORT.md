# Point-In-Time S&P 500 Universe Report

Last updated: 2026-05-13

## Executive Summary

The MCI-GRU point-in-time (PIT) universe system is designed to remove the two
main universe-contamination objections in stock-level S&P 500 backtests:

1. Survivorship bias from using only names that survived to a later date.
2. Future membership leakage from allowing a stock to be scored before it had
   joined the S&P 500, or after it had left.

The current implementation uses a fixed union stock axis plus daily masks. The
model can keep historical price history for feature lookbacks, but the loss,
prediction export, portfolio candidate set, and PIT-aware backtest only use the
stocks that were active S&P 500 members on each date.

On 2026-05-13, we pulled a new LSEG PIT-union market data file through LSEG
Workspace and verified that it passes the strict masked-panel breadth checks
that the older anchored data failed.

## Core Definition

For any date `D`, the PIT S&P 500 universe is:

```text
{ stock | stock.valid_from <= D <= stock.valid_to }
```

The membership source is an interval table:

```text
kdcode, valid_from, valid_to
```

Example:

```text
A.N,2016-01-01,2026-05-13
AA.N,2016-11-01,2016-11-01
AABA.OQ,2016-01-01,2017-06-16
AABA.OQ^J19,2016-01-01,2017-06-16
```

This means the universe is date-dependent. It is not the current S&P 500 list,
not a fixed 2016 list, and not a list of stocks that survived across the whole
experiment.

## Current Data Artifacts

### PIT membership intervals

Path:

```text
data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
```

Purpose:

- Defines when each identifier is an S&P 500 constituent.
- Provides the `valid_from` and `valid_to` bounds used to build daily active
  membership masks.
- Includes historical/dead LSEG identifiers where needed.

This file controls tradability. A stock can have price history before it joins,
but it is not tradable or loss-eligible until `valid_from`.

### New LSEG PIT-union market panel

Path:

```text
data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv
```

Metadata:

```text
data/raw/market/sp500_pit_union_lseg_20150101_20260513.meta.json
```

Source:

```text
refinitiv.data.get_history
```

The pull was run through the authenticated local LSEG Workspace session.

Observed metadata:

```text
requested_identifiers:          870
resolved_identifiers_with_rows: 759
rows:                           1,849,404
date_min:                       2015-01-02
date_max:                       2026-05-13
columns:                        kdcode, dt, open, high, low, close, volume, turnover
```

The output file is also fingerprinted inside the successful experiment
metadata:

```text
data_file_sha256:      84e1f3f2b79a798246e001e17a372c8daf8bcfc658873ad3d352a99ad993840f
data_file_size_bytes:  125721665
data_file_mtime_iso:   2026-05-13T20:19:57.980381+00:00
```

### Remaining unresolved identifiers

The first bulk pull reported 8 initial failures. Two timeout names, `A.N` and
`AA.N`, were recovered on retry. The current unresolved original failure count
is 6:

```text
AABA.OQ
ABMD.OQ
AET.N
AGN.N
AIRC.N
ALXN.OQ
```

These are mostly unsuffixed dead aliases. Several corresponding historical
suffixed RICs are present and usable, for example:

```text
AABA.OQ^J19
ABMD.OQ^L22
AET.N^K18
AGN.N^E20
AIRC.N^F24
ALXN.OQ^G21
```

This is a normal LSEG historical-identifier issue, not evidence that the panel
is a current-survivor panel. The metadata keeps the unresolved list so future
alias-cleanup work can audit it directly.

### Alias coverage audit

On 2026-05-16, `scripts/audit_pit_lseg_alias_coverage.py` audited those six
unresolved originals against the PIT universe and the LSEG PIT-union market
panel. The exported artifacts are:

```text
docs/PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16.md
docs/PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16_candidates.csv
docs/PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16_daily_impact.csv
```

Result:

- All six originals have a suffixed historical candidate in the PIT universe.
- All six candidates have market rows in the PIT-union market panel.
- In the 2022-2025 validation window, only `ABMD.OQ` has an active unresolved
  original interval. Its suffixed candidate `ABMD.OQ^L22` covers all 245 active
  member-days and all 245 scoreable member-days.
- The unresolved-original breadth impact for 2022-2025 is therefore 0 uncovered
  active member-days and 0 unscoreable active member-days.

## Why The Previous Data Was Not Enough

Before this LSEG PIT-union pull, the available Google Drive/local market files
were anchored universe datasets. They had real LSEG-style stock data, but the
underlying stock set was anchored to a specific snapshot rather than the full
rolling PIT union.

The strict masked-panel test exposed this immediately.

Using the old file:

```text
data/raw/market/sp500_2016_universe_data.csv
```

the 2022 PIT experiment failed before training:

```text
PIT train breadth below 450
first failing date: 2016-01-19
scoreable names:   about 346
```

That failure is the intended behavior. If a market file lacks enough historical
leavers and joiners, the PIT system refuses to treat it as a clean rolling
universe.

Using the new LSEG PIT-union file, the same strict run passed.

## How Masked-Panel PIT Works

The active mode is:

```yaml
data:
  use_pit_universe: true
  pit_universe_mode: masked_panel
  pit_min_scoreable_stocks: 450
  pit_breadth_policy: error
```

The ready-made 2022 preset is:

```text
configs/experiment/pit_temporal_2022.yaml
```

The important behavior is:

1. Load all market rows from the PIT-union price panel.
2. Load PIT membership intervals from the Joiner/Leaver interval CSV.
3. Build a fixed `kdcode_list` containing every stock with a PIT interval
   overlapping the experiment and with available market data.
4. Build daily masks over `(date, stock)`.
5. Keep the full union axis internally.
6. Mask out invalid stock/date pairs for loss, prediction, and evaluation.

This avoids the historical failure mode where the pipeline filtered to stocks
with complete coverage across all splits. Complete-coverage filtering collapses
the data to continuous stayers, which is exactly what PIT is meant to avoid.

## The Four PIT Masks

The core implementation lives in:

```text
mci_gru/data/pit.py
```

For each sample date and stock, it builds four masks.

### `active_member`

True when the stock is an S&P 500 member on that date:

```text
valid_from <= date <= valid_to
```

This is the direct point-in-time membership mask.

### `feature_ready`

True when the stock has a complete historical lookback window of length
`his_t` before the sample date.

Pre-membership OHLCV is allowed here because those prices were public at the
time. This lets a future joiner have features ready on its join date without
pretending it was tradable before joining.

### `tradable`

True when:

```text
active_member AND feature_ready
```

This is the mask used for prediction export and candidate selection.

### `loss`

True when:

```text
tradable AND forward label is observable
```

This prevents the training objective from learning on inactive names or names
whose forward return cannot be computed.

Invalid labels are stored as `NaN`, not zero. The masked losses and IC metrics
ignore `NaN` targets.

## Data Flow In The Experiment

The PIT path runs through:

```text
mci_gru/pipeline.py
```

High-level flow:

```text
LSEG PIT-union prices
  + PIT membership intervals
  -> feature engineering
  -> train-only normalization fit
  -> fixed PIT union axis
  -> sliding windows
  -> active/feature/loss/tradable masks
  -> graph construction
  -> masked training and validation
  -> PIT-filtered prediction CSVs
  -> PIT-aware backtest
```

Important no-lookahead controls:

- Normalization statistics are fit only through `data.train_end`.
- The static graph is built with returns strictly before `graph_static_valid_from`.
- Dynamic graph snapshots, when enabled, use only data before each snapshot date.
- Feature windows use historical `his_t` observations before each sample date.
- Forward returns are labels only; labels are masked out where unavailable.
- Prediction CSVs only contain same-day `tradable` names.
- PIT-aware backtests filter both candidates and benchmark rows by the PIT
  interval table.

## Model And Graph Handling

The collate contract remains the repository invariant 9-tuple:

```text
(time_series, labels, graph_features, edge_index, edge_weight,
 n_stocks, batch_dates, edge_index_sector, edge_weight_sector)
```

In masked-panel PIT mode, `batch_dates` becomes a metadata dict:

```python
{"dates": [...], "stock_mask": ...}
```

The graph can still be built on the union axis, but at batch time edges touching
inactive nodes are removed. The model also zeros inactive node outputs so
inactive stocks cannot influence graph/self-attention behavior.

## Validation Performed On 2026-05-13

### 1. Unit tests for PIT mechanics

Command:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_pit_masked_panel.py -v --basetemp .codex_tmp\pytest-tmp
```

Result:

```text
11 passed
```

These tests cover:

- Future joiners using pre-membership price history while staying inactive
  before `valid_from`.
- Leaver/rejoiner intervals turning off and back on.
- Invalid labels being preserved as `NaN`.
- Candidate breadth reporting.
- Graph edges being removed when source or destination nodes are inactive.
- The 9-tuple collate contract being preserved.
- Masked MSE and IC ignoring `NaN` targets.
- Prediction rows filtering to the tradable mask.
- The masked-panel pipeline keeping the union axis instead of complete-stock
  filtering.

### 2. Old anchored data failed strict PIT breadth

Command shape:

```powershell
.venv\Scripts\python.exe run_experiment.py +experiment=pit_temporal_2022 training.num_epochs=1 training.num_models=1 tracking.enabled=false
```

With the old anchored file, the run failed before training because breadth fell
below the strict 450-name threshold:

```text
2016-01-19 scoreable names: about 346
```

This confirmed the old data could not support a true rolling S&P 500 PIT panel.

### 3. New LSEG PIT-union file passed strict PIT breadth

Command:

```powershell
.venv\Scripts\python.exe run_experiment.py +experiment=pit_temporal_2022 training.num_epochs=1 training.num_models=1 tracking.enabled=false data.filename=data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv output_dir=results/pit_smoke_lseg
```

Run directory:

```text
results/pit_smoke_lseg/pit_true_rolling_2022/20260513_162011
```

Metadata:

```text
results/pit_smoke_lseg/pit_true_rolling_2022/20260513_162011/run_metadata.json
```

Observed PIT setup:

```text
pit_universe_mode: masked_panel
train_end:         2020-12-31
union axis:        696 stocks in the 2016-2022 experiment window
```

Scoreable breadth:

```text
split   min   median   max
train   503   505      507
val     504   505      506
test    502   503      506
```

This is the strongest practical evidence that the new file contains the real
rolling opportunity set rather than a survivor-only or anchored subset.

### 4. Prediction export matched PIT masks

The 2022 strict smoke generated:

```text
237 prediction files
0 prediction-count mismatches against scoreable PIT masks
```

Examples:

```text
2022-01-24: 505 scoreable names, 505 prediction rows
2022-12-30: 503 scoreable names, 503 prediction rows
```

This confirms the model did not emit predictions for inactive union nodes or
drop to a fixed survivor list.

### 5. PIT-aware backtest completed

Command:

```powershell
$env:MPLBACKEND='Agg'
.venv\Scripts\python.exe -X utf8 tests\backtest_sp500_daily.py --predictions_dir results\pit_smoke_lseg\pit_true_rolling_2022\20260513_162011\averaged_predictions --data_file data\raw\market\sp500_pit_union_lseg_20150101_20260513.csv --pit_universe_csv data\raw\constituents\sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv --test_start 2022-01-24 --test_end 2022-12-30 --label_t 5 --auto_save --backtest_suffix _agg
```

Output:

```text
results/pit_smoke_lseg/pit_true_rolling_2022/20260513_162011/backtest_agg
```

The run used:

```text
119,366 predictions
237 prediction dates
520 unique prediction stocks
235 valid trading days
```

The one-epoch results are not performance evidence. They only prove that the
PIT-clean data, prediction export, and PIT-aware backtest can run end to end.

## What We Can Claim

We can claim the following with current evidence:

1. The new market file was pulled directly from LSEG via the local Workspace
   session, not generated from a later survivor list.
2. The input universe was the PIT Joiner/Leaver interval table, not the current
   S&P 500.
3. The market panel includes historical identifiers and delisted-style suffixed
   RICs, which is inconsistent with a pure current-survivor pull.
4. The older anchored data failed strict PIT breadth; the new LSEG PIT-union
   file passed with roughly 500 scoreable names per normal date.
5. The model pipeline uses daily masks to restrict training loss and prediction
   output to same-day valid PIT candidates.
6. The prediction files match the PIT scoreable mask counts exactly for the
   validated 2022 smoke run.
7. The input file hash, size, and modification time are recorded in
   `run_metadata.json` for provenance.

## What We Should Not Overclaim

We should not claim every vendor value is independently perfect. Remaining
data-quality caveats:

- Some original unsuffixed dead aliases did not resolve directly.
- A dedicated alias-resolution pass should eventually map all dead aliases to
  their preferred historical LSEG RICs.
- We have not yet audited corporate-action adjustments against an independent
  source.
- We have run a one-epoch smoke, not a full trained research result.
- The 2022 validation proves the current mechanics and data breadth for that
  window; the 2023, 2024, and 2025 presets should be run separately.

The precise claim is:

```text
The 2026-05-13 LSEG PIT-union panel is clean enough to support the repository's
true PIT masked-panel workflow, and the 2022 strict smoke confirms that the
pipeline is no longer collapsing to a survivor or anchored universe.
```

## Operational Rules

When running true PIT experiments:

1. Use `data.pit_universe_mode=masked_panel`.
2. Use a PIT-union market file, not an anchored universe file.
3. Keep `data.pit_breadth_policy=error` unless deliberately diagnosing data
   coverage.
4. Treat a breadth failure as a data problem first, not a model problem.
5. Do not re-enable complete-stock filtering in masked-panel mode.
6. Do not evaluate headline PIT results without passing `--pit_universe_csv` to
   the backtest.
7. Do not compare PIT-clean results against older survivor-filtered runs without
   labeling the difference clearly.

## Recommended Next Checks

1. Run strict one-epoch smokes for `pit_temporal_2023`, `pit_temporal_2024`, and
   `pit_temporal_2025` using the new LSEG PIT-union file.
2. Add a saved LSEG pull script so the data artifact can be reproduced without
   relying on an inline notebook or shell snippet.
3. Re-run the alias coverage audit whenever the PIT Joiner/Leaver export or
   PIT-union market pull is regenerated.
4. Upload the new PIT-union market CSV and `.meta.json` to the shared Google
   Drive location used by Colab.
5. Run a proper multi-epoch, multi-seed PIT research experiment only after the
   remaining year presets pass breadth validation.

## Key Files

Implementation:

```text
mci_gru/data/pit.py
mci_gru/pipeline.py
mci_gru/data/data_manager.py
mci_gru/models/mci_gru.py
mci_gru/training/losses.py
mci_gru/training/trainer.py
mci_gru/evaluation/portfolio.py
```

Tests:

```text
tests/test_pit_masked_panel.py
tests/backtest_sp500_daily.py
tests/backtest_sp500.py
```

Configs:

```text
configs/experiment/pit_temporal_2022.yaml
configs/experiment/pit_temporal_2023.yaml
configs/experiment/pit_temporal_2024.yaml
configs/experiment/pit_temporal_2025.yaml
```

Data and validation artifacts:

```text
data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv
data/raw/market/sp500_pit_union_lseg_20150101_20260513.meta.json
results/pit_smoke_lseg/pit_true_rolling_2022/20260513_162011/run_metadata.json
results/pit_smoke_lseg/pit_true_rolling_2022/20260513_162011/backtest_agg/backtest_results.csv
```
