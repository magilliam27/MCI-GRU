# Canonical Regime Data Contract (Hybrid Local + Colab)

This document defines the schema and lag policy for global regime inputs used in the hybrid workflow: local LSEG export plus Colab FRED merge.

## Canonical CSV Schema

The **canonical regime inputs** file consumed by the training pipeline must have exactly these columns (order does not matter for loading, but column names are case-sensitive):

| Column | Type | Description |
|--------|------|-------------|
| `dt` | string (YYYY-MM-DD) | Calendar date; one row per date. |
| `regime_market` | float | Market proxy level (e.g. S&P 500 index or close). |
| `regime_yield_curve` | float | 10Y yield minus 3M yield (spread). |
| `regime_oil` | float | Oil proxy level (e.g. WTI close). |
| `regime_copper` | float | Copper proxy level (e.g. LME or Refinitiv close). |
| `regime_stock_bond_corr` | float | Rolling correlation of market returns vs 10Y yield changes (e.g. 63d, min 21). |

- No extra columns are required; optional columns (e.g. `yield_10y`, `yield_3m`) may be present but are ignored by the regime feature module.
- Missing values may be present; the pipeline will forward-fill/backfill as in existing logic. Rows with all regime values missing for a date should be avoided where possible.

## Lag Policy (No Look-Ahead)

- **Decision time:** Model inputs for date `T` must use only information that would have been available at or before the close of day `T` (or at a defined cutoff, e.g. previous close).
- **Recommended:** Apply a **1-day lag** to all macro/commodity series before merging: the value assigned to date `T` is the value as of `T-1`. This avoids look-ahead when data is published with a delay.
- **Config:** When using the CSV override, an optional `features.regime_enforce_lag_days` (default 0) can be set to 1 so the loader shifts the loaded regime columns by that many calendar days before use. If the CSV was already built with lag applied (e.g. in Colab reconciliation), set to 0.

## Local LSEG Export (Partial File)

When running **locally** (LSEG-only), export at minimum:

- `dt`, `regime_copper`

Optionally, if entitled: `regime_market`, `regime_oil`. Yields and stock-bond correlation are typically sourced from FRED in Colab.

- **Naming:** e.g. `lseg_regime_export_YYYYMMDD.csv` or `regime_lseg_partial_<start>_<end>.csv`.
- **Metadata:** Save a small companion file or header comment with date range, RICs used, and row counts for reproducibility.

## Colab Reconciliation

Use the script `scripts/colab_regime_reconcile.py` in Colab:

1. Set env vars (or edit the script defaults): `LSEG_REGIME_PATH` to the path of your local LSEG export CSV (e.g. in Drive), `REGIME_START`, `REGIME_END`, `REGIME_OUTPUT`.
2. Run: `%run scripts/colab_regime_reconcile.py`. The script fetches FRED series with 1-day lag (SP500, DGS10, DGS3MO, DCOILWTICO), merges the LSEG export on `dt`, computes `regime_yield_curve` and `regime_stock_bond_corr`, and writes `regime_inputs_reconciled.csv`.
3. In your experiment config set `regime_inputs_csv` to that output path (e.g. `data/raw/market/regime_inputs_reconciled.csv`) so the pipeline uses the canonical file instead of live APIs.
4. Version the output file per run for reproducibility against FRED revisions.

## Validation

- The regime feature module (`mci_gru.features.regime`) expects a DataFrame with `dt` plus exactly: `regime_market`, `regime_yield_curve`, `regime_oil`, `regime_copper`, `regime_stock_bond_corr`.
- CSV override loaders should validate presence of these columns and optionally check for duplicate dates and monotonic date order after sort.
