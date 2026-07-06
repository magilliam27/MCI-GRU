# Regime Data Contract

This document defines the regime input contract for global scalar regime
features. The canonical column lists live in `mci_gru/regime_contract.py`
(`REGIME_REQUIRED_VARIABLES`, `REGIME_OPTIONAL_VARIABLES`, `REGIME_VARIABLES`).
The canonical workflow is the live FRED/LSEG-backed loader in
`DataManager.load_regime_inputs`; CSV regime inputs are a deprecated legacy
escape hatch.

## Canonical Live Workflow

Leave `features.regime_inputs_csv` unset or `null`. With global regime enabled,
the pipeline loads point-in-time-safe live inputs and derives the full
seven-variable surface:

| Column | Source / derivation | Description |
|--------|----------------------|-------------|
| `dt` | loader date index | Calendar date; one row per date. |
| `regime_market` | LSEG market RIC or FRED `SP500` fallback | Market proxy level. |
| `regime_yield_curve` | FRED/LSEG 10Y minus 3M yield | Yield curve spread. |
| `regime_oil` | FRED WTI or LSEG oil RIC fallback | Oil proxy level. |
| `regime_copper` | LSEG copper RIC or FRED copper fallback | Copper proxy level. |
| `regime_stock_bond_corr` | derived | 756-trading-day rolling correlation of market returns vs 10Y yield changes. |
| `regime_monetary_policy` | lagged 3M yield | Monetary policy / T-bill yield proxy. |
| `regime_volatility` | FRED `VIXCLS` or LSEG VIX fallback | Volatility / VIX proxy. |

The live loader applies a 1-day lag to FRED series before merging. It then
forward/backward fills sparse market holidays for raw live fields after all
series are loaded. `regime_stock_bond_corr` uses a 756-trading-day rolling
window with `min_periods=756`; its initial warmup rows remain `NaN` and only
later gaps are forward-filled, so the first valid correlation is not backfilled
into dates where the full 3-year window was unavailable.

## Requirements

- `FRED_API_KEY` should be set for the normal regime workflow.
- When `data.source=lseg`, configured LSEG RICs may supplement or replace
  specific live series where available.
- `features.regime_inputs_csv` should remain `null` in production configs.

## Deprecated CSV Escape Hatch

`features.regime_inputs_csv` still exists only for legacy offline experiments.
When set, it bypasses the live loader, emits a `DeprecationWarning`, and must
provide `dt` plus all seven regime variables listed above. Five-variable CSVs
are no longer accepted because they silently drop the paper-guided monetary
policy and volatility dimensions.

If the deprecated CSV path is used:

- no extra columns are required;
- optional helper columns such as `yield_10y` or `yield_3m` are ignored;
- all seven regime variables must be numeric or coercible to numeric;
- `features.regime_enforce_lag_days` may shift the loaded values forward for
  point-in-time safety;
- after any configured lag, the loader forward-fills only and never backfills
  leading gaps.

## Retired Colab Reconciliation

`scripts/colab_regime_reconcile.py` and `scripts/export_lseg_regime.py` are
deprecated. They no longer write regime CSV files, because the live loader is
the canonical source for seven-variable regime inputs. Older notebooks that
reference these scripts should be updated to leave `features.regime_inputs_csv`
unset and rely on `FRED_API_KEY`.

## Validation

The regime feature module consumes `dt` plus:

- `regime_market`
- `regime_yield_curve`
- `regime_oil`
- `regime_copper`
- `regime_stock_bond_corr`
- `regime_monetary_policy`
- `regime_volatility`

Direct calls to `compute_regime_monthly_features` still tolerate missing
optional columns for in-memory synthetic tests and older callers, but persisted
CSV overrides must provide the full seven-variable contract.
