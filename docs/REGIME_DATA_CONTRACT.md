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
| `dt` | session grid | One row per weekday session. |
| `regime_market` | LSEG market RIC or FRED `SP500` fallback | Market proxy level. |
| `regime_yield_curve` | FRED/LSEG 10Y minus 3M yield | Yield curve spread. |
| `regime_oil` | FRED WTI or LSEG oil RIC fallback | Oil proxy level. |
| `regime_copper` | LSEG copper RIC or FRED copper fallback | Copper proxy level. |
| `regime_stock_bond_corr` | derived | 756-trading-day rolling correlation of market returns vs 10Y yield changes. |
| `regime_monetary_policy` | lagged 3M yield | Monetary policy / T-bill yield proxy. |
| `regime_volatility` | FRED `VIXCLS` or LSEG VIX fallback | Volatility / VIX proxy. |

## Historical Availability Rules (#224)

The owner rulings of 2026-10-03 on #224 apply to the six FRED inputs the
frozen recipe enables. They are written per role (market level, 10-year yield,
3-month yield, oil, copper, volatility), so a later source for the same role
inherits them, and they are declared conventions, not measured publication
latency. The code lives in `mci_gru/data/auxiliary_quality.py`; the proof is
`tests/test_auxiliary_quality.py`, through `DataManager.load_regime_inputs` in
replay mode with providers and network off.

| Rule | Behaviour |
|------|-----------|
| Clock | The forecast for session t is made at 20:00 America/New_York on t. |
| Sessions | Weekdays. On an exchange trading date this gives the same availability as the exchange calendar; an exchange holiday counts as one session of carry. |
| Daily availability | A value dated D is known from 20:00 New York on the next session, so session t sees only values dated before t. |
| Copper availability | The value for month M is known from the first session of month M+2. |
| Carry | A daily value carries at most 5 sessions; a monthly value only through its month M+2. A longer gap stops the run, naming the role, the last observation and the first session over the limit. |
| Missing markers | FRED `.` or a blank is a genuine gap. Any other non-numeric token, an infinite value, or a repeated observation date stops the run. |
| Valid values | Market level, volatility and copper must be positive. The two yields and oil may be any finite value. Units are as FRED documents them. |
| Leading gaps | No back-fill. Sessions before a role's first usable value stay empty, so the regime features keep their neutral 0 there (`add_regime_features`), and the count is recorded. The training window is unchanged. |
| Revisions | Unchecked. Values as served at capture stand in for history. |

Derived columns follow their inputs on the same session: the yield curve is
10-year minus 3-month, monetary policy is the 3-month yield, and
`regime_stock_bond_corr` is a 756-session rolling correlation of market returns
against 10-year yield changes with `min_periods=756`, so its warmup stays
empty.

`DataManager.regime_input_receipt` carries one verdict per role (`role`,
`label`, `source`, `series_id`, `observation_id`, the rules and scope applied,
first observation, first usable session, leading-gap sessions, missing
observations, the longest carry seen, and `revisions: unchecked`) plus the
leading-gap count of each of the seven output columns. The same verdicts go
into #223's admission ledger (`DataManager.admission`) as one `valid` item per
role, rule `regime_historical_availability`, role `fred.<column>`, with the
verdict as evidence and the leading-gap counts as `coverage["regime"]`, so they
reach `admission.json` on success. A stop raises an `InputSnapshotError` at
stage `validate` whose facts carry the reason and dates; it is also recorded as
an `invalid` item, and the ledger so far travels with the error into
`run_failure.json`.

Index mode (`load_index_series`), standalone VIX, credit and the legacy CSV
below are disabled in the recipe and keep their earlier behaviour.

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

The Colab reconciliation exporters were retired in 2026-09 (readable at tag
`archive/pre-cleanup-2026-09`). The live loader is the canonical source for
seven-variable regime inputs; leave `features.regime_inputs_csv` unset and rely
on `FRED_API_KEY`.

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
