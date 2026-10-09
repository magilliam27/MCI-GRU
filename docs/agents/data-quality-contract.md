# Input Admission Contract

Status: **implemented for the inputs the frozen recipe enables** (the market
panel and the PIT universe file). Ticket: [#223](https://github.com/magilliam27/MCI-GRU/issues/223).
Owner rulings: [2026-10-03 decisions](https://github.com/magilliam27/MCI-GRU/issues/223#issuecomment-5971665014).
Code: `mci_gru/data/quality_contract.py`. Proofs: `tests/test_data_quality_contract.py`.

Identity (#191) records what was read. Admission decides whether what was read
may be trained on. Neither repairs a source: no row is filled, deduplicated,
dropped or substituted to make admission pass.

## Seams

| Step | Where | What happens |
| --- | --- | --- |
| Resolve | `DataManager._read_selected_csv` | A required selected file (`data.filename`, `data.index_filename`) resolves by its explicit path or relative to the project root, never by a same-named basename search (`resolve_project_data_path(..., allow_basename_fallback=False)`). |
| Read | same | Resolve, read and parse failures stop at once with role, source, stage and reason. |
| Validate | `DataManager._load_from_csv`, `load_pit_intervals` | Rules run on the rows the native reader returned, before any transform, and record verdicts in the window's `AdmissionLedger`. |
| Enforce | `prepare_data`, `prepare_data_index_level` | `AdmissionLedger.require_admitted` runs after the panel read (before any auxiliary provider), after auxiliary and PIT loading (before feature work), and before returning. |
| Report | `run_experiment.py` | One catch around the preparation dispatch, for `AdmissionError` and `InputObservationError` only, writes `run_failure.json` in the window folder, logs role/source/stage/reason, and re-raises. No trainer is built. |

A successful window writes `admission.json` (schema `mci_gru.admission.v1`)
next to `run_metadata.json`, with every verdict and the per-stock coverage counts.

## Verdicts

`valid`, `invalid`, `insufficient_evidence`, `unresolved`. A **required** item
whose verdict is anything but `valid` blocks admission. #224 and #225 record
their verdicts in the same ledger (`DataManager.admission`, or the
`admission=` keyword on the preparation entry points); there is no second
collector and no second report.

## Market panel rules (`data.filename`)

Settled on #193, not re-asked:

- Columns `kdcode, dt, open, high, low, close, volume`; extra columns are allowed.
- Nonblank identifiers; `dt` is a `YYYY-MM-DD` date; one row per stock/session.
- Observed OHLC finite and positive; observed volume finite and nonnegative.
  A blank cell is genuine missingness, not malformed; any other non-numeric
  token is malformed.
- **Frozen history**: per stock and OHLC field, `n` = distinct dates with a
  finite value and `u` = distinct values. All four fields with `n >= 2` and
  `u = 1` is `invalid`. A single constant field is reported as `valid`
  (`constant_field_reported`). Any field with `n < 2` is
  `insufficient_evidence`, which stops the run (owner decision 2).

Owner decision 2: **no per-stock missing-price budget.** Gaps are counted per
stock (`rows_with_missing_price`, `absent_sessions_within_history`) and stop
nothing. Only the session breadth floor (`pit_min_scoreable_stocks` with
`pit_breadth_policy: error`, 104 on the 110-name recipe) or the `n < 2` rule
stops a run. A breadth-floor failure is an `AdmissionError` and reaches
`run_failure.json`.

## PIT universe rules (`data.pit_universe_csv`)

Owner decision 3:

- A blank `valid_to` is membership through `data.pit_export_cutoff`
  (`2026-07-31` for `gics_top10_110_2016`). With no cutoff declared, a blank
  `valid_to` stops the run. It is never dropped.
- Blank identifiers, blank or unparseable `valid_from`, and unparseable
  `valid_to` stop the run.
- An inverted interval (`valid_from > valid_to`) stops the run.
- Two intervals for one name that share a date stop the run. Adjacent
  intervals (one ends the day before the next starts) do not.
- A name with membership in the experiment period and no row in the market
  panel stops the run. It is no longer dropped from the union. The one
  exception is a name the data config declares in `data.pit_absent_kdcodes`
  because its package has no source for it (`DD.N^I17` in
  `gics_top10_110_2016_eodhd`, #281). Admission records the declared gap as a
  valid item and the name stays off the stock axis. The declaration describes
  the package, so it holds for any window, including one the name is not a
  member in. A declared name that has panel rows, or is in no PIT interval,
  stops the run.

Findings name the stock and list up to 20 offending rows; counts are complete.

## `run_failure.json` (`mci_gru.run_failure.v1`)

| Field | Meaning |
| --- | --- |
| `outcome`, `stage` | `failed`, `preparation`. |
| `walkforward_window`, `resolved_config`, `created_at_utc` | The window and resolved config the runner had already written. |
| `failures` | One entry per blocking item: `role`, `source`, `configured_path`, `stage` (`resolve`/`read`/`parse`/`validate`/`admit`), `rule`, `verdict`, `reason_code`, `reason`, `required`, `evidence`. |
| `admission` | The whole ledger at the point of failure, including non-blocking items and coverage. |
| `input_observations` | The window's sealed #191 observations, as read; nothing is reopened or rehashed. |

If the report cannot be written, the runner logs that and still fails.

## Deferred while the recipe leaves them disabled

Index-family value rules, sector mapping rules and exporter-side validation.
Index mode still enforces resolution, read failures and any recorded
verdicts. Masking of a session whose own OHLC is blank is with the PIT
mask/label work (#225); labels that need a missing close are already masked.
