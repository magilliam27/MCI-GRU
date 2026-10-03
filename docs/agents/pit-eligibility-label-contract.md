# PIT eligibility and label contract (#225)

This is the implemented contract for dated cessation eligibility and label coverage
in masked-panel preparation. The rulings are Mac's, recorded on #225 on 2026-10-03
([comment 5971662968](https://github.com/magilliam27/MCI-GRU/issues/225#issuecomment-5971662968)).
The settled principles they build on (effective-and-known eligibility, single source,
no repair, preserved selection) are in the #225 body and are not repeated here.

## Prediction clock

The forecast for sample date D is made at **20:00 America/New_York on D**
(`data.prediction_clock_time`, `data.prediction_clock_timezone`). A test fixture may
declare another clock through the same fields. The clock is a declared convention, not
measured latency.

## Cessation evidence

`data.pit_cessation_events_csv` declares the event file. It requires
`data.use_pit_universe=true`. `null` declares no cessation evidence and excludes
nothing; the PIT fragment records that it was undeclared. The file is read once,
as text, through the #191 input carrier under the role
`data.pit_cessation_events_csv`.

| Column | Required | Meaning |
| --- | --- | --- |
| `event_id` | yes | Unique, non-blank. |
| `kdcode` | yes | The original security code on the PIT union axis. |
| `effective_at` | yes | When the cessation takes effect: `YYYY-MM-DD` or a timezone-aware ISO timestamp. |
| `known_from` | yes | When the cessation was knowable: `YYYY-MM-DD` or a timezone-aware ISO timestamp. |
| `acquired_at` | no | When the evidence was collected. Reported, **never** used as availability. |
| `evidence` | no | Source reference. Reported only. |

Resolution, with every instant converted to UTC:

- A timestamp holds from that instant.
- A date-only `known_from` K holds from the prediction clock on the **first panel
  session after K** (ruling 2). Evidence dated D is first usable for the forecast
  on the next session. It is never read as midnight.
- A date-only `effective_at` E holds from the start of E in the clock's timezone, so
  a cessation dated E is in effect for the forecast made on E. *This is an
  implementation default the rulings did not fix; it only matters once the
  cessation is also known, which date-only knowledge never is before the next
  session.*
- A stock is excluded from daily eligibility on D when, for some event,
  `max(effective, known) <= forecast(D)`. Equality counts. The exclusion persists
  on later dates. Monthly selection, the union axis and the original rows are never
  changed.

The run stops with `mci_gru.data.pit.PITEligibilityError` before any tensor is built
when the file is missing a required column, has a blank or duplicate `event_id`, a
blank `kdcode`, a malformed or timezone-naive timestamp, or a cessation with a blank
`effective_at` or `known_from` (no dated evidence, ruling 2). A missing or unreadable
declared file stops the same way (`event_file_unreadable`), under the settled
single-source rule. The error is an `AdmissionError` (#223) with one `invalid` item
(`rule="cessation_known_by"`, `stage="pit_eligibility"`, the reason code, and the
named stocks as evidence). Preparation records it in the window's admission ledger
and attaches the sealed input observations, so the runner writes it to
`run_failure.json` (`docs/agents/data-quality-contract.md`).

## Populations

All are `(dates, stocks)` booleans on the fixed union axis
(`mci_gru.data.pit.PITMaskSet`):

| Mask | Definition |
| --- | --- |
| `active_member` | Monthly PIT selection. Unchanged by this contract. |
| `cessation_excluded` | Effective and known by the forecast, as above. |
| `eligible` | `active_member & ~cessation_excluded`. Never reads labels. |
| `feature_ready` | Complete `his_t` lookback before D. |
| `price_observed` | A finite close on D itself. A genuine missing price masks that session only (#223 ruling 13). |
| `tradable` | `eligible & feature_ready & price_observed`: the prediction population. |
| `label_available` | The fixed-session label is observable. |
| `loss` | `tradable & label_available`: rows that train and score. |

## Labels (ruling 4)

`label(D) = close[session D + label_t] / close[session D + 1] - 1`, where sessions are
the panel's own trading dates: every date any selected stock has a row on
(`mci_gru.data.preprocessing.label_session_axis`). With `label_t=5` the return spans
four close-to-close intervals. A stock without a close at either endpoint has an
unobservable label. There is no fill, no next-available substitute and no terminal
valuation. `compute_labels`, `label_available_mask` and
`assert_training_labels_respect_embargo` all call `resolve_label_endpoints`, so the
embargo verifies exactly the exit close each label reads.

Non-masked runs keep their existing same-day mean then zero fill for unobservable
labels; only the endpoint rule changed for them, and only where a selected stock has
a gap.

## Coverage report (ruling 3)

Training and evaluation use observable labels only. `prepare_data` returns the
`pit_eligibility` fragment (schema `mci_gru.pit_eligibility.v1`), which run metadata
records next to `pit_breadth` and the admission record carries as
`coverage.pit_eligibility`. It holds the clock, the endpoint rule, every declared
event with its original text and resolved UTC instants, and per split:

- `daily`: `selected`, `cessation_excluded`, `eligible`, `feature_ready`,
  `price_gap`, `predictions`, `label_observable`, `label_omitted`;
- `price_gaps_by_stock`: eligible sessions masked for a missing close, per stock;
- `totals`: the same counts summed;
- `label_coverage`: `label_observable / predictions`, with numerator and denominator
  named;
- `omitted_labels`: every prediction without an observable label, with its entry and
  exit sessions and a reason: `entry_close_missing`, `exit_close_missing`,
  `entry_and_exit_close_missing`, `endpoint_after_panel_end` or `non_finite_return`.

No complete economic-return claim is made until #129 values terminal outcomes.

## Proof

`tests/test_pit_eligibility_contract.py` runs public `prepare_data` on the packet's
two-name fixture (29 declared sessions; KEEP on all, STOP through 2024-02-05; D =
2024-02-05, forecast at 2024-02-06T01:00Z) with these event variants:

| Case | Evidence | STOP at D |
| --- | --- | --- |
| K | effective 22:00Z, known 21:30Z on D | excluded |
| L | known 02:00Z on D+1, acquired a month earlier | kept; label omitted |
| D1 | known date-only 2024-02-02 (counts from the D forecast) | excluded |
| D2 | known date-only 2024-02-05 (counts from the D+1 forecast) | kept; label omitted |
| F | effective 22:00Z on D+1 | kept; label omitted |
| U | blank `known_from` | run rejected naming STOP |

KEEP's label at D is `128/124 - 1` in every case. A second fixture with a missing
2024-02-07 bar gives `104/100 - 1` under fixed sessions where row shifts gave
`105/100 - 1`, and all three label consumers agree on it.
