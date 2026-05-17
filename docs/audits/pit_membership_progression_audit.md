# PIT Membership Progression Audit

This audit makes the Joiner/Leaver point-in-time membership progression visible for the masked-panel validation workflow.

## Sources

- Changes: `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_changes.csv`
- Snapshots: `data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_snapshots.csv`
- PIT-union market panel: `data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv`
- Snapshot progression CSV: `docs/audits/pit_membership_progression_snapshot_counts.csv`

## Summary

- Changes file totals: 251 joiners and 252 leavers.
- Snapshot active membership range: 503-509 names.
- Validation-year guard: snapshot membership changes within 2022-2025.

## Joiners And Leavers By Year

| year | joiners | leavers | total_changes |
| --- | --- | --- | --- |
| 2016 | 40 | 39 | 79 |
| 2017 | 34 | 34 | 68 |
| 2018 | 28 | 28 | 56 |
| 2019 | 24 | 24 | 48 |
| 2020 | 17 | 17 | 34 |
| 2021 | 24 | 24 | 48 |
| 2022 | 18 | 20 | 38 |
| 2023 | 18 | 18 | 36 |
| 2024 | 17 | 17 | 34 |
| 2025 | 22 | 22 | 44 |
| 2026 | 9 | 9 | 18 |

## Validation-Year Snapshot Transition Counts

| year | joined_count | left_count | transition_count |
| --- | --- | --- | --- |
| 2022 | 18 | 20 | 38 |
| 2023 | 18 | 18 | 36 |
| 2024 | 17 | 17 | 34 |
| 2025 | 22 | 22 | 44 |

## Representative Snapshot Transitions

| as_of_date | year | direction | constituent_ric | market_kdcode | market_row_count | market_first_dt | market_last_dt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-02-02 | 2022 | Joiner | CEG.OQ | CEG.OQ | 1082 | 2022-01-19 | 2026-05-12 |
| 2022-02-03 | 2022 | Leaver | GAP.N | GAP.N | 2856 | 2015-01-02 | 2026-05-12 |
| 2023-01-04 | 2023 | Joiner | GEHC.OQ | GEHC.OQ | 853 | 2022-12-15 | 2026-05-12 |
| 2023-01-05 | 2023 | Leaver | VNO.N | VNO.N | 2857 | 2015-01-02 | 2026-05-13 |
| 2024-03-18 | 2024 | Joiner | DECK.N | DECK.N | 2856 | 2015-01-02 | 2026-05-12 |
| 2024-03-18 | 2024 | Leaver | WHR.N | WHR.N | 2857 | 2015-01-02 | 2026-05-13 |
| 2025-02-24 | 2025 | Joiner | SNDK.OQ | SNDK.OQ | 312 | 2025-02-13 | 2026-05-12 |
| 2025-02-10 | 2025 | Leaver | MRP.N | MRP.N | 318 | 2025-02-05 | 2026-05-12 |
