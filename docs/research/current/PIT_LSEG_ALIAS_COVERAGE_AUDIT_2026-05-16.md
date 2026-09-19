# PIT LSEG Alias Coverage Audit

This audit checks whether unresolved original LSEG identifiers have suffixed historical RIC candidates in the PIT universe and market rows in the PIT-union market panel.

## Summary

| metric | value |
| --- | --- |
| validation_start | 2022-01-01 |
| validation_end | 2025-12-31 |
| validation_dates | 1003 |
| his_t | 10 |
| label_t | 5 |
| unresolved_originals | 6 |
| originals_with_any_candidate | 6 |
| originals_with_candidate_market_rows | 6 |
| candidate_rows | 6 |
| distinct_candidates | 6 |
| candidates_with_market_rows | 6 |
| original_active_member_days | 245 |
| candidate_covered_active_days | 245 |
| candidate_scoreable_active_days | 245 |
| uncovered_active_member_days | 0 |
| unscoreable_active_member_days | 0 |
| max_daily_original_active_count | 1 |
| max_daily_uncovered_active_count | 0 |
| max_daily_unscoreable_active_count | 0 |

## Candidate Coverage

| original | candidate | candidate_valid_from | candidate_valid_to | has_market_rows | market_rows | market_date_min | market_date_max | overlaps_validation | active_days_in_validation | scoreable_days_in_validation | loss_days_in_validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AABA.OQ | AABA.OQ^J19 | 2016-01-01 | 2017-06-16 | True | 1196 | 2015-01-02 | 2019-10-02 | False | 0 | 0 | 0 |
| ABMD.OQ | ABMD.OQ^L22 | 2018-05-31 | 2022-12-21 | True | 2008 | 2015-01-02 | 2022-12-21 | True | 245 | 245 | 240 |
| AET.N | AET.N^K18 | 2016-01-01 | 2018-11-30 | True | 985 | 2015-01-02 | 2018-11-28 | False | 0 | 0 | 0 |
| AGN.N | AGN.N^E20 | 2016-01-01 | 2020-05-11 | True | 1347 | 2015-01-02 | 2020-05-08 | False | 0 | 0 | 0 |
| AIRC.N | AIRC.N^F24 | 2020-12-15 | 2020-12-18 | True | 888 | 2020-12-15 | 2024-06-27 | False | 0 | 0 | 0 |
| ALXN.OQ | ALXN.OQ^G21 | 2016-01-01 | 2021-07-20 | True | 1648 | 2015-01-02 | 2021-07-20 | False | 0 | 0 | 0 |

## Validation Breadth Impact By Year

| year | original_active_member_days | candidate_covered_active_days | candidate_scoreable_active_days | uncovered_active_member_days | unscoreable_active_member_days |
| --- | --- | --- | --- | --- | --- |
| 2022 | 245 | 245 | 245 | 0 | 0 |
| 2023 | 0 | 0 | 0 | 0 | 0 |
| 2024 | 0 | 0 | 0 | 0 | 0 |
| 2025 | 0 | 0 | 0 | 0 | 0 |

## Daily Impact Rows

Rows are shown for dates with non-zero unresolved-original activity. For long outputs this table shows the first and last 20 such dates; use the companion daily CSV for the full daily series.

| date | original_active_count | covered_by_candidate_count | scoreable_by_candidate_count | uncovered_active_count | unscoreable_active_count |
| --- | --- | --- | --- | --- | --- |
| 2022-01-03 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-04 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-05 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-06 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-07 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-10 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-11 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-12 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-13 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-14 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-18 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-19 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-20 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-21 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-24 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-25 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-26 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-27 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-28 | 1 | 1 | 1 | 0 | 0 |
| 2022-01-31 | 1 | 1 | 1 | 0 | 0 |
| 2022-11-23 | 1 | 1 | 1 | 0 | 0 |
| 2022-11-25 | 1 | 1 | 1 | 0 | 0 |
| 2022-11-28 | 1 | 1 | 1 | 0 | 0 |
| 2022-11-29 | 1 | 1 | 1 | 0 | 0 |
| 2022-11-30 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-01 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-02 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-05 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-06 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-07 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-08 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-09 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-12 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-13 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-14 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-15 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-16 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-19 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-20 | 1 | 1 | 1 | 0 | 0 |
| 2022-12-21 | 1 | 1 | 1 | 0 | 0 |
