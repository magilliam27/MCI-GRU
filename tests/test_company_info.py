"""Earnings releases are dated by when they became public, and owned by the right company (#393).

``mci_gru.data.company_info`` turns EODHD earnings histories into one row per release
for the panel. These tests pin that:

* a release before the open is known at that session's close; one after the close, or
  at an unrecorded time, at the next session's close; one on a non-trading day at the
  next session's close either way;
* a release goes only to the panel code whose symbol segment covers its report date,
  so a ticker reused by another company never lends its releases;
* releases outside the pull's dates are left out, so no row reaches the locked holdout;
* a malformed events file is refused.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mci_gru.data import company_info as ci

SESSIONS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


def test_a_release_before_the_open_is_known_that_day_and_after_the_close_the_next():
    known = ci.known_session(
        ["2024-01-03", "2024-01-03", "2024-01-03"], ["before", "after", "unknown"], SESSIONS
    )
    assert known.tolist() == [1, 2, 2]


def test_a_release_on_a_non_trading_day_is_known_at_the_next_close():
    known = ci.known_session(["2024-01-06", "2024-01-06"], ["before", "after"], SESSIONS)
    assert known.tolist() == [4, 4]


def test_a_release_after_the_last_session_points_past_the_axis():
    known = ci.known_session(["2024-01-08", "2024-02-01"], ["after", "before"], SESSIONS)
    assert known.tolist() == [5, 5]


def test_eodhd_timing_maps_unknown_values_to_unknown():
    assert ci.eodhd_timing("BeforeMarket") == "before"
    assert ci.eodhd_timing("AfterMarket") == "after"
    assert ci.eodhd_timing(None) == "unknown"
    assert ci.eodhd_timing("DuringMarket") == "unknown"


def _release(report, actual=1.0, estimate=0.9, timing="AfterMarket"):
    return {
        "reportDate": report,
        "date": "2023-12-31",
        "beforeAfterMarket": timing,
        "epsActual": actual,
        "epsEstimate": estimate,
    }


def test_a_reused_ticker_lends_its_releases_only_inside_each_companys_segment():
    record = {
        "OLD.N^A20": {"segments": [{"symbol": "TKR.US", "start": None, "end": "2020-01-31"}]},
        "NEW.N": {"segments": [{"symbol": "TKR.US", "start": "2021-06-01", "end": None}]},
    }
    history = {
        "TKR.US": {
            "a": _release("2019-11-01"),
            "b": _release("2020-11-01"),  # between the segments: nobody's
            "c": _release("2022-02-01"),
        }
    }
    events = ci.earnings_from_eodhd(history, ci.symbol_segments(record), "2015-01-01", "2025-12-31")
    assert events[["kdcode", "report_date"]].values.tolist() == [
        ["NEW.N", "2022-02-01"],
        ["OLD.N^A20", "2019-11-01"],
    ]


def test_releases_outside_the_pull_dates_are_left_out():
    record = {"A.N": {"segments": [{"symbol": "A.US"}]}}
    history = {
        "A.US": {
            "a": _release("2014-12-31"),
            "b": _release("2025-12-31"),
            "c": _release("2026-01-22"),
            "d": {"reportDate": None, "epsActual": 1.0},
        }
    }
    events = ci.earnings_from_eodhd(history, ci.symbol_segments(record), "2015-01-01", "2025-12-31")
    assert events["report_date"].tolist() == ["2025-12-31"]


def test_a_release_without_figures_keeps_its_date_with_nan_eps():
    record = {"A.N": {"segments": [{"symbol": "A.US"}]}}
    history = {"A.US": {"a": _release("2024-05-01", actual=None, estimate="n/a")}}
    events = ci.check_earnings(
        ci.earnings_from_eodhd(history, ci.symbol_segments(record), "2015-01-01", "2025-12-31")
    )
    assert len(events) == 1
    assert events["eps_actual"].isna().all() and events["eps_estimate"].isna().all()


def test_a_malformed_events_file_is_refused():
    good = pd.DataFrame(
        [["A.N", "2024-01-03", "after", "2023-12-31", 1.0, 0.9, "A.US"]],
        columns=list(ci.EARNINGS_COLUMNS),
    )
    ci.check_earnings(good)
    with pytest.raises(ValueError, match="lack columns"):
        ci.check_earnings(good.drop(columns=["timing"]))
    with pytest.raises(ValueError, match="unknown earnings timings"):
        ci.check_earnings(good.assign(timing="midday"))
    with pytest.raises(ValueError, match="not a YYYY-MM-DD"):
        ci.check_earnings(good.assign(report_date="03/01/2024"))
