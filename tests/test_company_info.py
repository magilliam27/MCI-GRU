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

import numpy as np
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


# --- SEC first-filed fundamentals -------------------------------------------------


def _fact(val, end, filed, form="10-Q", start=None, accn=None):
    item = {"val": val, "end": end, "filed": filed, "form": form, "accn": accn or f"a-{filed}"}
    if start:
        item["start"] = start
    return item


def _companyfacts(us_gaap: dict, dei: dict | None = None) -> dict:
    def wrap(block, unit):
        return {name: {"units": {unit: items}} for name, items in block.items()}

    facts = {"us-gaap": wrap(us_gaap, "USD")}
    if dei:
        facts["dei"] = {name: {"units": {"shares": items}} for name, items in dei.items()}
    return {"cik": 1, "entityName": "Example Corp", "facts": facts}


def test_the_first_filed_value_wins_over_a_later_restatement():
    facts = ci.first_filed_facts(
        _companyfacts(
            {
                "Assets": [
                    _fact(110.0, "2019-12-31", "2021-02-25", form="10-K"),  # restated copy
                    _fact(100.0, "2019-12-31", "2020-02-20", form="10-K"),
                ]
            }
        )
    )
    assert facts[["field", "end", "value", "filed"]].values.tolist() == [
        ["assets", "2019-12-31", 100.0, "2020-02-20"]
    ]


def test_within_a_period_the_better_concept_wins_and_the_next_fills_gaps():
    facts = ci.first_filed_facts(
        _companyfacts(
            {
                "SalesRevenueNet": [
                    _fact(50.0, "2017-03-31", "2017-05-01", start="2017-01-01"),
                    _fact(55.0, "2018-03-31", "2018-05-01", start="2018-01-01"),
                ],
                "Revenues": [_fact(56.0, "2018-03-31", "2018-05-01", start="2018-01-01")],
            }
        )
    )
    revenue = facts.loc[facts["field"] == "revenue"].sort_values("end")
    assert revenue["value"].tolist() == [50.0, 56.0]
    assert revenue["concept"].tolist() == ["SalesRevenueNet", "Revenues"]


def test_values_from_registration_statements_are_left_out():
    facts = ci.first_filed_facts(
        _companyfacts({"Assets": [_fact(100.0, "2019-12-31", "2020-01-10", form="S-4")]})
    )
    assert facts.empty


def test_a_fourth_quarter_is_the_year_less_nine_months_known_when_both_were_filed():
    facts = ci.first_filed_facts(
        _companyfacts(
            {
                "NetIncomeLoss": [
                    _fact(10.0, "2019-03-31", "2019-05-01", start="2019-01-01"),
                    _fact(11.0, "2019-06-30", "2019-08-01", start="2019-04-01"),
                    _fact(12.0, "2019-09-30", "2019-11-01", start="2019-07-01"),
                    _fact(33.0, "2019-09-30", "2019-11-01", start="2019-01-01"),
                    _fact(50.0, "2019-12-31", "2020-02-20", start="2019-01-01", form="10-K"),
                ]
            }
        )
    )
    quarters = ci.quarterly_flows(facts, "net_income")
    assert quarters[["end", "value", "filed"]].values.tolist() == [
        ["2019-03-31", 10.0, "2019-05-01"],
        ["2019-06-30", 11.0, "2019-08-01"],
        ["2019-09-30", 12.0, "2019-11-01"],
        ["2019-12-31", 17.0, "2020-02-20"],
    ]


def test_share_counts_move_onto_the_panels_split_basis():
    splits = pd.DataFrame({"dt": ["2020-08-31"], "ratio": [4.0]})
    factor = ci.split_factor_after(["2020-06-30", "2020-08-31", "2020-09-30"], splits)
    assert factor.tolist() == [4.0, 1.0, 1.0]
    facts = ci.first_filed_facts(
        _companyfacts(
            {},
            dei={
                "EntityCommonStockSharesOutstanding": [
                    _fact(1000.0, "2020-07-17", "2020-07-31"),
                    _fact(4100.0, "2020-10-16", "2020-10-30", form="10-K"),
                ]
            },
        )
    )
    rows = ci.fundamental_rows(facts, "AAPL.OQ", "0000320193", splits)
    shares = rows.loc[rows["field"] == "shares_adj"].sort_values("end")
    assert shares["value"].tolist() == [4000.0, 4100.0]


def _year_of_filings() -> pd.DataFrame:
    """Five quarters of one company's first-filed rows, as the fundamentals file holds."""
    rows = []
    ends = ["2018-12-31", "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31"]
    filed = ["2019-02-20", "2019-05-01", "2019-08-01", "2019-11-01", "2020-02-20"]
    for i, (end, when) in enumerate(zip(ends, filed, strict=True)):
        rows += [
            ["X.N", "1", "equity", end, 50.0 + i, when],
            ["X.N", "1", "assets", end, 100.0 + 10 * i, when],
            ["X.N", "1", "shares_adj", end, 10.0, when],
            ["X.N", "1", "net_income_q", end, 2.0 + i, when],
            ["X.N", "1", "gross_profit_q", end, 5.0, when],
        ]
    return pd.DataFrame(rows, columns=list(ci.FUNDAMENTAL_ROW_COLUMNS))


def test_a_value_is_first_read_the_session_after_its_filing_date():
    sessions = ["2019-10-31", "2019-11-01", "2019-11-04", "2019-11-05"]
    assert ci.filed_known_session(["2019-11-01", "2019-11-02"], sessions).tolist() == [2, 2]


def test_the_timeline_uses_only_values_filed_before_each_session():
    sessions = list(pd.bdate_range("2019-01-01", "2020-06-30").strftime("%Y-%m-%d"))
    timeline = ci.fundamentals_timeline(_year_of_filings(), sessions)
    position = {s: i for i, s in enumerate(sessions)}
    assert timeline["session"].tolist() == [
        position[d] for d in ("2019-02-21", "2019-05-02", "2019-08-02", "2019-11-04", "2020-02-21")
    ]
    third, last = timeline.iloc[3], timeline.iloc[4]
    # By November 2019 four quarters are known: 2 + 3 + 4 + 5.
    assert third["net_income_ttm"] == 14.0 and third["newest_end"] == "2019-09-30"
    assert np.isnan(timeline.iloc[2]["net_income_ttm"])  # three quarters only
    assert last["net_income_ttm"] == 3.0 + 4.0 + 5.0 + 6.0
    assert last["gross_profit_ttm"] == 20.0
    assert last["asset_growth"] == pytest.approx(140.0 / 100.0 - 1.0)
    assert last["equity"] == 54.0


def test_four_quarters_with_a_gap_make_no_trailing_year():
    rows = _year_of_filings()
    rows = rows.loc[~((rows["field"] == "net_income_q") & (rows["end"] == "2019-06-30"))]
    sessions = list(pd.bdate_range("2019-01-01", "2020-06-30").strftime("%Y-%m-%d"))
    timeline = ci.fundamentals_timeline(rows, sessions)
    assert np.isnan(timeline.iloc[-1]["net_income_ttm"])
