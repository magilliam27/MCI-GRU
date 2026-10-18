"""Russell 2000 membership from IWM's SEC holdings filings (E14)."""

from __future__ import annotations

import pandas as pd
import pytest

from mci_gru.filings.iwm_holdings import (
    HoldingsError,
    holding_key,
    nport_holdings,
    nq_holdings,
    price_error,
    split_factor,
)
from scripts.data.build_r2k_members import membership


def _row(*cells: str) -> str:
    return "<TR>" + "".join(f"<TD>{cell}</TD>" for cell in cells) + "</TR>"


NQ = (
    _row("iShares S&amp;P 500 Value ETF")
    + _row("Other Fund Corp.", "1,000", "2,000")
    + '<P><FONT SIZE="4">iSHARES<SUP>&reg;</SUP> RUSSELL 2000 ETF</FONT></P>'
    + _row("Security", "Shares", "Value")
    + _row("COMMON STOCKS &#151; 99.79%")
    + _row("ADVERTISING &#151; 0.11%")
    + _row("Harte-Hanks Inc.", "663,519", "$", "5,706,263")
    + _row('Marin Software Inc.<SUP STYLE="x">a</SUP>', "141,687", "1,450,875")
    + '<P><FONT SIZE="4">iSHARES<SUP>&reg;</SUP> RUSSELL 2000 ETF</FONT></P>'
    + _row("AAR Corp.", "613,349", "13,481,411")
    + _row("24,134,866")
    + _row("WARRANTS &#151; 0.00%")
    + _row("Magnum Hunter Resources Corp. (Expires 04/15/16)", "287,302", "3")
    + _row("TOTAL INVESTMENTS IN SECURITIES", "21,000,000,000")
    + _row("Not A Holding Inc.", "1", "1")
    + "<P>NET ASSETS</P>"
    + _row("iShares Russell 2000 ETF", "5,000", "600,000")
)


def test_the_n_q_schedule_runs_from_the_first_page_header_to_the_net_assets_line():
    table = nq_holdings(NQ)

    assert list(table["name"]) == [
        "Harte-Hanks Inc.",
        "Marin Software Inc.",
        "AAR Corp.",
        "Magnum Hunter Resources Corp. (Expires 04/15/16)",
    ]
    assert list(table["section"]) == ["COMMON STOCKS"] * 3 + ["WARRANTS"]
    assert table.iloc[0][["shares", "value"]].tolist() == [663519.0, 5706263.0]


def test_another_funds_holding_of_iwm_is_not_a_page_header():
    # Without the registered mark, "iShares Russell 2000 ETF" is a holding, not a header.
    with pytest.raises(HoldingsError):
        nq_holdings(_row("iShares Russell 2000 ETF", "5,000", "600,000") + "<P>NET ASSETS</P>")


NPORT = b"""<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport" xmlns:com="http://www.sec.gov/edgar/common"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sec.gov/edgar/nport x.xsd">
<formData><genInfo><seriesId>S000004344</seriesId><repPdEnd>2020-03-31</repPdEnd><repPdDate>2019-09-30</repPdDate></genInfo>
<invstOrSecs>
<invstOrSec><name>Blackbaud Inc</name><cusip>09227Q100</cusip><identifiers><isin value="US09227Q1004"/></identifiers>
<balance>1096504.00</balance><units>NS</units><valUSD>99058171.36</valUSD><assetCat>EC</assetCat><invCountry>US</invCountry></invstOrSec>
<invstOrSec><name>BlackRock Cash Funds</name><cusip>000000000</cusip><identifiers><other value="X"/></identifiers>
<balance>5.0</balance><valUSD>5.0</valUSD><assetCat>STIV</assetCat><invCountry>US</invCountry></invstOrSec>
</invstOrSecs></formData></edgarSubmission>"""


def test_an_n_port_gives_its_series_date_and_holdings_with_isin():
    series, date, table = nport_holdings(NPORT)

    assert (series, date) == ("S000004344", "2019-09-30")
    assert list(table["name"]) == ["Blackbaud Inc", "BlackRock Cash Funds"]
    assert table.iloc[0][["isin", "cusip", "asset_cat"]].tolist() == [
        "US09227Q1004",
        "09227Q100",
        "EC",
    ]
    assert pd.isna(table.iloc[1]["isin"])
    assert table.iloc[0]["value"] / table.iloc[0]["shares"] == pytest.approx(90.34, abs=0.01)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("HB Fuller Co.", "H.B. Fuller Company"),
        ("Casella Waste Systems Inc.Class A", "Casella Waste Systems Inc"),
        ("United Security Bancshares/Fresno CA", "United Security Bancshares"),
        ("TechTarget, Inc. Common Stock", "TechTarget Inc."),
        ("Aaron’s Inc.", "Aaron's Inc"),
        ("Piper Sandler Cos", "Piper Sandler Companies"),
    ],
)
def test_holding_keys_ignore_punctuation_state_suffixes_and_legal_words(left, right):
    assert holding_key(left) == holding_key(right)


def test_different_companies_keep_different_keys():
    assert holding_key("First Financial Corp.") != holding_key("First Financial Bankshares Inc.")


def test_a_split_after_the_report_date_is_undone_before_comparing():
    # Parker Drilling: the fund marked it at 6.52 on 30 June 2014; EODHD's close
    # is 97.80 after its 1-for-15 reverse split of July 2018.
    splits = [
        {"date": "1981-02-03", "split": "2.000000/1.000000"},
        {"date": "2018-07-27", "split": "1.000000/15.000000"},
    ]
    factor = split_factor(splits, "2014-06-30")

    assert factor == pytest.approx(1 / 15)
    assert price_error(6.52, 97.80) > 0.9
    assert price_error(6.52, 97.80, factor) < 0.001


def test_a_missing_or_zero_price_never_matches():
    assert price_error(10.0, 0.0) == float("inf")
    assert price_error(float("nan"), 10.0) == float("inf")


def _held(snap: str, cik: str, how: str = "price") -> dict:
    return {"snap": snap, "cik": cik, "name": f"Co {cik}", "symbol": f"S{cik}.US", "how": how}


def test_membership_runs_from_a_report_date_to_the_day_before_the_next_date_it_is_missing():
    rows = [
        _held("2012-12-31", "1"),
        _held("2013-06-30", "1"),
        _held("2013-06-30", "2"),
        _held("2013-12-31", "2"),
        _held("2014-06-30", "1"),
        _held("2014-06-30", "2"),
        _held("2012-12-31", "3"),
    ]
    frame = membership(pd.DataFrame(rows), "2013-07-01").set_index("cik")

    assert frame.at["1", "spells"] == "2012-12-31:2013-12-30;2014-06-30:"
    assert frame.at["2", "spells"] == "2013-06-30:"
    # Company 3 left (was missing on 30 June 2013) before the frame starts.
    assert "3" not in frame.index
