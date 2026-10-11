"""The EDGAR filing index and when each filing was knowable (#289).

The pull needs EDGAR; these tests drive the same functions with submissions
blocks shaped like EDGAR's, and with the two acceptance timestamps that were
checked against their filing headers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mci_gru.data.pit import PredictionClock
from mci_gru.filings.edgar import (
    ALPHABET_CIK,
    GOOGLE_INC_CIK,
    FilingIndexError,
    build_filing_index,
    filings_from_block,
    first_forecast_session,
    parse_acceptance,
    submissions_blocks,
)


def _block(rows: list[tuple[str, str, str, str, str]]) -> dict[str, list[str]]:
    """Parallel arrays from (accession, form, report_date, acceptance, document) rows."""
    return {
        "accessionNumber": [row[0] for row in rows],
        "form": [row[1] for row in rows],
        "filingDate": [row[3][:10] for row in rows],
        "reportDate": [row[2] for row in rows],
        "acceptanceDateTime": [row[3] for row in rows],
        "primaryDocument": [row[4] for row in rows],
    }


def test_acceptance_is_true_utc_as_checked_against_filing_headers():
    """The header says 16:01:36 New York; the submissions file says 20:01:36Z (summer)."""
    summer = parse_acceptance("2026-07-22T20:01:36.000Z")
    winter = parse_acceptance("2026-02-04T21:01:37.000Z")
    assert summer.tz_convert("America/New_York").strftime("%H:%M:%S") == "16:01:36"
    assert winter.tz_convert("America/New_York").strftime("%H:%M:%S") == "16:01:37"


def test_naive_acceptance_is_refused():
    with pytest.raises(FilingIndexError):
        parse_acceptance("2026-07-22T20:01:36")


def test_filing_accepted_before_the_clock_counts_that_session():
    clock = PredictionClock()
    sessions = ["2026-07-21", "2026-07-22", "2026-07-23"]
    accepted = parse_acceptance("2026-07-22T20:01:36.000Z")  # 16:01 New York
    assert first_forecast_session(accepted, sessions, clock) == "2026-07-22"


def test_filing_accepted_after_the_clock_counts_the_next_session():
    clock = PredictionClock()
    sessions = ["2026-02-04", "2026-02-05"]
    accepted = parse_acceptance("2026-02-05T01:30:00.000Z")  # 20:30 New York on the 4th
    assert first_forecast_session(accepted, sessions, clock) == "2026-02-05"


def test_filing_accepted_exactly_at_the_clock_counts_that_session():
    """#225: equality counts."""
    clock = PredictionClock()
    accepted = clock.at("2026-02-04")
    assert first_forecast_session(accepted, ["2026-02-04", "2026-02-05"], clock) == "2026-02-04"


def test_filing_after_the_last_session_has_no_forecast():
    clock = PredictionClock()
    accepted = parse_acceptance("2026-02-06T15:00:00.000Z")
    assert first_forecast_session(accepted, ["2026-02-04", "2026-02-05"], clock) is None


def test_report_filed_under_both_ciks_is_kept_once_at_its_earliest_acceptance():
    google = filings_from_block(
        GOOGLE_INC_CIK,
        _block(
            [
                ("0001288776-15-000046", "10-Q", "2015-09-30", "2015-10-29T20:10:00.000Z", "g.htm"),
                ("0001288776-15-000010", "10-K", "2014-12-31", "2015-02-06T21:00:00.000Z", "k.htm"),
            ]
        ),
    )
    alphabet = filings_from_block(
        ALPHABET_CIK,
        _block(
            [
                ("0001652044-15-000005", "10-Q", "2015-09-30", "2015-10-29T20:12:00.000Z", "a.htm"),
                ("0001652044-16-000012", "10-K", "2015-12-31", "2016-02-11T21:00:00.000Z", "k.htm"),
            ]
        ),
    )
    index = build_filing_index(
        google + alphabet, forms=("10-K", "10-Q"), accepted_from="2014-01-01"
    )
    kept = [record.accession_number for record in index.filings]
    assert kept == ["0001288776-15-000010", "0001288776-15-000046", "0001652044-16-000012"]
    assert [record.accession_number for record in index.duplicates] == ["0001652044-15-000005"]


def test_tie_on_acceptance_goes_to_the_preferred_filer():
    same = "2015-10-29T20:10:00.000Z"
    records = filings_from_block(
        GOOGLE_INC_CIK, _block([("g-1", "10-Q", "2015-09-30", same, "g.htm")])
    ) + filings_from_block(ALPHABET_CIK, _block([("a-1", "10-Q", "2015-09-30", same, "a.htm")]))
    index = build_filing_index(records, forms=("10-Q",), accepted_from="2014-01-01")
    assert [record.cik for record in index.filings] == [ALPHABET_CIK]


def test_one_accession_listed_under_both_filers_is_kept_once_under_the_preferred_filer():
    same = "2015-10-29T20:10:00.000Z"
    records = filings_from_block(
        GOOGLE_INC_CIK, _block([("0001652044-15-000001", "10-Q", "2015-09-30", same, "g.htm")])
    ) + filings_from_block(
        ALPHABET_CIK, _block([("0001652044-15-000001", "10-Q", "2015-09-30", same, "a.htm")])
    )
    index = build_filing_index(records, forms=("10-Q",), accepted_from="2014-01-01")
    assert [record.cik for record in index.filings] == [ALPHABET_CIK]
    assert index.duplicates == ()


def test_index_keeps_only_requested_forms_from_the_start_date():
    records = filings_from_block(
        ALPHABET_CIK,
        _block(
            [
                ("a-1", "10-K", "2012-12-31", "2013-01-29T21:00:00.000Z", "old.htm"),
                ("a-2", "10-K", "2013-12-31", "2014-02-03T21:00:00.000Z", "k13.htm"),
                ("a-3", "8-K", "2014-04-16", "2014-04-16T20:05:00.000Z", "e.htm"),
                ("a-4", "4", "", "2014-04-17T20:05:00.000Z", "f4.xml"),
            ]
        ),
    )
    index = build_filing_index(records, forms=("10-K",), accepted_from="2014-01-01")
    assert [record.accession_number for record in index.filings] == ["a-2"]
    assert index.filings[0].fiscal_year == 2013


def test_older_submission_files_are_listed_for_reading():
    main = {
        "filings": {"recent": _block([]), "files": [{"name": "CIK0001652044-submissions-001.json"}]}
    }
    recent, names = submissions_blocks(main)
    assert recent == _block([])
    assert names == ["CIK0001652044-submissions-001.json"]


def test_ragged_submissions_block_is_refused():
    block = _block([("a-1", "10-K", "2013-12-31", "2014-02-03T21:00:00.000Z", "k.htm")])
    block["form"].append("10-Q")
    with pytest.raises(FilingIndexError):
        filings_from_block(ALPHABET_CIK, block)


def test_document_url_uses_unpadded_cik_and_bare_accession():
    record = filings_from_block(
        ALPHABET_CIK,
        _block(
            [
                (
                    "0001652044-26-000012",
                    "8-K",
                    "2026-02-04",
                    "2026-02-04T21:01:37.000Z",
                    "goog-20260204.htm",
                )
            ]
        ),
    )[0]
    assert record.document_url == (
        "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000012/goog-20260204.htm"
    )
    assert isinstance(record.accepted_utc, pd.Timestamp)
