"""S&P 500 membership spells, name checks and the 10-Ks a membership calls for (#309)."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

import scripts.data.build_sp500_members as build_members
import scripts.data.export_sec_member_10k as member_export
import scripts.data.export_sec_risk_factors as edgar_export
from mci_gru.filings.membership import (
    Spell,
    apply_changes,
    member_filings,
    name_key,
    names_match,
    names_on,
    spells,
    successors,
)


def _day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


LISTS = [
    (_day("2013-01-02"), frozenset({"AAA", "BBB", "FB"})),
    (_day("2015-06-01"), frozenset({"AAA", "FB"})),
    (_day("2018-03-01"), frozenset({"AAA", "BBB", "FB"})),
    (_day("2022-06-09"), frozenset({"AAA", "BBB", "META"})),
]


def test_a_spell_runs_from_the_first_list_to_the_first_list_without_the_ticker():
    found = spells(LISTS)
    assert [spell for spell in found if spell.ticker == "BBB"] == [
        Spell("BBB", _day("2013-01-02"), _day("2015-06-01")),
        Spell("BBB", _day("2018-03-01"), None),
    ]
    assert Spell("FB", _day("2013-01-02"), _day("2022-06-09")) in found
    assert Spell("META", _day("2022-06-09"), None) in found


def test_a_spell_holds_from_its_start_to_the_day_before_its_end():
    spell = Spell("BBB", _day("2013-01-02"), _day("2015-06-01"))
    assert spell.holds(_day("2013-01-02")) and spell.holds(_day("2015-05-31"))
    assert not spell.holds(_day("2015-06-01")) and not spell.holds(_day("2013-01-01"))
    assert Spell("META", _day("2022-06-09"), None).holds(_day("2030-01-01"))


def test_changes_after_the_last_list_extend_it_and_earlier_ones_are_ignored():
    extended = apply_changes(
        LISTS,
        [
            (_day("2020-01-01"), "OLD", None),
            (_day("2026-09-21"), "NEW", "AAA"),
            (_day("2026-10-06"), None, "BBB"),
        ],
    )
    assert extended[: len(LISTS)] == LISTS
    assert extended[len(LISTS) :] == [
        (_day("2026-09-21"), frozenset({"BBB", "META", "NEW"})),
        (_day("2026-10-06"), frozenset({"META", "NEW"})),
    ]


@pytest.mark.parametrize(
    ("name", "other", "expected"),
    [
        ("Pepco Holdings", "PEPCO HOLDINGS INC", True),
        ("The Coca-Cola Company", "COCA COLA CO", True),
        ("AT&T", "AT&T INC.", True),
        ("Estée Lauder Companies", "ESTEE LAUDER COMPANIES INC", True),
        ("Spectra Energy", "Sea Ltd", False),
        ("Pepco Holdings", "Pepco Holdings Class B Trust", True),
        ("Alphabet Inc.", "Inc.", False),
        ("Discovery", "Discover Financial Services", False),
        ("A. O. Smith", "SMITH A O CORP", True),
        ("IBM", "INTERNATIONAL BUSINESS MACHINES CORP", True),
        ("IBM", "INTEL CORP", False),
        ("Amazon", "AMAZON COM INC", True),
        ("Casey's", "CASEYS GENERAL STORES INC", True),
        ("Honeywell Technologies", "HONEYWELL INTERNATIONAL INC", False),
    ],
)
def test_names_match_once_legal_words_are_dropped(name, other, expected):
    assert names_match(name, other) is expected


def test_name_key_drops_case_punctuation_and_legal_words():
    assert name_key("The J.M. Smucker Company") == "j m smucker"


def test_a_membership_calls_for_its_reports_and_the_one_before_each_entry():
    spell_list = [
        Spell("BBB", _day("2013-01-02"), _day("2015-06-01")),
        Spell("BBB", _day("2018-03-01"), None),
    ]
    filings = [(_day(f"{year}-02-20"), f"acc-{year}") for year in range(2012, 2021)]
    assert member_filings(filings, spell_list) == [
        ("acc-2012", "prior"),
        ("acc-2013", "member"),
        ("acc-2014", "member"),
        ("acc-2015", "member"),
        ("acc-2018", "prior"),
        ("acc-2019", "member"),
        ("acc-2020", "member"),
    ]


def test_two_reports_in_one_year_are_both_kept_in_filing_order():
    spell_list = [Spell("BBB", _day("2013-01-02"), None)]
    filings = [
        (_day("2016-02-20"), "acc-b"),
        (_day("2015-02-20"), "acc-a"),
        (_day("2016-09-01"), "acc-transition"),
    ]
    assert member_filings(filings, spell_list) == [
        ("acc-a", "member"),
        ("acc-b", "member"),
        ("acc-transition", "member"),
    ]


def test_a_member_from_its_first_report_has_no_prior():
    spell_list = [Spell("NEW", _day("2014-01-02"), None)]
    assert member_filings([(_day("2014-03-01"), "acc-1")], spell_list) == [("acc-1", "member")]


SUBMISSIONS = {
    "name": "Walt Disney Co",
    "formerNames": [
        {
            "name": "DISNEY ENTERPRISES INC",
            "from": "1996-02-09T05:00:00.000Z",
            "to": "2019-03-19T04:00:00.000Z",
        },
        {
            "name": "WALT DISNEY CO/",
            "from": "2019-03-20T04:00:00.000Z",
            "to": "2019-03-21T04:00:00.000Z",
        },
    ],
}


def test_names_on_a_day_are_the_dated_names_then_the_current_one():
    assert names_on(SUBMISSIONS, _day("2013-11-20")) == ["DISNEY ENTERPRISES INC"]
    assert names_on(SUBMISSIONS, _day("2020-11-20")) == ["Walt Disney Co"]
    assert names_on({"name": "ACME INC"}, _day("2001-01-01")) == ["ACME INC"]


def _filer(cik, first, last, first_names, last_names):
    return (cik, _day(first), _day(last), first_names, last_names)


def test_a_new_holding_company_follows_its_predecessor():
    filers = [
        _filer("new", "2019-11-21", "2025-11-13", ["Walt Disney Co"], ["Walt Disney Co"]),
        _filer("old", "1997-12-19", "2018-11-21", ["WALT DISNEY CO"], ["WALT DISNEY CO"]),
    ]
    assert successors("new", filers) == ["old", "new"]


@pytest.mark.parametrize(
    "other",
    [
        # An acquirer that took the name had filed 10-Ks all along (ACE, renamed Chubb).
        _filer("ace", "1994-03-01", "2026-02-26", ["ACE LTD"], ["Chubb Ltd"]),
        # A spin-off began while its parent went on filing (Arconic from Howmet).
        _filer("spin", "2020-03-30", "2023-02-21", ["Arconic Corp"], ["Arconic Corp"]),
        # A company under another name took the ticker (Fox Corp after Twenty-First Century Fox).
        _filer("fox", "2019-08-10", "2026-08-06", ["Fox Corp"], ["Fox Corp"]),
        # A namesake long after (Sandisk spun off in 2025, SanDisk bought in 2016).
        _filer("namesake", "2025-08-01", "2026-08-01", ["Sandisk Corp"], ["Sandisk Corp"]),
    ],
)
def test_no_other_company_follows(other):
    anchor = {
        "ace": _filer("anchor", "2013-02-28", "2015-02-26", ["CHUBB CORP"], ["CHUBB CORP"]),
        "spin": _filer("anchor", "1994-03-11", "2026-02-13", ["ALCOA INC"], ["Arconic Inc"]),
        "fox": _filer(
            "anchor",
            "2005-09-01",
            "2018-08-13",
            ["NEWS CORP"],
            ["TWENTY-FIRST CENTURY FOX, INC."],
        ),
        "namesake": _filer(
            "anchor", "1996-03-01", "2016-02-12", ["SANDISK CORP"], ["SANDISK CORP"]
        ),
    }[other[0]]
    assert successors("anchor", [anchor, other]) == ["anchor"]


# The company table and the members' pull, against a fake EDGAR (#309).

AGENT = "Test Agent test@example.com"


def _submissions(name, reports, former=(), sic=""):
    """A submissions file whose 10-Ks are ``(accepted date, accession)``."""
    return {
        "name": name,
        "sic": sic,
        "formerNames": [{"name": old, "from": start, "to": end} for old, start, end in former],
        "filings": {
            "recent": {
                "accessionNumber": [accession for _, accession in reports],
                "form": ["10-K"] * len(reports),
                "filingDate": [when for when, _ in reports],
                "reportDate": [f"{int(when[:4]) - 1}-12-31" for when, _ in reports],
                "acceptanceDateTime": [f"{when}T21:00:00.000Z" for when, _ in reports],
                "primaryDocument": ["doc.htm"] * len(reports),
            },
            "files": [],
        },
    }


def _reports(cik, years):
    return [(f"{year}-02-20", f"{cik}-{year}") for year in years]


FAKE_FILERS = {
    "0000000001": _submissions("ACME INC", _reports("1", range(2012, 2017))),
    "0000000002": _submissions("OLDCO CORP", _reports("2", range(2012, 2016))),
    "0000000003": _submissions("NEWCO INC", _reports("3", range(2014, 2017))),
    "0000000005": _submissions("LATE CORP", []),
    # A company that took the old company's ticker years later.
    "0000000007": _submissions("TOTALLY DIFFERENT INC", _reports("7", range(2013, 2017))),
}


class FakeSec:
    def __init__(self, filers):
        self.filers = filers

    def __call__(self, url, headers):
        for cik, body in self.filers.items():
            if url.endswith(f"CIK{cik}.json"):
                return json.dumps(body).encode("utf-8")
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)


def _html_table(columns, rows):
    head = "".join(f"<th>{column}</th>" for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _build_inputs(tmp_path: Path, manual_rows=()) -> list[str]:
    (tmp_path / "lists.csv").write_text(
        'date,tickers\n2012-12-28,"AAA,OLD"\n2015-06-01,"AAA,NEW"\n', encoding="utf-8"
    )
    (tmp_path / "changes.html").write_text(
        _html_table(
            ["Date", "Added", "Added name", "Removed", "Removed name", "Reason"],
            [
                ["January 4, 2016", "LATE", "Late Corp", "", "", ""],
                ["June 1, 2015", "NEW", "Newco", "OLD", "Oldco", ""],
            ],
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.html").write_text(
        _html_table(
            ["Symbol", "Security", "CIK"],
            [["AAA", "Acme", "1"], ["NEW", "Newco", "3"], ["LATE", "Late Corp", "5"]],
        ),
        encoding="utf-8",
    )
    (tmp_path / "tickers.json").write_text(
        json.dumps(
            {
                "0": {"ticker": "AAA", "cik_str": 1},
                "1": {"ticker": "NEW", "cik_str": 3},
                "2": {"ticker": "OLD", "cik_str": 7},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "lookup.txt").write_text(
        "OLDCO CORP:0000000002:\nACME INC:0000000001:\n", encoding="latin-1"
    )
    manual = "ticker,start,name,cik,note\n" + "".join(f"{row}\n" for row in manual_rows)
    (tmp_path / "manual.csv").write_text(manual, encoding="utf-8")
    return [
        "--user-agent", AGENT,
        "--cache-dir", str(tmp_path / "cache"),
        "--member-lists", str(tmp_path / "lists.csv"),
        "--wiki-current", str(tmp_path / "current.html"),
        "--wiki-changes", str(tmp_path / "changes.html"),
        "--sec-tickers", str(tmp_path / "tickers.json"),
        "--cik-lookup", str(tmp_path / "lookup.txt"),
        "--manual", str(tmp_path / "manual.csv"),
        "--as-of", "2026-10-15",
        "--output-dir", str(tmp_path / "out"),
    ]  # fmt: skip


@pytest.fixture(autouse=False)
def _no_throttle(monkeypatch):
    monkeypatch.setattr(edgar_export, "MIN_INTERVAL_SECONDS", 0.0)


@pytest.mark.usefixtures("_no_throttle")
def test_wikipedia_tables_are_read_with_spans_filled_and_no_html_library(tmp_path):
    # Wikipedia's changes table gives a date shared by several changes one
    # cell with a rowspan, under two header rows; pandas' read_html needs
    # lxml, which CI's locked environment does not carry.
    (tmp_path / "changes.html").write_text(
        "<p>intro</p><table>"
        '<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th>'
        '<th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>'
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
        '<tr><td rowspan="2">March 23, 2026</td><td>NEW</td><td>New &amp; Co.<sup>[1]</sup></td>'
        "<td>OLD</td><td>Old<br>Corp</td><td>Cap</td></tr>"
        "<tr><td>TWO</td><td>Two Inc.</td><td></td><td></td><td>Cap</td></tr>"
        "</table><table><tr><td>a later table</td></tr></table>",
        encoding="utf-8",
    )
    changes = build_members.read_wiki_changes(tmp_path / "changes.html")

    assert changes.to_dict("records") == [
        {
            "date": _day("2026-03-23"),
            "added": "NEW",
            "added_name": "New & Co.[1]",
            "removed": "OLD",
            "removed_name": "Old Corp",
        },
        {
            "date": _day("2026-03-23"),
            "added": "TWO",
            "added_name": "Two Inc.",
            "removed": None,
            "removed_name": None,
        },
    ]


def test_every_run_is_named_and_a_reused_ticker_is_not_taken(tmp_path):
    assert build_members.main(_build_inputs(tmp_path), transport=FakeSec(FAKE_FILERS)) == 0
    runs = pd.read_csv(tmp_path / "out" / "sp500_ticker_runs.csv", dtype=str).fillna("")
    by_ticker = {row.ticker: row for row in runs.itertuples(index=False)}
    assert by_ticker["AAA"].ciks == "0000000001" and by_ticker["AAA"].method == "wiki_cik"
    # SEC's ticker file gives OLD to another company now; the name check refuses it.
    assert by_ticker["OLD"].ciks == "0000000002" and by_ticker["OLD"].method == "name_lookup"
    assert by_ticker["NEW"].start == "2015-06-01"
    # The change after the last member list adds LATE; it has filed no 10-K yet.
    assert by_ticker["LATE"].status == "no 10-K while a member"
    companies = pd.read_csv(tmp_path / "out" / "sp500_companies.csv", dtype=str).fillna("")
    assert companies.set_index("key")["release"].to_dict() == {
        "aaa": "r1",
        "new": "r1",
        "old": "r2",
    }
    assert companies.set_index("key").loc["old", "spells"] == "OLD:2012-12-28:2015-06-01"


@pytest.mark.usefixtures("_no_throttle")
def test_a_manual_cik_whose_reports_carry_another_name_is_refused(tmp_path):
    argv = _build_inputs(tmp_path, ["OLD,2012-12-28,Oldco,0000000007,wrong company"])
    build_members.main(argv, transport=FakeSec(FAKE_FILERS))
    runs = pd.read_csv(tmp_path / "out" / "sp500_ticker_runs.csv", dtype=str).fillna("")
    old = runs[runs["ticker"] == "OLD"].iloc[0]
    assert old["status"] == "manual row not confirmed" and old["ciks"] == ""


def test_a_ticker_passed_to_another_company_hands_over_its_part_of_the_run():
    run = Spell("CB", _day("1996-01-02"), None)
    tried = {
        "ace": ("wiki_cik", [(_day(f"{year}-02-26"), f"ace-{year}") for year in range(2016, 2020)], "", 0),
        "chubb": ("name_lookup", [(_day(f"{year}-02-26"), f"chubb-{year}") for year in range(2013, 2016)], "", 0),
        "one": ("name_lookup", [(_day("2012-03-01"), "one-2012")], "", 0),
        # Its own reports end before the chosen company's; later it only co-filed
        # them (Xerox Corp under Xerox Holdings).
        "joint": ("name_lookup", [(_day("2013-02-26"), "joint-2013"), (_day("2014-02-26"), "joint-2014"),
                                  (_day("2017-02-26"), "ace-2017"), (_day("2018-02-26"), "ace-2018")], "", 0),
    }  # fmt: skip
    parts = build_members._handovers(run, ["ace"], list(tried), tried)
    before = Spell("CB", _day("1996-01-02"), _day("2016-02-26"))
    assert parts == [(before, "chubb"), (before, "joint")]


FIXTURE = Path(__file__).parent / "fixtures" / "filings" / "tenk_item1a_sample.htm"


class FakeEdgar(FakeSec):
    """Submissions as ``FakeSec``, and every primary document the fixture 10-K."""

    def __init__(self, filers, broken=(), documents=None):
        super().__init__(filers)
        self.broken = set(broken)
        self.documents = documents or {}

    def __call__(self, url, headers):
        if url.endswith("/doc.htm"):
            if any(accession.replace("-", "") in url for accession in self.broken):
                return b"<html><body><div>ITEM 1. BUSINESS</div></body></html>"
            for accession, document in self.documents.items():
                if accession.replace("-", "") in url:
                    return document.encode("utf-8")
            return FIXTURE.read_bytes()
        return super().__call__(url, headers)


MEMBER_FILERS = {
    "0000000001": _submissions("ACME INC", _reports("1", range(2011, 2017)), sic="2834"),
    "0000000002": _submissions("BETA CORP", _reports("2", range(2011, 2014)), sic="6021"),
    "0000000003": _submissions("BETA HOLDINGS", _reports("3", range(2014, 2017)), sic="6021"),
    "0000000004": _submissions("GAMMA INC", _reports("4", range(2011, 2017))),
}


def _member_inputs(tmp_path: Path, extra: list[str] | None = None) -> list[str]:
    pd.DataFrame(
        [
            {"key": "aaa", "name": "ACME INC", "ciks": "0000000001",
             "tickers": "AAA", "spells": "AAA:2014-01-02:", "release": "r1"},
            {"key": "bbb", "name": "BETA HOLDINGS", "ciks": "0000000002;0000000003",
             "tickers": "BBB", "spells": "BBB:2012-12-28:2015-06-01", "release": "r1"},
            {"key": "ccc", "name": "GAMMA INC", "ciks": "0000000004",
             "tickers": "CCC", "spells": "CCC:2015-03-01:2015-09-01", "release": "r1"},
            {"key": "ddd", "name": "DELISTED", "ciks": "0000000004",
             "tickers": "DDD", "spells": "DDD:2013-01-02:", "release": "r2"},
        ]
    ).to_csv(tmp_path / "companies.csv", index=False)  # fmt: skip
    return [
        "--user-agent", AGENT,
        "--companies", str(tmp_path / "companies.csv"),
        "--release", "r1",
        "--package-root", str(tmp_path / "package"),
        "--cache-dir", str(tmp_path / "cache"),
        "--accepted-from", "2013-01-01",
        "--accepted-to", "2015-12-31",
        "--manifest-output", str(tmp_path / "manifest.json"),
    ] + (extra or [])  # fmt: skip


@pytest.mark.usefixtures("_no_throttle")
def test_a_members_reports_are_those_filed_while_in_the_index_and_the_one_before(tmp_path):
    assert member_export.main(_member_inputs(tmp_path), transport=FakeEdgar(MEMBER_FILERS)) == 0
    index = pd.read_csv(tmp_path / "package" / "member_10k_index.csv", dtype=str).fillna("")
    rows = {
        (row.company, row.accession_number): (row.role, row.prior_accession)
        for row in index.itertuples(index=False)
    }
    assert rows == {
        # Joined in January 2014: its 2013 report is the pair partner, and the
        # window ends with 2015.
        ("aaa", "1-2013"): ("prior", ""),
        ("aaa", "1-2014"): ("member", "1-2013"),
        ("aaa", "1-2015"): ("member", "1-2014"),
        # A member from before the window: 2012 is the first report's partner,
        # and the successor's first report pairs with its predecessor's last.
        ("bbb", "2-2012"): ("prior", ""),
        ("bbb", "2-2013"): ("member", "2-2012"),
        ("bbb", "3-2014"): ("member", "2-2013"),
        ("bbb", "3-2015"): ("member", "3-2014"),
    }
    left_out = pd.read_csv(tmp_path / "package" / "member_left_out.csv", dtype=str)
    assert left_out["company"].tolist() == ["ccc"]
    risks = (tmp_path / "package" / "member_risk_factors.jsonl").read_text().splitlines()
    assert {json.loads(line)["company"] for line in risks} == {"aaa", "bbb"}
    meta = json.loads((tmp_path / "package" / "member_10k.meta.json").read_text())
    assert meta["member_reports_by_year"] == {"2013": 1, "2014": 2, "2015": 2}
    assert (tmp_path / "manifest.json").exists()


@pytest.mark.usefixtures("_no_throttle")
def test_a_report_that_fails_the_checks_stops_the_manifest_unless_its_risks_are_left_out(
    tmp_path,
):
    transport = FakeEdgar(MEMBER_FILERS, broken={"1-2014"})
    assert member_export.main(_member_inputs(tmp_path), transport=transport) == 1
    assert not (tmp_path / "manifest.json").exists()
    assert (
        member_export.main(_member_inputs(tmp_path, ["--leave-out-flagged"]), transport=transport)
        == 0
    )
    left_out = pd.read_csv(tmp_path / "package" / "member_left_out.csv", dtype=str)
    assert left_out.set_index("company")["reason"].to_dict() == {
        "ccc": "no 10-K accepted while a member in the window",
        "aaa": "no Item 1A section closed by Item 1B, 1C or 2",
    }
    assert left_out.set_index("company").loc[
        "aaa", ["accession_number", "year", "sic"]
    ].tolist() == [
        "1-2014",
        "2014",
        "2834",
    ]
    # The report stays in the index, its risks out, and its company's other
    # reports in.
    index = pd.read_csv(tmp_path / "package" / "member_10k_index.csv", dtype=str)
    published = index.set_index("accession_number")["risks_published"].to_dict()
    assert published["1-2014"] == "False"
    assert published["1-2015"] == "True"
    risks = (tmp_path / "package" / "member_risk_factors.jsonl").read_text().splitlines()
    assert "1-2014" not in {json.loads(line)["accession_number"] for line in risks}
    assert "1-2015" in {json.loads(line)["accession_number"] for line in risks}
    meta = json.loads((tmp_path / "package" / "member_10k.meta.json").read_text())
    assert meta["member_reports_risks_left_out_by_year"]["2014"] == {
        "reports": 2,
        "left_out": 1,
        "share": 0.5,
    }
    assert meta["member_reports_risks_left_out_by_sic_division"] == {
        "Finance, Insurance and Real Estate": {"reports": 3, "left_out": 0, "share": 0.0},
        "Manufacturing": {"reports": 2, "left_out": 1, "share": 0.5},
    }
    assert (tmp_path / "manifest.json").exists()


def _item_1a(paragraphs: list[str]) -> str:
    body = "".join(f"<div>{paragraph}</div>" for paragraph in paragraphs)
    return (
        "<html><body><div>ITEM 1. BUSINESS</div><div>We make things.</div>"
        f"<div>ITEM 1A. RISK FACTORS</div>{body}"
        "<div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div><div>None.</div></body></html>"
    )


_FILLER = " ".join(["Demand for our products depends on conditions we cannot control."] * 20)
#: Headings run into their paragraphs (Ford), which only the run-in layout reads.
RUN_IN_10K = _item_1a([f"<b>Risk number {n} may harm us.</b> {_FILLER}" for n in range(1, 5)])
#: A long Item 1A with no emphasis at all: no layout splits it into risks.
PLAIN_10K = _item_1a([_FILLER] * 4)
#: An Item 1A that only points to the annual report (Bank of New York Mellon).
POINTER_10K = _item_1a(
    [
        "The information required by this Item is set forth under the caption Risk Factors "
        "in the Annual Report to Shareholders, which is incorporated herein by reference."
    ]
)


@pytest.mark.usefixtures("_no_throttle")
def test_other_layouts_are_tried_and_whole_sections_kept_where_risks_fail(tmp_path):
    transport = FakeEdgar(
        MEMBER_FILERS,
        documents={"1-2014": RUN_IN_10K, "1-2015": PLAIN_10K, "3-2015": POINTER_10K},
    )
    assert (
        member_export.main(_member_inputs(tmp_path, ["--leave-out-flagged"]), transport=transport)
        == 0
    )
    package = tmp_path / "package"
    index = pd.read_csv(package / "member_10k_index.csv", dtype=str).set_index("accession_number")
    assert index.loc["1-2014", "layout"] == "run_in"
    assert index.loc["1-2013", "layout"] == "blocks"
    risks = [json.loads(line) for line in (package / "member_risk_factors.jsonl").open()]
    assert [risk["heading"] for risk in risks if risk["accession_number"] == "1-2014"] == [
        f"Risk number {n} may harm us." for n in range(1, 5)
    ]
    sections = {
        section["accession_number"]: section
        for section in map(json.loads, (package / "member_item_1a.jsonl").open())
    }
    # The plain section is published whole though its risks are not; the
    # pointer, and the short fixture sections, are not sections at all.
    assert set(sections) == {"1-2014", "1-2015"}
    assert sections["1-2015"]["risks_passed"] is False
    assert sections["1-2014"]["risks_passed"] is True
    assert sections["1-2015"]["section_words"] == 4 * len(_FILLER.split())
    assert sections["1-2015"]["text"] == "\n\n".join([_FILLER] * 4)
    assert index.loc["3-2015", "section_published"] == "False"
    left_out = pd.read_csv(package / "member_left_out.csv", dtype=str).fillna("")
    reasons = left_out.set_index("accession_number")["reason"].to_dict()
    assert reasons["1-2015"] == "Item 1A holds no heading followed by text"
    assert reasons["3-2015"] == "Item 1A holds no heading followed by text"
    assert left_out.set_index("accession_number").loc["1-2015", "section_published"] == "True"
