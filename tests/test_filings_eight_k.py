"""8-K events, their item sections and their filing-time context (#308, #311).

The module tests feed the functions 8-Ks shaped like the ones E10's ten companies
filed; the export tests drive the whole pull against a fake EDGAR.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import scripts.data.export_sec_8k_events as export
import scripts.data.export_sec_member_8k as member_export
import scripts.data.export_sec_risk_factors as export_10k
from mci_gru.data.input_manifest import read_input_manifest
from mci_gru.filings.eight_k import (
    MAX_TEXT_CHARS,
    ItemSection,
    document_text,
    event_text,
    index_accepted,
    index_exhibits,
    is_event,
    item_sections,
    listed_items,
    press_release,
    prior_items,
    tagged_revenue,
)
from mci_gru.filings.filers import FILERS, Filer

if TYPE_CHECKING:
    from pathlib import Path

AGENT = "MCI-GRU tests tests@example.com"
COVER = (
    "<html><body><div><b>FORM 8-K</b></div><div><b>ACME CORP</b></div>"
    "<div>Check the appropriate box below.</div>"
)
SIGNATURE = (
    "<div><b>SIGNATURE</b></div><div>Pursuant to the requirements of the Securities "
    "Exchange Act of 1934, the registrant has duly caused this report to be signed.</div>"
    "<div>/s/ Jane Doe</div></body></html>"
)


def _eight_k(body: str) -> str:
    return COVER + body + SIGNATURE


def test_listed_items_keep_their_order_without_blanks():
    assert listed_items("5.02,9.01") == ("5.02", "9.01")
    assert listed_items(" 1.01, ,2.03 ") == ("1.01", "2.03")
    assert listed_items("") == ()


@pytest.mark.parametrize(
    ("form", "items", "expected"),
    [
        ("8-K", ("5.02", "9.01"), True),
        ("8-K", ("1.01",), True),
        ("8-K/A", ("5.02",), False),
        ("8-K", ("2.02", "9.01"), False),
        ("10-K", (), False),
    ],
)
def test_an_event_is_an_original_8k_with_an_event_item(form, items, expected):
    assert is_event(form, items) is expected


def test_item_sections_run_from_their_heading_to_the_next_item_or_the_signature():
    sections = item_sections(
        _eight_k(
            "<div><b>Item 5.02. Departure of Directors or Certain Officers.</b></div>"
            "<div>Jane Smith will retire on June 30.</div><div>John Lee will succeed her.</div>"
            "<div><b>Item 9.01. Financial Statements and Exhibits.</b></div>"
            "<div>99.1 Press release</div>"
        )
    )
    assert [(section.code, section.body) for section in sections] == [
        ("5.02", "Jane Smith will retire on June 30.\nJohn Lee will succeed her."),
        ("9.01", "99.1 Press release"),
    ]
    assert sections[0].text.startswith("Item 5.02. Departure")


def test_a_signature_without_a_title_still_ends_the_last_section():
    # Caterpillar's January 2018 8-K opens its signature block with no title.
    sections = item_sections(
        COVER + "<div><b>Item 5.02. Departure of Directors.</b></div>"
        "<div>A director retired.</div>"
        "<div>Pursuant to the requirements of the Securities Exchange Act of 1934, the "
        "registrant has duly caused this report to be signed.</div><div>By: /s/ A. B.</div>"
        "</body></html>"
    )
    assert [(section.code, section.body) for section in sections] == [
        ("5.02", "A director retired.")
    ]


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("<div><b>1.01 Entry into a Material Definitive Agreement</b></div>", "1.01"),
        ("<div><b>ITEM 2.05 Costs Associated with Exit Activities</b></div>", "2.05"),
        ("<div>Item 5.02(b) Departure of Directors</div>", "5.02"),
        ("<div><b>Item 5.02Departure of Directors</b></div>", "5.02"),
        ("<div>1.01 Entry into a Material Definitive Agreement</div>", None),
        ("<div><b>0.75% Notes due 2026</b></div>", None),
        ("<div><b>1.875% Notes due 2026</b></div>", None),
    ],
)
def test_which_lines_open_an_item_section(line, code):
    # Caterpillar heads "1.01 Entry into ..." in bold with no "Item"; Agilent runs
    # the title on with no space; Coca-Cola's cover lists bold notes such as
    # "0.75% Notes due 2026".
    sections = item_sections(_eight_k(line + "<div>Terms were agreed.</div>"))
    assert [section.code for section in sections] == ([code] if code else [])


def test_one_item_may_have_several_sections():
    # JPMorgan heads "Item 5.02(b)" and "Item 5.02(d)" apart.
    sections = item_sections(
        _eight_k(
            "<div><b>Item 5.02(b) Departure of Directors</b></div><div>A director retired.</div>"
            "<div><b>Item 5.02(d) Election of Directors</b></div><div>A director joined.</div>"
        )
    )
    assert [(section.code, section.body) for section in sections] == [
        ("5.02", "A director retired."),
        ("5.02", "A director joined."),
    ]


def test_a_heading_followed_straight_by_another_items_shares_its_text():
    # Fifth Third heads "Item 1.01" and "Item 2.03" over one account of a deal.
    sections = item_sections(
        _eight_k(
            "<div><b>Item 1.01 Entry into a Material Definitive Agreement</b></div>"
            "<div><b>Item 2.03 Creation of a Direct Financial Obligation</b></div>"
            "<div>The bank issued notes.</div>"
            "<div><b>Item 5.02 Departure of Directors</b></div>"
            "<div><b>Item 9.01 Financial Statements and Exhibits</b></div><div>99.1</div>"
            "<div><b>Item 8.01 Other Events</b></div>"
        )
    )
    assert [(section.code, section.body, section.stacked) for section in sections] == [
        ("1.01", "", True),
        ("2.03", "The bank issued notes.", False),
        # Item 9.01's text is left out, and the signature ends the last item.
        ("5.02", "", False),
        ("9.01", "99.1", False),
        ("8.01", "", False),
    ]


def test_edgars_wrapper_header_is_not_text():
    html = (
        "<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>ex991.htm\n"
        "<DESCRIPTION>EX-99.1\n<TEXT>\n<html><body><p>Acme names a new CFO.</p></body></html>"
    )
    assert document_text(html) == "Acme names a new CFO."


INDEX = (
    "<table><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>"
    "<tr><td>1</td><td>8-K</td><td><a>acme-8k.htm</a> &nbsp;iXBRL</td><td>8-K</td></tr>"
    "{rows}</table>"
)


def _index(*exhibits: tuple[str, str]) -> str:
    rows = "".join(
        f"<tr><td>2</td><td>Exhibit</td><td><a>{name}</a></td><td>{kind}</td></tr>"
        for kind, name in exhibits
    )
    return INDEX.format(rows=rows)


@pytest.mark.parametrize(
    ("exhibits", "expected"),
    [
        ((("EX-99.2", "b.htm"), ("EX-99.1", "a.htm")), ("EX-99.1", "a.htm")),
        ((("EX-99", "a.htm"),), ("EX-99", "a.htm")),
        ((("EX-99.01", "a.htm"),), ("EX-99.01", "a.htm")),
        ((("EX-99", "a.htm"), ("EX-99", "b.htm")), None),
        ((("EX-99.2", "b.htm"),), None),
        ((("EX-10.1", "c.htm"),), None),
    ],
)
def test_the_press_release_is_exhibit_99_1_or_a_lone_unnumbered_99(exhibits, expected):
    listed = index_exhibits(_index(*exhibits))
    assert listed[0] == ("8-K", "acme-8k.htm")
    assert press_release(listed) == expected


def test_event_text_leaves_out_the_exhibit_list_and_adds_the_press_release():
    sections = [
        ItemSection("5.02", "Item 5.02. Departure", "Jane Smith retired."),
        ItemSection("9.01", "Item 9.01. Exhibits", "99.1 Press release"),
    ]
    text, truncated = event_text(sections, "Acme thanks Jane.", "EX-99.1")
    assert text == "Item 5.02. Departure\nJane Smith retired.\n\nExhibit 99.1\nAcme thanks Jane."
    assert truncated is False


def test_event_text_is_cut_at_the_limit_and_says_so():
    sections = [ItemSection("1.01", "Item 1.01. Agreement", "terms " * 5000)]
    text, truncated = event_text(sections, None)
    assert len(text) == MAX_TEXT_CHARS
    assert truncated is True
    exact, cut = event_text([ItemSection("1.01", "x" * MAX_TEXT_CHARS, "")], None)
    assert (len(exact), cut) == (MAX_TEXT_CHARS, False)


def _at(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_prior_items_are_the_other_8ks_of_the_90_days_before_oldest_first():
    filings = [
        (_at("2024-01-01T12:00"), "a-1", ("8.01",)),  # 91 days before
        (_at("2024-01-02T12:00"), "a-2", ("2.02", "9.01")),  # exactly 90 days before
        (_at("2024-03-01T12:00"), "a-4", ("7.01",)),
        (_at("2024-02-01T12:00"), "a-3", ("5.02",)),
        (_at("2024-04-01T12:00"), "a-5", ("5.02",)),  # the event itself
        (_at("2024-04-01T13:00"), "a-6", ("1.01",)),  # after it
    ]
    assert prior_items(filings, _at("2024-04-01T12:00"), "a-5") == [
        ["2.02", "9.01"],
        ["5.02"],
        ["7.01"],
    ]


def _fact(accession: str, start: str, end: str, value: float, form: str = "10-K") -> dict:
    return {"accn": accession, "start": start, "end": end, "val": value, "form": form}


def _facts(**concepts: list[dict]) -> dict:
    return {
        "facts": {"us-gaap": {name: {"units": {"USD": facts}} for name, facts in concepts.items()}}
    }


def test_revenue_is_the_10ks_own_full_year_not_a_comparative_or_a_quarter():
    facts = _facts(
        Revenues=[
            _fact("k-24", "2023-01-01", "2023-12-31", 90.0),  # comparative year
            _fact("k-24", "2024-10-01", "2024-12-31", 30.0),  # fourth quarter
            _fact("k-24", "2024-01-01", "2024-12-31", 100.0),
            _fact("k-25", "2025-01-01", "2025-12-31", 120.0),  # a later 10-K
        ]
    )
    assert tagged_revenue(facts, "k-24") == {
        "value": 100.0,
        "concept": "Revenues",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "fiscal_year": 2024,
    }


def test_revenue_falls_back_through_the_concepts_a_filing_tags():
    # Pfizer tagged SalesRevenueNet, later SalesRevenueGoodsNet, before Revenues.
    facts = _facts(
        Revenues=[_fact("k-22", "2022-01-01", "2022-12-31", 100.0)],
        SalesRevenueGoodsNet=[_fact("k-15", "2015-01-01", "2015-12-31", 50.0)],
    )
    assert tagged_revenue(facts, "k-15")["concept"] == "SalesRevenueGoodsNet"
    assert tagged_revenue(facts, "k-99") is None


def test_a_year_ending_in_january_is_that_years_fiscal_year():
    # Walmart's year ending 31 January 2024 is its fiscal 2024; a 52-week year
    # ending in the first week of January belongs to the year before.
    walmart = _facts(Revenues=[_fact("k", "2023-02-01", "2024-01-31", 648.0)])
    assert tagged_revenue(walmart, "k")["fiscal_year"] == 2024
    weeks = _facts(Revenues=[_fact("k", "2015-01-05", "2016-01-03", 70.0)])
    assert tagged_revenue(weeks, "k")["fiscal_year"] == 2015


# The export, against a fake EDGAR.

ACME = Filer("acme8k", "Acme Corp", ("0000000077",))
BASE = "https://www.sec.gov/Archives/edgar/data/77"


def _submissions(rows) -> dict:
    return {
        "accessionNumber": [r[0] for r in rows],
        "form": [r[1] for r in rows],
        "filingDate": [r[3][:10] for r in rows],
        "reportDate": [r[2] for r in rows],
        "acceptanceDateTime": [r[3] for r in rows],
        "primaryDocument": [r[4] for r in rows],
        "items": [r[5] for r in rows],
    }


def _acme(broken: bool = False) -> dict[str, bytes]:
    main = {
        "filings": {
            "recent": _submissions(
                [
                    (
                        "0000000077-24-000001",
                        "10-K",
                        "2023-12-31",
                        "2024-02-20T21:00:00.000Z",
                        "k23.htm",
                        "",
                    ),
                    (
                        "0000000077-24-000002",
                        "8-K",
                        "",
                        "2024-03-01T21:00:00.000Z",
                        "e1.htm",
                        "2.02,9.01",
                    ),
                    (
                        "0000000077-24-000003",
                        "8-K",
                        "",
                        "2024-04-01T21:00:00.000Z",
                        "e2.htm",
                        "5.02,9.01",
                    ),
                    (
                        "0000000077-24-000004",
                        "8-K/A",
                        "",
                        "2024-04-05T21:00:00.000Z",
                        "e3.htm",
                        "5.02",
                    ),
                    (
                        "0000000077-25-000001",
                        "10-K",
                        "2024-12-31",
                        "2025-02-20T21:00:00.000Z",
                        "k24.htm",
                        "",
                    ),
                    (
                        "0000000077-12-000001",
                        "8-K",
                        "",
                        "2012-06-01T21:00:00.000Z",
                        "e0.htm",
                        "5.02",
                    ),
                ]
            )
        }
    }
    facts = _facts(
        Revenues=[
            _fact("0000000077-24-000001", "2023-01-01", "2023-12-31", 500.0),
            _fact("0000000077-25-000001", "2024-01-01", "2024-12-31", 900.0),
        ]
    )
    event = _eight_k(
        "<div><b>Item 5.02. Departure of Directors or Certain Officers.</b></div>"
        + ("" if broken else "<div>Jane Smith resigned effective immediately.</div>")
        + "<div><b>Item 9.01. Financial Statements and Exhibits.</b></div><div>99.1</div>"
    )
    return {
        "https://data.sec.gov/submissions/CIK0000000077.json": json.dumps(main).encode(),
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000077.json": json.dumps(facts).encode(),
        f"{BASE}/000000007724000003/e2.htm": event.encode(),
        f"{BASE}/000000007724000003/0000000077-24-000003-index.htm": _index(
            ("EX-99.1", "pr.htm")
        ).encode(),
        f"{BASE}/000000007724000003/pr.htm": b"<html><body><p>Acme names an interim CEO.</p></body></html>",
    }


class Transport:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.seen: list[str] = []

    def __call__(self, url: str, headers: dict[str, str]) -> bytes:
        assert headers["User-Agent"] == AGENT
        self.seen.append(url)
        return self.responses[url]


@pytest.fixture(autouse=True)
def _acme_filer(monkeypatch):
    monkeypatch.setattr(export_10k, "MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setitem(FILERS, ACME.key, ACME)


def _run(tmp_path: Path, transport: Transport) -> int:
    return export.main(
        [
            "--user-agent",
            AGENT,
            "--package-root",
            str(tmp_path / "package"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--filer",
            ACME.key,
            "--manifest-output",
            str(tmp_path / "manifest.json"),
        ],
        transport=transport,
    )


def test_an_event_carries_its_text_and_only_what_was_public_when_it_was_filed(tmp_path):
    transport = Transport(_acme())
    assert _run(tmp_path, transport) == 0
    lines = (tmp_path / "package" / "8k_events.jsonl").read_text().splitlines()
    assert len(lines) == 1  # the 2.02 8-K, the 8-K/A and the 2012 8-K are not events
    event = json.loads(lines[0])
    assert {
        key: event[key]
        for key in (
            "event_id",
            "company",
            "company_name",
            "items",
            "text",
            "truncated",
            "prior_revenue_usd",
            "prior_revenue_fiscal_year",
            "prior_10k_accession",
            "prior_90d_items",
            "accepted_utc",
        )
    } == {
        "event_id": "0000000077-24-000003",
        "company": "acme8k",
        "company_name": "Acme Corp",
        "items": ["5.02", "9.01"],
        "text": "Item 5.02. Departure of Directors or Certain Officers.\n"
        "Jane Smith resigned effective immediately.\n\nExhibit 99.1\nAcme names an interim CEO.",
        "truncated": False,
        # The fiscal 2024 10-K came out after the event; fiscal 2023's is the latest before.
        "prior_revenue_usd": 500.0,
        "prior_revenue_fiscal_year": 2023,
        "prior_10k_accession": "0000000077-24-000001",
        "prior_90d_items": [["2.02", "9.01"]],
        "accepted_utc": "2024-04-01T21:00:00+00:00",
    }
    manifest = read_input_manifest(tmp_path / "manifest.json")
    assert manifest.manifest.package_id == "8k_events"
    assert not any(url.endswith(("e1.htm", "e3.htm", "e0.htm")) for url in transport.seen)


def test_an_item_with_no_text_blocks_the_manifest(tmp_path, capsys):
    assert _run(tmp_path, Transport(_acme(broken=True))) == 1
    assert not (tmp_path / "manifest.json").exists()
    assert "BLOCKING 0000000077-24-000003: item_section_empty" in capsys.readouterr().out
    checks = pd.read_csv(tmp_path / "package" / "8k_events_checks.csv")
    assert checks["check"].tolist() == ["item_section_empty"]


def test_an_item_heading_stacked_on_the_next_items_passes_the_checks(tmp_path):
    responses = _acme()
    url = f"{BASE}/000000007724000003/e2.htm"
    responses[url] = responses[url].replace(
        b"<div>Jane Smith",
        b"<div><b>Item 5.02(e) Compensatory Arrangements.</b></div><div>Jane Smith",
    )
    assert _run(tmp_path, Transport(responses)) == 0
    event = json.loads((tmp_path / "package" / "8k_events.jsonl").read_text())
    assert event["text"].startswith(
        "Item 5.02. Departure of Directors or Certain Officers.\n\n"
        "Item 5.02(e) Compensatory Arrangements.\nJane Smith resigned"
    )


def test_a_listed_item_with_no_section_blocks_the_manifest(tmp_path, capsys):
    responses = _acme()
    url = f"{BASE}/000000007724000003/e2.htm"
    responses[url] = responses[url].replace(b"Item 5.02.", b"Departure")
    assert _run(tmp_path, Transport(responses)) == 1
    assert "item_section_missing: item 5.02 is listed but has no section" in (
        capsys.readouterr().out
    )


# The acceptance time a filing's index page shows, and the members' pull (#311).


def _index_page(accepted: str, *exhibits: tuple[str, str]) -> str:
    """An index page as EDGAR lays it out, with its Eastern acceptance time."""
    return (
        '<div class="formGrouping"><div class="infoHead">Filing Date</div>'
        '<div class="info">2014-03-03</div>\n<div class="infoHead">Accepted</div>\n'
        f'<div class="info">{accepted}</div></div>' + _index(*exhibits)
    )


@pytest.mark.parametrize(
    ("accepted", "expected"),
    [
        # Eastern standard time in winter, daylight time in summer.
        ("2013-01-09 16:56:08", "2013-01-09T21:56:08+00:00"),
        ("2019-09-10 16:38:35", "2019-09-10T20:38:35+00:00"),
    ],
)
def test_the_index_page_gives_the_acceptance_time_in_eastern_time(accepted, expected):
    assert index_accepted(_index_page(accepted)).isoformat() == expected


def test_an_index_page_without_an_acceptance_time_gives_none():
    assert index_accepted(_index(("EX-99.1", "a.htm"))) is None


BETA = "https://www.sec.gov/Archives/edgar/data/88"


def _beta_8k(accession: str, listed: str, items: str = "5.02,9.01") -> tuple:
    return (accession, "8-K", "", listed, f"{accession[-3:]}.htm", items)


def _beta(broken: bool = False) -> dict[str, bytes]:
    """A member from 2014-01-02 until 2014-06-01, the first day it was out."""
    filings = [
        ("0000000088-14-000001", "10-K", "2013-12-31", "2014-02-20T21:00:00.000Z", "k.htm", ""),
        # SEC lists A five hours late: its index page says 16:00 Eastern.
        _beta_8k("0000000088-14-000011", "2014-03-04T02:00:00.000Z"),
        # B is listed in the first hour of 1 June Eastern; its page says 31 May.
        _beta_8k("0000000088-14-000012", "2014-06-01T05:00:00.000Z"),
        # C's page puts it on 1 June, the first day out.
        _beta_8k("0000000088-14-000013", "2014-06-02T01:00:00.000Z"),
        # D is weeks after the company left.
        _beta_8k("0000000088-14-000014", "2014-07-15T20:00:00.000Z"),
        # Earnings 8-Ks (item 2.02): five calendar days after A by the Eastern
        # day, and six after it.
        _beta_8k("0000000088-14-000021", "2014-03-08T22:00:00.000Z", "2.02,9.01"),
        _beta_8k("0000000088-14-000022", "2014-03-09T22:00:00.000Z", "2.02,9.01"),
        # Another nearby 8-K that is not about earnings.
        _beta_8k("0000000088-14-000023", "2014-03-05T22:00:00.000Z", "8.01"),
        # E has an empty item section when ``broken``; F's document is gone.
        _beta_8k("0000000088-14-000015", "2014-04-01T20:00:00.000Z"),
        _beta_8k("0000000088-14-000016", "2014-04-15T20:00:00.000Z"),
    ]
    pages = {
        "0000000088-14-000011": "2014-03-03 16:00:00",
        "0000000088-14-000012": "2014-05-31 20:00:00",
        "0000000088-14-000013": "2014-06-01 09:00:00",
        "0000000088-14-000015": "2014-04-01 16:00:00",
        "0000000088-14-000016": "2014-04-15 16:00:00",
    }
    responses = {
        "https://data.sec.gov/submissions/CIK0000000088.json": json.dumps(
            {"sic": "2834", "filings": {"recent": _submissions(filings)}}
        ).encode(),
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000088.json": json.dumps(
            _facts(Revenues=[_fact("0000000088-14-000001", "2013-01-01", "2013-12-31", 700.0)])
        ).encode(),
    }
    for accession, accepted in pages.items():
        folder = f"{BETA}/{accession.replace('-', '')}"
        empty = broken and accession.endswith("15")
        body = "" if empty else f"<div>Event {accession[-2:]}: an officer resigned.</div>"
        if not accession.endswith("16"):
            responses[f"{folder}/{accession[-3:]}.htm"] = _eight_k(
                f"<div><b>Item 5.02. Departure of Officers.</b></div>{body}"
            ).encode()
        responses[f"{folder}/{accession}-index.htm"] = _index_page(accepted).encode()
    return responses


def _member_run(tmp_path: Path, responses: dict[str, bytes], *extra: str) -> int:
    tmp_path.mkdir(parents=True, exist_ok=True)
    companies = tmp_path / "companies.csv"
    companies.write_text(
        "key,name,ciks,tickers,spells,release\n"
        "beta,Beta Corp,0000000088,BETA,BETA:2014-01-02:2014-06-01,r2\n",
        encoding="utf-8",
    )
    return member_export.main(
        [
            "--user-agent", AGENT,
            "--companies", str(companies),
            "--package-root", str(tmp_path / "package"),
            "--cache-dir", str(tmp_path / "cache"),
            "--manifest-output", str(tmp_path / "manifest.json"),
            *extra,
        ],
        transport=Transport(responses),
    )  # fmt: skip


def test_a_members_events_are_those_accepted_while_a_member_by_their_index_page(tmp_path):
    assert _member_run(tmp_path, _beta(), "--leave-out-flagged") == 0
    package = tmp_path / "package"
    events = [
        json.loads(line) for line in (package / "member_8k_events.jsonl").read_text().splitlines()
    ]
    assert [
        (event["event_id"], event["accepted_utc"], event["listed_accepted_utc"]) for event in events
    ] == [
        ("0000000088-14-000011", "2014-03-03T21:00:00+00:00", "2014-03-04T02:00:00+00:00"),
        ("0000000088-14-000015", "2014-04-01T20:00:00+00:00", "2014-04-01T20:00:00+00:00"),
        ("0000000088-14-000012", "2014-06-01T00:00:00+00:00", "2014-06-01T05:00:00+00:00"),
    ]
    first = events[0]
    assert (first["company"], first["cik"], first["release"], first["sic"]) == (
        "beta",
        "0000000088",
        "r2",
        "2834",
    )
    assert first["text"].endswith("Event 11: an officer resigned.")
    assert first["nearby_202_listed_utc"] == ["2014-03-08T22:00:00+00:00"]
    assert events[1]["nearby_202_listed_utc"] == []
    assert (first["prior_revenue_usd"], first["prior_10k_accession"]) == (
        700.0,
        "0000000088-14-000001",
    )
    left_out = pd.read_csv(package / "member_8k_left_out.csv", dtype=str)
    assert left_out[["event_id", "year", "reason"]].values.tolist() == [
        ["0000000088-14-000016", "2014", "document_unreadable"]
    ]
    meta = json.loads((package / "member_8k_events.meta.json").read_text())
    assert meta["events_by_item"] == {"5.02": 3}
    assert meta["events_by_year"] == {"2014": 3}
    assert meta["events_by_year_and_item"] == {"2014": {"5.02": 3}}
    assert meta["outside_membership_by_index_page"] == 1
    assert meta["acceptance_corrected_from_index_page"] == 3
    assert meta["events_left_out_by_year"]["2014"] == {
        "events": 4,
        "left_out": 1,
        "share_left_out": 0.25,
    }
    assert read_input_manifest(tmp_path / "manifest.json").manifest.package_id == (
        "member_8k_events"
    )


def test_a_flagged_event_is_left_out_or_stops_the_manifest(tmp_path):
    assert _member_run(tmp_path / "out", _beta(broken=True), "--leave-out-flagged") == 0
    left_out = pd.read_csv(tmp_path / "out" / "package" / "member_8k_left_out.csv", dtype=str)
    assert dict(zip(left_out["event_id"], left_out["reason"], strict=True)) == {
        "0000000088-14-000015": "item_section_empty",
        "0000000088-14-000016": "document_unreadable",
    }
    lines = (tmp_path / "out" / "package" / "member_8k_events.jsonl").read_text().splitlines()
    assert [json.loads(line)["event_id"] for line in lines] == [
        "0000000088-14-000011",
        "0000000088-14-000012",
    ]
    assert _member_run(tmp_path / "kept", _beta(broken=True)) == 1
    assert not (tmp_path / "kept" / "manifest.json").exists()
