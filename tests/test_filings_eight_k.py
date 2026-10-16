"""8-K events, their item sections and their filing-time context (#308).

The module tests feed the functions 8-Ks shaped like the ones E10's ten companies
filed; the export tests drive the whole pull against a fake EDGAR.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import scripts.data.export_sec_8k_events as export
import scripts.data.export_sec_risk_factors as export_10k
from mci_gru.data.input_manifest import read_input_manifest
from mci_gru.filings.eight_k import (
    MAX_TEXT_CHARS,
    ItemSection,
    document_text,
    event_text,
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
        ("<div>1.01 Entry into a Material Definitive Agreement</div>", None),
        ("<div><b>0.75% Notes due 2026</b></div>", None),
        ("<div><b>1.875% Notes due 2026</b></div>", None),
    ],
)
def test_which_lines_open_an_item_section(line, code):
    # Caterpillar heads "1.01 Entry into ..." in bold with no "Item"; Coca-Cola's
    # cover lists bold notes such as "0.75% Notes due 2026".
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


def test_a_listed_item_with_no_section_blocks_the_manifest(tmp_path, capsys):
    responses = _acme()
    url = f"{BASE}/000000007724000003/e2.htm"
    responses[url] = responses[url].replace(b"Item 5.02.", b"Departure")
    assert _run(tmp_path, Transport(responses)) == 1
    assert "item_section_missing: item 5.02 is listed but has no section" in (
        capsys.readouterr().out
    )
