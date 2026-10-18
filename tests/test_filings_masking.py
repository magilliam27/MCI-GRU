"""E13: what the name mask and the deep mask hide, and the leak check's options.

Nothing here calls Jev or loads a tagger: entity spans are given by hand.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

import scripts.data.identify_8k_company as identify
import scripts.data.mask_8k_events as mask_script
from mci_gru.filings.events import Event, event_state

if TYPE_CHECKING:
    from pathlib import Path

from mci_gru.filings.masking import (
    Identity,
    Vocabulary,
    company_vocabulary,
    core_tokens,
    display_name,
    mask_entities,
    mask_names,
    name_patterns,
    round_revenue,
    ticker_patterns,
)


def make_event(
    text: str, company: str = "ba", name: str = "BOEING CO", revenue: float = 76.6e9
) -> Event:
    return Event(
        event_id="0000012927-19-000099",
        company=company,
        company_name=name,
        items=("5.02",),
        text=text,
        prior_revenue_usd=revenue,
        prior_revenue_fiscal_year=2018,
        prior_90d_items=(),
    )


def patterns(identity: Identity, vocabulary: Vocabulary, own: set[str] = frozenset()):
    return (
        name_patterns(identity, vocabulary, own_words=set(own), own_capitals=set(own)),
        ticker_patterns(identity, vocabulary, own_words=set(own)),
    )


EMPTY = Vocabulary(lower={}, capital={})


def test_core_drops_suffixes_and_state_tags() -> None:
    assert core_tokens("CELGENE CORP /DE/") == ["CELGENE"]
    assert core_tokens("The Goldman Sachs Group, Inc.") == ["Goldman", "Sachs"]
    assert display_name("CELGENE CORP /DE/") == "Celgene Corp"
    assert display_name("AT&T INC.") == "AT&T Inc."


def test_name_mask_hides_the_filer_and_withholds_the_company_field() -> None:
    identity = Identity(names=("BOEING CO",), tickers=("BA",))
    event = make_event(
        "The Boeing Company (NYSE: BA) said Boeing’s CEO resigned. See www.boeing.com or "
        "write to ir@boeing.com. BOEING CO filed it."
    )
    masked, counts = mask_names(event, *patterns(identity, EMPTY))
    assert "oeing" not in masked.text and "BOEING" not in masked.text
    assert "NYSE: BA" not in masked.text
    assert "the Company’s CEO" in masked.text
    assert "[website]" in masked.text and "[email]" in masked.text
    assert masked.company_name == "withheld"
    assert event_state(masked)["company"] == "withheld"
    assert counts.names == 3 and counts.tickers == 1 and counts.web == 2


def test_a_common_word_core_is_masked_only_when_capitalised() -> None:
    vocabulary = Vocabulary(lower={"target": 500}, capital={"target": 500})
    identity = Identity(names=("TARGET CORP",), tickers=("TGT",))
    event = make_event(
        "Target Corporation set the target bonus. TARGET agreed.", "tgt", "TARGET CORP"
    )
    masked, _ = mask_names(event, *patterns(identity, vocabulary))
    assert masked.text == "The Company set the target bonus. The Company agreed."


def test_a_short_name_is_masked_alone_only_when_few_others_capitalise_it() -> None:
    identity = Identity(names=("SOUTHERN CO", "GOLDMAN SACHS GROUP INC"), tickers=())
    vocabulary = Vocabulary(lower={}, capital={"goldman": 5, "southern": 0})
    masked, _ = mask_names(make_event("Goldman hired a banker."), *patterns(identity, vocabulary))
    assert masked.text == "The Company hired a banker."
    crowded = Vocabulary(lower={}, capital={"goldman": 400})
    masked, _ = mask_names(make_event("Goldman hired a banker."), *patterns(identity, crowded))
    assert masked.text == "Goldman hired a banker."


def test_a_short_name_is_masked_only_when_capitalised() -> None:
    identity = Identity(names=("APARTMENT INVESTMENT & MANAGEMENT CO",), tickers=())
    masked, _ = mask_names(
        make_event("Apartment REIT bought an apartment."), *patterns(identity, EMPTY)
    )
    assert masked.text == "The Company REIT bought an apartment."


def test_an_ordinary_word_former_name_is_left_alone() -> None:
    vocabulary = Vocabulary(lower={"he": 700}, capital={"he": 700})
    identity = Identity(names=("RAYTHEON CO",), tickers=(), former=("HE HOLDINGS INC",))
    masked, _ = mask_names(
        make_event("He resigned. HE Holdings, Inc. was renamed."), *patterns(identity, vocabulary)
    )
    assert masked.text == "He resigned. The Company was renamed."


def test_a_former_name_is_masked_whole_but_never_by_its_first_word() -> None:
    identity = Identity(
        names=("SKYWORKS SOLUTIONS INC",), tickers=(), former=("ALPHA INDUSTRIES INC",)
    )
    text = "Alpha Industries, Inc. renamed itself. The alpha of the fund. Alpha Beta Inc. is a customer."
    masked, _ = mask_names(make_event(text), *patterns(identity, EMPTY))
    assert masked.text.startswith("The Company renamed itself.")
    assert "Alpha Beta Inc." in masked.text


def test_initials_of_a_long_name_are_masked_unless_an_ordinary_word() -> None:
    identity = Identity(names=("ADVANCED MICRO DEVICES INC",), tickers=())
    masked, _ = mask_names(make_event("AMD Appoints a director."), *patterns(identity, EMPTY))
    assert masked.text == "The Company Appoints a director."
    ordinary = Vocabulary(lower={"amd": 300}, capital={})
    masked, _ = mask_names(make_event("AMD Appoints a director."), *patterns(identity, ordinary))
    assert masked.text == "AMD Appoints a director."


def test_a_ticker_that_is_an_ordinary_word_is_masked_only_after_an_exchange() -> None:
    identity = Identity(names=("ALLSTATE CORP",), tickers=("ALL",))
    vocabulary = Vocabulary(lower={"all": 700}, capital={})
    event = make_event(
        "ALL RIGHTS RESERVED. Shares trade on the NYSE: ALL.", "all", "ALLSTATE CORP"
    )
    masked, counts = mask_names(event, *patterns(identity, vocabulary))
    assert masked.text == "ALL RIGHTS RESERVED. Shares trade on the [ticker]."
    assert counts.tickers == 1


def test_vocabulary_counts_filers_once_and_ignores_addresses() -> None:
    vocabulary = company_vocabulary(
        [("a", "the target rose"), ("a", "target again"), ("b", "Target at www.target.com")]
    )
    assert vocabulary.lower["target"] == 1
    assert vocabulary.capital["target"] == 1


def test_deep_mask_hides_people_and_places_but_keeps_titles_and_terms() -> None:
    text = (
        "Departure of Directors or Certain Officers. Mark Alles, Group President, of "
        "Summit, New Jersey, joins the Compensation Committee (the “Proxy Statement”). "
        "Call (908) 673-9969."
    )
    spans = []
    for found, label in [
        ("Certain Officers", "PERSON"),
        ("Mark Alles", "PERSON"),
        ("Group President", "ORG"),
        ("New Jersey", "GPE"),
        ("the Compensation Committee", "ORG"),
        ("the “Proxy Statement", "ORG"),
        ("1999", "DATE"),
    ]:
        start = text.find(found)
        if start >= 0:
            spans.append((start, start + len(found), label))
    masked, counts = mask_entities(make_event(text), spans)
    assert "Certain Officers" in masked.text and "Group President" in masked.text
    assert "Compensation Committee" in masked.text and "Proxy Statement" in masked.text
    assert "[person]" in masked.text and "[place]" in masked.text and "[phone]" in masked.text
    assert "Alles" not in masked.text and "Jersey" not in masked.text
    assert counts.entities == 2 and counts.phones == 1
    assert masked.prior_revenue_usd == 80e9


def test_revenue_keeps_one_significant_figure() -> None:
    assert round_revenue(76.6e9) == 80e9
    assert round_revenue(1.24e6) == 1e6
    assert round_revenue(None) is None


def _write_inputs(tmp_path: Path, texts: list[tuple[str, str, str, str]]) -> dict[str, Path]:
    events = tmp_path / "events.jsonl"
    with events.open("w", encoding="utf-8") as handle:
        for event_id, company, name, text in texts:
            handle.write(
                json.dumps(
                    {
                        "event_id": event_id,
                        "company": company,
                        "company_name": name,
                        "items": ["5.02"],
                        "text": text,
                        "prior_revenue_usd": 1e9,
                        "prior_revenue_fiscal_year": 2020,
                        "prior_90d_items": [],
                        "sic": "3720" if company == "ba" else "3721",
                    }
                )
                + "\n"
            )
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"event_id": [t[0] for t in texts]}).to_csv(sample, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame(
        {
            "key": ["ba", "lmt", "noc", "gd", "txt", "hon"],
            "name": [
                "BOEING CO",
                "LOCKHEED MARTIN CORP",
                "NORTHROP GRUMMAN CORP",
                "GENERAL DYNAMICS CORP",
                "TEXTRON INC",
                "HONEYWELL INTERNATIONAL INC",
            ],
            "ciks": ["1", "2", "3", "4", "5", "6"],
            "tickers": ["BA", "LMT", "NOC", "GD", "TXT", "HON"],
        }
    ).to_csv(companies, index=False)
    submissions = tmp_path / "subs"
    submissions.mkdir()
    return {"events": events, "sample": sample, "companies": companies, "subs": submissions}


def test_the_mask_script_writes_both_levels_and_records_hashes(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        [
            ("e1", "ba", "BOEING CO", "Boeing named Jane Roe CEO."),
            ("e2", "lmt", "LOCKHEED MARTIN CORP", "Lockheed Martin Corporation hired."),
        ],
    )

    class Tagger:
        def pipe(self, texts, **_):
            for text in texts:
                start = text.find("Jane Roe")
                ents = (
                    []
                    if start < 0
                    else [
                        type(
                            "E",
                            (),
                            {"start_char": start, "end_char": start + 8, "label_": "PERSON"},
                        )()
                    ]
                )
                yield type("D", (), {"ents": ents})()

    out = tmp_path / "masked"
    code = mask_script.main(
        [
            "--events",
            str(paths["events"]),
            "--sample",
            str(paths["sample"]),
            "--companies",
            str(paths["companies"]),
            "--sec-submissions",
            str(paths["subs"]),
            "--output-dir",
            str(out),
        ],
        tagger=Tagger(),
    )
    assert code == 0
    named = [json.loads(line) for line in (out / "name" / "events.jsonl").read_text().splitlines()]
    deep = [json.loads(line) for line in (out / "deep" / "events.jsonl").read_text().splitlines()]
    assert named[0]["text"] == "The Company named Jane Roe CEO."
    assert deep[0]["text"] == "The Company named [person] CEO."
    assert named[1]["text"] == "The Company hired."
    meta = json.loads((out / "masked.meta.json").read_text())
    assert set(meta["sha256"]) >= {"name/events.jsonl", "deep/events.jsonl", "identities.csv"}


def test_the_mask_script_refuses_when_a_name_survives(tmp_path: Path, monkeypatch) -> None:
    paths = _write_inputs(tmp_path, [("e1", "ba", "BOEING CO", "The Boeing Company hired.")])
    monkeypatch.setattr(mask_script, "name_patterns", lambda *a, **k: [])
    code = mask_script.main(
        [
            "--events",
            str(paths["events"]),
            "--sample",
            str(paths["sample"]),
            "--companies",
            str(paths["companies"]),
            "--sec-submissions",
            str(paths["subs"]),
            "--output-dir",
            str(tmp_path / "masked"),
            "--skip-deep",
        ]
    )
    assert code == 1
    assert not (tmp_path / "masked" / "name" / "events.jsonl").exists()


def test_leak_check_options_hold_the_filer_once_beside_same_industry_others(tmp_path: Path) -> None:
    sic = {"ba": "3720", "lmt": "3721", "noc": "3721", "gd": "3721", "txt": "3721", "hon": "9999"}
    names = {k: k.upper() for k in sic}
    event = make_event("text")
    table = identify.options_table([event], sic, names)
    row = table.iloc[0]
    shown = [row[slot] for slot in identify.OPTIONS]
    assert sorted(shown) == ["ba", "gd", "lmt", "noc", "txt"]
    assert row[row["truth"]] == "ba"
    again = identify.options_table([event], sic, names)
    assert again.equals(table)
