"""Automatic checks on one filing's Item 1A extraction (#292).

Each test feeds the parser a filing shaped like one of the defects found by hand
in Alphabet's 10-Ks during #289 (or the remaining way a risk can vanish) and
asserts that the matching check, and only it, fires. The fixture is the same
inline-XBRL-shaped 10-K the parser's own tests use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mci_gru.filings.checks import CHECKS, check_extraction
from mci_gru.filings.risk_factors import read_item_1a

FIXTURE = Path(__file__).parent / "fixtures" / "filings" / "tenk_item1a_sample.htm"
COMPETITION = '<div><span style="font-style:italic;font-weight:700">We face intense competition.'
BODY = "<div><span>Our business environment is rapidly evolving and intensely competitive.</span>"


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _checks(html: str) -> list[tuple[str, str]]:
    reading = read_item_1a(html)
    risks = reading.risk_factors(accession_number="a", fiscal_year=2025)
    return [(finding.check, finding.risk_id) for finding in check_extraction(reading, risks)]


def test_a_clean_filing_raises_no_finding(html):
    assert _checks(html) == []


def test_findings_are_named_by_the_published_check_list(html):
    html = html.replace(COMPETITION, "<div><b>General Risks</b></div>" + BODY + COMPETITION)
    assert {check for check, _ in _checks(html)} <= set(CHECKS)


@pytest.mark.parametrize("title", ["Risk Factors", "General Risks", "Risks Related to Taxes"])
def test_a_title_read_as_a_risk_heading_is_flagged(html, title):
    # A title with text under it reads as a risk: Alphabet's fiscal 2014-2019
    # "RISK FACTORS" did, over the section's introduction.
    html = html.replace(
        COMPETITION,
        f"<div><b>{title}</b></div><div><span>The risks below apply to any company.</span></div>"
        + COMPETITION,
    )
    assert _checks(html) == [("title_read_as_risk", "a#002")]


def test_a_heading_of_fewer_than_four_words_is_flagged(html):
    html = html.replace("We face intense competition.", "Competition.")
    assert _checks(html) == [("title_read_as_risk", "a#002")]


def test_a_heading_starting_mid_sentence_is_flagged(html):
    # The second half of a heading cut by a page break, after body text so the
    # parser cannot join it: what Alphabet's fiscal 2017 heading looked like
    # before the join.
    html = html.replace("We face intense competition.", "being secure, users may leave us.")
    assert _checks(html) == [("heading_split_across_page", "a#002")]


def test_a_brand_name_in_lower_case_is_not_a_split_heading(html):
    html = html.replace("We face intense competition.", "iPhone sales could fall sharply.")
    assert _checks(html) == []


def test_a_category_as_long_as_a_sentence_is_flagged(html):
    # The first half of a split heading read as a category.
    sentence = (
        "If our security measures are breached, or if our services are subject to attacks "
        "that degrade or deny the ability of users to access our products and services"
    )
    html = html.replace("Risks Related to Laws and Regulations", sentence)
    assert _checks(html) == [("heading_split_across_page", "a#003")]


@pytest.mark.parametrize(
    "furniture",
    ["competitive. Table of Contents", "competitive. 14. Customers", "competitive. 12"],
)
def test_page_furniture_left_in_a_body_is_flagged(html, furniture):
    # Alphabet's "11." page numbers from fiscal 2023, when the parser misses them.
    html = html.replace("intensely competitive.</span>", f"intensely {furniture}</span>")
    assert _checks(html) == [("page_furniture_in_text", "a#002")]


@pytest.mark.parametrize(
    "text",
    [
        "competitive in 2025. Revenue grew in 120 countries.",
        "competitive. 2024 Form 10-K filings changed the format.",
        "competitive. 10 of our largest customers left.",
        "competitive under Section 230. Courts may change that.",
        "competitive. 5G networks change how users reach us.",
    ],
)
def test_numbers_that_are_text_are_not_page_furniture(html, text):
    html = html.replace("intensely competitive.</span>", f"intensely {text}</span>")
    assert _checks(html) == []


def test_an_emphasised_paragraph_read_as_body_is_flagged(html):
    # A heading too long for the parser merges its risk into the one before:
    # Alphabet's 808-character fiscal 2018 heading under a 600-character limit.
    long_heading = "Data privacy concerns could damage our reputation and deter users. " * 16
    html = html.replace(
        COMPETITION,
        f'<div><span style="font-weight:700">{long_heading.strip()}</span></div>'
        "<div><span>Concerns are expressed about privacy.</span></div>" + COMPETITION,
    )
    assert _checks(html) == [("heading_read_as_body", "a#001")]


def test_a_heading_with_nothing_under_it_is_flagged(html):
    html = html.replace(
        '<div id="i1b">',
        '<div><span style="font-weight:700">We may not pay dividends.</span></div><div id="i1b">',
    )
    assert _checks(html) == [("heading_without_body", "")]
