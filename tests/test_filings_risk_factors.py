"""Item 1A of a 10-K, cut into individual risk factors (#289).

The fixture is shaped like an inline-XBRL 10-K: a hidden XBRL header, a table of
contents that also names Item 1A, bold category headings, bold-italic risk
headings, and page furniture inside the section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mci_gru.filings.risk_factors import (
    RiskFactorExtractionError,
    extract_risk_factors,
    html_blocks,
    item_1a_blocks,
    read_item_1a,
)

FIXTURE = Path(__file__).parent / "fixtures" / "filings" / "tenk_item1a_sample.htm"


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_hidden_xbrl_header_and_styles_are_not_text(html):
    texts = [block.text for block in html_blocks(html)]
    assert not any("Hidden facts" in text for text in texts)
    assert not any("color: red" in text for text in texts)


def test_item_1a_is_the_section_not_the_table_of_contents(html):
    section = [block.text for block in item_1a_blocks(html_blocks(html))]
    assert section[0].startswith("ITEM 1A.")
    assert any("advertising" in text for text in section)
    assert not any("UNRESOLVED STAFF COMMENTS" in text for text in section)
    assert not any("holding company" in text for text in section)


def test_risks_are_split_at_emphasised_headings_with_their_bodies(html):
    risks = extract_risk_factors(html, accession_number="0001652044-26-000018", fiscal_year=2025)
    assert [risk.heading for risk in risks] == [
        "We generate a significant portion of our revenues from advertising. "
        "Reduced spending by advertisers could harm our business.",
        "We face intense competition.",
        "We are subject to increasing regulatory scrutiny, including antitrust investigations.",
    ]
    assert risks[0].body == (
        "We generated over 75% of total revenues from online advertising in 2025. "
        "Many advertisers can reduce or cancel their spending at any time."
    )


def test_a_heading_followed_by_a_heading_is_a_category(html):
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert [risk.category for risk in risks] == [
        "Risks Related to Our Business and Industry",
        "Risks Related to Our Business and Industry",
        "Risks Related to Laws and Regulations",
    ]


def test_page_numbers_and_running_headers_are_not_body_text(html):
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert (
        risks[1].body == "Our business environment is rapidly evolving and intensely competitive."
    )


def test_intro_paragraph_before_the_first_heading_is_not_a_risk(html):
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert not any("subject to various risks" in risk.body for risk in risks)


def test_risk_ids_are_stable_and_ordered(html):
    risks = extract_risk_factors(html, accession_number="0001652044-26-000018", fiscal_year=2025)
    assert [risk.risk_id for risk in risks] == [
        "0001652044-26-000018#001",
        "0001652044-26-000018#002",
        "0001652044-26-000018#003",
    ]
    assert {risk.fiscal_year for risk in risks} == {2025}


def test_filing_without_item_1a_is_refused():
    with pytest.raises(RiskFactorExtractionError):
        extract_risk_factors(
            "<html><body><div>ITEM 1. BUSINESS</div><div>Text.</div></body></html>",
            accession_number="a",
            fiscal_year=2025,
        )


def test_a_long_emphasised_paragraph_is_body_text_not_a_heading(html):
    emphasised = "Investors should read this whole section with care. " * 25
    html = html.replace(
        '<div><span style="font-style:italic;font-weight:700">We face intense competition.',
        f'<div><span style="font-weight:700">{emphasised.strip()}</span></div>'
        '<div><span style="font-style:italic;font-weight:700">We face intense competition.',
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert len(risks) == 3
    assert risks[0].body.endswith(emphasised.strip())


def test_a_heading_of_two_long_sentences_is_still_a_heading(html):
    # Shaped like Alphabet's fiscal 2018 data-privacy heading, 808 characters,
    # which an earlier 600-character limit merged into the risk before it.
    heading = (
        "Data privacy and security concerns relating to our technology and our "
        "practices could damage our reputation and deter current and potential users. "
    ) * 6
    html = html.replace(
        '<div><span style="font-style:italic;font-weight:700">We face intense competition.',
        f'<div><span style="font-style:italic;font-weight:700">{heading.strip()}</span></div>'
        "<div><span>Concerns are expressed about privacy.</span></div>"
        '<div><span style="font-style:italic;font-weight:700">We face intense competition.',
    )
    assert 800 < len(heading.strip()) < 1000
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert len(risks) == 4
    assert risks[1].heading == heading.strip()
    assert risks[1].body == "Concerns are expressed about privacy."


def test_a_line_break_inside_a_heading_is_a_wrap_not_a_category(html):
    html = html.replace(
        '<span style="font-style:italic;font-weight:700">We face intense competition.</span>',
        "<b><i>We face intense<br>competition.</i></b>",
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert risks[1].heading == "We face intense competition."
    assert risks[1].category == "Risks Related to Our Business and Industry"


@pytest.mark.parametrize("header", ["<b>Alphabet Inc.</b>", "Alphabet Inc."])
def test_a_running_page_header_is_dropped_whether_bold_or_not(html, header):
    running = f"<div>{header}</div>"
    html = (
        html.replace("<div><span>11</span></div>", f"<div><span>11</span></div>{running}")
        .replace("<div><span>Many advertisers", f"{running}<div><span>Many advertisers")
        .replace("<div><span>Governments", f"{running}<div><span>Governments")
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert len(risks) == 3
    assert not any("Alphabet Inc." in risk.text for risk in risks)


@pytest.mark.parametrize("title", ["RISK FACTORS", "Item 1A. Risk Factors"])
def test_a_section_title_on_its_own_line_is_not_a_risk(html, title):
    # Alphabet's 10-Ks for fiscal 2014 to 2019 put "ITEM 1A." and "RISK FACTORS"
    # in separate bold blocks, and Delta's from 2020 repeat "Item 1A. Risk
    # Factors"; the title must not open a risk over the intro.
    html = html.replace(
        "ITEM 1A.&#160;&#160;&#160;&#160;RISK FACTORS</span></div>",
        f'ITEM 1A.</span></div><div><span style="font-weight:700">{title}</span></div>',
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2019)
    assert len(risks) == 3
    assert risks[0].heading.startswith("We generate a significant portion")
    assert not any("subject to various risks" in risk.text for risk in risks)


def test_a_page_number_with_a_full_stop_is_not_body_text(html):
    # Alphabet's 10-Ks from fiscal 2023 number their pages "11." and break
    # paragraphs across them.
    html = html.replace("<div><span>11</span></div>", "<div><span>11.</span></div>")
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2024)
    assert (
        risks[1].body == "Our business environment is rapidly evolving and intensely competitive."
    )


def test_a_heading_broken_across_a_page_is_one_heading_not_a_category(html):
    # Alphabet's fiscal 2017 10-K breaks one risk heading across a page; the
    # second part starts mid-sentence, in lower case.
    html = html.replace(
        '<span style="font-style:italic;font-weight:700">We face intense competition.</span></div>',
        '<span style="font-style:italic;font-weight:700">We face intense</span></div>'
        "<div><span>11</span></div><div><span>Table of Contents</span></div>"
        '<div><span style="font-style:italic;font-weight:700">competition.</span></div>',
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2017)
    assert len(risks) == 3
    assert risks[1].heading == "We face intense competition."
    assert risks[1].category == "Risks Related to Our Business and Industry"


def test_a_lower_case_heading_after_body_text_starts_a_new_risk(html):
    # Only a heading with no body yet can be continued across a page; a heading
    # that merely starts in lower case ("iPhone ...") after a risk's body is a
    # new risk, not more of the previous heading.
    html = html.replace(
        "<div><span>Our business environment is rapidly evolving and intensely competitive."
        "</span></div>",
        "<div><span>Our business environment is rapidly evolving and intensely competitive."
        "</span></div>"
        '<div><span style="font-style:italic;font-weight:700">iPhone sales could fall.</span></div>'
        "<div><span>Demand for devices that run our apps can change quickly.</span></div>",
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert [risk.heading for risk in risks][1:3] == [
        "We face intense competition.",
        "iPhone sales could fall.",
    ]


def test_a_bullet_in_its_own_table_cell_is_kept_before_its_item(html):
    # Alphabet's fiscal 2013-2019 filings set each bullet in its own cell. Three
    # or more of them once read as a running header and were dropped.
    items = ("Restrictions on foreign ownership.", "Longer payment cycles.", "Currency risk.")
    rows = "".join(
        f"<tr><td><div>&#8226;</div></td><td><div>{item}</div></td></tr>" for item in items
    )
    html = html.replace(
        "intensely competitive.</span></div>",
        f"intensely competitive.</span></div><table>{rows}</table>",
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2019)
    assert risks[1].body == (
        "Our business environment is rapidly evolving and intensely competitive. "
        "• Restrictions on foreign ownership. • Longer payment cycles. • Currency risk."
    )


def test_a_bullet_run_into_its_item_is_followed_by_one_space(html):
    # From fiscal 2020 the bullet is a span with padding and no space after it.
    item = '<div><span>&#8226;</span><span style="padding-left:14.5pt">restrictions apply.</span></div>'
    html = html.replace(
        "intensely competitive.</span></div>", f"intensely competitive.</span></div>{item}"
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2020)
    assert risks[1].body.endswith("competitive. • restrictions apply.")


def test_an_emphasised_bullet_is_not_a_heading(html):
    bullets = "<div><b>&#8226;</b></div><div><span>Restrictions on foreign ownership.</span></div>"
    html = html.replace(
        "intensely competitive.</span></div>", f"intensely competitive.</span></div>{bullets}"
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2019)
    assert [risk.heading for risk in risks][1] == "We face intense competition."
    assert len(risks) == 3


COMPETITION = '<div><span style="font-style:italic;font-weight:700">We face intense competition.'


def test_a_plain_span_inside_a_bold_paragraph_is_plain():
    # Microsoft sets a risk's heading in a bold paragraph and its text in a
    # span that turns bold off.
    blocks = html_blocks(
        '<p style="font-weight:bold">We face competition'
        '<span style="font-weight:normal">. Rivals are many.</span></p>'
    )
    assert not blocks[0].emphasised
    assert blocks[0].lead == "We face competition"


def test_a_full_stop_left_plain_does_not_make_a_heading_body_text(html):
    # Caterpillar's headings end in a full stop set outside the bold.
    html = html.replace(
        '<span style="font-style:italic;font-weight:700">We face intense competition.</span>',
        "<b>We face intense competition</b>.",
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert [risk.heading for risk in risks][1] == "We face intense competition."
    assert len(risks) == 3


@pytest.mark.parametrize(
    "footer", ["Pfizer Inc. | 2015 Form 10-K | {page}", "{page} Delta Air Lines"]
)
def test_a_footer_with_its_page_number_is_dropped(html, footer):
    def line(page: int) -> str:
        return f"<div><span>{footer.format(page=page)}</span></div>"

    html = (
        html.replace("<div><span>Many advertisers", f"{line(12)}<div><span>Many advertisers")
        .replace("<div><span>11</span></div>", line(13))
        .replace("<div><span>Governments", f"{line(14)}<div><span>Governments")
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2015)
    assert len(risks) == 3
    assert not any("Form 10-K" in risk.text or "Delta" in risk.text for risk in risks)


def test_a_part_line_is_page_furniture(html):
    html = html.replace("<div><span>11</span></div>", "<div><span>11</span></div><div>PART I</div>")
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2025)
    assert (
        risks[1].body == "Our business environment is rapidly evolving and intensely competitive."
    )


@pytest.mark.parametrize(
    "bullet", ["<div>&#8226;</div><div>{item}</div>", "<div>&#8226; {item}</div>"]
)
def test_a_list_item_repeated_across_risks_is_text_not_a_header(html, bullet):
    # JPMorgan ends several lists with "harm to JPMorgan Chase's reputation."
    item = bullet.format(item="harm to our reputation.")
    html = (
        html.replace("<div><span>Many advertisers", f"{item}<div><span>Many advertisers")
        .replace("<div><span>11</span></div>", f"{item}<div><span>11</span></div>")
        .replace("<div><span>Governments", f"{item}<div><span>Governments")
    )
    risks = extract_risk_factors(html, accession_number="a", fiscal_year=2020)
    assert all("• harm to our reputation." in risk.body for risk in risks)


@pytest.mark.parametrize(
    "title", ["Cautionary Statements", "Summary of Risk Factors", "Forward-Looking Statements"]
)
def test_an_introduction_title_drops_the_text_under_it(html, title):
    # UnitedHealth's "Cautionary Statements" and the risk summaries JPMorgan and
    # others print are not risks.
    html = html.replace(
        COMPETITION,
        f"<div><b>{title}</b></div><div><span>This summary is not complete.</span></div>"
        + COMPETITION,
    )
    reading = read_item_1a(html)
    risks = reading.risk_factors(accession_number="a", fiscal_year=2025)
    assert [risk.heading for risk in risks][1] == "We face intense competition."
    assert len(risks) == 3
    assert not any("not complete" in risk.text for risk in risks)
    assert reading.unheaded_text == ()


def test_a_long_emphasised_preamble_before_the_first_risk_is_introduction(html):
    # Pfizer opens Item 1A with bold forward-looking-statement paragraphs.
    preamble = "This report contains forward-looking statements about our business. " * 3
    html = html.replace(
        '<div><span style="font-weight:700">Risks Related to Our Business and Industry</span></div>',
        f"<div><b>{preamble.strip()}</b></div>"
        '<div><span style="font-weight:700">Risks Related to Our Business and Industry</span></div>',
    )
    reading = read_item_1a(html)
    risks = reading.risk_factors(accession_number="a", fiscal_year=2025)
    assert len(risks) == 3
    assert risks[0].category == "Risks Related to Our Business and Industry"
    assert reading.lost_headings == ()


RUN_IN = (
    "<html><body><div>ITEM 1A. RISK FACTORS</div>"
    "<div><b>Supply and Demand</b></div>"
    "<div>Our results depend on the factors below.</div>"
    "<div><i>Economic conditions.</i> The demand for energy follows growth.</div>"
    "<div>Recessions reduce demand.</div>"
    "<div><i>Other demand-related factors</i>: Weather matters.</div>"
    "<div><b>Operational Factors</b></div>"
    "<div>Some factors are within our control.</div>"
    "<div><i>Cybersecurity.</i> Attacks may disrupt us.</div>"
    "<div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div></body></html>"
)


def test_run_in_headings_open_their_paragraphs():
    # Exxon Mobil heads each risk with an italic lead in its first paragraph,
    # under bold categories that carry an introduction of their own.
    reading = read_item_1a(RUN_IN, layout="run_in")
    risks = reading.risk_factors(accession_number="a", fiscal_year=2025)
    assert [(risk.category, risk.heading, risk.body) for risk in risks] == [
        (
            "Supply and Demand",
            "Economic conditions.",
            "The demand for energy follows growth. Recessions reduce demand.",
        ),
        ("Supply and Demand", "Other demand-related factors:", "Weather matters."),
        ("Operational Factors", "Cybersecurity.", "Attacks may disrupt us."),
    ]
    assert reading.unheaded_text == ()


SENTENCES = (
    "<html><body><div>ITEM 1A. RISK FACTORS</div>"
    "<div>STRATEGIC AND COMPETITIVE RISKS</div>"
    '<p style="font-weight:bold">We face intense competition'
    '<span style="font-weight:normal">. Rivals are many.</span></p>'
    "<div><b>Competition among platforms</b></div>"
    "<div>Platforms compete for developers.</div>"
    "<div><b>Our investments may not pay off.</b></div>"
    "<div>New products may fail.</div>"
    "<div>GENERAL RISKS</div>"
    "<div><b>Economic conditions may harm us.</b> Demand may fall.</div>"
    "<div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div></body></html>"
)


def test_sentence_headings_keep_their_titled_sub_headings_as_text():
    # Microsoft: risk headings are emphasised sentences (a block, or a lead
    # whose full stop may be plain), titled sub-headings sit inside a risk, and
    # categories are lines in capitals.
    reading = read_item_1a(SENTENCES, layout="sentences")
    risks = reading.risk_factors(accession_number="a", fiscal_year=2025)
    assert [(risk.category, risk.heading, risk.body) for risk in risks] == [
        (
            "STRATEGIC AND COMPETITIVE RISKS",
            "We face intense competition.",
            "Rivals are many. Competition among platforms Platforms compete for developers.",
        ),
        (
            "STRATEGIC AND COMPETITIVE RISKS",
            "Our investments may not pay off.",
            "New products may fail.",
        ),
        ("GENERAL RISKS", "Economic conditions may harm us.", "Demand may fall."),
    ]
    assert reading.unheaded_text == ()


def test_an_unknown_layout_is_refused(html):
    with pytest.raises(ValueError):
        read_item_1a(html, layout="columns")
