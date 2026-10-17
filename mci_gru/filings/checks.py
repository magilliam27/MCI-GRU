"""Automatic checks on one filing's Item 1A extraction (#292).

Each check is the general form of a defect first found by hand in Alphabet's
10-Ks during #289, so that the next filing to show it is caught on the pull
rather than after scoring. #297 widened them to ten more companies, whose
layouts showed which parts were Alphabet's habits rather than rules:

- ``title_read_as_risk``: a section, category or introduction title became a
  risk heading. Alphabet's fiscal 2014 to 2019 filings set "RISK FACTORS" apart
  from "Item 1A." in bold, and it opened a fake risk over the section's
  introduction; UnitedHealth's "CAUTIONARY STATEMENTS" did the same. Flagged: a
  heading that starts like such a title ("Risk Factors", "Item 1A", "Risks
  Related to ...", "General Risks", "Cautionary Statements", "Summary") and is
  not a sentence: Honeywell's "Risks related to our defined benefit pension
  plans may adversely impact ..." is a risk. A short heading alone is not
  flagged: Pfizer heads its risks with topic titles such as "Outsourcing".
- ``heading_split_across_page``: one heading cut in two. Alphabet's fiscal 2017
  filing broke a heading across a page; the second part became a heading of its
  own. Flagged: a heading whose first word is in lower case. (The first part,
  too long for a category, is a lost heading, below.)
- ``page_furniture_in_text``: a page number or "Table of Contents" line left in
  a heading or body. Alphabet's filings from fiscal 2023 number pages "11.",
  which an earlier pattern missed. Flagged: "table of contents" anywhere, or a
  one-to-three-digit number standing alone between sentences, within two of the
  page numbers the parser dropped from the section (any such number when it
  dropped none), so that "737 MAX" in a Boeing risk is text.
- ``heading_read_as_body``: a heading not recognised, so its risk merged into
  the one before. Alphabet's 808-character fiscal 2018 data-privacy heading
  exceeded an earlier 600-character limit. Flagged: any emphasised paragraph
  read as body text.
- ``heading_without_body``: a heading after the first risk with nothing under
  it, which the extraction drops: at the end of the section, or over another
  heading while too long to be a category. Before the first risk such text is
  introduction (Pfizer's emphasised forward-looking-statement paragraphs).
- ``text_without_heading``: text after the first risk that the extraction drops
  because no heading sits over it, outside an introduction. Microsoft's fiscal
  2021 to 2025 filings set one risk heading's full stop in plain type; the
  heading went unrecognised and the whole risk vanished. Flagged: any such
  paragraph. Introduction under a "Cautionary Statements" or "Summary" title,
  or under an Exxon Mobil category, is not text without a heading.
- ``section_text_not_in_risks``: the risks hold less than ``MIN_RISK_SHARE`` of
  the section's words, and at least ``MIN_LOST_WORDS`` are missing. The checks above look at what was read; this one counts
  what was not. In the S&P 500 pull (#309) about one report in fifty passed
  every other check while most of its section was lost: Marriott's fiscal 2018
  risks run their headings into the text, the default reading found one
  category over them, and its one risk held 441 of 9,335 words. Introduction
  and summaries are dropped on purpose, which a clean reading's risks rarely
  lose more than a tenth of the section to.

A finding blocks the export's manifest, as an unreadable Item 1A does: fix the
parser (or, for a false alarm, the check) and pull again.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mci_gru.filings.risk_factors import RiskFactor, SectionReading

CHECKS = (
    "title_read_as_risk",
    "heading_split_across_page",
    "page_furniture_in_text",
    "heading_read_as_body",
    "heading_without_body",
    "text_without_heading",
    "section_text_not_in_risks",
)
#: The risks of a clean reading hold at least this share of the section's words,
#: unless fewer than ``MIN_LOST_WORDS`` are missing: an introduction of a few
#: paragraphs is most of a short section.
MIN_RISK_SHARE = 0.7
MIN_LOST_WORDS = 300
#: A dropped page number this close to a number in text makes it page furniture.
PAGE_NUMBER_SLACK = 2

_TITLE_LIKE = re.compile(
    r"^(item\s*1a\b|risk\s+factors\b|risks?\s+(related|relating|specific)\s+to\b|general\s+risks\b|"
    r"cautionary\s+statements?\b|forward[-\s]looking\s+statements\b|summary\b)",
    re.I,
)
#: A heading of more words than this that ends a sentence is a risk, not a title.
MAX_TITLE_WORDS = 6
#: Nor is one that ends a sentence with a verb ("Forward-looking statements may
#: prove inaccurate.").
_VERB = re.compile(
    r"\b(may|might|could|can|cannot|will|would|should|must|is|are|was|were|has|have|had|"
    r"be|been|being|do|does|did)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"[.?!][\"\u201d\u2019)]?$")
#: A heading whose first word is all lower case continues a sentence ("being
#: secure, ..."); a brand such as "iPhone" or "eBay" is not that.
_LOWER_CASE_WORD = re.compile(r"[a-z]+\b")
_TABLE_OF_CONTENTS = re.compile(r"table\s+of\s+contents", re.I)
#: A page number between sentences: at the start, or after a sentence's end and
#: a space, and followed by the end or a space and a capital, bullet or quote.
_PAGE_NUMBER = re.compile(
    r"(?:^|(?<=[.!?;\u201d\"\u2019)] ))(\d{1,3})\.?(?= [A-Z\u2022\u201c\"]|$)"
)


@dataclass(frozen=True)
class CheckFinding:
    """One check that failed on one filing; ``risk_id`` is empty for dropped text."""

    check: str
    risk_id: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _excerpt(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _furniture(text: str, pages: tuple[int, ...]) -> str | None:
    match = _TABLE_OF_CONTENTS.search(text)
    if match is None:
        match = next(
            (
                found
                for found in _PAGE_NUMBER.finditer(text)
                if not pages
                or min(pages) - PAGE_NUMBER_SLACK
                <= int(found.group(1))
                <= max(pages) + PAGE_NUMBER_SLACK
            ),
            None,
        )
    if match is None:
        return None
    start = max(0, match.start() - 30)
    return text[start : match.end() + 30]


def check_extraction(reading: SectionReading, risks: list[RiskFactor]) -> list[CheckFinding]:
    """Every check's findings for one filing, in the order of ``CHECKS``."""
    findings: list[CheckFinding] = []
    for risk in risks:
        if _TITLE_LIKE.match(risk.heading) and not (
            _SENTENCE_END.search(risk.heading)
            and (len(risk.heading.split()) > MAX_TITLE_WORDS or _VERB.search(risk.heading))
        ):
            findings.append(
                CheckFinding(
                    "title_read_as_risk",
                    risk.risk_id,
                    f"heading reads like a title: {_excerpt(risk.heading)!r}",
                )
            )
    for risk in risks:
        if _LOWER_CASE_WORD.match(risk.heading):
            findings.append(
                CheckFinding(
                    "heading_split_across_page",
                    risk.risk_id,
                    f"heading starts in lower case: {_excerpt(risk.heading)!r}",
                )
            )
    for risk in risks:
        for part, text in (("heading", risk.heading), ("body", risk.body)):
            context = _furniture(text, reading.page_numbers)
            if context is not None:
                findings.append(
                    CheckFinding(
                        "page_furniture_in_text",
                        risk.risk_id,
                        f"page furniture in the {part}: {context!r}",
                    )
                )
    for paragraph in reading.emphasised_body:
        holder = next((risk.risk_id for risk in risks if paragraph in risk.body), "")
        findings.append(
            CheckFinding(
                "heading_read_as_body",
                holder,
                f"emphasised paragraph of {len(paragraph)} characters read as body: "
                f"{_excerpt(paragraph)!r}",
            )
        )
    for heading in reading.lost_headings:
        findings.append(
            CheckFinding(
                "heading_without_body",
                "",
                f"heading with no text under it, dropped: {_excerpt(heading)!r}",
            )
        )
    for paragraph in reading.unheaded_text:
        findings.append(
            CheckFinding(
                "text_without_heading",
                "",
                f"paragraph under no heading, dropped: {_excerpt(paragraph)!r}",
            )
        )
    held = sum(len(risk.heading.split()) + len(risk.body.split()) for risk in risks)
    if (
        held < MIN_RISK_SHARE * reading.section_words
        and reading.section_words - held >= MIN_LOST_WORDS
    ):
        findings.append(
            CheckFinding(
                "section_text_not_in_risks",
                "",
                f"risks hold {held:,} of the section's {reading.section_words:,} words "
                f"({held / reading.section_words:.0%})",
            )
        )
    return findings
