"""Automatic checks on one filing's Item 1A extraction (#292).

Each check is the general form of a defect first found by hand in Alphabet's
10-Ks during #289, so that the next filing to show it is caught on the pull
rather than after scoring:

- ``title_read_as_risk``: a section or category title became a risk heading.
  Alphabet's fiscal 2014 to 2019 filings set "RISK FACTORS" apart from
  "Item 1A." in bold, and it opened a fake risk over the section's
  introduction. Flagged: a heading that starts like a title ("Risk Factors",
  "Item 1A", "Risks Related to ...", "General Risks") or has fewer than four
  words. Alphabet's shortest real heading has ten.
- ``heading_split_across_page``: one heading cut in two, its first part read as
  a category and its second as the heading. Alphabet's fiscal 2017 filing broke
  a heading across a page. Flagged: a heading whose first word is in lower case,
  or a category longer than a title (``MAX_CATEGORY_CHARS``).
- ``page_furniture_in_text``: a page number or "Table of Contents" line left in
  a heading or body. Alphabet's filings from fiscal 2023 number pages "11.",
  which an earlier pattern missed. Flagged: "table of contents" anywhere, or a
  one-to-three-digit number standing alone between sentences.
- ``heading_read_as_body``: a heading not recognised, so its risk merged into
  the one before. Alphabet's 808-character fiscal 2018 data-privacy heading
  exceeded an earlier 600-character limit. Flagged: any emphasised paragraph
  read as body text.
- ``heading_without_body``: a heading at the end of the section with nothing
  under it, which the extraction drops. No Alphabet filing has one; it is the
  remaining way a risk can vanish without an error.

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
)
#: A real risk heading is a sentence; Alphabet's shortest has ten words.
MIN_HEADING_WORDS = 4
#: Categories are titles ("Risks Related to Ownership of Our Stock"); Alphabet's
#: longest is 49 characters.
MAX_CATEGORY_CHARS = 100

_TITLE_LIKE = re.compile(
    r"^(item\s*1a\b|risk\s+factors\b|risks?\s+(related|relating|specific)\s+to\b|general\s+risks\b)",
    re.I,
)
#: A heading whose first word is all lower case continues a sentence ("being
#: secure, ..."); a brand such as "iPhone" or "eBay" is not that.
_LOWER_CASE_WORD = re.compile(r"[a-z]+\b")
_TABLE_OF_CONTENTS = re.compile(r"table\s+of\s+contents", re.I)
#: A page number between sentences: at the start, or after a sentence's end and
#: a space, and followed by the end or a space and a capital, bullet or quote.
_PAGE_NUMBER = re.compile(r"(?:^|(?<=[.!?;:\u201d\"\u2019)] ))\d{1,3}\.?(?= [A-Z\u2022\u201c\"]|$)")


@dataclass(frozen=True)
class CheckFinding:
    """One check that failed on one filing; ``risk_id`` is empty for a lost heading."""

    check: str
    risk_id: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _excerpt(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _furniture(text: str) -> str | None:
    match = _TABLE_OF_CONTENTS.search(text) or _PAGE_NUMBER.search(text)
    if match is None:
        return None
    start = max(0, match.start() - 30)
    return text[start : match.end() + 30]


def check_extraction(reading: SectionReading, risks: list[RiskFactor]) -> list[CheckFinding]:
    """Every check's findings for one filing, in the order of ``CHECKS``."""
    findings: list[CheckFinding] = []
    for risk in risks:
        if _TITLE_LIKE.match(risk.heading) or len(risk.heading.split()) < MIN_HEADING_WORDS:
            findings.append(
                CheckFinding(
                    "title_read_as_risk",
                    risk.risk_id,
                    f"heading reads like a title: {_excerpt(risk.heading)!r}",
                )
            )
    seen_categories: set[str] = set()
    for risk in risks:
        if _LOWER_CASE_WORD.match(risk.heading):
            findings.append(
                CheckFinding(
                    "heading_split_across_page",
                    risk.risk_id,
                    f"heading starts in lower case: {_excerpt(risk.heading)!r}",
                )
            )
        category = risk.category
        if category and category not in seen_categories:
            seen_categories.add(category)
            if len(category) > MAX_CATEGORY_CHARS:
                findings.append(
                    CheckFinding(
                        "heading_split_across_page",
                        risk.risk_id,
                        f"category of {len(category)} characters: {_excerpt(category)!r}",
                    )
                )
    for risk in risks:
        for part, text in (("heading", risk.heading), ("body", risk.body)):
            context = _furniture(text)
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
    if reading.unclosed_heading is not None:
        findings.append(
            CheckFinding(
                "heading_without_body",
                "",
                f"heading with no text under it, dropped: {_excerpt(reading.unclosed_heading)!r}",
            )
        )
    return findings
