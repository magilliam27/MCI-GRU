"""8-K events: which filings count, their item sections, and filing-time context (#308).

Experiment E10 asks how good or bad an 8-K event is for shareholders. What this
module decides:

- **Which 8-Ks are events.** An original 8-K (amendments are left out) that
  carries any item in ``EVENT_ITEMS``. EDGAR's submissions files list each
  filing's item codes, so no document is read to decide.
- **The text of an event.** Each item section of the primary document, from its
  "Item N.NN" heading to the next item or the signature (an item may have more
  than one section: JPMorgan heads "Item 5.02(b)" and "Item 5.02(d)" apart), then the press-release
  exhibit (99.1, or a lone unnumbered 99), cut at ``MAX_TEXT_CHARS``. Item 9.01
  only lists exhibits and is left out.
- **Context known at filing time.** The company's revenue as tagged in its
  latest 10-K accepted before the 8-K, and the item codes of its other 8-Ks
  accepted in the ``PRIOR_DAYS`` before it. Nothing dated after the event.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from mci_gru.filings.risk_factors import html_blocks

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

EVENT_FORM = "8-K"
EVENT_ITEMS = ("1.01", "1.02", "1.03", "2.01", "2.05", "2.06", "3.01", "4.01", "4.02", "5.02")
#: Item 9.01 lists the exhibits; its text says nothing about the event.
LEFT_OUT_ITEMS = ("9.01",)
#: The longest text an event may carry; the scoring loader refuses longer ones.
MAX_TEXT_CHARS = 20_000
PRIOR_DAYS = 90
#: An item heading is a short line; a paragraph that cites an item is body.
MAX_ITEM_HEADING_CHARS = 300
#: The XBRL concepts companies tag total revenue with, in order of preference.
#: Which one a company uses changes over the years
#: (Pfizer: SalesRevenueNet, then SalesRevenueGoodsNet, then
#: RevenueFromContractWithCustomer..., then Revenues), so each row names its own.
REVENUE_CONCEPTS = (
    "Revenues",
    "RevenuesNetOfInterestExpense",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
#: A fiscal year's duration, in days, allowing for 52- and 53-week years.
FISCAL_YEAR_DAYS = (350, 380)
#: The last January day on which a 52- or 53-week fiscal year can end.
LAST_DAY_OF_YEAR_END_WEEK = 7

#: An item number ends at a word boundary, a bracket ("5.02(b)") or the title
#: run on without a space (Agilent's "Item 5.02Departure of Directors").
_ITEM_HEADING = re.compile(r"^(item\s*)?(\d)\s*\.\s*(\d{2})(?=\b|\(|[A-Za-z])", re.I)
#: After a bare item number, the item's title ("1.01 Entry into ...").
_BARE_ITEM_TITLE = re.compile(r"^\d\.\d{2}\s+[A-Z]")
#: The signature block, by its title or, when a filing gives it none
#: (Caterpillar's January 2018 8-K), by its opening words.
_SIGNATURE = re.compile(
    r"^(signatures?\b|pursuant to the requirements of the securities exchange act)", re.I
)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
#: The acceptance time a filing's index page shows, in EDGAR's Eastern time.
_ACCEPTED = re.compile(
    r">\s*Accepted\s*</div>\s*<div[^>]*>\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*<", re.I
)
EDGAR_TIMEZONE = "America/New_York"
_PRESS_RELEASE = re.compile(r"^EX-99(\.0?1)?$", re.I)
#: The header lines of EDGAR's SGML wrapper ("<TYPE>EX-99.1"), which open some
#: documents and have no closing tag, so a parser reads them as text.
_SGML_HEADER = re.compile(r"^<(TYPE|SEQUENCE|FILENAME|DESCRIPTION)>[^\n]*$", re.M)


def listed_items(value: str) -> tuple[str, ...]:
    """EDGAR's comma-separated item codes, in their order, without blanks."""
    return tuple(code.strip() for code in value.split(",") if code.strip())


def is_event(form: str, items: Iterable[str]) -> bool:
    return form == EVENT_FORM and bool(set(items) & set(EVENT_ITEMS))


@dataclass(frozen=True)
class ItemSection:
    """One item of an 8-K: its code, heading line and the text under it.

    ``stacked`` marks a heading followed straight by another item's heading,
    which shares that item's text (Fifth Third heads "Item 1.01" and "Item 2.03"
    over one account of a deal). A heading followed by item 9.01's is not
    stacked: that text is left out.
    """

    code: str
    heading: str
    body: str
    stacked: bool = False

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}" if self.body else self.heading


def item_sections(html: str) -> list[ItemSection]:
    """Every item section of an 8-K's primary document, in document order."""
    sections: list[ItemSection] = []
    code: str | None = None
    heading = ""
    body: list[str] = []

    def _close(next_code: str | None = None) -> None:
        if code is not None:
            text = "\n".join(body)
            stacked = next_code not in (None, *LEFT_OUT_ITEMS) and not any(
                character.isalnum() for character in text
            )
            sections.append(ItemSection(code, heading, text, stacked))

    for block in html_blocks(_SGML_HEADER.sub("", html)):
        match = _ITEM_HEADING.match(block.text)
        if (
            match
            and len(block.text) <= MAX_ITEM_HEADING_CHARS
            # Without the word "Item", only an emphasised line is a heading
            # (Caterpillar's "1.01 Entry into a Material Definitive Agreement").
            and (
                match.group(1) is not None
                or (block.emphasised and _BARE_ITEM_TITLE.match(block.text) is not None)
            )
        ):
            next_code = f"{match.group(2)}.{match.group(3)}"
            _close(next_code)
            code, heading, body = next_code, block.text, []
        elif code is not None and _SIGNATURE.match(block.text):
            _close()
            code = None
        elif code is not None:
            body.append(block.text)
    _close()
    return sections


def index_exhibits(index_html: str) -> list[tuple[str, str]]:
    """``(type, document)`` for each document a filing's ``-index.htm`` page lists."""
    rows = []
    for row in _ROW.finditer(index_html):
        cells = [
            html_lib.unescape(_TAG.sub("", cell)).replace("\xa0", " ").strip()
            for cell in _CELL.findall(row.group(1))
        ]
        if len(cells) >= 4 and cells[2]:
            rows.append((cells[3], cells[2].split()[0]))
    return rows


def index_accepted(index_html: str) -> pd.Timestamp | None:
    """The acceptance time a filing's ``-index.htm`` page shows, as a UTC instant.

    The page gives EDGAR's own Eastern clock time. SEC's submissions files give
    the same moment as UTC, but for some filers four or five hours late
    (JPMorgan's 8-K accepted at 16:56 Eastern on 9 January 2013 is listed at
    02:56 UTC the next day, which would be 21:56 Eastern, after the 17:30 cut-off
    its filing date shows it beat). None when the page shows no time.
    """
    match = _ACCEPTED.search(index_html)
    if match is None:
        return None
    return pd.Timestamp(match.group(1)).tz_localize(EDGAR_TIMEZONE).tz_convert("UTC")


def press_release(exhibits: Iterable[tuple[str, str]]) -> tuple[str, str] | None:
    """Exhibit 99.1, or the only exhibit 99 when it carries no number."""
    nineties = [(kind, document) for kind, document in exhibits if kind.upper().startswith("EX-99")]
    numbered = [entry for entry in nineties if _PRESS_RELEASE.match(entry[0]) and "." in entry[0]]
    if numbered:
        return numbered[0]
    if len(nineties) == 1 and _PRESS_RELEASE.match(nineties[0][0]):
        return nineties[0]
    return None


def document_text(html: str) -> str:
    """A document's text, one block a line, without EDGAR's wrapper header."""
    return "\n".join(block.text for block in html_blocks(_SGML_HEADER.sub("", html)))


def event_text(
    sections: Iterable[ItemSection], exhibit: str | None, exhibit_type: str | None = None
) -> tuple[str, bool]:
    """The item sections and press release as one text, and whether it was cut."""
    parts = [section.text for section in sections if section.code not in LEFT_OUT_ITEMS]
    if exhibit:
        parts.append(f"Exhibit {(exhibit_type or 'EX-99.1').removeprefix('EX-')}\n{exhibit}")
    text = "\n\n".join(parts)
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS], True


def prior_items(
    filings: Iterable[tuple[pd.Timestamp, str, tuple[str, ...]]],
    accepted: pd.Timestamp,
    accession: str,
) -> list[list[str]]:
    """Item codes of each other 8-K accepted in the ``PRIOR_DAYS`` before, oldest first.

    ``filings`` holds ``(accepted, accession, items)`` for the company's
    original 8-Ks.
    """
    start = accepted - pd.Timedelta(days=PRIOR_DAYS)
    earlier = sorted(
        (when, other, items)
        for when, other, items in filings
        if other != accession and start <= when < accepted
    )
    return [list(items) for _, _, items in earlier]


def fiscal_year_of(period_end: str) -> int:
    """The fiscal year a period ending on this date belongs to, as for 10-Ks."""
    year, month, day = (int(part) for part in period_end[:10].split("-"))
    return year - 1 if month == 1 and day <= LAST_DAY_OF_YEAR_END_WEEK else year


def tagged_revenue(facts: Mapping[str, Any], accession: str) -> dict[str, Any] | None:
    """Annual revenue a 10-K tags for its own fiscal year, from SEC company facts.

    Of the facts reported in that filing, the first concept in
    ``REVENUE_CONCEPTS`` with a full-year USD value for the filing's latest
    period wins. None when the filing tags none of them.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in REVENUE_CONCEPTS:
        values = [
            fact
            for fact in gaap.get(concept, {}).get("units", {}).get("USD", [])
            if fact.get("accn") == accession and fact.get("start") and fact.get("end")
        ]
        annual = [
            fact
            for fact in values
            if FISCAL_YEAR_DAYS[0]
            <= (pd.Timestamp(fact["end"]) - pd.Timestamp(fact["start"])).days
            <= FISCAL_YEAR_DAYS[1]
        ]
        if not annual:
            continue
        latest = max(fact["end"] for fact in annual)
        fact = next(fact for fact in annual if fact["end"] == latest)
        return {
            "value": float(fact["val"]),
            "concept": concept,
            "period_start": fact["start"],
            "period_end": fact["end"],
            "fiscal_year": fiscal_year_of(fact["end"]),
        }
    return None
