"""Russell 2000 membership from the iShares Russell 2000 ETF's SEC holdings filings (E14).

There is no free history of who was in the Russell 2000 and when. The iShares
Russell 2000 ETF (IWM, series S000004344 of iShares Trust) holds nearly every
member and files its full holdings with the SEC: a schedule of investments in
each N-Q (30 June and 31 December, to the end of 2018) and an XML N-PORT every
quarter from 30 September 2019. What this module decides:

- **What the fund held on a date.** ``nq_holdings`` reads the IWM schedule out
  of iShares Trust's combined N-Q (one HTML document for every fund in the
  trust); ``nport_holdings`` reads an N-PORT. Both give the holding's name,
  shares and value; N-PORT adds ISIN and CUSIP.
- **Which listing a holding is.** Value divided by shares is the price the fund
  marked the holding at, which is that day's close, so a candidate price series
  is the holding's when its close agrees (``price_error``), after undoing any
  split the vendor has adjusted for since (``split_factor``). A reused ticker
  fails this test.
- **Who was a member when.** A company is a member from a holdings date until
  the next holdings date it is missing from (``mci_gru.filings.membership.spells``
  over the dated lists of companies held).
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

HOLDING_COLUMNS = ["name", "shares", "value", "section", "isin", "cusip", "asset_cat", "country"]
#: The schedule's top-level sections; industry headings sit between them.
_SECTIONS = (
    "COMMON STOCKS",
    "RIGHTS",
    "WARRANTS",
    "SHORT-TERM INVESTMENTS",
    "MONEY MARKET FUNDS",
    "INVESTMENT COMPANIES",
    "PREFERRED STOCKS",
    "OTHER INTERESTS",
    "TOTAL",
    "CORPORATE BONDS",
)
#: The fund's page header: "iShares(R) Russell 2000 ETF" (or "Index Fund" before
#: July 2013), with the registered mark as an entity and tags in between. The mark
#: is what tells a page header from another fund's holding of IWM.
_PAGE_HEADER = re.compile(
    r"iSHARES(?:<[^>]+>|\s|&reg;|®|&#0*174;|&nbsp;)*RUSSELL\s+2000\s+(?:ETF|INDEX\s+FUND)",
    re.IGNORECASE,
)
_MARK = re.compile(r"&reg;|®|&#0*174;")
_ROW = re.compile(r"<TR[^>]*>(.*?)</TR>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<TD[^>]*>(.*?)</TD>", re.IGNORECASE | re.DOTALL)
_FOOTNOTE = re.compile(r"<SUP[^>]*>.*?</SUP>", re.IGNORECASE | re.DOTALL)
_NUMBER = re.compile(r"\d+(\.\d+)?")
#: Words that say what kind of company a holding is, not which one.
_LEGAL_WORDS = frozenset(
    (
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "plc",
        "llc",
        "lp",
        "the",
        "sa",
        "nv",
        "ag",
        "se",
        "holdings",
        "holding",
        "group",
        "trust",
        "class",
        "a",
        "b",
        "c",
        "new",
        "reit",
        "de",
        "of",
        "common",
        "stock",
        "shares",
        "ordinary",
        "series",
        "cl",
        "com",
        "par",
        "usd",
        "value",
        "voting",
        "non",
        "md",
        "del",
        "us",
        "cos",
        "companies",
        "nvs",
    )
)


class HoldingsError(ValueError):
    """A holdings document that does not read as the fund's schedule."""


def _cell_text(cell: str) -> str:
    text = re.sub(r"<[^>]+>", "", _FOOTNOTE.sub("", cell))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _number(text: str) -> float | None:
    plain = text.replace(",", "").replace("$", "").strip()
    return float(plain) if _NUMBER.fullmatch(plain) else None


def nq_holdings(document: str) -> pd.DataFrame:
    """IWM's holdings from iShares Trust's combined N-Q, one row per holding.

    The schedule runs from the fund's first page header to the "Net Assets" line
    after its last one. A row with a name, shares and value is a holding; a row
    with only a heading changes the section when it starts with one of the
    top-level sections. Footnote marks (superscripts) are dropped from names.
    """
    heads = [m.start() for m in _PAGE_HEADER.finditer(document) if _MARK.search(m.group(0))]
    if not heads:
        raise HoldingsError("no iShares Russell 2000 page header")
    end = re.search(r"NET\s+ASSETS", document[heads[-1] :], re.IGNORECASE)
    if end is None:
        raise HoldingsError("no Net Assets line after the last page header")
    schedule = document[heads[0] : heads[-1] + end.start() + 100]
    section: str | None = None
    rows: list[dict[str, Any]] = []
    for row in _ROW.findall(schedule):
        cells = [
            text for text in (_cell_text(c) for c in _CELL.findall(row)) if text and text != "$"
        ]
        if not cells:
            continue
        head = cells[0].upper()
        if head.startswith(("NET ASSETS", "TOTAL INVESTMENTS")):
            break
        if len(cells) == 1 or (len(cells) == 2 and _number(cells[1]) is None):
            section = next((name for name in _SECTIONS if head.startswith(name)), section)
            continue
        if len(cells) >= 3 and _number(cells[-1]) is not None and _number(cells[-2]) is not None:
            rows.append(
                {
                    "name": cells[0],
                    "shares": _number(cells[-2]),
                    "value": _number(cells[-1]),
                    "section": section,
                }
            )
    if not rows:
        raise HoldingsError("no holdings rows in the schedule")
    return pd.DataFrame(rows).reindex(columns=HOLDING_COLUMNS)


def nport_holdings(document: bytes | str) -> tuple[str, str, pd.DataFrame]:
    """``(series id, report date, holdings)`` from an N-PORT's ``primary_doc.xml``."""
    text = document.decode("utf-8") if isinstance(document, bytes) else document
    # Namespaces only get in the way of finding plain tags.
    text = re.sub(
        r'\s[\w:]+="[^"]*"',
        lambda m: "" if ":" in m.group(0).split("=")[0] or "xmlns" in m.group(0) else m.group(0),
        text,
    )
    text = re.sub(r"<(/?)\w+:", r"<\1", text)
    root = ET.fromstring(text)
    series = root.findtext(".//seriesId") or ""
    date = root.findtext(".//repPdDate") or ""
    rows = []
    for item in root.iter("invstOrSec"):
        isin = item.find("identifiers/isin")
        rows.append(
            {
                "name": item.findtext("name"),
                "shares": float(item.findtext("balance") or 0),
                "value": float(item.findtext("valUSD") or 0),
                "section": None,
                "isin": isin.get("value") if isin is not None else None,
                "cusip": item.findtext("cusip"),
                "asset_cat": item.findtext("assetCat"),
                "country": item.findtext("invCountry"),
            }
        )
    return series, date, pd.DataFrame(rows).reindex(columns=HOLDING_COLUMNS)


def holding_key(name: str) -> str:
    """A holding's name without case, punctuation, state suffixes or legal words.

    Runs of single letters join ("H.B. Fuller" and "HB Fuller" are "hb fuller"),
    a trailing ``/OH``-style state goes, and so do words like Inc and Class A.
    """
    text = str(name).lower().replace("&", " and ").replace("’", "").replace("'", "")
    text = re.sub(r"/[a-z ]*$", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"(inc|corp|co|ltd)\.?(class)", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text.replace(".", " "))
    joined: list[str] = []
    run = ""
    for word in text.split():
        if len(word) == 1:
            run += word
            continue
        if run:
            joined.append(run)
            run = ""
        joined.append(word)
    if run:
        joined.append(run)
    return " ".join(word for word in joined if word not in _LEGAL_WORDS)


def split_factor(splits: Iterable[Mapping[str, str]], day: str) -> float:
    """Product of new/old shares over splits after ``day``: raw close = adjusted close x factor.

    ``splits`` is EODHD's ``/api/splits`` list (``date``, ``split`` as "new/old").
    """
    factor = 1.0
    for entry in splits:
        if entry["date"] > day:
            new, old = (float(part) for part in entry["split"].split("/"))
            if new > 0 and old > 0:
                factor *= new / old
    return factor


def price_error(implied: float, close: float, factor: float = 1.0) -> float:
    """How far the fund's price is from a series' close, as a share, with and without splits."""
    if not (close > 0 and implied > 0):
        return float("inf")
    error = abs(implied / close - 1)
    if factor != 1.0:
        error = min(error, abs(implied / (close * factor) - 1))
    return error
