"""Item 1A of a 10-K, cut into individual risk factors (#289).

What this module decides:

- **What counts as text.** The document is read with the standard-library HTML
  parser into blocks (paragraph-level elements), each knowing whether all of its
  text is bold or italic. Hidden content (an inline-XBRL header inside a
  ``display:none`` element), scripts, styles and the head are not text.
- **Where Item 1A is.** A 10-K names Item 1A twice: in the table of contents and
  at the section itself. Each short block starting ``Item 1A`` opens a candidate
  span that ends at the next ``Item 1B``, ``Item 1C`` or ``Item 2``; the longest
  span is the section.
- **Where one risk ends and the next begins.** Each risk factor opens with an
  emphasised heading followed by plain paragraphs. A heading followed directly by
  another heading is a category ("Risks Related to ..."), carried by the risks
  under it. Text before the first heading is the section's introduction, and
  page numbers and running "Table of Contents" lines are page furniture.

None of this is verified against every year's layout here: milestone 1 of #289
checks the extraction by hand on three fiscal years before any answer is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "body",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
VOID_TAGS = frozenset({"area", "br", "col", "hr", "img", "input", "link", "meta", "wbr"})
SKIPPED_TAGS = frozenset({"head", "script", "style", "title"})
BOLD_TAGS = frozenset({"b", "strong"})
ITALIC_TAGS = frozenset({"em", "i"})

#: A heading is at most this many characters; longer emphasised text is body.
MAX_HEADING_CHARS = 600
#: Item headings are short; a long paragraph that mentions Item 1A is not one.
MAX_ITEM_HEADING_CHARS = 100

_SPACE = re.compile(r"\s+")
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(bold|bolder|[6-9]00)", re.I)
_ITALIC_STYLE = re.compile(r"font-style\s*:\s*(italic|oblique)", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_ITEM_1A = re.compile(r"^item\s*1a\b", re.I)
_ITEM_END = re.compile(r"^item\s*(1b|1c|2)\b", re.I)
_FURNITURE = re.compile(r"^(\d{1,3}|table of contents)$", re.I)


class RiskFactorExtractionError(ValueError):
    """A document has no Item 1A section that can be read."""


@dataclass(frozen=True)
class Block:
    """One paragraph-level element's text, and whether all of it is emphasised."""

    text: str
    emphasised: bool


@dataclass(frozen=True)
class RiskFactor:
    """One risk factor: its heading, its body and the category it sits under."""

    accession_number: str
    fiscal_year: int
    index: int
    category: str | None
    heading: str
    body: str

    @property
    def risk_id(self) -> str:
        return f"{self.accession_number}#{self.index:03d}"

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "accession_number": self.accession_number,
            "fiscal_year": self.fiscal_year,
            "index": self.index,
            "category": self.category,
            "heading": self.heading,
            "body": self.body,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> RiskFactor:
        return cls(
            accession_number=str(row["accession_number"]),
            fiscal_year=int(row["fiscal_year"]),
            index=int(row["index"]),
            category=row.get("category"),
            heading=str(row["heading"]),
            body=str(row["body"]),
        )


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._stack: list[tuple[str, bool, bool, bool]] = []
        self._runs: list[tuple[str, bool]] = []

    def _state(self) -> tuple[bool, bool, bool]:
        bold = any(entry[1] for entry in self._stack)
        italic = any(entry[2] for entry in self._stack)
        hidden = any(entry[3] for entry in self._stack)
        return bold, italic, hidden

    def _flush(self) -> None:
        text = _SPACE.sub(" ", "".join(run for run, _ in self._runs)).strip()
        if text:
            emphasised = all(flag for run, flag in self._runs if run.strip())
            self.blocks.append(Block(text, emphasised))
        self._runs = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self._flush()
        if tag in VOID_TAGS:
            return
        style = " ".join(value or "" for name, value in attrs if name == "style")
        self._stack.append(
            (
                tag,
                tag in BOLD_TAGS or bool(_BOLD_STYLE.search(style)),
                tag in ITALIC_TAGS or bool(_ITALIC_STYLE.search(style)),
                tag in SKIPPED_TAGS or bool(_HIDDEN_STYLE.search(style)),
            )
        )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS:
            self._flush()
        for position in range(len(self._stack) - 1, -1, -1):
            if self._stack[position][0] == tag:
                del self._stack[position:]
                break

    def handle_data(self, data: str) -> None:
        bold, italic, hidden = self._state()
        if not hidden:
            self._runs.append((data, bold or italic))

    def close(self) -> None:
        super().close()
        self._flush()


def html_blocks(html: str) -> list[Block]:
    """The document's paragraph-level text blocks, in order."""
    parser = _BlockParser()
    parser.feed(html)
    parser.close()
    return parser.blocks


def item_1a_blocks(blocks: list[Block]) -> list[Block]:
    """The blocks of the Item 1A section, its own heading first."""
    best: tuple[int, int] | None = None
    for start, block in enumerate(blocks):
        if len(block.text) > MAX_ITEM_HEADING_CHARS or not _ITEM_1A.match(block.text):
            continue
        for end in range(start + 1, len(blocks)):
            candidate = blocks[end].text
            if len(candidate) <= MAX_ITEM_HEADING_CHARS and _ITEM_END.match(candidate):
                if best is None or end - start > best[1] - best[0]:
                    best = (start, end)
                break
    if best is None:
        raise RiskFactorExtractionError("no Item 1A section closed by Item 1B, 1C or 2")
    return blocks[best[0] : best[1]]


def _is_heading(block: Block) -> bool:
    return block.emphasised and len(block.text) <= MAX_HEADING_CHARS


def extract_risk_factors(html: str, *, accession_number: str, fiscal_year: int) -> list[RiskFactor]:
    """Item 1A's risk factors, in filing order."""
    section = item_1a_blocks(html_blocks(html))
    risks: list[RiskFactor] = []
    category: str | None = None
    heading: str | None = None
    body: list[str] = []

    def _emit() -> None:
        risks.append(
            RiskFactor(
                accession_number=accession_number,
                fiscal_year=fiscal_year,
                index=len(risks) + 1,
                category=category,
                heading=heading or "",
                body=" ".join(body),
            )
        )

    for block in section[1:]:
        if _FURNITURE.match(block.text):
            continue
        if _is_heading(block):
            if heading is not None and not body:
                category = heading
            elif heading is not None:
                _emit()
            heading, body = block.text, []
        elif heading is not None:
            body.append(block.text)
    if heading is not None and body:
        _emit()
    if not risks:
        raise RiskFactorExtractionError("Item 1A holds no heading followed by text")
    return risks
