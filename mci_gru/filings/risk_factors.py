"""Item 1A of a 10-K, cut into individual risk factors (#289).

What this module decides:

- **What counts as text.** The document is read with the standard-library HTML
  parser into blocks (paragraph-level elements), each knowing whether all of its
  words are bold or italic, and which emphasised words open it (its lead).
  Emphasis is set by the innermost element that says so, so a plain span inside
  a bold paragraph is plain (Microsoft's fiscal 2017-2022 filings), and a
  full stop or other mark left unemphasised does not make a heading body text
  (Caterpillar's). Hidden content (an inline-XBRL header inside a
  ``display:none`` element), scripts, styles and the head are not text.
- **Where Item 1A is.** A 10-K names Item 1A twice: in the table of contents and
  at the section itself. Each short block starting ``Item 1A`` opens a candidate
  span that ends at the next ``Item 1B``, ``Item 1C`` or ``Item 2``; the longest
  span is the section.
- **Where one risk ends and the next begins.** Each risk factor opens with an
  emphasised heading followed by plain paragraphs. A heading followed directly by
  another heading is a category ("Risks Related to ..."), carried by the risks
  under it, unless it is longer than a title (``MAX_CATEGORY_CHARS``): then it
  is emphasised introduction (Pfizer's forward-looking-statement paragraphs)
  and dropped. Text before the first heading is the section's introduction; a
  "Risk Factors" or "Item 1A. Risk Factors" title set apart from the item
  heading is part of the section's own heading, not a risk (Alphabet's fiscal
  2014 to 2019 filings, Delta's from 2020). A heading such as "Cautionary
  Statements" or "Summary" opens more introduction, dropped with the text under
  it until the next heading (UnitedHealth's cautionary statements, the risk
  summaries JPMorgan and others print before the risks themselves). A
  ``<br>`` inside a paragraph is a line wrap, not a new block, and a heading
  that a page break cuts in two (its second part starting in lower case,
  Alphabet's fiscal 2017 filing) is one heading, not a category and a heading.
- **What is page furniture.** Page numbers (with or without a full stop, as
  in Alphabet's filings from fiscal 2023), "Table of Contents" and "Part I"
  lines, and any short line repeated on several pages of the section, once a
  page number at either end is set aside (a running header such as "Alphabet
  Inc.", bold or not, or a footer such as "Pfizer Inc. | 2015 Form 10-K | 20"),
  are dropped wherever they fall.
- **Layouts other than Alphabet's.** ``read_item_1a`` takes a layout. The
  default, ``blocks``, is the rule above. ``run_in`` is for filers whose risk
  headings open the paragraph they head (Exxon Mobil: "*Economic conditions.*
  The demand for energy ..."): each emphasised lead ending in a full stop or
  colon starts a risk, and whole emphasised blocks are categories.
  ``sentences`` is for filers who put titled sub-headings under each risk
  (Microsoft): a risk heading is an emphasised sentence, as a block or a lead;
  an emphasised title without a full stop is a sub-heading kept in the risk's
  text; and a short line in capitals is a category.
- **How a list item reads.** A bullet is kept and followed by one space, whether
  the filing sets it in its own table cell (Alphabet's fiscal 2013 to 2019
  filings) or in a span run into the item's text (fiscal 2020 on), so the same
  list reads the same in every year. A block of bullets or other marks with no
  words is never a heading.

``read_item_1a`` also records what a reader would otherwise lose silently: an
emphasised paragraph too long to be a heading, read as body text; a heading
after the first risk with no text under it; text after the first risk with no
heading over it, outside introduction; and the page numbers it dropped.
``mci_gru.filings.checks`` turns those, and the output itself, into per-filing
checks (#292).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "body",
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
#: Alphabet's longest heading is 808 characters (fiscal 2018); no emphasised
#: block in its fiscal 2013-2025 risk sections falls between that and 1,000.
MAX_HEADING_CHARS = 1000
#: Item headings are short; a long paragraph that mentions Item 1A is not one.
MAX_ITEM_HEADING_CHARS = 100
#: A short line seen this many times in one Item 1A is a running page header.
RUNNING_HEADER_REPEATS = 3
RUNNING_HEADER_MAX_CHARS = 100
#: A category is a title; a longer emphasised block over a heading is introduction.
MAX_CATEGORY_CHARS = 100
#: A category in title case may run longer (Bank of America's "Market Risk is the
#: Risk that Values of Assets and Liabilities or Revenues will be Adversely
#: Affected by Changes in Market Conditions ...", 379 characters).
MAX_TITLE_CASE_CATEGORY_CHARS = 400
#: A run-in heading is the emphasised start of a paragraph, at most this long.
MAX_RUN_IN_CHARS = 400
LAYOUTS = ("blocks", "run_in", "sentences")

_SPACE = re.compile(r"\s+")
_WEIGHT_STYLE = re.compile(r"font-weight\s*:\s*([a-z0-9]+)", re.I)
_SLANT_STYLE = re.compile(r"font-style\s*:\s*([a-z]+)", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
#: "Item", even with markup splitting the word ("I tem 1B" in Verizon's fiscal 2015).
_ITEM_1A = re.compile(r"^i\s?t\s?e\s?m\s*1a\b", re.I)
_ITEM_END = re.compile(r"^i\s?t\s?e\s?m\s*(1b|1c|2)\b", re.I)
#: The title of a 10-K section that can follow the risk factors, when a filing
#: is laid out by topic rather than by item (Honeywell from fiscal 2020).
_NEXT_SECTION_TITLE = re.compile(
    r"^(unresolved\s+staff\s+comments|cybersecurity|properties|legal\s+proceedings|"
    r"mine\s+safety\s+disclosures|market\s+for\s+(the\s+)?registrant|selected\s+financial\s+data|"
    r"management.s\s+discussion\s+and\s+analysis|quantitative\s+and\s+qualitative\s+disclosures?)",
    re.I,
)
_FURNITURE = re.compile(
    r"^(\d{1,3}\.?|table of contents|part\s+[ivx]+|parts\s+[ivx]+\s+(and|&)\s+[ivx]+)$", re.I
)
_PAGE_NUMBER_BLOCK = re.compile(r"^(\d{1,3})\.?$")
#: A page number at either end of a running header or footer, with its separator.
_PAGE_AFFIX = re.compile(r"^\d{1,3}\s*[|\u2022\u00b7-]?\s+|\s+[|\u2022\u00b7-]?\s*\d{1,3}$")
#: The section's own title, when a filing sets it apart from the item heading.
_SECTION_TITLE = re.compile(r"^(item\s*1a\.?\s*)?risk\s+factors\.?$", re.I)
#: The section's title again, as a line of its own anywhere in the section: a
#: page's continuation header (Ford's "Item 1A. Risk Factors (Continued)", AIG's
#: "ITEM 1A | Risk Factors") or a title set apart ("Risk Factors Discussion").
_SECTION_TITLE_LINE = re.compile(
    r"^(item\s*1a\.?\s*[|:\u2013\u2014-]?\s*)?risk\s+factors\s*"
    r"(,?\s*\(?(continued|cont.d)\.?\)?|\(.*\)|discussion)?\s*[.:]?$",
    re.I,
)
#: A heading that opens introduction rather than a risk.
_INTRO_TITLE = re.compile(
    r"^(cautionary\s+(statements?|notes?)(\s+(about|regarding|concerning|relating\s+to|on|for)\s+"
    r"forward[-\s]looking\s+(statements|information))?|forward[-\s]looking\s+statements|"
    r"(risk\s+factors?\s+)?summary(\s+of\s+(the\s+)?risk\s+factors)?|summary\s+risk\s+factors)\.?$",
    re.I,
)
#: Words that make a title introduction when it has no verb: forward-looking
#: statements, a safe harbour, or a summary of the risks ("Forward-Looking
#: Statements and Risk Factors", "Summary of Risks Associated with Our Business").
_INTRO_WORDS = re.compile(
    r"forward[-\s]looking|cautionary\s+statement|safe\s+harbou?r|"
    r"^(risk\s+factors?\s+)?summary\s*[.:]?$|^summary\s+of\s+(the\s+)?(\w+\s+)?(risks?|risk\s+factors)\b",
    re.I,
)
#: An introduction's title is at most this long (Cautionary Statement for
#: Purposes of the "Safe Harbor" Provisions of the Private Securities Litigation
#: Reform Act of 1995).
MAX_INTRO_TITLE_CHARS = 200
#: A category title, with any text under it introducing the category (NextEra's
#: "Risks Relating to NEE's and FPL's Business", CenterPoint's "Risk Factors
#: Affecting Our Interests in Enable Midstream"). Pfizer's "Risks Affecting
#: International Operations" is a risk.
_CATEGORY_TITLE = re.compile(
    r"^((risks?|risk\s+factors)\s+(related|relating|specific|pertaining)\s+to\b|"
    r"risk\s+factors\s+affecting\b|general\s+risks?\b)",
    re.I,
)
#: A verb makes a heading that names forward-looking statements a risk, not an
#: introduction's title: "Forward-looking statements may prove inaccurate."
_VERB = re.compile(
    r"\b(may|might|could|can|cannot|will|would|should|must|is|are|was|were|has|have|had|"
    r"be|been|being|do|does|did)\b",
    re.I,
)
#: A sentence ends in a full stop, question or exclamation mark, perhaps quoted.
_SENTENCE_END = re.compile(r"[.?!][\"\u201d\u2019)]?$")
_RUN_IN_END = re.compile(r"[.:][\"\u201d\u2019)]?$")
#: A lead's own full stop set in plain type, opening the rest of its paragraph.
_PLAIN_STOP = re.compile(r"^[.:?!][\"\u201d\u2019)]?(?=\s|$)")
#: A bullet opening a block, and whatever space follows it.
_LEADING_BULLET = re.compile(r"^([\u2022\u25cf\u25aa\u25e6\u2023\u00b7])\s*")
#: A bullet on its own, its item in the next block.
_BARE_BULLET = re.compile(r"^[\u2022\u25cf\u25aa\u25e6\u2023\u00b7]$")


class RiskFactorExtractionError(ValueError):
    """A document has no Item 1A section that can be read."""


@dataclass(frozen=True)
class Block:
    """One paragraph-level element's text, whether all of its words are emphasised,
    and the emphasised words that open it (all of it when ``emphasised``)."""

    text: str
    emphasised: bool
    lead: str = ""


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


def _clean(text: str) -> str:
    text = _SPACE.sub(" ", text).strip()
    return _LEADING_BULLET.sub(r"\1 ", text).strip()


def _weight(style: str) -> bool | None:
    """Whether a style makes text bold, makes it plain, or says nothing of weight."""
    match = _WEIGHT_STYLE.search(style)
    if match is None:
        return None
    value = match.group(1).lower()
    if value in ("bold", "bolder") or (value.isdigit() and int(value) >= 600):
        return True
    if value in ("normal", "lighter") or value.isdigit():
        return False
    return None


def _slant(style: str) -> bool | None:
    """Whether a style makes text italic, makes it upright, or says nothing of slant."""
    match = _SLANT_STYLE.search(style)
    if match is None:
        return None
    value = match.group(1).lower()
    if value in ("italic", "oblique"):
        return True
    return False if value == "normal" else None


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._stack: list[tuple[str, bool | None, bool | None, bool]] = []
        self._runs: list[tuple[str, bool]] = []

    def _state(self) -> tuple[bool, bool, bool]:
        # The innermost element that sets a weight or a slant decides it.
        bold = next((entry[1] for entry in reversed(self._stack) if entry[1] is not None), False)
        italic = next((entry[2] for entry in reversed(self._stack) if entry[2] is not None), False)
        hidden = any(entry[3] for entry in self._stack)
        return bold, italic, hidden

    def _flush(self) -> None:
        text = _clean("".join(run for run, _ in self._runs))
        if text:
            worded = [flag for run, flag in self._runs if _has_words(run)]
            emphasised = bool(worded) and all(worded)
            lead_runs: list[str] = []
            for run, flag in self._runs:
                if _has_words(run) and not flag:
                    break
                lead_runs.append(run)
            lead = _clean("".join(lead_runs)) if any(worded[:1]) else ""
            self.blocks.append(Block(text, emphasised, text if emphasised else lead))
        self._runs = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self._flush()
        if tag == "body":
            # The body ends the head even when the head is never closed
            # (Pfizer's 8-Ks before 2018).
            heads = [position for position, entry in enumerate(self._stack) if entry[0] == "head"]
            if heads:
                del self._stack[heads[0] :]
        if tag == "br":
            self._runs.append((" ", False))
        if tag in VOID_TAGS:
            return
        style = " ".join(value or "" for name, value in attrs if name == "style")
        self._stack.append(
            (
                tag,
                True if tag in BOLD_TAGS else _weight(style),
                True if tag in ITALIC_TAGS else _slant(style),
                tag in SKIPPED_TAGS or bool(_HIDDEN_STYLE.search(style)),
            )
        )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self._flush()
        if tag == "br":
            self._runs.append((" ", False))

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


def _longest_span(
    blocks: list[Block], opens: Callable[[Block], bool], closes: Callable[[Block], bool]
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for start, block in enumerate(blocks):
        if len(block.text) > MAX_ITEM_HEADING_CHARS or not opens(block):
            continue
        for end in range(start + 1, len(blocks)):
            if len(blocks[end].text) <= MAX_ITEM_HEADING_CHARS and closes(blocks[end]):
                if best is None or end - start > best[1] - best[0]:
                    best = (start, end)
                break
    return best


def item_1a_blocks(blocks: list[Block]) -> list[Block]:
    """The blocks of the Item 1A section, its own heading first."""
    best = _longest_span(
        blocks,
        lambda block: bool(_ITEM_1A.match(block.text)),
        lambda block: bool(_ITEM_END.match(block.text)),
    )
    if best is None or not any(
        len(block.text) > MAX_ITEM_HEADING_CHARS for block in blocks[best[0] : best[1]]
    ):
        # Item 1A only points elsewhere (an index of items at the back of a
        # filing laid out by topic): the section titled "Risk Factors", up to
        # the next section's title.
        titled = _longest_span(
            blocks,
            lambda block: block.emphasised and bool(_SECTION_TITLE.match(block.text)),
            lambda block: bool(
                _NEXT_SECTION_TITLE.match(block.text) or _ITEM_END.match(block.text)
            ),
        )
        best = titled or best
    if best is None:
        raise RiskFactorExtractionError("no Item 1A section closed by Item 1B, 1C or 2")
    return blocks[best[0] : best[1]]


def _has_words(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _furniture_key(text: str) -> str:
    """A short line with any page number at either end set aside."""
    return _PAGE_AFFIX.sub("", text).strip() if len(text) <= RUNNING_HEADER_MAX_CHARS else text


def _listed(section: list[Block]) -> list[bool]:
    """Whether each block is a list item: it opens with a bullet, or follows one."""
    after = [False] + [_BARE_BULLET.match(block.text) is not None for block in section[:-1]]
    return [
        follows or (_LEADING_BULLET.match(block.text) is not None and _has_words(block.text))
        for block, follows in zip(section, after, strict=True)
    ]


def _running_headers(section: list[Block]) -> set[str]:
    # A list item repeated across risks ("harm to JPMorgan Chase's reputation.")
    # is text, not a header.
    counts = Counter(
        _furniture_key(block.text)
        for block, listed in zip(section, _listed(section), strict=True)
        if not listed
    )
    return {
        key
        for key, seen in counts.items()
        if seen >= RUNNING_HEADER_REPEATS
        and len(key) <= RUNNING_HEADER_MAX_CHARS
        and _has_words(key)
    }


def _is_heading(block: Block) -> bool:
    return block.emphasised and _has_words(block.text) and len(block.text) <= MAX_HEADING_CHARS


def _is_title_case(text: str) -> bool:
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", text) if len(word) > 3]
    return bool(words) and sum(word[0].isupper() for word in words) >= 0.75 * len(words)


def _is_capitals_title(block: Block) -> bool:
    letters = [character for character in block.text if character.isalpha()]
    return (
        len(block.text) <= MAX_CATEGORY_CHARS
        and len(letters) >= 4
        and all(character.isupper() for character in letters)
        and not _SENTENCE_END.search(block.text)
    )


def _is_intro_title(text: str) -> bool:
    return bool(_INTRO_TITLE.match(text)) or (
        len(text) <= MAX_INTRO_TITLE_CHARS
        and not _VERB.search(text)
        and bool(_INTRO_WORDS.search(text))
    )


def _run_in(block: Block, *, sentence: bool) -> tuple[str, str] | None:
    """A paragraph's emphasised opening as a heading, and the rest as body, or None."""
    lead = block.lead
    if block.emphasised or not lead or len(lead) > MAX_RUN_IN_CHARS or not _has_words(lead):
        return None
    end = _SENTENCE_END if sentence else _RUN_IN_END
    rest = block.text[len(lead) :].strip() if block.text.startswith(lead) else ""
    stop = _PLAIN_STOP.match(rest)
    if stop and not end.search(lead):
        # Microsoft sets some leads' full stops in plain type.
        lead, rest = lead + stop.group(), rest[stop.end() :].strip()
    if not _has_words(rest):
        return None
    if not end.search(lead) and (sentence or not rest[:1].isupper()):
        # Chevron's run-in leads from fiscal 2015 end with no mark at all; the
        # plain text after them starts a sentence.
        return None
    return lead, rest


@dataclass(frozen=True)
class _Draft:
    category: str | None
    heading: str
    body: tuple[str, ...]


@dataclass(frozen=True)
class SectionReading:
    """How one Item 1A was read: its risks, and what would otherwise be lost silently."""

    drafts: tuple[_Draft, ...]
    #: Emphasised paragraphs with words, too long for a heading, read as body text.
    emphasised_body: tuple[str, ...]
    #: Headings after the first risk with no text under them: dropped.
    lost_headings: tuple[str, ...]
    #: Page numbers dropped as page furniture, in filing order.
    page_numbers: tuple[int, ...] = ()
    #: Paragraphs after the first risk under no heading, outside introduction: dropped.
    unheaded_text: tuple[str, ...] = ()
    #: The section's paragraphs after its heading, page furniture and titles aside:
    #: the whole text, however it splits into risks.
    paragraphs: tuple[str, ...] = ()

    @property
    def section_words(self) -> int:
        return sum(len(paragraph.split()) for paragraph in self.paragraphs)

    @property
    def unclosed_heading(self) -> str | None:
        """The last lost heading, kept for readers of #292's field."""
        return self.lost_headings[-1] if self.lost_headings else None

    def risk_factors(self, *, accession_number: str, fiscal_year: int) -> list[RiskFactor]:
        if not self.drafts:
            raise RiskFactorExtractionError("Item 1A holds no heading followed by text")
        return [
            RiskFactor(
                accession_number=accession_number,
                fiscal_year=fiscal_year,
                index=position,
                category=draft.category,
                heading=draft.heading,
                body=" ".join(draft.body),
            )
            for position, draft in enumerate(self.drafts, start=1)
        ]


def read_item_1a(html: str, *, layout: str = "blocks") -> SectionReading:
    """Item 1A cut into risks, with the blocks that fit no risk cleanly."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    section = item_1a_blocks(html_blocks(html))
    drafts: list[_Draft] = []
    emphasised_body: list[str] = []
    lost: list[str] = []
    page_numbers: list[int] = []
    unheaded: list[str] = []
    introduction = True
    # Introduction under a summary title, whose topics may be headed.
    summary = False
    category: str | None = None
    heading: str | None = None
    body: list[str] = []

    def _drop(text: str) -> None:
        # Text under no heading: introduction, or a risk the reading missed.
        if not introduction and _has_words(text):
            unheaded.append(text)

    def _close() -> None:
        # The current heading ends: as a risk if text sits under it.
        nonlocal heading, body
        if heading is not None and body:
            drafts.append(_Draft(category, heading, tuple(body)))
        elif heading is not None and drafts:
            lost.append(heading)
        heading, body = None, []

    def _open(text: str, first_body: str | None = None) -> None:
        nonlocal heading, body, category
        if heading is not None and not body:
            # A heading over a heading: a category if it is a title, else
            # emphasised introduction.
            if len(heading) <= MAX_CATEGORY_CHARS or (
                len(heading) <= MAX_TITLE_CASE_CATEGORY_CHARS and _is_title_case(heading)
            ):
                category = heading
            elif drafts:
                lost.append(heading)
            heading = None
        _close()
        _start(text, first_body)

    def _start(text: str, first_body: str | None = None) -> None:
        nonlocal heading, body, introduction, summary
        heading, body, introduction, summary = (
            text,
            [first_body] if first_body else [],
            False,
            False,
        )

    furniture = _running_headers(section)
    paragraphs: list[str] = []
    for position, block in enumerate(section[1:], start=1):
        following = section[position + 1].text if position + 1 < len(section) else ""
        number = _PAGE_NUMBER_BLOCK.match(block.text)
        if number:
            page_numbers.append(int(number.group(1)))
            continue
        if _FURNITURE.match(block.text) or _furniture_key(block.text) in furniture:
            continue
        if (not drafts and heading is None and _SECTION_TITLE.match(block.text)) or (
            _SECTION_TITLE_LINE.match(block.text)
        ):
            continue
        paragraphs.append(block.text)
        if layout == "run_in":
            if _is_heading(block) and len(block.text) > MAX_CATEGORY_CHARS:
                # A heading on its own line, too long for a category (Chevron's
                # fiscal 2020 COVID-19 and Noble acquisition risks).
                _close()
                _start(block.text)
                continue
            if _is_heading(block):
                # A category, with its own introduction under it (Exxon Mobil).
                _close()
                category, introduction, summary = block.text, True, False
                continue
            run_in = _run_in(block, sentence=False)
            if run_in is not None:
                _close()
                _start(*run_in)
            elif heading is not None:
                body.append(block.text)
            else:
                _drop(block.text)
            continue
        if layout == "sentences":
            if _is_capitals_title(block):
                _close()
                category = block.text
                continue
            if _is_heading(block) and not _SENTENCE_END.search(block.text):
                if heading is not None:
                    body.append(block.text)
                else:
                    category = block.text
                continue
            run_in = _run_in(block, sentence=True)
            if run_in is not None:
                _close()
                _start(*run_in)
                continue
        if not block.emphasised and _is_capitals_title(block):
            # A category set in capitals but not emphasised (Honeywell from
            # fiscal 2020).
            _close()
            category = block.text
            continue
        if _is_heading(block):
            if _is_intro_title(block.text):
                _close()
                introduction, summary = True, "summary" in block.text.lower()
                continue
            if (
                summary
                and len(block.text) <= MAX_CATEGORY_CHARS
                and (_LEADING_BULLET.match(following) or _BARE_BULLET.match(following))
            ):
                # A topic heading over a summary's bullets (Bank of America's
                # fiscal 2021).
                continue
            if heading is not None and not body and block.text[:1].islower():
                heading = f"{heading} {block.text}"
                continue
            if body and block.text[:1].islower() and not _SENTENCE_END.search(body[-1]):
                # A sentence carried across a page, quoting another risk's
                # heading in emphasis (American Airlines' fiscal 2020).
                body.append(block.text)
                continue
            if (
                len(block.text) <= MAX_TITLE_CASE_CATEGORY_CHARS
                and _CATEGORY_TITLE.match(block.text)
                and not _SENTENCE_END.search(block.text)
            ):
                # A category title, with text under it introducing the category
                # (NextEra). One ending in a full stop is a risk (Honeywell's
                # "Risks related to our ... pension plans may ...", and CVS's
                # "Risks related to the seasonality of our business.").
                _close()
                category, introduction, summary = block.text, True, False
                continue
            _open(block.text)
        elif heading is not None:
            body.append(block.text)
            if block.emphasised and _has_words(block.text):
                emphasised_body.append(block.text)
        else:
            _drop(block.text)
    _close()
    return SectionReading(
        drafts=tuple(drafts),
        emphasised_body=tuple(emphasised_body),
        lost_headings=tuple(lost),
        page_numbers=tuple(page_numbers),
        unheaded_text=tuple(unheaded),
        paragraphs=tuple(paragraphs),
    )


def extract_risk_factors(
    html: str, *, accession_number: str, fiscal_year: int, layout: str = "blocks"
) -> list[RiskFactor]:
    """Item 1A's risk factors, in filing order."""
    return read_item_1a(html, layout=layout).risk_factors(
        accession_number=accession_number, fiscal_year=fiscal_year
    )
