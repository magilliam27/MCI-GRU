"""8-K events with the filer's identity hidden, for E13 (the name-masking test).

E11 found that Jev's 8-K rating lines up with the 2-day return, but Jev reads the
company name and may remember how the story ended. Glasserman and Lin (2023,
arXiv 2309.17322) remove company identifiers from the text and compare. This
module decides, in code, what is removed. Two levels:

- **Name mask** (:func:`mask_names`): the filer's own identifiers. The state's
  company field says ``withheld``; in the text, the filer's current and former
  legal names, their cores without suffixes (``Boeing`` for ``BOEING CO``), a
  distinctive first word of a longer current core (``Goldman`` of
  ``Goldman Sachs``, capitalised or in capitals)
  become ``the Company`` and its tickers ``[ticker]``; every web address becomes ``[website]`` and
  every e-mail address ``[email]``.
- **Deep mask** (:func:`mask_entities`): on top of the name mask, named people,
  other organisations, places and products found by a named-entity tagger, and
  phone numbers, become bracketed placeholders; prior revenue is rounded to one
  significant figure, so scale stays but the exact figure cannot point at a
  company. The tagger runs in the script; this module takes its spans.

Whether a word is safe to mask everywhere is measured, not guessed: a word is
**distinctive** when few other filers' texts spell it in lower case
(:func:`company_vocabulary`): many filers write "target" and "general", almost
none write "boeing". A core that is an ordinary word (``Target``) is masked only
in its capitalised and upper-case spellings. A ticker, or the initials of a
name of three words or more (``AMD``), is masked on its own only when it is
not an ordinary word, and a ticker always after an exchange name
(``NYSE: BA``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING

from mci_gru.filings.events import ITEM_TITLES, MAX_TEXT_CHARS, Event

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

PLACEHOLDER = "the Company"
WITHHELD = "withheld"
TICKER = "[ticker]"
#: Words dropped from the end (and a leading "The") of a legal name to reach its core.
NAME_SUFFIXES = frozenset(
    {
        "ag",
        "co",
        "companies",
        "company",
        "corp",
        "corporation",
        "group",
        "holding",
        "holdings",
        "inc",
        "incorporated",
        "limited",
        "llc",
        "lp",
        "ltd",
        "nv",
        "plc",
        "sa",
        "se",
        "the",
        "trust",
    }
)
#: A word is distinctive when at most this many other filers' texts spell it in lower case.
DISTINCTIVE_MAX_OTHER_FILERS = 20
#: A name's first word is masked on its own only when at most this many other filers
#: write it capitalised: "Boeing" passes, "American" and "Texas" do not.
SHORT_NAME_MAX_OTHER_FILERS = 100
#: Words left out of a name's initials (``Bank of America`` has none worth using).
INITIAL_STOPWORDS = frozenset({"of", "and", "&", "the", "for"})
MIN_INITIALS = 3
MIN_SHORT_NAME_CHARS = 4
EXCHANGES = (
    r"(?:NYSE(?:\s+(?:American|Arca|MKT))?|NASDAQ|Nasdaq(?:GS|GM)?|Nasdaq Global Select Market)"
)
#: Entity labels the deep mask hides, and what each becomes.
ENTITY_PLACEHOLDERS = {
    "PERSON": "[person]",
    "ORG": "[organization]",
    "GPE": "[place]",
    "LOC": "[place]",
    "FAC": "[place]",
    "PRODUCT": "[product]",
}
#: Organisations that say what happened, not who filed: kept in the deep mask.
KEPT_ORGANISATION_WORDS = re.compile(
    r"\b(board|committee|commission|sec|exchange|nyse|nasdaq|court|department|irs|"
    r"fasb|pcaob|council|company)\b",
    re.IGNORECASE,
)

#: Titles and roles a tagger mistakes for people or organisations ("Certain Officers",
#: "Group President"): a span made only of these, and small words, is kept.
ROLE_WORDS = frozenset(
    {
        "accounting",
        "appointment",
        "arrangements",
        "board",
        "certain",
        "chairman",
        "chief",
        "compensatory",
        "controller",
        "counsel",
        "departure",
        "director",
        "directors",
        "election",
        "executive",
        "financial",
        "general",
        "group",
        "officer",
        "officers",
        "operating",
        "president",
        "principal",
        "secretary",
        "senior",
        "treasurer",
        "vice",
        "of",
        "and",
        "the",
        "or",
        "a",
        "an",
    }
)
_TITLE_PARTS = tuple(
    sorted(
        {part.strip().lower() for title in ITEM_TITLES.values() for part in title.split(";")},
        key=len,
        reverse=True,
    )
)
_WORD = re.compile(r"[a-z][a-z0-9]*")
_LOWER_WORD = re.compile(r"(?<![A-Za-z0-9])[a-z][a-z0-9]*(?![A-Za-z0-9])")
_CAPITAL_WORD = re.compile(r"(?<![A-Za-z0-9])[A-Z][a-z][a-z0-9]*(?![A-Za-z0-9])")
_STATE_TAG = re.compile(r"/[A-Za-z]{2,4}/?")
_NAME_TOKEN = re.compile(r"[A-Za-z0-9&'.\-]+")
_URL = re.compile(
    r"\b(?:https?://)?(?:[A-Za-z0-9\-]+\.)+(?:com|net|org|io|co|us|biz|info)\b(?:/[^\s)\"”]*)?",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")
_PHONE = re.compile(r"(?:\+?1[\s.\-]?)?\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")
_DEFINED_AGAIN = re.compile(
    r"the Company\s*\(\s*(?:the\s+)?[“\"]?(?:the\s+)?Company[”\"]?\s*\)", re.IGNORECASE
)


@dataclass(frozen=True)
class Identity:
    """Everything that names one filer: legal names and tickers.

    ``former`` names are masked as whole names only, never by a short form or
    initials: "Alpha Industries" was Skyworks, but "Alpha" alone is not.
    """

    names: tuple[str, ...]
    tickers: tuple[str, ...]
    former: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaskCounts:
    """What one masking pass replaced in one event's text."""

    names: int = 0
    tickers: int = 0
    web: int = 0
    entities: int = 0
    phones: int = 0
    cut_to_limit: bool = False


def clean_name(name: str) -> str:
    """A legal name without SEC state tags: ``CELGENE CORP /DE/`` becomes ``CELGENE CORP``."""
    return re.sub(r"\s+", " ", _STATE_TAG.sub(" ", name)).strip(" ,")


def name_tokens(name: str) -> list[str]:
    return [token.strip(".,'") for token in _NAME_TOKEN.findall(clean_name(name))]


def core_tokens(name: str) -> list[str]:
    """The name's words without a leading "The" or trailing suffixes, such as ``Inc``."""
    tokens = [t for t in name_tokens(name) if t]
    while tokens and tokens[-1].lower().replace(".", "") in NAME_SUFFIXES:
        tokens.pop()
    while tokens and tokens[0].lower() == "the":
        tokens.pop(0)
    return tokens


def display_name(name: str) -> str:
    """A legal name as a reader writes it: ``CELGENE CORP /DE/`` becomes ``Celgene Corp``."""
    cleaned = clean_name(name)
    if cleaned.isupper():
        cleaned = " ".join(word if "&" in word else word.capitalize() for word in cleaned.split())
    return cleaned


def text_words(text: str) -> set[str]:
    """The words a text spells in lower case, web and e-mail addresses left out.

    A filer's name is written capitalised; the same word written in lower case
    by many filers is an ordinary word ("target", "general").
    """
    return set(_LOWER_WORD.findall(_URL.sub(" ", _EMAIL.sub(" ", text))))


def capital_words(text: str) -> set[str]:
    """The words a text spells with a capital first letter, lower-cased, addresses left out."""
    return {w.lower() for w in _CAPITAL_WORD.findall(_URL.sub(" ", _EMAIL.sub(" ", text)))}


@dataclass(frozen=True)
class Vocabulary:
    """For every word, how many filers' texts spell it in lower case, and capitalised."""

    lower: dict[str, int]
    capital: dict[str, int]


def company_vocabulary(texts: Iterable[tuple[str, str]]) -> Vocabulary:
    """Word counts by filer over (company, text) pairs; a company with several texts counts once."""
    lower: dict[str, set[str]] = {}
    capital: dict[str, set[str]] = {}
    for company, text in texts:
        lower.setdefault(company, set()).update(text_words(text))
        capital.setdefault(company, set()).update(capital_words(text))
    return Vocabulary(lower=_count(lower), capital=_count(capital))


def _count(by_company: Mapping[str, set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for words in by_company.values():
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    return counts


def _phrase(tokens: Sequence[str]) -> str:
    """Tokens as a regex that allows any spacing or punctuation between them."""
    return r"[\s,.]*".join(re.escape(token) for token in tokens)


@dataclass(frozen=True)
class NamePattern:
    """One way of writing the filer's name, and whether case must match."""

    regex: str
    case_sensitive: bool


def name_patterns(
    identity: Identity,
    vocabulary: Vocabulary,
    *,
    own_words: set[str],
    own_capitals: set[str],
    max_other: int = DISTINCTIVE_MAX_OTHER_FILERS,
) -> list[NamePattern]:
    """Every pattern that names the filer, longest first.

    ``vocabulary`` comes from :func:`company_vocabulary` over all filers;
    ``own_words`` and ``own_capitals`` are the filer's own, so its own use is
    not counted against a word.
    """

    def others(word: str) -> int:
        word = word.lower()
        return vocabulary.lower.get(word, 0) - (1 if word in own_words else 0)

    def capitalised_elsewhere(word: str) -> int:
        word = word.lower()
        return vocabulary.capital.get(word, 0) - (1 if word in own_capitals else 0)

    suffix = r"(?:[\s,]+(?i:" + "|".join(sorted(NAME_SUFFIXES - {"the"})) + r")\b\.?)*"
    found: dict[str, NamePattern] = {}
    for name in identity.names + identity.former:
        core = core_tokens(name)
        if not core:
            continue
        current = name in identity.names
        whole = [t for t in name_tokens(name) if t]
        if len(whole) > 1:
            # The legal name as written, suffixes and all ("HE Holdings, Inc.").
            regex = r"(?:\bthe\s+)?\b" + _phrase(whole) + r"\.?(?![A-Za-z0-9])"
            found.setdefault(regex.lower(), NamePattern(regex, case_sensitive=False))
        # "AT&T" and "3M" are names, whatever their pieces are.
        distinctive = not core[0].isalpha() or others(core[0]) <= max_other
        if len(core) > 1 or distinctive:
            regex = r"(?:\bthe\s+)?\b" + _phrase(core) + suffix + r"(?![A-Za-z0-9])"
            found.setdefault(regex.lower(), NamePattern(regex, case_sensitive=False))
        elif current:
            # An ordinary word as the whole core ("Target"): its capitalised and upper-case
            # spellings. A former name of this kind ("He Holdings") is left alone.
            for spelling in {core[0].capitalize(), core[0].upper()}:
                regex = r"\b" + re.escape(spelling) + suffix + r"(?![A-Za-z0-9])"
                found.setdefault(regex, NamePattern(regex, case_sensitive=True))
        first = core[0]
        if (
            current
            and len(core) > 1
            and len(first) >= MIN_SHORT_NAME_CHARS
            and first.isalpha()
            and others(first) <= max_other
            and capitalised_elsewhere(first) <= SHORT_NAME_MAX_OTHER_FILERS
        ):
            # Capitalised and upper case only: "Apartment" names AIMCO, "apartment" does not.
            for spelling in {first.capitalize(), first.upper()}:
                regex = r"\b" + re.escape(spelling) + r"(?![A-Za-z0-9])"
                found.setdefault(regex, NamePattern(regex, case_sensitive=True))
        words = [w for w in core if w.lower() not in INITIAL_STOPWORDS and w[:1].isalpha()]
        initials = "".join(w[0] for w in words).upper()
        if current and len(words) >= MIN_INITIALS and others(initials) <= max_other:
            regex = r"(?<![A-Za-z0-9])" + re.escape(initials) + r"(?![A-Za-z0-9])"
            found.setdefault(regex, NamePattern(regex, case_sensitive=True))
    return sorted(found.values(), key=lambda p: len(p.regex), reverse=True)


def ticker_patterns(
    identity: Identity,
    vocabulary: Vocabulary,
    *,
    own_words: set[str],
    max_other: int = DISTINCTIVE_MAX_OTHER_FILERS,
) -> list[str]:
    """Case-sensitive patterns for the filer's tickers.

    After an exchange name or "symbol" always; on their own only when at least
    two characters long and not an ordinary word ("ALL", "KEY").
    """
    patterns: list[str] = []
    for ticker in sorted(set(identity.tickers), key=len, reverse=True):
        escaped = re.escape(ticker)
        patterns.append(r"\b" + EXCHANGES + r"\s*:\s*" + escaped + r"(?![A-Za-z0-9])")
        patterns.append(r"(?:ticker\s+)?symbols?\s+[“\"]?" + escaped + r"[”\"]?(?![A-Za-z0-9])")
        lowered = ticker.lower()
        other = vocabulary.lower.get(lowered, 0) - (1 if lowered in own_words else 0)
        if len(ticker) >= 2 and other <= max_other:
            patterns.append(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])")
    return patterns


def _placeholder(text: str, start: int, matched: str) -> str:
    """``the Company``, capitalised at the start of a sentence or when the match was."""
    before = text[:start].rstrip()
    if matched[:1].isupper() and (not before or before[-1] in '.!?:;\n"“(' or matched[:3] == "The"):
        return "The Company"
    return PLACEHOLDER


def _substitute(pattern: re.Pattern[str], text: str, make: str | None = None) -> tuple[str, int]:
    count = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return make if make is not None else _placeholder(text, match.start(), match.group(0))

    return pattern.sub(swap, text), count


@cache
def _compiled(regex: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(regex, flags)


def mask_text_names(
    text: str, names: Sequence[NamePattern], tickers: Sequence[str]
) -> tuple[str, MaskCounts]:
    """The text with the filer's names, tickers, web and e-mail addresses replaced."""
    # Addresses first, so "www.boeing.com" goes whole rather than as "www.the Company.com".
    text, emails = _substitute(_EMAIL, text, "[email]")
    text, sites = _substitute(_URL, text, "[website]")
    name_count = 0
    for pattern in names:
        flags = 0 if pattern.case_sensitive else re.IGNORECASE
        text, n = _substitute(_compiled(pattern.regex, flags), text)
        name_count += n
    ticker_count = 0
    for regex in tickers:
        text, n = _substitute(_compiled(regex), text, TICKER)
        ticker_count += n
    text = _DEFINED_AGAIN.sub(lambda m: m.group(0)[:3] + " Company", text)
    return text, MaskCounts(names=name_count, tickers=ticker_count, web=emails + sites)


def _cut(text: str) -> tuple[str, bool]:
    return (text[:MAX_TEXT_CHARS], True) if len(text) > MAX_TEXT_CHARS else (text, False)


def mask_names(
    event: Event, names: Sequence[NamePattern], tickers: Sequence[str]
) -> tuple[Event, MaskCounts]:
    """The event as the name mask leaves it: company withheld, own identifiers out of the text."""
    text, counts = mask_text_names(event.text, names, tickers)
    text, cut = _cut(text)
    return replace(event, company_name=WITHHELD, text=text), replace(counts, cut_to_limit=cut)


def round_revenue(usd: float | None) -> float | None:
    """Revenue to one significant figure: 76.6 billion becomes 80 billion."""
    if usd is None or usd == 0:
        return usd
    digits = math.floor(math.log10(abs(usd)))
    return float(round(usd / 10**digits) * 10**digits)


def mask_entities(event: Event, spans: Iterable[tuple[int, int, str]]) -> tuple[Event, MaskCounts]:
    """A name-masked event with tagged entities and phone numbers replaced, revenue rounded.

    ``spans`` are (start, end, label) offsets into ``event.text`` from a tagger.
    Left alone: spans already holding a placeholder, defined terms in quotes,
    organisations in :data:`KEPT_ORGANISATION_WORDS`, spans made only of
    :data:`ROLE_WORDS`, and anything inside an 8-K item title.
    """
    text = event.text
    lowered = text.lower()
    titles = [
        (m.start(), m.end()) for part in _TITLE_PARTS for m in re.finditer(re.escape(part), lowered)
    ]
    pieces: list[str] = []
    last = 0
    count = 0
    for start, end, label in sorted(spans):
        if start < last or label not in ENTITY_PLACEHOLDERS:
            continue
        found = text[start:end]
        if "Company" in found or "[" in found or any(q in found for q in '“”"'):
            continue
        if set(_WORD.findall(found.lower())) <= ROLE_WORDS:
            continue
        if any(a <= start and end <= b for a, b in titles):
            continue
        if label == "ORG" and KEPT_ORGANISATION_WORDS.search(found):
            continue
        pieces.append(text[last:start])
        pieces.append(ENTITY_PLACEHOLDERS[label])
        last = end
        count += 1
    pieces.append(text[last:])
    text, phones = _substitute(_PHONE, "".join(pieces), "[phone]")
    text, cut = _cut(text)
    masked = replace(event, text=text, prior_revenue_usd=round_revenue(event.prior_revenue_usd))
    return masked, MaskCounts(entities=count, phones=phones, cut_to_limit=cut)
