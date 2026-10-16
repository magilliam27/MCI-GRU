"""S&P 500 membership over time, and which 10-Ks a membership calls for (#309).

Experiment E7 tests whether what Jev reads in 10-Ks predicts returns, on the
companies that were in the S&P 500 when each report was filed. What this module
decides:

- **Who was a member when.** Dated member lists (one full list per change date)
  become spells: a ticker's first list date to the first list date it is gone
  from. A ticker's spells are then joined to the company it belonged to, so a
  company keeps its membership across a ticker change (FB to META).
- **Whose name is whose.** A ticker can pass to another company (Pepco Holdings'
  POM), so a ticker's company is confirmed by name against SEC's names for the
  CIK (current and former). ``names_match`` is the test.
- **Which 10-Ks to read.** Every 10-K a company filed while a member, and, for
  each of those whose previous 10-K was filed before the company joined, that
  previous one too, so every member's report can be paired with the one before
  it. Reports are paired by filing date, not fiscal year, so a company that
  changed its year end keeps both of that year's reports.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: Words that say what kind of company a name is, not which one.
_LEGAL_WORDS = frozenset(
    (
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "companies",
        "cos",
        "plc",
        "ltd",
        "limited",
        "llc",
        "lp",
        "group",
        "holdings",
        "holding",
        "hldgs",
        "the",
        "new",
        "class",
        "a",
        "b",
        "c",
        "common",
        "stock",
        "shares",
        "trust",
        "sa",
        "nv",
        "ag",
        "de",
        "del",
        "and",
        "of",
    )
)
_NON_WORD = re.compile(r"[^a-z0-9]+")
#: Names this close, after legal words are dropped, are the same name.
NAME_RATIO = 0.85
#: A successor files its first 10-K within this long of its predecessor's last
#: (a year, plus a late filing); after longer, a matching name is a new company
#: (SanDisk, bought in 2016, and Sandisk, spun off in 2025).
MAX_SUCCESSOR_GAP = pd.Timedelta(days=550)


@dataclass(frozen=True)
class Spell:
    """One unbroken run of a ticker in the index; ``end`` is the first day out."""

    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp | None

    def holds(self, when: pd.Timestamp) -> bool:
        return self.start <= when and (self.end is None or when < self.end)


def apply_changes(
    lists: Sequence[tuple[pd.Timestamp, frozenset[str]]],
    changes: Iterable[tuple[pd.Timestamp, str | None, str | None]],
) -> list[tuple[pd.Timestamp, frozenset[str]]]:
    """Member lists extended by ``(date, added, removed)`` changes after the last list."""
    lists = sorted(lists, key=lambda entry: entry[0])
    last_date, members = lists[-1][0], set(lists[-1][1])
    extended = list(lists)
    by_date: dict[pd.Timestamp, list[tuple[str | None, str | None]]] = {}
    for date, added, removed in changes:
        if date > last_date:
            by_date.setdefault(date, []).append((added, removed))
    for date in sorted(by_date):
        for added, removed in by_date[date]:
            if removed:
                members.discard(removed)
            if added:
                members.add(added)
        extended.append((date, frozenset(members)))
    return extended


def spells(lists: Sequence[tuple[pd.Timestamp, frozenset[str]]]) -> list[Spell]:
    """Every ticker's runs in the index, from dated member lists."""
    opened: dict[str, pd.Timestamp] = {}
    found: list[Spell] = []
    for date, members in sorted(lists, key=lambda entry: entry[0]):
        for ticker in sorted(set(opened) - members):
            found.append(Spell(ticker, opened.pop(ticker), date))
        for ticker in sorted(members - set(opened)):
            opened[ticker] = date
    found.extend(Spell(ticker, start, None) for ticker, start in opened.items())
    return sorted(found, key=lambda spell: (spell.ticker, spell.start))


def name_key(name: str) -> str:
    """A company name without accents, punctuation, case or legal words."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    plain = plain.lower().replace("&", " and ").replace("'", "").replace("\u2019", "")
    words = [word for word in _NON_WORD.split(plain) if word and word not in _LEGAL_WORDS]
    return " ".join(words)


def names_match(name: str, other: str) -> bool:
    """Whether two names are one company's, once legal words go.

    They match when equal, equal in another word order (SEC files people's
    names surname first: "SMITH A O CORP" is A. O. Smith), when one is the
    other's initials ("IBM"), when one is the start of the other in words
    ("Amazon" and "AMAZON COM INC"), or when nearly equal by ``NAME_RATIO``.
    """
    left, right = name_key(name), name_key(other)
    if not left or not right:
        return False
    if left.replace(" ", "") == right.replace(" ", ""):
        return True
    short, long = sorted((left.split(), right.split()), key=len)
    if sorted(short) == sorted(long):
        return True
    if len(short) == 1 and len(long) > 1 and short[0] == "".join(word[0] for word in long):
        return True
    if long[: len(short)] == short:
        return True
    return SequenceMatcher(None, left, right).ratio() >= NAME_RATIO


def member_filings(
    filings: Sequence[tuple[pd.Timestamp, str]], company_spells: Iterable[Spell]
) -> list[tuple[str, str]]:
    """``(accession, role)`` for the 10-Ks a company's membership calls for.

    ``filings`` holds ``(accepted, accession)`` for every one of the company's
    10-Ks. ``role`` is ``member`` for a report filed while the company was in
    the index, and ``prior`` for the report just before a member report that
    was not itself filed while a member: its pair partner.
    """
    ordered = sorted(filings)
    company_spells = list(company_spells)
    member = [any(spell.holds(when) for spell in company_spells) for when, _ in ordered]
    roles: dict[int, str] = {}
    for position, is_member in enumerate(member):
        if is_member:
            roles[position] = "member"
            if position > 0 and not member[position - 1]:
                roles.setdefault(position - 1, "prior")
    return [(ordered[position][1], roles[position]) for position in sorted(roles)]


def names_on(submissions: Mapping[str, Any], when: pd.Timestamp) -> list[str]:
    """The names a CIK went by on a date, from its SEC submissions file.

    ``formerNames`` dates each name; the current name holds after the last
    former one ends, and whenever no dated name covers the day.
    """
    current = str(submissions.get("name", ""))
    dated = []
    for entry in submissions.get("formerNames", []) or []:
        start = _naive(entry.get("from"))
        end = _naive(entry.get("to"))
        dated.append((str(entry.get("name", "")), start, end))
    held = [name for name, start, end in dated if start <= when <= end]
    others_end = max((end for name, _, end in dated if name != current), default=None)
    if not held or others_end is None or when > others_end:
        held.append(current)
    return list(dict.fromkeys(name for name in held if name))


def _naive(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(str(value)) if value else pd.Timestamp.min
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


def successors(
    anchor: str,
    filers: Sequence[tuple[str, pd.Timestamp, pd.Timestamp, Sequence[str], Sequence[str]]],
) -> list[str]:
    """``anchor`` with the CIKs that carried the same company's reports before or after it.

    ``filers`` holds, per candidate CIK, its first and last 10-K ever and its
    names when it filed them. One CIK follows another when it filed no 10-K
    until the other had filed its last, and its first name matches the other's
    last, within ``MAX_SUCCESSOR_GAP``: a new holding company (Disney in 2019). An acquirer that took the
    ticker (ACE renamed Chubb) filed 10-Ks before; a spin-off whose parent kept
    filing (Arconic from Howmet) began while the parent went on; and a company
    that took another's place under a new name (Fox from Twenty-First Century
    Fox) has a different name. None of those follows.
    """
    by_cik = {entry[0]: entry for entry in filers}
    chain = [anchor]

    def _follows(later: tuple, earlier: tuple) -> bool:
        return earlier[2] < later[1] <= earlier[2] + MAX_SUCCESSOR_GAP and any(
            names_match(name, other) for name in later[3] for other in earlier[4]
        )

    while True:
        before = [
            entry for entry in filers if entry[0] not in chain and _follows(by_cik[chain[0]], entry)
        ]
        if not before:
            break
        chain.insert(0, max(before, key=lambda entry: entry[2])[0])
    while True:
        after = [
            entry
            for entry in filers
            if entry[0] not in chain and _follows(entry, by_cik[chain[-1]])
        ]
        if not after:
            break
        chain.append(min(after, key=lambda entry: entry[1])[0])
    return chain
