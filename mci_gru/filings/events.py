"""8-K events for E10 (#305): what Jev sees, and the baseline it must beat.

E10 asks how good or bad an 8-K event is for shareholders. This module decides:

- **The event record** the Data lane publishes, one JSON object per 8-K:
  ``event_id`` (unique, such as the accession number), ``company`` (the key used
  across the filings data, such as ``jpm``), ``company_name``, ``items`` (the
  8-K's item codes, such as ``["5.02", "9.01"]``), ``text`` (item sections plus
  press-release exhibit 99.1, already cut at ``MAX_TEXT_CHARS``),
  ``prior_revenue_usd`` and ``prior_revenue_fiscal_year`` (from the company's
  latest 10-K filed before the 8-K, or null), and ``prior_90d_items`` (one list
  of item codes per other 8-K the company filed in the previous 90 days, oldest
  first). Every field is known at filing time.
- **The state** Jev reads (:func:`event_state`): the company name, the item
  codes with their titles, the prior revenue in words, the earlier 8-Ks' item
  codes and the text. Nothing dated after the filing.
- **The baseline**, fixed on the card before any label: a reader who knows the
  item code and counts words. First the item code's lean
  (:func:`item_lean`: negative-leaning items above auditor changes above the
  rest), then, within a lean, the Loughran-McDonald net tone (negative minus
  positive words as a share of all words). :func:`baseline_badness` folds both
  into one number whose order is exactly that ranking.
- **Jev's scores**: the rating score weights each option's probability from -2
  (very bad) to +2 (very good); the bad score is the probability of bad plus
  very bad.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The card's negative-leaning items: agreement ended, bankruptcy, restructuring,
#: impairment, delisting notice, restatement.
NEGATIVE_LEANING_ITEMS = frozenset({"1.02", "1.03", "2.05", "2.06", "3.01", "4.02"})
AUDITOR_CHANGE_ITEMS = frozenset({"4.01"})
RATING_WEIGHTS = {"very_bad": -2, "bad": -1, "neutral": 0, "good": 1, "very_good": 2}
RATING_LABELS = tuple(RATING_WEIGHTS)
SIZE_LABELS = ("under_1_percent", "between_1_and_5_percent", "over_5_percent", "no_amount_stated")
MAX_TEXT_CHARS = 20_000
#: Form 8-K item titles, as the SEC's form names them.
ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation or an Obligation under an "
    "Off-Balance Sheet Arrangement of a Registrant",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation "
    "or an Obligation under an Off-Balance Sheet Arrangement",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or "
    "Standard; Transfer of Listing",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements or a Related Audit "
    "Report or Completed Interim Review",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; "
    "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "5.05": "Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the "
    "Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
EVENT_FIELDS = (
    "event_id",
    "company",
    "company_name",
    "items",
    "text",
    "prior_revenue_usd",
    "prior_revenue_fiscal_year",
    "prior_90d_items",
)

_WORD = re.compile(r"[a-z]+")


class EventDataError(ValueError):
    """An event record breaks the contract above."""


@dataclass(frozen=True)
class Event:
    event_id: str
    company: str
    company_name: str
    items: tuple[str, ...]
    text: str
    prior_revenue_usd: float | None
    prior_revenue_fiscal_year: int | None
    prior_90d_items: tuple[tuple[str, ...], ...]


def load_events(path: Path) -> list[Event]:
    """Every event in a JSON-lines file, checked against the contract; ids must be unique."""
    events: list[Event] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [field for field in EVENT_FIELDS if field not in raw]
        if missing:
            raise EventDataError(f"{path}:{number}: missing {', '.join(missing)}")
        if not raw["items"]:
            raise EventDataError(f"{path}:{number}: no item codes")
        if len(raw["text"]) > MAX_TEXT_CHARS:
            raise EventDataError(f"{path}:{number}: text over {MAX_TEXT_CHARS:,} characters")
        revenue, year = raw["prior_revenue_usd"], raw["prior_revenue_fiscal_year"]
        events.append(
            Event(
                event_id=str(raw["event_id"]),
                company=str(raw["company"]),
                company_name=str(raw["company_name"]),
                items=tuple(str(code) for code in raw["items"]),
                text=str(raw["text"]),
                prior_revenue_usd=None if revenue is None else float(revenue),
                prior_revenue_fiscal_year=None if year is None else int(year),
                prior_90d_items=tuple(
                    tuple(str(code) for code in codes) for codes in raw["prior_90d_items"]
                ),
            )
        )
    ids = [event.event_id for event in events]
    if len(set(ids)) != len(ids):
        raise EventDataError(f"{path}: an event id repeats")
    return events


def item_label(code: str) -> str:
    """An item code with its title, such as ``5.02 Departure of Directors ...``."""
    title = ITEM_TITLES.get(code)
    return f"{code} {title}" if title else code


def revenue_words(usd: float | None, fiscal_year: int | None) -> str:
    """Prior annual revenue as Jev reads it, such as ``$211.9 billion (fiscal 2023 10-K)``."""
    if usd is None:
        return "not available"
    if abs(usd) >= 1e9:
        amount = f"${usd / 1e9:,.1f} billion"
    elif abs(usd) >= 1e6:
        amount = f"${usd / 1e6:,.1f} million"
    else:
        amount = f"${usd:,.0f}"
    return f"{amount} (fiscal {fiscal_year} 10-K)" if fiscal_year is not None else amount


def event_state(event: Event) -> dict[str, Any]:
    """What Jev reads for one event: context known at filing time, then the text."""
    return {
        "company": event.company_name,
        "item_codes": [item_label(code) for code in event.items],
        "prior_annual_revenue": revenue_words(
            event.prior_revenue_usd, event.prior_revenue_fiscal_year
        ),
        "other_8k_item_codes_in_prior_90_days": [
            ", ".join(codes) for codes in event.prior_90d_items
        ]
        or ["none"],
        "event_text": event.text,
    }


def item_lean(items: tuple[str, ...] | list[str]) -> int:
    """The item code's lean: 2 negative-leaning, 1 auditor change, 0 the rest."""
    codes = set(items)
    if codes & NEGATIVE_LEANING_ITEMS:
        return 2
    if codes & AUDITOR_CHANGE_ITEMS:
        return 1
    return 0


def tone_shares(
    text: str, negative: frozenset[str], positive: frozenset[str]
) -> tuple[float, float]:
    """Shares of a text's words (every occurrence) on the negative and positive lists."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0, 0.0
    return (
        sum(word in negative for word in words) / len(words),
        sum(word in positive for word in words) / len(words),
    )


def baseline_badness(lean: int, net_tone: float) -> float:
    """Lean first, net tone within it: higher is worse.

    Net tone lies in [-1, 1], so ``(1 + net_tone) / 3`` lies in [0, 2/3] and an
    event can never pass one with a stronger lean.
    """
    return lean + (1.0 + net_tone) / 3.0


def rating_score(probabilities: dict[str, float]) -> float:
    """Each rating option's probability times its weight, -2 (very bad) to +2."""
    return sum(weight * probabilities.get(name, 0.0) for name, weight in RATING_WEIGHTS.items())


def bad_score(probabilities: dict[str, float]) -> float:
    """The probability of bad plus very bad."""
    return probabilities.get("bad", 0.0) + probabilities.get("very_bad", 0.0)
