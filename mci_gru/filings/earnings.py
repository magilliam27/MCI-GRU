"""Earnings releases read against the previous one (E17).

An earnings 8-K (item 2.02) is upbeat by design, so experiment E17 asks Jev
whether a quarter came in better or worse than the company's previous release
had set investors up to expect. What this module decides:

- **Which release is the previous one.** The company's latest original 8-K
  listing item 2.02 accepted before this one. A pair counts when that release
  was accepted ``PREVIOUS_MIN_DAYS`` to ``PREVIOUS_MAX_DAYS`` days earlier (one
  quarter, give or take) and no other 2.02 8-K came in the ``CLEAR_DAYS``
  before this one: such a near repeat is usually a correction or a second part
  of the same news, and which of the two is "the release" is unclear.
- **Which candidates are drawn.** Each event's place in a draw is its
  ``sha256(seed:event_id)``, so a draw never depends on the order or size of the
  candidate list it is taken from, and draws taken one after another from the
  same pool never overlap.
- **What Jev reads.** The company, the filing day and both releases' texts (item
  sections plus press-release exhibit, as an E10 event text), each cut at
  ``MAX_TEXT_CHARS``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mci_gru.filings.jev import Question

EARNINGS_ITEM = "2.02"
PREVIOUS_MIN_DAYS = 45
PREVIOUS_MAX_DAYS = 120
CLEAR_DAYS = 14
#: A release shorter than this says too little to compare (a bare "see exhibit").
MIN_RELEASE_CHARS = 500
#: The setup question's options and their weights, worst first.
SETUP_WEIGHTS = {"much_worse": -2, "worse": -1, "in_line": 0, "better": 1, "much_better": 2}
GUIDANCE_LABELS = ("raised", "reaffirmed", "lowered", "withdrawn", "introduced", "none")
PROMISE_LABELS = ("kept", "mixed", "missed", "none_set")
QUESTION_LABELS = {
    "setup": tuple(SETUP_WEIGHTS),
    "guidance": GUIDANCE_LABELS,
    "promise": PROMISE_LABELS,
}


def previous_release(
    eight_ks: Iterable[tuple[pd.Timestamp, str, tuple[str, ...]]],
    accepted: pd.Timestamp,
    accession: str,
) -> tuple[tuple[pd.Timestamp, str] | None, str]:
    """The previous earnings release ``(accepted, accession)``, or None, and why not.

    ``eight_ks`` holds ``(accepted, accession, items)`` for the company's original
    8-Ks. The reason is empty for a usable pair, else ``no_previous``,
    ``previous_within_clear_days``, ``previous_too_recent`` or ``previous_too_old``.
    """
    earlier = sorted(
        (when, other)
        for when, other, items in eight_ks
        if other != accession and EARNINGS_ITEM in items and when < accepted
    )
    if not earlier:
        return None, "no_previous"
    when, other = earlier[-1]
    days = (accepted - when) / pd.Timedelta(days=1)
    if days <= CLEAR_DAYS:
        return None, "previous_within_clear_days"
    if days < PREVIOUS_MIN_DAYS:
        return None, "previous_too_recent"
    if days > PREVIOUS_MAX_DAYS:
        return None, "previous_too_old"
    return (when, other), ""


def draw_rank(event_id: str, seed: int) -> int:
    """An event's place in the draw with ``seed``: lower is drawn first."""
    return int(hashlib.sha256(f"{seed}:{event_id}".encode()).hexdigest()[:16], 16)


def draw(event_ids: Sequence[str], draws: Sequence[tuple[str, int, int]]) -> dict[str, str]:
    """Each drawn event's draw name, for ``(name, size, seed)`` draws taken in turn.

    A size below zero takes every event left. An event goes to the first draw
    that ranks it within its size, so later draws never repeat an earlier one's.
    """
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("an event id repeats in the candidates")
    left = list(event_ids)
    drawn: dict[str, str] = {}
    for name, size, seed in draws:
        ranked = sorted(left, key=lambda event_id: draw_rank(event_id, seed))
        chosen = ranked if size < 0 else ranked[:size]
        drawn.update(dict.fromkeys(chosen, name))
        taken = set(chosen)
        left = [event_id for event_id in left if event_id not in taken]
    return drawn


def release_state(
    company: str,
    filed: str,
    current_text: str,
    previous_filed: str,
    previous_text: str,
) -> dict[str, Any]:
    """What Jev reads for one earnings release: it, and the previous one."""
    return {
        "company": company,
        "filed": filed,
        "current_results_release": current_text,
        "previous_results_release": {"filed": previous_filed, "text": previous_text},
    }


def question_order(question: Question, event_id: str) -> list[str]:
    """The event's own shuffle of one question's options, the same on every run.

    Seeded by event and question together, so the questions asked in one call
    are shuffled independently.
    """
    names = [name for name, _ in question.options]
    key = f"{event_id}:{question.question_id}".encode()
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16)
    return [names[i] for i in np.random.default_rng(seed).permutation(len(names))]


def setup_score(probabilities: dict[str, float]) -> float:
    """Each setup option's probability times its weight, much worse -2 to much better +2."""
    return sum(weight * probabilities.get(name, 0.0) for name, weight in SETUP_WEIGHTS.items())
