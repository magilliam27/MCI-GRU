"""An 8-K read with the company's latest earnings release beside it (E18).

E16 rated most Russell 2000 8-Ks neutral. Mac's idea is that an officer change
or a loan amendment reads differently after a bad quarter than after a good one,
so experiment E18 asks E16's rating question again with the company's latest
earnings release in the state. What this module decides:

- **Which release is the latest.** The company's latest original 8-K listing
  item 2.02 accepted before the 8-K, if it came no more than
  ``RELEASE_MAX_DAYS`` days earlier (about one quarter and a 10-K's lag);
  older than that, it describes a quarter the market has already moved past,
  and the event has no context.
- **Which events are drawn.** Draws are taken in turn, each from one E16 rating
  stratum (neutral, bad, good), ranked by
  :func:`mci_gru.filings.earnings.draw_rank`, so a draw never depends on the
  frame's order and later draws never repeat an earlier one's events.
- **What Jev reads.** E16's state unchanged, with the release (its filing day
  and text, cut at ``MAX_TEXT_CHARS`` as every event text) placed just before the
  8-K's own text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from mci_gru.filings.earnings import EARNINGS_ITEM, draw_rank

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

RELEASE_MAX_DAYS = 120
#: The state key the release is given under, and the key it goes before.
RELEASE_KEY = "latest_earnings_release_before_this_8k"
EVENT_TEXT_KEY = "event_text"
#: E16's rating answers, folded into the strata draws are taken from.
STRATA = {
    "very_bad": "bad",
    "bad": "bad",
    "neutral": "neutral",
    "good": "good",
    "very_good": "good",
}


def latest_release(
    eight_ks: Iterable[tuple[pd.Timestamp, str, tuple[str, ...]]],
    accepted: pd.Timestamp,
    accession: str,
) -> tuple[tuple[pd.Timestamp, str] | None, str]:
    """The latest earnings release ``(accepted, accession)`` before an 8-K, and why not.

    ``eight_ks`` holds ``(accepted, accession, items)`` for the company's original
    8-Ks. The reason is empty when a release is found, else ``no_release`` or
    ``release_too_old``.
    """
    earlier = sorted(
        (when, other)
        for when, other, items in eight_ks
        if other != accession and EARNINGS_ITEM in items and when < accepted
    )
    if not earlier:
        return None, "no_release"
    when, other = earlier[-1]
    if (accepted - when) / pd.Timedelta(days=1) > RELEASE_MAX_DAYS:
        return None, "release_too_old"
    return (when, other), ""


def draw_strata(
    frame: Sequence[tuple[str, str]], draws: Sequence[tuple[str, str, int, int]]
) -> dict[str, str]:
    """Each drawn event's draw name, for ``(name, stratum, size, seed)`` draws taken in turn.

    ``frame`` holds ``(event_id, stratum)``. A size below zero takes every event
    left in the stratum. An event goes to the first draw that ranks it within its
    size.
    """
    ids = [event_id for event_id, _ in frame]
    if len(set(ids)) != len(ids):
        raise ValueError("an event id repeats in the frame")
    unknown = {stratum for _, stratum, _, _ in draws} - set(STRATA.values())
    if unknown:
        raise ValueError(f"unknown strata: {', '.join(sorted(unknown))}")
    drawn: dict[str, str] = {}
    for name, stratum, size, seed in draws:
        left = [e for e, s in frame if s == stratum and e not in drawn]
        ranked = sorted(left, key=lambda event_id: draw_rank(event_id, seed))
        drawn.update(dict.fromkeys(ranked if size < 0 else ranked[:size], name))
    return drawn


def context_state(state: dict[str, Any], release_filed: str, release_text: str) -> dict[str, Any]:
    """An 8-K's state with its latest earnings release placed before the 8-K's text."""
    if RELEASE_KEY in state or EVENT_TEXT_KEY not in state:
        raise ValueError(f"the state must hold {EVENT_TEXT_KEY} and no {RELEASE_KEY}")
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key == EVENT_TEXT_KEY:
            out[RELEASE_KEY] = {"filed": release_filed, "text": release_text}
        out[key] = value
    return out
