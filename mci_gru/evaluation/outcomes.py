"""The one evaluation definition of returns, rankability, exits and ties (#335).

The in-run summary, the replay and the side modules each used to decide for
themselves what a stock returned after signal date ``D``, who could be ranked on
``D``, what happens when a holding stops trading, and how tied scores order. This
module is the single answer they call:

* **Realized return** reads closes on the panel's own session axis through
  ``resolve_labels``, the resolver the training label uses, so a missing bar never
  moves an endpoint. The default legs (close ``D+1`` to close ``D+5``) are exactly
  the training label.
* **Rankable on D** is the PIT tradable mask on ``D``. It never reads label
  availability, so a name without a future price keeps its rank.
* **Missing close (E3).** With no exit close, the last close inside the hold is
  booked and the slot holds cash until the tranche's exit. With no entry close the
  position never opens and the slot holds cash. Both are recorded as events.
* **Ties** order by score descending, then kdcode ascending.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mci_gru.config import Legs
from mci_gru.data.preprocessing import LabelResolution, resolve_labels

if TYPE_CHECKING:
    from mci_gru.data.pit import PITMaskSet


@dataclass(frozen=True)
class MissingCloseEvent:
    """A position whose entry or exit close did not exist."""

    date: str
    kdcode: str
    kind: str  # "missing_entry" or "missing_exit"
    scheduled_session: str
    booked_session: str | None  # None when the position never opened


@dataclass(frozen=True)
class RealizedReturns:
    """Booked return per ``(signal date, stock)`` under one set of legs.

    Where both closes exist, ``returns`` is the training label's value, whether or
    not the stock printed on the signal date. The E3 rules apply only to a stock with
    a close on the signal date (one that could have been picked): a missing exit books
    the last close in the hold, then cash to the exit (``booked_exit_dates`` names the
    session booked); a missing entry books cash (0.0). A stock with no close on the
    signal date and an incomplete label is NaN, as is every cell of a date whose hold
    runs past the panel's last session.
    """

    dates: list[str]
    kdcodes: list[str]
    legs: Legs
    entry_dates: list[str | None]
    exit_dates: list[str | None]
    returns: np.ndarray  # (dates, stocks)
    booked_exit_dates: list[list[str | None]]
    events: tuple[MissingCloseEvent, ...]


def realized_returns(
    panel: pd.DataFrame,
    dates: list[str],
    legs: Legs | None = None,
    kdcodes: list[str] | None = None,
) -> RealizedReturns:
    """Booked return for each signal date and stock, on the panel's session axis.

    ``panel`` needs ``kdcode``, ``dt`` and ``close``. ``kdcodes`` fixes the column
    order and defaults to the panel's kdcodes, sorted.
    """
    legs = legs or Legs()
    dates = [str(date) for date in dates]
    if kdcodes is None:
        kdcodes = sorted(panel["kdcode"].astype(str).unique())
    kdcodes = list(kdcodes)

    # The session axis comes from every name in the panel, never from the requested
    # columns alone: a subset would rebuild the axis from its own rows, and a missing
    # bar would then move an endpoint as a per-stock row shift does.
    universe = sorted(set(panel["kdcode"].astype(str)) | set(kdcodes))
    position = {kdcode: j for j, kdcode in enumerate(universe)}
    columns = [position[kdcode] for kdcode in kdcodes]

    def at(offset: int) -> LabelResolution:
        # resolve_labels(label_t=k) reads the close at session D+k as its exit.
        resolved = resolve_labels(panel, universe, dates, offset)
        return dataclasses.replace(
            resolved,
            kdcodes=kdcodes,
            entry_close=resolved.entry_close[:, columns],
            exit_close=resolved.exit_close[:, columns],
        )

    entry = at(legs.entry_session)
    exit_ = at(legs.exit_session)
    entry_dates = entry.exit_dates
    exit_dates = exit_.exit_dates
    entry_close = entry.exit_close
    exit_close = exit_.exit_close.copy()
    booked = np.empty(exit_close.shape, dtype=object)
    booked[:] = np.asarray(exit_dates, dtype=object)[:, None]

    resolved = np.array(
        [e is not None and x is not None for e, x in zip(entry_dates, exit_dates, strict=True)]
    )
    holdable = _closes_on_signal_date(panel, kdcodes, dates) & resolved[:, None]
    opened = holdable & np.isfinite(entry_close)
    never_opened = holdable & ~opened
    missing_exit = opened & ~np.isfinite(exit_close)

    # Missing exit: walk back through the hold to the last close the stock printed.
    # The entry close itself ends the walk, so every opened position finds one.
    pending = missing_exit.copy()
    for offset in range(legs.exit_session - 1, legs.entry_session - 1, -1):
        if not pending.any():
            break
        earlier = at(offset)
        found = pending & np.isfinite(earlier.exit_close)
        exit_close[found] = earlier.exit_close[found]
        booked[found] = np.broadcast_to(
            np.asarray(earlier.exit_dates, dtype=object)[:, None], booked.shape
        )[found]
        pending &= ~found

    with np.errstate(divide="ignore", invalid="ignore"):
        returns = exit_close / entry_close - 1.0
    returns[~resolved] = np.nan
    returns[never_opened] = 0.0  # no fill: the slot holds cash
    booked[~np.isfinite(returns) | never_opened] = None

    events = [
        MissingCloseEvent(dates[i], kdcodes[j], "missing_entry", entry_dates[i], None)
        if never_opened[i, j]
        else MissingCloseEvent(dates[i], kdcodes[j], "missing_exit", exit_dates[i], booked[i, j])
        for i, j in zip(*np.nonzero(missing_exit | never_opened), strict=True)
    ]

    return RealizedReturns(
        dates=dates,
        kdcodes=kdcodes,
        legs=legs,
        entry_dates=entry_dates,
        exit_dates=exit_dates,
        returns=returns,
        booked_exit_dates=booked.tolist(),
        events=tuple(events),
    )


def _closes_on_signal_date(panel: pd.DataFrame, kdcodes: list[str], dates: list[str]) -> np.ndarray:
    """True where the stock printed a finite close on the signal date itself."""
    rows = panel.loc[panel["kdcode"].isin(kdcodes), ["kdcode", "dt", "close"]]
    rows = rows[np.isfinite(pd.to_numeric(rows["close"], errors="coerce"))]
    present = set(zip(rows["dt"].astype(str), rows["kdcode"].astype(str), strict=True))
    out = [[(date, kdcode) in present for kdcode in kdcodes] for date in dates]
    return np.array(out, dtype=bool).reshape(len(dates), len(kdcodes))


def rankable(masks: PITMaskSet) -> np.ndarray:
    """Who may be ranked on each date: the PIT tradable mask, never label availability."""
    return np.asarray(masks.tradable, dtype=bool).copy()


def order_by_score(scores: np.ndarray, kdcodes: list[str]) -> np.ndarray:
    """Indices by score descending, ties by kdcode ascending; NaN scores last."""
    arr = np.asarray(scores, dtype=np.float64)
    keys = np.asarray(list(kdcodes))
    if arr.shape != keys.shape:
        raise ValueError(f"scores and kdcodes differ in shape: {arr.shape} != {keys.shape}")
    return np.lexsort((keys, -arr))


def select_top_k(
    scores: np.ndarray,
    kdcodes: list[str],
    top_k: int,
    rankable: np.ndarray | None = None,
) -> np.ndarray:
    """Top-k indices among scored, rankable names under the core tie rule."""
    if top_k <= 0:
        return np.asarray([], dtype=int)
    arr = np.asarray(scores, dtype=np.float64)
    allowed = np.isfinite(arr)
    if rankable is not None:
        allowed &= np.asarray(rankable, dtype=bool)
    order = order_by_score(arr, kdcodes)
    return order[allowed[order]][:top_k]


def top_k_tranche_returns(
    scores: np.ndarray,
    rankable_mask: np.ndarray,
    realized: RealizedReturns,
    top_k: int,
) -> np.ndarray:
    """Equal-weight booked return of each date's top-k tranche.

    Selection reads only scores and the rankable mask; what a pick returned never
    decides whether it was picked. A date whose hold is unresolved is NaN. A pick
    with no booked return (masks built from another panel) also makes its date NaN,
    so the mismatch shows instead of being averaged away. The caller aligns the
    rows and columns of ``scores`` and ``rankable_mask`` with ``realized``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(rankable_mask, dtype=bool)
    expected = (len(realized.dates), len(realized.kdcodes))
    if scores.shape != expected or mask.shape != expected:
        raise ValueError(
            f"scores {scores.shape} and rankable {mask.shape} must match realized {expected}"
        )
    out = np.full(len(realized.dates), np.nan)
    for i in range(len(realized.dates)):
        if realized.exit_dates[i] is None or realized.entry_dates[i] is None:
            continue
        picks = select_top_k(scores[i], realized.kdcodes, top_k, rankable=mask[i])
        if picks.size:
            out[i] = float(np.mean(realized.returns[i, picks]))
    return out
