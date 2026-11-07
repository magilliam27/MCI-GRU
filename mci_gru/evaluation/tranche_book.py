"""The staggered tranche book a label implies, and its daily P&L (#336).

A label of close ``D+1`` to close ``D+5`` is a four-session hold opened on every
signal date, so consecutive dates' returns share three sessions. Summing or
annualising those returns as if each were one day counts every session about four
times. The book here is the strategy those returns describe, traded session by
session:

* The book is ``tranches`` sleeves, each holding a share of the capital. A signal
  date opens a tranche every ``hold / tranches`` dates, in turn across the sleeves, so
  each sleeve's next tranche enters at the close its previous one exits on.
* Inside a tranche, the picks are equally weighted at the entry close and then held
  (buy and hold, no daily rebalancing). A sleeve keeps its own capital from tranche to
  tranche, and the sleeves are never re-equalised.
* Closes come from the panel's own session axis. A pick with no close on a session
  carries its last close, so a missing exit books the last close in the hold and then
  cash (E3), and a pick with no entry close holds cash. A tranche's full-hold growth
  therefore equals ``1 +`` the core's booked return for its date.
* A sleeve holds cash before its first tranche and after its last one, and in place
  of a tranche with no picks. The book runs from the first scheduled entry to the
  last scheduled exit, so it ramps in over the first ``hold - 1`` sessions and out
  over the last ones, as a staggered book does when traded. On a short window this
  dilutes the absolute numbers (about 4% of a one-year Sharpe on four sleeves);
  excess numbers net it, since the universe book ramps on the same schedule.

Selection reads only the scores and the rankable mask, through the core's tie rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mci_gru.evaluation.outcomes import RealizedReturns, select_top_k, top_k_tranche_returns

if TYPE_CHECKING:
    from mci_gru.config import Legs


@dataclass(frozen=True)
class TrancheBook:
    """Daily P&L of one staggered book; ``top_k=None`` is the equal-weight universe."""

    legs: Legs
    tranches: int
    top_k: int | None
    sessions: list[str]  # the close each daily return ends on
    daily_returns: np.ndarray
    tranches_opened: int
    # Tranche dates left in cash because a pick had no booked return, which only
    # happens when the scores or mask come from another panel than ``realized``.
    tranches_unbooked: int

    @property
    def cumulative_return(self) -> float:
        """Compounded return of the book over its sessions; NaN for an empty book."""
        if not self.daily_returns.size:
            return float("nan")
        return float(np.prod(1.0 + self.daily_returns) - 1.0)


def signal_date_closes(panel: pd.DataFrame, realized: RealizedReturns) -> np.ndarray:
    """True where the stock printed a finite close on the signal date itself.

    The rankable set for a run without PIT masks: a name that could have been bought.
    """
    closes = _close_matrix(panel, realized.kdcodes)
    out = np.zeros((len(realized.dates), len(realized.kdcodes)), dtype=bool)
    for i, date in enumerate(realized.dates):
        if date in closes.index:
            out[i] = np.isfinite(closes.loc[date].to_numpy(dtype=np.float64))
    return out


def book_picks(
    scores: np.ndarray,
    rankable_row: np.ndarray,
    kdcodes: list[str],
    top_k: int | None,
) -> np.ndarray:
    """Names one tranche holds: the top-k by the core rule, or every rankable name."""
    if top_k is None:
        return np.flatnonzero(np.asarray(rankable_row, dtype=bool))
    return select_top_k(scores, kdcodes, top_k, rankable=rankable_row)


def period_returns(
    scores: np.ndarray,
    rankable_mask: np.ndarray,
    realized: RealizedReturns,
    top_k: int | None,
) -> np.ndarray:
    """Each signal date's equal-weight booked return over the whole hold.

    For a top-k this is the core's ``top_k_tranche_returns``; ``top_k=None`` is the
    equal-weight rankable universe under the same rules. A date whose hold is
    unresolved, or a pick without a booked return, is NaN.
    """
    if top_k is not None:
        return top_k_tranche_returns(scores, rankable_mask, realized, top_k)
    mask = _checked_mask(scores, rankable_mask, realized)
    out = np.full(len(realized.dates), np.nan)
    for i in range(len(realized.dates)):
        if realized.entry_dates[i] is None or realized.exit_dates[i] is None:
            continue
        picks = np.flatnonzero(mask[i])
        if picks.size:
            out[i] = float(np.mean(realized.returns[i, picks]))
    return out


def tranche_book(
    panel: pd.DataFrame,
    scores: np.ndarray,
    rankable_mask: np.ndarray,
    realized: RealizedReturns,
    *,
    top_k: int | None,
    tranches: int,
) -> TrancheBook:
    """Daily P&L of the ``tranches``-sleeve book over ``realized``'s signal dates.

    ``panel`` must be the frame ``realized`` was built from (``kdcode``, ``dt``,
    ``close``), so both read one session axis. ``scores`` and ``rankable_mask`` align
    with ``realized``'s dates and kdcodes.
    """
    legs = realized.legs
    hold = legs.hold_sessions
    if isinstance(tranches, bool) or not isinstance(tranches, int) or tranches <= 0:
        raise ValueError(f"tranches must be a positive integer, got {tranches!r}")
    if hold % tranches:
        raise ValueError(f"tranches ({tranches}) must divide the hold of {hold} sessions")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive or None, got {top_k!r}")
    scores = np.asarray(scores, dtype=np.float64)
    mask = _checked_mask(scores, rankable_mask, realized)

    closes = _close_matrix(panel, realized.kdcodes)
    sessions = closes.index.tolist()
    position = {session: t for t, session in enumerate(sessions)}
    close = closes.to_numpy(dtype=np.float64)

    stride = hold // tranches
    sleeves: list[list[tuple[int, int, np.ndarray]]] = [[] for _ in range(tranches)]
    opened = unbooked = 0
    # The book spans the schedule, not the picks, so every book over the same
    # ``realized`` covers the same sessions and their daily returns subtract.
    span: list[int] = []
    for i in range(0, len(realized.dates), stride):
        entry, exit_ = realized.entry_dates[i], realized.exit_dates[i]
        if entry is None or exit_ is None:
            continue
        span += [position[entry], position[exit_]]
        picks = book_picks(scores[i], mask[i], realized.kdcodes, top_k)
        if picks.size == 0:
            continue
        if not np.all(np.isfinite(realized.returns[i, picks])):
            unbooked += 1
            continue
        a, b = position[entry], position[exit_]
        sleeves[(i // stride) % tranches].append((a, b, _tranche_growth(close[a : b + 1, picks])))
        opened += 1

    if not span:
        return TrancheBook(legs, tranches, top_k, [], np.asarray([], dtype=np.float64), 0, 0)

    first, last = min(span), max(span)
    value = np.zeros(last - first + 1)
    for sleeve in sleeves:
        value += _sleeve_value(sleeve, first, last, 1.0 / tranches)
    daily = value[1:] / value[:-1] - 1.0
    return TrancheBook(
        legs=legs,
        tranches=tranches,
        top_k=top_k,
        sessions=sessions[first + 1 : last + 1],
        daily_returns=daily,
        tranches_opened=opened,
        tranches_unbooked=unbooked,
    )


def _close_matrix(panel: pd.DataFrame, kdcodes: list[str]) -> pd.DataFrame:
    """Closes on the panel's session axis (every date in ``panel``), one column per kdcode."""
    frame = panel[["kdcode", "dt", "close"]].copy()
    frame["dt"] = frame["dt"].astype(str)
    frame["kdcode"] = frame["kdcode"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    sessions = sorted(frame["dt"].unique())
    pivot = frame.pivot_table(index="dt", columns="kdcode", values="close", aggfunc="mean")
    return pivot.reindex(index=sessions, columns=list(kdcodes))


def _checked_mask(
    scores: np.ndarray, rankable_mask: np.ndarray, realized: RealizedReturns
) -> np.ndarray:
    mask = np.asarray(rankable_mask, dtype=bool)
    expected = (len(realized.dates), len(realized.kdcodes))
    if np.shape(scores) != expected or mask.shape != expected:
        raise ValueError(
            f"scores {np.shape(scores)} and rankable {mask.shape} must match realized {expected}"
        )
    return mask


def _tranche_growth(path: np.ndarray) -> np.ndarray:
    """Value of 1 invested equally at the first row's closes, held to each later row.

    A missing close carries the stock's last close (cash from then on); a stock with
    no entry close holds cash throughout.
    """
    filled = pd.DataFrame(path).ffill().to_numpy(dtype=np.float64)
    entry = path[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = filled / entry
    ratio[:, ~np.isfinite(entry)] = 1.0
    return ratio.mean(axis=1)


def _sleeve_value(
    tranches: list[tuple[int, int, np.ndarray]], first: int, last: int, capital: float
) -> np.ndarray:
    """One sleeve's value on sessions ``first..last``; cash between its tranches."""
    value = np.full(last - first + 1, np.nan)
    since = first
    for a, b, growth in sorted(tranches, key=lambda tranche: tranche[0]):
        if a < since:
            raise ValueError("a sleeve's tranches overlap: signal dates must be in session order")
        value[since - first : a - first + 1] = capital
        value[a - first : b - first + 1] = capital * growth
        capital *= float(growth[-1])
        since = b
    value[since - first :] = capital
    return value
