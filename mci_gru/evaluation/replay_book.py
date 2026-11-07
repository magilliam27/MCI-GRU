"""The book a saved-prediction replay trades, and its daily P&L (#340).

The replay used to trade a one-session open-to-open hold, read each stock's return
from a per-stock row shift (so a missing bar stretched one "day" over two sessions),
drop a holding with no next price and re-weight the rest, and hold Alphabet twice.
This module is the strategy it trades instead:

* **Default (E2).** Buy at close ``D+1`` and sell at close ``D+5``: the training
  label's legs, so the replay trades exactly the return the model learned. The book is
  ``tranches`` sleeves opened in turn, as the in-run summary's tranche book is
  (``mci_gru/evaluation/tranche_book.py``); with no costs and no multi-class issuer the
  two books are the same numbers. The paper's open ``D+1`` to open ``D+2`` daily hold
  stays as the labelled ``open_to_open`` variant, on the same machinery.
* **Weights.** Equal at entry, then buy and hold inside a tranche: weights drift with
  prices and are never re-equalised until the sleeve's next tranche.
* **Costs** are charged on the session a trade happens, on the notional traded: the
  sleeve's rebalance from its old tranche to its new one, and a forced sale (below).
* **Returns, exits and ties come from the core** (``mci_gru/evaluation/outcomes.py``):
  prices on the panel's own session axis, so a missing bar never moves an endpoint. A
  holding with no exit price books its last price inside the hold, is sold there, and
  the slot holds cash to the tranche's exit (E3); a pick with no entry price holds cash.
  Each such event is recorded. Ties order by score, then kdcode.
* **One share class per issuer (E4).** A portfolio holds at most one class of an
  issuer (the higher score; the next name fills the slot), and the benchmark weights
  each issuer once, split across its classes.
* **Rankable on D** is the PIT membership on ``D``, a close printed on ``D`` and a
  saved score: the run writes a score for every name its PIT tradable mask admits, so
  the scored set is the run's tradable universe. The benchmark is that universe, on
  the same date and tranche schedule as the candidates.

Price returns only: the panel carries no dividends (E6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mci_gru.config import Legs
from mci_gru.evaluation.outcomes import order_by_score, realized_returns

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mci_gru.evaluation.outcomes import RealizedReturns

CLOSE_TRANCHES = "close_tranches"
OPEN_TO_OPEN = "open_to_open"
STRATEGIES = (CLOSE_TRANCHES, OPEN_TO_OPEN)

# Ticker roots of the S&P 500 issuers that have listed more than one share class since
# 2016. A kdcode's root is the text before its exchange suffix (``GOOGL.OQ`` ->
# ``GOOGL``). Pass ``issuers`` (the CLI's ``--issuer_map``) for any other pair.
SHARE_CLASS_ISSUERS: dict[str, str] = {
    "GOOG": "Alphabet",
    "GOOGL": "Alphabet",
    "FOX": "Fox",
    "FOXA": "Fox",
    "NWS": "News Corp",
    "NWSA": "News Corp",
    "DISCA": "Discovery",
    "DISCB": "Discovery",
    "DISCK": "Discovery",
    "UA": "Under Armour",
    "UAA": "Under Armour",
    "LBTYA": "Liberty Global",
    "LBTYB": "Liberty Global",
    "LBTYK": "Liberty Global",
}


@dataclass(frozen=True)
class ReplayStrategy:
    """What the replay buys, at which price, and how the hold is staggered."""

    name: str
    price: str  # the panel column traded: "close" or "open"
    legs: Legs
    tranches: int

    def __post_init__(self) -> None:
        if self.name not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.name!r}")
        if self.price not in ("close", "open"):
            raise ValueError(f"price must be 'close' or 'open', got {self.price!r}")
        if isinstance(self.tranches, bool) or not isinstance(self.tranches, int):
            raise ValueError(f"tranches must be a positive integer, got {self.tranches!r}")
        if self.tranches <= 0 or self.legs.hold_sessions % self.tranches:
            raise ValueError(
                f"tranches ({self.tranches}) must be positive and divide the hold of "
                f"{self.legs.hold_sessions} sessions"
            )

    @classmethod
    def close_tranches(cls, label_t: int, tranches: int | None = None) -> ReplayStrategy:
        """The default: the training label's legs at the close, one sleeve per session."""
        legs = Legs.from_label(label_t)
        return cls(CLOSE_TRANCHES, "close", legs, tranches or legs.hold_sessions)

    @classmethod
    def open_to_open(
        cls, holding_period: int = 1, rebalance_style: str = "staggered"
    ) -> ReplayStrategy:
        """The paper's variant: buy at open ``D+1``, hold ``holding_period`` sessions.

        ``staggered`` runs one sleeve per held session; ``block`` one sleeve that
        rebalances every ``holding_period`` signal dates.
        """
        if rebalance_style not in ("staggered", "block"):
            raise ValueError(
                f"rebalance_style must be 'staggered' or 'block', got {rebalance_style!r}"
            )
        legs = Legs(entry_session=1, exit_session=1 + holding_period)
        tranches = holding_period if rebalance_style == "staggered" else 1
        return cls(OPEN_TO_OPEN, "open", legs, tranches)

    @property
    def stride(self) -> int:
        """Signal dates between two tranche openings."""
        return self.legs.hold_sessions // self.tranches

    def describe(self) -> str:
        legs = self.legs
        return (
            f"{self.name}: buy at {self.price} D+{legs.entry_session}, sell at {self.price} "
            f"D+{legs.exit_session} ({legs.hold_sessions} sessions), {self.tranches} "
            f"staggered tranche{'s' if self.tranches != 1 else ''}, buy-and-hold weights"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "price": self.price,
            "entry_session": self.legs.entry_session,
            "exit_session": self.legs.exit_session,
            "hold_sessions": self.legs.hold_sessions,
            "tranches": self.tranches,
        }


@dataclass(frozen=True)
class CostModel:
    """Retail costs: half the round-trip spread plus slippage on each side traded."""

    bid_ask_spread: float = 0.0
    slippage: float = 0.0

    @property
    def per_side(self) -> float:
        return self.bid_ask_spread / 2.0 + self.slippage


@dataclass(frozen=True)
class ReplayBook:
    """Daily P&L of the replay's book and its benchmark, one session axis for both.

    ``sessions[t]`` is the session daily return ``t`` ends on, from the first entry
    close (cash before it) to the last exit close. ``costs`` and
    ``turnover`` are fractions of the book's value at the previous session; turnover
    is one-way (half the notional bought plus sold).
    """

    strategy: ReplayStrategy
    top_k: int
    sessions: list[str]
    gross_returns: np.ndarray
    net_returns: np.ndarray
    benchmark_returns: np.ndarray
    costs: np.ndarray
    turnover: np.ndarray
    signal_dates: list[str]  # the dates a tranche was opened on
    period_returns: np.ndarray  # each opened tranche's booked hold return, gross
    benchmark_period_returns: np.ndarray
    holdings: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    tranches_opened: int = 0
    tranches_unbooked: int = 0
    rank_gate: dict[str, Any] = field(default_factory=dict)

    @property
    def excess_period_returns(self) -> np.ndarray:
        return self.period_returns - self.benchmark_period_returns


def issuer_keys(kdcodes: list[str], issuers: Mapping[str, str] | None = None) -> list[str]:
    """The issuer of each kdcode; a name with one listed class is its own issuer."""
    out = []
    for kdcode in kdcodes:
        code = str(kdcode)
        if issuers and code in issuers:
            out.append(str(issuers[code]))
            continue
        root = code.split(".", 1)[0].split("^", 1)[0]
        out.append(SHARE_CLASS_ISSUERS.get(root, code))
    return out


def select_picks(
    scores: np.ndarray,
    rankable: np.ndarray,
    kdcodes: list[str],
    issuers: list[str],
    top_k: int,
    *,
    held: list[int] | None = None,
    held_ranks: Mapping[int, int] | None = None,
    min_rank_drop: int | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Top-k picks by the core tie rule, at most one share class per issuer.

    Returns the picks and every rankable name's 1-based rank. With ``min_rank_drop``
    (the rank-drop gate), a name in ``held`` is kept unless its rank worsened by at
    least ``min_rank_drop`` since ``held_ranks``; the freed slots are refilled from the
    top of today's ranking.
    """
    arr = np.asarray(scores, dtype=np.float64)
    allowed = np.isfinite(arr) & np.asarray(rankable, dtype=bool)
    order = order_by_score(arr, kdcodes)
    ranked = [int(j) for j in order if allowed[j]]
    ranks = {j: r + 1 for r, j in enumerate(ranked)}

    candidates = ranked
    if min_rank_drop is not None and held and held_ranks is not None:
        survivors = [
            j
            for j in sorted((j for j in held if j in ranks), key=ranks.__getitem__)
            if j not in held_ranks or ranks[j] - held_ranks[j] < min_rank_drop
        ]
        candidates = survivors + [j for j in ranked if j not in set(survivors)]

    picks: list[int] = []
    taken: set[str] = set()
    for j in candidates:
        if len(picks) == top_k:
            break
        if issuers[j] in taken:
            continue
        picks.append(j)
        taken.add(issuers[j])
    return np.asarray(picks, dtype=int), ranks


def issuer_weights(names: np.ndarray, issuers: list[str]) -> np.ndarray:
    """Equal weight per issuer, split equally across the issuer's classes in ``names``."""
    names = np.asarray(names, dtype=int)
    if names.size == 0:
        return np.asarray([], dtype=np.float64)
    keys = [issuers[j] for j in names]
    counts = pd.Series(keys).value_counts()
    weights = np.array([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weights / len(counts)


def price_matrix(panel: pd.DataFrame, kdcodes: list[str], price: str) -> pd.DataFrame:
    """``price`` on the panel's session axis (every date in ``panel``), one column per name."""
    frame = panel[["kdcode", "dt", price]].copy()
    frame["dt"] = frame["dt"].astype(str)
    frame["kdcode"] = frame["kdcode"].astype(str)
    frame[price] = pd.to_numeric(frame[price], errors="coerce")
    sessions = sorted(frame["dt"].unique())
    pivot = frame.pivot_table(index="dt", columns="kdcode", values=price, aggfunc="mean")
    return pivot.reindex(index=sessions, columns=list(kdcodes))


@dataclass
class _Tranche:
    signal: int  # row in the signal dates
    sleeve: int
    entry: int  # session index of the entry price
    exit: int
    names: np.ndarray
    weights: np.ndarray  # sum to 1
    growth: np.ndarray  # (exit - entry + 1, names): value of 1 in each name, cash once sold
    sold_at: np.ndarray  # session index each name is sold at; -1 if it never opened


def _tranche(signal, sleeve, entry, exit_, names, weights, prices) -> _Tranche:
    path = prices[entry : exit_ + 1][:, names]
    finite = np.isfinite(path)
    filled = pd.DataFrame(path).ffill().to_numpy(dtype=np.float64)
    first = path[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = filled / first
    opened = np.isfinite(first)
    growth[:, ~opened] = 1.0
    # The last session in the hold each name printed a price: its sale (E3 when early).
    last = np.where(finite.any(axis=0), finite.shape[0] - 1 - np.argmax(finite[::-1], axis=0), 0)
    sold_at = np.where(opened, entry + last, -1)
    return _Tranche(signal, sleeve, entry, exit_, names, weights, growth, sold_at)


def _run_sleeve(
    tranches: list[_Tranche],
    first: int,
    last: int,
    capital: float,
    per_side: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """One sleeve's value on sessions ``first..last`` after trades, and its trades.

    Returns the value path, the cost and the notional traded on each session, and the
    trades as ``(session, name, signed notional)`` (positive buys).
    """
    n = last - first + 1
    value = np.full(n, np.nan)
    cost = np.zeros(n)
    traded = np.zeros(n)
    trades: list[tuple[int, int, float]] = []
    cash = capital
    positions: dict[int, float] = {}  # name -> value held at the close of ``since``
    since = first

    def trade(session: int, name: int, notional: float) -> float:
        if notional == 0.0:
            return 0.0
        trades.append((session, name, notional))
        traded[session - first] += abs(notional)
        charge = per_side * abs(notional)
        cost[session - first] += charge
        return charge

    def sell_all(session: int) -> None:
        nonlocal cash
        for name, held in positions.items():
            cash += held - trade(session, name, -held)
        positions.clear()

    for tranche in sorted(tranches, key=lambda t: t.entry):
        a, b = tranche.entry, tranche.exit
        if a < since:
            raise ValueError("a sleeve's tranches overlap: signal dates must be in session order")
        if a > since:
            sell_all(since)
            value[since - first : a - first + 1] = cash
        book_value = cash + sum(positions.values())
        targets = {
            int(name): book_value * float(w)
            for name, w, sold in zip(tranche.names, tranche.weights, tranche.sold_at, strict=True)
            if sold >= 0
        }
        charges = 0.0
        for name in sorted(set(positions) | set(targets)):
            charges += trade(a, name, targets.get(name, 0.0) - positions.get(name, 0.0))
        # The cost comes out of the whole sleeve in proportion.
        invested = book_value - charges
        scale = invested / book_value if book_value else 0.0
        stock = tranche.growth * (scale * book_value * tranche.weights)[None, :]
        path = stock.sum(axis=1)
        positions = {}
        liquidation = np.zeros(b - a + 1)
        for col, (name, sold) in enumerate(zip(tranche.names, tranche.sold_at, strict=True)):
            if sold < 0:
                continue
            if sold < b:  # E3: sold at its last price, cash after
                liquidation[sold - a :] += trade(sold, int(name), -float(stock[sold - a, col]))
            else:
                positions[int(name)] = float(stock[-1, col])
        path = path - liquidation
        value[a - first : b - first + 1] = path
        cash = float(path[-1]) - sum(positions.values())
        since = b

    sell_all(since)
    value[since - first :] = cash
    return value, cost, traded, trades


def build_replay_book(
    panel: pd.DataFrame,
    scores: np.ndarray,
    dates: list[str],
    kdcodes: list[str],
    *,
    strategy: ReplayStrategy,
    top_k: int,
    members: np.ndarray | None = None,
    costs: CostModel | None = None,
    min_rank_drop: int | None = None,
    issuers: Mapping[str, str] | None = None,
) -> ReplayBook:
    """Trade ``scores`` (``dates`` x ``kdcodes``, NaN unscored) under ``strategy``.

    ``panel`` holds ``kdcode``, ``dt``, ``close`` (and ``open`` for the open variant);
    its dates are the session axis, so pass every name the run could hold. ``members``
    is PIT membership on each signal date (all members when None).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k!r}")
    costs = costs or CostModel()
    dates = [str(d) for d in dates]
    kdcodes = [str(k) for k in kdcodes]
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(dates), len(kdcodes)):
        raise ValueError(f"scores {scores.shape} must be (dates, kdcodes)")
    member = (
        np.ones(scores.shape, dtype=bool) if members is None else np.asarray(members, dtype=bool)
    )
    keys = issuer_keys(kdcodes, issuers)

    panel = panel.assign(kdcode=panel["kdcode"].astype(str), dt=panel["dt"].astype(str))
    traded_panel = panel[["kdcode", "dt", strategy.price]].rename(columns={strategy.price: "close"})
    realized = realized_returns(traded_panel, dates, strategy.legs, kdcodes=kdcodes)
    priced = price_matrix(traded_panel, kdcodes, "close")
    closes = price_matrix(panel, kdcodes, "close")
    # Scored at close D, and holdable as the core sees it: a price in the traded column
    # on D (the open, for the open-to-open variant).
    printed = (closes.reindex(dates).notna() & priced.reindex(dates).notna()).to_numpy()
    rankable = member & printed & np.isfinite(scores)

    sessions = priced.index.tolist()
    position = {session: t for t, session in enumerate(sessions)}
    prices = priced.to_numpy(dtype=np.float64)

    stride, n_sleeves = strategy.stride, strategy.tranches
    book: list[list[_Tranche]] = [[] for _ in range(n_sleeves)]
    bench: list[list[_Tranche]] = [[] for _ in range(n_sleeves)]
    sleeve_state: list[tuple[list[int], dict[int, int]] | None] = [None] * n_sleeves
    signal_rows: list[int] = []
    period, bench_period = [], []
    holdings: list[dict[str, Any]] = []
    span: list[int] = []
    unbooked = gate_exits = 0

    for i in range(0, len(dates), stride):
        entry, exit_ = realized.entry_dates[i], realized.exit_dates[i]
        if entry is None or exit_ is None:
            continue
        a, b = position[entry], position[exit_]
        span += [a, b]
        sleeve = (i // stride) % n_sleeves
        state = sleeve_state[sleeve]
        picks, ranks = select_picks(
            scores[i],
            rankable[i],
            kdcodes,
            keys,
            top_k,
            held=state[0] if state else None,
            held_ranks=state[1] if state else None,
            min_rank_drop=min_rank_drop,
        )
        if state and min_rank_drop is not None:
            gate_exits += sum(
                1
                for j in state[0]
                if j in ranks and j in state[1] and ranks[j] - state[1][j] >= min_rank_drop
            )
        # The gate compares against what the sleeve holds; an empty or unbooked tranche
        # leaves it in cash.
        sleeve_state[sleeve] = None
        universe = np.flatnonzero(rankable[i])
        if universe.size:
            w = issuer_weights(universe, keys)
            bench[sleeve].append(_tranche(i, sleeve, a, b, universe, w, prices))
        if picks.size == 0:
            continue
        if not np.all(np.isfinite(realized.returns[i, picks])):
            unbooked += 1
            continue
        weights = np.full(picks.size, 1.0 / picks.size)
        book[sleeve].append(_tranche(i, sleeve, a, b, picks, weights, prices))
        sleeve_state[sleeve] = (picks.tolist(), ranks)
        signal_rows.append(i)
        period.append(float(weights @ realized.returns[i, picks]))
        bench_period.append(
            float(issuer_weights(universe, keys) @ realized.returns[i, universe])
            if universe.size
            else np.nan
        )
        holdings += _holding_rows(realized, i, sleeve, picks, weights, ranks, scores, keys)

    if not span:
        empty = np.asarray([], dtype=np.float64)
        return ReplayBook(strategy, top_k, [], empty, empty, empty, empty, empty, [], empty, empty)

    first, last = min(span), max(span)
    capital = 1.0 / n_sleeves
    gross_value = np.zeros(last - first + 1)
    net_value = np.zeros(last - first + 1)
    bench_value = np.zeros(last - first + 1)
    cost = np.zeros(last - first + 1)
    traded = np.zeros(last - first + 1)
    trade_rows: list[dict[str, Any]] = []
    for sleeve in range(n_sleeves):
        gross_value += _run_sleeve(book[sleeve], first, last, capital, 0.0)[0]
        value, c, notional, trades = _run_sleeve(book[sleeve], first, last, capital, costs.per_side)
        net_value += value
        cost += c
        traded += notional
        bench_value += _run_sleeve(bench[sleeve], first, last, capital, 0.0)[0]
        trade_rows += [
            {
                "date": sessions[s],
                "sleeve": sleeve,
                "kdcode": kdcodes[name],
                "action": "BUY" if notional > 0 else "SELL",
                "notional": abs(notional),
            }
            for s, name, notional in trades
        ]

    held_events = {(h["signal_date"], h["kdcode"]) for h in holdings}
    events = [
        {
            "signal_date": e.date,
            "kdcode": e.kdcode,
            "kind": e.kind,
            "scheduled_session": e.scheduled_session,
            "booked_session": e.booked_session,
        }
        for e in realized.events
        if (e.date, e.kdcode) in held_events
    ]
    # The book holds 1.0 in cash before the first entry close, so the first session's
    # return is that close's buying cost (and zero gross).
    gross_value, net_value, bench_value = (
        np.concatenate([[1.0], v]) for v in (gross_value, net_value, bench_value)
    )
    previous = net_value[:-1]
    return ReplayBook(
        strategy=strategy,
        top_k=top_k,
        sessions=sessions[first : last + 1],
        gross_returns=gross_value[1:] / gross_value[:-1] - 1.0,
        net_returns=net_value[1:] / previous - 1.0,
        benchmark_returns=bench_value[1:] / bench_value[:-1] - 1.0,
        costs=cost / previous,
        turnover=traded / previous / 2.0,
        signal_dates=[dates[i] for i in signal_rows],
        period_returns=np.asarray(period, dtype=np.float64),
        benchmark_period_returns=np.asarray(bench_period, dtype=np.float64),
        holdings=holdings,
        trades=sorted(trade_rows, key=lambda r: (r["date"], r["sleeve"], r["kdcode"])),
        events=events,
        tranches_opened=len(signal_rows),
        tranches_unbooked=unbooked,
        rank_gate={
            "enabled": min_rank_drop is not None,
            "min_rank_drop": min_rank_drop,
            "gate_exits": gate_exits if min_rank_drop is not None else 0,
        },
    )


def _holding_rows(
    realized: RealizedReturns,
    i: int,
    sleeve: int,
    picks: np.ndarray,
    weights: np.ndarray,
    ranks: Mapping[int, int],
    scores: np.ndarray,
    keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for j, w in zip(picks.tolist(), weights.tolist(), strict=True):
        booked = float(realized.returns[i, j])
        rows.append(
            {
                "signal_date": realized.dates[i],
                "entry_date": realized.entry_dates[i],
                "exit_date": realized.exit_dates[i],
                "booked_exit_date": realized.booked_exit_dates[i][j],
                "sleeve": sleeve,
                "kdcode": realized.kdcodes[j],
                "issuer": keys[j],
                "rank": ranks[j],
                "score": float(scores[i, j]),
                "weight": w,
                "stock_return": booked,
                "contribution": w * booked,
            }
        )
    return rows
