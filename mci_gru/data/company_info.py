"""Point-in-time company information for the candidate models (#393).

Daily price and volume look used up on S&P 500 names: five model families and two
one-line signals all land at the same pooled IC (the go/no-go review of 2026-11-12).
This module holds information none of them has seen, dated by when it became public,
so a score on signal date ``D`` can read only what was known at the close of ``D``.

**Earnings events.** One row per quarterly earnings release: the company's panel code
(``kdcode``), the day it was released (``report_date``), whether before the open or
after the close (``timing``), and the reported and consensus EPS. They come from
EODHD's earnings history (``Earnings::History`` of its fundamentals endpoint), which
covers delisted companies and dates each release. Its EPS figures are split-adjusted
to the day they were served, so a surprise is scaled by a split-adjusted close
(``mci_gru.evaluation.company_candidates``).

When a release is public by the close: a release before the open on session ``r`` is
known at the close of ``r``; one after the close, or at a time not recorded, is known
at the close of the next session. The second rule is the cautious one: a release whose
time is unknown and that came out before the open is only used a session late.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

EARNINGS_COLUMNS = (
    "kdcode",
    "report_date",
    "timing",
    "fiscal_period_end",
    "eps_actual",
    "eps_estimate",
    "source_symbol",
)
TIMINGS = ("before", "after", "unknown")
_EODHD_TIMING = {"BeforeMarket": "before", "AfterMarket": "after"}


def eodhd_timing(value: object) -> str:
    """EODHD's ``beforeAfterMarket`` as ``before``, ``after`` or ``unknown``."""
    return _EODHD_TIMING.get(value, "unknown") if isinstance(value, str) else "unknown"


def symbol_segments(symbol_record: Mapping[str, Mapping]) -> pd.DataFrame:
    """``kdcode, symbol, start, end`` from a price pull's ``*_symbols.json`` record.

    Each panel code was priced from one or more EODHD symbols over dated segments; an
    open start or end is the widest date. The same segments decide whose earnings are
    whose, so a ticker another company took over never lends its releases.
    """
    rows = []
    for kdcode, record in symbol_record.items():
        for segment in record.get("segments", []):
            symbol = segment.get("symbol")
            if not symbol:
                continue
            rows.append(
                (
                    str(kdcode),
                    str(symbol),
                    str(segment.get("start") or "0000-01-01"),
                    str(segment.get("end") or "9999-12-31"),
                )
            )
    return pd.DataFrame(rows, columns=["kdcode", "symbol", "start", "end"])


def earnings_from_eodhd(
    history_by_symbol: Mapping[str, Mapping[str, Mapping]],
    segments: pd.DataFrame,
    first_date: str,
    last_date: str,
) -> pd.DataFrame:
    """The panel's earnings events from EODHD ``Earnings::History`` blocks.

    ``history_by_symbol`` maps an EODHD symbol to its history (keyed by fiscal period
    end). A release goes to the panel code whose segment for that symbol covers its
    report date; one inside no segment, without a report date, or outside
    ``first_date..last_date`` is left out. A release without a reported EPS is kept
    with NaN, so its date still counts.
    """
    rows = []
    for symbol, history in history_by_symbol.items():
        owned = segments.loc[segments["symbol"] == symbol]
        if owned.empty or not history:
            continue
        for period_end, item in history.items():
            report = item.get("reportDate")
            if not isinstance(report, str) or not first_date <= report <= last_date:
                continue
            covering = owned.loc[(owned["start"] <= report) & (owned["end"] >= report)]
            for kdcode in covering["kdcode"]:
                rows.append(
                    {
                        "kdcode": kdcode,
                        "report_date": report,
                        "timing": eodhd_timing(item.get("beforeAfterMarket")),
                        "fiscal_period_end": str(item.get("date") or period_end),
                        "eps_actual": _number(item.get("epsActual")),
                        "eps_estimate": _number(item.get("epsEstimate")),
                        "source_symbol": symbol,
                    }
                )
    events = pd.DataFrame(rows, columns=list(EARNINGS_COLUMNS))
    return events.sort_values(["kdcode", "report_date"], kind="stable").reset_index(drop=True)


def _number(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def check_earnings(events: pd.DataFrame) -> pd.DataFrame:
    """The events with their columns checked and typed, or a ValueError."""
    missing = [c for c in EARNINGS_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"earnings events lack columns {missing}")
    out = events.loc[:, list(EARNINGS_COLUMNS)].copy()
    for column in ("kdcode", "report_date", "timing", "fiscal_period_end", "source_symbol"):
        out[column] = out[column].astype(str)
    bad_timing = sorted(set(out["timing"]) - set(TIMINGS))
    if bad_timing:
        raise ValueError(f"unknown earnings timings {bad_timing}")
    dates = pd.to_datetime(out["report_date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("an earnings report_date is not a YYYY-MM-DD date")
    for column in ("eps_actual", "eps_estimate"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float64)
    return out


def load_earnings(path: str) -> pd.DataFrame:
    """A preserved earnings-events CSV, checked."""
    return check_earnings(pd.read_csv(path, dtype={"kdcode": str, "report_date": str}))


def known_session(
    report_dates: Sequence[str], timings: Sequence[str], sessions: Sequence[str]
) -> np.ndarray:
    """The index on ``sessions`` of the first close by which each release was public.

    A release before the open is public at the close of the first session on or after
    its report date; any other at the close of the first session after it. A release
    later than every session gives ``len(sessions)``.
    """
    axis = np.asarray(sessions, dtype=object)
    reports = np.asarray([str(d) for d in report_dates], dtype=object)
    on_or_after = np.searchsorted(axis, reports, side="left")
    after = np.searchsorted(axis, reports, side="right")
    before = np.asarray([t == "before" for t in timings], dtype=bool)
    return np.where(before, on_or_after, after).astype(np.int64)


def kdcodes_with_events(events: pd.DataFrame, kdcodes: Iterable[str]) -> list[str]:
    """The given panel codes that have at least one release."""
    have = set(events["kdcode"].astype(str))
    return [code for code in kdcodes if code in have]
