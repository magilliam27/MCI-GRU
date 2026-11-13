"""Candidates that read company information, and the control they are read against (#393).

The go/no-go review of 2026-11-12 set the bar: new information earns its place only if
adding it to 1-week reversal plus 12-1 momentum lifts the pooled IC with a t of about 2.
So the control is a candidate too, and each information arm has a twin that adds it to
the control. Every arm is unfitted: equal weights on ranks, fixed before any test year
was scored, so there is nothing to tune on the test years.

* ``reversal_momentum``: the mean of the centred ranks of 1-week reversal and 12-1
  momentum (``mci_gru.evaluation.baselines``).
* ``earnings_surprise``: the mean of the centred ranks of the latest release's EPS
  surprise and its announcement return, while that release is at most
  ``MAX_EVENT_AGE`` sessions old.
* ``reversal_momentum_earnings``: the mean of the three centred ranks: reversal,
  momentum and ``earnings_surprise``.

**What an earnings score on date ``D`` reads.** The latest release known by the close of
``D`` (``mci_gru.data.company_info.known_session``), call its session ``k``:

* surprise: reported minus consensus EPS, over the close of ``k-2``, the last close
  before any release time the rule allows;
* announcement return: the stock's return from the close of ``k-2`` to the close of
  ``k``, minus the mean return of every panel name over the same two sessions. Two
  sessions, so a release whose time is not recorded still has its reaction inside.

A name with no release in the last ``MAX_EVENT_AGE`` sessions gets the middle rank.

**Where the releases come from.** The preserved events CSV named by the environment
variable ``MCI_GRU_EARNINGS_EVENTS`` (``scripts/data/export_company_info.py`` writes
it). Its sha256 goes into every result's controls.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mci_gru.data.company_info import known_session, load_earnings
from mci_gru.data.preprocessing import label_session_axis
from mci_gru.evaluation.baselines import momentum_scores, reversal_scores
from mci_gru.evaluation.candidates import CandidateContext, CandidateResult
from mci_gru.evaluation.residual_linear import centred_ranks

if TYPE_CHECKING:
    import pandas as pd

EARNINGS_EVENTS_ENV = "MCI_GRU_EARNINGS_EVENTS"
#: A release older than this many sessions (about a quarter) no longer scores.
MAX_EVENT_AGE = 63
#: The announcement window, in sessions, ending at the session the release is known.
ANNOUNCEMENT_SESSIONS = 2
REVERSAL_LOOKBACK = 5


@dataclass(frozen=True)
class EarningsSignals:
    """The latest known release's signals on each test date, for each stock."""

    surprise: np.ndarray  # (test dates, stocks); NaN without a recent release
    announcement_return: np.ndarray
    age: np.ndarray  # sessions since the release was known; -1 without one
    events_used: int


def _closes(panel: pd.DataFrame, kdcodes: list[str]) -> tuple[list[str], np.ndarray]:
    rows = panel.loc[:, ["kdcode", "dt", "close"]].copy()
    rows["dt"] = rows["dt"].astype(str)
    rows["kdcode"] = rows["kdcode"].astype(str)
    sessions = label_session_axis(rows, list(kdcodes))
    table = rows.pivot_table(index="dt", columns="kdcode", values="close", aggfunc="mean")
    closes = table.reindex(index=sessions, columns=list(kdcodes)).to_numpy(dtype=np.float64)
    return sessions, closes


def earnings_signals(
    panel: pd.DataFrame,
    kdcodes: list[str],
    dates: list[str],
    events: pd.DataFrame,
    max_age: int = MAX_EVENT_AGE,
) -> EarningsSignals:
    """Surprise, announcement return and age of each stock's latest known release."""
    sessions, closes = _closes(panel, kdcodes)
    n_sessions = len(sessions)
    lag = ANNOUNCEMENT_SESSIONS
    with np.errstate(divide="ignore", invalid="ignore"):
        window = np.full_like(closes, np.nan)
        window[lag:] = closes[lag:] / closes[:-lag] - 1.0
    window[~np.isfinite(window)] = np.nan
    counts = np.isfinite(window).sum(axis=1)
    market = np.where(counts > 0, np.nansum(window, axis=1) / np.maximum(counts, 1), np.nan)

    position = {session: i for i, session in enumerate(sessions)}
    rows = np.asarray([position.get(str(d), -1) for d in dates], dtype=np.int64)
    shape = (len(dates), len(kdcodes))
    surprise = np.full(shape, np.nan)
    announcement = np.full(shape, np.nan)
    age = np.full(shape, -1, dtype=np.int64)
    column = {code: j for j, code in enumerate(kdcodes)}
    used = 0
    for kdcode, group in events.groupby("kdcode", sort=False):
        j = column.get(str(kdcode))
        if j is None:
            continue
        known = known_session(group["report_date"], group["timing"], sessions)
        keep = (known >= lag) & (known < n_sessions)
        if not keep.any():
            continue
        known = known[keep]
        order = np.argsort(known, kind="stable")
        known = known[order]
        actual = group["eps_actual"].to_numpy(dtype=np.float64)[keep][order]
        estimate = group["eps_estimate"].to_numpy(dtype=np.float64)[keep][order]
        base = closes[known - lag, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            event_surprise = np.where(base > 0, (actual - estimate) / base, np.nan)
        event_return = window[known, j] - market[known]
        used += len(known)
        # Two releases known on one session (a correction): the later row wins.
        latest = np.searchsorted(known, rows, side="right") - 1
        for i, (row, which) in enumerate(zip(rows, latest, strict=True)):
            if row < 0 or which < 0:
                continue
            sessions_since = int(row - known[which])
            if sessions_since > max_age:
                continue
            age[i, j] = sessions_since
            surprise[i, j] = event_surprise[which]
            announcement[i, j] = event_return[which]
    surprise[~np.isfinite(surprise)] = np.nan
    announcement[~np.isfinite(announcement)] = np.nan
    return EarningsSignals(surprise, announcement, age, used)


def _scorable(context: CandidateContext, *arrays: np.ndarray) -> np.ndarray:
    if context.rankable_mask is not None:
        return np.asarray(context.rankable_mask, dtype=bool)
    keep = np.zeros(arrays[0].shape, dtype=bool)
    for values in arrays:
        keep |= np.isfinite(values)
    return keep


def _mean_rank(parts: list[np.ndarray], scorable: np.ndarray) -> np.ndarray:
    ranked = [centred_ranks(part, scorable) for part in parts]
    return np.where(scorable, np.mean(ranked, axis=0), np.nan)


def _events_path() -> str:
    path = os.environ.get(EARNINGS_EVENTS_ENV)
    if not path:
        raise ValueError(f"set {EARNINGS_EVENTS_ENV} to the preserved earnings-events CSV")
    return path


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _earnings_part(context: CandidateContext) -> tuple[np.ndarray, np.ndarray, dict]:
    path = _events_path()
    events = load_earnings(path)
    signals = earnings_signals(context.panel, context.kdcodes, context.test_dates, events)
    scorable = _scorable(context, signals.surprise)
    score = _mean_rank([signals.surprise, signals.announcement_return], scorable)
    covered = scorable & (signals.age >= 0)
    controls = {
        "earnings_events_csv": os.path.basename(path),
        "earnings_events_sha256": _sha256(path),
        "earnings_events_used": signals.events_used,
        "max_event_age_sessions": MAX_EVENT_AGE,
        "announcement_sessions": ANNOUNCEMENT_SESSIONS,
        "share_scorable_with_recent_release": float(covered.sum() / max(scorable.sum(), 1)),
    }
    return score, scorable, controls


def _control_parts(context: CandidateContext) -> list[np.ndarray]:
    return [
        reversal_scores(context.panel, context.kdcodes, context.test_dates, REVERSAL_LOOKBACK),
        momentum_scores(context.panel, context.kdcodes, context.test_dates),
    ]


def reversal_momentum(context: CandidateContext) -> CandidateResult:
    """Mean centred rank of 1-week reversal and 12-1 momentum."""
    parts = _control_parts(context)
    scorable = _scorable(context, *parts)
    return CandidateResult(
        _mean_rank(parts, scorable),
        {"definition": "mean centred rank of reversal_1w and momentum_12_1"},
    )


def earnings_surprise(context: CandidateContext) -> CandidateResult:
    """Mean centred rank of the latest release's EPS surprise and announcement return."""
    score, scorable, controls = _earnings_part(context)
    return CandidateResult(
        np.where(scorable, score, np.nan),
        {"definition": "mean centred rank of EPS surprise and announcement return", **controls},
    )


def reversal_momentum_earnings(context: CandidateContext) -> CandidateResult:
    """Mean centred rank of reversal, momentum and ``earnings_surprise``."""
    earnings, scorable, controls = _earnings_part(context)
    parts = [*_control_parts(context), earnings]
    return CandidateResult(
        _mean_rank(parts, scorable),
        {
            "definition": "mean centred rank of reversal_1w, momentum_12_1, earnings_surprise",
            **controls,
        },
    )
