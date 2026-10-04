"""
Historical availability rules for the regime input roles (#224).

These are the owner rulings of 2026-10-03, written per role rather than per
provider series so a later source for the same role inherits them. They are
declared conventions, not measured publication latency:

- Prediction clock: the forecast for session t is made at 20:00 America/New_York on t.
- A daily value dated D is known from 20:00 New York on the next session after D,
  so session t sees only values dated before t.
- A monthly value for month M is known from the first session of month M+2.
- A daily value may carry forward at most ``MAX_CARRY_SESSIONS`` sessions, and a
  monthly value only through its own month M+2. A longer gap stops the run.
- The source's missing marker or a blank is a genuine gap. Any other
  non-numeric token is malformed and stops the run. Positive roles must be > 0;
  the others may take any finite value.
- No back-fill: sessions before a role's first usable value stay empty and are
  counted. Revisions are unchecked: values as served at capture stand in for history.

A session is a weekday. On any exchange trading date this gives the same
availability as the exchange calendar, because a value dated D is usable at
session t exactly when D < t; an exchange holiday counts as one session of carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.data.quality_contract import AdmissionItem, Verdict

PREDICTION_CLOCK = "20:00 America/New_York"
SESSION_CALENDAR = "weekdays"
MAX_CARRY_SESSIONS = 5
MONTHLY_RELEASE_LAG_MONTHS = 2
VALID_FOR_STATED_SCOPE = "valid for stated scope"
ADMISSION_RULE = "regime_historical_availability"
_MAX_REPORTED_DATES = 10


@dataclass(frozen=True)
class RegimeRole:
    """One regime input role and its value rules, independent of the source."""

    column: str
    label: str
    cadence: str
    positive: bool


REGIME_INPUT_ROLES: dict[str, RegimeRole] = {
    role.column: role
    for role in (
        RegimeRole("regime_market", "market level", "daily", positive=True),
        RegimeRole("yield_10y", "10-year yield", "daily", positive=False),
        RegimeRole("yield_3m", "3-month yield", "daily", positive=False),
        RegimeRole("regime_oil", "oil", "daily", positive=False),
        RegimeRole("regime_copper", "copper", "monthly", positive=True),
        RegimeRole("regime_volatility", "volatility", "daily", positive=True),
    )
}


class RegimeInputError(ValueError):
    """A regime input broke a ruled admission rule; ``facts`` name the role and dates."""

    def __init__(self, role: RegimeRole, reason: str, message: str, **details: Any) -> None:
        self.facts = {
            "reason": reason,
            "reason_code": reason,
            "regime_role": role.column,
            "regime_role_label": role.label,
            **details,
        }
        super().__init__(f"Regime input {role.label} ({role.column}): {message}")


@dataclass(frozen=True)
class QualifiedRole:
    """A role's values on the session grid and its verdict for the run report."""

    values: pd.Series
    verdict: dict[str, Any]


def session_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Sessions from start to end inclusive."""
    return pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())


def _dates(index: pd.Index) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in index[:_MAX_REPORTED_DATES]]


def parse_observations(
    raw: pd.Series, role: RegimeRole, missing_markers: tuple[str, ...]
) -> tuple[pd.Series, int]:
    """
    Validate original observations and return (values, gap count).

    Values are float on a sorted date index with genuine gaps as NaN. Raises
    RegimeInputError for malformed tokens, non-finite or non-positive values,
    and unparseable or repeated observation dates.
    """
    try:
        index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    except (TypeError, ValueError) as error:
        raise RegimeInputError(
            role, "malformed_observation_date", f"unparseable observation dates ({error})"
        ) from None
    values = pd.Series(raw.to_numpy(dtype=object), index=index).sort_index(kind="stable")
    repeated = values.index[values.index.duplicated()]
    if len(repeated):
        raise RegimeInputError(
            role,
            "duplicate_observation_date",
            f"repeated observation dates {_dates(repeated)}",
            dates=_dates(repeated),
            count=len(repeated),
        )

    text = values.map(lambda value: value.strip() if isinstance(value, str) else value)
    gap = text.isna() | text.map(
        lambda value: isinstance(value, str) and value in ("", *missing_markers)
    )
    numeric = pd.to_numeric(text.where(~gap), errors="coerce").astype(float)
    for reason, bad, rule in (
        ("malformed_value", numeric.isna() & ~gap, "non-numeric tokens"),
        ("non_finite_value", np.isinf(numeric), "non-finite values"),
        (
            "non_positive_value",
            (numeric <= 0) if role.positive else None,
            "values that are not positive",
        ),
    ):
        if bad is None or not bad.any():
            continue
        dates = _dates(numeric.index[bad])
        raise RegimeInputError(
            role, reason, f"{rule} on {dates}", dates=dates, count=int(bad.sum())
        )
    return numeric, int(gap.sum())


def _carry_violation(
    role: RegimeRole, sessions: pd.DatetimeIndex, over: np.ndarray, last_seen: pd.Index, limit: str
) -> RegimeInputError:
    first = int(np.argmax(over))
    return RegimeInputError(
        role,
        "carry_forward_limit_exceeded",
        f"no value since {last_seen[first].strftime('%Y-%m-%d')} by session "
        f"{sessions[first].strftime('%Y-%m-%d')} ({limit})",
        last_observation=last_seen[first].strftime("%Y-%m-%d"),
        first_session_over_limit=sessions[first].strftime("%Y-%m-%d"),
        sessions_over_limit=_dates(sessions[over]),
        count=int(over.sum()),
    )


def _align_daily(
    observed: pd.Series, role: RegimeRole, sessions: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """As-of values and carry ages: session t uses the last value dated before t."""
    dates = observed.index.to_numpy()
    calendar = session_calendar(min(observed.index[0], sessions[0]), sessions[-1])
    position = np.searchsorted(dates, sessions.to_numpy(), side="left") - 1
    known = position >= 0
    position = position.clip(min=0)
    available_from = np.searchsorted(calendar.to_numpy(), dates[position], side="right")
    age = np.searchsorted(calendar.to_numpy(), sessions.to_numpy()) - available_from
    over = known & (age > MAX_CARRY_SESSIONS)
    if over.any():
        raise _carry_violation(
            role,
            sessions,
            over,
            observed.index[position],
            f"carry is limited to {MAX_CARRY_SESSIONS} sessions",
        )
    values = np.where(known, observed.to_numpy()[position], np.nan)
    return values, np.where(known, age, 0)


def _align_monthly(observed: pd.Series, role: RegimeRole, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Session t in month X uses the value for month X-2, and nothing older."""
    monthly = observed.groupby(observed.index.to_period("M")).last()
    months = monthly.index.year.to_numpy() * 12 + monthly.index.month.to_numpy()
    target = sessions.year.to_numpy() * 12 + sessions.month.to_numpy() - MONTHLY_RELEASE_LAG_MONTHS
    position = np.searchsorted(months, target, side="right") - 1
    known = position >= 0
    position = position.clip(min=0)
    over = known & (months[position] < target)
    if over.any():
        raise _carry_violation(
            role,
            sessions,
            over,
            monthly.index[position].to_timestamp(),
            f"month M is used only during month M+{MONTHLY_RELEASE_LAG_MONTHS}",
        )
    return np.where(known, monthly.to_numpy()[position], np.nan)


def qualify_role(
    raw: pd.Series,
    role: RegimeRole,
    sessions: pd.DatetimeIndex,
    *,
    source: str,
    series_id: str,
    missing_markers: tuple[str, ...],
) -> QualifiedRole:
    """Apply the ruled value, availability, carry and leading-gap rules to one role."""
    numeric, gap_count = parse_observations(raw, role, missing_markers)
    observed = numeric.dropna()
    if observed.empty or observed.index[0] >= sessions[-1]:
        raise RegimeInputError(
            role,
            "no_usable_value",
            f"no observation dated before the last session {sessions[-1].strftime('%Y-%m-%d')}",
        )
    if role.cadence == "daily":
        values, ages = _align_daily(observed, role, sessions)
        max_carry: int | None = int(ages.max())
    else:
        values = _align_monthly(observed, role, sessions)
        max_carry = None
    usable = ~np.isnan(values)
    if not usable.any():
        raise RegimeInputError(
            role,
            "no_usable_value",
            f"no value becomes usable by the last session {sessions[-1].strftime('%Y-%m-%d')}",
        )
    first_usable = int(np.argmax(usable))
    verdict = {
        "role": role.column,
        "label": role.label,
        "source": source,
        "series_id": series_id,
        "verdict": VALID_FOR_STATED_SCOPE,
        "scope": {
            "first_session": sessions[0].strftime("%Y-%m-%d"),
            "last_session": sessions[-1].strftime("%Y-%m-%d"),
            "prediction_clock": PREDICTION_CLOCK,
            "session_calendar": SESSION_CALENDAR,
        },
        "rules": {
            "availability": "next_session"
            if role.cadence == "daily"
            else f"first_session_of_month_plus_{MONTHLY_RELEASE_LAG_MONTHS}",
            "max_carry_sessions": MAX_CARRY_SESSIONS if role.cadence == "daily" else None,
            "missing_markers": ["", *missing_markers],
            "positive": role.positive,
            "back_fill": False,
        },
        "first_observation": observed.index[0].strftime("%Y-%m-%d"),
        "last_observation": observed.index[-1].strftime("%Y-%m-%d"),
        "first_usable_session": sessions[first_usable].strftime("%Y-%m-%d"),
        "leading_gap_sessions": first_usable,
        "missing_observations": gap_count,
        "max_carry_sessions_seen": max_carry,
        "revisions": "unchecked",
    }
    return QualifiedRole(pd.Series(values, index=sessions, name=role.column), verdict)


def admission_item(verdict: dict[str, Any]) -> AdmissionItem:
    """A qualified role's verdict as one #223 ledger item (``valid`` for its stated scope)."""
    return AdmissionItem(
        role=f"{verdict['source']}.{verdict['role']}",
        rule=ADMISSION_RULE,
        verdict=Verdict.VALID,
        reason_code="valid_for_stated_scope",
        reason=f"{verdict['label']} meets the #224 availability rules for the stated scope",
        source=verdict["source"],
        evidence=verdict,
    )


def rejection_item(facts: dict[str, Any]) -> AdmissionItem:
    """A role stopped by a #224 rule, from the stop's own facts, as an ``invalid`` ledger item."""
    return AdmissionItem(
        role=facts["role"],
        rule=ADMISSION_RULE,
        verdict=Verdict.INVALID,
        reason_code=facts["reason_code"],
        reason=facts.get("detail", facts["reason_code"]),
        stage=facts["stage"],
        source=facts["source"],
        evidence={
            key: value
            for key, value in facts.items()
            if key not in {"role", "source", "stage", "reason", "reason_code", "detail"}
        },
    )
