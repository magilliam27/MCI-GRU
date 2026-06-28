from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import pandas as pd


def _active_pit_members(pit_universe: pd.DataFrame, date: pd.Timestamp) -> set[str]:
    active = pit_universe[(pit_universe["valid_from"] <= date) & (pit_universe["valid_to"] >= date)]
    return set(active["kdcode"])


def _count_stale_members(
    market_data: pd.DataFrame,
    *,
    active_codes: set[str],
    observed_codes: set[str],
    date: pd.Timestamp,
    stale_after_days: int,
) -> int:
    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative")

    stale_codes = active_codes - observed_codes
    if not stale_codes:
        return 0

    prior_observations = market_data[
        (market_data["dt"] < date) & (market_data["kdcode"].isin(stale_codes))
    ]
    if prior_observations.empty:
        return 0

    last_seen = prior_observations.groupby("kdcode")["dt"].max()
    age_days = (date - last_seen).dt.days
    return int((age_days >= stale_after_days).sum())


def build_pit_availability_report(
    market_data: pd.DataFrame,
    pit_universe: pd.DataFrame,
    *,
    min_price: float,
    min_dollar_volume: float,
    stale_after_days: int,
    calendar: Iterable[str | pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """Build a report-only PIT availability/tradability summary.

    This helper intentionally works on copies and returns metadata only. It does
    not filter masked-panel tensors, PIT masks, or any training inputs.
    """

    market = market_data.copy()
    pit = pit_universe.copy()
    market["dt"] = pd.to_datetime(market["dt"])
    pit["valid_from"] = pd.to_datetime(pit["valid_from"])
    pit["valid_to"] = pd.to_datetime(pit["valid_to"])
    market["dollar_volume"] = market["close"] * market["volume"]
    report_dates = _report_dates(market, calendar)

    dates: list[dict[str, Any]] = []
    for date in report_dates:
        day = market[market["dt"] == date]
        active_codes = _active_pit_members(pit, date)
        day_active = day[day["kdcode"].isin(active_codes)].copy()
        observed_codes = set(day_active["kdcode"])
        tradable = day_active[
            (day_active["open"] >= min_price)
            & (day_active["close"] >= min_price)
            & (day_active["volume"] > 0)
            & (day_active["dollar_volume"] >= min_dollar_volume)
        ]

        dates.append(
            {
                "dt": date.strftime("%Y-%m-%d"),
                "active_members": int(len(active_codes)),
                "observed_members": int(day_active["kdcode"].nunique()),
                "missing_members": int(len(active_codes - observed_codes)),
                "zero_volume_count": int((day_active["volume"] <= 0).sum()),
                "stale_count": _count_stale_members(
                    market,
                    active_codes=active_codes,
                    observed_codes=observed_codes,
                    date=date,
                    stale_after_days=stale_after_days,
                ),
                "tradable_count": int(tradable["kdcode"].nunique()),
            }
        )

    return {
        "schema_version": 1,
        "pit_union_kdcodes": int(pit["kdcode"].nunique()),
        "policy": {
            "masked_panel_preserved": True,
            "report_only": True,
            "calendar_scope": "explicit" if calendar is not None else "market_observed",
            "min_price": float(min_price),
            "min_dollar_volume": float(min_dollar_volume),
            "stale_after_days": int(stale_after_days),
        },
        "dates": dates,
    }


def _report_dates(
    market: pd.DataFrame, calendar: Iterable[str | pd.Timestamp] | None
) -> list[pd.Timestamp]:
    if calendar is None:
        return [pd.Timestamp(date) for date in sorted(market["dt"].dropna().unique())]
    parsed = pd.to_datetime(list(calendar))
    return [pd.Timestamp(date) for date in sorted(parsed.dropna().unique())]
