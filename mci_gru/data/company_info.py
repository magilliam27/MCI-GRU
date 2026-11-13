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


# --- SEC first-filed fundamentals -------------------------------------------------
#
# **Fundamentals.** One row per company, field and fiscal period, holding the value the
# company *first* filed for that period: SEC's companyfacts lists every value of every
# filing, including later filings' restated copies of old periods, and only the
# earliest-filed copy is what an investor could have read at the time. EODHD's
# fundamentals are not used for values, because they are restated yet stamped with the
# original filing date. A value is known from the close of the first session after
# its filing date. EDGAR dates a filing accepted after 17:30 Eastern to the next
# business day, so no filing is read before it was public; one accepted during the
# day is read a session late, which costs a slow signal nothing.

#: Each field and the us-gaap (or dei) concepts that carry it, best first. Within a
#: period the first concept with a value wins, so a company that moved from one tag to
#: another (ASC 606 revenue) keeps one series.
FUNDAMENTAL_CONCEPTS: dict[str, tuple[tuple[str, str], ...]] = {
    "equity": (
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ),
    "assets": (("us-gaap", "Assets"),),
    "net_income": (("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")),
    "gross_profit": (("us-gaap", "GrossProfit"),),
    "revenue": (
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ),
    "cost_of_revenue": (
        ("us-gaap", "CostOfRevenue"),
        ("us-gaap", "CostOfGoodsAndServicesSold"),
        ("us-gaap", "CostOfGoodsSold"),
    ),
    "shares": (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ),
}
#: Fields measured at an instant (a balance-sheet date); the rest are flows.
INSTANT_FIELDS = frozenset({"equity", "assets", "shares"})
#: Periodic reports whose values count. A registration statement's pro forma figures
#: are not a company's own reported results.
PERIODIC_FORMS = frozenset(
    {"10-K", "10-K/A", "10-KT", "10-KT/A", "10-Q", "10-Q/A", "10-QT", "20-F", "40-F"}
)
QUARTER_DAYS = (80, 100)
NINE_MONTH_DAYS = (260, 285)
YEAR_DAYS = (350, 380)
FACT_COLUMNS = ("field", "concept", "start", "end", "value", "accn", "form", "filed")


def first_filed_facts(companyfacts: Mapping) -> pd.DataFrame:
    """Every field's first-filed value per period, from one company's companyfacts JSON.

    A period is ``(start, end)`` for a flow and ``end`` alone for an instant. Of a
    concept's copies of one period, the one with the earliest ``filed`` date wins (ties
    to the lowest accession). Values in a unit other than USD (or shares, for
    ``shares``) and values from non-periodic forms are left out. Then, per field and
    period, the first concept in ``FUNDAMENTAL_CONCEPTS`` order with a value is kept.
    """
    facts = companyfacts.get("facts") or {}
    rows = []
    for field, concepts in FUNDAMENTAL_CONCEPTS.items():
        unit = "shares" if field == "shares" else "USD"
        for priority, (taxonomy, concept) in enumerate(concepts):
            entries = (((facts.get(taxonomy) or {}).get(concept) or {}).get("units") or {}).get(
                unit
            ) or []
            for item in entries:
                if item.get("form") not in PERIODIC_FORMS:
                    continue
                value = _number(item.get("val"))
                if not np.isfinite(value) or not item.get("end") or not item.get("filed"):
                    continue
                start = "" if field in INSTANT_FIELDS else str(item.get("start") or "")
                if field not in INSTANT_FIELDS and not start:
                    continue
                rows.append(
                    (
                        field,
                        concept,
                        priority,
                        start,
                        str(item["end"]),
                        value,
                        str(item.get("accn") or ""),
                        str(item["form"]),
                        str(item["filed"]),
                    )
                )
    columns = ["field", "concept", "priority", "start", "end", "value", "accn", "form", "filed"]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table.drop(columns="priority")
    table = table.sort_values(["field", "start", "end", "priority", "filed", "accn"], kind="stable")
    first_by_concept = table.drop_duplicates(["field", "concept", "start", "end"], keep="first")
    best = first_by_concept.drop_duplicates(["field", "start", "end"], keep="first")
    return best.drop(columns="priority").reset_index(drop=True)


def _days(start: pd.Series, end: pd.Series) -> pd.Series:
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.days


def quarterly_flows(facts: pd.DataFrame, field: str) -> pd.DataFrame:
    """A flow field's quarters: ``end, value, accn, filed``, one row per quarter end.

    A quarter is a reported three-month value or, for a fourth quarter companies seldom
    report alone, the year less the first nine months (both first-filed, same start);
    the derived quarter is known when the later of the two was filed. A reported
    quarter beats a derived one for the same end.
    """
    rows = facts.loc[facts["field"] == field].copy()
    columns = ["end", "value", "accn", "filed"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["days"] = _days(rows["start"], rows["end"])
    quarters = rows.loc[rows["days"].between(*QUARTER_DAYS), columns].assign(derived=False)
    years = rows.loc[rows["days"].between(*YEAR_DAYS)]
    nines = rows.loc[rows["days"].between(*NINE_MONTH_DAYS)]
    derived = years.merge(nines, on="start", suffixes=("", "_9m"))
    derived = derived.loc[derived["end_9m"] < derived["end"]]
    if not derived.empty:
        later = derived["filed_9m"] > derived["filed"]
        q4 = pd.DataFrame(
            {
                "end": derived["end"],
                "value": derived["value"] - derived["value_9m"],
                "accn": np.where(later, derived["accn_9m"], derived["accn"]),
                "filed": np.where(later, derived["filed_9m"], derived["filed"]),
                "derived": True,
            }
        )
        quarters = pd.concat([quarters, q4], ignore_index=True)
    quarters = quarters.sort_values(["end", "derived", "filed"], kind="stable")
    quarters = quarters.drop_duplicates("end", keep="first")
    return quarters.loc[:, columns].reset_index(drop=True)


FUNDAMENTAL_ROW_COLUMNS = ("kdcode", "cik", "field", "end", "value", "filed")
#: The fields a fundamentals file carries: balance-sheet instants and quarterly flows.
ROW_FIELDS = ("equity", "assets", "shares_adj", "net_income_q", "gross_profit_q")
#: Four quarter ends one after another span about 273 days.
TTM_SPAN_DAYS = (250, 300)
YEAR_AGO_DAYS = (350, 380)


def split_factor_after(dates: Sequence[str], splits: pd.DataFrame | None) -> np.ndarray:
    """Per date, the product of the ratios of every split dated after it.

    ``splits`` holds ``dt, ratio`` (shares after per share before), the records the
    price panel was split-adjusted with, so a share count times this factor is on the
    panel's basis and times the panel's close is a market value.
    """
    days = np.asarray([str(d) for d in dates], dtype=object)
    factor = np.ones(len(days))
    if splits is None or splits.empty:
        return factor
    for split_dt, ratio in zip(splits["dt"].astype(str), splits["ratio"], strict=True):
        factor[days < split_dt] *= float(ratio)
    return factor


def fundamental_rows(
    facts: pd.DataFrame, kdcode: str, cik: str, splits: pd.DataFrame | None
) -> pd.DataFrame:
    """One company's fundamentals file rows (``FUNDAMENTAL_ROW_COLUMNS``).

    ``equity`` and ``assets`` are first-filed instants; ``shares_adj`` is the share
    count on the panel's split basis; ``net_income_q`` and ``gross_profit_q`` are
    quarters (``quarterly_flows``). Gross profit is the reported figure, or else
    revenue less cost of revenue for a quarter that has both, known when the later of
    the two was filed.
    """
    parts = []
    for field in ("equity", "assets"):
        rows = facts.loc[facts["field"] == field, ["end", "value", "filed"]]
        parts.append(rows.assign(field=field))
    shares = facts.loc[facts["field"] == "shares", ["end", "value", "filed"]].copy()
    if not shares.empty:
        shares["value"] = shares["value"] * split_factor_after(shares["end"].tolist(), splits)
        shares = shares.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    parts.append(shares.assign(field="shares_adj"))
    income = quarterly_flows(facts, "net_income")
    parts.append(income.loc[:, ["end", "value", "filed"]].assign(field="net_income_q"))
    gross = quarterly_flows(facts, "gross_profit")
    revenue = quarterly_flows(facts, "revenue")
    cost = quarterly_flows(facts, "cost_of_revenue")
    if not revenue.empty and not cost.empty:
        both = revenue.merge(cost, on="end", suffixes=("", "_cost"))
        built = pd.DataFrame(
            {
                "end": both["end"],
                "value": both["value"] - both["value_cost"],
                "filed": np.maximum(both["filed"], both["filed_cost"]),
            }
        )
        built = built.loc[~built["end"].isin(set(gross["end"]))]
        gross = pd.concat([gross.loc[:, ["end", "value", "filed"]], built], ignore_index=True)
    parts.append(gross.loc[:, ["end", "value", "filed"]].assign(field="gross_profit_q"))
    out = pd.concat(parts, ignore_index=True).assign(kdcode=kdcode, cik=cik)
    out = out.loc[:, list(FUNDAMENTAL_ROW_COLUMNS)]
    return out.sort_values(["field", "end", "filed"], kind="stable").reset_index(drop=True)


def check_fundamentals(rows: pd.DataFrame) -> pd.DataFrame:
    """A fundamentals file's rows checked and typed, or a ValueError."""
    missing = [c for c in FUNDAMENTAL_ROW_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"fundamentals lack columns {missing}")
    out = rows.loc[:, list(FUNDAMENTAL_ROW_COLUMNS)].copy()
    for column in ("kdcode", "cik", "field", "end", "filed"):
        out[column] = out[column].astype(str)
    unknown = sorted(set(out["field"]) - set(ROW_FIELDS))
    if unknown:
        raise ValueError(f"unknown fundamentals fields {unknown}")
    for column in ("end", "filed"):
        if pd.to_datetime(out[column], format="%Y-%m-%d", errors="coerce").isna().any():
            raise ValueError(f"a fundamentals {column} is not a YYYY-MM-DD date")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(np.float64)
    return out


def load_fundamentals(path: str) -> pd.DataFrame:
    """A preserved fundamentals CSV, checked."""
    return check_fundamentals(pd.read_csv(path, dtype={"cik": str, "kdcode": str}))


def filed_known_session(filed: Sequence[str], sessions: Sequence[str]) -> np.ndarray:
    """The first session strictly after each filing date: when the value is first read."""
    axis = np.asarray(sessions, dtype=object)
    return np.searchsorted(axis, np.asarray([str(d) for d in filed], dtype=object), side="right")


def _ttm(quarters: pd.DataFrame) -> float:
    """The sum of the four newest quarters, if their ends run one quarter apart."""
    if len(quarters) < 4:
        return np.nan
    newest = quarters.sort_values("end").tail(4)
    span = (pd.Timestamp(newest["end"].iloc[-1]) - pd.Timestamp(newest["end"].iloc[0])).days
    if not TTM_SPAN_DAYS[0] <= span <= TTM_SPAN_DAYS[1]:
        return np.nan
    return float(newest["value"].sum())


def _newest(rows: pd.DataFrame) -> tuple[float, str]:
    if rows.empty:
        return np.nan, ""
    last = rows.sort_values(["end", "filed"]).iloc[-1]
    return float(last["value"]), str(last["end"])


def fundamentals_timeline(rows: pd.DataFrame, sessions: Sequence[str]) -> pd.DataFrame:
    """One company's fundamentals as known at each session a new value arrived.

    Columns: ``session`` (index on ``sessions``), ``equity``, ``assets``,
    ``shares_adj``, ``net_income_ttm``, ``gross_profit_ttm``, ``asset_growth`` and
    ``newest_end`` (the latest period end known). Each row uses only values first read
    on or before its session (``filed_known_session``); a later restatement never
    enters because the file holds first-filed values only.
    """
    known = filed_known_session(rows["filed"].tolist(), sessions)
    rows = rows.assign(known=known)
    rows = rows.loc[rows["known"] < len(sessions)]
    out = []
    for session in sorted(set(rows["known"])):
        seen = rows.loc[rows["known"] <= session]
        by_field = {f: seen.loc[seen["field"] == f] for f in ROW_FIELDS}
        equity, _ = _newest(by_field["equity"])
        assets, assets_end = _newest(by_field["assets"])
        shares, _ = _newest(by_field["shares_adj"])
        growth = np.nan
        if assets_end:
            older = by_field["assets"]
            gap = (pd.Timestamp(assets_end) - pd.to_datetime(older["end"])).dt.days
            year_ago = older.loc[gap.between(*YEAR_AGO_DAYS)]
            if not year_ago.empty:
                base = float(year_ago.sort_values("end").iloc[-1]["value"])
                growth = assets / base - 1.0 if base > 0 else np.nan
        ends = [r["end"].max() for r in by_field.values() if not r.empty]
        out.append(
            {
                "session": int(session),
                "equity": equity,
                "assets": assets,
                "shares_adj": shares,
                "net_income_ttm": _ttm(by_field["net_income_q"]),
                "gross_profit_ttm": _ttm(by_field["gross_profit_q"]),
                "asset_growth": growth,
                "newest_end": max(ends) if ends else "",
            }
        )
    return pd.DataFrame(out)
