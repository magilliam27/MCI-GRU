"""Sector tone from rated 8-Ks, and the sector ETF rotation test (E20, #394).

What this module decides:

- **Which sector a filing belongs to.** The filer's SIC code as recorded in that
  filing's own SEC header (``ASSIGNED-SIC``), so the code is the one the SEC had
  at acceptance, not today's. :func:`sic_to_sector` maps it to one of eleven
  GICS-style sectors, each traded by a Select Sector SPDR; :func:`sic_to_industry`
  maps a narrower subset to nine industry ETFs. Both maps are fixed hand maps,
  written before any price was looked at. A code neither map covers returns
  ``None`` and the filing is left out.
- **What is known before an entry.** A window ends at 09:30 US Eastern on the
  entry day and is half open, ``[start, end)``: a filing accepted at exactly
  09:30 belongs to the next window. Nothing accepted at or after the entry
  instant can reach that entry's signal (:func:`assign_windows`).
- **What "abnormal" tone means.** A sector's mean score in the window minus the
  mean of the same sector's filings accepted in the 52 weeks before the window
  starts. The history ends where the window starts, so the baseline never sees
  the window or anything later (:func:`abnormal_tone`).
- **How a period is scored.** The Spearman rank correlation across the eligible
  assets between signal and later return (:func:`cross_sectional_ic`), averaged
  over periods with a moving-block bootstrap interval
  (:func:`block_bootstrap_interval`), since neighbouring weeks are not
  independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

SECTOR_ETFS: dict[str, str] = {
    "energy": "XLE",
    "materials": "XLB",
    "industrials": "XLI",
    "discretionary": "XLY",
    "staples": "XLP",
    "health": "XLV",
    "financials": "XLF",
    "technology": "XLK",
    "communication": "XLC",
    "utilities": "XLU",
    "real_estate": "XLRE",
}

# (low, high, sector), inclusive. The first matching range wins, so narrow
# exceptions come before the broad range they carve out of.
_SECTOR_RANGES: tuple[tuple[int, int, str], ...] = (
    # Exceptions first.
    (1220, 1241, "energy"),  # coal
    (1531, 1531, "discretionary"),  # operative builders (homebuilders)
    (2833, 2836, "health"),  # pharmaceuticals, biologicals
    (2840, 2844, "staples"),  # soap, detergents, cosmetics
    (3080, 3089, "materials"),  # plastics products
    (3410, 3412, "materials"),  # metal cans
    (3570, 3579, "technology"),  # computers and office equipment
    (3630, 3639, "discretionary"),  # household appliances
    (3650, 3652, "discretionary"),  # household audio and video
    (3660, 3679, "technology"),  # communications equipment, electronics, semis
    (3710, 3714, "discretionary"),  # motor vehicles and parts
    (3751, 3751, "discretionary"),  # motorcycles, bicycles
    (3826, 3826, "health"),  # laboratory analytical instruments
    (3820, 3829, "technology"),  # measuring and controlling instruments
    (3840, 3851, "health"),  # medical instruments and supplies
    (3860, 3861, "technology"),  # photographic equipment
    (3870, 3873, "discretionary"),  # watches
    (4610, 4619, "energy"),  # crude and refined pipelines
    (4922, 4922, "energy"),  # natural gas transmission
    (4950, 4959, "industrials"),  # sanitary services, waste
    (5045, 5045, "technology"),  # computer wholesale
    (5047, 5047, "health"),  # medical wholesale
    (5122, 5122, "health"),  # drugs wholesale
    (5140, 5149, "staples"),  # grocery wholesale
    (5171, 5172, "energy"),  # petroleum wholesale
    (5180, 5182, "staples"),  # beer, wine, spirits wholesale
    (5331, 5331, "staples"),  # variety stores (hypermarkets)
    (5400, 5499, "staples"),  # food stores
    (5912, 5912, "staples"),  # drug stores
    (6324, 6324, "health"),  # hospital and medical service plans
    (6500, 6599, "real_estate"),
    (6798, 6798, "real_estate"),  # REITs
    (7310, 7319, "communication"),  # advertising
    (7370, 7379, "technology"),  # computer programming, software, data
    (7812, 7899, "communication"),  # motion pictures, video
    (8731, 8731, "health"),  # commercial biological research
    # Broad ranges.
    (100, 999, "staples"),  # agriculture
    (1000, 1099, "materials"),  # metal mining
    (1300, 1399, "energy"),  # oil and gas extraction, services
    (1400, 1499, "materials"),  # nonmetallic minerals
    (1500, 1799, "industrials"),  # construction
    (2000, 2199, "staples"),  # food, tobacco
    (2200, 2399, "discretionary"),  # textiles, apparel
    (2400, 2499, "materials"),  # lumber, wood
    (2500, 2599, "discretionary"),  # furniture
    (2600, 2699, "materials"),  # paper, containers
    (2700, 2799, "communication"),  # printing, publishing
    (2800, 2899, "materials"),  # chemicals
    (2900, 2999, "energy"),  # petroleum refining
    (3000, 3199, "discretionary"),  # rubber, leather, footwear
    (3200, 3399, "materials"),  # stone, clay, glass, primary metals
    (3400, 3799, "industrials"),  # fabricated metal, machinery, transport eq.
    (3800, 3899, "industrials"),  # remaining instruments (e.g. 3812 defense)
    (3900, 3999, "discretionary"),  # misc. manufacturing (toys, jewelry)
    (4000, 4799, "industrials"),  # transportation
    (4800, 4899, "communication"),  # telephone, cable, broadcasting
    (4900, 4999, "utilities"),
    (5000, 5199, "industrials"),  # wholesale
    (5200, 5999, "discretionary"),  # retail
    (6000, 6799, "financials"),
    (7000, 7299, "discretionary"),  # hotels, personal services
    (7300, 7399, "industrials"),  # business services
    (7500, 7599, "discretionary"),  # auto repair, rental
    (7900, 7999, "discretionary"),  # amusement, recreation
    (8000, 8099, "health"),  # health services
    (8200, 8299, "discretionary"),  # education
    (8700, 8799, "industrials"),  # engineering, management services
)

INDUSTRY_ETFS: tuple[str, ...] = ("XBI", "KRE", "XOP", "SMH", "ITB", "XRT", "XME", "KIE", "IYT")

# (low, high, etf, excluded (low, high) ranges)
_INDUSTRY_RANGES: tuple[tuple[int, int, str, tuple[tuple[int, int], ...]], ...] = (
    (2833, 2836, "XBI", ()),
    (8731, 8731, "XBI", ()),
    (6020, 6036, "KRE", ()),
    (1310, 1389, "XOP", ()),
    (3674, 3674, "SMH", ()),
    (1520, 1531, "ITB", ()),
    (5200, 5999, "XRT", ((5400, 5499), (5812, 5812))),
    (1000, 1099, "XME", ()),
    (1200, 1241, "XME", ()),
    (3310, 3399, "XME", ()),
    (6310, 6411, "KIE", ((6324, 6324),)),
    (4000, 4581, "IYT", ()),
)


def _as_int(sic: object) -> int | None:
    if sic is None:
        return None
    try:
        if isinstance(sic, float) and np.isnan(sic):
            return None
        return int(str(sic).strip().split(".")[0])
    except ValueError:
        return None


def sic_to_sector(sic: object) -> str | None:
    """Map a SIC code to a sector key of :data:`SECTOR_ETFS`, or ``None``."""
    code = _as_int(sic)
    if code is None:
        return None
    for low, high, sector in _SECTOR_RANGES:
        if low <= code <= high:
            return sector
    return None


def sic_to_industry(sic: object) -> str | None:
    """Map a SIC code to one of :data:`INDUSTRY_ETFS`, or ``None``."""
    code = _as_int(sic)
    if code is None:
        return None
    for low, high, etf, excluded in _INDUSTRY_RANGES:
        if low <= code <= high and not any(a <= code <= b for a, b in excluded):
            return etf
    return None


def parse_header_sic(header: str, cik: str | int) -> str:
    """Return the ``ASSIGNED-SIC`` of ``cik``'s block in an SEC ``.hdr.sgml``.

    A header can list several companies (filer, subject company, co-registrants);
    only the block whose ``<CIK>`` equals ``cik`` counts. Returns ``""`` when the
    CIK is absent or its block carries no SIC.
    """
    want = int(cik)
    for block in header.split("<COMPANY-DATA>")[1:]:
        block = block.split("</COMPANY-DATA>")[0]
        cik_value = sic_value = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("<CIK>"):
                cik_value = int(line[5:].strip())
            elif line.startswith("<ASSIGNED-SIC>"):
                sic_value = line[len("<ASSIGNED-SIC>") :].strip()
        if cik_value == want:
            return sic_value or ""
    return ""


def entry_instants(sessions: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """09:30 US Eastern on each session date, as UTC instants."""
    days = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize()
    local = (days + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York")
    return local.tz_convert("UTC")


def weekly_entry_days(sessions: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading session of each calendar (ISO) week."""
    days = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize().sort_values()
    iso = days.isocalendar()
    key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    first = ~pd.Series(key.values).duplicated().values
    return days[first]


def assign_windows(accepted_utc: pd.Series, instants: pd.DatetimeIndex) -> np.ndarray:
    """Index of the entry whose window ``[instants[i-1], instants[i])`` holds each time.

    Returns ``-1`` for a time before the first instant's window start or at/after
    the last instant (no entry can use it).
    """
    times = pd.to_datetime(accepted_utc, utc=True).values.astype("datetime64[ns]")
    edges = instants.tz_convert("UTC").tz_localize(None).values.astype("datetime64[ns]")
    idx = np.searchsorted(edges, times, side="right")
    idx = np.where((idx >= 1) & (idx < len(edges)), idx, -1)
    return idx


def window_tone(
    events: pd.DataFrame,
    instants: pd.DatetimeIndex,
    group_col: str,
    value_col: str,
    lookback: int = 1,
) -> pd.DataFrame:
    """Per entry and group: events, mean, sum and bad share over the window.

    ``lookback`` counts entries: window ``k`` covers ``[instants[k-lookback],
    instants[k])``. ``events`` needs ``accepted_utc``, ``group_col``,
    ``value_col`` and optionally ``bad`` (1 when Jev's most likely answer is bad
    or very bad).
    """
    ev = events.dropna(subset=[group_col, value_col]).copy()
    ev["k"] = assign_windows(ev["accepted_utc"], instants)
    ev = ev[ev["k"] >= 0]
    if "bad" not in ev:
        ev["bad"] = np.nan
    bucket = ev.groupby(["k", group_col]).agg(
        s=(value_col, "sum"), n=(value_col, "size"), b=("bad", "sum")
    )
    groups = sorted(ev[group_col].unique())
    full = pd.MultiIndex.from_product([range(len(instants)), groups], names=["k", group_col])
    bucket = bucket.reindex(full, fill_value=0.0)
    rolled = (
        bucket.groupby(level=group_col)
        .rolling(lookback, min_periods=lookback)
        .sum()
        .droplevel(0)
        .sort_index()
    )
    out = rolled.reset_index()
    out = out[out["k"] >= lookback]
    out["entry_utc"] = instants[out["k"].to_numpy()]
    out["window_start_utc"] = instants[(out["k"] - lookback).to_numpy()]
    out["mean"] = np.where(out["n"] > 0, out["s"] / out["n"].where(out["n"] > 0, 1), np.nan)
    out["bad_share"] = np.where(out["n"] > 0, out["b"] / out["n"].where(out["n"] > 0, 1), np.nan)
    return out.rename(columns={"s": "sum"}).drop(columns="b").reset_index(drop=True)


def abnormal_tone(
    tone: pd.DataFrame,
    events: pd.DataFrame,
    group_col: str,
    value_col: str,
    history_weeks: int = 52,
    min_history_weeks: int = 26,
) -> pd.Series:
    """Window mean minus the group's mean over the ``history_weeks`` before the window.

    History is every event of the group accepted in ``[window_start -
    history_weeks, window_start)``. NaN when the group's first event is less than
    ``min_history_weeks`` before the window start, or the history is empty.
    """
    ev = events.dropna(subset=[group_col, value_col])
    times = pd.to_datetime(ev["accepted_utc"], utc=True)
    result = pd.Series(np.nan, index=tone.index)
    span = pd.Timedelta(weeks=history_weeks)
    need = pd.Timedelta(weeks=min_history_weeks)
    for group, rows in tone.groupby(group_col):
        mask = (ev[group_col] == group).to_numpy()
        t = times[mask].sort_values()
        v = ev.loc[mask, value_col].reindex(t.index).to_numpy(dtype=float)
        tv = t.values.astype("datetime64[ns]")
        csum = np.concatenate([[0.0], np.cumsum(v)])
        starts = pd.to_datetime(rows["window_start_utc"], utc=True)
        lo = np.searchsorted(tv, (starts - span).values.astype("datetime64[ns]"), side="left")
        hi = np.searchsorted(tv, starts.values.astype("datetime64[ns]"), side="left")
        count = hi - lo
        hist_mean = np.where(count > 0, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)
        if len(tv) == 0:
            continue
        ok = (starts - need).values.astype("datetime64[ns]") >= tv[0]
        result.loc[rows.index] = np.where(ok, rows["mean"].to_numpy() - hist_mean, np.nan)
    return result


def cross_sectional_ic(
    panel: pd.DataFrame,
    signal: str,
    outcome: str,
    period: str = "entry_utc",
    min_assets: int = 5,
) -> pd.Series:
    """Spearman correlation across assets per period; periods with too few assets drop."""
    out = {}
    for key, rows in panel.dropna(subset=[signal, outcome]).groupby(period):
        if len(rows) < min_assets:
            continue
        a = rows[signal].rank().to_numpy()
        b = rows[outcome].rank().to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        out[key] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out, dtype=float).sort_index()


def rank_regression(
    panel: pd.DataFrame,
    outcome: str,
    signals: Sequence[str],
    period: str = "entry_utc",
    min_assets: int = 5,
) -> pd.DataFrame:
    """Per period, OLS of the outcome's ranks on the signals' ranks (all scaled to [-1, 1])."""
    rows_out = {}
    cols = list(signals)
    for key, rows in panel.dropna(subset=[outcome, *cols]).groupby(period):
        n = len(rows)
        if n < max(min_assets, len(cols) + 2):
            continue

        def scaled(x: pd.Series, half: float = max((n - 1) / 2, 1)) -> np.ndarray:
            r = x.rank().to_numpy()
            return (r - r.mean()) / half

        X = np.column_stack([np.ones(n)] + [scaled(rows[c]) for c in cols])
        y = scaled(rows[outcome])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows_out[key] = coef[1:]
    return pd.DataFrame.from_dict(rows_out, orient="index", columns=cols).sort_index()


def block_bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    block: int,
    draws: int = 10_000,
    seed: int = 394,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and moving-block bootstrap interval of a series kept in time order."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(draws, -1)[:, :n]
    means = x[idx].mean(axis=1)
    tail = (1 - level) / 2
    return (float(x.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)))


def top_minus_bottom(
    panel: pd.DataFrame, signal: str, outcome: str, k: int = 3, period: str = "entry_utc"
) -> pd.Series:
    """Per period, mean outcome of the top ``k`` assets by signal minus the bottom ``k``."""
    out = {}
    for key, rows in panel.dropna(subset=[signal, outcome]).groupby(period):
        if len(rows) < 2 * k:
            continue
        ordered = rows.sort_values(signal)
        out[key] = float(ordered[outcome].iloc[-k:].mean() - ordered[outcome].iloc[:k].mean())
    return pd.Series(out, dtype=float).sort_index()
