"""Sector tone from rated 8-Ks and the sector ETF rotation test (E20, #394)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.filings.sector_tone import (
    SECTOR_ETFS,
    abnormal_tone,
    assign_windows,
    block_bootstrap_interval,
    cross_sectional_ic,
    entry_instants,
    parse_header_sic,
    rank_regression,
    sic_to_industry,
    sic_to_sector,
    top_minus_bottom,
    weekly_entry_days,
    window_tone,
)


def _et(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC")


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        ("2834", "health"),  # pharma carved out of chemicals
        ("2821", "materials"),
        ("2844", "staples"),  # cosmetics carved out of chemicals
        ("1531", "discretionary"),  # homebuilders carved out of construction
        ("1600", "industrials"),
        ("3674", "technology"),  # semis carved out of industrial machinery range
        ("3826", "health"),  # lab instruments before the 3820-3829 tech range
        ("3825", "technology"),
        ("6798", "real_estate"),  # REITs carved out of financials
        ("6324", "health"),  # health plans carved out of insurance
        ("6311", "financials"),
        ("4922", "energy"),  # gas pipelines carved out of utilities
        ("4911", "utilities"),
        ("4953", "industrials"),  # waste carved out of utilities
        ("5411", "staples"),  # food stores carved out of retail
        ("5961", "discretionary"),
        ("7372", "technology"),  # software carved out of business services
        ("7389", "industrials"),
        ("4813", "communication"),
        ("1311", "energy"),
        ("1221", "energy"),  # coal before the materials mining ranges
        (" 3720 ", "industrials"),
        (2834.0, "health"),
    ],
)
def test_sic_to_sector_exceptions_take_precedence(sic: object, sector: str) -> None:
    assert sic_to_sector(sic) == sector


@pytest.mark.parametrize("sic", ["9995", "", None, "abc", np.nan, "8880"])
def test_sic_to_sector_unmapped_is_none(sic: object) -> None:
    assert sic_to_sector(sic) is None


def test_every_sector_has_an_etf() -> None:
    mapped = {sic_to_sector(code) for code in range(100, 10000)} - {None}
    assert mapped == set(SECTOR_ETFS)


@pytest.mark.parametrize(
    ("sic", "etf"),
    [
        ("2836", "XBI"),
        ("8731", "XBI"),
        ("6022", "KRE"),
        ("1311", "XOP"),
        ("3674", "SMH"),
        ("1531", "ITB"),
        ("5651", "XRT"),
        ("5411", None),  # food stores excluded from retail
        ("5812", None),  # restaurants excluded from retail
        ("3312", "XME"),
        ("6331", "KIE"),
        ("6324", None),  # health plans excluded from insurance
        ("4512", "IYT"),
        ("4612", None),  # pipelines are not transport
        ("7372", None),
    ],
)
def test_sic_to_industry(sic: str, etf: str | None) -> None:
    assert sic_to_industry(sic) == etf


HEADER = """<SEC-HEADER>0000000000-13-000001.hdr.sgml : 20130403
<TYPE>8-K
<FILER>
<COMPANY-DATA>
<CONFORMED-NAME>PARENT CO
<CIK>0000001111
<ASSIGNED-SIC>6022
</COMPANY-DATA>
</FILER>
<FILER>
<COMPANY-DATA>
<CONFORMED-NAME>SUBSIDIARY CO
<CIK>0000002222
<ASSIGNED-SIC>4911
</COMPANY-DATA>
</FILER>
<FILER>
<COMPANY-DATA>
<CONFORMED-NAME>NO SIC CO
<CIK>0000003333
</COMPANY-DATA>
</FILER>
"""


def test_parse_header_sic_reads_the_requested_filer_block() -> None:
    assert parse_header_sic(HEADER, "0000002222") == "4911"
    assert parse_header_sic(HEADER, 1111) == "6022"
    assert parse_header_sic(HEADER, "3333") == ""
    assert parse_header_sic(HEADER, "9999") == ""


def test_weekly_entry_days_take_first_session_of_each_week() -> None:
    sessions = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-16"]
    )
    # 2024-01-01 and 2024-01-15 are holidays, so Tuesday opens those weeks.
    got = weekly_entry_days(sessions)
    assert list(got.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-08", "2024-01-16"]


def test_entry_instants_are_0930_eastern_across_dst() -> None:
    got = entry_instants(pd.to_datetime(["2024-01-08", "2024-07-08"]))
    assert list(got.strftime("%H:%M")) == ["14:30", "13:30"]


def test_assign_windows_is_half_open_at_the_entry_instant() -> None:
    instants = entry_instants(pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22"]))
    times = pd.Series(
        [
            _et("2024-01-08 09:29:59"),  # before first window start: unusable
            _et("2024-01-08 09:30:00"),  # starts window 1
            _et("2024-01-15 09:29:59"),  # last second of window 1
            _et("2024-01-15 09:30:00"),  # exactly at entry 1: not known, goes to window 2
            _et("2024-01-22 09:30:00"),  # at the last instant: no entry can use it
        ]
    )
    assert list(assign_windows(times, instants)) == [-1, 1, 1, 2, -1]


def _events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accepted_utc": [_et(t) for t, _, _ in rows],
            "sector": [s for _, s, _ in rows],
            "score": [v for _, _, v in rows],
        }
    )


def test_window_tone_means_and_rolling_lookback() -> None:
    instants = entry_instants(pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]))
    ev = _events(
        [
            ("2024-01-08 12:00", "a", 1.0),
            ("2024-01-08 18:00", "a", -1.0),
            ("2024-01-09 12:00", "a", 2.0),
            ("2024-01-09 12:00", "b", 0.5),
            ("2024-01-10 09:31", "a", 9.0),  # after the last instant: unusable
        ]
    )
    one = window_tone(ev, instants, "sector", "score", lookback=1).set_index(["k", "sector"])
    assert one.loc[(1, "a"), "n"] == 2
    assert one.loc[(1, "a"), "mean"] == pytest.approx(0.0)
    assert one.loc[(2, "a"), "mean"] == pytest.approx(2.0)
    assert np.isnan(one.loc[(1, "b"), "mean"])
    two = window_tone(ev, instants, "sector", "score", lookback=2).set_index(["k", "sector"])
    assert list(two.index.get_level_values("k").unique()) == [2]
    assert two.loc[(2, "a"), "n"] == 3
    assert two.loc[(2, "a"), "sum"] == pytest.approx(2.0)
    assert two.loc[(2, "a"), "window_start_utc"] == instants[0]


def test_abnormal_tone_history_stops_at_window_start() -> None:
    weekly = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    instants = entry_instants(weekly)
    rows = [(f"{d:%Y-%m-%d} 12:00", "a", 0.0) for d in weekly[:-2]]
    ev = _events(rows)
    # Window events and anything later are extreme; neither may enter the baseline.
    k = 58
    ev.loc[ev["accepted_utc"] >= instants[k - 1], "score"] = 5.0
    ev = pd.concat([ev, _events([(f"{weekly[-1]:%Y-%m-%d} 12:00", "a", 100.0)])], ignore_index=True)
    tone = window_tone(ev, instants, "sector", "score", lookback=1)
    tone["abn"] = abnormal_tone(tone, ev, "sector", "score")
    row = tone[(tone["k"] == k) & (tone["sector"] == "a")].iloc[0]
    assert row["mean"] == pytest.approx(5.0)
    assert row["abn"] == pytest.approx(5.0)  # history mean is exactly 0


def test_abnormal_tone_history_is_a_trailing_52_week_mean() -> None:
    weekly = pd.date_range("2022-01-03", periods=120, freq="W-MON")
    instants = entry_instants(weekly)
    # Score equals the week number, so the trailing mean is known exactly.
    ev = _events([(f"{d:%Y-%m-%d} 12:00", "a", float(i)) for i, d in enumerate(weekly)])
    tone = window_tone(ev, instants, "sector", "score", lookback=1)
    tone["abn"] = abnormal_tone(tone, ev, "sector", "score")
    row = tone[tone["k"] == 100].iloc[0]
    # Window k=100 holds week 99; history is weeks [99-52, 99) = 47..98.
    assert row["mean"] == pytest.approx(99.0)
    assert row["abn"] == pytest.approx(99.0 - np.mean(np.arange(47, 99)))


def test_abnormal_tone_needs_26_weeks_of_history() -> None:
    weekly = pd.date_range("2023-01-02", periods=40, freq="W-MON")
    instants = entry_instants(weekly)
    ev = _events([(f"{d:%Y-%m-%d} 12:00", "a", 1.0) for d in weekly])
    tone = window_tone(ev, instants, "sector", "score", lookback=1)
    tone["abn"] = abnormal_tone(tone, ev, "sector", "score")
    by_k = tone.set_index("k")["abn"]
    assert by_k.loc[:26].isna().all()
    assert by_k.loc[28:].notna().all()


def test_cross_sectional_ic_and_min_assets() -> None:
    panel = pd.DataFrame(
        {
            "entry_utc": [1] * 5 + [2] * 5 + [3] * 3,
            "sig": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
            "ret": [10, 20, 30, 40, 50, 5, 4, 3, 2, 1, 1, 2, 3],
        }
    )
    ic = cross_sectional_ic(panel, "sig", "ret", min_assets=5)
    assert list(ic.index) == [1, 2]
    assert ic.loc[1] == pytest.approx(1.0)
    assert ic.loc[2] == pytest.approx(-1.0)


def test_top_minus_bottom() -> None:
    panel = pd.DataFrame(
        {"entry_utc": [1] * 6, "sig": [6, 5, 4, 3, 2, 1], "ret": [3.0, 3.0, 3.0, 0.0, 0.0, 0.0]}
    )
    assert top_minus_bottom(panel, "sig", "ret", k=3).loc[1] == pytest.approx(3.0)


def test_rank_regression_separates_signals() -> None:
    rng = np.random.default_rng(0)
    frames = []
    for p in range(200):
        a = rng.normal(size=11)
        b = rng.normal(size=11)
        frames.append(
            pd.DataFrame({"entry_utc": p, "a": a, "b": b, "y": a + 0.05 * rng.normal(size=11)})
        )
    coefs = rank_regression(pd.concat(frames), "y", ["a", "b"])
    assert coefs["a"].mean() > 0.8
    assert abs(coefs["b"].mean()) < 0.05


def test_block_bootstrap_interval_is_deterministic_and_brackets_mean() -> None:
    x = np.random.default_rng(1).normal(0.1, 1.0, size=500)
    first = block_bootstrap_interval(x, block=8, draws=2000, seed=394)
    again = block_bootstrap_interval(x, block=8, draws=2000, seed=394)
    assert first == again
    mean, lo, hi = first
    assert mean == pytest.approx(x.mean())
    assert lo < mean < hi
    assert 0.1 < hi - lo < 0.3  # about 4 standard errors of 1/sqrt(500)
