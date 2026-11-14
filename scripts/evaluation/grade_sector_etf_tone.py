"""Grade E20: weekly Jev 8-K tone by sector against sector ETF returns (#394).

Reads the frozen inputs named in the experiment card (rated events, header SIC
codes, EODHD ETF prices) and writes ``summary.json`` plus per-period panels.
Every choice here is the card's; see ``mci_gru.filings.sector_tone`` for the
rules that keep the signal free of lookahead.

    python scripts/evaluation/grade_sector_etf_tone.py \
        --events events_all.csv --sic header_sic.csv \
        --prices data/etf_prices --out grading
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mci_gru.filings.sector_tone import (  # noqa: E402
    INDUSTRY_ETFS,
    SECTOR_ETFS,
    abnormal_tone,
    block_bootstrap_interval,
    cross_sectional_ic,
    entry_instants,
    rank_regression,
    sic_to_industry,
    sic_to_sector,
    top_minus_bottom,
    weekly_entry_days,
    window_tone,
)

START = pd.Timestamp("2013-01-01")
MIN_EVENTS = 5
SEED = 394
DRAWS = 10_000


def load_prices(folder: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        with gzip.open(folder / f"{sym}.US.json.gz", "rt") as fh:
            df = pd.DataFrame(json.load(fh))
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[(df["close"] > 0) & (df["open"] > 0)]
        factor = df["adjusted_close"] / df["close"]
        out[sym] = pd.DataFrame({"open": df["open"] * factor, "close": df["adjusted_close"]})
    return out


def point_in_time_sic(events: pd.DataFrame, header: pd.DataFrame) -> tuple[pd.Series, dict]:
    ok = header[(header["status"] == "ok") & header["sic"].notna() & (header["sic"] != "")]
    by_event = ok.set_index("event_id")["sic"].astype(str)
    by_cik = ok.groupby("cik")["sic"].agg(lambda s: s.astype(str).mode().iloc[0])
    sic = events["event_id"].map(by_event)
    source = pd.Series(np.where(sic.notna(), "own_header", ""), index=events.index)
    fill = sic.isna()
    sic[fill] = events.loc[fill, "cik"].map(by_cik)
    source[fill & sic.notna()] = "cik_header"
    fill = sic.isna()
    sic[fill] = events.loc[fill, "current_sic"]
    source[fill & sic.notna()] = "current_sec"
    source[sic.isna()] = "none"
    differs = (sic.notna() & events["current_sic"].notna()) & (
        sic.astype(str).str.split(".").str[0]
        != events["current_sic"].astype(str).str.split(".").str[0]
    )
    counts = {k: int(v) for k, v in source.value_counts().items()}
    counts["differs_from_current_sec_sic"] = int(differs.sum())
    return sic, counts


def momentum(prices: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex, sym: str) -> pd.DataFrame:
    """12-1 and 1-month ETF minus SPY momentum at each session's open (closes before it)."""
    px = prices[sym]["close"].reindex(sessions)
    spy = prices["SPY"]["close"].reindex(sessions)
    rel12 = (px.shift(21) / px.shift(252) - 1) - (spy.shift(21) / spy.shift(252) - 1)
    rel1 = (px.shift(1) / px.shift(22) - 1) - (spy.shift(1) / spy.shift(22) - 1)
    return pd.DataFrame({"mom12_1": rel12, "mom1": rel1})


def build_panel(
    events: pd.DataFrame,
    group_col: str,
    group_etf: dict[str, str],
    prices: dict[str, pd.DataFrame],
    sessions: pd.DatetimeIndex,
    entry_days: pd.DatetimeIndex,
    instants: pd.DatetimeIndex,
    lookback: int,
    exit_kind: str,
) -> pd.DataFrame:
    """One row per entry and group: signals, momentum and the ETF-minus-SPY outcome."""
    jev = window_tone(events, instants, group_col, "rating_score", lookback)
    jev["jev_abn"] = abnormal_tone(jev, events, group_col, "rating_score")
    words = window_tone(events, instants, group_col, "base_good", lookback)
    words["word_abn"] = abnormal_tone(words, events, group_col, "base_good")
    panel = jev.rename(columns={"mean": "jev_raw", "sum": "jev_sum"})
    panel["neg_bad_share"] = -panel["bad_share"]
    panel = panel.merge(
        words[["k", group_col, "mean", "word_abn"]].rename(columns={"mean": "word_raw"}),
        on=["k", group_col],
    )
    panel = panel[panel["n"] >= MIN_EVENTS].copy()
    panel["entry_day"] = entry_days[panel["k"].to_numpy()]
    panel["etf"] = panel[group_col].map(group_etf)
    pos = {d: i for i, d in enumerate(sessions)}
    spy = prices["SPY"]
    rows = []
    for etf, part in panel.groupby("etf"):
        px = prices[etf]
        first_ok = px.index[0]
        mom = momentum(prices, sessions, etf)
        for idx, r in part.iterrows():
            d = r["entry_day"]
            i = pos[d]
            if i < 5 or sessions[i - 5] < first_ok:
                continue
            if exit_kind == "weekly_open":
                k = int(r["k"])
                if k + 1 >= len(entry_days):
                    continue
                a_day, b_day, a_col, b_col = d, entry_days[k + 1], "open", "open"
            elif exit_kind == "weekly_close":
                # Close-entry variant: entry_day is the session before the weekly D.
                k = int(r["k"])
                if k + 1 >= len(entry_days):
                    continue
                a_day, b_day, a_col, b_col = d, entry_days[k + 1], "close", "close"
            else:
                hold = int(exit_kind.split("_")[1])
                if i + hold - 1 >= len(sessions):
                    continue
                a_day, b_day, a_col, b_col = d, sessions[i + hold - 1], "open", "close"
            try:
                etf_ret = px.at[b_day, b_col] / px.at[a_day, a_col] - 1
                spy_ret = spy.at[b_day, b_col] / spy.at[a_day, a_col] - 1
            except KeyError:
                continue
            rows.append(
                {
                    "idx": idx,
                    "outcome": etf_ret - spy_ret,
                    "mom12_1": mom.at[d, "mom12_1"],
                    "mom1": mom.at[d, "mom1"],
                }
            )
    extra = pd.DataFrame(rows).set_index("idx")
    return panel.join(extra, how="inner")


SIGNALS = [
    "jev_abn",
    "word_abn",
    "jev_raw",
    "word_raw",
    "jev_sum",
    "neg_bad_share",
    "mom12_1",
    "mom1",
]


def summarise(panel: pd.DataFrame, block: int, min_assets: int, periods_per_year: float) -> dict:
    out: dict = {
        "periods": int(panel["entry_utc"].nunique()),
        "rows": int(len(panel)),
        "events_used": int(panel["n"].sum()),
    }
    ics = {}
    for sig in SIGNALS:
        ic = cross_sectional_ic(panel, sig, "outcome", min_assets=min_assets)
        ics[sig] = ic
        m, lo, hi = block_bootstrap_interval(ic.to_numpy(), block, DRAWS, SEED)
        out[f"ic_{sig}"] = {"mean": m, "lo": lo, "hi": hi, "periods": int(len(ic))}
    both = pd.concat([ics["jev_abn"], ics["word_abn"]], axis=1, join="inner")
    diff = both.iloc[:, 0] - both.iloc[:, 1]
    m, lo, hi = block_bootstrap_interval(diff.to_numpy(), block, DRAWS, SEED)
    out["jev_minus_words_abn"] = {"mean": m, "lo": lo, "hi": hi, "periods": int(len(diff))}
    coefs = rank_regression(
        panel, "outcome", ["jev_abn", "mom12_1", "word_abn"], min_assets=min_assets
    )
    out["rank_regression"] = {}
    for col in coefs:
        m, lo, hi = block_bootstrap_interval(coefs[col].to_numpy(), block, DRAWS, SEED)
        out["rank_regression"][col] = {"mean": m, "lo": lo, "hi": hi, "periods": int(len(coefs))}
    for sig in ["jev_abn", "word_abn", "mom12_1"]:
        ls = top_minus_bottom(panel, sig, "outcome", k=3)
        m, lo, hi = block_bootstrap_interval(ls.to_numpy(), block, DRAWS, SEED)
        out[f"top3_minus_bottom3_{sig}"] = {
            "mean_per_period": m,
            "lo": lo,
            "hi": hi,
            "annualised_mean": m * periods_per_year,
            "periods": int(len(ls)),
        }
    ic = ics["jev_abn"]
    years = pd.to_datetime(ic.index).year
    out["by_year_jev_abn"] = {
        int(y): {"mean": float(v.mean()), "periods": int(len(v))} for y, v in ic.groupby(years)
    }
    out["years_positive_jev_abn"] = int(sum(v["mean"] > 0 for v in out["by_year_jev_abn"].values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--sic", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events, dtype={"cik": str, "current_sic": str})
    header = pd.read_csv(args.sic, dtype={"cik": str, "sic": str})
    header["cik"] = header["cik"].str.zfill(10)
    events["cik"] = events["cik"].str.zfill(10)
    events["sic"], sic_counts = point_in_time_sic(events, header)
    events["sector"] = events["sic"].map(sic_to_sector)
    events["industry"] = events["sic"].map(sic_to_industry)
    events["bad"] = events["rating_choice"].isin(["bad", "very_bad"]).astype(float)
    events["accepted_utc"] = pd.to_datetime(events["accepted_utc"], utc=True)

    symbols = ["SPY", *SECTOR_ETFS.values(), *INDUSTRY_ETFS]
    prices = load_prices(Path(args.prices), symbols)
    sessions = prices["SPY"].index
    sessions = sessions[sessions >= START - pd.Timedelta(days=400)]
    test_sessions = sessions[sessions >= START]
    weekly_days = weekly_entry_days(test_sessions)
    weekly_instants = entry_instants(weekly_days)
    # Close-entry variant: 15:45 ET on the session before each weekly D.
    pos = {d: i for i, d in enumerate(sessions)}
    prev_days = pd.DatetimeIndex([sessions[pos[d] - 1] for d in weekly_days])
    close_instants = (
        (prev_days + pd.Timedelta(hours=15, minutes=45))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    daily_days = test_sessions
    daily_instants = entry_instants(daily_days)

    summary: dict = {
        "events": int(len(events)),
        "sic_source": sic_counts,
        "events_by_sector": {
            k: int(v) for k, v in events["sector"].value_counts(dropna=False).items()
        },
        "events_by_industry": {k: int(v) for k, v in events["industry"].value_counts().items()},
    }
    universes = {
        "both": events,
        "sp500": events[events["universe"] == "sp500"],
        "r2k": events[events["universe"] == "r2k"],
    }
    runs = {}
    for uname, ev in universes.items():
        runs[f"sector_weekly_{uname}"] = (
            ev,
            "sector",
            SECTOR_ETFS,
            weekly_days,
            weekly_instants,
            1,
            "weekly_open",
            8,
            5,
            52,
        )
    ind_map = {e: e for e in INDUSTRY_ETFS}
    runs["industry_weekly_both"] = (
        events,
        "industry",
        ind_map,
        weekly_days,
        weekly_instants,
        1,
        "weekly_open",
        8,
        4,
        52,
    )
    runs["sector_daily2_both"] = (
        events,
        "sector",
        SECTOR_ETFS,
        daily_days,
        daily_instants,
        5,
        "hold_2",
        40,
        5,
        252 / 2,
    )
    runs["sector_daily5_both"] = (
        events,
        "sector",
        SECTOR_ETFS,
        daily_days,
        daily_instants,
        5,
        "hold_5",
        40,
        5,
        252 / 5,
    )
    runs["industry_daily2_both"] = (
        events,
        "industry",
        ind_map,
        daily_days,
        daily_instants,
        5,
        "hold_2",
        40,
        4,
        252 / 2,
    )
    runs["industry_daily5_both"] = (
        events,
        "industry",
        ind_map,
        daily_days,
        daily_instants,
        5,
        "hold_5",
        40,
        4,
        252 / 5,
    )
    runs["sector_weekly_close_entry_both"] = (
        events,
        "sector",
        SECTOR_ETFS,
        prev_days,
        close_instants,
        1,
        "weekly_close",
        8,
        5,
        52,
    )

    for name, (
        ev,
        gcol,
        gmap,
        days,
        inst,
        lookback,
        exit_kind,
        block,
        min_assets,
        ppy,
    ) in runs.items():
        panel = build_panel(ev, gcol, gmap, prices, sessions, days, inst, lookback, exit_kind)
        panel.to_csv(out_dir / f"panel_{name}.csv.gz", index=False)
        summary[name] = summarise(panel, block, min_assets, ppy)
        print(name, json.dumps(summary[name]["ic_jev_abn"]), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
