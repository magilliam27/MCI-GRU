"""
Export point-in-time S&P 500 membership snapshots from LSEG.

The training code currently uses fixed constituent CSVs. For PIT-safe universe
filtering, we need an as-of dated record of which RICs belonged to the index.
This script queries LSEG with the ``SDate`` parameter and writes both the raw
snapshots and compact validity intervals.

Requires Refinitiv Workspace to be running.

Examples:
    python scripts/data/export_sp500_pit_membership.py --start 2016-01-01 --end 2026-05-04
    python scripts/data/export_sp500_pit_membership.py --frequency daily --delay 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import refinitiv.data as rd
from tqdm import tqdm

DEFAULT_FIELDS = [
    "TR.IndexConstituentRIC",
    "TR.CommonName",
    "TR.CompanyMarketCap",
]

FALLBACK_FIELDS = [
    "TR.RIC",
    "TR.CommonName",
    "TR.CompanyMarketCap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PIT S&P 500 membership from LSEG.")
    parser.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD; default: today")
    parser.add_argument(
        "--frequency",
        choices=["adaptive", "daily", "weekly", "monthly", "quarterly"],
        default="adaptive",
        help=(
            "Snapshot cadence. adaptive pulls monthly anchors, then binary-searches "
            "changed windows to find exact business-day membership boundaries."
        ),
    )
    parser.add_argument("--chain-ric", default="0#.SPX", help="LSEG chain RIC")
    parser.add_argument("--fallback-ric", default=".SPX", help="Fallback index RIC")
    parser.add_argument(
        "--output-dir",
        default="data/raw/constituents",
        help="Directory for PIT snapshot outputs",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="Seconds between calls")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional max dates for smoke tests"
    )
    return parser.parse_args()


def build_asof_dates(start: str, end: str, frequency: str) -> list[str]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if frequency == "adaptive":
        dates = pd.date_range(start_ts, end_ts, freq="BME")
    elif frequency == "daily":
        dates = pd.bdate_range(start_ts, end_ts)
    elif frequency == "weekly":
        dates = pd.date_range(start_ts, end_ts, freq="W-FRI")
    elif frequency == "monthly":
        dates = pd.date_range(start_ts, end_ts, freq="BME")
    else:
        dates = pd.date_range(start_ts, end_ts, freq="BQE")

    if len(dates) == 0 or dates[-1] != end_ts:
        dates = dates.union(pd.DatetimeIndex([end_ts]))
    return [d.strftime("%Y-%m-%d") for d in dates if start_ts <= d <= end_ts]


def _normalise_snapshot(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    rename_map = {
        "TR.IndexConstituentRIC": "constituent_ric",
        "Index Constituent RIC": "constituent_ric",
        "Constituent RIC": "constituent_ric",
        "TR.RIC": "constituent_ric",
        "RIC": "constituent_ric",
        "Instrument": "instrument",
        "TR.CommonName": "company_name",
        "Company Common Name": "company_name",
        "Common Name": "company_name",
        "TR.IndexConstituentName": "company_name",
        "Index Constituent Name": "company_name",
        "TR.CompanyMarketCap": "company_market_cap",
        "Company Market Cap": "company_market_cap",
    }
    out = out.rename(columns={c: rename_map.get(c, c) for c in out.columns})

    if "constituent_ric" in out.columns:
        out["constituent_ric"] = out["constituent_ric"].replace("", pd.NA)

    if "constituent_ric" not in out.columns and "instrument" in out.columns:
        out["constituent_ric"] = out["instrument"]
    elif "instrument" in out.columns:
        out["constituent_ric"] = out["constituent_ric"].combine_first(out["instrument"])
    if "instrument" not in out.columns:
        out["instrument"] = out["constituent_ric"]

    out["as_of_date"] = as_of_date
    keep = ["as_of_date", "instrument", "constituent_ric", "company_name", "company_market_cap"]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[keep]
    out = out.dropna(subset=["constituent_ric"])
    out = out.drop_duplicates(subset=["as_of_date", "constituent_ric"], keep="first")
    return out.sort_values(["as_of_date", "constituent_ric"]).reset_index(drop=True)


def fetch_snapshot(chain_ric: str, fallback_ric: str, as_of_date: str) -> pd.DataFrame:
    try:
        df = rd.get_data(
            universe=[chain_ric],
            fields=DEFAULT_FIELDS,
            parameters={"SDate": as_of_date},
        )
    except Exception:
        df = None

    if df is None or df.empty:
        try:
            df = rd.get_data(
                universe=[fallback_ric],
                fields=DEFAULT_FIELDS,
                parameters={"SDate": as_of_date},
            )
        except Exception:
            df = None

    if df is None or df.empty:
        df = rd.get_data(
            universe=[chain_ric],
            fields=FALLBACK_FIELDS,
            parameters={"SDate": as_of_date},
        )

    return _normalise_snapshot(df, as_of_date)


def _snapshot_members(snapshot: pd.DataFrame) -> frozenset[str]:
    return frozenset(snapshot["constituent_ric"].dropna().astype(str))


def fetch_snapshot_cached(
    cache: dict[str, pd.DataFrame],
    chain_ric: str,
    fallback_ric: str,
    as_of_date: str,
    delay: float,
) -> pd.DataFrame:
    if as_of_date not in cache:
        cache[as_of_date] = fetch_snapshot(chain_ric, fallback_ric, as_of_date)
        time.sleep(delay)
    return cache[as_of_date]


def refine_changed_window(
    cache: dict[str, pd.DataFrame],
    chain_ric: str,
    fallback_ric: str,
    start: str,
    end: str,
    delay: float,
) -> None:
    start_snap = fetch_snapshot_cached(cache, chain_ric, fallback_ric, start, delay)
    end_snap = fetch_snapshot_cached(cache, chain_ric, fallback_ric, end, delay)
    if _snapshot_members(start_snap) == _snapshot_members(end_snap):
        return

    bdays = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    if len(bdays) <= 2:
        return

    mid = bdays[len(bdays) // 2].strftime("%Y-%m-%d")
    fetch_snapshot_cached(cache, chain_ric, fallback_ric, mid, delay)
    refine_changed_window(cache, chain_ric, fallback_ric, start, mid, delay)
    refine_changed_window(cache, chain_ric, fallback_ric, mid, end, delay)


def fetch_adaptive_snapshots(
    chain_ric: str,
    fallback_ric: str,
    start: str,
    end: str,
    delay: float,
    limit: int | None = None,
) -> tuple[list[pd.DataFrame], list[dict[str, str]]]:
    anchors = build_asof_dates(start, end, "adaptive")
    if limit is not None:
        anchors = anchors[:limit]

    cache: dict[str, pd.DataFrame] = {}
    failures = []

    print(f"  Anchor snapshots: {len(anchors)}")
    for as_of_date in tqdm(anchors, desc="Fetching PIT anchors"):
        try:
            snap = fetch_snapshot_cached(cache, chain_ric, fallback_ric, as_of_date, delay)
            if snap.empty:
                failures.append({"as_of_date": as_of_date, "reason": "empty"})
        except Exception as exc:
            failures.append({"as_of_date": as_of_date, "reason": str(exc)})

    print("  Refining changed anchor windows...")
    for prev_date, next_date in tqdm(
        list(zip(anchors, anchors[1:], strict=False)), desc="Refining PIT"
    ):
        if prev_date not in cache or next_date not in cache:
            continue
        try:
            refine_changed_window(cache, chain_ric, fallback_ric, prev_date, next_date, delay)
        except Exception as exc:
            failures.append({"as_of_date": f"{prev_date}..{next_date}", "reason": str(exc)})

    snapshots = [cache[d] for d in sorted(cache) if not cache[d].empty]
    return snapshots, failures


def build_intervals(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(
            columns=[
                "constituent_ric",
                "company_name",
                "valid_from",
                "valid_to",
                "snapshot_count",
            ]
        )

    dates = sorted(pd.to_datetime(snapshots["as_of_date"]).unique())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}

    rows = []
    for ric, group in snapshots.groupby("constituent_ric", sort=True):
        group_dates = sorted(pd.to_datetime(group["as_of_date"]).unique())
        start = group_dates[0]
        prev = group_dates[0]
        count = 1
        names = group.dropna(subset=["company_name"])["company_name"].astype(str)
        name = names.iloc[-1] if len(names) else ""

        for current in group_dates[1:]:
            if next_date.get(prev) == current:
                prev = current
                count += 1
                continue
            rows.append(
                {
                    "constituent_ric": ric,
                    "company_name": name,
                    "valid_from": pd.Timestamp(start).strftime("%Y-%m-%d"),
                    "valid_to": pd.Timestamp(prev).strftime("%Y-%m-%d"),
                    "snapshot_count": count,
                }
            )
            start = prev = current
            count = 1

        rows.append(
            {
                "constituent_ric": ric,
                "company_name": name,
                "valid_from": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "valid_to": pd.Timestamp(prev).strftime("%Y-%m-%d"),
                "snapshot_count": count,
            }
        )

    return pd.DataFrame(rows).sort_values(["constituent_ric", "valid_from"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    asof_dates = build_asof_dates(args.start, end, args.frequency)
    if args.limit is not None:
        asof_dates = asof_dates[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_start = args.start.replace("-", "")
    safe_end = end.replace("-", "")
    prefix = f"sp500_pit_{args.frequency}_{safe_start}_{safe_end}"
    snapshots_path = out_dir / f"{prefix}_snapshots.csv"
    intervals_path = out_dir / f"{prefix}_intervals.csv"
    meta_path = out_dir / f"{prefix}_meta.json"

    print("=" * 70)
    print("  S&P 500 PIT Membership Export")
    print("=" * 70)
    print(f"  Date range: {args.start} to {end}")
    print(f"  Frequency:  {args.frequency}")
    print(f"  Snapshots:  {len(asof_dates)} requested")
    print(f"  Output:     {snapshots_path}")

    rd.open_session()
    try:
        if args.frequency == "adaptive":
            snapshots, failures = fetch_adaptive_snapshots(
                args.chain_ric,
                args.fallback_ric,
                args.start,
                end,
                args.delay,
                args.limit,
            )
        else:
            snapshots = []
            failures = []
            for as_of_date in tqdm(asof_dates, desc="Fetching PIT snapshots"):
                try:
                    snap = fetch_snapshot(args.chain_ric, args.fallback_ric, as_of_date)
                    if snap.empty:
                        failures.append({"as_of_date": as_of_date, "reason": "empty"})
                    else:
                        snapshots.append(snap)
                except Exception as exc:
                    failures.append({"as_of_date": as_of_date, "reason": str(exc)})
                time.sleep(args.delay)
    finally:
        rd.close_session()

    if not snapshots:
        print("ERROR: No PIT snapshots were fetched.")
        return 1

    snapshot_df = pd.concat(snapshots, ignore_index=True)
    snapshot_df = snapshot_df.drop_duplicates(
        subset=["as_of_date", "constituent_ric"], keep="first"
    )
    interval_df = build_intervals(snapshot_df)

    snapshot_df.to_csv(snapshots_path, index=False)
    interval_df.to_csv(intervals_path, index=False)

    counts = snapshot_df.groupby("as_of_date")["constituent_ric"].nunique()
    meta = {
        "start": args.start,
        "end": end,
        "frequency": args.frequency,
        "chain_ric": args.chain_ric,
        "fallback_ric": args.fallback_ric,
        "snapshot_dates_requested": len(asof_dates),
        "snapshot_dates_fetched": int(counts.shape[0]),
        "unique_constituents": int(snapshot_df["constituent_ric"].nunique()),
        "min_constituents_per_snapshot": int(counts.min()),
        "max_constituents_per_snapshot": int(counts.max()),
        "failures": failures,
        "outputs": {
            "snapshots": str(snapshots_path),
            "intervals": str(intervals_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {snapshots_path} ({len(snapshot_df):,} rows)")
    print(f"Saved {intervals_path} ({len(interval_df):,} intervals)")
    print(f"Saved {meta_path}")
    if failures:
        print(f"Warnings: {len(failures)} snapshot dates failed; see metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
