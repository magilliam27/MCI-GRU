"""
Export S&P 500 point-in-time membership from LSEG Joiner/Leaver data.

LSEG's simple ``0#.SPX`` + ``SDate`` chain calls can return a static current
membership set under some entitlements. The Joiner/Leaver endpoint provides
the actual change log, including a direction field whose LSEG mnemonic is the
typo-looking ``TR.IndexJLConstituentituentChange``.

This script starts from the current/live chain membership and walks backward
through Joiner/Leaver events to produce validity intervals suitable for a
future PIT universe filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import refinitiv.data as rd


JL_FIELDS = [
    "TR.IndexJLConstituentChangeDate",
    "TR.IndexJLConstituentRIC",
    "TR.IndexJLConstituentName",
    "TR.IndexJLConstituentituentChange",
]

CURRENT_FIELDS = ["TR.RIC", "TR.CommonName", "TR.CompanyMarketCap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export S&P 500 PIT Joiner/Leaver data.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-05-04")
    parser.add_argument("--index-ric", default=".SPX")
    parser.add_argument("--chain-ric", default="0#.SPX")
    parser.add_argument("--output-dir", default="data/raw/constituents")
    return parser.parse_args()


def previous_business_day(date: pd.Timestamp) -> str:
    return (date - pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def normalise_ric(value: object) -> str:
    return str(value).strip()


def fetch_current_members(chain_ric: str) -> pd.DataFrame:
    df = rd.get_data(universe=[chain_ric], fields=CURRENT_FIELDS)
    out = df.rename(
        columns={
            "Instrument": "constituent_ric",
            "Common Name": "company_name",
            "Company Common Name": "company_name",
            "Company Market Cap": "company_market_cap",
        }
    ).copy()
    out["constituent_ric"] = out["constituent_ric"].map(normalise_ric)
    keep = ["constituent_ric", "company_name", "company_market_cap"]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].drop_duplicates(subset=["constituent_ric"], keep="first")


def fetch_joiner_leaver(index_ric: str, start: str, end: str) -> pd.DataFrame:
    df = rd.get_data(
        universe=[index_ric],
        fields=JL_FIELDS,
        parameters={"SDATE": start, "EDATE": end, "IC": "B"},
    )
    out = df.rename(
        columns={
            "Date": "change_date",
            "Constituent RIC": "constituent_ric",
            "Constituent Name": "company_name",
            "Change": "change",
        }
    ).copy()
    out["change_date"] = pd.to_datetime(out["change_date"]).dt.strftime("%Y-%m-%d")
    out["constituent_ric"] = out["constituent_ric"].map(normalise_ric)
    out["change"] = out["change"].astype(str).str.strip()
    out = out.dropna(subset=["change_date", "constituent_ric", "change"])
    out = out.sort_values(["change_date", "change", "constituent_ric"]).reset_index(drop=True)
    return out[["change_date", "constituent_ric", "company_name", "change"]]


def reconstruct_intervals(
    current_members: pd.DataFrame,
    changes: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    names = {
        row.constituent_ric: row.company_name
        for row in current_members.itertuples(index=False)
        if pd.notna(row.company_name)
    }
    for row in changes.itertuples(index=False):
        if pd.notna(row.company_name):
            names[row.constituent_ric] = row.company_name

    active_end = {ric: end for ric in current_members["constituent_ric"].astype(str)}
    intervals: list[dict[str, str]] = []

    for change_date, group in changes.sort_values("change_date", ascending=False).groupby(
        "change_date", sort=False
    ):
        change_ts = pd.Timestamp(change_date)
        before_date = previous_business_day(change_ts)

        joiners = group[group["change"].str.lower() == "joiner"]
        leavers = group[group["change"].str.lower() == "leaver"]

        for row in joiners.itertuples(index=False):
            ric = row.constituent_ric
            if ric in active_end:
                intervals.append(
                    {
                        "constituent_ric": ric,
                        "company_name": names.get(ric, ""),
                        "valid_from": change_date,
                        "valid_to": active_end[ric],
                    }
                )
                del active_end[ric]

        for row in leavers.itertuples(index=False):
            ric = row.constituent_ric
            active_end.setdefault(ric, before_date)

    for ric, valid_to in active_end.items():
        intervals.append(
            {
                "constituent_ric": ric,
                "company_name": names.get(ric, ""),
                "valid_from": start,
                "valid_to": valid_to,
            }
        )

    out = pd.DataFrame(intervals)
    out = out.sort_values(["constituent_ric", "valid_from", "valid_to"]).reset_index(drop=True)
    return out


def build_change_date_snapshots(intervals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = sorted(
        set(intervals["valid_from"].tolist())
        | set((pd.to_datetime(intervals["valid_to"]) + pd.offsets.BDay(1)).dt.strftime("%Y-%m-%d"))
        | {start, end}
    )
    dates = [d for d in dates if start <= d <= end]
    rows = []
    interval_dates = intervals.copy()
    interval_dates["valid_from_dt"] = pd.to_datetime(interval_dates["valid_from"])
    interval_dates["valid_to_dt"] = pd.to_datetime(interval_dates["valid_to"])
    for date in dates:
        dt = pd.Timestamp(date)
        active = interval_dates[
            (interval_dates["valid_from_dt"] <= dt) & (interval_dates["valid_to_dt"] >= dt)
        ]
        for row in active.itertuples(index=False):
            rows.append(
                {
                    "as_of_date": date,
                    "constituent_ric": row.constituent_ric,
                    "company_name": row.company_name,
                }
            )
    return pd.DataFrame(rows).sort_values(["as_of_date", "constituent_ric"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_start = args.start.replace("-", "")
    safe_end = args.end.replace("-", "")
    prefix = f"sp500_pit_joiner_leaver_{safe_start}_{safe_end}"

    rd.open_session()
    try:
        current_members = fetch_current_members(args.chain_ric)
        changes = fetch_joiner_leaver(args.index_ric, args.start, args.end)
    finally:
        rd.close_session()

    intervals = reconstruct_intervals(current_members, changes, args.start, args.end)
    snapshots = build_change_date_snapshots(intervals, args.start, args.end)

    current_path = out_dir / f"{prefix}_current_members.csv"
    changes_path = out_dir / f"{prefix}_changes.csv"
    intervals_path = out_dir / f"{prefix}_intervals.csv"
    snapshots_path = out_dir / f"{prefix}_snapshots.csv"
    meta_path = out_dir / f"{prefix}_meta.json"

    current_members.to_csv(current_path, index=False)
    changes.to_csv(changes_path, index=False)
    intervals.to_csv(intervals_path, index=False)
    snapshots.to_csv(snapshots_path, index=False)

    counts = snapshots.groupby("as_of_date")["constituent_ric"].nunique()
    meta = {
        "start": args.start,
        "end": args.end,
        "index_ric": args.index_ric,
        "chain_ric": args.chain_ric,
        "current_members": int(current_members["constituent_ric"].nunique()),
        "change_rows": int(len(changes)),
        "joiners": int((changes["change"].str.lower() == "joiner").sum()),
        "leavers": int((changes["change"].str.lower() == "leaver").sum()),
        "interval_rows": int(len(intervals)),
        "snapshot_dates": int(snapshots["as_of_date"].nunique()),
        "min_members_per_snapshot": int(counts.min()) if len(counts) else 0,
        "max_members_per_snapshot": int(counts.max()) if len(counts) else 0,
        "outputs": {
            "current_members": str(current_path),
            "changes": str(changes_path),
            "intervals": str(intervals_path),
            "snapshots": str(snapshots_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {current_path} ({len(current_members):,} rows)")
    print(f"Saved {changes_path} ({len(changes):,} rows)")
    print(f"Saved {intervals_path} ({len(intervals):,} rows)")
    print(f"Saved {snapshots_path} ({len(snapshots):,} rows)")
    print(f"Saved {meta_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
