"""Export a monthly PIT S&P 500 top-N-by-market-cap GICS universe from LSEG.

For each as-of snapshot date, this script reconstructs point-in-time S&P 500
membership from Joiner/Leaver intervals, fetches contemporaneous market cap and
GICS sector metadata, selects the top N names per sector, and writes compact
``kdcode, valid_from, valid_to`` intervals for masked-panel training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mci_gru.data.lseg_loader import LSEGLoader  # noqa: E402
from scripts.data.export_sp500_gics_top10_mcap import (  # noqa: E402
    GICS_SECTOR_CANDIDATES,
    _find_sector_column,
    _normalise_metadata,
    select_top_by_sector,
)
from scripts.data.export_sp500_joiner_leaver_pit import (  # noqa: E402
    fetch_current_members,
    fetch_joiner_leaver,
    reconstruct_intervals,
)

METADATA_BASE_FIELDS = ["TR.CommonName", "TR.CompanyMarketCap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export monthly point-in-time S&P 500 top-N-by-market-cap names "
            "within each GICS sector."
        )
    )
    parser.add_argument("--start", default="2018-01-02", help="Selector start date")
    parser.add_argument("--end", required=True, help="Selector end date")
    parser.add_argument("--history-start", default="2015-01-01")
    parser.add_argument("--history-end", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--frequency",
        choices=["monthly", "weekly", "daily"],
        default="monthly",
        help="Selection snapshot cadence",
    )
    parser.add_argument("--index-ric", default=".SPX")
    parser.add_argument("--chain-ric", default="0#.SPX")
    parser.add_argument("--expected-sectors", type=int, default=11)
    parser.add_argument("--metadata-batch-size", type=int, default=250)
    parser.add_argument("--history-batch-size", type=int, default=25)
    parser.add_argument("--snapshot-delay", type=float, default=0.25)
    parser.add_argument("--batch-delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit snapshots for smoke runs")
    parser.add_argument(
        "--membership-intervals-csv",
        type=Path,
        default=None,
        help="Optional prebuilt membership intervals; otherwise fetch Joiner/Leaver data",
    )
    parser.add_argument(
        "--constituents-dir",
        type=Path,
        default=Path("data/raw/constituents"),
    )
    parser.add_argument(
        "--market-dir",
        type=Path,
        default=Path("data/raw/market"),
    )
    parser.add_argument("--skip-history", action="store_true")
    return parser.parse_args()


def _date_str(value: pd.Timestamp | str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_asof_dates(
    start: str,
    end: str,
    frequency: str = "monthly",
    limit: int | None = None,
) -> list[str]:
    """Return business-aware as-of dates including the first business day and end."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    business_days = pd.bdate_range(start_ts, end_ts)
    if business_days.empty:
        raise ValueError(f"No business days in selector range: {start} to {end}")

    if frequency == "daily":
        dates = business_days
    elif frequency == "weekly":
        dates = pd.date_range(start_ts, end_ts, freq="W-FRI")
    else:
        dates = pd.date_range(start_ts, end_ts, freq="BME")

    anchors = pd.DatetimeIndex([business_days[0]]).union(dates)
    anchors = anchors.union(pd.DatetimeIndex([end_ts]))
    anchors = anchors[(anchors >= start_ts) & (anchors <= end_ts)]
    out = sorted({_date_str(date) for date in anchors})
    if limit is not None:
        out = out[:limit]
    return out


def normalise_membership_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    frame = intervals.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "constituent_ric" not in frame.columns and "kdcode" in frame.columns:
        frame = frame.rename(columns={"kdcode": "constituent_ric"})
    elif "constituent_ric" in frame.columns and "kdcode" in frame.columns:
        frame["constituent_ric"] = frame["constituent_ric"].combine_first(frame["kdcode"])
    required = {"constituent_ric", "valid_from", "valid_to"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Membership intervals missing columns: {sorted(missing)}")
    frame = frame[["constituent_ric", "valid_from", "valid_to"]].copy()
    frame = frame.dropna(subset=["constituent_ric", "valid_from", "valid_to"])
    frame["constituent_ric"] = frame["constituent_ric"].astype(str).str.strip()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"]).dt.strftime("%Y-%m-%d")
    frame["valid_to"] = pd.to_datetime(frame["valid_to"]).dt.strftime("%Y-%m-%d")
    frame = frame[frame["constituent_ric"] != ""]
    return frame.sort_values(["constituent_ric", "valid_from", "valid_to"]).reset_index(drop=True)


def active_constituents_on_date(intervals: pd.DataFrame, as_of_date: str) -> list[str]:
    frame = normalise_membership_intervals(intervals)
    mask = (frame["valid_from"] <= as_of_date) & (frame["valid_to"] >= as_of_date)
    return sorted(frame.loc[mask, "constituent_ric"].astype(str).unique())


def load_or_fetch_membership_intervals(
    loader: LSEGLoader,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    if args.membership_intervals_csv is not None:
        intervals = normalise_membership_intervals(pd.read_csv(args.membership_intervals_csv))
        return intervals, None, None

    current_members = fetch_current_members(args.chain_ric)
    changes = fetch_joiner_leaver(args.index_ric, args.start, args.end)
    intervals = reconstruct_intervals(current_members, changes, args.start, args.end)
    return normalise_membership_intervals(intervals), current_members, changes


def _batch(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def fetch_asof_metadata(
    loader: LSEGLoader,
    rics: list[str],
    as_of_date: str,
    batch_size: int,
    delay: float,
) -> tuple[pd.DataFrame, str]:
    """Fetch contemporaneous market cap and GICS metadata for active RICs."""
    assert loader.rd is not None
    errors: list[str] = []
    for sector_field in GICS_SECTOR_CANDIDATES:
        frames: list[pd.DataFrame] = []
        for batch in _batch(rics, batch_size):
            try:
                raw = loader.rd.get_data(
                    universe=batch,
                    fields=METADATA_BASE_FIELDS + [sector_field],
                    parameters={"SDate": as_of_date},
                )
            except Exception as exc:
                errors.append(f"{as_of_date} {sector_field}: {type(exc).__name__}: {exc}")
                raw = None

            if raw is not None and not raw.empty:
                sector_column = _find_sector_column(raw, sector_field)
                if sector_column is None:
                    errors.append(f"{as_of_date} {sector_field}: could not infer sector column")
                else:
                    frames.append(_normalise_metadata(raw, sector_column, sector_field))

            if len(rics) > batch_size:
                time.sleep(delay)

        if not frames:
            continue

        metadata = pd.concat(frames, ignore_index=True)
        metadata = metadata.drop_duplicates(subset=["kdcode"], keep="first")
        metadata = metadata.sort_values(
            ["gics_sector", "company_market_cap"], ascending=[True, False]
        )
        if metadata["gics_sector"].nunique() >= 10:
            return metadata.reset_index(drop=True), sector_field

    raise RuntimeError(
        f"Could not fetch usable GICS metadata for {as_of_date}. Recent errors: {errors[-5:]}"
    )


def validate_snapshot_selection(
    selected: pd.DataFrame,
    *,
    as_of_date: str,
    top_n: int,
    expected_sectors: int,
) -> None:
    counts = selected.groupby("gics_sector")["kdcode"].nunique()
    short = counts[counts != top_n]
    if counts.shape[0] != expected_sectors or not short.empty:
        raise ValueError(
            f"{as_of_date} selected {counts.shape[0]} sectors; "
            f"expected {expected_sectors} with {top_n} names each. "
            f"Counts: {counts.to_dict()}"
        )


def build_selected_snapshots(
    loader: LSEGLoader,
    membership_intervals: pd.DataFrame,
    as_of_dates: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    selected_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for as_of_date in tqdm(as_of_dates, desc="Selecting monthly PIT GICS top-N"):
        active = active_constituents_on_date(membership_intervals, as_of_date)
        metadata, sector_field = fetch_asof_metadata(
            loader,
            active,
            as_of_date,
            batch_size=args.metadata_batch_size,
            delay=args.batch_delay,
        )
        selected = select_top_by_sector(metadata, args.top_n)
        validate_snapshot_selection(
            selected,
            as_of_date=as_of_date,
            top_n=args.top_n,
            expected_sectors=args.expected_sectors,
        )

        metadata = metadata.copy()
        metadata.insert(0, "as_of_date", as_of_date)
        selected = selected.copy()
        selected.insert(0, "as_of_date", as_of_date)
        selected["active_sp500_count"] = len(active)
        selected["metadata_rows"] = int(metadata["kdcode"].nunique())

        metadata_frames.append(metadata)
        selected_frames.append(selected)
        counts = selected.groupby("gics_sector")["kdcode"].nunique().to_dict()
        audit_rows.append(
            {
                "as_of_date": as_of_date,
                "active_sp500_count": len(active),
                "metadata_rows": int(metadata["kdcode"].nunique()),
                "selected_rows": int(len(selected)),
                "selected_unique_kdcodes": int(selected["kdcode"].nunique()),
                "sector_field": sector_field,
                "sector_counts": {str(k): int(v) for k, v in counts.items()},
            }
        )
        time.sleep(args.snapshot_delay)

    return (
        pd.concat(selected_frames, ignore_index=True),
        pd.concat(metadata_frames, ignore_index=True),
        audit_rows,
    )


def _coalesce_selection_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    frame = intervals.copy()
    frame["_from"] = pd.to_datetime(frame["valid_from"])
    frame["_to"] = pd.to_datetime(frame["valid_to"])
    frame = frame.sort_values(["kdcode", "_from", "_to"])
    rows: list[dict[str, str]] = []

    for kdcode, group in frame.groupby("kdcode", sort=True):
        current_from: pd.Timestamp | None = None
        current_to: pd.Timestamp | None = None
        for valid_from, valid_to in group[["_from", "_to"]].itertuples(
            index=False,
            name=None,
        ):
            if current_from is None or current_to is None:
                current_from = valid_from
                current_to = valid_to
                continue
            if valid_from <= current_to + pd.Timedelta(days=1):
                current_to = max(current_to, valid_to)
                continue
            rows.append(
                {
                    "kdcode": str(kdcode),
                    "valid_from": _date_str(current_from),
                    "valid_to": _date_str(current_to),
                }
            )
            current_from = valid_from
            current_to = valid_to

        if current_from is not None and current_to is not None:
            rows.append(
                {
                    "kdcode": str(kdcode),
                    "valid_from": _date_str(current_from),
                    "valid_to": _date_str(current_to),
                }
            )

    return pd.DataFrame(rows, columns=["kdcode", "valid_from", "valid_to"])


def build_selection_intervals(selected_snapshots: pd.DataFrame, end: str) -> pd.DataFrame:
    """Convert selection snapshots into compact PIT intervals."""
    if selected_snapshots.empty:
        return pd.DataFrame(columns=["kdcode", "valid_from", "valid_to"])

    dates = sorted(pd.to_datetime(selected_snapshots["as_of_date"]).unique())
    valid_to_by_date: dict[str, str] = {}
    for i, as_of_ts in enumerate(dates):
        if i + 1 < len(dates):
            valid_to = pd.Timestamp(dates[i + 1]) - pd.Timedelta(days=1)
        else:
            valid_to = pd.Timestamp(end)
        valid_to_by_date[_date_str(pd.Timestamp(as_of_ts))] = _date_str(valid_to)

    rows = []
    for row in (
        selected_snapshots[["as_of_date", "kdcode"]].drop_duplicates().itertuples(index=False)
    ):
        as_of_date = _date_str(row.as_of_date)
        rows.append(
            {
                "kdcode": str(row.kdcode).strip(),
                "valid_from": as_of_date,
                "valid_to": valid_to_by_date[as_of_date],
            }
        )

    intervals = pd.DataFrame(rows)
    intervals = intervals[intervals["kdcode"] != ""]
    return (
        _coalesce_selection_intervals(intervals)
        .sort_values(["kdcode", "valid_from", "valid_to"])
        .reset_index(drop=True)
    )


def _write_optional(frame: pd.DataFrame | None, path: Path) -> str | None:
    if frame is None:
        return None
    frame.to_csv(path, index=False)
    return str(path)


def write_constituent_outputs(
    selected_snapshots: pd.DataFrame,
    metadata_snapshots: pd.DataFrame,
    pit_universe: pd.DataFrame,
    membership_intervals: pd.DataFrame,
    current_members: pd.DataFrame | None,
    changes: pd.DataFrame | None,
    audit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    args.constituents_dir.mkdir(parents=True, exist_ok=True)
    safe_start = args.start.replace("-", "")
    safe_end = args.end.replace("-", "")
    prefix = f"sp500_pit_gics_top{args.top_n}_mcap_{args.frequency}_{safe_start}_{safe_end}"

    snapshots_path = args.constituents_dir / f"{prefix}_snapshots.csv"
    metadata_path = args.constituents_dir / f"{prefix}_all_metadata_snapshots.csv"
    pit_path = args.constituents_dir / f"{prefix}_pit_universe.csv"
    membership_path = args.constituents_dir / f"{prefix}_sp500_membership_intervals.csv"
    current_path = args.constituents_dir / f"{prefix}_current_members.csv"
    changes_path = args.constituents_dir / f"{prefix}_changes.csv"
    meta_path = args.constituents_dir / f"{prefix}_meta.json"

    selected_snapshots.to_csv(snapshots_path, index=False)
    metadata_snapshots.to_csv(metadata_path, index=False)
    pit_universe.to_csv(pit_path, index=False)
    membership_intervals.to_csv(membership_path, index=False)

    counts = selected_snapshots.groupby("as_of_date")["kdcode"].nunique()
    union_count = int(pit_universe["kdcode"].nunique()) if len(pit_universe) else 0
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "refinitiv.data",
        "selector": f"{args.frequency}_sp500_gics_top{args.top_n}_by_market_cap",
        "start": args.start,
        "end": args.end,
        "history_start": args.history_start,
        "history_end": args.history_end,
        "frequency": args.frequency,
        "top_n_per_sector": args.top_n,
        "expected_sectors": args.expected_sectors,
        "index_ric": args.index_ric,
        "chain_ric": args.chain_ric,
        "snapshot_dates": int(selected_snapshots["as_of_date"].nunique()),
        "selected_rows": int(len(selected_snapshots)),
        "min_selected_per_snapshot": int(counts.min()),
        "max_selected_per_snapshot": int(counts.max()),
        "pit_interval_rows": int(len(pit_universe)),
        "pit_union_kdcodes": union_count,
        "audit": audit_rows,
        "outputs": {
            "selected_snapshots": str(snapshots_path),
            "all_metadata_snapshots": str(metadata_path),
            "pit_universe": str(pit_path),
            "sp500_membership_intervals": str(membership_path),
            "current_members": _write_optional(current_members, current_path),
            "changes": _write_optional(changes, changes_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    meta["outputs"]["metadata"] = str(meta_path)
    return meta, prefix


def fetch_history(
    loader: LSEGLoader,
    pit_universe: pd.DataFrame,
    prefix: str,
    args: argparse.Namespace,
    constituent_meta: dict[str, Any],
) -> dict[str, Any]:
    args.market_dir.mkdir(parents=True, exist_ok=True)
    start_safe = args.history_start.replace("-", "")
    end_safe = args.history_end.replace("-", "")
    price_path = args.market_dir / f"{prefix}_lseg_{start_safe}_{end_safe}.csv"
    price_meta_path = price_path.with_suffix(".meta.json")

    rics = sorted(pit_universe["kdcode"].dropna().astype(str).unique())
    prices = loader.get_historical_prices(
        rics,
        start=args.history_start,
        end=args.history_end,
        batch_size=args.history_batch_size,
        delay_between_batches=args.batch_delay,
    )
    prices.to_csv(price_path, index=False)

    coverage = (
        prices.groupby("kdcode")["dt"]
        .agg(row_count="size", first_dt="min", last_dt="max")
        .reset_index()
    )
    coverage_path = price_path.with_name(price_path.stem + "_coverage.csv")
    coverage.to_csv(coverage_path, index=False)
    missing = sorted(set(rics) - set(coverage["kdcode"].astype(str)))

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "refinitiv.data.get_history",
        "selected_pit_universe": constituent_meta["outputs"]["pit_universe"],
        "start": args.history_start,
        "end": args.history_end,
        "requested_identifiers": len(rics),
        "resolved_identifiers_with_rows": int(coverage["kdcode"].nunique()),
        "missing_identifiers": missing,
        "rows": int(len(prices)),
        "date_min": str(prices["dt"].min()) if len(prices) else None,
        "date_max": str(prices["dt"].max()) if len(prices) else None,
        "outputs": {
            "prices": str(price_path),
            "coverage": str(coverage_path),
        },
    }
    with price_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    meta["outputs"]["metadata"] = str(price_meta_path)
    return meta


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    as_of_dates = build_asof_dates(args.start, args.end, args.frequency, args.limit)
    print("=" * 80)
    print("S&P 500 PIT GICS Top-N Market-Cap Export")
    print("=" * 80)
    print(f"Selector range: {args.start} to {args.end}")
    print(f"Frequency:      {args.frequency}")
    print(f"Snapshots:      {len(as_of_dates)}")
    print(f"History range:  {args.history_start} to {args.history_end}")

    loader = LSEGLoader()
    loader.connect()
    try:
        membership_intervals, current_members, changes = load_or_fetch_membership_intervals(
            loader,
            args,
        )
        selected_snapshots, metadata_snapshots, audit_rows = build_selected_snapshots(
            loader,
            membership_intervals,
            as_of_dates,
            args,
        )
        pit_universe = build_selection_intervals(selected_snapshots, args.end)
        constituent_meta, prefix = write_constituent_outputs(
            selected_snapshots,
            metadata_snapshots,
            pit_universe,
            membership_intervals,
            current_members,
            changes,
            audit_rows,
            args,
        )
        print(json.dumps(constituent_meta, indent=2))

        if not args.skip_history:
            price_meta = fetch_history(loader, pit_universe, prefix, args, constituent_meta)
            print(json.dumps(price_meta, indent=2))
    finally:
        loader.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
