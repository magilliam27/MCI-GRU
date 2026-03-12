"""
Incremental LSEG data refresh for the paper trading pipeline.

Fetches new OHLCV bars since the last date in the master CSV, appends them,
and runs validation checks. Uses the same proven reshape logic as
scripts/data/download_all_universes.py.

Requires: Refinitiv Workspace desktop app to be running.

Usage:
    python paper_trade/scripts/refresh_data.py
    python paper_trade/scripts/refresh_data.py --csv data/raw/market/sp500_2019_universe_data_through_2026.csv
    python paper_trade/scripts/refresh_data.py --constituents data/raw/constituents/sp500_constituents_2019.csv
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BATCH_SIZE = 50
BATCH_DELAY = 2.0
MAX_RETRIES = 3
RETRY_BACKOFF = 4.0
MIN_COVERAGE_DEFAULT = 0.80
FETCH_LOOKBACK_DAYS = 7
BENCHMARK_RIC = "SPY.P"

LSEG_FIELDS = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]

COLUMN_MAPPING = {
    'MKT_OPEN': 'open', 'MKT_HIGH': 'high', 'MKT_LOW': 'low',
    'TRDPRC_1': 'close', 'ACVOL_UNS': 'volume',
    'OPEN_PRC': 'open', 'HIGH_1': 'high', 'LOW_1': 'low', 'HST_CLOSE': 'close',
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume',
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
}

STANDARD_COLS = ['kdcode', 'dt', 'open', 'high', 'low', 'close', 'volume', 'turnover']


def load_rics(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    return df['Instrument'].dropna().tolist()


def get_last_date(csv_path: str) -> str:
    df = pd.read_csv(csv_path, usecols=['dt'])
    return df['dt'].max()


def _fetch_batch(rd, batch: list, start: str, end: str) -> pd.DataFrame | None:
    """Fetch a single batch from LSEG and return the DataFrame (or None)."""
    try:
        df = rd.get_history(
            universe=batch,
            fields=LSEG_FIELDS,
            start=start,
            end=end,
            interval="1D",
        )
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        raise e
    return None


def _extract_rics_from_batch(df: pd.DataFrame) -> set:
    """Extract the set of RIC codes present in a returned batch DataFrame."""
    if df is None or df.empty:
        return set()
    if isinstance(df.columns, pd.MultiIndex):
        rics = df.columns.get_level_values(0).unique().tolist()
        return {r for r in rics if r not in ("Date", "Instrument", "index")}
    return set()


def download_incremental(rd, rics: list, start: str, end: str) -> tuple:
    """
    Download daily OHLCV from LSEG in batches with retry logic.

    Returns (list_of_dataframes, set_of_fetched_rics).
    Failed/partial batches are retried up to MAX_RETRIES times
    with exponential backoff. After all batches, any still-missing
    RICs are retried individually in smaller batches.
    """
    all_data = []
    fetched_rics = set()
    requested_rics = set(rics)

    # --- Pass 1: batch download with retries ---
    for i in tqdm(range(0, len(rics), BATCH_SIZE), desc="Fetching"):
        batch = rics[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE

        df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = _fetch_batch(rd, batch, start, end)
                if df is not None:
                    break
                wait = RETRY_BACKOFF * attempt
                print(f"    Batch {batch_num} attempt {attempt}/{MAX_RETRIES} "
                      f"returned empty (retry in {wait:.0f}s)")
                time.sleep(wait)
            except Exception as e:
                wait = RETRY_BACKOFF * attempt
                print(f"    Batch {batch_num} attempt {attempt}/{MAX_RETRIES} "
                      f"failed: {e}  (retry in {wait:.0f}s)")
                time.sleep(wait)

        if df is not None:
            batch_rics = _extract_rics_from_batch(df)
            fetched_rics.update(batch_rics)
            all_data.append(df)
            if len(batch_rics) < len(batch):
                print(f"    Batch {batch_num}: got {len(batch_rics)}/{len(batch)} RICs")
        else:
            print(f"    Batch {batch_num}: FAILED after {MAX_RETRIES} attempts "
                  f"({len(batch)} RICs lost)")

        time.sleep(BATCH_DELAY)

    # --- Pass 2: retry missing RICs in smaller batches ---
    missing = sorted(requested_rics - fetched_rics)
    if missing:
        retry_batch_size = 10
        n_micro = (len(missing) + retry_batch_size - 1) // retry_batch_size
        print(f"\n  Retrying {len(missing)} missing RICs in {n_micro} "
              f"micro-batches of {retry_batch_size}...")
        for i in range(0, len(missing), retry_batch_size):
            micro = missing[i:i + retry_batch_size]
            micro_num = i // retry_batch_size
            df = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df = _fetch_batch(rd, micro, start, end)
                    if df is not None:
                        break
                    time.sleep(RETRY_BACKOFF * attempt)
                except Exception:
                    time.sleep(RETRY_BACKOFF * attempt)

            if df is not None:
                batch_rics = _extract_rics_from_batch(df)
                fetched_rics.update(batch_rics)
                all_data.append(df)
            else:
                print(f"    Micro-batch {micro_num}: failed "
                      f"({', '.join(micro[:3])}{'...' if len(micro) > 3 else ''})")
            time.sleep(BATCH_DELAY)

        recovered = len(fetched_rics) - (len(requested_rics) - len(missing))
        print(f"  Recovered {recovered}/{len(missing)} RICs in Pass 2")

    return all_data, fetched_rics


def reshape_to_standard(combined: pd.DataFrame) -> pd.DataFrame:
    """Reshape LSEG MultiIndex DataFrame to flat OHLCV + turnover format."""
    if not isinstance(combined.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from rd.get_history")

    instruments = combined.columns.get_level_values(0).unique().tolist()
    records = []

    for instrument in instruments:
        if instrument in ('Date', 'Instrument', 'index'):
            continue
        try:
            inst_data = combined[instrument].copy()
            inst_data = inst_data.rename(columns=COLUMN_MAPPING)
            inst_data['kdcode'] = instrument
            inst_data['dt'] = combined.index
            records.append(inst_data)
        except Exception as e:
            print(f"  Skipping {instrument}: {e}")

    df = pd.concat(records, ignore_index=True)

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after reshape: {missing}")

    df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['close'])
    df = df.drop_duplicates(subset=['kdcode', 'dt'], keep='first')
    df['turnover'] = df['volume'] * df['close']
    df = df[STANDARD_COLS]
    df = df.sort_values(['kdcode', 'dt']).reset_index(drop=True)
    return df


def validate(master_df: pd.DataFrame, new_dates: list, expected_stock_count: int):
    """Run basic data quality checks after refresh."""
    issues = []

    latest = master_df['dt'].max()
    today = datetime.now().strftime('%Y-%m-%d')

    if not new_dates:
        issues.append(f"WARNING: No new dates were added (latest in CSV: {latest})")
    else:
        print(f"  New dates added: {len(new_dates)} ({min(new_dates)} to {max(new_dates)})")

    stock_count = master_df['kdcode'].nunique()
    if stock_count < expected_stock_count * 0.8:
        issues.append(
            f"WARNING: Only {stock_count} stocks in data "
            f"(expected ~{expected_stock_count})"
        )

    latest_day = master_df[master_df['dt'] == latest]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        null_pct = latest_day[col].isna().mean()
        if null_pct > 0.05:
            issues.append(f"WARNING: {col} has {null_pct:.1%} nulls on {latest}")

    if issues:
        print("\n  VALIDATION ISSUES:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  Validation passed: all checks OK")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Incremental LSEG data refresh.")
    parser.add_argument(
        "--csv",
        default="data/raw/market/sp500_2019_universe_data_through_2026.csv",
        help="Path to master CSV to update",
    )
    parser.add_argument(
        "--constituents",
        default="data/raw/constituents/sp500_constituents_2019.csv",
        help="Path to constituents CSV with RIC codes",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for fetch (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate but do not write to CSV",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_COVERAGE_DEFAULT,
        help=(
            f"Minimum fraction of RICs that must return data "
            f"(default: {MIN_COVERAGE_DEFAULT}). "
            f"Exits with error if coverage is below this threshold."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-fetch from the last date already in the CSV instead of "
            "last_date + 1. Use this to repair an incomplete prior fetch."
        ),
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    constituents_path = PROJECT_ROOT / args.constituents

    if not csv_path.exists():
        print(f"ERROR: Master CSV not found: {csv_path}")
        sys.exit(1)
    if not constituents_path.exists():
        print(f"ERROR: Constituents CSV not found: {constituents_path}")
        sys.exit(1)

    print("=" * 70)
    print("  LSEG Data Refresh")
    print("=" * 70)

    last_date = get_last_date(str(csv_path))
    if args.force:
        fetch_start = last_date
    else:
        fetch_start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    fetch_end = args.end or datetime.now().strftime('%Y-%m-%d')

    api_start = (
        pd.Timestamp(fetch_start) - timedelta(days=FETCH_LOOKBACK_DAYS)
    ).strftime('%Y-%m-%d')

    print(f"  Master CSV:     {csv_path}")
    print(f"  Last date:      {last_date}")
    if args.force:
        print(f"  Mode:           --force (re-fetching from {last_date})")
    print(f"  Target range:   {fetch_start} to {fetch_end}")
    print(f"  API range:      {api_start} to {fetch_end} "
          f"(+{FETCH_LOOKBACK_DAYS}d lookback for LSEG reliability)")

    if fetch_start > fetch_end:
        print("  Data is already up to date. Nothing to fetch.")
        return

    rics = load_rics(str(constituents_path))
    if BENCHMARK_RIC not in rics:
        rics.append(BENCHMARK_RIC)
    print(f"  Constituents:   {len(rics)} RICs from {constituents_path.name}")

    import refinitiv.data as rd
    print("\n  Connecting to Refinitiv Workspace...")
    rd.open_session()

    try:
        raw_batches, fetched_rics = download_incremental(
            rd, rics, api_start, fetch_end,
        )
    finally:
        rd.close_session()
        print("  Disconnected from Refinitiv.")

    if not raw_batches:
        print("  No new data returned from LSEG. Market may be closed.")
        return

    coverage = len(fetched_rics) / len(rics) if rics else 0
    print(f"\n  Coverage: {len(fetched_rics)}/{len(rics)} RICs ({coverage:.1%})")

    still_missing = sorted(set(rics) - fetched_rics)
    if still_missing:
        print(f"  Missing RICs ({len(still_missing)}): "
              f"{', '.join(still_missing[:20])}"
              f"{'...' if len(still_missing) > 20 else ''}")

    if coverage < args.min_coverage:
        print(
            f"\n  ERROR: Coverage {coverage:.1%} is below minimum "
            f"{args.min_coverage:.0%}. Aborting to avoid corrupt data.\n"
            f"  Possible causes:\n"
            f"    - Market data not yet available (run after market close)\n"
            f"    - LSEG Workspace not running / session issues\n"
            f"    - API rate limits\n"
            f"  Re-run the pipeline once the data is available, or use\n"
            f"  --min-coverage 0 to force the update."
        )
        sys.exit(1)

    print(f"  Combining {len(raw_batches)} batches...")
    combined = pd.concat(raw_batches)
    all_data = reshape_to_standard(combined)
    all_api_dates = sorted(all_data['dt'].unique().tolist())
    print(f"  API returned: {len(all_data):,} rows across {len(all_api_dates)} dates")

    new_data = all_data[all_data['dt'] >= fetch_start].copy()
    new_dates = sorted(new_data['dt'].unique().tolist())
    if len(all_api_dates) > len(new_dates):
        print(f"  Filtered to target range: {len(new_data):,} rows across "
              f"{len(new_dates)} dates (dropped {len(all_api_dates) - len(new_dates)} "
              f"lookback dates)")
    print(f"  New rows: {len(new_data):,} across {len(new_dates)} dates")

    if args.dry_run:
        print("\n  DRY RUN: not writing to CSV.")
        print(f"  Would append {len(new_data):,} rows.")
        print(f"  New date range: {min(new_dates)} to {max(new_dates)}")
        return

    print(f"\n  Reading existing master CSV...")
    master_df = pd.read_csv(str(csv_path))
    before_rows = len(master_df)

    merged = pd.concat([master_df, new_data], ignore_index=True)
    merged = merged.drop_duplicates(subset=['kdcode', 'dt'], keep='last')
    merged = merged.sort_values(['kdcode', 'dt']).reset_index(drop=True)

    added_rows = len(merged) - before_rows

    validate(merged, new_dates, expected_stock_count=len(rics))

    merged.to_csv(str(csv_path), index=False)
    print(f"\n  SAVED: {csv_path}")
    print(f"    Rows before: {before_rows:,}")
    print(f"    Rows after:  {len(merged):,}")
    print(f"    Rows added:  {added_rows:,}")
    print(f"    Date range:  {merged['dt'].min()} to {merged['dt'].max()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
