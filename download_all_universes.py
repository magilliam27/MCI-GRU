"""
Download historical OHLCV data for all S&P 500 constituent universes.

Reads each sp500_constituents_YEAR.csv file and downloads daily price data
from Refinitiv, then reshapes and saves to sp500_YEAR_universe_data.csv.

Same reshape logic as scratch1.py / lseg_loader.py (proven to work).

Requires: Refinitiv Workspace desktop app to be running.
"""

import refinitiv.data as rd
import pandas as pd
from tqdm import tqdm
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Each entry: (constituent CSV, output CSV, data start date, data end date)
# Start date is 1 year before the constituent year (buffer for correlation).
# End date extends through 2025 for full coverage.
UNIVERSES = [
    {
        'constituents_csv': 'sp500_constituents_2016.csv',
        'output_csv': 'sp500_2016_universe_data.csv',
        'start': '2015-01-01',
        'end': '2025-12-31',
        'label': '2016',
    },
    {
        'constituents_csv': 'sp500_constituents_2017.csv',
        'output_csv': 'sp500_2017_universe_data.csv',
        'start': '2016-01-01',
        'end': '2025-12-31',
        'label': '2017',
    },
    {
        'constituents_csv': 'sp500_constituents_2018.csv',
        'output_csv': 'sp500_2018_universe_data.csv',
        'start': '2017-01-01',
        'end': '2025-12-31',
        'label': '2018',
    },
]

# Skip universes that already have a saved output file
SKIP_EXISTING = True

# Refinitiv API settings
BATCH_SIZE = 50        # RICs per API call (avoid rate limits)
BATCH_DELAY = 0.5      # seconds between batches

# Column name mapping from LSEG field names -> standard OHLCV
# (covers all known field name variants returned by rd.get_history)
COLUMN_MAPPING = {
    'MKT_OPEN': 'open', 'MKT_HIGH': 'high', 'MKT_LOW': 'low',
    'TRDPRC_1': 'close', 'ACVOL_UNS': 'volume',
    'OPEN_PRC': 'open', 'HIGH_1': 'high', 'LOW_1': 'low', 'HST_CLOSE': 'close',
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume',
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
}


def load_rics(csv_path):
    """Load RIC codes from a constituents CSV file."""
    df = pd.read_csv(csv_path)
    rics = df['Instrument'].dropna().tolist()
    return rics


def download_history(rics, start, end):
    """
    Download daily OHLCV data from Refinitiv in batches.

    Returns a list of raw DataFrames (one per successful batch).
    Each batch calls rd.get_history() which returns a MultiIndex DataFrame
    with (Instrument, Field) columns and a Date index.
    """
    all_data = []

    for i in tqdm(range(0, len(rics), BATCH_SIZE), desc="  Downloading"):
        batch = rics[i:i + BATCH_SIZE]

        try:
            df = rd.get_history(
                universe=batch,
                start=start,
                end=end,
                interval='1D'
            )
            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    Batch {i // BATCH_SIZE} failed: {e}")

        time.sleep(BATCH_DELAY)

    return all_data


def reshape_to_standard(combined):
    """
    Reshape Refinitiv MultiIndex DataFrame to flat OHLCV format.

    Input:  MultiIndex columns (Instrument, Field), Date index
    Output: Flat DataFrame with [kdcode, dt, open, high, low, close, volume, turnover]

    This is the same reshape logic used in scratch1.py and lseg_loader.py.
    """
    if not isinstance(combined.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from rd.get_history")

    instruments = combined.columns.get_level_values(0).unique().tolist()
    fields = combined.columns.get_level_values(1).unique().tolist()
    print(f"  Instruments found: {len(instruments)}")
    print(f"  Fields: {fields}")

    records = []

    for instrument in tqdm(instruments, desc="  Reshaping"):
        # Skip metadata-like column names that aren't real instruments
        if instrument in ['Date', 'Instrument', 'index']:
            continue
        try:
            # Extract this instrument's data: columns become the field names
            instrument_data = combined[instrument].copy()

            # Map LSEG field names to standard open/high/low/close/volume
            instrument_data = instrument_data.rename(columns=COLUMN_MAPPING)

            # Add stock identifier (full RIC, e.g. "AAPL.OQ")
            instrument_data['kdcode'] = instrument

            # Add date from the DataFrame index
            instrument_data['dt'] = combined.index

            records.append(instrument_data)
        except Exception as e:
            print(f"    Skipping {instrument}: {e}")
            continue

    final_df = pd.concat(records, ignore_index=True)

    # Verify required OHLCV columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in final_df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Available columns: {final_df.columns.tolist()}")
        raise KeyError(f"Missing required columns: {missing}")

    # Format date as YYYY-MM-DD string
    final_df['dt'] = pd.to_datetime(final_df['dt']).dt.strftime('%Y-%m-%d')

    # Drop rows where close price is NaN (no trade that day)
    final_df = final_df.dropna(subset=['close'])

    # Remove duplicate (stock, date) rows
    before = len(final_df)
    final_df = final_df.drop_duplicates(subset=['kdcode', 'dt'], keep='first')
    dropped = before - len(final_df)
    if dropped > 0:
        print(f"  Dropped {dropped} duplicate rows")

    # Compute turnover (volume * close price)
    final_df['turnover'] = final_df['volume'] * final_df['close']

    # Keep only the standard columns in order
    final_df = final_df[['kdcode', 'dt', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

    # Sort by stock then date
    final_df = final_df.sort_values(['kdcode', 'dt']).reset_index(drop=True)

    return final_df


def process_universe(universe_config):
    """
    Full pipeline for one universe: load RICs -> download -> reshape -> save.
    """
    csv_path = universe_config['constituents_csv']
    output_path = universe_config['output_csv']
    start = universe_config['start']
    end = universe_config['end']
    label = universe_config['label']

    print(f"\n{'=' * 70}")
    print(f"  UNIVERSE: {label}")
    print(f"  Constituents: {csv_path}")
    print(f"  Date range:   {start} to {end}")
    print(f"  Output:       {output_path}")
    print(f"{'=' * 70}")

    # Check if output already exists
    if SKIP_EXISTING and os.path.exists(output_path):
        existing = pd.read_csv(output_path, nrows=5)
        print(f"  SKIPPING - output file already exists ({output_path})")
        print(f"  Set SKIP_EXISTING = False to re-download.")
        return None

    # Check constituent file exists
    if not os.path.exists(csv_path):
        print(f"  ERROR: Constituent file not found: {csv_path}")
        return None

    # Step 1: Load RICs
    rics = load_rics(csv_path)
    print(f"  Loaded {len(rics)} RICs from {csv_path}")

    # Step 2: Download historical data
    print(f"  Downloading data from {start} to {end}...")
    raw_batches = download_history(rics, start, end)

    if not raw_batches:
        print(f"  ERROR: No data downloaded for {label}")
        return None

    # Step 3: Combine and reshape
    print(f"  Combining {len(raw_batches)} batches and reshaping...")
    combined = pd.concat(raw_batches)
    final_df = reshape_to_standard(combined)

    # Step 4: Save
    final_df.to_csv(output_path, index=False)
    print(f"\n  SAVED: {output_path}")
    print(f"    Rows:       {len(final_df):,}")
    print(f"    Stocks:     {final_df['kdcode'].nunique()}")
    print(f"    Date range: {final_df['dt'].min()} to {final_df['dt'].max()}")
    print(f"    Columns:    {final_df.columns.tolist()}")
    print(f"    Sample:")
    print(final_df.head(3).to_string(index=False))

    return final_df


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Connecting to Refinitiv Workspace...")
    rd.open_session()

    results = {}
    for universe_config in UNIVERSES:
        try:
            df = process_universe(universe_config)
            results[universe_config['label']] = df
        except Exception as e:
            print(f"\n  FAILED for {universe_config['label']}: {e}")
            results[universe_config['label']] = None

    rd.close_session()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for label, df in results.items():
        if df is not None:
            print(f"  {label}: {len(df):>10,} rows, {df['kdcode'].nunique():>4} stocks")
        else:
            skipped = os.path.exists(f'sp500_{label}_universe_data.csv')
            status = "SKIPPED (already exists)" if skipped else "FAILED"
            print(f"  {label}: {status}")
    print(f"{'=' * 70}")
    print("Done!")
