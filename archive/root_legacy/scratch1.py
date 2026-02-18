"""
Download historical data for S&P 500 constituents from 2019
Uses the same reshape logic as lseg_loader.py (proven to work)
"""

import refinitiv.data as rd
import pandas as pd
from tqdm import tqdm
import time

# Column name mapping from LSEG (same as lseg_loader.py)
COLUMN_MAPPING = {
    'MKT_OPEN': 'open', 'MKT_HIGH': 'high', 'MKT_LOW': 'low',
    'TRDPRC_1': 'close', 'ACVOL_UNS': 'volume',
    'OPEN_PRC': 'open', 'HIGH_1': 'high', 'LOW_1': 'low', 'HST_CLOSE': 'close',
    'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume',
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
}

# Load constituents
constituents = pd.read_csv('sp500_constituents_2019.csv')
rics = constituents['Instrument'].dropna().tolist()
print(f"Loaded {len(rics)} RICs")

# Connect
rd.open_session()

# Download historical data
print("Downloading historical data 2018-01-01 to 2025-12-31...")

all_data = []
batch_size = 50

for i in tqdm(range(0, len(rics), batch_size)):
    batch = rics[i:i+batch_size]

    try:
        df = rd.get_history(
            universe=batch,
            start='2018-01-01',
            end='2025-12-31',
            interval='1D'
        )

        if df is not None and len(df) > 0:
            all_data.append(df)
    except Exception as e:
        print(f"Batch {i//batch_size} failed: {e}")

    time.sleep(0.5)

rd.close_session()

# Combine batches
print("Reshaping data...")
combined = pd.concat(all_data)

# Print raw column info for debugging
print(f"MultiIndex columns: {isinstance(combined.columns, pd.MultiIndex)}")
if isinstance(combined.columns, pd.MultiIndex):
    instruments = combined.columns.get_level_values(0).unique().tolist()
    fields = combined.columns.get_level_values(1).unique().tolist()
    print(f"Instruments: {len(instruments)}")
    print(f"Fields: {fields}")

# Reshape - same logic as lseg_loader.py _reshape_to_standard_format
records = []

if isinstance(combined.columns, pd.MultiIndex):
    instruments = combined.columns.get_level_values(0).unique().tolist()

    for instrument in tqdm(instruments, desc="Reshaping"):
        if instrument in ['Date', 'Instrument', 'index']:
            continue
        try:
            # Extract this instrument's data (columns = field names)
            instrument_data = combined[instrument].copy()

            # Rename LSEG field names to standard OHLCV
            instrument_data = instrument_data.rename(columns=COLUMN_MAPPING)

            # Add stock identifier (keep full RIC like sp500_data.csv does)
            instrument_data['kdcode'] = instrument

            # Add date from index
            instrument_data['dt'] = combined.index

            records.append(instrument_data)
        except Exception as e:
            print(f"  Skipping {instrument}: {e}")
            continue
else:
    raise ValueError("Expected MultiIndex columns from LSEG get_history")

final_df = pd.concat(records, ignore_index=True)

# Verify we have all OHLCV columns
print(f"Columns after reshape: {final_df.columns.tolist()}")

required = ['open', 'high', 'low', 'close', 'volume']
missing = [c for c in required if c not in final_df.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    print("Available columns:", final_df.columns.tolist())
    raise KeyError(f"Missing required columns: {missing}")

# Format date
final_df['dt'] = pd.to_datetime(final_df['dt']).dt.strftime('%Y-%m-%d')

# Drop rows where close is NaN
final_df = final_df.dropna(subset=['close'])

# Drop duplicates
before = len(final_df)
final_df = final_df.drop_duplicates(subset=['kdcode', 'dt'], keep='first')
print(f"Dropped {before - len(final_df)} duplicate rows")

# Add turnover
final_df['turnover'] = final_df['volume'] * final_df['close']

# Keep only needed columns
final_df = final_df[['kdcode', 'dt', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

# Sort
final_df = final_df.sort_values(['kdcode', 'dt']).reset_index(drop=True)

# Save
final_df.to_csv('sp500_2019_universe_data.csv', index=False)
print(f"\nSaved {len(final_df)} rows to sp500_2019_universe_data.csv")
print(f"Stocks: {final_df['kdcode'].nunique()}")
print(f"Date range: {final_df['dt'].min()} to {final_df['dt'].max()}")
print(f"Columns: {final_df.columns.tolist()}")
print(f"\nSample:")
print(final_df.head(5))
