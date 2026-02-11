# test_fred_credit_spreads.py
from fredapi import Fred
import pandas as pd

# Initialize FRED API
fred = Fred(api_key='c14e4bb7d2c2e2cea539f52ef6672ff8')

# Define date range
start_date = '2016-01-01'
end_date = '2025-12-31'

print("Fetching credit spreads from FRED...\n")

# Fetch IG spread (ICE BofA US Corporate Index OAS)
print("Fetching IG spread (BAMLC0A0CM)...")
ig_spread = fred.get_series('BAMLC0A0CM', observation_start=start_date, observation_end=end_date)
print(f"✓ IG spread: {len(ig_spread)} observations")
print(f"  Latest value: {ig_spread.iloc[-1]:.2f} bps")
print(f"  Date range: {ig_spread.index[0]} to {ig_spread.index[-1]}")

# Fetch HY spread (ICE BofA US High Yield OAS)
print("\nFetching HY spread (BAMLH0A0HYM2)...")
hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start=start_date, observation_end=end_date)
print(f"✓ HY spread: {len(hy_spread)} observations")
print(f"  Latest value: {hy_spread.iloc[-1]:.2f} bps")
print(f"  Date range: {hy_spread.index[0]} to {hy_spread.index[-1]}")

# Combine into DataFrame
df = pd.DataFrame({
    'dt': ig_spread.index,
    'ig_spread': ig_spread.values,
})

# Merge HY spread
hy_df = pd.DataFrame({
    'dt': hy_spread.index,
    'hy_spread': hy_spread.values
})

df = df.merge(hy_df, on='dt', how='outer')

# Calculate HY-IG differential (credit quality spread)
df['hy_ig_diff'] = df['hy_spread'] - df['ig_spread']

# Sort by date
df = df.sort_values('dt').reset_index(drop=True)

# Format date as string
df['dt'] = df['dt'].astype(str)

print("\n" + "="*60)
print("Combined DataFrame:")
print("="*60)
print(df.head(10))
print("\n...")
print(df.tail(10))

print(f"\nShape: {df.shape}")
print(f"\nSummary statistics:")
print(df[['ig_spread', 'hy_spread', 'hy_ig_diff']].describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Save to CSV (optional)
df.to_csv('credit_spreads.csv', index=False)
print("\n✓ Saved to credit_spreads.csv")