import pandas as pd

df = pd.read_csv('sp500_2019_universe_data.csv')
print(f"Total rows: {len(df)}")
print(f"Stocks: {df['kdcode'].nunique()}")
print(f"Dates: {df['dt'].nunique()}")
print(f"Expected rows (stocks x dates): {df['kdcode'].nunique() * df['dt'].nunique()}")

# Check for duplicates
dupes = df.groupby(['kdcode', 'dt']).size()
print(f"\nMax rows per stock/date: {dupes.max()}")
print(f"Mean rows per stock/date: {dupes.mean():.1f}")

# Sample a stock
sample_stock = df['kdcode'].value_counts().index[0]
print(f"\nSample stock '{sample_stock}': {df[df['kdcode']==sample_stock].shape[0]} rows")
print(df[df['kdcode']==sample_stock].head(10))

import pandas as pd

df = pd.read_csv('sp500_2019_universe_data.csv')
print(f"Before: {len(df)} rows")

df = df.drop_duplicates(subset=['kdcode', 'dt'], keep='first')
print(f"After: {len(df)} rows")

df.to_csv('sp500_2019_universe_data.csv', index=False)
print("Saved!")