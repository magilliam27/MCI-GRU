# generate_datasets.py (run locally with Refinitiv Workspace open)
from mci_gru.data.lseg_loader import LSEGLoader

datasets = {
    "sp500": ("sp500", "2016-01-01", "2025-12-31"),
}

loader = LSEGLoader()
loader.connect()

for name, (universe, start, end) in datasets.items():
    print(f"\nGenerating {name}...")
    df = loader.fetch_universe_data(universe, start, end)

    # Add VIX (handles permission errors gracefully)
    vix_df = loader.get_vix(start, end)
    if vix_df is not None and len(vix_df) > 0:
        df = df.merge(vix_df, on="dt", how="left")
        df["vix"] = df["vix"].ffill().fillna(20)
    else:
        print(f"  Using default VIX value (20) for {name}")
        df["vix"] = 20.0

    df.to_csv(f"{name}_data.csv", index=False)
    print(f"Saved {name}_data.csv ({len(df):,} rows)")

loader.disconnect()
