"""
Credit spread feature engineering for MCI-GRU.

Market-wide IG/HY option-adjusted spreads merged from FRED (via fred_loader).
Broadcast to all stocks by merge on dt, following the VIX pattern.
"""

import pandas as pd

# Credit spread feature columns (market-wide, broadcast to all stocks)
CREDIT_FEATURES = [
    "ig_spread",
    "hy_spread",
    "ig_spread_change",
    "hy_spread_change",
    "ig_spread_zscore",
    "hy_spread_zscore",
    "credit_spread_diff",
]

# Rolling window for z-score (approx 3 months)
CREDIT_ZSCORE_WINDOW = 63


def add_credit_features(
    df: pd.DataFrame,
    credit_df: pd.DataFrame,
    zscore_window: int = CREDIT_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Merge credit spread data and add derived features.

    Credit spreads are market-level; the same values are replicated for all
    stocks on each date (merge on dt).

    Features added:
        - ig_spread: Investment grade OAS (basis points)
        - hy_spread: High yield OAS (basis points)
        - ig_spread_change: Daily pct change in IG OAS
        - hy_spread_change: Daily pct change in HY OAS
        - ig_spread_zscore: Rolling z-score of IG spread
        - hy_spread_zscore: Rolling z-score of HY spread
        - credit_spread_diff: HY - IG (risk premium / risk appetite)

    Args:
        df: DataFrame with stock data (must have 'dt' column)
        credit_df: DataFrame with columns [dt, ig_spread, hy_spread]
        zscore_window: Rolling window for z-score (default 63 days)

    Returns:
        DataFrame with credit features merged
    """
    print("Merging credit spread features...")
    df = df.copy()
    credit = credit_df.copy()

    # Normalize dt for merging
    if "dt" not in credit.columns and credit.index.name in ("dt", "Date", "date"):
        credit = credit.reset_index()
        credit = credit.rename(columns={credit.columns[0]: "dt"})
    credit["dt"] = pd.to_datetime(credit["dt"]).dt.strftime("%Y-%m-%d")
    credit = credit.sort_values("dt")

    # Derived features on the credit series
    credit["ig_spread_change"] = credit["ig_spread"].pct_change().fillna(0)
    credit["hy_spread_change"] = credit["hy_spread"].pct_change().fillna(0)
    ig_std = credit["ig_spread"].rolling(zscore_window, min_periods=1).std()
    hy_std = credit["hy_spread"].rolling(zscore_window, min_periods=1).std()
    credit["ig_spread_zscore"] = (
        credit["ig_spread"] - credit["ig_spread"].rolling(zscore_window, min_periods=1).mean()
    ) / (ig_std + 1e-8)
    credit["hy_spread_zscore"] = (
        credit["hy_spread"] - credit["hy_spread"].rolling(zscore_window, min_periods=1).mean()
    ) / (hy_std + 1e-8)
    credit["ig_spread_zscore"] = credit["ig_spread_zscore"].fillna(0).clip(-3, 3)
    credit["hy_spread_zscore"] = credit["hy_spread_zscore"].fillna(0).clip(-3, 3)
    credit["credit_spread_diff"] = credit["hy_spread"] - credit["ig_spread"]

    merge_cols = ["dt"] + CREDIT_FEATURES
    credit_merge = credit[merge_cols]

    # Merge with stock data (same credit values for all stocks on each date)
    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    df = df.merge(credit_merge, on="dt", how="left")

    # Forward-fill missing (e.g. weekends/holidays in stock data)
    # and log if we still needed fallback fills after merge.
    missing_before = {
        col: int(df[col].isna().sum())
        for col in CREDIT_FEATURES
        if col in df.columns
    }
    for col in CREDIT_FEATURES:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(0)

    total_missing_before = sum(missing_before.values())
    if total_missing_before > 0:
        print(
            f"  WARNING: credit merge had {total_missing_before} missing values before fallback fill; "
            "verify date alignment in credit_df vs stock data."
        )

    print(f"  Added credit features: {CREDIT_FEATURES}")
    if "ig_spread" in df.columns and df["ig_spread"].notna().any():
        print(f"  IG spread range: {df['ig_spread'].min():.1f} to {df['ig_spread'].max():.1f} bps")
    return df
