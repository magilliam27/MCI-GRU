"""
Build sp500.py-compatible dataset from yfinance.

This script downloads S&P 500 stock data and prepares it for the MCI-GRU model.

Paper features (6 total):
    1. close  - Closing price
    2. open   - Opening price  
    3. high   - High price
    4. low    - Low price
    5. volume - Trading volume
    6. turnover - Close * Volume (computed from close and volume)

Output CSV columns:
    kdcode, dt, open, high, low, close, volume, prev_close, turnover
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def load_tickers_from_csv(filepath):
    df = pd.read_csv(filepath)
    for col in ["ticker", "symbol", "Ticker", "Symbol", "TICKER", "SYMBOL"]:
        if col in df.columns:
            return (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .tolist()
            )
    return (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )


def download_sp500_dataframe(tickers, start_date, end_date, min_days=200):
    """
    Download OHLCV data for all tickers and prepare for MCI-GRU model.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data download
        end_date: End date for data download
        min_days: Minimum number of trading days required
    
    Returns:
        combined: DataFrame with all stock data
        failed: List of tickers that failed to download
    """
    rows = []
    failed = []
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < min_days:
                failed.append(ticker)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df["kdcode"] = ticker
            df["dt"] = df.index.strftime("%Y-%m-%d")
            df["prev_close"] = df["Close"].shift(1)
            
            # Paper feature #6: Turnover = Close * Volume
            df["turnover"] = df["Close"] * df["Volume"]
            
            df = df.dropna()
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            rows.append(
                df[
                    ["kdcode", "dt", "open", "high", "low", "close", "volume", "prev_close", "turnover"]
                ]
            )
        except Exception as e:
            failed.append(ticker)

    if not rows:
        raise ValueError("No data downloaded for provided tickers/date range.")

    combined = pd.concat(rows, ignore_index=True)
    return combined, failed


def enforce_numeric(df):
    """
    Ensure all numeric columns are properly typed.
    
    Required columns for MCI-GRU paper:
        - open, high, low, close, volume (OHLCV)
        - prev_close (for return calculation)
        - turnover (Close * Volume, paper feature #6)
    """
    df = df.loc[:, ~df.columns.duplicated()]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x not in (None, "")]) for col in df.columns
        ]
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Paper requires 6 features: close, open, high, low, volume, turnover
    required = ["open", "high", "low", "close", "volume", "prev_close", "turnover"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after normalization: {missing}")

    for col in required:
        if not isinstance(df[col], pd.DataFrame):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required)


def clean_dataset(df):
    """
    Basic data cleaning without strict alignment.
    The sp500.py code handles varying sequence counts.
    """
    print("\nCleaning dataset...")
    
    initial_rows = len(df)
    initial_stocks = df["kdcode"].nunique()
    initial_dates = df["dt"].nunique()
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Sort by date then stock for consistent ordering
    df = df.sort_values(["dt", "kdcode"]).reset_index(drop=True)
    
    final_stocks = df["kdcode"].nunique()
    final_dates = df["dt"].nunique()
    
    print(f"  Stocks: {initial_stocks} -> {final_stocks}")
    print(f"  Dates: {initial_dates} -> {final_dates}")
    print(f"  Rows: {initial_rows:,} -> {len(df):,}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build sp500.py-compatible dataset from yfinance."
    )
    parser.add_argument("--tickers", required=True, help="Path to ticker CSV.")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--min-days", type=int, default=200)
    parser.add_argument("--out", default="sp500_yf_download.csv")
    args = parser.parse_args()

    tickers = load_tickers_from_csv(args.tickers)
    df, failed = download_sp500_dataframe(
        tickers, args.start, args.end, min_days=args.min_days
    )
    df = enforce_numeric(df)
    
    # Basic cleaning (no strict alignment needed)
    df = clean_dataset(df)

    df.to_csv(args.out, index=False)
    print(f"\nSaved dataset to {args.out} ({len(df):,} rows)")
    if failed:
        print(f"Failed/insufficient data for {len(failed)} tickers")


if __name__ == "__main__":
    main()
