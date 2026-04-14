#!/usr/bin/env python
"""
Build Historical Stock Universe

Creates a universe of top 500 stocks by market cap as of a specific date.
Uses yfinance to get historical prices and current shares outstanding.

Usage:
    python build_historical_universe.py --date 2019-01-02 --output sp500_2019.csv
    python build_historical_universe.py --date 2019-01-02 --top_n 500 --download_data
"""

import argparse
import time
from datetime import timedelta

import pandas as pd

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Error: yfinance required. Install with: pip install yfinance")

from tqdm import tqdm


# Broad universe of US stocks to screen from
# This combines multiple sources to get comprehensive coverage
def get_broad_universe() -> list[str]:
    """
    Get a broad list of US stocks to screen for top market cap.
    Combines Russell 1000, S&P 500, and other large/mid cap stocks.
    """

    # Current S&P 500 (will include many that were there in 2019)
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        tables = pd.read_html(sp500_url)
        sp500_df = tables[0]
        sp500_tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  Loaded {len(sp500_tickers)} S&P 500 tickers from Wikipedia")
    except Exception as e:
        print(f"  Warning: Could not load S&P 500 from Wikipedia: {e}")
        sp500_tickers = []

    # Historical S&P 500 changes - stocks removed since 2019
    # These are major ones that were in S&P 500 in 2019 but removed since
    historical_sp500 = [
        # Removed 2019-2024 (major ones)
        "CELG",  # Celgene (acquired by Bristol-Myers)
        "RTN",  # Raytheon (merged with UTX to form RTX)
        "UTX",  # United Technologies (merged to form RTX)
        "ETFC",  # E*TRADE (acquired by Morgan Stanley)
        "FLIR",  # FLIR Systems (acquired by Teledyne)
        "CXO",  # Concho Resources (acquired by ConocoPhillips)
        "TIF",  # Tiffany (acquired by LVMH)
        "NBL",  # Noble Energy (acquired by Chevron)
        "VAR",  # Varian Medical (acquired by Siemens)
        "MXIM",  # Maxim Integrated (acquired by Analog Devices)
        "AGN",  # Allergan (acquired by AbbVie)
        "XLNX",  # Xilinx (acquired by AMD)
        "INFO",  # IHS Markit (merged with S&P Global)
        "PBCT",  # People's United Financial (acquired by M&T Bank)
        "KSU",  # Kansas City Southern (acquired by CP Rail)
        "ALXN",  # Alexion (acquired by AstraZeneca)
        "ATVI",  # Activision Blizzard (acquired by Microsoft)
        "CTXS",  # Citrix (taken private)
        "CERN",  # Cerner (acquired by Oracle)
    ]

    # Additional large caps that might not be in current S&P 500
    additional_large_caps = [
        # Large tech/growth
        "GOOGL",
        "GOOG",
        "AMZN",
        "AAPL",
        "MSFT",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "ADBE",
        "CRM",
        "ORCL",
        "INTC",
        "AMD",
        "AVGO",
        "QCOM",
        # Finance
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "BLK",
        "SCHW",
        "AXP",
        # Healthcare
        "JNJ",
        "UNH",
        "PFE",
        "MRK",
        "ABBV",
        "TMO",
        "ABT",
        "LLY",
        "BMY",
        # Consumer
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        # Industrial
        "BA",
        "CAT",
        "GE",
        "HON",
        "UPS",
        "UNP",
        "RTX",
        "LMT",
        "MMM",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PXD",
        "OXY",
        # Telecom/Media
        "VZ",
        "T",
        "CMCSA",
        "DIS",
        "NFLX",
    ]

    # Combine all tickers
    all_tickers = list(set(sp500_tickers + historical_sp500 + additional_large_caps))

    # Clean tickers (remove any with special characters except hyphen)
    all_tickers = [t for t in all_tickers if t and isinstance(t, str)]
    all_tickers = [t.strip().upper() for t in all_tickers]

    print(f"  Total universe: {len(all_tickers)} tickers")

    return all_tickers


def get_market_cap_at_date(
    tickers: list[str], target_date: str, batch_size: int = 50
) -> pd.DataFrame:
    """
    Get market cap for each ticker at a specific date.

    Uses: price at target_date * current shares outstanding
    (Shares outstanding changes slowly for large caps, so this is a reasonable approximation)
    """

    target_dt = pd.to_datetime(target_date)
    start_date = (target_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (target_dt + timedelta(days=7)).strftime("%Y-%m-%d")

    results = []
    failed = []

    print(f"\nFetching market cap data for {len(tickers)} tickers...")
    print(f"  Target date: {target_date}")
    print(f"  Fetching prices from {start_date} to {end_date}")

    # Process in batches
    for i in tqdm(range(0, len(tickers), batch_size), desc="Processing batches"):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)

        try:
            # Download historical data for batch
            data = yf.download(
                batch_str, start=start_date, end=end_date, progress=False, threads=True
            )

            # Get individual ticker info for shares outstanding
            for ticker in batch:
                try:
                    # Get price at target date
                    if len(batch) == 1:
                        prices = data["Close"]
                    else:
                        if ticker not in data["Close"].columns:
                            failed.append(ticker)
                            continue
                        prices = data["Close"][ticker]

                    # Find price closest to target date
                    prices = prices.dropna()
                    if len(prices) == 0:
                        failed.append(ticker)
                        continue

                    # Get price closest to target date
                    price_date = prices.index[prices.index <= target_dt]
                    if len(price_date) == 0:
                        price_date = prices.index[0]
                        price = prices.iloc[0]
                    else:
                        price_date = price_date[-1]
                        price = prices.loc[price_date]

                    # Get shares outstanding
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    shares = info.get("sharesOutstanding")

                    if shares is None:
                        # Try alternative field
                        shares = info.get("impliedSharesOutstanding")

                    if shares is None or shares == 0:
                        failed.append(ticker)
                        continue

                    market_cap = price * shares

                    results.append(
                        {
                            "ticker": ticker,
                            "price": price,
                            "price_date": price_date,
                            "shares_outstanding": shares,
                            "market_cap": market_cap,
                            "name": info.get("shortName", ticker),
                        }
                    )

                except Exception:
                    failed.append(ticker)
                    continue

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"\n  Batch failed: {e}")
            failed.extend(batch)
            continue

    print(f"\n  Successfully processed: {len(results)} tickers")
    print(f"  Failed: {len(failed)} tickers")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("market_cap", ascending=False)
        return df
    else:
        return pd.DataFrame()


def download_historical_data(
    tickers: list[str], start_date: str, end_date: str, output_file: str
) -> pd.DataFrame:
    """
    Download full historical data for selected tickers.
    """

    print(f"\nDownloading historical data for {len(tickers)} tickers...")
    print(f"  Period: {start_date} to {end_date}")

    all_data = []

    # Download in batches
    batch_size = 50
    for i in tqdm(range(0, len(tickers), batch_size), desc="Downloading"):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)

        try:
            data = yf.download(
                batch_str,
                start=start_date,
                end=end_date,
                progress=False,
                threads=True,
                group_by="ticker",
            )

            # Reshape to long format
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data.copy()
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            continue
                        ticker_data = data[ticker].copy()

                    ticker_data = ticker_data.reset_index()
                    ticker_data["ticker"] = ticker
                    ticker_data.columns = [
                        c.lower() if isinstance(c, str) else c for c in ticker_data.columns
                    ]

                    # Rename columns to match expected format
                    ticker_data = ticker_data.rename(
                        columns={"date": "dt", "adj close": "adj_close", "ticker": "kdcode"}
                    )

                    all_data.append(ticker_data)

                except Exception:
                    continue

            time.sleep(0.1)

        except Exception as e:
            print(f"\n  Batch download failed: {e}")
            continue

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)

        # Ensure required columns exist
        if "turnover" not in full_df.columns:
            # Calculate turnover as volume * close price (approximate)
            full_df["turnover"] = full_df["volume"] * full_df["close"]

        # Format date
        full_df["dt"] = pd.to_datetime(full_df["dt"]).dt.strftime("%Y-%m-%d")

        # Save to CSV
        full_df.to_csv(output_file, index=False)
        print(f"\n  Saved {len(full_df)} rows to {output_file}")
        print(f"  Date range: {full_df['dt'].min()} to {full_df['dt'].max()}")
        print(f"  Stocks: {full_df['kdcode'].nunique()}")

        return full_df
    else:
        print("  No data downloaded!")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Build historical stock universe by market cap")

    parser.add_argument(
        "--date",
        type=str,
        default="2019-01-02",
        help="Target date for market cap ranking (default: 2019-01-02)",
    )

    parser.add_argument(
        "--top_n", type=int, default=500, help="Number of top stocks to select (default: 500)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="top500_universe.csv",
        help="Output file for universe list (default: top500_universe.csv)",
    )

    parser.add_argument(
        "--download_data",
        action="store_true",
        help="Download full historical data for selected stocks",
    )

    parser.add_argument(
        "--data_start",
        type=str,
        default="2016-01-01",
        help="Start date for historical data download (default: 2016-01-01)",
    )

    parser.add_argument(
        "--data_end",
        type=str,
        default="2025-12-31",
        help="End date for historical data download (default: 2025-12-31)",
    )

    parser.add_argument(
        "--data_output",
        type=str,
        default="sp500_historical.csv",
        help="Output file for historical data (default: sp500_historical.csv)",
    )

    args = parser.parse_args()

    if not HAS_YFINANCE:
        print("Please install yfinance: pip install yfinance")
        return

    print("=" * 80)
    print("BUILDING HISTORICAL STOCK UNIVERSE")
    print("=" * 80)
    print(f"Target date: {args.date}")
    print(f"Top N stocks: {args.top_n}")
    print("=" * 80)

    # Step 1: Get broad universe
    print("\nStep 1: Getting broad stock universe...")
    tickers = get_broad_universe()

    # Step 2: Get market cap at target date
    print("\nStep 2: Calculating market caps...")
    market_caps = get_market_cap_at_date(tickers, args.date)

    if len(market_caps) == 0:
        print("Error: No market cap data retrieved")
        return

    # Step 3: Select top N
    print(f"\nStep 3: Selecting top {args.top_n} by market cap...")
    top_stocks = market_caps.head(args.top_n)

    # Save universe
    top_stocks.to_csv(args.output, index=False)
    print(f"\n  Universe saved to: {args.output}")

    # Print top 20
    print(f"\n  Top 20 by market cap as of {args.date}:")
    print("  " + "-" * 60)
    for _i, row in top_stocks.head(20).iterrows():
        mcap_b = row["market_cap"] / 1e9
        print(f"  {row['ticker']:6s} {row['name'][:30]:30s} ${mcap_b:>8.1f}B")
    print("  " + "-" * 60)

    # Step 4: Download historical data if requested
    if args.download_data:
        print("\nStep 4: Downloading historical data...")
        selected_tickers = top_stocks["ticker"].tolist()

        download_historical_data(
            tickers=selected_tickers,
            start_date=args.data_start,
            end_date=args.data_end,
            output_file=args.data_output,
        )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
