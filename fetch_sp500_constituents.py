"""
Fetch historical S&P 500 constituents for specific years using Refinitiv Data API.

This mimics the approach used to create sp500_constituents_2019.csv.
Uses rd.get_data() with the .SPX index and a historical date parameter (SDate)
to retrieve the index composition as of that date.

Requires: Refinitiv Workspace desktop app to be running.
"""

import refinitiv.data as rd
import pandas as pd

# Years to fetch constituents for
YEARS = [2016, 2017, 2018]

# Connect to Refinitiv Workspace
rd.open_session()

for year in YEARS:
    # Use end-of-year date to capture the full-year composition.
    # SDate tells Refinitiv to return the index members as of that date,
    # rather than the current (live) composition.
    sdate = f'{year}-12-31'
    print(f"\n{'='*60}")
    print(f"Fetching S&P 500 constituents as of {sdate}...")
    print(f"{'='*60}")

    try:
        # Method 1: Use the chain RIC '0#.SPX' with date parameter.
        # The fields match the columns in sp500_constituents_2019.csv:
        #   - Instrument (auto-included by rd.get_data)
        #   - TR.IndexConstituentRIC  -> "Constituent RIC"
        #   - TR.CommonName           -> "Company Common Name"
        #   - TR.CompanyMarketCap     -> "Company Market Cap"
        constituents = rd.get_data(
            universe=['0#.SPX'],
            fields=['TR.IndexConstituentRIC', 'TR.CommonName', 'TR.CompanyMarketCap'],
            parameters={'SDate': sdate}
        )

        if constituents is None or len(constituents) == 0:
            print(f"  Method 1 returned no data. Trying alternative method...")
            # Method 2: Query the index directly (without 0# chain prefix)
            constituents = rd.get_data(
                universe=['.SPX'],
                fields=['TR.IndexConstituentRIC', 'TR.IndexConstituentName',
                        'TR.CompanyMarketCap'],
                parameters={'SDate': sdate}
            )

        if constituents is not None and len(constituents) > 0:
            output_file = f'sp500_constituents_{year}.csv'
            constituents.to_csv(output_file, index=False)
            print(f"  Saved {len(constituents)} constituents to {output_file}")
            print(f"  Columns: {constituents.columns.tolist()}")
            print(f"  Sample:")
            print(constituents.head(5).to_string(index=False))
        else:
            print(f"  ERROR: No constituents found for {year}")

    except Exception as e:
        print(f"  ERROR fetching {year}: {e}")
        print(f"  Trying fallback approach...")

        # Fallback: use simpler field set
        try:
            constituents = rd.get_data(
                universe=['0#.SPX'],
                fields=['TR.RIC', 'TR.CommonName', 'TR.CompanyMarketCap'],
                parameters={'SDate': sdate}
            )
            if constituents is not None and len(constituents) > 0:
                output_file = f'sp500_constituents_{year}.csv'
                constituents.to_csv(output_file, index=False)
                print(f"  Fallback succeeded! Saved {len(constituents)} to {output_file}")
            else:
                print(f"  Fallback also returned no data for {year}")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")

rd.close_session()
print(f"\nDone! Check for sp500_constituents_YEAR.csv files.")
