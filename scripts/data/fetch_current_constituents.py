"""
Fetch current S&P 500 constituents from LSEG and save to CSV.

Queries the live chain RIC (0#.SPX) to get today's index membership.
Saves in the same format as existing constituent files so it can be
used directly by download_all_universes.py.

Requires: Refinitiv Workspace desktop app to be running.

Usage:
    python scripts/data/fetch_current_constituents.py
    python scripts/data/fetch_current_constituents.py --universe russell1000
    python scripts/data/fetch_current_constituents.py --output data/raw/constituents/sp500_constituents_current.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Fetch current index constituents from LSEG.")
    parser.add_argument(
        "--universe",
        default="sp500",
        help="Universe name from universes.py (default: sp500)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/raw/constituents/<universe>_constituents_current.csv)",
    )
    args = parser.parse_args()

    from mci_gru.data.lseg_loader import LSEGLoader
    from mci_gru.data.universes import get_universe_info

    info = get_universe_info(args.universe)
    print(f"Fetching current constituents for {info['name']}...")

    loader = LSEGLoader()
    loader.connect()

    try:
        import refinitiv.data as rd

        chain_ric = info["chain_ric"]
        constituents = rd.get_data(
            universe=[chain_ric],
            fields=["TR.RIC", "TR.CommonName", "TR.CompanyMarketCap"],
        )

        if constituents is None or len(constituents) == 0:
            raise RuntimeError(f"No constituents returned for {chain_ric}")

        constituents = constituents.rename(
            columns={
                "Instrument": "Instrument",
                "RIC": "Constituent RIC",
                "Company Common Name": "Company Common Name",
                "Company Market Cap": "Company Market Cap",
            }
        )

        keep_cols = [
            c
            for c in [
                "Instrument",
                "Constituent RIC",
                "Company Common Name",
                "Company Market Cap",
            ]
            if c in constituents.columns
        ]
        constituents = constituents[keep_cols]

        out_path = args.output
        if out_path is None:
            out_path = f"data/raw/constituents/{args.universe}_constituents_current.csv"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        constituents.to_csv(out_path, index=False)

        print(f"Saved {len(constituents)} constituents to {out_path}")
        print(f"Fetched on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\nFirst 5 entries:")
        print(constituents.head().to_string(index=False))
    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()
