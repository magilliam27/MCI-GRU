"""
Local LSEG regime export for hybrid workflow.

Run on a machine with Refinitiv Workspace (LSEG) access. Exports LSEG-only
regime series (copper required; market and oil optional) to CSV plus a
metadata JSON for Colab reconciliation.

Usage:
  python scripts/export_lseg_regime.py
  python scripts/export_lseg_regime.py --start 2016-01-01 --end 2025-12-31 --output-dir data/raw/market

Output:
  - lseg_regime_export_<start>_<end>.csv  (columns: dt, regime_copper [, regime_market, regime_oil])
  - lseg_regime_export_<start>_<end>_meta.json  (date range, RICs, row counts)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Export LSEG-only regime series for Colab merge.")
    p.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    p.add_argument(
        "--output-dir",
        default="data/raw/market",
        help="Directory for CSV and metadata output",
    )
    p.add_argument(
        "--copper-ric",
        default=".MXCOPPFE",
        help="LSEG RIC for copper",
    )
    p.add_argument(
        "--market-ric",
        default=".SPX",
        help="LSEG RIC for market (optional)",
    )
    p.add_argument(
        "--oil-ric",
        default="CLc1",
        help="LSEG RIC for oil (optional)",
    )
    p.add_argument(
        "--no-market",
        action="store_true",
        help="Skip market series (copper only)",
    )
    p.add_argument(
        "--no-oil",
        action="store_true",
        help="Skip oil series",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from mci_gru.data.lseg_loader import LSEGLoader

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = args.start
    end = args.end
    loader = LSEGLoader()
    loader.connect()

    meta = {
        "start": start,
        "end": end,
        "rics": {},
        "row_counts": {},
        "columns_exported": ["dt"],
    }

    series_list = []
    # Copper (required)
    copper = loader.get_series(args.copper_ric, start, end, "regime_copper")
    if copper is None or len(copper) == 0:
        loader.disconnect()
        raise RuntimeError(f"Failed to fetch copper series {args.copper_ric}")
    series_list.append(copper)
    meta["rics"]["regime_copper"] = args.copper_ric
    meta["row_counts"]["regime_copper"] = len(copper)
    meta["columns_exported"].append("regime_copper")

    if not args.no_market:
        market = loader.get_series(args.market_ric, start, end, "regime_market")
        if market is not None and len(market) > 0:
            series_list.append(market)
            meta["rics"]["regime_market"] = args.market_ric
            meta["row_counts"]["regime_market"] = len(market)
            meta["columns_exported"].append("regime_market")

    if not args.no_oil:
        oil = loader.get_series(args.oil_ric, start, end, "regime_oil")
        if oil is not None and len(oil) > 0:
            series_list.append(oil)
            meta["rics"]["regime_oil"] = args.oil_ric
            meta["row_counts"]["regime_oil"] = len(oil)
            meta["columns_exported"].append("regime_oil")

    loader.disconnect()

    base = series_list[0]
    for df in series_list[1:]:
        base = base.merge(df, on="dt", how="outer")
    base["dt"] = pd.to_datetime(base["dt"])
    base = base.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
    base["dt"] = base["dt"].dt.strftime("%Y-%m-%d")

    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    csv_name = f"lseg_regime_export_{safe_start}_{safe_end}.csv"
    meta_name = f"lseg_regime_export_{safe_start}_{safe_end}_meta.json"
    csv_path = out_dir / csv_name
    meta_path = out_dir / meta_name

    base.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {csv_path} ({len(base)} rows)")
    print(f"Saved {meta_path}")
    print("Columns:", meta["columns_exported"])


if __name__ == "__main__":
    main()
