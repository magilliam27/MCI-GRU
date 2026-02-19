"""
Colab-side regime reconciliation: merge FRED series with LSEG export.

Run in Colab after uploading the local LSEG export (e.g. lseg_regime_export_*.csv).
Fetches FRED series with 1-day lag, merges on dt, computes regime_yield_curve
and regime_stock_bond_corr, and saves regime_inputs_reconciled.csv.

Usage from Colab:
  %cd /content/MCI-GRU
  LSEG_REGIME_PATH = "/content/drive/MyDrive/MCI-GRU-Data/lseg_regime_export_20160101_20251231.csv"
  REGIME_START = "2010-01-01"
  REGIME_END = "2025-12-31"
  REGIME_OUTPUT = "data/raw/market/regime_inputs_reconciled.csv"
  %run scripts/colab_regime_reconcile.py

Or copy the logic below into a notebook cell and set the variables there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Defaults (override via env or before %run)
LSEG_REGIME_PATH = os.environ.get("LSEG_REGIME_PATH", "")
REGIME_START = os.environ.get("REGIME_START", "2010-01-01")
REGIME_END = os.environ.get("REGIME_END", "2025-12-31")
REGIME_OUTPUT = os.environ.get("REGIME_OUTPUT", "data/raw/market/regime_inputs_reconciled.csv")
FRED_LAG_DAYS = 1


def main():
    if not LSEG_REGIME_PATH or not Path(LSEG_REGIME_PATH).exists():
        print("LSEG_REGIME_PATH not set or file not found. Skip reconciliation.")
        return

    from mci_gru.data.fred_loader import (
        FREDLoader,
        FRED_SERIES_SP500,
        FRED_SERIES_10Y,
        FRED_SERIES_3M,
        FRED_SERIES_OIL_WTI,
    )

    fred = FREDLoader()
    start_ts = pd.Timestamp(REGIME_START) - pd.Timedelta(days=FRED_LAG_DAYS + 31)
    start = start_ts.strftime("%Y-%m-%d")
    end = REGIME_END

    yield_10y = fred.get_series(FRED_SERIES_10Y, start, end, "yield_10y", lag_days=FRED_LAG_DAYS)
    yield_3m = fred.get_series(FRED_SERIES_3M, start, end, "yield_3m", lag_days=FRED_LAG_DAYS)
    oil = fred.get_series(FRED_SERIES_OIL_WTI, start, end, "regime_oil", lag_days=FRED_LAG_DAYS)
    market = fred.get_series(FRED_SERIES_SP500, start, end, "regime_market", lag_days=FRED_LAG_DAYS)

    base = (
        yield_10y.merge(yield_3m, on="dt", how="outer")
        .merge(oil, on="dt", how="outer")
        .merge(market, on="dt", how="outer")
    )
    base["dt"] = pd.to_datetime(base["dt"])
    base = base.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")

    lseg = pd.read_csv(LSEG_REGIME_PATH)
    lseg["dt"] = pd.to_datetime(lseg["dt"])
    # Merge LSEG-only columns (copper required; market/oil optional - use suffixes to prefer LSEG)
    lseg_extra = [c for c in ["regime_copper", "regime_market", "regime_oil"] if c in lseg.columns and c not in base.columns]
    lseg_cols = ["dt"] + lseg_extra if lseg_extra else ["dt"]
    if lseg_extra:
        base = base.merge(lseg[["dt"] + lseg_extra].drop_duplicates(subset=["dt"], keep="last"), on="dt", how="outer")
    base = base.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")

    base["yield_10y"] = pd.to_numeric(base["yield_10y"], errors="coerce")
    base["yield_3m"] = pd.to_numeric(base["yield_3m"], errors="coerce")
    base["regime_yield_curve"] = base["yield_10y"] - base["yield_3m"]
    base["regime_market"] = pd.to_numeric(base["regime_market"], errors="coerce")
    market_ret = base["regime_market"].pct_change()
    yield_change = base["yield_10y"].diff()
    base["regime_stock_bond_corr"] = market_ret.rolling(63, min_periods=21).corr(yield_change)

    for col in ["regime_market", "regime_yield_curve", "regime_oil", "regime_copper", "regime_stock_bond_corr"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").ffill().bfill()

    out_cols = ["dt", "regime_market", "regime_yield_curve", "regime_oil", "regime_copper", "regime_stock_bond_corr"]
    base = base[[c for c in out_cols if c in base.columns]].copy()
    base["dt"] = base["dt"].dt.strftime("%Y-%m-%d")

    out_path = Path(REGIME_OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(base)} rows). Use regime_inputs_csv: '{REGIME_OUTPUT}' in config.")


if __name__ == "__main__":
    main()
