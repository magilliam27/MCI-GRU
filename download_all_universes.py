from pathlib import Path
import pandas as pd

from mci_gru.data.lseg_loader import LSEGLoader
from mci_gru.data.fred_loader import (
    FREDLoader,
    FRED_SERIES_SP500,
    FRED_SERIES_10Y,
    FRED_SERIES_3M,
    FRED_SERIES_OIL_WTI,
    FRED_SERIES_COPPER,
)

start = "2016-01-01"
end = "2025-12-31"

# ---- Try FRED client (optional fallback) ----
fred = None
try:
    fred = FREDLoader()
except Exception:
    fred = None

def try_fred(series_id, value_name):
    if fred is None:
        return None, "FRED unavailable (no API key or init failure)"
    try:
        df = fred.get_series(series_id, start, end, value_name, lag_days=1)
        return df, None
    except Exception as e:
        return None, f"FRED error: {e}"

# ---- Pull from FRED primary/fallback ----
yield_10y, err_y10_f = try_fred(FRED_SERIES_10Y, "yield_10y")
yield_3m, err_y3m_f = try_fred(FRED_SERIES_3M, "yield_3m")
oil_fred, err_oil_f = try_fred(FRED_SERIES_OIL_WTI, "regime_oil")
mkt_fred, err_mkt_f = try_fred(FRED_SERIES_SP500, "regime_market")
cop_fred, err_cop_f = try_fred(FRED_SERIES_COPPER, "regime_copper")

# ---- Pull from LSEG ----
market_lseg = copper_lseg = y10_lseg = y3m_lseg = oil_lseg = None
lseg_errors = {}

loader = LSEGLoader()
try:
    loader.connect()
    market_lseg = loader.get_series(".SPX", start, end, "regime_market")
    copper_lseg = loader.get_series(".MXCOPPFE", start, end, "regime_copper")
    y10_lseg = loader.get_series("US10YT=RR", start, end, "yield_10y")
    y3m_lseg = loader.get_series("US3MT=RR", start, end, "yield_3m")
    oil_lseg = loader.get_series("CLc1", start, end, "regime_oil")
finally:
    loader.disconnect()

# ---- Hybrid choose (same intended priority) ----
# FRED-primary: yields, oil
yield_10y = yield_10y if yield_10y is not None else y10_lseg
yield_3m = yield_3m if yield_3m is not None else y3m_lseg
oil = oil_fred if oil_fred is not None else oil_lseg

# LSEG-primary: market, copper
market = market_lseg if market_lseg is not None else mkt_fred
copper = copper_lseg if copper_lseg is not None else cop_fred

series_map = {
    "yield_10y": yield_10y,
    "yield_3m": yield_3m,
    "regime_oil": oil,
    "regime_market": market,
    "regime_copper": copper,
}

# ---- Merge whatever exists ----
available = [v for v in series_map.values() if v is not None and len(v) > 0]
if not available:
    raise RuntimeError("No regime series could be fetched from LSEG/FRED.")

base = available[0].copy()
for df in available[1:]:
    base = base.merge(df, on="dt", how="outer")

base["dt"] = pd.to_datetime(base["dt"])
base = base.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")

# Compute derived fields if source columns exist
if "yield_10y" in base.columns and "yield_3m" in base.columns:
    base["yield_10y"] = pd.to_numeric(base["yield_10y"], errors="coerce")
    base["yield_3m"] = pd.to_numeric(base["yield_3m"], errors="coerce")
    base["regime_yield_curve"] = base["yield_10y"] - base["yield_3m"]

if "regime_market" in base.columns and "yield_10y" in base.columns:
    base["regime_market"] = pd.to_numeric(base["regime_market"], errors="coerce")
    market_ret = base["regime_market"].pct_change()
    yield_change = base["yield_10y"].diff()
    base["regime_stock_bond_corr"] = market_ret.rolling(63, min_periods=21).corr(yield_change)

# keep only desired date span
base = base[(base["dt"] >= pd.Timestamp(start)) & (base["dt"] <= pd.Timestamp(end))]
base["dt"] = base["dt"].dt.strftime("%Y-%m-%d")

# ---- Save outputs ----
out_dir = Path("data/raw/market")
out_dir.mkdir(parents=True, exist_ok=True)

data_path = out_dir / "regime_inputs_partial_2016_2025.csv"
base.to_csv(data_path, index=False)

missing = []
for k, v in series_map.items():
    if v is None or len(v) == 0:
        missing.append({"series": k, "status": "missing"})
    else:
        missing.append({"series": k, "status": "ok", "rows": len(v)})

missing_df = pd.DataFrame(missing)
missing_path = out_dir / "regime_inputs_missing_report_2016_2025.csv"
missing_df.to_csv(missing_path, index=False)

print(f"Saved partial data: {data_path}")
print(f"Saved missing report: {missing_path}")
print("Columns in partial file:", list(base.columns))
print("Date range:", base["dt"].min(), "to", base["dt"].max())