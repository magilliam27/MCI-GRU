"""
Shared LSEG DataFrame reshaping utilities.

Centralizes the MultiIndex-to-flat-OHLCV transformation used by both
``lseg_loader.py`` and ``paper_trade/scripts/refresh_data.py``.
"""

import pandas as pd

# LSEG field names → standard OHLCV column names.
# Covers the official daily summary fields, alternative field names,
# and simple/title-case variants that the API may return.
COLUMN_MAPPING = {
    "MKT_OPEN": "open",
    "MKT_HIGH": "high",
    "MKT_LOW": "low",
    "TRDPRC_1": "close",
    "ACVOL_UNS": "volume",
    "OPEN_PRC": "open",
    "HIGH_1": "high",
    "LOW_1": "low",
    "HST_CLOSE": "close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

STANDARD_COLS = ["kdcode", "dt", "open", "high", "low", "close", "volume", "turnover"]


def reshape_lseg_to_standard(combined: pd.DataFrame) -> pd.DataFrame:
    """Reshape an LSEG MultiIndex DataFrame to flat OHLCV + turnover format.

    The Refinitiv ``get_history()`` API returns DataFrames with a MultiIndex
    column structure ``(Instrument, Field)``.  This function normalises that
    into a tidy DataFrame with columns matching ``STANDARD_COLS``.

    Raises ``ValueError`` if *combined* does not have MultiIndex columns or
    if required OHLCV columns are missing after the reshape.
    """
    if not isinstance(combined.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from rd.get_history")

    instruments = combined.columns.get_level_values(0).unique().tolist()
    records = []

    for instrument in instruments:
        if instrument in ("Date", "Instrument", "index"):
            continue
        try:
            inst_data = combined[instrument].copy()
            inst_data = inst_data.rename(columns=COLUMN_MAPPING)
            inst_data["kdcode"] = instrument
            inst_data["dt"] = combined.index
            records.append(inst_data)
        except Exception as exc:
            print(f"  Skipping {instrument}: {exc}")

    df = pd.concat(records, ignore_index=True)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after reshape: {missing}")

    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["close"])
    df = df.drop_duplicates(subset=["kdcode", "dt"], keep="first")
    df["turnover"] = df["volume"] * df["close"]
    df = df[STANDARD_COLS]
    df = df.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    return df
