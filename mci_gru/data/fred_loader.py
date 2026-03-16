"""
FRED API loader for credit spread and other macroeconomic data.

Fetches ICE BofA IG/HY option-adjusted spreads from FRED.
No CSV fallback: if the API fails, the caller should soft-fail (log and continue without credit features).
"""

import os
from typing import Optional

import pandas as pd

# FRED series IDs for credit spreads (daily, basis points)
FRED_SERIES_IG = "BAMLC0A0CM"   # ICE BofA US Corporate Index OAS
FRED_SERIES_HY = "BAMLH0A0HYM2"  # ICE BofA US High Yield Index OAS
FRED_SERIES_SP500 = "SP500"
FRED_SERIES_10Y = "DGS10"
FRED_SERIES_3M = "DGS3MO"
FRED_SERIES_OIL_WTI = "DCOILWTICO"
FRED_SERIES_COPPER = "PCOPPUSDM"


class FREDLoader:
    """
    Load credit spread (and optionally other FRED) series via the FRED API.
    Requires FRED_API_KEY environment variable. Raises on missing key or fetch failure.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable or pass api_key to FREDLoader."
            )
        self._fred = None

    def _get_client(self):
        """Lazy-initialize fredapi.Fred to avoid import at module load if key is missing."""
        if self._fred is None:
            from fredapi import Fred
            self._fred = Fred(api_key=self._api_key)
        return self._fred

    def get_credit_spreads(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch IG and HY option-adjusted spreads from FRED and return a daily DataFrame.

        - Forward-fills weekends/holidays so each calendar day has a value (aligned with
          typical trading calendar usage after merge).
        - Applies 1-day lag: the value assigned to date T is the spread from T-1,
          to avoid look-ahead bias (FRED data may not be available until after market close).

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            DataFrame with columns [dt, ig_spread, hy_spread]. dt is string YYYY-MM-DD.
            Spreads are in basis points (as published by FRED).

        Raises:
            ValueError: If API key is missing.
            Exception: On network/API errors (caller should catch and soft-fail).
        """
        fred = self._get_client()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        buffered_start = (start_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        observation_start = start_ts.strftime("%Y-%m-%d")
        observation_end = end_ts.strftime("%Y-%m-%d")

        ig = fred.get_series(
            FRED_SERIES_IG,
            observation_start=buffered_start,
            observation_end=observation_end,
        )
        hy = fred.get_series(
            FRED_SERIES_HY,
            observation_start=buffered_start,
            observation_end=observation_end,
        )

        df = pd.DataFrame({"ig_spread": ig, "hy_spread": hy})
        df = df.sort_index()
        df = df.ffill()
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError(
                f"No credit spread data returned from FRED for {observation_start} to {observation_end}."
            )

        df = df.bfill()

        # 1-day lag: value at T = spread from T-1 (avoid look-ahead bias)
        df = df.shift(1)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df.dropna(how="all")

        if df.empty:
            raise ValueError(
                f"No lagged credit spread data available for requested range {observation_start} to {observation_end}."
            )

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "dt"})
        df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")

        return df[["dt", "ig_spread", "hy_spread"]]

    def get_series(
        self,
        series_id: str,
        start: str,
        end: str,
        value_name: str,
        lag_days: int = 1,
        buffer_days: int = 31,
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series with a point-in-time-safe lag.

        Args:
            series_id: FRED series identifier.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            value_name: Name for output value column.
            lag_days: Number of calendar days to lag to avoid look-ahead.
            buffer_days: Lookback buffer to preserve values after lagging.

        Returns:
            DataFrame with columns [dt, <value_name>], dt as YYYY-MM-DD.
        """
        fred = self._get_client()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        buffered_start = (start_ts - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

        values = fred.get_series(
            series_id,
            observation_start=buffered_start,
            observation_end=end_ts.strftime("%Y-%m-%d"),
        )
        df = pd.DataFrame({value_name: values}).sort_index()
        df = df.replace(".", pd.NA).apply(pd.to_numeric, errors="coerce")
        df = df.ffill().bfill()
        if lag_days > 0:
            df = df.shift(lag_days)

        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError(f"No data available for FRED series {series_id} in range {start} to {end}")

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "dt"})
        df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
        return df[["dt", value_name]]
