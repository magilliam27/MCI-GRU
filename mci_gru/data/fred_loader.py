"""
FRED API loader for credit spread and other macroeconomic data.

Fetches ICE BofA IG/HY option-adjusted spreads from FRED.
Required input failures stop preparation. Offline replay never constructs a client.
"""

import os
from importlib import import_module

import pandas as pd

from mci_gru.data.input_snapshots import InputSnapshots

# FRED series IDs for credit spreads (daily, basis points)
FRED_SERIES_IG = "BAMLC0A0CM"  # ICE BofA US Corporate Index OAS
FRED_SERIES_HY = "BAMLH0A0HYM2"  # ICE BofA US High Yield Index OAS
FRED_SERIES_SP500 = "SP500"
FRED_SERIES_10Y = "DGS10"
FRED_SERIES_3M = "DGS3MO"
FRED_SERIES_OIL_WTI = "DCOILWTICO"
FRED_SERIES_COPPER = "PCOPPUSDM"
FRED_SERIES_VIX = "VIXCLS"


class FREDLoader:
    """
    Load credit spread (and optionally other FRED) series via the FRED API.
    Requires FRED_API_KEY environment variable. Raises on missing key or fetch failure.
    """

    def __init__(
        self, api_key: str | None = None, *, snapshots: InputSnapshots | None = None
    ) -> None:
        self.snapshots = snapshots if snapshots is not None else InputSnapshots()
        self._api_key = (
            None if self.snapshots.mode == "replay" else api_key or os.environ.get("FRED_API_KEY")
        )
        if self.snapshots.mode != "replay" and not self._api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable or pass api_key to FREDLoader."
            )
        self._fred = None

    def _get_client(self):
        """Lazy-initialize fredapi.Fred to avoid import at module load if key is missing."""
        if self._fred is None:
            self._fred = import_module("fredapi").Fred(api_key=self._api_key)
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
            Exception: On acquisition or retained-input failures.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        buffered_start = (start_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        observation_start = start_ts.strftime("%Y-%m-%d")
        observation_end = end_ts.strftime("%Y-%m-%d")

        observations = {}
        for name, series_id in (("ig_spread", FRED_SERIES_IG), ("hy_spread", FRED_SERIES_HY)):
            observations[name] = self.snapshots.observe(
                role=f"fred.{name}",
                source="fred",
                request={
                    "series_id": series_id,
                    "observation_start": buffered_start,
                    "observation_end": observation_end,
                },
                acquire=lambda selected=series_id: self._get_client().get_series(
                    selected, observation_start=buffered_start, observation_end=observation_end
                ),
            )
        with self.snapshots.accepted(*observations.values(), role="credit"):
            df = pd.DataFrame({name: item.data for name, item in observations.items()})
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
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        buffered_start = (start_ts - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

        request = {
            "series_id": series_id,
            "observation_start": buffered_start,
            "observation_end": end_ts.strftime("%Y-%m-%d"),
        }
        observation = self.snapshots.observe(
            role=f"fred.{value_name}",
            source="fred",
            request=request,
            acquire=lambda: self._get_client().get_series(
                series_id,
                observation_start=request["observation_start"],
                observation_end=request["observation_end"],
            ),
        )
        with self.snapshots.accepted(observation):
            values = observation.data
            df = pd.DataFrame({value_name: values}).sort_index()
            df = df.replace(".", pd.NA).apply(pd.to_numeric, errors="coerce")
            df = df.ffill().bfill()
            if lag_days > 0:
                df = df.shift(lag_days)

            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            df = df.dropna(how="all")
            if df.empty:
                raise ValueError(
                    f"No data available for FRED series {series_id} in range {start} to {end}"
                )

            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "dt"})
            df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
            return df[["dt", value_name]]
