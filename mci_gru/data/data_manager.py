"""
Unified data manager for MCI-GRU.

This module provides a unified interface for loading data from
different sources (CSV, LSEG) and preparing it for model training.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mci_gru.data.path_resolver import resolve_project_data_path

if TYPE_CHECKING:
    from mci_gru.config import DataConfig
    from mci_gru.graph.builder import GraphSchedule


class DataManager:
    """
    Unified data manager for loading and preparing stock data.

    Supports loading from CSV files or LSEG/Refinitiv API.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.df: pd.DataFrame | None = None
        self.vix_df: pd.DataFrame | None = None
        self.credit_df: pd.DataFrame | None = None
        self.regime_df: pd.DataFrame | None = None
        self.kdcode_list: list[str] | None = None

    def load(self) -> pd.DataFrame:
        if self.config.source == "csv":
            return self._load_from_csv()
        elif self.config.source == "lseg":
            return self._load_from_lseg()
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")

    def load_index_series(self) -> pd.DataFrame:
        """
        Load a single index series for index-level experiment mode (no survivorship bias).

        Uses index_filename CSV if set (columns: dt, close; optional open, high, low, volume),
        otherwise FRED SP500 with 1-day lag. Returns DataFrame with kdcode='INDEX' and
        standard OHLCV columns so feature pipeline can run unchanged.
        """
        start_ts = pd.Timestamp(self.config.train_start) - pd.Timedelta(days=365 * 2)
        start = start_ts.strftime("%Y-%m-%d")
        end = self.config.test_end

        if self.config.index_filename:
            resolved = resolve_project_data_path(self.config.index_filename)
            df = pd.read_csv(resolved)
            df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
            if "close" not in df.columns:
                raise ValueError(f"Index CSV must have 'close' column: {resolved}")
            for col in ["open", "high", "low", "volume"]:
                if col not in df.columns:
                    df[col] = df["close"] if col in ("open", "high", "low") else 0.0
            if "turnover" not in df.columns:
                df["turnover"] = df["volume"] * df["close"]
            df = df[["dt", "open", "high", "low", "close", "volume", "turnover"]]
        else:
            from mci_gru.data.fred_loader import FRED_SERIES_SP500, FREDLoader

            fred = FREDLoader()
            s = fred.get_series(FRED_SERIES_SP500, start, end, "close", lag_days=1)
            df = s.copy()
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 0.0
            df["turnover"] = 0.0
            df = df[["dt", "open", "high", "low", "close", "volume", "turnover"]]

        df["kdcode"] = "INDEX"
        df = df[["kdcode", "dt", "open", "high", "low", "close", "volume", "turnover"]]
        df = df.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
        df = df[(df["dt"] >= start) & (df["dt"] <= end)]
        self.df = df
        self.kdcode_list = ["INDEX"]
        return df

    def _load_from_csv(self) -> pd.DataFrame:
        resolved_path = resolve_project_data_path(self.config.filename)
        print(f"Loading data from {resolved_path}...")

        df = pd.read_csv(resolved_path)

        print(f"  Loaded {len(df)} rows")
        print(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")
        print(f"  Stocks: {df['kdcode'].nunique()}")

        self.df = df
        return df

    def _load_from_lseg(self) -> pd.DataFrame:
        from mci_gru.data.lseg_loader import LSEGLoader

        loader = LSEGLoader()
        try:
            loader.connect()

            # Determine date range (need data before training for correlation)
            start_date = self.config.train_start
            end_date = self.config.test_end

            df = loader.fetch_universe_data(
                universe=self.config.universe,
                start=start_date,
                end=end_date,
                include_vix=False,  # We'll handle VIX separately
            )

            self.df = df
            return df

        finally:
            loader.disconnect()

    def load_vix(self) -> pd.DataFrame:
        if self.config.source == "lseg":
            from mci_gru.data.lseg_loader import LSEGLoader

            loader = LSEGLoader()
            try:
                loader.connect()
                vix_df = loader.get_vix(self.config.train_start, self.config.test_end)
                self.vix_df = vix_df
                return vix_df
            finally:
                loader.disconnect()
        else:
            try:
                vix_path = resolve_project_data_path("vix_data.csv")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "VIX data not found. Create vix_data.csv under data/raw/market "
                    "or use source='lseg'"
                ) from e
            vix_df = pd.read_csv(vix_path)
            self.vix_df = vix_df
            return vix_df

    def load_credit_spreads(self) -> pd.DataFrame:
        """
        Load credit spread data (IG/HY OAS) from FRED API.

        Requires FRED_API_KEY environment variable. If the key is missing or
        the fetch fails, raises an exception; the caller should catch and
        soft-fail (e.g. continue without credit features).

        Returns:
            DataFrame with columns [dt, ig_spread, hy_spread]
        """
        from mci_gru.data.fred_loader import FREDLoader

        # Align credit history to the earliest loaded stock date when available.
        # This avoids large pre-merge gaps (and zero fallback fills) when the
        # stock CSV includes a pre-train buffer period (e.g., 2018 for 2019 train).
        credit_start = self.config.train_start
        if self.df is not None and "dt" in self.df.columns and len(self.df) > 0:
            try:
                earliest_stock_dt = pd.to_datetime(self.df["dt"], errors="coerce").dropna().min()
                if pd.notna(earliest_stock_dt):
                    credit_start = earliest_stock_dt.strftime("%Y-%m-%d")
            except Exception:
                # Fall back to configured train_start on unexpected date parsing issues.
                credit_start = self.config.train_start

        loader = FREDLoader()
        credit_df = loader.get_credit_spreads(
            start=credit_start,
            end=self.config.test_end,
        )
        self.credit_df = credit_df
        return credit_df

    def load_regime_inputs(
        self,
        lseg_market_ric: str = ".SPX",
        lseg_copper_ric: str = ".MXCOPPFE",
        lseg_yield_10y_ric: str = "US10YT=RR",
        lseg_yield_3m_ric: str = "US3MT=RR",
        lseg_oil_ric: str = "CLc1",
        lseg_vix_ric: str = "VIX",
        regime_inputs_csv: str | None = None,
        regime_enforce_lag_days: int = 0,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Load Phase-1 global regime input series with hybrid sourcing.

        If regime_inputs_csv is set, load from that file (and optionally apply lag);
        otherwise use FRED/LSEG APIs.

        Args:
            end: Optional override for the fetch end date (ISO string). When None,
                defaults to self.config.test_end. Pass the inference/prediction date
                here so FRED series are fetched through the live date, not the frozen
                training config end date.

        Output columns:
            dt, regime_market, regime_yield_curve, regime_oil, regime_copper,
            regime_stock_bond_corr, regime_monetary_policy, regime_volatility
        """
        from mci_gru.features.regime import (
            REGIME_OPTIONAL_VARIABLES,
            REGIME_REQUIRED_VARIABLES,
            REGIME_VARIABLES,
        )

        if regime_inputs_csv:
            resolved = resolve_project_data_path(regime_inputs_csv)
            base = pd.read_csv(resolved)
            base["dt"] = pd.to_datetime(base["dt"])
            base = base.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
            required = {"dt"} | set(REGIME_REQUIRED_VARIABLES)
            missing = sorted(required - set(base.columns))
            if missing:
                raise ValueError(
                    f"Regime CSV {regime_inputs_csv} is missing required columns: {missing}. "
                    "See docs/REGIME_DATA_CONTRACT.md."
                )
            for col in REGIME_OPTIONAL_VARIABLES:
                if col not in base.columns:
                    base[col] = np.nan
            base = base[["dt"] + REGIME_VARIABLES].copy()
            for col in REGIME_VARIABLES:
                base[col] = pd.to_numeric(base[col], errors="coerce")
            if regime_enforce_lag_days > 0:
                base[REGIME_VARIABLES] = base[REGIME_VARIABLES].shift(regime_enforce_lag_days)
            base[REGIME_VARIABLES] = base[REGIME_VARIABLES].ffill().bfill()
            base["dt"] = base["dt"].dt.strftime("%Y-%m-%d")
            self.regime_df = base
            return base

        from mci_gru.data.fred_loader import (
            FRED_SERIES_3M,
            FRED_SERIES_10Y,
            FRED_SERIES_COPPER,
            FRED_SERIES_OIL_WTI,
            FRED_SERIES_SP500,
            FRED_SERIES_VIX,
            FREDLoader,
        )

        start_ts = pd.Timestamp(self.config.train_start) - pd.Timedelta(days=365 * 15)
        start = start_ts.strftime("%Y-%m-%d")
        end = end if end is not None else self.config.test_end

        fred = None
        try:
            fred = FREDLoader()
        except Exception:
            fred = None

        def try_fred(series_id: str, value_name: str):
            if fred is None:
                return None
            try:
                return fred.get_series(series_id, start, end, value_name, lag_days=1)
            except Exception:
                return None

        # FRED-primary variables.
        yield_10y = try_fred(FRED_SERIES_10Y, "yield_10y")
        yield_3m = try_fred(FRED_SERIES_3M, "yield_3m")
        oil = try_fred(FRED_SERIES_OIL_WTI, "regime_oil")
        volatility = try_fred(FRED_SERIES_VIX, "regime_volatility")

        # FRED fallback candidates.
        market_fallback = try_fred(FRED_SERIES_SP500, "regime_market")
        copper_fallback = try_fred(FRED_SERIES_COPPER, "regime_copper")

        market = None
        copper = None
        lseg_yield_10y = None
        lseg_yield_3m = None
        lseg_oil = None
        lseg_volatility = None

        if self.config.source == "lseg":
            from mci_gru.data.lseg_loader import LSEGLoader

            loader = LSEGLoader()
            try:
                loader.connect()
                market = loader.get_series(lseg_market_ric, start, end, "regime_market")
                copper = loader.get_series(lseg_copper_ric, start, end, "regime_copper")
                if yield_10y is None:
                    lseg_yield_10y = loader.get_series(lseg_yield_10y_ric, start, end, "yield_10y")
                if yield_3m is None:
                    lseg_yield_3m = loader.get_series(lseg_yield_3m_ric, start, end, "yield_3m")
                if oil is None:
                    lseg_oil = loader.get_series(lseg_oil_ric, start, end, "regime_oil")
                if volatility is None:
                    lseg_volatility = loader.get_series(
                        lseg_vix_ric, start, end, "regime_volatility"
                    )
            finally:
                loader.disconnect()

        if yield_10y is None:
            yield_10y = lseg_yield_10y
        if yield_3m is None:
            yield_3m = lseg_yield_3m
        if oil is None:
            oil = lseg_oil
        if volatility is None:
            volatility = lseg_volatility

        if market is None:
            market = market_fallback
        if copper is None:
            copper = copper_fallback

        required_series = {
            "yield_10y": yield_10y,
            "yield_3m": yield_3m,
            "regime_oil": oil,
            "regime_market": market,
            "regime_copper": copper,
            "regime_volatility": volatility,
        }
        missing = [
            name for name, value in required_series.items() if value is None or len(value) == 0
        ]
        if missing:
            raise ValueError(
                "Unable to load required regime input series. Missing: "
                f"{missing}. Provide FRED_API_KEY and/or LSEG entitlements for configured RICs."
            )

        base = (
            yield_10y.merge(yield_3m, on="dt", how="outer")
            .merge(oil, on="dt", how="outer")
            .merge(market, on="dt", how="outer")
            .merge(copper, on="dt", how="outer")
            .merge(volatility, on="dt", how="outer")
        )
        base["dt"] = pd.to_datetime(base["dt"])
        base = base.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")

        base["yield_10y"] = pd.to_numeric(base["yield_10y"], errors="coerce")
        base["yield_3m"] = pd.to_numeric(base["yield_3m"], errors="coerce")
        base["regime_yield_curve"] = base["yield_10y"] - base["yield_3m"]
        base["regime_monetary_policy"] = base["yield_3m"]
        base["regime_market"] = pd.to_numeric(base["regime_market"], errors="coerce")
        base["regime_volatility"] = pd.to_numeric(base["regime_volatility"], errors="coerce")

        # Stock-bond correlation proxy: rolling corr between market returns and yield-10y changes.
        market_ret = base["regime_market"].pct_change()
        yield_change = base["yield_10y"].diff()
        base["regime_stock_bond_corr"] = market_ret.rolling(63, min_periods=21).corr(yield_change)

        # Fill sparse macro holidays/weekends while preserving time direction.
        for col in [
            "regime_market",
            "regime_yield_curve",
            "regime_oil",
            "regime_copper",
            "regime_stock_bond_corr",
            "regime_monetary_policy",
            "regime_volatility",
        ]:
            base[col] = pd.to_numeric(base[col], errors="coerce").ffill().bfill()

        base = base[
            [
                "dt",
                "regime_market",
                "regime_yield_curve",
                "regime_oil",
                "regime_copper",
                "regime_stock_bond_corr",
                "regime_monetary_policy",
                "regime_volatility",
            ]
        ].copy()
        base["dt"] = base["dt"].dt.strftime("%Y-%m-%d")
        self.regime_df = base
        return base

    def filter_complete_stocks(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Filter to stocks with complete data across all periods.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame and list of stock codes
        """
        print("Filtering stocks with complete data...")

        date_mask = (df["dt"] >= self.config.train_start) & (df["dt"] <= self.config.test_end)
        df_period = df[date_mask].copy()
        period_dates = sorted(df_period["dt"].unique())

        print(
            f"  Period: {len(period_dates)} trading days from {period_dates[0]} to {period_dates[-1]}"
        )

        kdcode_counts = df_period["kdcode"].value_counts()
        kdcode_list = kdcode_counts[kdcode_counts == len(period_dates)].index.tolist()
        kdcode_list = sorted(kdcode_list)

        print(f"  Stocks with complete data: {len(kdcode_list)}")

        if len(kdcode_list) == 0:
            raise ValueError("No stocks have complete data across the entire period!")

        df_filtered = df_period[df_period["kdcode"].isin(kdcode_list)].copy()
        df_filtered = df_filtered.sort_values(["dt", "kdcode"]).reset_index(drop=True)

        self.kdcode_list = kdcode_list
        return df_filtered, kdcode_list

    def filter_complete_stocks_per_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Keep stocks that appear on every trading day within train, val, and test separately."""
        print("Filtering stocks with per-split complete data...")

        train_mask = (df["dt"] >= self.config.train_start) & (df["dt"] <= self.config.train_end)
        val_mask = (df["dt"] >= self.config.val_start) & (df["dt"] <= self.config.val_end)
        test_mask = (df["dt"] >= self.config.test_start) & (df["dt"] <= self.config.test_end)

        def _complete_kdcodes(mask: pd.Series) -> set[str]:
            sub = df[mask]
            dates = sorted(sub["dt"].unique())
            if not dates:
                return set()
            n = len(dates)
            counts = sub["kdcode"].value_counts()
            return set(counts[counts == n].index.astype(str))

        k_train = _complete_kdcodes(train_mask)
        k_val = _complete_kdcodes(val_mask)
        k_test = _complete_kdcodes(test_mask)
        kdcode_list = sorted(k_train & k_val & k_test)

        print(f"  Train-complete: {len(k_train)}, val-complete: {len(k_val)}, test-complete: {len(k_test)}")
        print(f"  Intersection (usable in all splits): {len(kdcode_list)}")

        if len(kdcode_list) == 0:
            raise ValueError("No stocks have complete data in every split after per-split filtering!")

        df_filtered = df[df["kdcode"].isin(kdcode_list)].copy()
        df_filtered = df_filtered.sort_values(["dt", "kdcode"]).reset_index(drop=True)
        self.kdcode_list = kdcode_list
        return df_filtered, kdcode_list

    def split_by_period(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test periods.

        Args:
            df: Filtered DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_mask = (df["dt"] >= self.config.train_start) & (df["dt"] <= self.config.train_end)
        val_mask = (df["dt"] >= self.config.val_start) & (df["dt"] <= self.config.val_end)
        test_mask = (df["dt"] >= self.config.test_start) & (df["dt"] <= self.config.test_end)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        train_dates = sorted(train_df["dt"].unique())
        val_dates = sorted(val_df["dt"].unique())
        test_dates = sorted(test_df["dt"].unique())

        print(f"  Training: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
        print(f"  Validation: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
        print(f"  Test: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

        return train_df, val_df, test_df


class CombinedDataset(Dataset):
    """
    Combined dataset for synchronized time series, graph features, and labels.

    This ensures time series and graph data stay aligned when shuffling.
    """

    def __init__(self, X_time_series, X_graph, y, sample_dates: list[str] | None = None):
        self.X_time_series = X_time_series
        self.X_graph = X_graph
        self.y = y
        self.sample_dates = sample_dates

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {
            "time_series": self.X_time_series[idx],
            "graph_features": self.X_graph[idx],
            "label": self.y[idx],
        }
        if self.sample_dates is not None:
            item["date"] = self.sample_dates[idx]
        return item


def _snapshot_age_days(sample_date: str, valid_from: str) -> float:
    ds = datetime.strptime(sample_date[:10], "%Y-%m-%d").date()
    dv = datetime.strptime(valid_from[:10], "%Y-%m-%d").date()
    return float((ds - dv).days)


def combined_collate_fn(
    batch,
    edge_index,
    edge_weight,
    graph_schedule=None,
    append_snapshot_age_days: bool = False,
    static_graph_valid_from: str | None = None,
    edge_index_sector: torch.Tensor | None = None,
    edge_weight_sector: torch.Tensor | None = None,
    use_sector_relation: bool = False,
):
    """
    Custom collate function to create properly batched graph data.

    PyG batches graphs by concatenating nodes and shifting edge indices.
    This function replicates that behavior while keeping time series aligned.

    When *graph_schedule* is provided and samples carry dates, each sample
    looks up its own graph snapshot from the schedule.  Otherwise the fixed
    *edge_index* / *edge_weight* are replicated across the batch (static
    mode).

    When ``append_snapshot_age_days`` is True, appends one column to 2-D
    ``edge_weight`` (multi-feature edges only) using calendar age from the
    active snapshot ``valid_from`` to the sample date.

    Returns a **9-tuple** (trainer contract): the first seven entries match
    the historical layout; entries 7–8 are optional sector ``edge_index``
    and ``edge_weight`` batched the same way (or ``None`` when disabled).
    """
    num_stocks = batch[0]["graph_features"].shape[0]

    time_series = torch.stack([item["time_series"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    has_dates = "date" in batch[0]
    use_schedule = graph_schedule is not None and has_dates

    graph_features_list = []
    edge_index_list = []
    edge_weight_list = []

    for i, item in enumerate(batch):
        graph_features_list.append(item["graph_features"])

        if use_schedule:
            ei, ew = graph_schedule.get_graph_for_date(item["date"])
            valid_from = graph_schedule.snapshot_valid_from_for_date(item["date"])
        else:
            ei, ew = edge_index, edge_weight
            valid_from = static_graph_valid_from

        if append_snapshot_age_days and ew.dim() == 2 and valid_from is not None and has_dates:
            age = _snapshot_age_days(item["date"], valid_from)
            col = torch.full((ew.shape[0], 1), age, dtype=ew.dtype, device=ew.device)
            ew = torch.cat([ew, col], dim=-1)

        edge_index_list.append(ei + (i * num_stocks))
        edge_weight_list.append(ew)

    batched_graph_features = torch.cat(graph_features_list, dim=0)
    batched_edge_index = torch.cat(edge_index_list, dim=1)
    batched_edge_weight = torch.cat(edge_weight_list, dim=0)

    batch_dates = [item["date"] for item in batch] if has_dates else None

    batched_ei_sec = None
    batched_ew_sec = None
    if use_sector_relation and edge_index_sector is not None and edge_weight_sector is not None:
        esi_list = []
        esw_list = []
        for i in range(len(batch)):
            esi_list.append(edge_index_sector + (i * num_stocks))
            esw_list.append(edge_weight_sector)
        batched_ei_sec = torch.cat(esi_list, dim=1)
        batched_ew_sec = torch.cat(esw_list, dim=0)

    return (
        time_series,
        labels,
        batched_graph_features,
        batched_edge_index,
        batched_edge_weight,
        num_stocks,
        batch_dates,
        batched_ei_sec,
        batched_ew_sec,
    )


def create_data_loaders(
    stock_features_train: np.ndarray,
    x_graph_train: np.ndarray,
    train_labels: np.ndarray,
    stock_features_val: np.ndarray,
    x_graph_val: np.ndarray,
    val_labels: np.ndarray,
    stock_features_test: np.ndarray,
    x_graph_test: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    batch_size: int = 32,
    train_dates: list[str] | None = None,
    val_dates: list[str] | None = None,
    test_dates: list[str] | None = None,
    dynamic_graph: bool = False,
    graph_schedule: GraphSchedule | None = None,
    append_snapshot_age_days: bool = False,
    static_graph_valid_from: str | None = None,
    edge_index_sector: torch.Tensor | None = None,
    edge_weight_sector: torch.Tensor | None = None,
    use_sector_relation: bool = False,
) -> tuple:
    """
    Create train/val/test data loaders.

    Args:
        stock_features_train: Training time series (days, stocks, seq_len, features)
        x_graph_train: Training graph features (days, stocks, features)
        train_labels: Training labels (days, stocks)
        stock_features_val: Validation time series
        x_graph_val: Validation graph features
        val_labels: Validation labels
        stock_features_test: Test time series
        x_graph_test: Test graph features
        edge_index: Graph edge indices (static fallback when graph_schedule is None)
        edge_weight: Graph edge weights
        batch_size: Batch size for training
        train_dates: Per-sample dates aligned with train samples
        val_dates: Per-sample dates aligned with val samples
        test_dates: Per-sample dates aligned with test samples
        dynamic_graph: When True, attach dates to datasets and disable shuffle
                       so the collate function can look up per-sample graphs.
        graph_schedule: Precomputed graph snapshots.  When provided, each
                        sample's graph is resolved by date in the collate
                        function, allowing batch_size > 1 in dynamic mode.
        append_snapshot_age_days: Append one ``edge_attr`` column from snapshot age (multi-feature only).
        static_graph_valid_from: ``valid_from`` string for static graphs when snapshot age is on.
        edge_index_sector / edge_weight_sector: Static sector-branch graph (optional).
        use_sector_relation: When True, batch sector edges as tuple slots 7–8.

    Returns:
        Tuple of (train_loader, val_loader, test_loader). Each batch is a **9-tuple**
        from ``combined_collate_fn`` (see that function's docstring).
    """
    print("Creating data loaders...")

    X_train_ts = torch.from_numpy(stock_features_train).float()
    X_train_graph = torch.from_numpy(x_graph_train).float()
    y_train = torch.from_numpy(train_labels).float()

    X_val_ts = torch.from_numpy(stock_features_val).float()
    X_val_graph = torch.from_numpy(x_graph_val).float()
    y_val = torch.from_numpy(val_labels).float()

    X_test_ts = torch.from_numpy(stock_features_test).float()
    X_test_graph = torch.from_numpy(x_graph_test).float()
    y_test_dummy = torch.zeros(len(X_test_ts), X_test_graph.shape[1], dtype=torch.float32)

    print(f"  Train: ts={X_train_ts.shape}, graph={X_train_graph.shape}, labels={y_train.shape}")
    print(f"  Val: ts={X_val_ts.shape}, graph={X_val_graph.shape}, labels={y_val.shape}")
    print(f"  Test: ts={X_test_ts.shape}, graph={X_test_graph.shape}")

    attach_dates = dynamic_graph or append_snapshot_age_days
    train_dataset = CombinedDataset(
        X_train_ts,
        X_train_graph,
        y_train,
        sample_dates=train_dates if attach_dates else None,
    )
    val_dataset = CombinedDataset(
        X_val_ts,
        X_val_graph,
        y_val,
        sample_dates=val_dates if attach_dates else None,
    )
    test_dataset = CombinedDataset(
        X_test_ts,
        X_test_graph,
        y_test_dummy,
        sample_dates=test_dates if attach_dates else None,
    )

    collate_fn = partial(
        combined_collate_fn,
        edge_index=edge_index,
        edge_weight=edge_weight,
        graph_schedule=graph_schedule,
        append_snapshot_age_days=append_snapshot_age_days,
        static_graph_valid_from=static_graph_valid_from,
        edge_index_sector=edge_index_sector,
        edge_weight_sector=edge_weight_sector,
        use_sector_relation=use_sector_relation,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not dynamic_graph,
        drop_last=False,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"  Created loaders: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
