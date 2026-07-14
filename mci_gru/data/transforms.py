"""Shared feature imputation, normalization, and single-date tensor helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def impute_feature_nans_by_day(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Per-day cross-sectional mean fill, then zero-fill remaining NaNs."""
    parts = []
    for _, df_day in df.groupby("dt"):
        df_day = df_day.copy()
        for col in feature_cols:
            if col in df_day.columns:
                df_day[col] = df_day[col].fillna(df_day[col].mean())
        df_day = df_day.fillna(0.0)
        parts.append(df_day)
    return pd.concat(parts)


def compute_zscore_norm_stats(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_start: str,
    train_end: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return mean/std fitted only on the inclusive training date range."""
    train_df = df[(df["dt"] >= train_start) & (df["dt"] <= train_end)]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in feature_cols:
        if col in train_df.columns:
            means[col] = train_df[col].mean()
            stds[col] = train_df[col].std()
            if stds[col] == 0:
                stds[col] = 1.0
    return means, stds


def normalize_features_zscore(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    means: Mapping[str, float],
    stds: Mapping[str, float],
    *,
    clip_sigma: float = 3.0,
    default_mean: float | None = 0.0,
    default_std: float | None = 1.0,
) -> pd.DataFrame:
    """3-sigma clip + z-score. Moved from pipeline._apply_normalisation;
    default_mean/default_std params reconcile pipeline's `.get(col, 0.0/1.0)`
    fallback with infer.py's direct `means[col]` indexing — infer.py callers
    pass default_mean=None and default_std=None (KeyError on missing col)."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        if default_mean is None:
            m = means[col]
        else:
            m = means.get(col, default_mean)
        if default_std is None:
            s = stds[col]
        else:
            s = stds.get(col, default_std)
        df[col] = np.clip(df[col], m - clip_sigma * s, m + clip_sigma * s)
        df[col] = (df[col] - m) / s
    return df


def build_single_date_tensors(
    df_norm: pd.DataFrame,
    kdcode_list: Sequence[str],
    feature_cols: Sequence[str],
    his_t: int,
    target_date: str,
    *,
    use_polars: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (time_series (1,S,his_t,F), graph_features (1,S,F)) for one date.

    Single-date special case of generate_time_series_features/generate_graph_features.
    """
    del use_polars  # reserved for WS-K parity with generate_time_series_features

    all_dates = sorted(df_norm["dt"].unique())
    if target_date not in all_dates:
        raise ValueError(f"Target date {target_date} not found in data")

    target_idx = all_dates.index(target_date)
    if target_idx < his_t:
        raise ValueError(
            f"Need at least {his_t} days before target date; only {target_idx} available"
        )

    window_dates = all_dates[target_idx - his_t : target_idx + 1]
    df_window = df_norm[df_norm["dt"].isin(window_dates)].copy()

    stock_to_idx = {kd: i for i, kd in enumerate(kdcode_list)}
    n_stocks = len(kdcode_list)
    n_features = len(feature_cols)

    lookback_dates = window_dates[:his_t]
    pivot = np.zeros((his_t, n_stocks, n_features), dtype=np.float32)

    for _, row in df_window[df_window["dt"].isin(lookback_dates)].iterrows():
        kd = row["kdcode"]
        dt_val = row["dt"]
        if kd in stock_to_idx and dt_val in lookback_dates:
            s_idx = stock_to_idx[kd]
            d_idx = lookback_dates.index(dt_val)
            pivot[d_idx, s_idx, :] = row[feature_cols].values.astype(np.float32)

    time_series = pivot.transpose(1, 0, 2)
    time_series = time_series[np.newaxis, ...]

    graph_date = window_dates[-1]
    graph_features = np.zeros((1, n_stocks, n_features), dtype=np.float32)
    df_graph_day = df_window[df_window["dt"] == graph_date]
    for _, row in df_graph_day.iterrows():
        kd = row["kdcode"]
        if kd in stock_to_idx:
            s_idx = stock_to_idx[kd]
            graph_features[0, s_idx, :] = row[feature_cols].values.astype(np.float32)

    return time_series, graph_features
