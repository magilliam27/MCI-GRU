"""
Data preprocessing utilities for MCI-GRU.

Contains pure data-transformation functions extracted from run_experiment.py:
- generate_time_series_features: sliding-window tensor construction
- generate_graph_features: per-day graph node features
- compute_labels: forward-return label computation
- apply_rank_labels: cross-sectional rank percentile conversion
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def generate_time_series_features(
    df: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    his_t: int,
) -> np.ndarray:
    """Build sliding-window feature tensors for all stocks.

    Returns array of shape (num_usable_days, num_stocks, his_t, num_features).
    """
    all_dates = sorted(df["dt"].unique())
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)
    num_usable_days = len(all_dates) - his_t

    print(f"  Allocating feature array: ({num_usable_days}, {num_stocks}, {his_t}, {num_features})")

    df_subset = df[df["kdcode"].isin(kdcode_list)][["kdcode", "dt"] + feature_cols].copy()
    # Last row wins for duplicate (dt, kdcode), matching legacy iterrows overwrite semantics.
    df_subset = df_subset.drop_duplicates(subset=["dt", "kdcode"], keep="last")

    pivot_data = np.zeros((len(all_dates), num_stocks, num_features), dtype=np.float32)
    for fi, col in enumerate(
        tqdm(feature_cols, desc="  Building pivot (per-feature)", leave=False)
    ):
        wide = df_subset.pivot_table(
            index="dt",
            columns="kdcode",
            values=col,
            aggfunc="last",
            fill_value=0.0,
        )
        wide = wide.reindex(index=all_dates, columns=kdcode_list, fill_value=0.0)
        pivot_data[:, :, fi] = wide.to_numpy(dtype=np.float32, copy=False)

    # (T, S, F) -> sliding windows along time -> (T - his_t + 1, S, F, his_t) -> keep num_usable_days
    windows = sliding_window_view(pivot_data, his_t, axis=0)
    windows = windows[:num_usable_days, ...]
    # (num_usable_days, S, F, his_t) -> (num_usable_days, S, his_t, F)
    stock_features = np.transpose(windows, (0, 1, 3, 2)).astype(np.float32, copy=False)

    return stock_features


def generate_graph_features(
    df: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    dates: list[str],
) -> np.ndarray:
    """Build per-day graph node feature tensors.

    Returns array of shape (num_dates, num_stocks, num_features).
    """
    num_dates = len(dates)
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)

    x_graph = np.zeros((num_dates, num_stocks, num_features), dtype=np.float32)
    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_list)}

    df_subset = df[df["dt"].isin(dates) & df["kdcode"].isin(kdcode_list)]

    for date_idx, date in enumerate(dates):
        df_day = df_subset[df_subset["dt"] == date]
        for _, row in df_day.iterrows():
            stock_idx = stock_to_idx.get(row["kdcode"])
            if stock_idx is not None:
                x_graph[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)

    return x_graph


def apply_rank_labels(labels: np.ndarray) -> np.ndarray:
    """Convert raw return labels to cross-sectional rank percentiles per day.

    Each day's returns are ranked across stocks and divided by the stock count
    to yield percentiles in (0, 1].  Only same-day information is used, so this
    does **not** introduce look-ahead bias.
    """
    from scipy.stats import rankdata

    ranked = np.empty_like(labels)
    for i in range(labels.shape[0]):
        ranked[i] = rankdata(labels[i]) / labels.shape[1]
    return ranked.astype(np.float32)


def compute_labels(
    df: pd.DataFrame,
    kdcode_list: list[str],
    dates: list[str],
    label_t: int,
) -> np.ndarray:
    """Compute forward-return labels for the given dates.

    For each (stock, date) pair the label is:
        close[date + label_t] / close[date + 1] - 1

    NaN labels (e.g. near the end of the dataset) are filled with the
    cross-sectional mean for that day, then with zero as a final fallback.
    """
    df_subset = df[df["kdcode"].isin(kdcode_list)].copy()
    df_subset = df_subset.sort_values(["kdcode", "dt"])

    df_subset["future_close"] = df_subset.groupby("kdcode")["close"].shift(-label_t)
    df_subset["next_close"] = df_subset.groupby("kdcode")["close"].shift(-1)
    df_subset["forward_return"] = df_subset["future_close"] / df_subset["next_close"] - 1

    df_subset = df_subset[df_subset["dt"].isin(dates)]
    pivot = df_subset.pivot_table(index="dt", columns="kdcode", values="forward_return")
    pivot = pivot.reindex(index=dates, columns=kdcode_list)

    for date in dates:
        if date in pivot.index:
            row_mean = pivot.loc[date].mean()
            pivot.loc[date] = pivot.loc[date].fillna(row_mean)
    pivot = pivot.fillna(0)

    return pivot.values.astype(np.float32)
