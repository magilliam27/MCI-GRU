"""Regression tests for vectorised preprocessing (matches legacy semantics)."""

import numpy as np
import pandas as pd
import pytest

from mci_gru.data.preprocessing import generate_time_series_features


def _legacy_generate_time_series_features(
    df: pd.DataFrame,
    kdcode_list: list[str],
    feature_cols: list[str],
    his_t: int,
) -> np.ndarray:
    """Reference implementation (iterrows) for equivalence checks."""
    all_dates = sorted(df["dt"].unique())
    num_stocks = len(kdcode_list)
    num_features = len(feature_cols)
    num_usable_days = len(all_dates) - his_t

    stock_features = np.zeros((num_usable_days, num_stocks, his_t, num_features), dtype=np.float32)

    stock_to_idx = {stock: idx for idx, stock in enumerate(kdcode_list)}
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}

    df_subset = df[df["kdcode"].isin(kdcode_list)][["kdcode", "dt"] + feature_cols].copy()
    pivot_data = np.zeros((len(all_dates), num_stocks, num_features), dtype=np.float32)

    for _, row in df_subset.iterrows():
        kdcode = row["kdcode"]
        dt = row["dt"]
        if kdcode in stock_to_idx and dt in date_to_idx:
            stock_idx = stock_to_idx[kdcode]
            date_idx = date_to_idx[dt]
            pivot_data[date_idx, stock_idx, :] = row[feature_cols].values.astype(np.float32)

    for day_offset in range(num_usable_days):
        stock_features[day_offset, :, :, :] = pivot_data[
            day_offset : day_offset + his_t, :, :
        ].transpose(1, 0, 2)

    return stock_features


def test_generate_time_series_features_matches_legacy_small_grid():
    """Vectorised path must match iterrows reference on a tiny panel."""
    dates = [f"2020-01-{d:02d}" for d in range(1, 11)]
    kdcodes = ["A", "B", "C"]
    feature_cols = ["f1", "f2"]
    rows = []
    rng = np.random.default_rng(0)
    for dt in dates:
        for k in kdcodes:
            rows.append(
                {
                    "kdcode": k,
                    "dt": dt,
                    "f1": float(rng.random()),
                    "f2": float(rng.random()),
                }
            )
    df = pd.DataFrame(rows)
    his_t = 3

    got = generate_time_series_features(df, kdcodes, feature_cols, his_t)
    want = _legacy_generate_time_series_features(df, kdcodes, feature_cols, his_t)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_generate_time_series_features_duplicate_dt_kdcode_last_wins():
    """When duplicate (dt, kdcode) rows exist, last row wins (legacy iterrows behaviour)."""
    df = pd.DataFrame(
        [
            {"kdcode": "X", "dt": "2020-01-01", "f1": 1.0, "f2": 0.0},
            {"kdcode": "X", "dt": "2020-01-01", "f1": 99.0, "f2": 1.0},
            {"kdcode": "X", "dt": "2020-01-02", "f1": 2.0, "f2": 0.0},
            {"kdcode": "X", "dt": "2020-01-03", "f1": 3.0, "f2": 0.0},
            {"kdcode": "X", "dt": "2020-01-04", "f1": 4.0, "f2": 0.0},
        ]
    )
    kdcode_list = ["X"]
    feature_cols = ["f1", "f2"]
    his_t = 2

    got = generate_time_series_features(df, kdcode_list, feature_cols, his_t)
    want = _legacy_generate_time_series_features(df, kdcode_list, feature_cols, his_t)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("his_t", [1, 2, 5])
def test_generate_time_series_features_shape(his_t: int):
    """Output shape matches (num_dates - his_t, n_stocks, his_t, n_features)."""
    dates = [f"2020-01-{d:02d}" for d in range(1, 21)]
    stocks = ["S1", "S2"]
    feature_cols = ["a", "b"]
    rows = []
    for dt in dates:
        for s in stocks:
            rows.append({"kdcode": s, "dt": dt, "a": 1.0, "b": 2.0})
    df = pd.DataFrame(rows)
    out = generate_time_series_features(df, stocks, feature_cols, his_t)
    assert out.shape == (len(dates) - his_t, len(stocks), his_t, len(feature_cols))
