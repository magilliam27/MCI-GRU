"""Unit tests for mci_gru.data.transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.data.transforms import (
    build_single_date_tensors,
    compute_zscore_norm_stats,
    impute_feature_nans_by_day,
    normalize_features_zscore,
)


def _sample_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kdcode": ["A", "B", "A", "B", "A", "B"],
            "dt": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
            ],
            "feat_a": [1.0, np.nan, 3.0, 5.0, np.nan, 7.0],
            "feat_b": [np.nan, 2.0, 4.0, np.nan, 6.0, 8.0],
        }
    )


def test_impute_feature_nans_by_day_mean_then_zero_fill() -> None:
    df = _sample_panel()
    out = impute_feature_nans_by_day(df, ["feat_a", "feat_b"])

    day1 = out[out["dt"] == "2020-01-01"]
    assert day1.loc[day1["kdcode"] == "B", "feat_a"].iloc[0] == pytest.approx(1.0)
    assert day1.loc[day1["kdcode"] == "A", "feat_b"].iloc[0] == pytest.approx(2.0)

    day2 = out[out["dt"] == "2020-01-02"]
    assert day2.loc[day2["kdcode"] == "B", "feat_b"].iloc[0] == pytest.approx(4.0)

    day3 = out[out["dt"] == "2020-01-03"]
    assert day3.loc[day3["kdcode"] == "A", "feat_a"].iloc[0] == pytest.approx(7.0)
    assert not out[["feat_a", "feat_b"]].isna().any().any()


def test_compute_zscore_norm_stats_train_period_only() -> None:
    df = pd.DataFrame(
        {
            "dt": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "feat": [0.0, 2.0, 100.0, 200.0],
        }
    )
    means, stds = compute_zscore_norm_stats(df, ["feat"], train_end="2020-01-02")

    assert means["feat"] == pytest.approx(1.0)
    assert stds["feat"] == pytest.approx(np.std([0.0, 2.0], ddof=1))
    assert "feat" in means
    assert compute_zscore_norm_stats(df, ["missing"], "2020-01-02") == ({}, {})


def test_normalize_features_zscore_clip_and_scale() -> None:
    df = pd.DataFrame({"dt": ["2020-01-01"], "feat": [10.0]})
    means = {"feat": 0.0}
    stds = {"feat": 2.0}

    out = normalize_features_zscore(df, ["feat"], means, stds, clip_sigma=3.0)
    assert out["feat"].iloc[0] == pytest.approx(3.0)

    out2 = normalize_features_zscore(
        pd.DataFrame({"dt": ["2020-01-01"], "feat": [1.0]}),
        ["feat"],
        {"feat": 2.0},
        {"feat": 1.0},
    )
    assert out2["feat"].iloc[0] == pytest.approx(-1.0)


def test_normalize_features_zscore_default_fallback_vs_keyerror() -> None:
    df = pd.DataFrame({"dt": ["2020-01-01"], "feat": [5.0], "other": [1.0]})
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    out = normalize_features_zscore(df, ["feat"], means, stds)
    assert out["feat"].iloc[0] == pytest.approx(3.0)

    with pytest.raises(KeyError):
        normalize_features_zscore(
            df,
            ["feat"],
            means,
            stds,
            default_mean=None,
            default_std=None,
        )


def test_build_single_date_tensors_shapes() -> None:
    dates = [f"2020-01-{d:02d}" for d in range(1, 8)]
    rows = []
    for dt in dates:
        for kd in ["A", "B"]:
            rows.append(
                {
                    "kdcode": kd,
                    "dt": dt,
                    "f0": float(dates.index(dt)),
                    "f1": float(dates.index(dt) + 1),
                }
            )
    df_norm = pd.DataFrame(rows)
    kdcode_list = ["A", "B"]
    feature_cols = ["f0", "f1"]
    target_date = "2020-01-05"
    his_t = 3

    time_series, graph_features = build_single_date_tensors(
        df_norm,
        kdcode_list,
        feature_cols,
        his_t,
        target_date,
    )

    assert time_series.shape == (1, 2, 3, 2)
    assert graph_features.shape == (1, 2, 2)
    assert time_series.dtype == np.float32
    assert graph_features.dtype == np.float32

    lookback_start = dates[dates.index(target_date) - his_t]
    assert time_series[0, 0, 0, 0] == pytest.approx(float(dates.index(lookback_start)))
    assert graph_features[0, 0, 0] == pytest.approx(float(dates.index(target_date)))
