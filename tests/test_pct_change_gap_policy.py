"""Return features treat an input gap as pandas 2 always did, under pandas 2 and 3 alike.

pandas 2's ``pct_change`` padded gaps forward by default. pandas 3 leaves them NaN,
silently. The owner chose pandas 2's behaviour for every call site (#245), so each
public boundary below must give the same return-derived output for a gapped input
as for that input with its gaps forward-filled (per stock where the source is a
panel). These tests must pass identically under pandas 2 and pandas 3.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.capacity import add_lagged_capacity_inputs
from mci_gru.features.credit import add_credit_features
from mci_gru.features.momentum import add_momentum_continuous
from mci_gru.features.volatility import (
    add_vix_features,
    add_volatility_features,
    add_volatility_targeting_features,
)
from mci_gru.graph.correlation import compute_correlation_matrix
from mci_gru.utils.returns import padded_pct_change

DATES = [f"2020-01-{day:02d}" for day in range(1, 13)]
# AAA has an interior two-day gap; BBB starts late and has a one-day gap.
CLOSES = {
    "AAA": [10.0, np.nan, np.nan, 12.0, 12.6, 12.0, 13.2, 13.0, 12.5, 13.5, 14.0, 13.3],
    "BBB": [np.nan, 50.0, 51.0, 49.0, np.nan, 52.0, 53.0, 51.5, 52.5, 54.0, 53.0, 55.0],
}


def _panel(*, filled: bool) -> pd.DataFrame:
    frames = []
    for code, closes in CLOSES.items():
        close = pd.Series(closes)
        frames.append(
            pd.DataFrame(
                {
                    "kdcode": code,
                    "dt": DATES,
                    "close": close.ffill() if filled else close,
                    "volume": 1000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _series(values: list[float], *, filled: bool) -> pd.Series:
    series = pd.Series(values, dtype=float)
    return series.ffill() if filled else series


def _assert_same_columns(gapped: pd.DataFrame, filled: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        pd.testing.assert_series_equal(
            gapped[column].reset_index(drop=True),
            filled[column].reset_index(drop=True),
            check_names=False,
            obj=column,
        )


def test_daily_return_across_a_gap_is_measured_from_the_last_observed_close() -> None:
    result = add_volatility_features(_panel(filled=False), short_window=2, long_window=3)
    aaa = result[result["kdcode"] == "AAA"]["_daily_return"].to_numpy()
    bbb = result[result["kdcode"] == "BBB"]["_daily_return"].to_numpy()
    # Gap days carry no change; the first day after a gap spans the whole gap.
    np.testing.assert_allclose(aaa[:5], [np.nan, 0.0, 0.0, 0.2, 0.05])
    np.testing.assert_allclose(
        bbb[:6], [np.nan, np.nan, 0.02, 49.0 / 51.0 - 1, 0.0, 52.0 / 49.0 - 1]
    )


def test_volatility_features_treat_a_gap_as_the_forward_filled_close() -> None:
    gapped = add_volatility_features(_panel(filled=False), short_window=2, long_window=3)
    filled = add_volatility_features(_panel(filled=True), short_window=2, long_window=3)
    _assert_same_columns(
        gapped, filled, ["_daily_return", "volatility_2d", "volatility_3d", "vol_ratio"]
    )


def test_volatility_targeting_features_treat_a_gap_as_the_forward_filled_close() -> None:
    gapped = add_volatility_targeting_features(_panel(filled=False), interaction_return_window=3)
    filled = add_volatility_targeting_features(_panel(filled=True), interaction_return_window=3)
    features = [column for column in gapped.columns if column.startswith("vol_target_")]
    assert any("_ret3_lag2_" in column for column in features)
    _assert_same_columns(gapped, filled, features)


def test_momentum_features_treat_a_gap_as_the_forward_filled_close() -> None:
    kwargs = {"fast_window": 2, "slow_window": 3, "dynamic_min_history": 1}
    gapped = add_momentum_continuous(_panel(filled=False), **kwargs)
    filled = add_momentum_continuous(_panel(filled=True), **kwargs)
    features = [
        column
        for column in gapped.columns
        if column not in {"kdcode", "dt", "close", "volume"}
        and pd.api.types.is_numeric_dtype(gapped[column])
    ]
    assert {"fast_momentum", "slow_momentum", "weekly_momentum"} <= set(features)
    _assert_same_columns(gapped, filled, features)


def test_correlation_matrix_treats_a_gap_as_the_forward_filled_close() -> None:
    args = (["AAA", "BBB"], "2020-01-13", 20)
    gapped = compute_correlation_matrix(_panel(filled=False), *args)
    filled = compute_correlation_matrix(_panel(filled=True), *args)
    pd.testing.assert_frame_equal(gapped, filled)


def test_capacity_volatility_treats_a_gap_as_the_forward_filled_close() -> None:
    gapped = add_lagged_capacity_inputs(_panel(filled=False), lookback_days=3)
    filled = add_lagged_capacity_inputs(_panel(filled=True), lookback_days=3)
    _assert_same_columns(gapped, filled, ["daily_return", "lagged_volatility"])


def test_vix_change_treats_a_gap_as_the_forward_filled_level() -> None:
    levels = [20.0, np.nan, 22.0, 21.0, np.nan, np.nan, 24.0]
    panel = pd.DataFrame({"kdcode": "AAA", "dt": DATES[:7], "close": 1.0})

    def vix_change(*, filled: bool) -> pd.Series:
        vix = pd.DataFrame({"dt": DATES[:7], "vix": _series(levels, filled=filled)})
        return add_vix_features(panel, vix)["vix_change"]

    pd.testing.assert_series_equal(vix_change(filled=False), vix_change(filled=True))
    np.testing.assert_allclose(vix_change(filled=False).iloc[2], 0.1)


@pytest.mark.parametrize("column", ["ig_spread", "hy_spread"])
def test_credit_spread_change_treats_a_gap_as_the_forward_filled_spread(column: str) -> None:
    spreads = [1.0, np.nan, 1.2, 1.1, np.nan, 1.32, 1.3]
    panel = pd.DataFrame({"kdcode": "AAA", "dt": DATES[:7], "close": 1.0})

    def change(*, filled: bool) -> pd.Series:
        credit = pd.DataFrame(
            {
                "dt": DATES[:7],
                "ig_spread": 1.0,
                "hy_spread": 4.0,
            }
        )
        credit[column] = _series(spreads, filled=filled)
        return add_credit_features(panel, credit)[f"{column}_change"]

    pd.testing.assert_series_equal(change(filled=False), change(filled=True))
    np.testing.assert_allclose(change(filled=False).iloc[2], 0.2)


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        (1, [np.nan, np.nan, 0.0, 0.0, 0.2, 0.05, np.nan, 0.0, 0.2, 0.0, 0.0]),
        (3, [np.nan, np.nan, np.nan, np.nan, 0.2, 0.26, np.nan, np.nan, np.nan, 0.2, 0.2]),
    ],
)
def test_padded_change_matches_pandas_2_default_within_each_group(periods, expected) -> None:
    # Values recorded from pandas 2.3.3's default groupby pct_change on this input.
    groups = pd.Series(list("aaaaaabbbbb"))
    values = pd.Series([np.nan, 10, np.nan, np.nan, 12, 12.6, 5, np.nan, 6, np.nan, np.nan])
    np.testing.assert_allclose(padded_pct_change(values, groups, periods=periods), expected)


def test_package_has_no_pct_change_call_without_an_explicit_fill_method() -> None:
    package = Path(__file__).resolve().parents[1] / "mci_gru"
    bare = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pct_change"
                and not any(keyword.arg == "fill_method" for keyword in node.keywords)
            ):
                bare.append(f"{path.relative_to(package.parent).as_posix()}:{node.lineno}")
    assert bare == []
