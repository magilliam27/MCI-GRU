import numpy as np
import pandas as pd
import pytest

from mci_gru.config import FeatureConfig
from mci_gru.features.registry import FeatureEngineer, build_feature_list
from mci_gru.features.volatility import (
    add_volatility_targeting_features,
    get_volatility_targeting_features,
)


def _make_panel(periods: int = 36) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows = []
    scenarios = {
        "AAA": np.linspace(0.004, 0.022, periods),
        "BBB": np.linspace(-0.003, 0.018, periods),
    }
    for kdcode, returns in scenarios.items():
        close = 100.0 if kdcode == "AAA" else 80.0
        for dt, daily_return in zip(dates, returns, strict=True):
            close *= 1.0 + daily_return
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _expected_ewm_vol(close: pd.Series, half_life: int, target_vol: float) -> pd.Series:
    lagged_returns = close.pct_change().shift(2)
    vol = (
        lagged_returns.ewm(halflife=half_life, min_periods=2, adjust=False)
        .std(bias=False)
        .mul(np.sqrt(252))
    )
    return vol.fillna(target_vol)


def test_harvey_style_volatility_targeting_feature_values():
    panel = _make_panel()

    out = add_volatility_targeting_features(
        panel,
        half_lives=[2, 4],
        target_vol=0.10,
        scale_clip=(0.25, 4.0),
        interaction_return_window=3,
    )
    aaa = out[out["kdcode"] == "AAA"].reset_index(drop=True)

    expected_vol_hl2 = _expected_ewm_vol(aaa["close"], half_life=2, target_vol=0.10)
    expected_vol_hl4 = _expected_ewm_vol(aaa["close"], half_life=4, target_vol=0.10)
    expected_scale_hl2 = (0.10 / (expected_vol_hl2 + 1e-8)).clip(0.25, 4.0)
    expected_scale_hl4 = (0.10 / (expected_vol_hl4 + 1e-8)).clip(0.25, 4.0)
    expected_vol_change = expected_vol_hl2 / (expected_vol_hl4 + 1e-8) - 1.0
    expected_vol_of_vol = (
        expected_vol_hl2.diff()
        .ewm(halflife=2, min_periods=2, adjust=False)
        .std(bias=False)
        .mul(np.sqrt(252))
        .fillna(0.0)
    )
    expected_ret3_x_scale = aaa["close"].pct_change(3).shift(2).fillna(0.0) * expected_scale_hl2

    pd.testing.assert_series_equal(
        aaa["vol_target_ewm_vol_hl2"],
        expected_vol_hl2,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        aaa["vol_target_scale_hl2"],
        expected_scale_hl2,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        aaa["vol_target_scale_hl4"],
        expected_scale_hl4,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        aaa["vol_target_vol_change_hl2_hl4"],
        expected_vol_change,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        aaa["vol_target_vol_of_vol_hl2"],
        expected_vol_of_vol,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        aaa["vol_target_ret3_lag2_x_scale_hl2"],
        expected_ret3_x_scale,
        check_names=False,
    )


def test_volatility_targeting_uses_harvey_ex_ante_lag():
    panel = _make_panel()
    changed = panel.copy()
    aaa_dates = changed.loc[changed["kdcode"] == "AAA", "dt"].tolist()
    changed_date = aaa_dates[20]
    changed.loc[
        (changed["kdcode"] == "AAA") & (changed["dt"] == changed_date),
        "close",
    ] *= 100.0

    base_out = add_volatility_targeting_features(
        panel,
        half_lives=[2, 4],
        interaction_return_window=3,
    )
    changed_out = add_volatility_targeting_features(
        changed,
        half_lives=[2, 4],
        interaction_return_window=3,
    )

    unaffected_dates = set(aaa_dates[:22])
    cols = get_volatility_targeting_features(
        half_lives=[2, 4],
        interaction_return_window=3,
    )
    base_aaa = base_out[
        (base_out["kdcode"] == "AAA") & (base_out["dt"].isin(unaffected_dates))
    ].reset_index(drop=True)
    changed_aaa = changed_out[
        (changed_out["kdcode"] == "AAA") & (changed_out["dt"].isin(unaffected_dates))
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(base_aaa[cols], changed_aaa[cols])


def test_volatility_targeting_is_per_stock_isolated():
    panel = _make_panel()
    changed = panel.copy()
    changed.loc[changed["kdcode"] == "BBB", "close"] *= np.linspace(
        1.0,
        5.0,
        (changed["kdcode"] == "BBB").sum(),
    )

    base_out = add_volatility_targeting_features(panel, half_lives=[2, 4])
    changed_out = add_volatility_targeting_features(changed, half_lives=[2, 4])
    cols = get_volatility_targeting_features(half_lives=[2, 4])

    pd.testing.assert_frame_equal(
        base_out.loc[base_out["kdcode"] == "AAA", cols].reset_index(drop=True),
        changed_out.loc[changed_out["kdcode"] == "AAA", cols].reset_index(drop=True),
    )


def test_volatility_targeting_registry_and_config_wiring():
    feature_cols = get_volatility_targeting_features(
        half_lives=[20, 60, 90],
        interaction_return_window=21,
    )
    cfg = FeatureConfig(include_volatility_targeting=True)
    engineer = FeatureEngineer(cfg)

    assert feature_cols == [
        "vol_target_ewm_vol_hl20",
        "vol_target_ewm_vol_hl60",
        "vol_target_ewm_vol_hl90",
        "vol_target_scale_hl20",
        "vol_target_scale_hl60",
        "vol_target_scale_hl90",
        "vol_target_vol_change_hl20_hl90",
        "vol_target_vol_of_vol_hl20",
        "vol_target_ret21_lag2_x_scale_hl20",
    ]
    assert all(col in engineer.get_feature_columns() for col in feature_cols)
    assert all(
        col
        in build_feature_list(
            include_momentum=False,
            include_volatility_targeting=True,
        )
        for col in feature_cols
    )

    transformed = engineer.transform(_make_panel())
    assert all(col in transformed.columns for col in feature_cols)


def test_volatility_targeting_config_rejects_invalid_controls():
    with pytest.raises(ValueError, match="volatility_targeting_half_lives"):
        FeatureConfig(volatility_targeting_half_lives=[20, 20])

    with pytest.raises(ValueError, match="volatility_target_scale_clip"):
        FeatureConfig(volatility_target_scale_clip=[4.0, 0.25])

    with pytest.raises(ValueError, match="volatility_target_vol"):
        FeatureConfig(volatility_target_vol=0.0)
