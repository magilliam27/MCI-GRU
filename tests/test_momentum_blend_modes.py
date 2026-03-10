import pandas as pd
import pytest
from omegaconf import OmegaConf

from mci_gru.config import FeatureConfig, get_preset
from mci_gru.features.momentum import (
    _estimate_dynamic_fast_weights_for_group,
    _solve_dynamic_speed,
    add_momentum_binary,
)


def _make_state_df() -> pd.DataFrame:
    """Build four stocks whose 2-day windows map to Bull/Correction/Bear/Rebound."""
    rows = []
    scenarios = {
        "BULL": [0.10, 0.10],
        "CORRECTION": [0.40, -0.10],
        "BEAR": [-0.10, -0.10],
        "REBOUND": [-0.40, 0.10],
    }
    for kdcode, returns in scenarios.items():
        close = 100.0
        for idx, daily_return in enumerate(returns):
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": f"2020-01-0{idx + 1}",
                    "close": close,
                }
            )
            close *= 1.0 + daily_return
        rows.append(
            {
                "kdcode": kdcode,
                "dt": "2020-01-03",
                "close": close,
            }
        )
    return pd.DataFrame(rows)


def _state_snapshot(kwargs: dict) -> pd.DataFrame:
    """Return the first row with valid slow momentum for each stock."""
    out = add_momentum_binary(
        _make_state_df(),
        fast_window=1,
        slow_window=2,
        include_weekly_momentum=False,
        **kwargs,
    )
    return out[out["dt"] == "2020-01-03"].set_index("kdcode").sort_index()


def test_static_blend_supports_custom_intermediate_speed():
    snapshot = _state_snapshot(
        {
            "blend_mode": "static",
            "blend_fast_weight": 0.25,
        }
    )

    assert snapshot.loc["BULL", "momentum_blend"] == pytest.approx(1.0)
    assert snapshot.loc["BEAR", "momentum_blend"] == pytest.approx(-1.0)
    assert snapshot.loc["CORRECTION", "momentum_blend"] == pytest.approx(0.50)
    assert snapshot.loc["REBOUND", "momentum_blend"] == pytest.approx(-0.50)


def test_binary_momentum_keeps_warmup_rows_neutral():
    out = add_momentum_binary(
        _make_state_df(),
        fast_window=1,
        slow_window=2,
        blend_mode="static",
        include_weekly_momentum=False,
    )
    first_rows = out[out["dt"] == "2020-01-01"]
    second_rows = out[out["dt"] == "2020-01-02"]

    assert (first_rows["slow_signal"] == 0.0).all()
    assert (first_rows["fast_signal"] == 0.0).all()
    assert (first_rows["momentum_blend"] == 0.0).all()
    assert (first_rows[["cycle_bull", "cycle_correction", "cycle_bear"]] == 0.0).all().all()
    assert (second_rows["slow_signal"] == 0.0).all()
    assert (second_rows["momentum_blend"] == 0.0).all()
    assert (second_rows[["cycle_bull", "cycle_correction", "cycle_bear"]] == 0.0).all().all()


def test_dynamic_formula_matches_proposition_9():
    index = pd.Index([0])
    a_co, a_re = _solve_dynamic_speed(
        mu_bu=pd.Series([0.04], index=index),
        mu_be=pd.Series([-0.02], index=index),
        mu_co=pd.Series([0.01], index=index),
        mu_re=pd.Series([0.03], index=index),
        second_bu=pd.Series([0.09], index=index),
        second_be=pd.Series([0.16], index=index),
        second_co=pd.Series([0.04], index=index),
        second_re=pd.Series([0.09], index=index),
        p_bu=pd.Series([0.40], index=index),
        p_be=pd.Series([0.20], index=index),
        fallback_correction_fast_weight=0.15,
        fallback_rebound_fast_weight=0.70,
        valid_correction_mask=pd.Series([True], index=index),
        valid_rebound_mask=pd.Series([True], index=index),
    )

    assert a_co.iloc[0] == pytest.approx(0.275)
    assert a_re.iloc[0] == pytest.approx(0.7666666667)


def test_dynamic_estimator_uses_only_prior_observations():
    group = pd.DataFrame(
        {
            "kdcode": ["AAA"] * 6,
            "_daily_return": [0.00, 0.30, -0.20, 0.02, 0.15, 0.99],
        }
    )
    slow_position = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, 1.0], index=group.index)
    fast_position = pd.Series([1.0, -1.0, -1.0, 1.0, -1.0, 1.0], index=group.index)

    weights_a = _estimate_dynamic_fast_weights_for_group(
        group=group,
        slow_position=slow_position,
        fast_position=fast_position,
        fallback_correction_fast_weight=0.15,
        fallback_rebound_fast_weight=0.70,
        lookback_periods=0,
        min_history=1,
        min_state_observations=1,
    )

    group_changed = group.copy()
    group_changed.loc[5, "_daily_return"] = -0.99
    weights_b = _estimate_dynamic_fast_weights_for_group(
        group=group_changed,
        slow_position=slow_position,
        fast_position=fast_position,
        fallback_correction_fast_weight=0.15,
        fallback_rebound_fast_weight=0.70,
        lookback_periods=0,
        min_history=1,
        min_state_observations=1,
    )

    # Weight at row 4 is applied using information available through row 4 only.
    assert weights_a.iloc[4] == pytest.approx(weights_b.iloc[4])


def test_dynamic_estimator_activates_after_sufficient_history():
    group = pd.DataFrame(
        {
            "kdcode": ["AAA"] * 16,
            # forward_return at row t is _daily_return at row t+1
            "_daily_return": [
                0.00,
                0.04, -0.02, 0.01, 0.03,
                0.04, -0.02, 0.01, 0.03,
                0.04, -0.02, 0.01, 0.03,
                0.04, -0.02, 0.01,
            ],
        }
    )
    slow_position = pd.Series([1.0, -1.0, 1.0, -1.0] * 4, index=group.index)
    fast_position = pd.Series([1.0, -1.0, -1.0, 1.0] * 4, index=group.index)

    weights = _estimate_dynamic_fast_weights_for_group(
        group=group,
        slow_position=slow_position,
        fast_position=fast_position,
        fallback_correction_fast_weight=0.15,
        fallback_rebound_fast_weight=0.70,
        lookback_periods=0,
        min_history=4,
        min_state_observations=3,
    )

    # By the 4th Correction/Rebound observation there is enough prior history
    # for the paper-style estimator, so weights should move off the fallbacks.
    # In this synthetic setup, the Correction estimate clips to 0.0 and the
    # Rebound estimate lands at 0.611111..., both distinct from 0.15 / 0.70.
    assert weights.iloc[14] == pytest.approx(0.0)
    assert weights.iloc[15] == pytest.approx(0.6111111111)


def test_feature_config_validates_dynamic_controls_and_yaml_merge():
    cfg = FeatureConfig(
        momentum_blend_mode="dynamic",
        momentum_dynamic_correction_fast_weight=0.15,
        momentum_dynamic_rebound_fast_weight=0.70,
        momentum_dynamic_lookback_periods=0,
        momentum_dynamic_min_history=252,
        momentum_dynamic_min_state_observations=3,
    )
    assert cfg.momentum_blend_mode == "dynamic"

    preset = get_preset("momentum_dynamic")
    assert preset.features.momentum_blend_mode == "dynamic"
    assert preset.features.momentum_dynamic_lookback_periods == 0
    assert preset.features.momentum_dynamic_min_history == 252
    assert preset.features.momentum_dynamic_min_state_observations == 3

    base = OmegaConf.load("configs/features/with_momentum.yaml")
    override = OmegaConf.load("configs/experiment/momentum_dynamic.yaml")
    merged = OmegaConf.merge({"features": base}, override)
    merged_cfg = FeatureConfig(**OmegaConf.to_container(merged.features, resolve=True))

    assert merged_cfg.momentum_blend_mode == "dynamic"
    assert merged_cfg.momentum_dynamic_correction_fast_weight == pytest.approx(0.15)
    assert merged_cfg.momentum_dynamic_rebound_fast_weight == pytest.approx(0.70)
    assert merged_cfg.momentum_dynamic_lookback_periods == 0
    assert merged_cfg.momentum_dynamic_min_history == 252
    assert merged_cfg.momentum_dynamic_min_state_observations == 3

    with pytest.raises(ValueError):
        FeatureConfig(momentum_blend_mode="adaptive")

    with pytest.raises(ValueError):
        FeatureConfig(momentum_dynamic_min_history=0)
