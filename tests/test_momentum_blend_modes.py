import pandas as pd
import pytest

from mci_gru.config import FeatureConfig, get_preset
from mci_gru.features.momentum import add_momentum_binary


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


def _state_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return the first row with valid slow momentum for each stock."""
    out = add_momentum_binary(
        _make_state_df(),
        fast_window=1,
        slow_window=2,
        include_weekly_momentum=False,
        **df,
    )
    return out[out["dt"] == "2020-01-03"].set_index("kdcode").sort_index()


def test_dynamic_blend_uses_cycle_specific_weights():
    snapshot = _state_snapshot(
        {
            "blend_mode": "dynamic",
            "blend_fast_weight": 0.5,
            "dynamic_correction_fast_weight": 0.15,
            "dynamic_rebound_fast_weight": 0.70,
        }
    )

    assert snapshot.loc["BULL", "momentum_blend"] == pytest.approx(1.0)
    assert snapshot.loc["BEAR", "momentum_blend"] == pytest.approx(-1.0)
    assert snapshot.loc["CORRECTION", "momentum_blend"] == pytest.approx(0.70)
    assert snapshot.loc["REBOUND", "momentum_blend"] == pytest.approx(0.40)


def test_static_blend_supports_custom_intermediate_speed():
    snapshot = _state_snapshot(
        {
            "blend_mode": "static",
            "blend_fast_weight": 0.25,
            "dynamic_correction_fast_weight": 0.15,
            "dynamic_rebound_fast_weight": 0.70,
        }
    )

    assert snapshot.loc["BULL", "momentum_blend"] == pytest.approx(1.0)
    assert snapshot.loc["BEAR", "momentum_blend"] == pytest.approx(-1.0)
    assert snapshot.loc["CORRECTION", "momentum_blend"] == pytest.approx(0.50)
    assert snapshot.loc["REBOUND", "momentum_blend"] == pytest.approx(-0.50)


def test_feature_config_validates_blend_controls_and_dynamic_preset():
    cfg = FeatureConfig(
        momentum_blend_mode="dynamic",
        momentum_dynamic_correction_fast_weight=0.15,
        momentum_dynamic_rebound_fast_weight=0.70,
    )
    assert cfg.momentum_blend_mode == "dynamic"

    preset = get_preset("momentum_dynamic")
    assert preset.features.momentum_blend_mode == "dynamic"
    assert preset.features.momentum_dynamic_correction_fast_weight == pytest.approx(0.15)
    assert preset.features.momentum_dynamic_rebound_fast_weight == pytest.approx(0.70)

    with pytest.raises(ValueError):
        FeatureConfig(momentum_blend_mode="adaptive")

    with pytest.raises(ValueError):
        FeatureConfig(momentum_blend_fast_weight=1.1)
