import pandas as pd
import pytest
from omegaconf import OmegaConf

from mci_gru.config import FeatureConfig, get_preset
from mci_gru.features.registry import FeatureEngineer
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


def _make_long_history_df() -> pd.DataFrame:
    """Build 252-day histories that hit each cycle under the live 21/252 windows."""
    rows = []
    end_date = pd.Timestamp("2020-12-31")
    dates = pd.bdate_range(end_date - pd.offsets.BDay(252), end_date)
    scenarios = {
        "BULL": [0.0] * 231 + [0.01] * 21,
        "CORRECTION": [0.002] * 231 + [-0.01] * 21,
        "BEAR": [0.0] * 231 + [-0.01] * 21,
        "REBOUND": [-0.002] * 231 + [0.01] * 21,
    }
    for kdcode, returns in scenarios.items():
        close = 100.0
        rows.append(
            {
                "kdcode": kdcode,
                "dt": dates[0].strftime("%Y-%m-%d"),
                "close": close,
                "open": close,
                "high": close,
                "low": close,
                "volume": 1_000_000,
            }
        )
        for dt, daily_return in zip(dates[1:], returns):
            close *= 1.0 + daily_return
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "close": close,
                    "open": close,
                    "high": close,
                    "low": close,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


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


def test_feature_engineer_dynamic_mode_works_with_live_windows():
    fe = FeatureEngineer(
        include_momentum=True,
        include_weekly_momentum=False,
        momentum_encoding="binary",
        momentum_blend_mode="dynamic",
        momentum_dynamic_correction_fast_weight=0.15,
        momentum_dynamic_rebound_fast_weight=0.70,
    )
    out = fe.transform(_make_long_history_df())
    snapshot = out[out["dt"] == out["dt"].max()].set_index("kdcode").sort_index()

    assert snapshot.loc["BULL", "momentum_blend"] == pytest.approx(1.0)
    assert snapshot.loc["BEAR", "momentum_blend"] == pytest.approx(-1.0)
    assert snapshot.loc["CORRECTION", "momentum_blend"] == pytest.approx(0.70)
    assert snapshot.loc["REBOUND", "momentum_blend"] == pytest.approx(0.40)


def test_hydra_dynamic_experiment_yaml_merges_into_feature_config():
    base = OmegaConf.load("configs/features/with_momentum.yaml")
    override = OmegaConf.load("configs/experiment/momentum_dynamic.yaml")
    merged = OmegaConf.merge({"features": base}, override)
    cfg = FeatureConfig(**OmegaConf.to_container(merged.features, resolve=True))

    assert cfg.momentum_blend_mode == "dynamic"
    assert cfg.momentum_blend_fast_weight == pytest.approx(0.5)
    assert cfg.momentum_dynamic_correction_fast_weight == pytest.approx(0.15)
    assert cfg.momentum_dynamic_rebound_fast_weight == pytest.approx(0.70)
