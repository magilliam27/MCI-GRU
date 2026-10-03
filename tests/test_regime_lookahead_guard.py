"""Regime forward context is refused unless the matching excludes the current month.

``regime_include_subsequent_returns=True`` emits returns that follow each similar
historical month. With ``regime_exclusion_months=0`` the current month can match
itself, and its subsequent return is the future. Two copies of the guard refuse that
combination: ``FeatureConfig.__post_init__`` at config time and
``compute_regime_monthly_features`` at computation time. Until issue 251 nothing
pinned either one: removing both left the suite green.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import FeatureConfig, create_config_from_dict
from mci_gru.features import FeatureEngineer
from mci_gru.features.regime import add_regime_features, compute_regime_monthly_features

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
GUARD_MESSAGE = r"exclusion_months (must be )?>= 1"


def _regime_daily() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=1200, freq="D")
    x = np.linspace(0, 12, len(dates))
    return pd.DataFrame(
        {
            "dt": dates.strftime("%Y-%m-%d"),
            "regime_market": 1000 + 25 * np.sin(x) + np.linspace(0, 100, len(dates)),
            "regime_yield_curve": 1.5 + 0.2 * np.cos(x / 1.7),
            "regime_oil": 60 + 8 * np.sin(x / 1.3),
            "regime_copper": 3.2 + 0.3 * np.cos(x / 2.3),
            "regime_stock_bond_corr": -0.2 + 0.35 * np.sin(x / 2.0),
        }
    )


def _stock_panel() -> pd.DataFrame:
    dates = pd.date_range("2002-01-01", periods=30, freq="D").strftime("%Y-%m-%d")
    rows = []
    for kdcode, offset in [("AAA", 0.0), ("BBB", 20.0)]:
        for day, dt in enumerate(dates):
            close = 100.0 + day + offset
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1000.0,
                    "turnover": close * 1000,
                }
            )
    return pd.DataFrame(rows)


def test_feature_config_refuses_subsequent_returns_without_exclusion() -> None:
    with pytest.raises(ValueError, match=GUARD_MESSAGE):
        FeatureConfig(regime_include_subsequent_returns=True, regime_exclusion_months=0)


def test_feature_config_accepts_the_combinations_the_guard_allows() -> None:
    """The guard is specific: it refuses only forward context with no exclusion."""
    FeatureConfig(regime_include_subsequent_returns=True, regime_exclusion_months=1)
    FeatureConfig(regime_include_subsequent_returns=False, regime_exclusion_months=0)


def test_create_config_from_dict_refuses_subsequent_returns_without_exclusion() -> None:
    with pytest.raises(ValueError, match=GUARD_MESSAGE):
        create_config_from_dict(
            {
                "features": {
                    "regime_include_subsequent_returns": True,
                    "regime_exclusion_months": 0,
                }
            }
        )


def test_hydra_override_refuses_subsequent_returns_without_exclusion() -> None:
    """The path ``run_experiment.py`` takes: ``compose`` then ``create_config_from_dict``."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "features.include_global_regime=true",
                "features.regime_include_subsequent_returns=true",
                "features.regime_exclusion_months=0",
            ],
        )
    with pytest.raises(ValueError, match=GUARD_MESSAGE):
        create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


def test_regime_computation_refuses_subsequent_returns_without_exclusion() -> None:
    with pytest.raises(ValueError, match=GUARD_MESSAGE):
        compute_regime_monthly_features(
            regime_df=_regime_daily(),
            exclusion_months=0,
            include_subsequent_returns=True,
        )


def test_regime_computation_accepts_the_combinations_the_guard_allows() -> None:
    monthly = compute_regime_monthly_features(
        regime_df=_regime_daily(), exclusion_months=1, include_subsequent_returns=True
    )
    current_only = compute_regime_monthly_features(
        regime_df=_regime_daily(), exclusion_months=0, include_subsequent_returns=False
    )
    assert len(monthly) and len(current_only)


def test_feature_engineer_refuses_the_combination_when_the_config_guard_is_bypassed() -> None:
    """Keyword construction skips ``FeatureConfig``; the computation-time copy still refuses."""
    engineer = FeatureEngineer(
        include_momentum=False,
        include_weekly_momentum=False,
        include_global_regime=True,
        regime_include_subsequent_returns=True,
        regime_exclusion_months=0,
    )
    with pytest.raises(ValueError, match=GUARD_MESSAGE):
        engineer.transform(_stock_panel(), regime_df=_regime_daily())


def test_add_regime_features_refuses_subsequent_returns_without_exclusion() -> None:
    with pytest.raises(ValueError, match=GUARD_MESSAGE):
        add_regime_features(
            _stock_panel(),
            _regime_daily(),
            exclusion_months=0,
            include_subsequent_returns=True,
        )
