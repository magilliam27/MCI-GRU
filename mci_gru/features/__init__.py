"""
Feature engineering for MCI-GRU experiments.

Modules:
- base: Base OHLCV features and derived price/volume features
- momentum: Momentum features (binary, continuous, buffered)
- volatility: Volatility, VIX, RSI, and moving average features
- credit: Credit spread features (IG/HY OAS from FRED)
- regime: Global scalar regime features
- registry: FeatureEngineer pipeline and feature set composition
"""

from mci_gru.features.base import add_base_features
from mci_gru.features.momentum import (
    add_momentum_binary,
    add_momentum_buffered,
    add_momentum_continuous,
)
from mci_gru.features.regime import (
    REGIME_FEATURES,
    add_regime_features,
)
from mci_gru.features.registry import (
    FEATURE_SETS,
    FeatureEngineer,
    build_feature_list,
)
from mci_gru.features.volatility import (
    add_vix_features,
    add_volatility_features,
)

__all__ = [
    "add_base_features",
    "add_momentum_binary",
    "add_momentum_continuous",
    "add_momentum_buffered",
    "add_volatility_features",
    "add_vix_features",
    "add_regime_features",
    "REGIME_FEATURES",
    "FeatureEngineer",
    "build_feature_list",
    "FEATURE_SETS",
]
