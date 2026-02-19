"""
Feature engineering for MCI-GRU experiments.

Modules:
- base: Base OHLCV features
- momentum: Momentum features (binary, continuous, buffered)
- volatility: Volatility and VIX features
- registry: Feature set registry and composition
"""

from mci_gru.features.base import add_base_features
from mci_gru.features.momentum import (
    add_momentum_binary,
    add_momentum_continuous,
    add_momentum_buffered,
)
from mci_gru.features.volatility import (
    add_volatility_features,
    add_vix_features,
)
from mci_gru.features.regime import (
    add_regime_features,
    REGIME_FEATURES,
)
from mci_gru.features.registry import (
    FeatureEngineer,
    build_feature_list,
    FEATURE_SETS,
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
