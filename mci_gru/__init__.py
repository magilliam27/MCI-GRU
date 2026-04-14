"""
MCI-GRU: Multi-head Cross-attention and Improved GRU for Stock Prediction.

A modular experiment framework for testing stock prediction models with:
- Dynamic correlation graph updates
- Multiple stock universes (S&P 500, Russell 1000, MSCI World)
- LSEG/Refinitiv data integration
- Configurable feature engineering (momentum, volatility, VIX)
- Hydra-based experiment configuration
"""

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)

__version__ = "0.1.0"
__all__ = [
    "DataConfig",
    "FeatureConfig",
    "GraphConfig",
    "ModelConfig",
    "TrainingConfig",
    "TrackingConfig",
    "ExperimentConfig",
]
