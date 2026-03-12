"""
Model architectures for MCI-GRU experiments.

Modules:
- mci_gru: Main MCI-GRU model and components
"""

from mci_gru.models.mci_gru import (
    StockPredictionModel,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
    AttentionResetGRUCell,
    GATBlock,
    GATLayer,      # backward-compatible alias for GATBlock
    GATLayer_1,    # backward-compatible alias for GATBlock
    SelfAttention,
    MarketLatentStateLearner,
    create_model,
)

__all__ = [
    "StockPredictionModel",
    "ImprovedGRU",
    "MultiScaleTemporalEncoder",
    "AttentionResetGRUCell",
    "GATBlock",
    "GATLayer",
    "GATLayer_1",
    "SelfAttention",
    "MarketLatentStateLearner",
    "create_model",
]
