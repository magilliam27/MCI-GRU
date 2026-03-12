"""
Model architectures for MCI-GRU experiments.

Components (all in mci_gru.models.mci_gru):
- AttentionResetGRUCell / ImprovedGRU / MultiScaleTemporalEncoder: temporal encoding
- GATBlock: unified two-layer Graph Attention block (replaces GATLayer, GATLayer_1)
- SelfAttention: optional cross-stock feature mixing
- MarketLatentStateLearner: multi-head cross-attention for latent market states
- StockPredictionModel: full model combining all components
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
