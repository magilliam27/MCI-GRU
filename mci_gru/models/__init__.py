"""
Model architectures for MCI-GRU experiments.

Components (split across temporal/graph/attention/latent/trunk/factory;
mci_gru.models.mci_gru remains as a compatibility re-export shim):
- AttentionResetGRUCell / ImprovedGRU / MultiScaleTemporalEncoder: temporal encoding
- GATBlock: unified two-layer Graph Attention block (replaces GATLayer, GATLayer_1)
- SelfAttention: optional cross-stock feature mixing
- MarketLatentStateLearner: multi-head cross-attention for latent market states
- StockPredictionModel: full model combining all components
"""

from mci_gru.models.attention import ResidualCrossSectionBlock, SelfAttention
from mci_gru.models.factory import create_model
from mci_gru.models.graph import GATBlock, GATLayer, GATLayer_1
from mci_gru.models.latent import MarketLatentStateLearner
from mci_gru.models.temporal import (
    AttentionResetGRUCell,
    GRUWithAttention,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
)
from mci_gru.models.trunk import StockPredictionModel

__all__ = [
    "StockPredictionModel",
    "ImprovedGRU",
    "GRUWithAttention",
    "MultiScaleTemporalEncoder",
    "AttentionResetGRUCell",
    "GATBlock",
    "GATLayer",
    "GATLayer_1",
    "SelfAttention",
    "ResidualCrossSectionBlock",
    "MarketLatentStateLearner",
    "create_model",
]
