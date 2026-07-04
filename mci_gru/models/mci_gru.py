"""Compatibility re-export shim for legacy ``from mci_gru.models.mci_gru import ...`` imports."""

from mci_gru.models.attention import SelfAttention
from mci_gru.models.factory import create_model
from mci_gru.models.graph import GATBlock, GATLayer, GATLayer_1
from mci_gru.models.latent import MarketLatentStateLearner
from mci_gru.models.temporal import (
    AttentionResetGRUCell,
    CausalTransformerEncoder,
    GRUWithAttention,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
)
from mci_gru.models.trunk import StockPredictionModel

__all__ = [
    "AttentionResetGRUCell",
    "ImprovedGRU",
    "GRUWithAttention",
    "CausalTransformerEncoder",
    "MultiScaleTemporalEncoder",
    "GATBlock",
    "GATLayer",
    "GATLayer_1",
    "SelfAttention",
    "MarketLatentStateLearner",
    "StockPredictionModel",
    "create_model",
]
