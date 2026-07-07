import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


def _make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported GAT activation: {name!r}. Choose 'elu' or 'relu'.")


class GATBlock(nn.Module):
    """Two-layer Graph Attention block.

    Layer 1: multi-head GAT (in_channels → hidden × heads, concatenated)
    Layer 2: single-head GAT (hidden × heads → out_channels)

    When ``inter_layer_dropout > 0`` and training, dropout is applied on node
    features between the two GAT convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        heads: int = 1,
        activation: str = "elu",
        edge_feature_dim: int = 1,
        inter_layer_dropout: float = 0.0,
    ):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.gat1 = GATConv(
            in_channels, hidden, heads=heads, concat=True, edge_dim=edge_feature_dim
        )
        self.gat2 = GATConv(
            hidden * heads, out_channels, heads=1, concat=False, edge_dim=edge_feature_dim
        )
        self.act = _make_activation(activation)
        self.inter_layer_dropout = inter_layer_dropout
        if inter_layer_dropout > 0.0:
            self.drop_mid = nn.Dropout(inter_layer_dropout)
        else:
            self.drop_mid = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        x = self.gat1(x, edge_index, edge_weight)
        x = self.act(x)
        if self.drop_mid is not None and self.training:
            x = self.drop_mid(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


# Backward-compatible aliases
GATLayer = GATBlock
GATLayer_1 = GATBlock
