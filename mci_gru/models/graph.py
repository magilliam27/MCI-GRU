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

    With ``residual=False`` (the legacy form) a node's own features reach its
    output only as a self-loop neighbour, and GAT v1's attention ranks a node's
    sources the same way whatever the node is, so nodes with the same closed
    neighbourhood get (near) identical outputs whatever their own inputs.
    ``residual=True`` adds a learned linear map of each node's own input to both
    layers' outputs (``GATConv(residual=True)``), so a linked node keeps its own
    signal (issue #320). The two forms have different parameters, so a
    checkpoint belongs to the form that produced it.
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
        residual: bool = False,
    ):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.gat1 = GATConv(
            in_channels,
            hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_feature_dim,
            residual=residual,
        )
        self.gat2 = GATConv(
            hidden * heads,
            out_channels,
            heads=1,
            concat=False,
            edge_dim=edge_feature_dim,
            residual=residual,
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
