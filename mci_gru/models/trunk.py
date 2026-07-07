import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge

from mci_gru.models.attention import SelfAttention
from mci_gru.models.graph import GATBlock
from mci_gru.models.latent import MarketLatentStateLearner
from mci_gru.models.temporal import (
    CausalTransformerEncoder,
    GRUWithAttention,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
)


def _make_output_activation(name: str) -> nn.Module:
    if name == "none" or name == "identity":
        return nn.Identity()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(
        f"Unsupported output activation: {name!r}. Choose 'none', 'elu', 'relu', or 'sigmoid'."
    )


def _maybe_ln_ch(dim: int, on: bool) -> nn.Module:
    return nn.LayerNorm(dim) if on else nn.Identity()


def _maybe_drop(p: float, on: bool) -> nn.Module:
    if not on or p <= 0.0:
        return nn.Identity()
    return nn.Dropout(p)


def _apply_edge_dropout(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (not training) or p <= 0.0 or p >= 1.0 or edge_index.numel() == 0:
        return edge_index, edge_weight
    e_new, edge_mask = dropout_edge(edge_index, p=p, force_undirected=False, training=True)
    w = edge_weight[edge_mask] if edge_weight is not None else edge_weight
    return e_new, w


class StockPredictionModel(nn.Module):
    """
    MCI-GRU: temporal encoder + GAT + latent cross-attn + (optional) self-attn
    + prediction GAT. Four streams [A1, A2, B1, B2] are built in that order
    in ``forward``; do not change without updating :class:`SelfAttention` docs.
    """

    def __init__(
        self,
        input_size: int,
        gru_hidden_sizes: list[int] = None,
        hidden_size_gat1: int = 32,
        output_gat1: int = 4,
        gat_heads: int = 4,
        hidden_size_gat2: int = 32,
        num_hidden_states: int = 32,
        cross_attn_heads: int = 4,
        slow_kernel: int = 5,
        slow_stride: int = 2,
        use_multi_scale: bool = True,
        use_self_attention: bool = True,
        activation: str = "elu",
        output_activation: str = "none",
        latent_init_scale: float = 0.02,
        edge_feature_dim: int = 1,
        use_group_type_embed: bool = False,
        use_trunk_regularisation: bool = False,
        trunk_dropout: float = 0.1,
        use_nn_multihead_attention: bool = False,
        temporal_encoder: str = "legacy",
        drop_edge_p: float = 0.0,
        use_sector_relation: bool = False,
        use_a1_a2_cross_attention: bool = False,
        cross_a2_num_heads: int = 4,
    ):
        super().__init__()
        if gru_hidden_sizes is None:
            gru_hidden_sizes = [32, 10]

        self._align_dim = hidden_size_gat1
        tr = use_trunk_regularisation
        tdrop = trunk_dropout if tr else 0.0
        self._use_trunk_regularisation = tr
        gat_inter_drop = tdrop if tr else 0.0
        self.use_sector_relation = use_sector_relation
        self.use_a1_a2_cross_attention = use_a1_a2_cross_attention

        # Temporal: legacy ImprovedGRU vs GRU+attention (or multi-scale variants)
        if use_multi_scale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                input_size,
                hidden_sizes=gru_hidden_sizes,
                slow_kernel=slow_kernel,
                slow_stride=slow_stride,
                temporal_encoder=temporal_encoder,
            )
        elif temporal_encoder == "legacy":
            self.temporal_encoder = ImprovedGRU(input_size, hidden_sizes=gru_hidden_sizes)
        elif temporal_encoder == "transformer":
            self.temporal_encoder = CausalTransformerEncoder(input_size, gru_hidden_sizes[-1])
        else:
            self.temporal_encoder = GRUWithAttention(input_size, gru_hidden_sizes)

        gru_output_size = self.temporal_encoder.output_size
        self.edge_feature_dim = edge_feature_dim
        self.gat_layer = GATBlock(
            in_channels=input_size,
            hidden=hidden_size_gat1,
            out_channels=output_gat1,
            heads=gat_heads,
            activation=activation,
            edge_feature_dim=edge_feature_dim,
            inter_layer_dropout=gat_inter_drop,
        )
        if use_sector_relation:
            self.gat_layer_sector = GATBlock(
                in_channels=input_size,
                hidden=hidden_size_gat1,
                out_channels=output_gat1,
                heads=gat_heads,
                activation=activation,
                edge_feature_dim=1,
                inter_layer_dropout=gat_inter_drop,
            )
            self.gat_stream_fuse = nn.Linear(output_gat1 * 2, output_gat1)
        else:
            self.gat_layer_sector = None
            self.gat_stream_fuse = None

        self.align_dim = hidden_size_gat1
        self.proj_temporal = nn.Linear(gru_output_size, self.align_dim)
        self.proj_cross = nn.Linear(output_gat1, self.align_dim)

        if use_a1_a2_cross_attention:
            self.proj_a1_seq = nn.Linear(gru_output_size, self.align_dim)
            self.cross_a1_a2 = nn.MultiheadAttention(
                self.align_dim,
                cross_a2_num_heads,
                batch_first=True,
                dropout=tdrop,
            )
        else:
            self.proj_a1_seq = None
            self.cross_a1_a2 = None

        self.ln_a1 = _maybe_ln_ch(self.align_dim, tr)
        self.ln_a2 = _maybe_ln_ch(self.align_dim, tr)

        self.latent_learner = MarketLatentStateLearner(
            feature_dim=self.align_dim,
            num_latent_states=num_hidden_states,
            num_heads=cross_attn_heads,
            latent_init_scale=latent_init_scale,
            use_nn_multihead_attention=use_nn_multihead_attention,
            attn_dropout=tdrop,
        )
        self.concat_size = 4 * self.align_dim
        self.ln_z = _maybe_ln_ch(self.concat_size, tr)
        self.drop_z = _maybe_drop(tdrop, tr)

        if use_self_attention:
            self.self_attention: SelfAttention | None = SelfAttention(
                embed_dim=self.concat_size,
                align_dim=self.align_dim,
                use_group_type_embed=use_group_type_embed,
            )
        else:
            self.self_attention = None

        self.final_gat = GATBlock(
            in_channels=self.concat_size,
            hidden=hidden_size_gat2,
            out_channels=1,
            heads=gat_heads,
            activation=activation,
            edge_feature_dim=edge_feature_dim,
            inter_layer_dropout=gat_inter_drop,
        )
        self.output_act = _make_output_activation(output_activation)
        self.drop_edge_p = float(drop_edge_p)

    def _temporal_fast_sequence(self, x_time_series: torch.Tensor) -> torch.Tensor:
        enc = self.temporal_encoder
        if isinstance(enc, MultiScaleTemporalEncoder):
            return enc.forward_fast_sequence(x_time_series)
        if hasattr(enc, "forward_sequence"):
            return enc.forward_sequence(x_time_series)
        raise TypeError("Temporal encoder does not expose a sequence for cross-attention")

    def forward(
        self,
        x_time_series: torch.Tensor,
        x_graph: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_stocks: int | None = None,
        edge_index_sector: torch.Tensor | None = None,
        edge_weight_sector: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        training = self.training
        batch_size = x_time_series.shape[0]
        if num_stocks is None:
            num_stocks = x_time_series.shape[1]

        node_mask = None
        if stock_mask is not None:
            stock_mask = stock_mask.to(dtype=torch.bool, device=x_time_series.device)
            node_mask = stock_mask.reshape(batch_size * num_stocks, 1).to(x_graph.dtype)
            x_time_series = x_time_series * stock_mask[:, :, None, None].to(x_time_series.dtype)
            x_graph = x_graph * node_mask

        e_idx, e_wt = _apply_edge_dropout(
            edge_index, edge_weight, self.drop_edge_p, training=training
        )
        if (
            self.use_sector_relation
            and self.gat_layer_sector is not None
            and edge_index_sector is not None
            and edge_weight_sector is not None
        ):
            e_idx_s, e_wt_s = _apply_edge_dropout(
                edge_index_sector, edge_weight_sector, self.drop_edge_p, training=training
            )
        else:
            e_idx_s, e_wt_s = None, None

        a1_raw = self.temporal_encoder(x_time_series)
        a1_raw = a1_raw.reshape(batch_size * num_stocks, -1)
        a1 = self.proj_temporal(a1_raw)
        a1 = self.ln_a1(a1)
        if node_mask is not None:
            a1 = a1 * node_mask.to(a1.dtype)

        a2_corr = self.gat_layer(x_graph, e_idx, e_wt)
        if self.gat_layer_sector is not None and e_idx_s is not None and e_wt_s is not None:
            a2_sec = self.gat_layer_sector(x_graph, e_idx_s, e_wt_s)
            a2_raw = self.gat_stream_fuse(torch.cat([a2_corr, a2_sec], dim=-1))
        else:
            a2_raw = a2_corr

        a2 = self.proj_cross(a2_raw)
        if (
            self.use_a1_a2_cross_attention
            and self.cross_a1_a2 is not None
            and self.proj_a1_seq is not None
        ):
            seq = self._temporal_fast_sequence(x_time_series)
            seq = self.proj_a1_seq(seq)
            bn = batch_size * num_stocks
            tlen = seq.shape[2]
            q = a2.view(batch_size, num_stocks, -1).reshape(bn, 1, self.align_dim)
            kv = seq.reshape(bn, tlen, self.align_dim)
            cross_out, _ = self.cross_a1_a2(q, kv, kv, need_weights=False)
            a2 = a2 + cross_out.reshape(batch_size * num_stocks, -1)
        a2 = self.ln_a2(a2)
        if node_mask is not None:
            a2 = a2 * node_mask.to(a2.dtype)

        b1, b2 = self.latent_learner(a1, a2)
        if node_mask is not None:
            b1 = b1 * node_mask.to(b1.dtype)
            b2 = b2 * node_mask.to(b2.dtype)
        # Contract: A1, A2, B1, B2 order for SelfAttention group_type_embed slots 0..3
        z = torch.cat([a1, a2, b1, b2], dim=-1)
        z = self.ln_z(z)
        z = self.drop_z(z)
        if node_mask is not None:
            z = z * node_mask.to(z.dtype)

        if self.self_attention is not None:
            z = z.view(batch_size, num_stocks, -1)
            z = self.self_attention(z, stock_mask=stock_mask)
            z = z.view(batch_size * num_stocks, -1)
        out = self.final_gat(z, e_idx, e_wt)
        out = self.output_act(out)
        if node_mask is not None:
            out = out * node_mask.to(out.dtype)
        return out.view(batch_size, num_stocks)
