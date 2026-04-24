"""
MCI-GRU Model: Multi-head Cross-attention and Improved GRU for Stock Prediction.

This module contains all model components:
- AttentionResetGRUCell: GRU cell with attention-based reset gate
- ImprovedGRU: Multi-layer improved GRU
- GRUWithAttention: CuDNN-fused nn.GRU + post-hoc attention (Phase 2)
- MultiScaleTemporalEncoder: Fast/slow temporal processing
- GATLayer / GATLayer_1: Graph Attention layers
- SelfAttention: Optional feature-mixing layer before final prediction
- MarketLatentStateLearner: Cross-attention for market states
- StockPredictionModel: Main model combining all components
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
from typing import List, Tuple, Optional, Dict, Any


class AttentionResetGRUCell(nn.Module):
    """
    GRU cell with attention mechanism replacing the reset gate.

    Paper methodology:
    - Instead of: r_t = sigmoid(W_r * x_t + U_r * h_{t-1})
    - We use: r'_t = Attention(h_{t-1}, x_t)
    - Query from h_{t-1}, Key/Value from x_t
    - Candidate: h_tilde = tanh(W_h(x) + r' * U_h(h))
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(AttentionResetGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        q_t = self.W_q(h_prev)
        k_t = self.W_k(x_t)
        v_t = self.W_v(x_t)

        attn_score = (
            torch.sum(q_t * k_t, dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        )
        alpha_t = torch.sigmoid(attn_score)

        r_prime_t = alpha_t * v_t
        h_tilde = torch.tanh(self.W_h(x_t) + r_prime_t * self.U_h(h_prev))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class ImprovedGRU(nn.Module):
    """
    Multi-layer Improved GRU for temporal feature extraction.
    Paper uses two layers with hidden sizes [32, 10].
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = None):
        super(ImprovedGRU, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 10]

        self.layers = nn.ModuleList()
        self.hidden_sizes = hidden_sizes
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(AttentionResetGRUCell(prev_size, hidden_size))
            prev_size = hidden_size
        self.output_size = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_stocks, seq_len, _ = x.shape
        device = x.device
        layer_input = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_size = self.hidden_sizes[layer_idx]
            h = torch.zeros(batch_size, num_stocks, hidden_size, device=device)
            outputs = []
            for t in range(seq_len):
                h = layer(layer_input[:, :, t, :], h)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=2)
        return layer_input[:, :, -1, :]

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return all time steps from the final layer, shape ``(B, N, T, H)``."""
        batch_size, num_stocks, seq_len, _ = x.shape
        device = x.device
        layer_input = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_size = self.hidden_sizes[layer_idx]
            h = torch.zeros(batch_size, num_stocks, hidden_size, device=device)
            outputs = []
            for t in range(seq_len):
                h = layer(layer_input[:, :, t, :], h)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=2)
        return layer_input


class GRUWithAttention(nn.Module):
    """
    Fused `nn.GRU` over time + single post-hoc scaled dot-product readout.

    Stacked `nn.GRU` uses one hidden size (the last in ``hidden_sizes``) for all
    layers — unlike :class:`ImprovedGRU`, which can use different per-layer sizes.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 10]
        self.hidden_sizes = hidden_sizes
        n_layers = len(hidden_sizes)
        d_h = hidden_sizes[-1]
        self.output_size = d_h
        self.gru = nn.GRU(
            input_size,
            d_h,
            num_layers=n_layers,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_h)
        self.scale = d_h**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_stocks, tlen, f_in = x.shape
        x2 = x.reshape(batch * num_stocks, tlen, f_in)
        out, _ = self.gru(x2)
        h_t = out[:, -1, :]
        scores = (out * h_t.unsqueeze(1)).sum(-1) * self.scale
        alpha = F.softmax(scores, dim=-1)
        ctx = (alpha.unsqueeze(-1) * out).sum(dim=1)
        y = self.ln(h_t + ctx)
        return y.view(batch, num_stocks, -1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """GRU hidden states at every step (before attention readout), ``(B, N, T, H)``."""
        batch, num_stocks, tlen, f_in = x.shape
        x2 = x.reshape(batch * num_stocks, tlen, f_in)
        out, _ = self.gru(x2)
        return out.view(batch, num_stocks, tlen, -1)


class CausalTransformerEncoder(nn.Module):
    """Causal Transformer over the fast temporal path (Phase 3)."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.d_model = d_model
        self.output_size = d_model

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_stocks, tlen, f_in = x.shape
        z = self.input_proj(x).reshape(batch * num_stocks, tlen, self.d_model)
        try:
            out = self.encoder(z, is_causal=True)
        except TypeError:
            t = tlen
            causal = torch.triu(
                torch.full((t, t), float("-inf"), device=z.device, dtype=z.dtype),
                diagonal=1,
            )
            out = self.encoder(z, mask=causal)
        return out.view(batch, num_stocks, tlen, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.forward_sequence(x)
        return seq[:, :, -1, :]


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder: fast and slow paths combined.

    The ``temporal_encoder`` string selects the recurrent backbone:
    ``"legacy"`` = :class:`ImprovedGRU` (paper cell, Python loop);
    ``"gru_attn"`` = :class:`GRUWithAttention` (CuDNN-fused + attention readout);
    ``"transformer"`` = causal :class:`CausalTransformerEncoder` on the fast path.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        slow_kernel: int = 5,
        slow_stride: int = 2,
        temporal_encoder: str = "legacy",
    ):
        super(MultiScaleTemporalEncoder, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 10]
        self.hidden_sizes = hidden_sizes
        self.slow_kernel = slow_kernel
        self.slow_stride = slow_stride
        self.temporal_encoder = temporal_encoder

        if temporal_encoder == "legacy":
            self.fast_gru = ImprovedGRU(input_size, hidden_sizes)
            self.slow_gru = ImprovedGRU(input_size, hidden_sizes)
        elif temporal_encoder == "transformer":
            d_h = hidden_sizes[-1]
            self.fast_gru = CausalTransformerEncoder(input_size, d_h)
            self.slow_gru = GRUWithAttention(input_size, hidden_sizes)
        else:
            self.fast_gru = GRUWithAttention(input_size, hidden_sizes)
            self.slow_gru = GRUWithAttention(input_size, hidden_sizes)

        self.slow_aggregator = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=slow_kernel,
            stride=slow_stride,
            padding=slow_kernel // 2,
        )
        self.combiner = nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1])
        self.output_size = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_stocks, seq_len, input_size = x.shape
        fast_out = self.fast_gru(x)
        x_reshaped = x.view(batch_size * num_stocks, seq_len, input_size)
        x_reshaped = x_reshaped.transpose(1, 2)
        x_slow = self.slow_aggregator(x_reshaped)
        x_slow = x_slow.transpose(1, 2)
        seq_len_slow = x_slow.shape[1]
        x_slow = x_slow.view(batch_size, num_stocks, seq_len_slow, input_size)
        slow_out = self.slow_gru(x_slow)
        combined = torch.cat([fast_out, slow_out], dim=-1)
        return self.combiner(combined)

    def forward_fast_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Fast-branch sequence ``(B, N, T, H)`` for cross-stream attention."""
        if hasattr(self.fast_gru, "forward_sequence"):
            return self.fast_gru.forward_sequence(x)
        if isinstance(self.fast_gru, CausalTransformerEncoder):
            return self.fast_gru.forward_sequence(x)
        raise TypeError("Fast temporal module does not expose forward_sequence")


def _make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported GAT activation: {name!r}. Choose 'elu' or 'relu'.")


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
        f"Unsupported output activation: {name!r}. "
        "Choose 'none', 'elu', 'relu', or 'sigmoid'."
    )


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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index, edge_weight)
        x = self.act(x)
        if self.drop_mid is not None and self.training:
            x = self.drop_mid(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


# Backward-compatible aliases
GATLayer = GATBlock
GATLayer_1 = GATBlock


class SelfAttention(nn.Module):
    """
    Self-attention over the cross-section (stocks) for each batch item.

    Input ``x`` has shape ``(B, N, 4*align_dim)`` where the last dimension is
    the concatenation **(A1, A2, B1, B2)** in that order (contract: do not permute
    in :class:`StockPredictionModel` without updating type indices below).
    When ``use_group_type_embed`` is True, a learned ``(4, align_dim)`` embedding
    is added to each of the four stream blocks before the linear Q/K/V.
    """

    def __init__(
        self,
        embed_dim: int,
        align_dim: int,
        use_group_type_embed: bool = False,
    ):
        super().__init__()
        if embed_dim != 4 * align_dim:
            raise ValueError("embed_dim must be 4 * align_dim for four-stream self-attention")
        self.embed_dim = embed_dim
        self.align_dim = align_dim
        self.use_group_type_embed = use_group_type_embed
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5
        if use_group_type_embed:
            self.type_embed = nn.Embedding(4, align_dim)
            nn.init.normal_(self.type_embed.weight, std=0.02)
        else:
            self.type_embed = None  # no extra params for checkpoint compat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.type_embed is not None:
            b, n, c = x.shape
            if c != 4 * self.align_dim:
                raise ValueError("concat shape mismatch for group type embedding")
            part = x.view(b, n, 4, self.align_dim) + self.type_embed.weight.view(1, 1, 4, -1)
            x = part.reshape(b, n, c)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attn = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        return torch.matmul(attn, v)


class MarketLatentStateLearner(nn.Module):
    """
    Multi-head cross-attention for learning latent market states (R1, R2),
    with either the legacy 8-Linear MHA (paper) or :class:`nn.MultiheadAttention`.
    """

    def __init__(
        self,
        feature_dim: int,
        num_latent_states: int = 32,
        num_heads: int = 4,
        latent_init_scale: float = 0.02,
        use_nn_multihead_attention: bool = False,
        attn_dropout: float = 0.0,
    ):
        super(MarketLatentStateLearner, self).__init__()
        self.num_latent_states = num_latent_states
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_nn_multihead_attention = use_nn_multihead_attention

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.R1 = nn.Parameter(
            torch.randn(num_latent_states, feature_dim) * latent_init_scale
        )
        self.R2 = nn.Parameter(
            torch.randn(num_latent_states, feature_dim) * latent_init_scale
        )

        if use_nn_multihead_attention:
            self.mha1 = nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
            self.mha2 = nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
        else:
            self.W_Q1 = nn.Linear(feature_dim, feature_dim)
            self.W_K1 = nn.Linear(feature_dim, feature_dim)
            self.W_V1 = nn.Linear(feature_dim, feature_dim)
            self.W_O1 = nn.Linear(feature_dim, feature_dim)
            self.W_Q2 = nn.Linear(feature_dim, feature_dim)
            self.W_K2 = nn.Linear(feature_dim, feature_dim)
            self.W_V2 = nn.Linear(feature_dim, feature_dim)
            self.W_O2 = nn.Linear(feature_dim, feature_dim)
            self.mha1 = None
            self.mha2 = None

    def multi_head_cross_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        W_Q: nn.Linear,
        W_K: nn.Linear,
        W_V: nn.Linear,
        W_O: nn.Linear,
    ) -> torch.Tensor:
        nq = query.shape[0]
        Q = W_Q(query)
        K = W_K(key_value)
        V = W_V(key_value)
        Q = Q.view(nq, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = F.softmax(scores, dim=-1)
        out = torch.matmul(w, V)
        out = out.transpose(0, 1).contiguous().view(nq, -1)
        return W_O(out)

    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_nn_multihead_attention and self.mha1 is not None:
            bn = A1.shape[0]
            r1 = self.R1.unsqueeze(0).expand(bn, -1, -1)
            r2 = self.R2.unsqueeze(0).expand(bn, -1, -1)
            b1, _ = self.mha1(A1.unsqueeze(1), r1, r1, need_weights=False)
            b2, _ = self.mha2(A2.unsqueeze(1), r2, r2, need_weights=False)
            return b1.squeeze(1), b2.squeeze(1)
        b1 = self.multi_head_cross_attention(
            A1, self.R1, self.W_Q1, self.W_K1, self.W_V1, self.W_O1
        )
        b2 = self.multi_head_cross_attention(
            A2, self.R2, self.W_Q2, self.W_K2, self.W_V2, self.W_O2
        )
        return b1, b2


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
    e_new, edge_mask = dropout_edge(
        edge_index, p=p, force_undirected=False, training=True
    )
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
        gru_hidden_sizes: List[int] = None,
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
        super(StockPredictionModel, self).__init__()
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
            self.temporal_encoder = CausalTransformerEncoder(
                input_size, gru_hidden_sizes[-1]
            )
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
            self.self_attention: Optional[SelfAttention] = SelfAttention(
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
        num_stocks: Optional[int] = None,
        edge_index_sector: Optional[torch.Tensor] = None,
        edge_weight_sector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        training = self.training
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

        batch_size = x_time_series.shape[0]
        if num_stocks is None:
            num_stocks = x_time_series.shape[1]

        a1_raw = self.temporal_encoder(x_time_series)
        a1_raw = a1_raw.reshape(batch_size * num_stocks, -1)
        a1 = self.proj_temporal(a1_raw)
        a1 = self.ln_a1(a1)

        a2_corr = self.gat_layer(x_graph, e_idx, e_wt)
        if self.gat_layer_sector is not None and e_idx_s is not None and e_wt_s is not None:
            a2_sec = self.gat_layer_sector(x_graph, e_idx_s, e_wt_s)
            a2_raw = self.gat_stream_fuse(torch.cat([a2_corr, a2_sec], dim=-1))
        else:
            a2_raw = a2_corr

        a2 = self.proj_cross(a2_raw)
        if self.use_a1_a2_cross_attention and self.cross_a1_a2 is not None and self.proj_a1_seq is not None:
            seq = self._temporal_fast_sequence(x_time_series)
            seq = self.proj_a1_seq(seq)
            bn = batch_size * num_stocks
            tlen = seq.shape[2]
            q = a2.view(batch_size, num_stocks, -1).reshape(bn, 1, self.align_dim)
            kv = seq.reshape(bn, tlen, self.align_dim)
            cross_out, _ = self.cross_a1_a2(q, kv, kv, need_weights=False)
            a2 = a2 + cross_out.reshape(batch_size * num_stocks, -1)
        a2 = self.ln_a2(a2)

        b1, b2 = self.latent_learner(a1, a2)
        # Contract: A1, A2, B1, B2 order for SelfAttention group_type_embed slots 0..3
        z = torch.cat([a1, a2, b1, b2], dim=-1)
        z = self.ln_z(z)
        z = self.drop_z(z)

        if self.self_attention is not None:
            z = z.view(batch_size, num_stocks, -1)
            z = self.self_attention(z)
            z = z.view(batch_size * num_stocks, -1)
        out = self.final_gat(z, e_idx, e_wt)
        out = self.output_act(out)
        return out.view(batch_size, num_stocks)


def create_model(input_size: int, config: Dict[str, Any]) -> StockPredictionModel:
    """
    Build a model. Missing keys default to *legacy* shapes so old ``config.yaml``
    in checkpoint dirs still load; new runs should set all keys in Hydra.
    """
    act = config.get("activation", "elu")
    out = config.get("output_activation", None)
    if out is None:
        out = act
    tr = config.get("use_trunk_regularisation", False)
    tdrop = config.get("trunk_dropout", 0.1) if tr else 0.0
    if not tr:
        tdrop = 0.0
    return StockPredictionModel(
        input_size=input_size,
        gru_hidden_sizes=config.get("gru_hidden_sizes", [32, 10]),
        hidden_size_gat1=config.get("hidden_size_gat1", 32),
        output_gat1=config.get("output_gat1", 4),
        gat_heads=config.get("gat_heads", 4),
        hidden_size_gat2=config.get("hidden_size_gat2", 32),
        num_hidden_states=config.get("num_hidden_states", 32),
        cross_attn_heads=config.get("cross_attn_heads", 4),
        slow_kernel=config.get("slow_kernel", 5),
        slow_stride=config.get("slow_stride", 2),
        use_multi_scale=config.get("use_multi_scale", True),
        use_self_attention=config.get("use_self_attention", True),
        activation=act,
        output_activation=out,
        latent_init_scale=config.get("latent_init_scale", 0.02),
        edge_feature_dim=config.get("edge_feature_dim", 1),
        use_group_type_embed=config.get("use_group_type_embed", False),
        use_trunk_regularisation=tr,
        trunk_dropout=tdrop,
        use_nn_multihead_attention=config.get("use_nn_multihead_attention", False),
        temporal_encoder=config.get("temporal_encoder", "legacy"),
        drop_edge_p=float(config.get("drop_edge_p", 0.0)),
        use_sector_relation=bool(config.get("use_sector_relation", False)),
        use_a1_a2_cross_attention=bool(config.get("use_a1_a2_cross_attention", False)),
        cross_a2_num_heads=int(config.get("cross_a2_num_heads", 4)),
    )