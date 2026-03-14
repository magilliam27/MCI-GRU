"""
MCI-GRU Model: Multi-head Cross-attention and Improved GRU for Stock Prediction.

This module contains all model components:
- AttentionResetGRUCell: GRU cell with attention-based reset gate
- ImprovedGRU: Multi-layer improved GRU
- MultiScaleTemporalEncoder: Fast/slow temporal processing
- GATLayer / GATLayer_1: Graph Attention layers
- SelfAttention: Optional feature-mixing layer before final prediction
- MarketLatentStateLearner: Cross-attention for market states
- StockPredictionModel: Main model combining all components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
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
        
        # Scaled dot-product score per paper Equation 6.
        # The score is a scalar per stock (shape: batch, num_stocks, 1) so softmax
        # over the last dim would always yield 1.0 -- a no-op.  Sigmoid produces a
        # meaningful gate in [0, 1] that modulates the value signal.
        attn_score = torch.sum(q_t * k_t, dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        alpha_t = torch.sigmoid(attn_score)

        # Attention-weighted value as reset signal (paper Equation 7: r'_t = a_t * v_t)
        r_prime_t = alpha_t * v_t  # (batch, num_stocks, hidden_size)
        
        # Candidate hidden state: h_tilde = tanh(W_h(x) + r' * U_h(h))
        h_tilde = torch.tanh(self.W_h(x_t) + r_prime_t * self.U_h(h_prev))
        
        # Final hidden state: h_t = (1 - z_t) * h + z_t * h_tilde
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


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder inspired by the Momentum Turning Points paper.
    
    The paper demonstrates that slow (12-month) and fast (1-month) momentum 
    signals capture different but complementary information. This encoder
    processes temporal features at multiple scales:
    
    1. Fast path: Standard ImprovedGRU on full sequence (captures recent patterns)
    2. Slow path: Temporal aggregation via Conv1d, then GRU (captures longer-term trends)
    
    The outputs are combined to provide a richer temporal representation that
    captures both fast-moving signals and slow-moving trends.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden sizes for each GRU layer [32, 10]
        slow_kernel: Kernel size for temporal convolution (default 5)
        slow_stride: Stride for temporal downsampling (default 2)
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None, 
                 slow_kernel: int = 5, slow_stride: int = 2):
        super(MultiScaleTemporalEncoder, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 10]
        
        self.hidden_sizes = hidden_sizes
        self.slow_kernel = slow_kernel
        self.slow_stride = slow_stride
        self.fast_gru = ImprovedGRU(input_size, hidden_sizes)
        self.slow_aggregator = nn.Conv1d(
            in_channels=input_size, 
            out_channels=input_size, 
            kernel_size=slow_kernel, 
            stride=slow_stride, 
            padding=slow_kernel // 2
        )
        self.slow_gru = ImprovedGRU(input_size, hidden_sizes)
        self.combiner = nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1])
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_stocks, seq_len, input_size = x.shape
        device = x.device
        
        fast_out = self.fast_gru(x)
        x_reshaped = x.view(batch_size * num_stocks, seq_len, input_size)
        x_reshaped = x_reshaped.transpose(1, 2)
        x_slow = self.slow_aggregator(x_reshaped)
        x_slow = x_slow.transpose(1, 2)
        seq_len_slow = x_slow.shape[1]
        x_slow = x_slow.view(batch_size, num_stocks, seq_len_slow, input_size)
        
        slow_out = self.slow_gru(x_slow)
        combined = torch.cat([fast_out, slow_out], dim=-1)
        output = self.combiner(combined)
        
        return output


def _make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r}. Choose 'elu' or 'relu'.")


class GATBlock(nn.Module):
    """Two-layer Graph Attention block.

    Layer 1: multi-head GAT (in_channels → hidden × heads, concatenated)
    Layer 2: single-head GAT (hidden × heads → out_channels)

    Used twice in the MCI-GRU architecture: once for cross-sectional feature
    extraction (Part B) and once for the final prediction head (Part D).
    The former ``GATLayer`` and ``GATLayer_1`` were near-identical classes
    that only differed in parameter naming — this unified class replaces both.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 heads: int = 1, activation: str = "elu"):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, edge_dim=1)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index, edge_weight)
        x = self.act(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


# Backward-compatible aliases so existing checkpoints and imports keep working.
GATLayer = GATBlock
GATLayer_1 = GATBlock


class SelfAttention(nn.Module):
    """
    Self-attention layer for mixing heterogeneous feature groups.

    Applied to the concatenated [A1, A2, B1, B2] vector before the final
    prediction GAT.  This lets the four feature groups (temporal, cross-
    sectional, and their latent-state-enriched variants) interact and
    normalise relative to each other.

    Present in the original paper code but absent from the paper text.
    Controlled by the ``use_self_attention`` model flag.
    """

    def __init__(self, embed_dim: int):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attn_weights = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1
        )
        return torch.matmul(attn_weights, v)


class MarketLatentStateLearner(nn.Module):
    """
    Multi-head cross-attention mechanism for learning latent market states.
    
    Paper methodology:
    - Learns two sets of latent state vectors (R1, R2) 
    - R1 interacts with temporal features (A1)
    - R2 interacts with cross-sectional features (A2)
    - Uses multi-head attention (4 heads per paper)
    """
    
    def __init__(self, feature_dim: int, num_latent_states: int = 32, num_heads: int = 4,
                 latent_init_scale: float = 0.02):
        super(MarketLatentStateLearner, self).__init__()
        
        self.num_latent_states = num_latent_states
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Learnable latent state vectors (scale is configurable: paper default
        # 0.02; original code used 1.0 via plain torch.randn)
        self.R1 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * latent_init_scale)
        self.R2 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * latent_init_scale)
        
        # Multi-head attention projections for R1-A1 interaction
        self.W_Q1 = nn.Linear(feature_dim, feature_dim)
        self.W_K1 = nn.Linear(feature_dim, feature_dim)
        self.W_V1 = nn.Linear(feature_dim, feature_dim)
        self.W_O1 = nn.Linear(feature_dim, feature_dim)
        
        # Multi-head attention projections for R2-A2 interaction
        self.W_Q2 = nn.Linear(feature_dim, feature_dim)
        self.W_K2 = nn.Linear(feature_dim, feature_dim)
        self.W_V2 = nn.Linear(feature_dim, feature_dim)
        self.W_O2 = nn.Linear(feature_dim, feature_dim)
    
    def multi_head_cross_attention(self, query: torch.Tensor, key_value: torch.Tensor, 
                                    W_Q: nn.Linear, W_K: nn.Linear, 
                                    W_V: nn.Linear, W_O: nn.Linear) -> torch.Tensor:
        N = query.shape[0]
        Q = W_Q(query)
        K = W_K(key_value)
        V = W_V(key_value)
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, -1)
        output = W_O(attn_output)
        
        return output
    
    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B1 = self.multi_head_cross_attention(A1, self.R1, self.W_Q1, self.W_K1, self.W_V1, self.W_O1)
        B2 = self.multi_head_cross_attention(A2, self.R2, self.W_Q2, self.W_K2, self.W_V2, self.W_O2)
        
        return B1, B2


class StockPredictionModel(nn.Module):
    """
    MCI-GRU Model: Multi-head Cross-attention and Improved GRU for Stock Prediction.

    Architecture:
      Part A  -- Temporal features via ImprovedGRU (or MultiScaleTemporalEncoder)
      Part B  -- Cross-sectional features via GAT
      Part C  -- Multi-head cross-attention for latent market states
      Part D  -- [optional SelfAttention] -> Prediction GAT
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
        latent_init_scale: float = 0.02,
    ):
        super(StockPredictionModel, self).__init__()
        if gru_hidden_sizes is None:
            gru_hidden_sizes = [32, 10]
        
        if use_multi_scale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                input_size, 
                hidden_sizes=gru_hidden_sizes,
                slow_kernel=slow_kernel,
                slow_stride=slow_stride
            )
        else:
            self.temporal_encoder = ImprovedGRU(input_size, hidden_sizes=gru_hidden_sizes)
        gru_output_size = self.temporal_encoder.output_size
        self.gat_layer = GATBlock(
            in_channels=input_size, hidden=hidden_size_gat1,
            out_channels=output_gat1, heads=gat_heads, activation=activation,
        )
        self.align_dim = hidden_size_gat1
        self.proj_temporal = nn.Linear(gru_output_size, self.align_dim)
        self.proj_cross = nn.Linear(output_gat1, self.align_dim)
        self.latent_learner = MarketLatentStateLearner(
            feature_dim=self.align_dim,
            num_latent_states=num_hidden_states,
            num_heads=cross_attn_heads,
            latent_init_scale=latent_init_scale,
        )
        concat_size = 4 * self.align_dim

        if use_self_attention:
            self.self_attention: Optional[SelfAttention] = SelfAttention(concat_size)
        else:
            self.self_attention = None

        self.final_gat = GATBlock(
            in_channels=concat_size, hidden=hidden_size_gat2,
            out_channels=1, heads=gat_heads, activation=activation,
        )
        self.output_act = _make_activation(activation)
        
    def forward(self, x_time_series: torch.Tensor, x_graph: torch.Tensor, 
                edge_index: torch.Tensor, edge_weight: torch.Tensor, 
                num_stocks: Optional[int] = None) -> torch.Tensor:
        batch_size = x_time_series.shape[0]
        if num_stocks is None:
            num_stocks = x_time_series.shape[1]
        
        A1_raw = self.temporal_encoder(x_time_series)
        A1_raw = A1_raw.reshape(batch_size * num_stocks, -1)
        A1 = self.proj_temporal(A1_raw)
        A2_raw = self.gat_layer(x_graph, edge_index, edge_weight)
        A2 = self.proj_cross(A2_raw)
        B1, B2 = self.latent_learner(A1, A2)
        Z = torch.cat([A1, A2, B1, B2], dim=-1)

        if self.self_attention is not None:
            # Reshape to (batch, num_stocks, concat_size) so self-attention
            # operates *across stocks* within each batch element, then flatten
            # back to (batch * num_stocks, concat_size) for the graph layers.
            Z = Z.view(batch_size, num_stocks, -1)
            Z = self.self_attention(Z)
            Z = Z.view(batch_size * num_stocks, -1)

        out = self.final_gat(Z, edge_index, edge_weight)
        out = self.output_act(out)
        
        return out.view(batch_size, num_stocks)


def create_model(input_size: int, config: Dict[str, Any]) -> StockPredictionModel:
    return StockPredictionModel(
        input_size=input_size,
        gru_hidden_sizes=config.get('gru_hidden_sizes', [32, 10]),
        hidden_size_gat1=config.get('hidden_size_gat1', 32),
        output_gat1=config.get('output_gat1', 4),
        gat_heads=config.get('gat_heads', 4),
        hidden_size_gat2=config.get('hidden_size_gat2', 32),
        num_hidden_states=config.get('num_hidden_states', 32),
        cross_attn_heads=config.get('cross_attn_heads', 4),
        slow_kernel=config.get('slow_kernel', 5),
        slow_stride=config.get('slow_stride', 2),
        use_multi_scale=config.get('use_multi_scale', True),
        use_self_attention=config.get('use_self_attention', True),
        activation=config.get('activation', 'elu'),
        latent_init_scale=config.get('latent_init_scale', 0.02),
    )
