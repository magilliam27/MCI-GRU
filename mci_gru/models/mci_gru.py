"""
MCI-GRU Model: Multi-head Cross-attention and Improved GRU for Stock Prediction.

This module contains all model components extracted from sp500.py:
- AttentionResetGRUCell: GRU cell with attention-based reset gate
- ImprovedGRU: Multi-layer improved GRU
- MultiScaleTemporalEncoder: Fast/slow temporal processing
- GATLayer: Graph Attention layers
- MarketLatentStateLearner: Cross-attention for market states
- StockPredictionModel: Main model combining all components
"""

import math
import numpy as np
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
        
        # Update gate (unchanged from standard GRU)
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        
        # Attention mechanism (replaces reset gate)
        self.W_q = nn.Linear(hidden_size, hidden_size)  # Query from h_{t-1}
        self.W_k = nn.Linear(input_size, hidden_size)   # Key from x_t
        self.W_v = nn.Linear(input_size, hidden_size)   # Value from x_t
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: input at time t, shape (batch, num_stocks, input_size)
            h_prev: hidden state from t-1, shape (batch, num_stocks, hidden_size)
        
        Returns:
            h_t: new hidden state, shape (batch, num_stocks, hidden_size)
        """
        # Update gate (standard GRU)
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        
        # Attention-based reset gate
        # Query from hidden state, Key/Value from input
        q_t = self.W_q(h_prev)  # (batch, num_stocks, hidden_size)
        k_t = self.W_k(x_t)     # (batch, num_stocks, hidden_size)
        v_t = self.W_v(x_t)     # (batch, num_stocks, hidden_size)
        
        # Scaled dot-product attention per paper Equation 6
        attn_score = torch.sum(q_t * k_t, dim=-1, keepdim=True) / np.sqrt(self.hidden_size)
        alpha_t = F.softmax(attn_score, dim=-1)  # Paper uses softmax

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
        """
        Args:
            x: input sequence, shape (batch, num_stocks, seq_len, input_size)
        
        Returns:
            output: final hidden state, shape (batch, num_stocks, output_size)
        """
        batch_size, num_stocks, seq_len, _ = x.shape
        device = x.device
        
        # Process through each layer
        layer_input = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_size = self.hidden_sizes[layer_idx]
            h = torch.zeros(batch_size, num_stocks, hidden_size, device=device)
            
            outputs = []
            for t in range(seq_len):
                h = layer(layer_input[:, :, t, :], h)
                outputs.append(h)
            
            # Stack outputs as input for next layer
            layer_input = torch.stack(outputs, dim=2)
        
        # Return final hidden state (last time step of last layer)
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
        
        # Fast path: standard ImprovedGRU on full sequence
        # Captures short-term, fast-moving patterns (like 1-month momentum)
        self.fast_gru = ImprovedGRU(input_size, hidden_sizes)
        
        # Slow path: temporal aggregation then GRU
        # Captures longer-term, slow-moving patterns (like 12-month momentum)
        # Conv1d aggregates nearby time steps before processing
        self.slow_aggregator = nn.Conv1d(
            in_channels=input_size, 
            out_channels=input_size, 
            kernel_size=slow_kernel, 
            stride=slow_stride, 
            padding=slow_kernel // 2
        )
        self.slow_gru = ImprovedGRU(input_size, hidden_sizes)
        
        # Combine both scales
        # Concatenate fast and slow representations, project back to output size
        self.combiner = nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1])
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal features at multiple scales.
        
        Args:
            x: input sequence, shape (batch, num_stocks, seq_len, input_size)
        
        Returns:
            output: combined multi-scale features, shape (batch, num_stocks, output_size)
        """
        batch_size, num_stocks, seq_len, input_size = x.shape
        device = x.device
        
        # Fast path: process full sequence through ImprovedGRU
        # Output: (batch, num_stocks, hidden_sizes[-1])
        fast_out = self.fast_gru(x)
        
        # Slow path: aggregate temporally then process
        # Reshape for Conv1d: (batch * num_stocks, input_size, seq_len)
        x_reshaped = x.view(batch_size * num_stocks, seq_len, input_size)
        x_reshaped = x_reshaped.transpose(1, 2)  # (B*N, input_size, seq_len)
        
        # Apply temporal aggregation
        x_slow = self.slow_aggregator(x_reshaped)  # (B*N, input_size, seq_len')
        
        # Transpose back and reshape for GRU
        x_slow = x_slow.transpose(1, 2)  # (B*N, seq_len', input_size)
        seq_len_slow = x_slow.shape[1]
        x_slow = x_slow.view(batch_size, num_stocks, seq_len_slow, input_size)
        
        # Process aggregated sequence through slow GRU
        # Output: (batch, num_stocks, hidden_sizes[-1])
        slow_out = self.slow_gru(x_slow)
        
        # Combine fast and slow representations
        combined = torch.cat([fast_out, slow_out], dim=-1)  # (batch, num_stocks, hidden*2)
        output = self.combiner(combined)  # (batch, num_stocks, hidden)
        
        return output


class GATLayer(nn.Module):
    """
    Two-layer GAT for cross-sectional feature extraction.
    Paper uses ELU activation (not ReLU).
    """
    def __init__(self, hidden_size_gat1: int, output_gat1: int, 
                 in_channels: int, out_channels: int, heads: int = 1):
        super(GATLayer, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_size_gat1, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat1 * heads, output_gat1, heads=1, concat=False, edge_dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index, edge_weight)
        x = F.elu(x)  # Paper uses ELU
        x = self.gat2(x, edge_index, edge_weight)
        return x
    

class GATLayer_1(nn.Module):
    """
    Final prediction GAT layer.
    Paper uses ELU activation (not ReLU).
    """
    def __init__(self, hidden_size_gat2: int, in_channels: int, 
                 out_channels: int, heads: int = 1):
        super(GATLayer_1, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_size_gat2, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat2 * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index, edge_weight)
        x = F.elu(x)  # Paper uses ELU
        x = self.gat2(x, edge_index, edge_weight)
        return x 


class MarketLatentStateLearner(nn.Module):
    """
    Multi-head cross-attention mechanism for learning latent market states.
    
    Paper methodology:
    - Learns two sets of latent state vectors (R1, R2) 
    - R1 interacts with temporal features (A1)
    - R2 interacts with cross-sectional features (A2)
    - Uses multi-head attention (4 heads per paper)
    """
    
    def __init__(self, feature_dim: int, num_latent_states: int = 32, num_heads: int = 4):
        super(MarketLatentStateLearner, self).__init__()
        
        self.num_latent_states = num_latent_states
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Learnable latent state vectors
        self.R1 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * 0.02)
        self.R2 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * 0.02)
        
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
        """
        Multi-head cross-attention.
        
        Args:
            query: (N, feature_dim) - A1 or A2 as query
            key_value: (num_latent_states, feature_dim) - R1 or R2 as key/value
            W_Q, W_K, W_V, W_O: projection layers
        
        Returns:
            output: (N, feature_dim) - enriched features
        """
        N = query.shape[0]
        
        # Project to Q, K, V
        Q = W_Q(query)      # (N, feature_dim)
        K = W_K(key_value)  # (num_latent_states, feature_dim)
        V = W_V(key_value)  # (num_latent_states, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, N, head_dim)
        K = K.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(self.num_latent_states, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (heads, N, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, -1)  # (N, feature_dim)
        
        # Output projection
        output = W_O(attn_output)
        
        return output
    
    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            A1: temporal features, shape (N, feature_dim)
            A2: cross-sectional features, shape (N, feature_dim)
        
        Returns:
            B1: enriched temporal features, shape (N, feature_dim)
            B2: enriched cross-sectional features, shape (N, feature_dim)
        """
        # Cross-attention between A1 (query) and R1 (key/value)
        B1 = self.multi_head_cross_attention(A1, self.R1, self.W_Q1, self.W_K1, self.W_V1, self.W_O1)
        
        # Cross-attention between A2 (query) and R2 (key/value)
        B2 = self.multi_head_cross_attention(A2, self.R2, self.W_Q2, self.W_K2, self.W_V2, self.W_O2)
        
        return B1, B2


class StockPredictionModel(nn.Module):
    """
    MCI-GRU Model: Multi-head Cross-attention and Improved GRU for Stock Prediction.
    
    Architecture per paper:
    1. Part A: Multi-scale temporal encoder for temporal features
       - Enhanced with multi-scale processing inspired by Momentum Turning Points paper
       - Fast path: captures short-term patterns (like 1-month momentum)
       - Slow path: captures longer-term trends (like 12-month momentum)
    2. Part B: GAT for cross-sectional features
    3. Part C: Multi-head cross-attention for latent market states
    4. Part D: Concatenate A1, A2, B1, B2 -> Prediction GAT
    """
    def __init__(
        self, 
        input_size: int, 
        gru_hidden_sizes: List[int] = None,  # Paper: [32, 10]
        hidden_size_gat1: int = 32,           # Paper: 32
        output_gat1: int = 4,                 # Paper: 4
        gat_heads: int = 4,                   # Paper: 4 heads
        hidden_size_gat2: int = 32,           # Paper: 32
        num_hidden_states: int = 32,          # Paper: 32 latent vectors
        cross_attn_heads: int = 4,            # Paper: 4 heads for cross-attention
        slow_kernel: int = 5,                 # Kernel size for slow temporal aggregation
        slow_stride: int = 2                  # Stride for temporal downsampling
    ):
        super(StockPredictionModel, self).__init__()
        if gru_hidden_sizes is None:
            gru_hidden_sizes = [32, 10]
        
        # Part A: Multi-scale temporal encoder for temporal features
        # Uses both fast (full sequence) and slow (downsampled) processing paths
        # Inspired by Momentum Turning Points paper's slow/fast momentum distinction
        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_size, 
            hidden_sizes=gru_hidden_sizes,
            slow_kernel=slow_kernel,
            slow_stride=slow_stride
        )
        gru_output_size = self.temporal_encoder.output_size  # 10 for paper config
        
        # Part B: GAT for cross-sectional features
        self.gat_layer = GATLayer(hidden_size_gat1, output_gat1, input_size, output_gat1, gat_heads)
        
        # Projection layers to align dimensions for cross-attention
        # Both A1 and A2 should have same dimension for the latent state learner
        self.align_dim = hidden_size_gat1  # Use GAT hidden size as alignment dimension
        self.proj_temporal = nn.Linear(gru_output_size, self.align_dim)
        self.proj_cross = nn.Linear(output_gat1, self.align_dim)
        
        # Part C: Multi-head cross-attention for latent market states
        self.latent_learner = MarketLatentStateLearner(
            feature_dim=self.align_dim,
            num_latent_states=num_hidden_states,
            num_heads=cross_attn_heads
        )
        
        # Part D: Prediction layer
        # Paper: Concatenate A1, A2, B1, B2 -> 4 * align_dim, then final GAT
        concat_size = 4 * self.align_dim
        self.final_gat = GATLayer_1(hidden_size_gat2, concat_size, 1, gat_heads)
        self.elu = nn.ELU()  # Paper uses ELU, not ReLU
        
    def forward(self, x_time_series: torch.Tensor, x_graph: torch.Tensor, 
                edge_index: torch.Tensor, edge_weight: torch.Tensor, 
                num_stocks: Optional[int] = None) -> torch.Tensor:
        """
        Batched forward pass supporting batch_size > 1.
        
        Args:
            x_time_series: (batch, num_stocks, seq_len, input_size)
            x_graph: (batch * num_stocks, input_size) - PyG batched graph nodes
            edge_index: (2, batch * num_edges) - PyG batched edge indices
            edge_weight: (batch * num_edges,) - PyG batched edge weights
            num_stocks: int - number of stocks per graph (required for batch > 1)
        
        Returns:
            predictions: (batch, num_stocks) - predicted returns for each stock
        """
        batch_size = x_time_series.shape[0]
        if num_stocks is None:
            num_stocks = x_time_series.shape[1]
        
        # Part A: Temporal features via Multi-scale Temporal Encoder
        # Input: (batch, num_stocks, seq_len, input_size)
        # Output: (batch, num_stocks, gru_output_size)
        A1_raw = self.temporal_encoder(x_time_series)
        
        # Flatten to (batch * num_stocks, gru_output_size) to match batched graph structure
        A1_raw = A1_raw.reshape(batch_size * num_stocks, -1)
        A1 = self.proj_temporal(A1_raw)  # (batch * num_stocks, align_dim)
        
        # Part B: Cross-sectional features via GAT
        # x_graph is already (batch * num_stocks, input_size) from batched collate
        A2_raw = self.gat_layer(x_graph, edge_index, edge_weight)
        A2 = self.proj_cross(A2_raw)  # (batch * num_stocks, align_dim)
        
        # Part C: Latent state learning via multi-head cross-attention
        B1, B2 = self.latent_learner(A1, A2)  # Both (batch * num_stocks, align_dim)
        
        # Part D: Concatenate and predict (NO self-attention per paper)
        Z = torch.cat([A1, A2, B1, B2], dim=-1)  # (batch * num_stocks, 4 * align_dim)
        
        # Final GAT for prediction
        out = self.final_gat(Z, edge_index, edge_weight)  # (batch * num_stocks, 1)
        out = self.elu(out)  # Paper uses ELU activation
        
        # Reshape back to (batch, num_stocks)
        return out.view(batch_size, num_stocks)


def create_model(input_size: int, config: Dict[str, Any]) -> StockPredictionModel:
    """
    Create model using hyperparameters from config.
    
    Args:
        input_size: Number of input features
        config: Dictionary with model hyperparameters:
            - gru_hidden_sizes: List[int] (default [32, 10])
            - hidden_size_gat1: int (default 32)
            - output_gat1: int (default 4)
            - gat_heads: int (default 4)
            - hidden_size_gat2: int (default 32)
            - num_hidden_states: int (default 32)
            - cross_attn_heads: int (default 4)
            - slow_kernel: int (default 5)
            - slow_stride: int (default 2)
    
    Returns:
        Configured StockPredictionModel instance
    """
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
        slow_stride=config.get('slow_stride', 2)
    )
