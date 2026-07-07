import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super().__init__()
        self.num_latent_states = num_latent_states
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_nn_multihead_attention = use_nn_multihead_attention

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.R1 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * latent_init_scale)
        self.R2 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * latent_init_scale)

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

    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
