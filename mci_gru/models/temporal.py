import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super().__init__()
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

        attn_score = torch.sum(q_t * k_t, dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
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

    def __init__(self, input_size: int, hidden_sizes: list[int] = None):
        super().__init__()
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

    def __init__(self, input_size: int, hidden_sizes: list[int] = None):
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


def _transformer_nhead_for_d_model(d_model: int, requested_nhead: int) -> int:
    """
    Largest nhead with ``1 <= nhead <= min(requested_nhead, d_model)`` and
    ``d_model % nhead == 0`` (required by ``nn.MultiheadAttention``).
    """
    if d_model < 1:
        raise ValueError(f"d_model must be >= 1, got {d_model}")
    if requested_nhead < 1:
        raise ValueError(f"transformer nhead must be >= 1, got {requested_nhead}")
    cap = min(requested_nhead, d_model)
    for h in range(cap, 0, -1):
        if d_model % h == 0:
            return h
    return 1


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
        nhead_r = _transformer_nhead_for_d_model(d_model, nhead)
        if nhead_r != nhead:
            warnings.warn(
                f"CausalTransformerEncoder: d_model={d_model} is not divisible by "
                f"nhead={nhead}; using nhead={nhead_r} for nn.TransformerEncoder. "
                "Set gru_hidden_sizes last value or nhead so they match.",
                UserWarning,
                stacklevel=2,
            )
        self.nhead = nhead_r
        self.input_proj = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead_r,
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
        hidden_sizes: list[int] = None,
        slow_kernel: int = 5,
        slow_stride: int = 2,
        temporal_encoder: str = "legacy",
    ):
        super().__init__()
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
