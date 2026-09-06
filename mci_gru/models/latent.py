import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketLatentStateLearner(nn.Module):
    """
    Multi-head cross-attention for learning latent market states (R1, R2),
    with either the legacy 8-Linear MHA (paper) or :class:`nn.MultiheadAttention`.

    ``market_latent_mode`` selects what the latents are:

    ``"static"`` (default, the shipped behaviour)
        ``R1`` and ``R2`` are plain parameters, frozen after training, so each
        stock's output is a function of its own vector alone. The streams
        cannot observe the date's market (issue #198).
    ``"data_dependent"``
        The latents first read the date's active cross-section, then every
        stock reads those date-conditioned latents. This is the Set Transformer
        induced-set construction. It costs ``O(N*k)`` in the number of names.

    The two modes hold different parameters, so a checkpoint belongs to the
    mode that produced it.
    """

    VALID_MODES = ("static", "data_dependent")

    def __init__(
        self,
        feature_dim: int,
        num_latent_states: int = 32,
        num_heads: int = 4,
        latent_init_scale: float = 0.02,
        use_nn_multihead_attention: bool = False,
        attn_dropout: float = 0.0,
        market_latent_mode: str = "static",
    ):
        super().__init__()
        if market_latent_mode not in self.VALID_MODES:
            raise ValueError(
                f"market_latent_mode must be one of {self.VALID_MODES}, got {market_latent_mode!r}"
            )
        self.num_latent_states = num_latent_states
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_nn_multihead_attention = use_nn_multihead_attention
        self.market_latent_mode = market_latent_mode

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.R1 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * latent_init_scale)
        self.R2 = nn.Parameter(torch.randn(num_latent_states, feature_dim) * latent_init_scale)

        if market_latent_mode == "data_dependent":
            # Per-date latents need per-date keys, which the legacy shared-key
            # path cannot express, so this mode builds its own attention. Record
            # that on the attribute too, so it does not claim a path this mode
            # never takes. ``ModelConfig`` refuses the contradictory combination
            # rather than letting a config silently mean something else.
            self.use_nn_multihead_attention = True
            self.gather1 = nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
            self.gather2 = nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
            self.mha1 = nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
            self.mha2 = nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
        elif use_nn_multihead_attention:
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

    def _gather_latents(
        self,
        latents: torch.Tensor,
        gather: nn.MultiheadAttention,
        stream: torch.Tensor,
        num_stocks: int,
        stock_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Let the latents read one date's active cross-section.

        ``stream`` arrives flattened as ``(batch * num_stocks, feature_dim)``.
        Returns per-date latents of shape ``(batch, num_latent_states, feature_dim)``.
        """
        batch = stream.shape[0] // num_stocks
        sequence = stream.view(batch, num_stocks, self.feature_dim)
        expanded = latents.unsqueeze(0).expand(batch, -1, -1)

        key_padding_mask = None
        if stock_mask is not None:
            # Excluding inactive names as keys is what keeps the gathered state
            # stable: merely zeroing their rows would still let them occupy a
            # slot in the softmax and dilute every weight.
            #
            # A date with no active names leaves every key padded out. That
            # returns zeros rather than NaN, which is a property of the
            # attention implementation rather than something enforced here;
            # ``test_a_date_with_no_active_names_does_not_produce_nan`` is the
            # canary if a future torch changes it.
            mask = stock_mask.to(dtype=torch.bool, device=stream.device).view(batch, num_stocks)
            key_padding_mask = ~mask

        update, _ = gather(
            expanded,
            sequence,
            sequence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return expanded + update

    def _forward_data_dependent(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        num_stocks: int,
        stock_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = A1.shape[0] // num_stocks
        r1 = self._gather_latents(self.R1, self.gather1, A1, num_stocks, stock_mask)
        r2 = self._gather_latents(self.R2, self.gather2, A2, num_stocks, stock_mask)

        q1 = A1.view(batch, num_stocks, self.feature_dim)
        q2 = A2.view(batch, num_stocks, self.feature_dim)
        b1, _ = self.mha1(q1, r1, r1, need_weights=False)
        b2, _ = self.mha2(q2, r2, r2, need_weights=False)
        return b1.reshape(-1, self.feature_dim), b2.reshape(-1, self.feature_dim)

    def forward(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        num_stocks: int | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.market_latent_mode == "data_dependent":
            if num_stocks is None:
                raise ValueError(
                    "market_latent_mode='data_dependent' needs num_stocks to group the "
                    "flattened stream by date; the caller must pass it."
                )
            return self._forward_data_dependent(A1, A2, num_stocks, stock_mask)
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
