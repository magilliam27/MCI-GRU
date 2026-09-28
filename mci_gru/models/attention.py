import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_stock_mask(values: torch.Tensor, stock_mask: torch.Tensor) -> torch.Tensor:
    """Zero the rows of ``values`` belonging to PIT-inactive stocks."""
    mask = stock_mask.to(dtype=torch.bool, device=values.device)
    return values * mask.unsqueeze(-1).to(values.dtype)


class ResidualCrossSectionBlock(nn.Module):
    """Pre-norm residual wrapper around a cross-stock attention module.

    ``SelfAttention`` alone *replaces* the trunk vector ``z`` with its output.
    When its attention is close to uniform, that hands every stock the same
    cross-sectional average and a stock's own features stop reaching the score
    head. Wrapping it as ``z + Attn(LayerNorm(z))`` makes the attention a
    correction to ``z`` rather than a substitute for it.

    The LayerNorm is unconditional: it is what makes the attention's correction
    depend on a stock's shape rather than on its level, so a per-stock constant
    offset leaves that correction unchanged.
    """

    def __init__(self, inner: nn.Module, embed_dim: int):
        super().__init__()
        self.inner = inner
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = x + self.inner(self.norm(x), stock_mask=stock_mask)
        if stock_mask is not None:
            # The residual reintroduces the raw input, including rows the inner
            # module zeroed, so the mask has to be applied again after the add.
            out = _apply_stock_mask(out, stock_mask)
        return out


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

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.type_embed is not None:
            b, n, c = x.shape
            if c != 4 * self.align_dim:
                raise ValueError("concat shape mismatch for group type embedding")
            part = x.view(b, n, 4, self.align_dim) + self.type_embed.weight.view(1, 1, 4, -1)
            x = part.reshape(b, n, c)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if stock_mask is not None:
            mask = stock_mask.to(dtype=torch.bool, device=x.device)
            key_mask = mask[:, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            no_valid_keys = ~mask.any(dim=-1)
            if no_valid_keys.any():
                scores[no_valid_keys] = 0.0
        attn = F.softmax(scores, dim=-1)
        if stock_mask is not None:
            mask_f = mask.to(attn.dtype)
            attn = attn * mask_f[:, None, :]
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out = torch.matmul(attn, v)
        if stock_mask is not None:
            out = _apply_stock_mask(out, mask)
        return out
