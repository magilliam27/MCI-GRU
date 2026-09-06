"""Scratchpad-only: the proposed fixes for the two attention blocks, as testable modules.

FIX A  ResidualCrossStock  - z + Attn(LN(z)), multi-head, mask re-applied.
       Fixes the cross-stock SelfAttention block's rank collapse.

FIX B  TwoWayLatentBlock   - gather then broadcast through the learned latents:
           R_d = R + MHA(query=R,  kv=Z_active)   latents read the date's cross-section
           Z'  = Z + MHA(query=Z,  kv=R_d)        each stock reads the date's latents
       Fixes MarketLatentStateLearner's fixed-latent cross-attention, which today
       cannot see the date at all. Set Transformer ISAB / Perceiver shape.
       O(N*k) instead of the cross-stock block's O(N^2).

Shared by the diagnostic and the smoke driver. Not repo code.
"""
import torch
import torch.nn as nn


class ResidualCrossStock(nn.Module):
    """FIX A. Wraps the repo's SelfAttention in a pre-norm residual block."""

    def __init__(self, inner, dim, pre_ln=True):
        super().__init__()
        self.inner = inner
        self.ln = nn.LayerNorm(dim) if pre_ln else nn.Identity()

    def forward(self, x, stock_mask=None):
        out = x + self.inner(self.ln(x), stock_mask=stock_mask)
        if stock_mask is not None:
            out = out * stock_mask.unsqueeze(-1).to(out.dtype)
        return out


class TwoWayLatentBlock(nn.Module):
    """FIX B. Data-dependent latents: gather from the cross-section, broadcast back.

    Drop-in for the cross-stock block position (input/output ``(B, N, D)``), so the
    same flag can select it. The same mechanism can instead replace the keys and
    values inside MarketLatentStateLearner; this position is the one that is
    testable without touching the four-stream contract.
    """

    def __init__(self, dim, num_latents=32, num_heads=4, dropout=0.0, latent_init_scale=0.02):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_latents = num_latents
        self.R = nn.Parameter(torch.randn(num_latents, dim) * latent_init_scale)
        self.ln_z_gather = nn.LayerNorm(dim)
        self.ln_r = nn.LayerNorm(dim)
        self.ln_z_broadcast = nn.LayerNorm(dim)
        self.gather = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.broadcast = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x, stock_mask=None):
        b = x.shape[0]
        r = self.R.unsqueeze(0).expand(b, -1, -1)

        key_pad = None
        if stock_mask is not None:
            mask = stock_mask.to(dtype=torch.bool, device=x.device)
            key_pad = ~mask
            # A date with no active names would make the softmax all -inf; let it
            # attend to everything and zero the result at the end instead.
            dead = key_pad.all(dim=-1)
            if dead.any():
                key_pad = key_pad.clone()
                key_pad[dead] = False

        z_kv = self.ln_z_gather(x)
        if stock_mask is not None:
            z_kv = z_kv * mask.unsqueeze(-1).to(z_kv.dtype)
        # gather: the latents read the date's active cross-section
        r_upd, _ = self.gather(
            self.ln_r(r), z_kv, z_kv, key_padding_mask=key_pad, need_weights=False
        )
        r_d = r + r_upd
        # broadcast: every stock reads the date-conditioned latents
        out, _ = self.broadcast(self.ln_z_broadcast(x), r_d, r_d, need_weights=False)
        out = x + out
        if stock_mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


def attach(model, variant, cfg):
    """Swap the cross-stock block of a built StockPredictionModel."""
    if variant in ("residual", "pre_ln_residual"):
        model.self_attention = ResidualCrossStock(
            model.self_attention, model.concat_size, pre_ln=(variant == "pre_ln_residual")
        )
    elif variant == "two_way_latent":
        model.self_attention = TwoWayLatentBlock(
            model.concat_size,
            num_latents=int(cfg.get("num_hidden_states", 32)),
            num_heads=int(cfg.get("cross_attn_heads", 4)),
            dropout=float(cfg.get("trunk_dropout", 0.0)) if cfg.get("use_trunk_regularisation") else 0.0,
            latent_init_scale=float(cfg.get("latent_init_scale", 0.02)),
        )
    elif variant not in ("baseline", "no_self_attention"):
        raise ValueError(f"unknown variant {variant!r}")
    return model
