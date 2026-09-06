"""Scratchpad-only initialisation diagnostics on the real MCI-GRU trunk.

Measures, at initialisation and averaged over seeds, how much cross-sectional
(across-stock) variation survives each stage of the trunk, how uniform the
cross-stock attention is, and how gradient norm is distributed across modules
under the IC loss. Compares the shipped trunk against residual variants of the
cross-stock block. Mechanics diagnostic only; not performance evidence.
"""
import sys

WT = r"C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea"
sys.path.insert(0, WT)
import mci_gru  # noqa: E402

print("mci_gru resolved to:", mci_gru.__file__)

import math  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from mci_gru.models.factory import create_model  # noqa: E402
from mci_gru.training.losses import ICLoss  # noqa: E402

cfg = yaml.safe_load(open(WT + "/configs/config.yaml"))["model"]
cfg["edge_feature_dim"] = 4
F_IN, B, N, T = 23, 8, 500, 10
CLUSTERS, DEG = 25, 10


def make_batch(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, T, F_IN, generator=g)
    xg = x[:, :, -1, :].reshape(B * N, F_IN).clone()
    # clustered graph: 25 clusters of 20 names, each node -> DEG random same-cluster neighbours
    size = N // CLUSTERS
    src, dst = [], []
    for c in range(CLUSTERS):
        nodes = torch.arange(c * size, (c + 1) * size)
        for u in nodes:
            nb = nodes[torch.randperm(size, generator=g)[:DEG]]
            src.append(torch.full((DEG,), int(u)))
            dst.append(nb)
    src = torch.cat(src)
    dst = torch.cat(dst)
    E = src.numel()
    off = (torch.arange(B) * N).repeat_interleave(E)
    ei = torch.stack([src.repeat(B) + off, dst.repeat(B) + off])
    ew = torch.rand(ei.shape[1], 4, generator=g)
    # labels: weak linear signal in last-day features plus noise, so IC gradients are meaningful
    w = torch.randn(F_IN, generator=g)
    sig = (x[:, :, -1, :] @ w)
    sig = (sig - sig.mean(1, keepdim=True)) / sig.std(1, keepdim=True)
    y = 0.05 * sig + torch.randn(B, N, generator=g)
    return x, xg, ei, ew, y


class ResidualWrap(nn.Module):
    """x + Attn(x) or x + Attn(LN(x)) around the repo SelfAttention."""

    def __init__(self, inner, dim, pre_ln):
        super().__init__()
        self.inner = inner
        self.ln = nn.LayerNorm(dim) if pre_ln else nn.Identity()

    def forward(self, x, stock_mask=None):
        return x + self.inner(self.ln(x), stock_mask=stock_mask)


def cross_sectional_stats(t):
    """t: (B, N, D). Returns mean pairwise cosine across stocks and cross-sectional variance share."""
    t = t.detach().float()
    tn = F.normalize(t, dim=-1)
    cos = torch.einsum("bnd,bmd->bnm", tn, tn)
    eye = torch.eye(t.shape[1], dtype=torch.bool)
    mean_cos = cos[:, ~eye].mean().item()
    total_var = t.var(dim=(1, 2), unbiased=False).mean().item()
    cs_var = t.var(dim=1, unbiased=False).mean().item()  # variance across stocks, averaged over dims
    # effective rank (participation ratio of singular values) of the centred per-date matrix
    c = t - t.mean(1, keepdim=True)
    s = torch.linalg.svdvals(c)  # (B, min(N,D))
    pr = ((s**2).sum(-1) ** 2 / (s**4).sum(-1).clamp_min(1e-12)).mean().item()
    return mean_cos, cs_var / max(total_var, 1e-12), pr


def attention_entropy(sa, x):
    """Entropy (nats) of the repo SelfAttention weights on input x (B, N, D); uniform = ln N."""
    with torch.no_grad():
        if sa.type_embed is not None:
            b, n, c = x.shape
            x = (x.view(b, n, 4, sa.align_dim) + sa.type_embed.weight.view(1, 1, 4, -1)).reshape(b, n, c)
        q, k = sa.W_q(x), sa.W_k(x)
        a = torch.softmax(q @ k.transpose(-2, -1) * sa.scale, dim=-1)
        ent = -(a * (a + 1e-12).log()).sum(-1).mean().item()
        logit_std = (q @ k.transpose(-2, -1) * sa.scale).std().item()
    return ent, logit_std


def run(variant, seed):
    torch.manual_seed(seed)
    if variant == "no_self_attention":
        m = create_model(F_IN, {**cfg, "use_self_attention": False})
    else:
        m = create_model(F_IN, cfg)
    if variant == "residual":
        m.self_attention = ResidualWrap(m.self_attention, m.concat_size, pre_ln=False)
    elif variant == "pre_ln_residual":
        m.self_attention = ResidualWrap(m.self_attention, m.concat_size, pre_ln=True)
    m.train()
    x, xg, ei, ew, y = make_batch(seed)

    caps = {}

    def hook(name):
        def _h(mod, inp, out):
            caps[name] = out

        return _h

    m.ln_a1.register_forward_hook(hook("a1"))
    m.ln_a2.register_forward_hook(hook("a2"))
    m.latent_learner.register_forward_hook(hook("b"))
    m.ln_z.register_forward_hook(hook("z_pre"))
    if m.self_attention is not None:
        m.self_attention.register_forward_hook(hook("z_post"))
    out = m(x, xg, ei, ew, num_stocks=N)
    loss = ICLoss()(out, y)
    m.zero_grad()
    loss.backward()

    res = {}
    res["a1"] = cross_sectional_stats(caps["a1"].view(B, N, -1))
    res["a2"] = cross_sectional_stats(caps["a2"].view(B, N, -1))
    res["b1"] = cross_sectional_stats(caps["b"][0].view(B, N, -1))
    res["b2"] = cross_sectional_stats(caps["b"][1].view(B, N, -1))
    res["z_pre"] = cross_sectional_stats(caps["z_pre"].view(B, N, -1))
    if "z_post" in caps:
        res["z_post"] = cross_sectional_stats(caps["z_post"].view(B, N, -1))
        inner = m.self_attention.inner if hasattr(m.self_attention, "inner") else m.self_attention
        res["attn"] = attention_entropy(inner, caps["z_pre"].view(B, N, -1))
    o = out.detach()
    res["out_cs_std_over_abs_mean"] = (o.std(1).mean() / o.abs().mean().clamp_min(1e-12)).item()
    res["loss"] = loss.item()
    gn = {}
    for name, sub in m.named_children():
        ps = [p for p in sub.parameters() if p.grad is not None]
        if ps:
            g = torch.sqrt(sum((p.grad**2).sum() for p in ps)).item()
            pn = torch.sqrt(sum((p**2).sum() for p in ps)).item()
            gn[name] = (g, g / max(pn, 1e-12))
    res["grad"] = gn
    return res


VARIANTS = ["baseline", "no_self_attention", "residual", "pre_ln_residual"]
SEEDS = [0, 1, 2]
print(f"\nShapes: B={B}, N={N}, T={T}, F={F_IN}; ln(N) = {math.log(N):.2f}\n")
for v in VARIANTS:
    rs = [run(v, s) for s in SEEDS]

    def avg(key, idx=None):
        vals = [r[key] if idx is None else r[key][idx] for r in rs]
        return sum(vals) / len(vals)

    print(f"===== {v} =====")
    print("stage        mean_cos  cs_var_share  eff_rank")
    for st in ["a1", "a2", "b1", "b2", "z_pre", "z_post"]:
        if st in rs[0]:
            print(f"{st:10s}   {avg(st,0):7.3f}   {avg(st,1):9.3f}   {avg(st,2):7.1f}")
    if "attn" in rs[0]:
        print(f"attention entropy {avg('attn',0):.3f} nats (uniform={math.log(N):.2f}); logit std {avg('attn',1):.3f}")
    print(f"output: cross-sectional std / mean|out| = {avg('out_cs_std_over_abs_mean'):.3f};  IC loss at init = {avg('loss'):+.4f}")
    print("grad-norm by module (abs, rel-to-param-norm):")
    for name in rs[0]["grad"]:
        g = sum(r["grad"][name][0] for r in rs) / len(rs)
        rel = sum(r["grad"][name][1] for r in rs) / len(rs)
        print(f"   {name:18s} {g:9.2e}  {rel:9.2e}")
    print()
