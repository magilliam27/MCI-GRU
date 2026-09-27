"""Scratchpad-only: does each proposed fix actually fix what it claims?

Checks, at initialisation on synthetic inputs at frozen-recipe shapes:
  1. cross-sectional variance survival and effective rank through the block
  2. PIT masking: inactive nodes stay zero AND cannot influence active ones
  3. date-sensitivity: does the block's output for a fixed stock change when the
     REST of the cross-section changes? (the property the shipped latent
     cross-attention provably lacks)
  4. parameter count and step time
Mechanics only. Not performance evidence.
"""
import sys
import time

WT = r"C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea"
SP = r"C:\Users\magil\AppData\Local\Temp\claude\C--Users-magil-MCI-GRU--claude-worktrees-looped-transformers-mci-gru-0592ea\ef6daf14-4d68-4b7f-b550-4a3167f9e125\scratchpad"
sys.path.insert(0, WT)
sys.path.insert(0, SP)
import mci_gru  # noqa: E402

print("mci_gru resolved to:", mci_gru.__file__)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from twoway_block import attach  # noqa: E402

from mci_gru.models.factory import create_model  # noqa: E402
from mci_gru.training.losses import ICLoss  # noqa: E402

cfg = yaml.safe_load(open(WT + "/configs/config.yaml"))["model"]
cfg["edge_feature_dim"] = 4
# Shapes match the universe the project actually uses: gics_top10_110_2016.
# 201-node PIT union axis, ~110 admissible names per session, so ~91 masked.
F_IN, B, N, T = 16, 8, 201, 10
N_ACTIVE = 110
CLUSTERS, DEG = 11, 8
VARIANTS = ["baseline", "no_self_attention", "pre_ln_residual", "two_way_latent"]


def make_batch(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, T, F_IN, generator=g)
    xg = x[:, :, -1, :].reshape(B * N, F_IN).clone()
    size = N // CLUSTERS
    src, dst = [], []
    for c in range(CLUSTERS):
        nodes = torch.arange(c * size, (c + 1) * size)
        for u in nodes:
            src.append(torch.full((DEG,), int(u)))
            dst.append(nodes[torch.randperm(size, generator=g)[:DEG]])
    src, dst = torch.cat(src), torch.cat(dst)
    e = src.numel()
    off = (torch.arange(B) * N).repeat_interleave(e)
    ei = torch.stack([src.repeat(B) + off, dst.repeat(B) + off])
    ew = torch.rand(ei.shape[1], 4, generator=g)
    w = torch.randn(F_IN, generator=g)
    sig = x[:, :, -1, :] @ w
    sig = (sig - sig.mean(1, keepdim=True)) / sig.std(1, keepdim=True)
    y = 0.05 * sig + torch.randn(B, N, generator=g)
    return x, xg, ei, ew, y


def cs_stats(t):
    t = t.detach().float()
    tn = F.normalize(t, dim=-1)
    cos = torch.einsum("bnd,bmd->bnm", tn, tn)
    eye = torch.eye(t.shape[1], dtype=torch.bool)
    total = t.var(dim=(1, 2), unbiased=False).mean().item()
    cs = t.var(dim=1, unbiased=False).mean().item()
    c = t - t.mean(1, keepdim=True)
    s = torch.linalg.svdvals(c)
    pr = ((s**2).sum(-1) ** 2 / (s**4).sum(-1).clamp_min(1e-12)).mean().item()
    return cos[:, ~eye].mean().item(), cs / max(total, 1e-12), pr


def build(variant, seed):
    torch.manual_seed(seed)
    c = dict(cfg)
    if variant == "no_self_attention":
        c["use_self_attention"] = False
    m = create_model(F_IN, c)
    return attach(m, variant, cfg)


print(f"\nShapes B={B} N={N} T={T} F={F_IN}\n")
x, xg, ei, ew, y = make_batch(0)
mask = torch.ones(B, N, dtype=torch.bool)
mask[:, N_ACTIVE:] = False

for v in VARIANTS:
    m = build(v, 0)
    caps = {}
    if m.self_attention is not None:
        m.ln_z.register_forward_hook(lambda mo, i, o: caps.__setitem__("pre", o))
        m.self_attention.register_forward_hook(lambda mo, i, o: caps.__setitem__("post", o))
    m.train()
    out = m(x, xg, ei, ew, num_stocks=N, stock_mask=mask)
    loss = ICLoss()(out.masked_fill(~mask, float("nan")), y)
    m.zero_grad()
    loss.backward()

    n_tot = sum(p.numel() for p in m.parameters())
    n_blk = 0 if m.self_attention is None else sum(p.numel() for p in m.self_attention.parameters())
    line = f"===== {v} =====\n  params total {n_tot:6d}  block {n_blk:6d}"
    if "post" in caps:
        pre = cs_stats(caps["pre"].view(B, N, -1))
        post = cs_stats(caps["post"].view(B, N, -1))
        line += (
            f"\n  z before block: cos {pre[0]:.3f} cs_var_share {pre[1]:.3f} eff_rank {pre[2]:5.1f}"
            f"\n  z after  block: cos {post[0]:.3f} cs_var_share {post[1]:.3f} eff_rank {post[2]:5.1f}"
        )
    print(line)

    # 2. masking: inactive nodes zero at the output
    m.eval()
    with torch.no_grad():
        o1 = m(x, xg, ei, ew, num_stocks=N, stock_mask=mask)
        inactive_zero = bool((o1[:, N_ACTIVE:].abs().max() == 0).item())
        # ... and perturbing an INACTIVE stock must not move any ACTIVE score
        x2 = x.clone()
        x2[:, -3, :, :] += 5.0
        xg2 = xg.clone().view(B, N, -1)
        xg2[:, -3, :] += 5.0
        o2 = m(x2, xg2.reshape(B * N, -1), ei, ew, num_stocks=N, stock_mask=mask)
        leak = (o1[:, :N_ACTIVE] - o2[:, :N_ACTIVE]).abs().max().item()
        # 3. date-sensitivity: perturb OTHER active stocks, does stock 0 move?
        x3 = x.clone()
        x3[:, 40:100, :, :] += 2.0
        xg3 = xg.clone().view(B, N, -1)
        xg3[:, 40:100, :] += 2.0
        o3 = m(x3, xg3.reshape(B * N, -1), ei, ew, num_stocks=N, stock_mask=mask)
        # 201 nodes / 11 clusters = 18 per cluster, so stock 0 is in cluster 0
        # (nodes 0-17) and the perturbed 40-100 sit in clusters 2-5: no graph path.
        # Raw change is misleading: ICLoss centres per date, so a block that shifts
        # every stock equally scores high here while contributing nothing. Measure
        # the change in stock 0's CENTRED score, which is what the loss sees.
        raw = (o1[:, 0] - o3[:, 0]).abs().mean().item()

        def centred(o):
            a = o.masked_fill(~mask, float("nan"))
            return o[:, 0] - torch.nanmean(a, dim=1)

        eff = (centred(o1) - centred(o3)).abs().mean().item()
    print(
        f"  masking: inactive_zero={inactive_zero}  leak_from_inactive={leak:.2e}"
        f"\n  date-sensitivity when stocks 40-100 move: raw |dscore_0| {raw:.4f}"
        f"  ->  CENTRED (what ICLoss sees) {eff:.4f}"
    )

    m.train()
    m(x, xg, ei, ew, num_stocks=N, stock_mask=mask).sum().backward()
    t0 = time.time()
    for _ in range(2):
        m.zero_grad()
        m(x, xg, ei, ew, num_stocks=N, stock_mask=mask).sum().backward()
    print(f"  fwd+bwd {(time.time() - t0) / 2:.2f}s\n")
