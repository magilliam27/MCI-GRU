"""Scratchpad-only probe of smoke checkpoints on real validation batches.

For each smoke variant, rebuilds the data (same overrides as smoke_variant.py),
scores the UNTRAINED model at the trainer's member seed and the TRAINED
checkpoint (checkpoints/model_0_best.pth) on the same validation batches, and
measures cross-stock attention entropy, diagonal self-weight, top-5 mass, the
cross-sectional variance share and effective rank before and after the block,
and validation IC. Mechanics diagnostic only; not performance evidence.

usage: python trained_attention_probe.py <runs_dir> [max_batches]
"""
import math
import os
import sys

WT = r"C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea"
sys.path.insert(0, WT)
os.chdir(WT)
import mci_gru  # noqa: E402

print("mci_gru resolved to:", mci_gru.__file__, flush=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twoway_block import attach  # noqa: E402

from mci_gru.config import create_config_from_dict  # noqa: E402
from mci_gru.data.data_manager import create_data_loaders  # noqa: E402
from mci_gru.features import FeatureEngineer  # noqa: E402
from mci_gru.graph.utils import edge_feature_dim  # noqa: E402
from mci_gru.models import create_model  # noqa: E402
from mci_gru.models.attention import SelfAttention  # noqa: E402
from mci_gru.pipeline import prepare_data  # noqa: E402
from mci_gru.utils.seeding import set_seed  # noqa: E402

runs_dir = sys.argv[1]
max_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "8")))

PROTECTED = "C:/Users/magil/MCI-GRU"
UNIVERSE = os.environ.get("SMOKE_UNIVERSE", "110")

if UNIVERSE == "110":
    DATA = [
        "data.source=csv",
        f"data.filename={PROTECTED}/data/raw/market/"
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.csv",
        "data.use_pit_universe=true",
        f"data.pit_universe_csv={PROTECTED}/data/raw/constituents/"
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260731_pit_universe.csv",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks=104",
        "data.pit_breadth_policy=error",
        "data.train_start=2016-01-04",
        "data.train_end=2023-12-31",
        "data.val_start=2024-01-22",
        "data.val_end=2024-12-31",
        "data.test_start=2025-01-22",
        "data.test_end=2025-12-31",
    ]
else:
    DATA = [
        "data.source=csv",
        f"data.filename={PROTECTED}/data/raw/market/sp500_2019_universe_data_through_2026.csv",
        "data.use_pit_universe=false",
    ]

OVERRIDES = DATA + [
    "features=with_momentum",
    "features.include_global_regime=false",
    "training.loss_type=ic",
    "training.label_type=returns",
    "training.selection_metric=val_ic",
    "training.shuffle_train=true",
    "model.label_t=5",
    "training.num_models=1",
    "tracking.enabled=false",
    "seed=1729",
    "graph.drop_edge_p=0.1",
]

with initialize_config_dir(version_base=None, config_dir=os.path.join(WT, "configs")):
    cfg = compose(config_name="config", overrides=OVERRIDES)
config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
data = prepare_data(config, FeatureEngineer(config.features))
_, val_loader, _ = create_data_loaders(
    stock_features_train=data["stock_features_train"],
    x_graph_train=data["x_graph_train"],
    train_labels=data["train_labels"],
    stock_features_val=data["stock_features_val"],
    x_graph_val=data["x_graph_val"],
    val_labels=data["val_labels"],
    stock_features_test=data["stock_features_test"],
    x_graph_test=data["x_graph_test"],
    edge_index=data["edge_index"],
    edge_weight=data["edge_weight"],
    batch_size=config.training.batch_size,
    train_dates=data["train_dates"],
    val_dates=data["val_dates"],
    test_dates=data["test_dates"],
    dynamic_graph=False,
    graph_schedule=None,
    shuffle_train=False,
    append_snapshot_age_days=False,
    static_graph_valid_from=data.get("graph_static_valid_from"),
    edge_index_sector=None,
    edge_weight_sector=None,
    use_sector_relation=False,
    train_stock_masks=data.get("train_tradable_mask"),
    val_stock_masks=data.get("val_tradable_mask"),
    test_stock_masks=data.get("test_tradable_mask"),
)
num_features = len(data["feature_cols"])
n_stocks = len(data["kdcode_list"])
print(f"features={num_features} stocks={n_stocks} val_batches={len(val_loader)}", flush=True)
val_batches = []
for bi, batch in enumerate(val_loader):
    if bi >= max_batches:
        break
    val_batches.append(batch)
model_cfg = {
    **config.model.to_dict(),
    "edge_feature_dim": edge_feature_dim(config.graph),
    "drop_edge_p": config.graph.drop_edge_p,
    "use_sector_relation": False,
}


def build(variant, seed=1729):
    vcfg = dict(model_cfg)
    if variant == "no_self_attention":
        vcfg["use_self_attention"] = False
    set_seed(seed)  # trainer seeds member 0 with config.seed + 0 before building
    m = create_model(num_features, vcfg)
    return attach(m, variant, model_cfg)


def batch_mask(bdates, b, n):
    """masked_panel mode carries stock_mask inside the batch metadata dict."""
    if isinstance(bdates, dict) and bdates.get("stock_mask") is not None:
        return bdates["stock_mask"].to(dtype=torch.bool)
    return None


def cs_stats(t, smask=None):
    """Cosine, cross-sectional variance share and effective rank over ACTIVE names."""
    t = t.detach().float()
    cos_l, share_l, pr_l = [], [], []
    for b in range(t.shape[0]):
        row = t[b] if smask is None else t[b][smask[b]]
        if row.shape[0] < 3:
            continue
        rn = F.normalize(row, dim=-1)
        c = rn @ rn.T
        eye = torch.eye(row.shape[0], dtype=torch.bool)
        cos_l.append(c[~eye].mean().item())
        total = row.var(unbiased=False).item()
        share_l.append(row.var(dim=0, unbiased=False).mean().item() / max(total, 1e-12))
        cen = row - row.mean(0, keepdim=True)
        s = torch.linalg.svdvals(cen)
        pr_l.append(((s**2).sum() ** 2 / (s**4).sum().clamp_min(1e-12)).item())
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return mean(cos_l), mean(share_l), mean(pr_l)


def attn_stats(sa, x):
    with torch.no_grad():
        if sa.type_embed is not None:
            b, n, c = x.shape
            x = (x.view(b, n, 4, sa.align_dim) + sa.type_embed.weight.view(1, 1, 4, -1)).reshape(b, n, c)
        logits = sa.W_q(x) @ sa.W_k(x).transpose(-2, -1) * sa.scale
        a = torch.softmax(logits, dim=-1)
        ent = -(a * (a + 1e-12).log()).sum(-1).mean().item()
        diag = torch.diagonal(a, dim1=-2, dim2=-1).mean().item()
        top5 = a.topk(5, dim=-1).values.sum(-1).mean().item()
    return ent, logits.std().item(), diag, top5


def daily_ic(pred, y):
    ics = []
    for p, t in zip(pred, y):
        m = torch.isfinite(p) & torch.isfinite(t)
        if m.sum() < 3:
            continue
        p, t = p[m], t[m]
        p = p - p.mean()
        t = t - t.mean()
        ics.append(((p * t).sum() / (p.norm() * t.norm() + 1e-8)).item())
    return ics


def evaluate(m, label):
    m.eval()
    caps = {}
    handles = []
    if m.self_attention is not None:
        handles.append(m.ln_z.register_forward_hook(lambda mod, i, o: caps.__setitem__("pre", o)))
        handles.append(m.self_attention.register_forward_hook(lambda mod, i, o: caps.__setitem__("post", o)))
    agg = {"ent": [], "lstd": [], "diag": [], "top5": [], "pre": [], "post": [], "ic": [], "out_cs": []}
    with torch.no_grad():
        for batch in val_batches:
            ts, labels, gf, ei, ew, ns, bdates, eis, ews = batch
            B = ts.shape[0]
            smask = batch_mask(bdates, B, ns)
            out = m(ts, gf, ei, ew, num_stocks=ns, stock_mask=smask)
            if smask is not None:
                # score only PIT-active names, as the trainer does
                out = out.masked_fill(~smask, float("nan"))
            if "pre" in caps:
                pre = caps["pre"].view(B, ns, -1)
                post = caps["post"].view(B, ns, -1)
                inner = m.self_attention.inner if hasattr(m.self_attention, "inner") else m.self_attention
                x_in = m.self_attention.ln(pre) if hasattr(m.self_attention, "ln") else pre
                if isinstance(inner, SelfAttention):
                    e, ls, dg, t5 = attn_stats(inner, x_in)
                    agg["ent"].append(e)
                    agg["lstd"].append(ls)
                    agg["diag"].append(dg)
                    agg["top5"].append(t5)
                agg["pre"].append(cs_stats(pre, smask))
                agg["post"].append(cs_stats(post, smask))
            agg["ic"].extend(daily_ic(out, labels))
            agg["out_cs"].append(torch.nanmean(out.std(1)).item())
    for h in handles:
        h.remove()
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    if agg["ent"]:
        pre = [mean([p[i] for p in agg["pre"]]) for i in range(3)]
        post = [mean([p[i] for p in agg["post"]]) for i in range(3)]
        print(
            f"  [{label}] attention entropy {mean(agg['ent']):.3f} nats (uniform={math.log(n_stocks):.2f}); "
            f"logit std {mean(agg['lstd']):.3f}; self-weight {mean(agg['diag']):.4f} (uniform={1 / n_stocks:.4f}); "
            f"top-5 mass {mean(agg['top5']):.3f}"
        )
        print(f"  [{label}] z before block: cos {pre[0]:.3f} cs_var_share {pre[1]:.3f} eff_rank {pre[2]:.1f}")
        print(f"  [{label}] z after  block: cos {post[0]:.3f} cs_var_share {post[1]:.3f} eff_rank {post[2]:.1f}")
    ics = agg["ic"]
    print(
        f"  [{label}] validation daily IC over {len(ics)} days: mean {mean(ics):+.4f}; "
        f"output cross-sectional std {mean(agg['out_cs']):.4f}",
        flush=True,
    )


for variant in ["baseline", "no_self_attention", "pre_ln_residual", "two_way_latent"]:
    print(f"===== {variant} =====")
    m = build(variant)
    evaluate(m, "untrained")
    ck = os.path.join(runs_dir, variant, "checkpoints", "model_0_best.pth")
    if not os.path.exists(ck):
        print(f"  no checkpoint at {ck}")
        continue
    sd = torch.load(ck, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  loaded checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
    evaluate(m, "trained")
