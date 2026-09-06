"""Scratchpad-only training-dynamics smoke.

Runs the repo's own run_experiment.main() on the local anchored 2019-snapshot
universe CSV with the cross-stock attention block swapped for a residual variant
via a monkeypatched create_model. One seed, few epochs, CPU, no global regime.
Mechanics and early-dynamics check only; not performance evidence.

usage: python smoke_variant.py <variant> <epochs> <seed> <outdir> [lr] [warmup]
variant in {baseline, no_self_attention, residual, pre_ln_residual}
"""
import os
import sys

WT = r"C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea"
sys.path.insert(0, WT)
os.chdir(WT)
import mci_gru  # noqa: E402

print("mci_gru resolved to:", mci_gru.__file__, flush=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

variant = sys.argv[1]
epochs = int(sys.argv[2])
seed = int(sys.argv[3])
outdir = sys.argv[4]
lr = sys.argv[5] if len(sys.argv) > 5 else "5e-5"
warmup = sys.argv[6] if len(sys.argv) > 6 else "40"
threads = int(os.environ.get("SMOKE_THREADS", "4"))
torch.set_num_threads(threads)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiment as rx  # noqa: E402
from twoway_block import attach  # noqa: E402

from mci_gru.models import factory as fac  # noqa: E402

_orig_create_model = fac.create_model


def patched_create_model(n_features, cfg):
    m = _orig_create_model(n_features, cfg)
    if m.self_attention is not None or variant == "no_self_attention":
        m = attach(m, variant, cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"[smoke] variant={variant} params={n}", flush=True)
    return m


fac.create_model = patched_create_model
for modname in list(sys.modules):
    mod = sys.modules[modname]
    if mod is not None and getattr(mod, "create_model", None) is _orig_create_model:
        setattr(mod, "create_model", patched_create_model)

PROTECTED = "C:/Users/magil/MCI-GRU"
UNIVERSE = os.environ.get("SMOKE_UNIVERSE", "110")

if UNIVERSE == "110":
    # The universe the project actually works on: GICS sector top-10 by market
    # cap, monthly rebalance, ~110 admissible names per session on a 206-name
    # PIT union axis. Settings copied from configs/data/gics_top10_110_2016.yaml
    # at ref 8c1f9c7 (that config is not on this worktree's branch).
    data_overrides = [
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
    data_overrides = [
        "data.source=csv",
        f"data.filename={PROTECTED}/data/raw/market/sp500_2019_universe_data_through_2026.csv",
        "data.use_pit_universe=false",
    ]

overrides = data_overrides + [
    "features=with_momentum",
    "features.include_global_regime=false",
    "training.loss_type=ic",
    "training.label_type=returns",
    "training.selection_metric=val_ic",
    "training.shuffle_train=true",
    "model.label_t=5",
    f"training.num_epochs={epochs}",
    "training.num_models=1",
    f"training.early_stopping_patience={epochs}",
    f"training.learning_rate={lr}",
    f"training.warmup_steps={warmup}",
    "training.use_amp=false",
    "tracking.enabled=false",
    "evaluation.bootstrap_enabled=false",
    f"seed={seed}",
    f"experiment_name=smoke_{variant}",
    "graph.drop_edge_p=0.1",
]
if variant == "no_self_attention":
    overrides.append("model.use_self_attention=false")

# Hydra resolves a relative config_path against the *calling* file, which here is
# this scratchpad script, so compose explicitly from the repo's absolute configs dir
# and call the undecorated main. main() falls back to os.getcwd() as its output
# path when no Hydra run is active, so chdir into the run directory first.
from hydra import compose, initialize_config_dir  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=os.path.join(WT, "configs")):
    cfg = compose(config_name="config", overrides=overrides)
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
print(f"[smoke] output dir {outdir}", flush=True)
rx.main.__wrapped__(cfg)
