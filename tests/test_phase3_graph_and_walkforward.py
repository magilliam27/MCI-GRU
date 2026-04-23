"""Phase 3: lead-lag toy, snapshot valid_from, walk-forward config, cross-attn smoke."""

from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mci_gru.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig, WalkforwardConfig
from mci_gru.data.data_manager import CombinedDataset, combined_collate_fn
from mci_gru.graph.builder import GraphBuilder, GraphSchedule
from mci_gru.models import create_model
from mci_gru.walkforward import generate_walkforward_configs


def test_graph_schedule_snapshot_valid_from():
    sched = GraphSchedule(
        [
            ("2020-01-01", torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)),
            ("2020-07-01", torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)),
        ]
    )
    assert sched.snapshot_valid_from_for_date("2020-03-01") == "2020-01-01"
    assert sched.snapshot_valid_from_for_date("2020-07-01") == "2020-07-01"


def test_lead_lag_prefers_aligned_lag_on_toy():
    """Stock B is two-day lag of A → best cross-corr at lag 2 vs contemporaneous."""
    dates = pd.bdate_range("2020-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, size=len(dates))
    a = np.cumsum(noise)
    b = np.roll(a, 2)
    b[:2] = 0.0
    rows = []
    for i, dt in enumerate(dates):
        rows.append({"kdcode": "A.N", "dt": dt, "close": 100.0 + float(a[i])})
        rows.append({"kdcode": "B.N", "dt": dt, "close": 100.0 + float(b[i])})
    df = pd.DataFrame(rows)
    gb = GraphBuilder(
        judge_value=0.3,
        corr_lookback_days=60,
        top_k=1,
        top_k_metric="abs_corr",
        use_multi_feature_edges=True,
        use_lead_lag_features=True,
        lead_lag_days=[1, 2, 3, 5],
    )
    kd = ["A.N", "B.N"]
    pivot = gb._daily_returns_pivot(df, kd, "2020-06-01")
    corr = pivot.corr()
    ei, ew = gb.build_edges(corr, kd, returns_pivot=pivot)
    assert ew.dim() == 2 and ew.shape[1] == 6
    mask = (ei[0] == 0) & (ei[1] == 1)
    if mask.any():
        row = ew[mask][0]
        lag_norm = float(row[4].item())
        assert lag_norm > 0.3


def test_walkforward_generates_at_least_one_window():
    base = ExperimentConfig(
        data=DataConfig(
            train_start="2020-01-01",
            train_end="2022-12-31",
            val_start="2023-02-10",
            val_end="2023-08-31",
            test_start="2023-11-10",
            test_end="2024-06-30",
            skip_embargo_check=False,
        ),
        model=ModelConfig(label_t=5),
        training=TrainingConfig(
            walkforward=WalkforwardConfig(
                enabled=True,
                window_train_years=2,
                window_val_months=3,
                test_span_months=2,
                step_months=6,
                max_windows=2,
            )
        ),
    )
    windows = generate_walkforward_configs(base)
    assert len(windows) >= 1
    for w in windows:
        assert w.data.train_start <= w.data.train_end


def test_cross_attention_forward_runs_and_grad():
    m = create_model(
        4,
        {
            "gru_hidden_sizes": [8, 8],
            "hidden_size_gat1": 8,
            "output_gat1": 4,
            "gat_heads": 2,
            "hidden_size_gat2": 8,
            "num_hidden_states": 4,
            "cross_attn_heads": 2,
            "use_multi_scale": True,
            "use_self_attention": False,
            "use_trunk_regularisation": False,
            "use_a1_a2_cross_attention": True,
            "cross_a2_num_heads": 2,
            "edge_feature_dim": 1,
            "temporal_encoder": "gru_attn",
        },
    )
    b, n, t, f = 2, 3, 5, 4
    x = torch.randn(b, n, t, f, requires_grad=True)
    gf = torch.randn(b * n, f)
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ew = torch.ones(ei.shape[1])
    y = m(x, gf, ei, ew, n)
    assert y.shape == (b, n)
    y.sum().backward()
    assert x.grad is not None


def test_collate_appends_snapshot_age_column():
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ew = torch.ones((2, 4))
    sched = GraphSchedule([("2020-01-01", ei, ew.clone())])
    ts = torch.randn(2, 2, 3, 2)
    gf = torch.randn(2, 2, 2)
    lab = torch.zeros(2, 2)
    ds = CombinedDataset(ts, gf, lab, sample_dates=["2020-06-01", "2020-06-02"])
    loader = DataLoader(
        ds,
        batch_size=2,
        collate_fn=partial(
            combined_collate_fn,
            edge_index=ei,
            edge_weight=ew,
            graph_schedule=sched,
            append_snapshot_age_days=True,
            static_graph_valid_from=None,
        ),
    )
    batch = next(iter(loader))
    assert len(batch) == 9
    _ts, _y, _gf, _ei, b_ew, _ns, _d, _s1, _s2 = batch
    assert b_ew.shape[1] == 5
