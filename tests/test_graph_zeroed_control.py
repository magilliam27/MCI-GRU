"""Graph-zeroed control arm (issue 165, semantics fixed by the issue-164 protocol).

``graph.zero_edges=true`` is the ablation's A0 control: the built correlation
edge tensor is forced to shape (2, 0) with the correct feature width, GAT
layers and parameters intact, self-loops only. Three maintainer-confirmed
seams are pinned here:

1. ``GraphConfig`` validation: the flag defaults off, and combining it with a
   dynamic schedule is a config error rather than a silent override.
2. Build contract: with the flag set, ``build_correlation_graph`` returns
   empty edge tensors of the width ``edge_feature_dim`` implies, and no
   schedule. The sector branch (arm A4) still builds.
3. Output invariance: with the flag set, model output is invariant to the
   correlation structure of the panel the graph would have been built from.
   This is the guard that a control arm cannot quietly still use the graph.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from mci_gru.config import GraphConfig
from mci_gru.models import create_model
from mci_gru.pipeline import PipelineFrames, build_correlation_graph

TRAIN_START = "2020-03-02"
TEST_END = "2020-04-30"


def _panel(flip_second_stock: bool) -> PipelineFrames:
    """Two-stock panel; the second stock's returns are either identical to the
    first (corr +1) or sign-flipped (corr -1), so the flag-off graph differs."""
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2020-01-02", periods=60).strftime("%Y-%m-%d")
    steps = rng.normal(0, 0.02, len(sessions))
    close_a = 100 * np.cumprod(1 + steps)
    close_b = 100 * np.cumprod(1 + (-steps if flip_second_stock else steps))
    df = pd.DataFrame(
        {
            "kdcode": ["AAA"] * len(sessions) + ["BBB"] * len(sessions),
            "dt": list(sessions) * 2,
            "close": np.concatenate([close_a, close_b]),
        }
    )
    return PipelineFrames(raw=df, normalized=df, filtered=df)


def _artifacts(frames: PipelineFrames, cfg: GraphConfig):
    return build_correlation_graph(
        frames, ["AAA", "BBB"], cfg, TRAIN_START, TEST_END, first_sample_date=None
    )


# ── seam 1: config validation ─────────────────────────────────────────────


def test_zero_edges_defaults_false():
    assert GraphConfig().zero_edges is False


def test_zero_edges_accepts_true():
    assert GraphConfig(zero_edges=True).zero_edges is True


def test_zero_edges_rejects_dynamic_schedule():
    with pytest.raises(ValueError, match="zero_edges"):
        GraphConfig(zero_edges=True, update_frequency_months=6)


# ── seam 2: build contract ────────────────────────────────────────────────


def test_zeroed_build_is_empty_with_multi_feature_width():
    art = _artifacts(_panel(False), GraphConfig(zero_edges=True))
    assert art.edge_index.shape == (2, 0)
    assert art.edge_index.dtype == torch.long
    assert art.edge_weight.shape == (0, 4)
    assert art.edge_weight.dtype == torch.float
    assert art.graph_schedule is None


def test_zeroed_build_scalar_and_lead_lag_widths():
    scalar = _artifacts(_panel(False), GraphConfig(zero_edges=True, use_multi_feature_edges=False))
    assert scalar.edge_weight.shape == (0,)
    lead_lag = _artifacts(_panel(False), GraphConfig(zero_edges=True, use_lead_lag_features=True))
    assert lead_lag.edge_weight.shape == (0, 6)


def test_zeroed_build_still_builds_sector_branch(tmp_path):
    # Arm A4: zero_edges composes with the sector relation.
    sector_csv = tmp_path / "sector_map.csv"
    sector_csv.write_text("kdcode,sector\nAAA,Tech\nBBB,Tech\n", encoding="utf-8")
    art = _artifacts(
        _panel(False),
        GraphConfig(zero_edges=True, use_sector_relation=True, sector_map_csv=str(sector_csv)),
    )
    assert art.edge_index.shape == (2, 0)
    assert art.edge_index_sector is not None
    assert art.edge_index_sector.shape[1] > 0


# ── seam 3: output invariance (the load-bearing guard) ────────────────────


def _small_model():
    torch.manual_seed(0)
    m = create_model(
        3,
        {
            "gru_hidden_sizes": [4, 4],
            "hidden_size_gat1": 8,
            "output_gat1": 4,
            "gat_heads": 2,
            "hidden_size_gat2": 8,
            "num_hidden_states": 4,
            "cross_attn_heads": 2,
            "use_multi_scale": False,
            "use_self_attention": False,
            "activation": "relu",
            "use_trunk_regularisation": False,
            "use_nn_multihead_attention": False,
            "temporal_encoder": "legacy",
            "edge_feature_dim": 4,
        },
    )
    m.eval()
    return m


def test_zeroed_output_invariant_to_panel_correlation_structure():
    """Two panels whose correlation structure is opposite (+1 vs -1) must give
    bit-identical model outputs when zero_edges is set - and different outputs
    when it is not, proving the test can detect edge influence."""
    zero_cfg = GraphConfig(zero_edges=True)
    live_cfg = GraphConfig()  # shipped threshold 0.8

    art_same_z = _artifacts(_panel(False), zero_cfg)
    art_flip_z = _artifacts(_panel(True), zero_cfg)
    art_same_live = _artifacts(_panel(False), live_cfg)
    art_flip_live = _artifacts(_panel(True), live_cfg)

    # Sanity of the fixture: flag off, the two panels yield different graphs,
    # and the corr=+1 panel yields at least one edge at judge_value=0.8.
    assert art_same_live.edge_index.shape[1] > 0
    assert art_same_live.edge_index.shape[1] != art_flip_live.edge_index.shape[1]

    m = _small_model()
    torch.manual_seed(1)
    x_ts = torch.randn(1, 2, 5, 3)
    x_graph = torch.randn(2, 3)

    with torch.no_grad():
        out_same_z = m(x_ts, x_graph, art_same_z.edge_index, art_same_z.edge_weight)
        out_flip_z = m(x_ts, x_graph, art_flip_z.edge_index, art_flip_z.edge_weight)
        out_same_live = m(x_ts, x_graph, art_same_live.edge_index, art_same_live.edge_weight)
        out_flip_live = m(x_ts, x_graph, art_flip_live.edge_index, art_flip_live.edge_weight)

    assert torch.equal(out_same_z, out_flip_z)
    assert not torch.allclose(out_same_live, out_flip_live)
