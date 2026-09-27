"""Twin-exclusion hygiene rule (issue 164 protocol; harness ticket 166).

``graph.exclude_edge_pairs`` removes named stock pairs from every constructed
adjacency — threshold, top-K, dynamic snapshots, and the sector relation — so
the GOOG.OQ/GOOGL.OQ same-company edge cannot ride any ablation arm. Five
maintainer-confirmed seams are pinned here:

1. ``GraphConfig`` validation: the field defaults empty (shipped behaviour
   unchanged), accepts kdcode pairs, and rejects malformed entries.
2. Threshold path: an above-threshold excluded pair yields no edge in either
   direction; unrelated edges are untouched.
3. Top-K path: exclusion is candidate-level, **before** selection — the
   excluded name never wins a slot and the node still receives K clean
   neighbours. Filtering after selection would leave K-1 and is the inversion
   this seam exists to catch.
4. Sector path: the excluded pair is skipped in both directions, the rest of
   the sector stays fully connected, and it composes with ``zero_edges``
   (arm A4).
5. ``build_correlation_graph`` honours the config field end to end on both
   the correlation and sector branches, and on precomputed dynamic snapshots.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from mci_gru.config import GraphConfig
from mci_gru.graph.builder import GraphBuilder
from mci_gru.graph.correlation import build_edges
from mci_gru.graph.sector_edges import build_sector_edges
from mci_gru.pipeline import PipelineFrames, build_correlation_graph

TRAIN_START = "2020-03-02"
TEST_END = "2020-04-30"

CODES = ["AAA.OQ", "BBB.OQ", "CCC.OQ", "DDD.OQ"]
TWIN = ("AAA.OQ", "BBB.OQ")

# AAA/BBB is the twin: 0.95, above the shipped 0.8 threshold. CCC co-moves
# with both; DDD is near-independent.
CORR = pd.DataFrame(
    np.array(
        [
            [1.00, 0.95, 0.90, 0.10],
            [0.95, 1.00, 0.85, 0.05],
            [0.90, 0.85, 1.00, 0.20],
            [0.10, 0.05, 0.20, 1.00],
        ]
    ),
    index=CODES,
    columns=CODES,
)


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in zip(edge_index[0], edge_index[1], strict=True)}


def _build(corr: pd.DataFrame, exclude, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    defaults = {
        "judge_value": 0.8,
        "top_k": 0,
        "top_k_metric": "corr",
        "use_multi_feature_edges": True,
        "use_lead_lag_features": False,
        "lead_lag_days": [1, 2, 3, 5],
    }
    defaults.update(kwargs)
    return build_edges(
        corr,
        CODES,
        False,
        None,
        exclude_pairs=exclude,
        **defaults,
    )


# ── seam 1: config validation ─────────────────────────────────────────────


def test_exclude_edge_pairs_defaults_empty():
    assert GraphConfig().exclude_edge_pairs == []


def test_exclude_edge_pairs_accepts_twin_pair():
    cfg = GraphConfig(exclude_edge_pairs=[["GOOG.OQ", "GOOGL.OQ"]])
    assert cfg.exclude_edge_pairs == [["GOOG.OQ", "GOOGL.OQ"]]


@pytest.mark.parametrize(
    "pairs",
    [
        [["GOOG.OQ"]],
        [["GOOG.OQ", "GOOGL.OQ", "GOOG.OQ"]],
        [["GOOG.OQ", "GOOG.OQ"]],
        [["GOOG.OQ", ""]],
        [["GOOG.OQ", "   "]],
        [[1, 2]],
    ],
)
def test_exclude_edge_pairs_rejects_malformed_entries(pairs):
    with pytest.raises(ValueError, match="exclude_edge_pairs"):
        GraphConfig(exclude_edge_pairs=pairs)


# ── seam 2: threshold path ────────────────────────────────────────────────


def test_threshold_excludes_twin_both_directions_and_keeps_the_rest():
    ei_off, _ = _build(CORR, None)
    ei_on, _ = _build(CORR, [TWIN])

    edges_off = _edge_set(ei_off)
    edges_on = _edge_set(ei_on)

    assert (0, 1) in edges_off and (1, 0) in edges_off
    assert (0, 1) not in edges_on and (1, 0) not in edges_on
    # Every other above-threshold edge survives untouched.
    assert edges_on == edges_off - {(0, 1), (1, 0)}


def test_threshold_exclusion_is_a_noop_for_absent_names():
    ei_off, _ = _build(CORR, None)
    ei_on, _ = _build(CORR, [("ZZZ.OQ", "YYY.OQ")])
    assert _edge_set(ei_on) == _edge_set(ei_off)


# ── seam 3: top-K path (candidate-level, budget refilled) ─────────────────


def test_topk_excluded_name_never_selected_and_budget_refilled():
    ei, _ = _build(CORR, [TWIN], top_k=2, judge_value=0.8)
    edges = _edge_set(ei)

    neighbours_a = {c for r, c in edges if r == 0}
    neighbours_b = {c for r, c in edges if r == 1}

    # Candidate-level exclusion: the twin never wins a slot...
    assert 1 not in neighbours_a
    assert 0 not in neighbours_b
    # ...and the slot is refilled with the next clean candidate, so each
    # twin node still has exactly K neighbours. A post-selection filter
    # would leave K-1 here, which is the inversion this test catches.
    assert neighbours_a == {2, 3}
    assert neighbours_b == {2, 3}


def test_topk_unrelated_rows_are_untouched():
    ei_off, _ = _build(CORR, None, top_k=2)
    ei_on, _ = _build(CORR, [TWIN], top_k=2)

    off_d = {(r, c) for r, c in _edge_set(ei_off) if r == 3}
    on_d = {(r, c) for r, c in _edge_set(ei_on) if r == 3}
    assert on_d == off_d


def test_topk_rank_pct_ranks_against_clean_candidates():
    _, attr = _build(CORR, [TWIN], top_k=2)
    # Multi-feature attr column 3 is rank_pct in (0, 1]; with a full clean
    # K=2 selection every row carries ranks {1.0, 0.5}, so the excluded
    # candidate demonstrably did not occupy a rank position.
    ranks = attr[:, 3].tolist()
    assert set(np.round(ranks, 6)) == {1.0, 0.5}


# ── seam 4: sector path ───────────────────────────────────────────────────


def test_sector_edges_skip_excluded_pair_both_directions():
    sector_map = {"AAA.OQ": "Tech", "BBB.OQ": "Tech", "CCC.OQ": "Tech"}
    ei_off, _ = build_sector_edges(CODES[:3], sector_map)
    ei_on, _ = build_sector_edges(CODES[:3], sector_map, exclude_pairs=[TWIN])

    assert _edge_set(ei_off) == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
    assert _edge_set(ei_on) == {(0, 2), (2, 0), (1, 2), (2, 1)}


# ── seam 5: pipeline end to end ───────────────────────────────────────────


def _panel(codes: list[str]) -> PipelineFrames:
    """Panel whose stocks all share one return series, so every off-diagonal
    correlation is +1 and every pair clears the shipped 0.8 threshold."""
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2020-01-02", periods=60).strftime("%Y-%m-%d")
    steps = rng.normal(0, 0.02, len(sessions))
    close = 100 * np.cumprod(1 + steps)
    df = pd.DataFrame(
        {
            "kdcode": np.repeat(codes, len(sessions)),
            "dt": list(sessions) * len(codes),
            "close": np.tile(close, len(codes)),
        }
    )
    return PipelineFrames(raw=df, normalized=df, filtered=df)


def test_pipeline_honours_exclusion_on_correlation_branch():
    codes = ["AAA.OQ", "BBB.OQ"]
    frames = _panel(codes)

    live = build_correlation_graph(frames, codes, GraphConfig(), TRAIN_START, TEST_END)
    excluded = build_correlation_graph(
        frames,
        codes,
        GraphConfig(exclude_edge_pairs=[["AAA.OQ", "BBB.OQ"]]),
        TRAIN_START,
        TEST_END,
    )

    assert live.edge_index.shape[1] > 0
    assert excluded.edge_index.shape[1] == 0
    # Width contract is preserved on the excluded-to-empty path.
    assert excluded.edge_weight.shape == (0, 4)


def test_pipeline_honours_exclusion_on_sector_branch_with_zero_edges(tmp_path):
    # Arm A4: zero_edges + sector relation; the twin is excluded from the
    # sector adjacency built on the zeroed branch too.
    codes = ["AAA.OQ", "BBB.OQ", "CCC.OQ"]
    frames = _panel(codes)
    sector_csv = tmp_path / "sector_map.csv"
    sector_csv.write_text(
        "kdcode,sector\nAAA.OQ,Tech\nBBB.OQ,Tech\nCCC.OQ,Tech\n", encoding="utf-8"
    )

    art = build_correlation_graph(
        frames,
        codes,
        GraphConfig(
            zero_edges=True,
            use_sector_relation=True,
            sector_map_csv=str(sector_csv),
            exclude_edge_pairs=[["AAA.OQ", "BBB.OQ"]],
        ),
        TRAIN_START,
        TEST_END,
    )

    assert art.edge_index.shape == (2, 0)
    assert art.edge_index_sector is not None
    assert _edge_set(art.edge_index_sector) == {(0, 2), (2, 0), (1, 2), (2, 1)}


def test_precomputed_snapshots_honour_exclusion():
    codes = ["AAA.OQ", "BBB.OQ"]
    frames = _panel(codes)

    def snapshots(exclude):
        builder = GraphBuilder(
            update_frequency_months=6,
            corr_lookback_days=30,
            use_multi_feature_edges=True,
            exclude_edge_pairs=exclude,
        )
        return builder.precompute_snapshots(frames.raw, codes, TRAIN_START, TEST_END)

    def edge_counts(schedule):
        return [schedule.get_graph_for_date(date)[0].shape[1] for date in schedule.snapshot_dates]

    live = snapshots(None)
    excluded = snapshots([("AAA.OQ", "BBB.OQ")])

    assert sum(edge_counts(live)) > 0
    assert all(count == 0 for count in edge_counts(excluded))
