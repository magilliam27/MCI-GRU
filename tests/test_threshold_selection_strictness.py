"""Strictness of the threshold comparison in `build_edges` (issue 170).

`_select_edges_threshold` keeps `corr > judge_value`, strictly. Before this file
existed, mutating that one character to `>=` passed the **entire** suite --
measured, not assumed. That was harmless while `GraphConfig` floored `judge_value`
above zero, because an exact tie between two real return series is measure-zero.
Issue 162 opened the bound to `[-1, 1)` and documented `-1` as *a floor rather
than a literal "every pair"*, a claim that rests entirely on the comparison being
strict -- and nothing pinned it.

The seam is `mci_gru.graph.correlation.build_edges` with a hand-built correlation
matrix, confirmed with the maintainer before these tests were written. It is
deliberately lower than `GraphBuilder.build_graph`, which cannot reach the
boundary at all: fed two series whose daily returns are exact negatives,
`_daily_returns_pivot` derives returns from `close` via `pct_change`, and
`exp(r) - 1` is not the exact negative of `exp(-r) - 1`, so the pair measures
-0.999930854629442 rather than -1.0. The tie has to be injected.

`mci_gru/graph/correlation.py` is not modified by this file's ticket. These tests
pin behaviour that is already correct, so their red phase is the mutation table
rather than a failing first run.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from mci_gru.graph.builder import GraphBuilder
from mci_gru.graph.correlation import build_edges

THRESHOLDS = [-1.0, -0.5, 0.0, 0.5, 0.8, 0.9999]


def _corr_frame(values: list[list[float]]) -> tuple[pd.DataFrame, list[str]]:
    """A correlation matrix built by hand, with the stock axis `build_edges` expects."""
    codes = [f"S{i}" for i in range(len(values))]
    return pd.DataFrame(values, index=codes, columns=codes, dtype=float), codes


def _threshold_edges(frame: pd.DataFrame, codes: list[str], judge_value: float) -> torch.Tensor:
    """Selection only: threshold path, scalar edges, no lead-lag, no PIT mask.

    `kdcode_list` and `show_progress` are accepted by `build_edges` but unused in
    its body; they are passed positionally to match the production call rather
    than to influence the result.
    """
    edge_index, _ = build_edges(
        frame,
        codes,
        False,
        None,
        judge_value=judge_value,
        top_k=0,
        top_k_metric="corr",
        use_multi_feature_edges=False,
        use_lead_lag_features=False,
        lead_lag_days=[],
    )
    return edge_index


def _directed_pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    rows, cols = edge_index.tolist()
    return set(zip(rows, cols, strict=True))


# --- The comparison is strict at every threshold -----------------------------


@pytest.mark.parametrize("judge_value", THRESHOLDS)
def test_a_pair_at_exactly_the_threshold_is_excluded(judge_value):
    """`corr > judge_value`, not `>=`. This is the assertion the mutation breaks."""
    frame, codes = _corr_frame([[1.0, judge_value], [judge_value, 1.0]])
    assert _threshold_edges(frame, codes, judge_value).shape[1] == 0


@pytest.mark.parametrize("judge_value", THRESHOLDS)
def test_a_pair_one_ulp_above_the_threshold_is_kept(judge_value):
    """Control. Without it, the test above passes for any rule that drops everything.

    One unit in the last place is the tightest margin a double can express, so
    this pins that selection turns on the comparison itself rather than on some
    wider tolerance around it.
    """
    just_above = float(np.nextafter(judge_value, 1.0))
    assert just_above > judge_value, "nextafter produced no gap; the control is vacuous"

    frame, codes = _corr_frame([[1.0, just_above], [just_above, 1.0]])
    assert _threshold_edges(frame, codes, judge_value).shape[1] == 2


# --- What `-1` being a floor actually means (issue 162's decision) ------------


def _floor_fixture() -> tuple[pd.DataFrame, list[str]]:
    """S0 and S1 exactly anti-correlated; S2 uncorrelated with both.

    A genuine correlation matrix -- eigenvalues 2, 1, 0 -- so the fixture is not
    a shape that could never occur. At `judge_value = -1.0` the S0-S1 pair sits
    exactly on the floor and the other two pairs sit above it, which is what lets
    one assertion separate "strict" from "inclusive" without being satisfiable by
    dropping or keeping everything.
    """
    return _corr_frame(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_the_floor_drops_the_pair_sitting_on_it_and_keeps_the_rest():
    """Issue 162 documented `-1` as a floor, not a literal "every pair". This is that.

    Both directions are pinned in one assertion. `>=` would return the complete
    directed graph of 6 edges; a threshold that dropped everything would return 0.
    Only strict `>` returns exactly the four edges not touching the floor.
    """
    frame, codes = _floor_fixture()
    edge_index = _threshold_edges(frame, codes, -1.0)

    assert _directed_pairs(edge_index) == {(0, 2), (2, 0), (1, 2), (2, 1)}


def test_the_floor_is_not_a_complete_directed_graph():
    """The counting form of the assertion above, stated against `n(n-1)`.

    `tests/test_graph_config_judge_value.py` asserts that `judge_value = -1`
    yields `n(n-1)` edges on a real panel, and it does -- because no real pair
    lands on the floor. Read alone it invites "-1 keeps every pair", which is the
    reading 162 explicitly rejected. This is the counterexample.
    """
    frame, codes = _floor_fixture()
    n = len(codes)
    assert _threshold_edges(frame, codes, -1.0).shape[1] == n * (n - 1) - 2


# --- The strictness survives the production facade ---------------------------


def test_graphbuilder_build_edges_preserves_the_strict_comparison():
    """`GraphBuilder.build_edges` is what `build_graph` calls; it must not soften this.

    The module-level tests above would still pass if the facade stopped forwarding
    `judge_value` correctly, so the seam is exercised one level up as well.
    """
    frame, codes = _floor_fixture()
    builder = GraphBuilder(judge_value=-1.0, use_multi_feature_edges=False)
    edge_index, _ = builder.build_edges(frame, codes, show_progress=False)

    assert _directed_pairs(edge_index) == {(0, 2), (2, 0), (1, 2), (2, 1)}
