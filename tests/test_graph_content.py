"""Constrain what the correlation graph *contains*, not merely its shape.

Before these tests, no test in the repository related the graph's edges to the
data they are supposed to describe. The full non-slow suite passed with a
fabricated fully-connected graph, and with a graph missing every edge touching
an odd-indexed node. Shape was checked, schedule geometry was checked, content
was not. See issue #135.

The design constraint is that these must fail under two mutations of
``GraphBuilder.build_graph``:

* ``complete`` - connect every ordered node pair. The maximal point-in-time
  violation: every name wired together regardless of membership or correlation.
* ``isolate_half`` - drop every edge touching an odd-indexed node, so half the
  graph silently loses all connectivity.

A test suite that passes both mutations is not constraining content. Each test
below names which mutation it is the oracle for.

The fixture builds two perfectly-correlated pairs that are mutually
uncorrelated, so the correct edge set is small, exactly known, and both
"too many edges" and "too few edges" are detectable.
"""

import numpy as np
import pandas as pd
import pytest

from mci_gru.graph.builder import GraphBuilder

# Two blocks: (AAA, BBB) move together, (CCC, DDD) move together, and the
# blocks are independent of each other. Index parity matters for the
# isolate_half oracle: BBB and DDD sit at odd positions.
KDCODES = ["AAA", "BBB", "CCC", "DDD"]
JUDGE_VALUE = 0.8
N_SESSIONS = 300


def _two_block_returns() -> pd.DataFrame:
    """Long-format OHLCV whose returns have a known two-block correlation."""
    rng = np.random.default_rng(20260730)
    dates = pd.bdate_range("2023-01-02", periods=N_SESSIONS)

    block_one = rng.normal(0.0, 0.01, N_SESSIONS)
    block_two = rng.normal(0.0, 0.01, N_SESSIONS)
    # Small idiosyncratic noise keeps correlation just under 1.0 without
    # dropping it near the threshold.
    noise = lambda: rng.normal(0.0, 0.0005, N_SESSIONS)  # noqa: E731

    series = {
        "AAA": block_one + noise(),
        "BBB": block_one + noise(),
        "CCC": block_two + noise(),
        "DDD": block_two + noise(),
    }

    rows = []
    for kdcode, rets in series.items():
        close = 100.0 * np.cumprod(1.0 + rets)
        for dt, px in zip(dates, close, strict=True):
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1_000_000.0,
                    "turnover": 1_000_000.0 * px,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _two_block_returns()


@pytest.fixture(scope="module")
def end_date(panel: pd.DataFrame) -> str:
    return str(panel["dt"].max())


def _edge_pairs(edge_index) -> set[tuple[str, str]]:
    """Edge tensor to a set of (source, destination) name pairs."""
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    return {(KDCODES[a], KDCODES[b]) for a, b in zip(src, dst, strict=True)}


def _within_block(pair: tuple[str, str]) -> bool:
    blocks = [{"AAA", "BBB"}, {"CCC", "DDD"}]
    return any({pair[0], pair[1]} <= block for block in blocks)


def test_fixture_has_the_correlation_structure_the_other_tests_assume(panel, end_date):
    """Guard the guard: if the fixture drifts, the oracles below mean nothing."""
    builder = GraphBuilder(judge_value=JUDGE_VALUE, corr_lookback_days=N_SESSIONS)
    builder.build_graph(panel, KDCODES, end_date, show_progress=False)
    corr = builder.correlation_matrix

    assert corr.loc["AAA", "BBB"] > 0.9
    assert corr.loc["CCC", "DDD"] > 0.9
    for a, b in [("AAA", "CCC"), ("AAA", "DDD"), ("BBB", "CCC"), ("BBB", "DDD")]:
        assert abs(corr.loc[a, b]) < 0.5, f"{a}-{b} should be near-independent"


def test_threshold_graph_excludes_uncorrelated_pairs(panel, end_date):
    """Oracle for the ``complete`` mutation.

    A fully-connected graph contains cross-block edges. Correlation between the
    blocks is near zero, far below the threshold, so no such edge is legitimate.
    """
    builder = GraphBuilder(judge_value=JUDGE_VALUE, corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)

    cross_block = {pair for pair in _edge_pairs(edge_index) if not _within_block(pair)}
    assert not cross_block, (
        f"Graph connects uncorrelated names: {sorted(cross_block)}. "
        f"With judge_value={JUDGE_VALUE} only within-block pairs qualify."
    )


def test_threshold_graph_includes_every_correlated_pair(panel, end_date):
    """Oracle for the ``isolate_half`` mutation.

    Dropping edges that touch odd-indexed nodes removes BBB and DDD, which
    destroys both legitimate pairs. Absence of edges is as much a content
    defect as excess.
    """
    builder = GraphBuilder(judge_value=JUDGE_VALUE, corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)

    pairs = _edge_pairs(edge_index)
    for a, b in [("AAA", "BBB"), ("CCC", "DDD")]:
        assert (a, b) in pairs or (b, a) in pairs, (
            f"{a}-{b} correlate above {JUDGE_VALUE} but have no edge. "
            f"Present edges: {sorted(pairs)}"
        )


def test_edge_weights_equal_the_computed_correlation(panel, end_date):
    """Content, not shape: a fabricated edge cannot carry a truthful weight.

    On the scalar path the weight is the signed correlation for that pair, so
    every edge is checkable against the matrix the builder itself computed.
    """
    builder = GraphBuilder(judge_value=JUDGE_VALUE, corr_lookback_days=N_SESSIONS)
    edge_index, edge_weight = builder.build_graph(panel, KDCODES, end_date, show_progress=False)
    corr = builder.correlation_matrix

    n_edges = edge_index.shape[1]
    assert n_edges > 0, "fixture should produce edges"
    weights = edge_weight.reshape(n_edges, -1)[:, 0].tolist()

    # Iterate the tensor in order rather than via _edge_pairs, which returns an
    # unordered set and would pair each weight with an arbitrary edge.
    for position in range(n_edges):
        src = KDCODES[int(edge_index[0][position])]
        dst = KDCODES[int(edge_index[1][position])]
        weight = weights[position]
        expected = float(corr.loc[src, dst])
        # Tolerance is float32-scale: edge tensors are float32 while the
        # correlation matrix is float64, and the difference accumulates over
        # the lookback. Still far tighter than any fabricated weight.
        assert weight == pytest.approx(expected, abs=1e-4), (
            f"edge {src}->{dst} carries weight {weight} but the computed correlation is {expected}"
        )
        assert weight > JUDGE_VALUE, (
            f"edge {src}->{dst} has weight {weight}, at or below "
            f"judge_value={JUDGE_VALUE}; it should not have been selected"
        )


def test_top_k_selects_the_most_correlated_neighbours(panel, end_date):
    """Oracle for both mutations on the top-K path.

    With K=1 each node must select its single strongest neighbour, which is its
    block partner. ``complete`` overshoots the budget; ``isolate_half`` removes
    the only legitimate choice for two of the four nodes.
    """
    builder = GraphBuilder(top_k=1, top_k_metric="corr", corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)

    pairs = _edge_pairs(edge_index)
    partner = {"AAA": "BBB", "BBB": "AAA", "CCC": "DDD", "DDD": "CCC"}

    outgoing: dict[str, set[str]] = {name: set() for name in KDCODES}
    for src, dst in pairs:
        outgoing[src].add(dst)

    for name in KDCODES:
        assert outgoing[name] == {partner[name]}, (
            f"{name} should select exactly its block partner {partner[name]} "
            f"at K=1, but selected {sorted(outgoing[name])}"
        )
