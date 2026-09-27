"""Edge-channel contract for the SHIPPED DEFAULT graph configuration.

`configs/config.yaml` ships `top_k=0` with `use_multi_feature_edges=true`, and
`docs/DEFAULT_EXPERIMENT_RECIPE.md` pins the same pair. Before issue 114 no test
anywhere constructed a GraphBuilder with that combination: every multi-feature
test in the suite passes `top_k` in {1, 2, 3}, and `tests/test_inference_edge_dim.py`
only calls the `edge_feature_dim` helper without ever reaching `build_edges`. A
mutation populating `rank_pct` on the threshold path passed the entire suite.

Two groups below, deliberately separated:

* the shape and the semantics of columns 0-2, which are the contract;
* `test_rank_pct_is_inert_in_threshold_mode`, which documents the *defect*.
  Delete that one when the defect is resolved rather than editing an assertion
  out of a larger block.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from mci_gru.graph.builder import GraphBuilder

DEFAULT_JUDGE_VALUE = 0.8


def _correlated_panel(n: int = 12, periods: int = 300, seed: int = 1729) -> pd.DataFrame:
    """A panel whose correlation matrix has strong positive AND negative structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=periods)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows = []
    for i in range(n):
        loading = 1.0 if i < n // 2 else -1.0
        noise = rng.normal(size=periods) * 0.15
        close = 100.0 * np.exp(np.cumsum(loading * base * 0.01 + noise * 0.01))
        rows.append(pd.DataFrame({"dt": dates, "kdcode": f"S{i}", "close": close}))
    return pd.concat(rows, ignore_index=True)


def _builder(**overrides) -> GraphBuilder:
    kwargs = {
        "judge_value": DEFAULT_JUDGE_VALUE,
        "top_k": 0,
        "top_k_metric": "corr",
        "use_multi_feature_edges": True,
        "corr_lookback_days": 250,
    }
    kwargs.update(overrides)
    return GraphBuilder(**kwargs)


def _edge_attr(builder: GraphBuilder, df: pd.DataFrame) -> np.ndarray:
    codes = sorted(df["kdcode"].unique())
    _, edge_attr = builder.build_graph(df, codes, end_date="2021-01-01", show_progress=False)
    return edge_attr.numpy()


# --- The contract: shape and columns 0-2 on the shipped default ---------------


def test_threshold_mode_emits_four_channels():
    attr = _edge_attr(_builder(), _correlated_panel())
    assert attr.ndim == 2
    assert attr.shape[1] == 4
    assert attr.shape[0] > 0, "fixture produced no edges; the test would be vacuous"


def test_threshold_mode_column_semantics():
    attr = _edge_attr(_builder(), _correlated_panel())
    corr = attr[:, 0]

    assert np.all(corr > DEFAULT_JUDGE_VALUE), "column 0 must be the thresholded correlation"
    np.testing.assert_allclose(attr[:, 1], np.abs(corr), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(attr[:, 2], corr.astype(np.float64) ** 2, rtol=1e-6, atol=1e-7)


def test_the_shipped_default_admits_only_positive_correlations():
    """At judge_value=0.8 every kept correlation is positive.

    This is a property of the shipped *threshold*, not of the validator. Issue 162
    relaxed the bound to [-1, 1), so the threshold path can now keep negatives --
    just not at this default. The bound itself, and what a negative threshold does
    to the graph, are covered by tests/test_graph_config_judge_value.py.
    """
    attr = _edge_attr(_builder(), _correlated_panel())
    assert np.all(attr[:, 0] > 0)


# --- The defect, isolated so it can be deleted when resolved ------------------


def test_rank_pct_is_inert_in_threshold_mode():
    """DEFECT (issue 114): column 3 is a constant zero on the shipped default.

    Delete this test when the defect is resolved -- do not relax it in place.
    """
    attr = _edge_attr(_builder(), _correlated_panel())
    assert np.all(attr[:, 3] == 0.0), "rank_pct is no longer inert; this test should be removed"


def test_abs_corr_duplicates_corr_in_threshold_mode():
    """DEFECT (issue 114): column 1 is a bit-identical copy of column 0.

    Structural at this threshold, not incidental: the path keeps `corr > judge_value`
    and this fixture builds at 0.8, so every kept correlation is positive. Since issue
    162 that is no longer true of threshold mode in general -- a negative judge_value
    makes column 1 informative. The shipped default is still 0.8, so the defect stands.
    Delete this test when the defect is resolved.
    """
    attr = _edge_attr(_builder(), _correlated_panel())
    assert np.array_equal(attr[:, 1], attr[:, 0])


# --- Controls: the assertions above must be able to fail ----------------------


def test_top_k_mode_populates_rank_pct_and_breaks_the_duplication():
    """Control for both defect tests. Same builder, top_k>0, on a panel with negatives.

    Without this, `rank_pct == 0` and `|corr| == corr` could hold for reasons
    unrelated to the selection path and the two tests above would prove nothing.
    """
    attr = _edge_attr(
        _builder(top_k=4, top_k_metric="abs_corr", judge_value=0.3), _correlated_panel()
    )

    assert not np.all(attr[:, 3] == 0.0), "rank_pct must be populated under top-K"
    assert attr[:, 3].max() == pytest.approx(1.0)
    assert np.any(attr[:, 0] < 0), "abs_corr metric must admit negative correlations"
    assert not np.array_equal(attr[:, 1], attr[:, 0]), "|corr| must differ once negatives are kept"


def test_scalar_mode_emits_one_channel():
    """Control on the shape assertion: the width is not unconditionally 4."""
    _, edge_weight = _builder(use_multi_feature_edges=False).build_graph(
        _correlated_panel(),
        sorted(_correlated_panel()["kdcode"].unique()),
        end_date="2021-01-01",
        show_progress=False,
    )
    assert edge_weight.dim() == 1


# --- The warning ---------------------------------------------------------------


def test_build_graph_warns_when_multi_feature_edges_meet_threshold_selection(caplog):
    with caplog.at_level(logging.WARNING, logger="mci_gru.graph.builder"):
        _edge_attr(_builder(), _correlated_panel())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the inert-channel combination must warn"
    assert "rank_pct" in warnings[0].getMessage()


@pytest.mark.parametrize("judge_value", [0.0, 0.5, DEFAULT_JUDGE_VALUE])
def test_the_rank_two_claim_holds_down_to_and_including_zero(judge_value, caplog):
    """Issue 170: zero is on the degenerate side of the branch, not the open side.

    `corr > judge_value` is strict, so a threshold of exactly 0.0 still admits
    positive correlations only and |corr| still duplicates corr. A branch written
    `judge_value > 0` rather than `>= 0` would emit the negative-threshold message
    at 0.0 and be wrong; only the 0.0 case separates the two, which is why the
    boundary is parametrised rather than left to the shipped 0.8.
    """
    with caplog.at_level(logging.WARNING, logger="mci_gru.graph.builder"):
        _edge_attr(_builder(judge_value=judge_value), _correlated_panel())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the inert-channel combination must warn"
    message = warnings[0].getMessage()
    assert "duplicates corr" in message, "|corr| is a copy of corr at a non-negative threshold"
    assert "rank 2" in message


@pytest.mark.parametrize("judge_value", [-0.001, -0.5, -1.0])
def test_negative_threshold_warns_without_claiming_rank_two(judge_value, caplog):
    """Issue 170: below zero the |corr| half of the warning stops being true.

    The warning must still fire. `rank_pct` is inert at every threshold, so gating
    the whole warning on a non-negative `judge_value` would leave the negative
    threshold arm with a silently dead channel -- the exact defect (issue 114) this
    warning exists to surface. What changes is the claim: `corr > judge_value`
    admits negative correlations here, so |corr| is no longer forced to duplicate
    corr and the tensor is not driven to numerical rank 2.

    `-0.001` is the boundary case. A branch written as `judge_value > 0` instead of
    `>= 0` sends 0.0 down the wrong arm; a branch written `>= -1` sends every value
    down this one. Parametrising both sides of zero is what separates them.
    """
    with caplog.at_level(logging.WARNING, logger="mci_gru.graph.builder"):
        _edge_attr(_builder(judge_value=judge_value), _correlated_panel())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "rank_pct is still inert below zero; the warning must not go silent"
    message = warnings[0].getMessage()
    assert "rank_pct" in message, "the still-true half of the warning must survive"
    assert "rank 2" not in message, (
        "the rank-2 claim is false below zero: |corr| is informative for any kept "
        "pair with corr < 0"
    )
    assert "rank at most 3" in message, "only rank_pct is guaranteed inert below zero"
    assert str(judge_value) in message, (
        "the operator has to be able to see which arm they are on from the warning"
    )


def test_no_warning_when_the_combination_is_not_degenerate(caplog):
    """Control: the warning must not fire for configurations that are fine.

    A negative `judge_value` is deliberately **not** in this list (issue 170). It
    makes the |corr| channel informative but leaves `rank_pct` identically zero, so
    it is less degenerate rather than not degenerate, and it keeps warning. The test
    directly above pins what it says instead.
    """
    for builder in (
        _builder(top_k=4, judge_value=0.3),
        _builder(use_multi_feature_edges=False),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mci_gru.graph.builder"):
            builder.build_graph(
                _correlated_panel(),
                sorted(_correlated_panel()["kdcode"].unique()),
                end_date="2021-01-01",
                show_progress=False,
            )
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
