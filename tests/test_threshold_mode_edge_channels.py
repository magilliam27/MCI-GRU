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

from mci_gru.config import GraphConfig
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


def test_threshold_mode_admits_only_positive_correlations():
    """GraphConfig forbids judge_value <= 0, so the threshold path cannot keep negatives."""
    attr = _edge_attr(_builder(), _correlated_panel())
    assert np.all(attr[:, 0] > 0)

    for bad in (-0.5, 0.0):
        with pytest.raises(ValueError):
            GraphConfig(judge_value=bad)


# --- The defect, isolated so it can be deleted when resolved ------------------


def test_rank_pct_is_inert_in_threshold_mode():
    """DEFECT (issue 114): column 3 is a constant zero on the shipped default.

    Delete this test when the defect is resolved -- do not relax it in place.
    """
    attr = _edge_attr(_builder(), _correlated_panel())
    assert np.all(attr[:, 3] == 0.0), "rank_pct is no longer inert; this test should be removed"


def test_abs_corr_duplicates_corr_in_threshold_mode():
    """DEFECT (issue 114): column 1 is a bit-identical copy of column 0.

    Structural, not incidental: the threshold path keeps `corr > judge_value` and
    judge_value must be positive, so every kept correlation is positive.
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


def test_no_warning_when_the_combination_is_not_degenerate(caplog):
    """Control: the warning must not fire for configurations that are fine."""
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
