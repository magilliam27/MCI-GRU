"""Admissible range for `graph.judge_value` (issue 162).

`GraphConfig` is the only gate on this value. `GraphBuilder.__init__` validates
`top_k` and `top_k_metric` but never `judge_value` (`mci_gru/graph/builder.py:64-69`),
and `build_correlation_graph` constructs the builder straight from the config
(`mci_gru/pipeline.py:682-691`). So the config is where the constraint lives, and
the behavioural tests below reach the builder *through* `GraphConfig` rather than
calling `GraphBuilder(judge_value=...)` directly -- that shortcut passes with or
without this change and would prove nothing. See `_edge_attr_through_config` for
where the mirror of production stops.
"""

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import GraphConfig
from mci_gru.graph.builder import GraphBuilder

DEFAULT_JUDGE_VALUE = 0.8
END_DATE = "2021-01-01"


def _two_block_panel(n: int = 12, periods: int = 300, seed: int = 1729) -> pd.DataFrame:
    """Half the names load +1 on a common factor and half load -1.

    Within a block, correlations are strongly positive; across blocks they are
    strongly negative. That is the structure the threshold path could never
    reach before this change.
    """
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


def _edge_attr_through_config(judge_value: float) -> np.ndarray:
    """Construct the builder from a validated config, as `build_correlation_graph` does.

    The *construction* mirrors `mci_gru/pipeline.py:682-691`. The *call* is
    deliberately narrower than production's: no `admissible_mask`, because PIT
    admissibility is not what these tests are about, and a shorter lookback than
    the shipped 252 so the fixture stays small.

    What matters is that judge_value reaches the selection path *through*
    `GraphConfig`. Calling `GraphBuilder(judge_value=...)` directly would bypass
    the only validation there is, so it would pass identically before and after
    this change.
    """
    cfg = GraphConfig(
        judge_value=judge_value,
        use_multi_feature_edges=True,
        corr_lookback_days=250,
    )
    builder = GraphBuilder(
        judge_value=cfg.judge_value,
        update_frequency_months=cfg.update_frequency_months,
        corr_lookback_days=cfg.corr_lookback_days,
        top_k=cfg.top_k,
        top_k_metric=cfg.top_k_metric,
        use_multi_feature_edges=cfg.use_multi_feature_edges,
        use_lead_lag_features=cfg.use_lead_lag_features,
        lead_lag_days=cfg.lead_lag_days,
    )
    df = _two_block_panel()
    _, edge_attr = builder.build_graph(
        df, sorted(df["kdcode"].unique()), end_date=END_DATE, show_progress=False
    )
    return edge_attr.numpy()


def test_negative_judge_value_is_accepted():
    """The threshold arm needs a negative threshold to admit negative correlations."""
    assert GraphConfig(judge_value=-0.5).judge_value == -0.5


def test_judge_value_below_the_correlation_domain_is_rejected():
    """Pearson correlation cannot be under -1, so a threshold under -1 means nothing.

    It is not harmlessly equivalent to -1 either: every value strictly below -1
    selects the same edge set, so it reads as a deliberate choice the selection
    path cannot express. Note the asymmetry -- -1 itself is *not* in that group,
    because a strict `corr > judge_value` drops an exact -1.0 pair at -1.0 and
    keeps it at -1.5.
    """
    with pytest.raises(ValueError):
        GraphConfig(judge_value=-1.5)


@pytest.mark.parametrize("value", [-1.0, -0.5, 0.0, 0.5, DEFAULT_JUDGE_VALUE, 0.9999])
def test_accepted_values_span_the_correlation_domain(value):
    """`-1.0` is inclusive; `0.0` is a legal threshold meaning "all positive correlations".

    The `-1.0` case is what distinguishes the shipped interval from `-1 < v < 1`;
    the `0.0` case distinguishes it from `0 <= v < 1`. Deleting either lets an
    inverted bound through.
    """
    assert GraphConfig(judge_value=value).judge_value == value


@pytest.mark.parametrize("value", [1.0, 1.5, -1.01, -1.5, float("nan")])
def test_rejected_values_stay_rejected(value):
    """`1.0` must keep raising: `corr <= 1` under a strict `>`, so it is a guaranteed-empty
    graph. `-1.01` catches a floor set loosely at, say, `-1.1`. NaN compares false against
    every bound and must not slip through as "unbounded".
    """
    with pytest.raises(ValueError):
        GraphConfig(judge_value=value)


# --- Seam 2: what a negative threshold actually does to the graph -------------


def test_negative_threshold_admits_negative_correlations():
    """The point of the change. Column 0 is the signed correlation."""
    attr = _edge_attr_through_config(-1.0)
    assert np.any(attr[:, 0] < 0), "a threshold of -1 must keep negatively correlated pairs"


def test_the_extreme_threshold_produces_a_complete_directed_graph():
    """`judge_value = -1` admits every non-NaN off-diagonal pair: `n(n-1)` edges.

    Asserting merely "more edges than 0.8" would not pin the extreme. This
    fixture's correlations are bimodal, so 0.8, 0.0 and -0.5 all select the same
    60 edges; a strict-inequality check would pass for any of them. Only -1
    reaches 132.
    """
    n = _two_block_panel()["kdcode"].nunique()
    shipped = _edge_attr_through_config(DEFAULT_JUDGE_VALUE)
    opened = _edge_attr_through_config(-1.0)

    assert shipped.shape[0] > 0, "fixture produced no edges at 0.8; the comparison is vacuous"
    assert shipped.shape[0] < n * (n - 1), "fixture is already complete at 0.8; nothing to open"
    assert opened.shape[0] == n * (n - 1)


def test_negative_threshold_makes_the_abs_corr_channel_informative():
    """Issue 114's degenerate `|corr|` channel stops being a copy of `corr`.

    114 established that column 1 duplicates column 0 on the threshold path
    *because* the threshold could only be positive. A negative threshold removes
    that cause. This does not resolve 114: the shipped default is still 0.8, and
    the control below pins that the duplication survives there untouched.
    """
    assert not np.array_equal(
        _edge_attr_through_config(-1.0)[:, 1], _edge_attr_through_config(-1.0)[:, 0]
    )


def test_the_shipped_default_still_duplicates_the_abs_corr_channel():
    """Control. Without this, the test above could pass for reasons unrelated to sign."""
    shipped = _edge_attr_through_config(DEFAULT_JUDGE_VALUE)
    assert np.array_equal(shipped[:, 1], shipped[:, 0])
    assert np.all(shipped[:, 0] > 0)
