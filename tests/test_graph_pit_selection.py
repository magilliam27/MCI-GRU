"""Point-in-time awareness in graph edge *selection*, not merely in output.

Issue #123. The correlation graph had no PIT awareness: ``precompute_snapshots``
took no universe, so a snapshot for 2022 could wire together names that had not
yet joined and names that had already left.

The subtlety measured in the decision brief is that filtering edges *after*
selection repairs nothing, because ``combined_collate_fn`` already does exactly
that per sample date. On the threshold path the two are bit-identical. The real
harm is on the ``top_k`` path, where inadmissible names **consume neighbour
slots and are then discarded**, so a node configured for K neighbours trains on
fewer, silently, and ``rank_pct`` is ranked against a contaminated candidate set.

These tests therefore assert on *selection*, which is the only place the defect
can be repaired:

* an admissible node's out-degree equals ``min(top_k, n_admissible - 1)``
* no edge is selected into an inadmissible name
* the threshold path is unchanged, because it has no budget to starve
"""

import numpy as np
import pandas as pd
import pytest

from mci_gru.graph.builder import GraphBuilder

KDCODES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
# Admissible on the snapshot date: AAA, BBB, CCC, DDD. Absent: EEE, FFF.
ADMISSIBLE = ["AAA", "BBB", "CCC", "DDD"]
N_SESSIONS = 300
JUDGE_VALUE = 0.8


# Idiosyncratic noise per name. The two inadmissible names carry the *least*
# noise, so they track the common factor most closely and are therefore the
# most attractive top-K candidates for everyone. That is what makes them win
# slots in an unmasked build and demonstrates the budget loss; a fixture where
# the admissible names were the strongest candidates would lose no budget and
# would not exercise the defect at all.
NOISE_SCALE = {
    "AAA": 0.0040,
    "BBB": 0.0045,
    "CCC": 0.0050,
    "DDD": 0.0055,
    "EEE": 0.0002,
    "FFF": 0.0003,
}


def _correlated_panel() -> pd.DataFrame:
    """Every name strongly correlated, so selection is budget-bound, not signal-bound.

    With all pairs above threshold, any shortfall in a node's out-degree is
    attributable to slots lost to inadmissible names rather than to weak
    correlation.
    """
    rng = np.random.default_rng(20260730)
    dates = pd.bdate_range("2023-01-02", periods=N_SESSIONS)
    common = rng.normal(0.0, 0.01, N_SESSIONS)

    rows = []
    for kdcode in KDCODES:
        rets = common + rng.normal(0.0, NOISE_SCALE[kdcode], N_SESSIONS)
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


def _admissible_mask() -> np.ndarray:
    return np.array([name in ADMISSIBLE for name in KDCODES], dtype=bool)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _correlated_panel()


@pytest.fixture(scope="module")
def end_date(panel: pd.DataFrame) -> str:
    return str(panel["dt"].max())


def _out_degrees(edge_index) -> dict[str, int]:
    counts = dict.fromkeys(KDCODES, 0)
    for node in edge_index[0].tolist():
        counts[KDCODES[node]] += 1
    return counts


def _destinations(edge_index) -> set[str]:
    return {KDCODES[node] for node in edge_index[1].tolist()}


def test_fixture_actually_exhibits_the_defect(panel, end_date):
    """Guard the guard: without a mask, admissible nodes must lose slots.

    If the inadmissible names were not attractive candidates they would never
    win a slot, no budget would be lost, and every assertion below would pass
    against a build that has no defect to fix.
    """
    builder = GraphBuilder(top_k=3, top_k_metric="corr", corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    wasted = sum(
        1
        for a, b in zip(src, dst, strict=True)
        if KDCODES[a] in ADMISSIBLE and KDCODES[b] not in ADMISSIBLE
    )
    assert wasted > 0, (
        "no admissible node selected an inadmissible neighbour, so this fixture "
        "does not exercise the budget-starvation defect at all"
    )


def test_top_k_budget_is_spent_only_on_admissible_neighbours(panel, end_date):
    """The #123 defect: slots lost to names the collate would discard.

    Without a mask, a node's K neighbours are drawn from all six names, so up
    to two of every four slots point at EEE or FFF and are dropped downstream.
    With the mask, every slot goes to a name that survives.
    """
    top_k = 3
    builder = GraphBuilder(top_k=top_k, top_k_metric="corr", corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(
        panel, KDCODES, end_date, show_progress=False, admissible_mask=_admissible_mask()
    )

    degrees = _out_degrees(edge_index)
    expected = min(top_k, len(ADMISSIBLE) - 1)

    for name in ADMISSIBLE:
        assert degrees[name] == expected, (
            f"{name} has out-degree {degrees[name]}, expected "
            f"min(top_k={top_k}, n_admissible-1={len(ADMISSIBLE) - 1}) = {expected}. "
            f"A shortfall means budget was spent on names excluded from selection."
        )


def test_no_edge_is_selected_into_an_inadmissible_name(panel, end_date):
    builder = GraphBuilder(top_k=3, top_k_metric="corr", corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(
        panel, KDCODES, end_date, show_progress=False, admissible_mask=_admissible_mask()
    )

    leaked = _destinations(edge_index) - set(ADMISSIBLE)
    assert not leaked, f"edges point at names outside the universe: {sorted(leaked)}"


def test_inadmissible_nodes_select_nothing(panel, end_date):
    builder = GraphBuilder(top_k=3, top_k_metric="corr", corr_lookback_days=N_SESSIONS)
    edge_index, _ = builder.build_graph(
        panel, KDCODES, end_date, show_progress=False, admissible_mask=_admissible_mask()
    )

    degrees = _out_degrees(edge_index)
    for name in KDCODES:
        if name not in ADMISSIBLE:
            assert degrees[name] == 0, (
                f"{name} is not in the universe on this date but has out-degree {degrees[name]}"
            )


def test_masking_raises_admissible_out_degree_versus_no_mask(panel, end_date):
    """Direct evidence for the brief's headline: the fix *adds* edges.

    This is the property that distinguishes filtering before selection from
    filtering after it. Filtering after cannot raise a node's degree.
    """
    top_k = 3
    builder = GraphBuilder(top_k=top_k, top_k_metric="corr", corr_lookback_days=N_SESSIONS)

    unmasked, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)
    masked, _ = builder.build_graph(
        panel, KDCODES, end_date, show_progress=False, admissible_mask=_admissible_mask()
    )

    unmasked_degrees = _out_degrees(unmasked)
    masked_degrees = _out_degrees(masked)

    # Count only slots that would survive a downstream PIT filter.
    def surviving(edge_index) -> dict[str, int]:
        counts = dict.fromkeys(KDCODES, 0)
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for a, b in zip(src, dst, strict=True):
            if KDCODES[a] in ADMISSIBLE and KDCODES[b] in ADMISSIBLE:
                counts[KDCODES[a]] += 1
        return counts

    survive_unmasked = surviving(unmasked)
    for name in ADMISSIBLE:
        assert masked_degrees[name] > survive_unmasked[name], (
            f"{name}: masked selection kept {masked_degrees[name]} usable edges "
            f"but unmasked selection kept only {survive_unmasked[name]} after a "
            f"downstream filter. Filtering before selection must recover budget. "
            f"(unmasked raw out-degree was {unmasked_degrees[name]})"
        )


def test_threshold_path_is_unchanged_by_masking_except_for_excluded_names(panel, end_date):
    """The threshold path has no budget, so masking removes edges and adds none.

    The brief measured this as bit-identical once the downstream filter is
    applied. Asserting the weaker, directly observable property: every edge the
    masked build produces is one the unmasked build also produced.
    """
    builder = GraphBuilder(judge_value=JUDGE_VALUE, top_k=0, corr_lookback_days=N_SESSIONS)

    unmasked, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)
    masked, _ = builder.build_graph(
        panel, KDCODES, end_date, show_progress=False, admissible_mask=_admissible_mask()
    )

    def pairs(edge_index) -> set[tuple[str, str]]:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        return {(KDCODES[a], KDCODES[b]) for a, b in zip(src, dst, strict=True)}

    unmasked_pairs = pairs(unmasked)
    masked_pairs = pairs(masked)

    assert masked_pairs <= unmasked_pairs, (
        "masking introduced edges the unmasked threshold build did not select: "
        f"{sorted(masked_pairs - unmasked_pairs)}"
    )
    both_admissible = {p for p in unmasked_pairs if p[0] in ADMISSIBLE and p[1] in ADMISSIBLE}
    assert masked_pairs == both_admissible, (
        "on the threshold path the masked edge set should be exactly the "
        "admissible-to-admissible subset of the unmasked set"
    )


def test_no_mask_preserves_existing_behaviour(panel, end_date):
    """The parameter is optional and defaults to today's behaviour."""
    builder = GraphBuilder(top_k=3, top_k_metric="corr", corr_lookback_days=N_SESSIONS)

    without_arg, _ = builder.build_graph(panel, KDCODES, end_date, show_progress=False)
    explicit_none, _ = builder.build_graph(
        panel, KDCODES, end_date, show_progress=False, admissible_mask=None
    )

    assert without_arg.shape == explicit_none.shape
    assert (without_arg == explicit_none).all()
    # Every node still selects K, drawn from the full axis.
    for name, degree in _out_degrees(without_arg).items():
        assert degree == 3, f"{name} out-degree {degree} without a mask, expected top_k=3"
