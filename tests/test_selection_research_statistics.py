import numpy as np
import pytest

from mci_gru.evaluation.selection_nulls import (
    DatedScoreOutcome,
    permute_scores_within_date,
    run_within_date_permutation_null,
)
from mci_gru.evaluation.statistics import (
    dated_daily_ic,
    empirical_one_sided_p_value,
    moving_block_mean_ci,
    newey_west_mean_inference,
    newey_west_std,
)


def test_daily_spearman_is_dated_and_preserves_invalid_statuses() -> None:
    observations = dated_daily_ic(
        dates=("2026-01-02", "2026-01-05", "2026-01-06"),
        predictions=(
            np.array([1.0, 2.0, 2.0, 4.0]),
            np.array([4.0, 3.0, 2.0, 1.0]),
            np.array([1.0, 2.0, np.nan, 4.0]),
        ),
        true_returns=(
            np.array([0.01, 0.02, 0.02, 0.04]),
            np.array([0.04, 0.03, 0.02, 0.01]),
            np.array([0.01, 0.02, 0.03, 0.04]),
        ),
        statuses=("VALID_PRIMARY", "MISSING_EXPECTED_SCORE", "VALID_PRIMARY"),
        method="spearman",
    )

    assert [item.signal_dt for item in observations] == [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
    ]
    assert observations[0].daily_ic == 1.0
    assert observations[0].status == "VALID_PRIMARY"
    assert observations[1].daily_ic is None
    assert observations[1].status == "MISSING_EXPECTED_SCORE"
    assert observations[2].daily_ic is None
    assert observations[2].status == "INVALID_IC_NONFINITE_VALUES"


def test_overlap_aware_hac_lag_is_enforced_for_mean_inference() -> None:
    values = np.array([0.10, 0.05, -0.02, 0.03, 0.08, -0.01])

    with pytest.raises(ValueError, match="at least label_horizon - 1"):
        newey_west_mean_inference(values, lags=3, label_horizon=5)

    result = newey_west_mean_inference(values, lags=4, label_horizon=5)

    assert result.method == "newey_west_bartlett_v1"
    assert result.n_obs == len(values)
    assert result.mean == pytest.approx(np.mean(values))
    assert result.standard_error == pytest.approx(
        newey_west_std(values, lags=4) / np.sqrt(len(values))
    )
    assert result.t_stat is not None
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0


def test_overlap_aware_moving_block_mean_interval_is_enforced_and_deterministic() -> None:
    values = np.array([0.10, 0.05, -0.02, 0.03, 0.08, -0.01])

    with pytest.raises(ValueError, match="at least label_horizon"):
        moving_block_mean_ci(
            values,
            block_size=4,
            label_horizon=5,
            n_resamples=100,
            seed=17,
            ci_level=0.95,
        )

    first = moving_block_mean_ci(
        values,
        block_size=5,
        label_horizon=5,
        n_resamples=100,
        seed=17,
        ci_level=0.95,
    )
    second = moving_block_mean_ci(
        values,
        block_size=5,
        label_horizon=5,
        n_resamples=100,
        seed=17,
        ci_level=0.95,
    )

    assert first == second
    assert first.method == "circular_moving_block_percentile_v1"
    assert first.n_obs == len(values)
    assert first.block_size == 5
    assert first.lower is not None
    assert first.upper is not None
    assert first.lower <= np.mean(values) <= first.upper


def test_empirical_one_sided_p_value_uses_plus_one_correction() -> None:
    null_statistics = np.array([0.0, 0.1, 0.2, 0.3, np.nan])

    p_value = empirical_one_sided_p_value(0.2, null_statistics)

    assert p_value == pytest.approx((1 + 2) / (1 + 4))


def test_permutation_preserves_daily_set_score_multiset_and_ties() -> None:
    keys = ("CCC", "AAA", "DDD", "BBB")
    scores = (3.0, 1.0, 2.0, 2.0)

    assignment = permute_scores_within_date(
        instrument_keys=keys,
        scores=scores,
        signal_dt="2026-01-02",
        null_seed=73,
        draw_id=11,
    )

    assert assignment.instrument_keys == keys
    assert assignment.permuted_scores == (2.0, 1.0, 2.0, 3.0)
    assert assignment.source_instrument_keys == ("DDD", "AAA", "BBB", "CCC")
    assert (
        assignment.assignment_digest
        == "8a5ed1a24b58025fcd75a1d73a69ca1c1686df0b499e1ca76f953d52326542ae"
    )
    assert sorted(assignment.permuted_scores) == sorted(scores)
    assert assignment.permuted_scores.count(2.0) == 2
    assert sorted(assignment.source_instrument_keys) == sorted(keys)
    assert len(assignment.assignment_digest) == 64


def test_permutation_is_deterministic_and_assignment_is_outcome_independent() -> None:
    dates = (
        DatedScoreOutcome(
            signal_dt="2026-01-02",
            instrument_keys=("AAA", "BBB", "CCC", "DDD"),
            scores=(4.0, 3.0, 2.0, 1.0),
            outcomes=(0.04, 0.03, 0.02, 0.01),
        ),
        DatedScoreOutcome(
            signal_dt="2026-01-05",
            instrument_keys=("AAA", "BBB", "CCC", "DDD"),
            scores=(1.0, 2.0, 2.0, 4.0),
            outcomes=(-0.01, 0.02, 0.01, 0.05),
        ),
    )
    changed_outcomes = tuple(
        DatedScoreOutcome(
            signal_dt=item.signal_dt,
            instrument_keys=item.instrument_keys,
            scores=item.scores,
            outcomes=tuple(-value for value in item.outcomes),
        )
        for item in dates
    )

    with pytest.raises(ValueError, match="at least 1000"):
        run_within_date_permutation_null(dates, n_draws=999, null_seed=73, top_k=2)

    first = run_within_date_permutation_null(dates, n_draws=1000, null_seed=73, top_k=2)
    second = run_within_date_permutation_null(
        tuple(reversed(dates)), n_draws=1000, null_seed=73, top_k=2
    )
    changed = run_within_date_permutation_null(
        changed_outcomes, n_draws=1000, null_seed=73, top_k=2
    )

    assert first == second
    assert first.method == "WITHIN_DATE_SCORE_PERMUTATION_V1"
    assert first.valid_draw_count == 1000
    assert len(first.rank_ic_means) == 1000
    assert len(first.top_k_spread_means) == 1000
    assert first.assignment_digest_method == "SHA256_ASSIGNMENT_INDEX_STREAM_V1"
    assert (
        first.assignment_digest
        == "a12d6a6444bcb30deea8d9a5aa9119861afe31442f837383483d300dad5cbd1b"
    )
    assert first.assignment_digest == changed.assignment_digest
    assert first.rank_ic_means != changed.rank_ic_means
