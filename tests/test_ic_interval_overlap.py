"""IC confidence intervals reach nominal coverage on overlapping labels (#356).

The daily IC of a close ``D+1`` to close ``D+label_t`` label is serially
dependent: consecutive dates share ``label_t - 2`` sessions of outcome, so the
series is an MA(``label_t - 2``). The in-run IC CIs and the paired-difference
CIs used a circular block bootstrap with ``block_size = label_t``, which weights
those autocovariances down like a short Bartlett kernel; on 238 dates the
#343 thread measured 88.6% coverage for a nominal 95%. Both now use one
interval: the mean, a uniform-weight (Hansen-Hodrick) standard error over the
overlap lags, and a Student-t critical value on the equivalent degrees of
freedom of that estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mci_gru.evaluation.metrics import compute_ic_metrics, ic_confidence_interval
from mci_gru.evaluation.paired_inference import paired_mean_inference
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    hansen_hodrick_mean_ci,
    moving_block_bootstrap_ci,
    overlap_hac_lags,
)

#: The first run's test span: 238 scored dates (#187).
FIRST_RUN_DATES = 238
LABEL_T = 5
REPS = 2000
#: Monte Carlo standard error of a 95% coverage over REPS draws is about 0.5%.
COVERAGE_TOLERANCE = 0.015


def _overlapping_ic_draws(reps: int, n: int, seed: int) -> np.ndarray:
    """Zero-mean MA(3) series: each date sums the 4 daily shocks its label spans."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((reps, n + 3))
    return shocks[:, 3:] + shocks[:, 2:-1] + shocks[:, 1:-2] + shocks[:, :-3]


def _coverage(draws: np.ndarray, interval) -> float:
    hits = 0
    for values in draws:
        lower, upper = interval(values)
        hits += bool(lower <= 0.0 <= upper)
    return hits / len(draws)


def test_label_overlap_is_label_t_minus_two_lags():
    # close D+1 to close D+5 spans 4 sessions; neighbours share 3 of them.
    assert overlap_hac_lags(5) == 3
    assert overlap_hac_lags(2) == 0
    assert overlap_hac_lags(1) == 0


def test_ic_interval_reaches_nominal_coverage_on_overlapping_labels():
    """The in-run IC CI covers the true mean about 95% of the time on the first run's length."""
    draws = _overlapping_ic_draws(REPS, FIRST_RUN_DATES, seed=356)

    def interval(values):
        ci = ic_confidence_interval(values, label_t=LABEL_T, ci_level=0.95)
        return ci["lower"], ci["upper"]

    coverage = _coverage(draws, interval)
    assert abs(coverage - 0.95) <= COVERAGE_TOLERANCE, coverage


def test_paired_interval_reaches_nominal_coverage_on_overlapping_labels():
    """The paired-difference CI uses the same interval, so it covers at the same rate."""
    draws = _overlapping_ic_draws(REPS, FIRST_RUN_DATES, seed=357)

    def interval(values):
        result = paired_mean_inference(
            values, arm="a", control="b", label_horizon=LABEL_T, ci_level=0.95
        )
        return result.ci_lower, result.ci_upper

    coverage = _coverage(draws, interval)
    assert abs(coverage - 0.95) <= COVERAGE_TOLERANCE, coverage


def test_the_block_bootstrap_at_label_t_undercovers():
    """Control: the scheme #356 replaces fails the coverage band, so the band has teeth.

    Fewer draws and resamples than the tests above keep the repository's
    bootstrap affordable; its shortfall (about 89%) is far outside the band.
    """
    draws = _overlapping_ic_draws(250, FIRST_RUN_DATES, seed=358)

    def interval(values):
        ci = moving_block_bootstrap_ci(values, np.mean, LABEL_T, 100, 1729, 0.95)
        return ci["lower"], ci["upper"]

    assert _coverage(draws, interval) < 0.95 - COVERAGE_TOLERANCE


def test_without_overlap_the_interval_is_a_t_interval_on_n_minus_one_df():
    rng = np.random.default_rng(4)
    values = rng.normal(0.01, 0.05, size=60)
    ci = ic_confidence_interval(values, label_t=1, ci_level=0.9)
    expected = stats.t.interval(
        0.9, df=values.size - 1, loc=values.mean(), scale=values.std(ddof=0) / np.sqrt(60)
    )
    assert (ci["lower"], ci["upper"]) == pytest.approx(expected, rel=1e-12)


def test_the_interval_records_its_method_lags_and_degrees_of_freedom():
    values = _overlapping_ic_draws(1, FIRST_RUN_DATES, seed=1)[0]
    interval = hansen_hodrick_mean_ci(values, lags=3, ci_level=0.95)
    assert interval.method == "hansen_hodrick_t_v1"
    assert interval.n_obs == FIRST_RUN_DATES
    assert interval.lags == 3
    # Equivalent degrees of freedom of a truncated-kernel long-run variance at frequency
    # zero, n / (2L + 1), less one for the estimated mean.
    assert interval.df == pytest.approx(FIRST_RUN_DATES / 7 - 1)
    assert interval.lower < float(np.mean(values)) < interval.upper


def test_a_non_positive_long_run_variance_gives_no_interval():
    alternating = np.tile([1.0, -1.0], 20)  # lag-1 and lag-3 covariances outweigh the rest
    interval = hansen_hodrick_mean_ci(alternating, lags=3, ci_level=0.95)
    assert interval.lower is None and interval.upper is None
    ci = ic_confidence_interval(alternating, label_t=5, ci_level=0.95)
    assert np.isnan(ci["lower"]) and np.isnan(ci["upper"])


def test_compute_ic_metrics_reports_the_overlap_interval_for_ic_and_rank_ic():
    rng = np.random.default_rng(15)
    labels = rng.normal(size=(80, 12))
    scores = labels * 0.2 + rng.normal(size=(80, 12))
    metrics = compute_ic_metrics(scores, labels, label_t=LABEL_T, ci_enabled=True, ci_level=0.9)
    for prefix, method in (("avg_ic", "pearson"), ("avg_rank_ic", "spearman")):
        ci = ic_confidence_interval(
            daily_ic_series(scores, labels, method), label_t=LABEL_T, ci_level=0.9
        )
        assert metrics[f"{prefix}_ci_lower"] == ci["lower"]
        assert metrics[f"{prefix}_ci_upper"] == ci["upper"]


def test_compute_ic_metrics_omits_intervals_unless_asked():
    rng = np.random.default_rng(16)
    labels = rng.normal(size=(30, 8))
    metrics = compute_ic_metrics(rng.normal(size=(30, 8)), labels, label_t=LABEL_T)
    assert not [key for key in metrics if "_ci_" in key]


def test_paired_inference_defaults_to_the_overlap_interval():
    rng = np.random.default_rng(3)
    delta = pd.Series(rng.normal(0.003, 0.03, size=120))
    result = paired_mean_inference(
        delta, arm="a", control="b", label_horizon=LABEL_T, ci_level=0.95
    )
    interval = hansen_hodrick_mean_ci(delta.to_numpy(), lags=3, ci_level=0.95)
    assert result.ci_method == "hansen_hodrick_t_v1"
    assert result.block_size is None
    assert (result.ci_lower, result.ci_upper) == (interval.lower, interval.upper)


def test_paired_inference_keeps_an_explicitly_requested_block_bootstrap():
    """The archived graph notebooks pass ``block_size``; they reproduce their own records."""
    rng = np.random.default_rng(3)
    delta = rng.normal(0.003, 0.03, size=120)
    result = paired_mean_inference(
        delta,
        arm="a",
        control="b",
        label_horizon=LABEL_T,
        block_size=5,
        n_resamples=200,
        seed=1729,
        ci_level=0.95,
    )
    expected = moving_block_bootstrap_ci(delta, np.mean, 5, 200, 1729, 0.95)
    assert result.ci_method == "circular_moving_block_percentile_v1"
    assert result.block_size == 5
    assert (result.ci_lower, result.ci_upper) == pytest.approx(
        (expected["lower"], expected["upper"]), abs=1e-15
    )


def test_a_block_bootstrap_needs_its_resamples_and_seed():
    with pytest.raises(ValueError, match="n_resamples and seed"):
        paired_mean_inference(
            np.zeros(10) + 0.01,
            arm="a",
            control="b",
            label_horizon=LABEL_T,
            ci_level=0.95,
            block_size=5,
        )
