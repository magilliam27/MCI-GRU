import numpy as np

from mci_gru.evaluation.statistics import (
    daily_ic_series,
    moving_block_bootstrap_ci,
    newey_west_sharpe,
)


def test_daily_ic_series_computes_per_day_correlations():
    predictions = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    returns = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])

    pearson = daily_ic_series(predictions, returns, method="pearson")
    spearman = daily_ic_series(predictions, returns, method="spearman")

    assert pearson.shape == (2,)
    assert pearson[0] > 0.98
    assert pearson[1] < -0.98
    np.testing.assert_allclose(spearman, np.array([1.0, -1.0]))


def test_newey_west_sharpe_differs_from_naive_on_autocorrelated_returns():
    returns = np.array([0.01, 0.012, -0.002, -0.004, 0.011, 0.013, -0.003, -0.005])
    naive = newey_west_sharpe(returns, lags=0)
    corrected = newey_west_sharpe(returns, lags=2)

    assert np.isfinite(corrected)
    assert corrected != naive


def test_moving_block_bootstrap_ci_is_deterministic_and_contains_mean():
    values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    first = moving_block_bootstrap_ci(
        values,
        statistic=np.mean,
        block_size=2,
        n_resamples=200,
        seed=123,
        ci_level=0.95,
    )
    second = moving_block_bootstrap_ci(
        values,
        statistic=np.mean,
        block_size=2,
        n_resamples=200,
        seed=123,
        ci_level=0.95,
    )

    assert first == second
    assert first["lower"] <= np.mean(values) <= first["upper"]
