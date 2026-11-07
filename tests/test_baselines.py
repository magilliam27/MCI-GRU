"""Simple baseline signals read only what was known at the signal date's close (#360).

``mci_gru.evaluation.baselines`` builds the reversal and 12-1 momentum scores from
past closes and fits the linear baseline's ridge. These tests pin that:

* each past-return signal is the stated ratio of two closes, counted in sessions on
  the axis the training label counts on, so a missing bar never moves an endpoint;
* no close after the signal date can change that date's score;
* the ridge reads only the training rows its mask allows, recovers a planted linear
  signal, ignores an input that holds one value per date for every stock, and keeps
  the penalty with the best validation IC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.baselines import (
    fit_cross_sectional_ridge,
    linear_scores,
    momentum_scores,
    past_return,
    reversal_scores,
)

STOCKS = ["A.N", "B.N", "C.N"]


def _panel(n_sessions: int = 40, seed: int = 0) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=n_sessions).strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    rows = []
    for j, kdcode in enumerate(STOCKS):
        closes = 50.0 * (j + 1) * np.cumprod(1.0 + rng.normal(0.0, 0.02, n_sessions))
        rows += [
            {"kdcode": kdcode, "dt": d, "close": c} for d, c in zip(dates, closes, strict=True)
        ]
    return pd.DataFrame(rows)


def _close(panel: pd.DataFrame, kdcode: str, date: str) -> float:
    return float(panel.loc[(panel.kdcode == kdcode) & (panel.dt == date), "close"].iloc[0])


# --- Past-return signals --------------------------------------------------------------


def test_past_return_is_the_ratio_of_the_two_closes_counted_in_sessions():
    panel = _panel()
    sessions = sorted(panel.dt.unique())
    dates = sessions[25:30]

    got = past_return(panel, STOCKS, dates, start_lag=21, end_lag=2)

    for i, date in enumerate(dates):
        at = sessions.index(date)
        for j, kdcode in enumerate(STOCKS):
            expected = (
                _close(panel, kdcode, sessions[at - 2]) / _close(panel, kdcode, sessions[at - 21])
                - 1.0
            )
            assert got[i, j] == pytest.approx(expected, rel=1e-12)


def test_reversal_is_minus_the_recent_return_and_momentum_skips_the_last_month():
    panel = _panel(n_sessions=300)
    sessions = sorted(panel.dt.unique())
    dates = sessions[260:263]

    np.testing.assert_array_equal(
        reversal_scores(panel, STOCKS, dates, lookback=5),
        -past_return(panel, STOCKS, dates, start_lag=5),
    )
    momentum = momentum_scores(panel, STOCKS, dates)
    at = sessions.index(dates[0])
    expected = (
        _close(panel, "B.N", sessions[at - 21]) / _close(panel, "B.N", sessions[at - 252]) - 1
    )
    assert momentum[0, 1] == pytest.approx(expected, rel=1e-12)


def test_a_missing_bar_empties_that_score_and_never_moves_an_endpoint():
    panel = _panel()
    sessions = sorted(panel.dt.unique())
    date = sessions[30]
    gap = sessions[25]  # B.N's close five sessions back is missing
    holed = panel[~((panel.kdcode == "B.N") & (panel.dt == gap))]

    full = past_return(panel, STOCKS, [date], start_lag=5)
    got = past_return(holed, STOCKS, [date], start_lag=5)

    assert np.isnan(got[0, 1])
    np.testing.assert_array_equal(got[0, [0, 2]], full[0, [0, 2]])
    # A row shift would have read B.N's close six sessions back instead.
    next_date = sessions[31]
    assert past_return(holed, STOCKS, [next_date], start_lag=5)[0, 1] == pytest.approx(
        _close(panel, "B.N", next_date) / _close(panel, "B.N", sessions[26]) - 1.0, rel=1e-12
    )


def test_no_close_after_the_signal_date_changes_its_score():
    panel = _panel(n_sessions=60)
    sessions = sorted(panel.dt.unique())
    dates = sessions[25:50]
    before = {
        "reversal": reversal_scores(panel, STOCKS, dates, lookback=5),
        "past": past_return(panel, STOCKS, dates, start_lag=21, end_lag=3),
    }
    for i, date in enumerate(dates):
        future = panel.copy()
        later = future.dt > date
        future.loc[later, "close"] = future.loc[later, "close"] * 3.0
        future = future.drop(future[later & (future.kdcode == "A.N")].index[:2])
        assert np.array_equal(
            reversal_scores(future, STOCKS, [date], lookback=5)[0],
            before["reversal"][i],
            equal_nan=True,
        ), date
        assert np.array_equal(
            past_return(future, STOCKS, [date], start_lag=21, end_lag=3)[0],
            before["past"][i],
            equal_nan=True,
        ), date


def test_a_date_without_enough_history_or_off_the_axis_has_no_score():
    panel = _panel()
    sessions = sorted(panel.dt.unique())

    got = past_return(panel, STOCKS, [sessions[4], "2021-01-02"], start_lag=5)

    assert np.isnan(got).all()


def test_past_return_refuses_lags_that_do_not_look_back():
    with pytest.raises(ValueError, match="end_lag"):
        past_return(_panel(), STOCKS, [], start_lag=5, end_lag=5)


# --- Cross-sectional ridge ------------------------------------------------------------


def _planted(n_dates: int, n_stocks: int, seed: int, beta: np.ndarray, noise: float = 0.01):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_dates, n_stocks, 2, beta.size // 2)).astype(np.float32)
    y = linear_scores(x, beta) + rng.normal(0.0, noise, (n_dates, n_stocks))
    return x, y


BETA = np.array([0.5, -1.0, 0.0, 0.25])


def test_the_ridge_recovers_a_planted_linear_signal_and_scores_with_it():
    x, y = _planted(80, 30, seed=1, beta=BETA)
    xv, yv = _planted(20, 30, seed=2, beta=BETA)

    fit = fit_cross_sectional_ridge(x, y, xv, yv, alphas=(0.0,))

    np.testing.assert_allclose(fit.coefficients, BETA, atol=0.01)
    assert fit.validation_ic[0.0] > 0.99
    assert fit.train_dates == 80 and fit.train_rows == 80 * 30


def test_the_ridge_reads_only_the_training_rows_its_mask_allows():
    x, y = _planted(40, 20, seed=3, beta=BETA, noise=0.5)
    xv, yv = _planted(10, 20, seed=4, beta=BETA)
    mask = np.ones(y.shape, dtype=bool)
    mask[:, :5] = False
    baseline = fit_cross_sectional_ridge(x, y, xv, yv, train_mask=mask, alphas=(0.1,))

    masked_out = y.copy()
    masked_out[:, :5] = 1e6
    unlabelled = y.copy()
    unlabelled[:, 5] = np.nan
    allowed = y.copy()
    allowed[:, 6] += np.arange(40)

    same = fit_cross_sectional_ridge(x, masked_out, xv, yv, train_mask=mask, alphas=(0.1,))
    np.testing.assert_array_equal(same.coefficients, baseline.coefficients)
    fewer = fit_cross_sectional_ridge(x, unlabelled, xv, yv, train_mask=mask, alphas=(0.1,))
    assert fewer.train_rows == baseline.train_rows - 40
    moved = fit_cross_sectional_ridge(x, allowed, xv, yv, train_mask=mask, alphas=(0.1,))
    assert not np.allclose(moved.coefficients, baseline.coefficients)


def test_an_input_that_is_the_same_for_every_stock_on_a_date_is_ignored():
    """Global inputs (regime, VIX, credit) cannot rank stocks; demeaning removes them."""
    x, y = _planted(60, 25, seed=5, beta=BETA, noise=0.3)
    xv, yv = _planted(15, 25, seed=6, beta=BETA)
    rng = np.random.default_rng(7)
    per_date = rng.normal(size=(60, 1, 2, 1)) * np.ones((1, 25, 2, 1))
    per_date_val = rng.normal(size=(15, 1, 2, 1)) * np.ones((1, 25, 2, 1))
    with_global = np.concatenate([x, per_date], axis=3)
    with_global_val = np.concatenate([xv, per_date_val], axis=3)
    # The label moves with the global input too: the fit must not care.
    shifted = y + 5.0 * per_date[:, :, 0, 0]

    plain = fit_cross_sectional_ridge(x, y, xv, yv, alphas=(1.0,))
    both = fit_cross_sectional_ridge(with_global, shifted, with_global_val, yv, alphas=(1.0,))

    stock_level = both.coefficients.reshape(2, 3)[:, :2].ravel()
    np.testing.assert_allclose(stock_level, plain.coefficients, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(both.coefficients.reshape(2, 3)[:, 2], 0.0, atol=1e-9)


def test_the_penalty_with_the_best_validation_ic_is_kept_and_ties_go_to_the_larger():
    x, y = _planted(30, 15, seed=8, beta=BETA, noise=2.0)
    xv, yv = _planted(10, 15, seed=9, beta=BETA, noise=0.0)

    fit = fit_cross_sectional_ridge(x, y, xv, yv)
    best = max(fit.validation_ic.values())
    assert fit.validation_ic[fit.alpha] == best
    assert fit.alpha == max(a for a, ic in fit.validation_ic.items() if ic == best)

    # One input: every penalty only rescales the scores, so every IC ties.
    rng = np.random.default_rng(12)
    one, one_val = rng.normal(size=(30, 15, 1, 1)), rng.normal(size=(10, 15, 1, 1))
    tied = fit_cross_sectional_ridge(
        one, one[..., 0, 0] + rng.normal(size=(30, 15)), one_val, one_val[..., 0, 0]
    )
    assert tied.alpha == max(tied.validation_ic)

    # A validation set the scores cannot rank (one stock per date) gives no IC at all.
    with pytest.raises(ValueError, match="finite validation IC"):
        fit_cross_sectional_ridge(x, y, xv[:, :1], yv[:, :1])


def test_validation_ic_reads_only_scores_the_validation_mask_allows():
    x, y = _planted(30, 15, seed=10, beta=BETA)
    xv, yv = _planted(10, 15, seed=11, beta=BETA)
    poisoned = yv.copy()
    poisoned[:, :5] = -100.0 * linear_scores(xv, BETA)[:, :5]
    mask = np.ones(yv.shape, dtype=bool)
    mask[:, :5] = False

    clean = fit_cross_sectional_ridge(x, y, xv, yv, val_mask=mask, alphas=(0.0,))
    masked = fit_cross_sectional_ridge(x, y, xv, poisoned, val_mask=mask, alphas=(0.0,))

    assert masked.validation_ic[0.0] == clean.validation_ic[0.0]
