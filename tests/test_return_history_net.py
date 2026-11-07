"""The return-history net reads only past closes and learns a planted signal (#385).

``mci_gru.evaluation.return_history_net`` is a candidate arm for the #384 harness: a
small network fed only each stock's cumulative return path. These tests pin that:

* every input is exactly ``past_return`` from #360 for its lag, so the arm counts
  sessions on the same axis as the labels and the baselines;
* no close after the signal date can change that date's inputs or score;
* a stock without a close on the signal date is unscored, and a missing past close is
  a neutral input rather than a dropped stock;
* ranking and target standardisation behave as documented;
* trained on a panel whose labels follow one-week reversal, the net ranks the test
  dates in the right direction, and the same seed gives the same scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation.baselines import past_return
from mci_gru.evaluation.candidate_registry import CANDIDATES
from mci_gru.evaluation.candidates import CandidateContext, check_no_lookahead, check_result
from mci_gru.evaluation.return_history_net import (
    DAILY_LAGS,
    INPUT_LAGS,
    MONTHLY_LAGS,
    NetSpec,
    _score_candidate,
    fit_predict,
    rank_within_date,
    return_history_inputs,
    return_net_cnn_lstm,
    return_net_fnn,
    standardise_within_date,
)
from mci_gru.evaluation.statistics import daily_ic_series

STOCKS = [f"S{i:02d}.N" for i in range(30)]


def _panel(n_sessions: int = 320, n_stocks: int = 30, seed: int = 0) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", periods=n_sessions).strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    rows = []
    for j, kdcode in enumerate(STOCKS[:n_stocks]):
        closes = 40.0 * (1 + j / 10) * np.cumprod(1.0 + rng.normal(0.0, 0.02, n_sessions))
        rows += [
            {"kdcode": kdcode, "dt": d, "close": c, "volume": 1e6}
            for d, c in zip(dates, closes, strict=True)
        ]
    return pd.DataFrame(rows)


def test_input_lags_are_the_daily_path_then_twelve_months():
    assert tuple(range(1, 21)) == DAILY_LAGS
    assert tuple(21 * k for k in range(1, 13)) == MONTHLY_LAGS
    assert INPUT_LAGS == DAILY_LAGS + MONTHLY_LAGS


def test_every_input_is_past_return_for_its_lag():
    panel = _panel()
    sessions = sorted(panel.dt.unique())
    dates = sessions[250:256] + sessions[5:7]  # the last two run off the axis for long lags

    raw, scoreable = return_history_inputs(panel, STOCKS, dates)

    assert raw.shape == (len(dates), len(STOCKS), len(INPUT_LAGS))
    assert scoreable.all()
    for k, lag in enumerate(INPUT_LAGS):
        np.testing.assert_array_equal(
            raw[:, :, k], past_return(panel, STOCKS, dates, start_lag=lag)
        )
    assert np.isnan(raw[-1, :, -1]).all()  # 252 sessions before session 6 do not exist


def test_no_close_after_the_signal_date_changes_its_inputs():
    panel = _panel()
    sessions = sorted(panel.dt.unique())
    date = sessions[270]
    before, _ = return_history_inputs(panel, STOCKS, [date])

    moved = panel.copy()
    later = moved.dt > date
    moved.loc[later, "close"] *= np.random.default_rng(1).uniform(0.5, 2.0, int(later.sum()))
    after, _ = return_history_inputs(moved, STOCKS, [date])

    np.testing.assert_array_equal(before, after)


def test_a_stock_without_a_close_on_the_date_is_unscored_and_a_missing_past_close_is_neutral():
    panel = _panel()
    sessions = sorted(panel.dt.unique())
    date = sessions[280]
    panel = panel[~((panel.kdcode == STOCKS[0]) & (panel.dt == date))]
    panel = panel[~((panel.kdcode == STOCKS[1]) & (panel.dt == sessions[275]))]

    raw, scoreable = return_history_inputs(panel, STOCKS, [date])
    ranked = rank_within_date(raw)

    assert not scoreable[0, 0] and scoreable[0, 1:].all()
    assert np.isnan(raw[0, 1, 4])  # close D-5 missing for the second stock
    assert ranked[0, 1, 4] == 0.0
    assert np.isfinite(raw[0, 1, 3])


def test_ranks_span_minus_one_to_one_with_ties_averaged():
    raw = np.array([[[3.0], [1.0], [np.nan], [2.0], [2.0]]]).reshape(1, 5, 1)

    ranked = rank_within_date(raw)[0, :, 0]

    np.testing.assert_allclose(ranked, [1.0, -1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_target_is_standardised_within_each_date_over_the_mask_only():
    labels = np.array([[1.0, 2.0, 3.0, 100.0], [np.nan, 5.0, 5.0, 1.0]])
    mask = np.array([[True, True, True, False], [True, True, True, True]])

    out = standardise_within_date(labels, mask)

    np.testing.assert_allclose(out[0, :3], [-1.224744871, 0.0, 1.224744871], atol=1e-8)
    assert np.isnan(out[0, 3]) and np.isnan(out[1, 0])
    assert out[1, 1:].mean() == pytest.approx(0.0) and out[1, 1:].std() == pytest.approx(1.0)


def _planted(panel: pd.DataFrame, dates: list[str]) -> np.ndarray:
    """Labels that follow one-week reversal plus noise."""
    signal = -past_return(panel, STOCKS, dates, start_lag=5)
    noise = np.random.default_rng(7).normal(0.0, 1.0, signal.shape) * np.nanstd(signal)
    return signal + 0.5 * noise


@pytest.mark.parametrize("architecture", ["fnn", "cnn_lstm"])
def test_the_net_learns_a_planted_reversal_and_is_reproducible(architecture):
    panel = _panel(n_sessions=420)
    sessions = sorted(panel.dt.unique())
    train, val, test = sessions[260:360], sessions[365:390], sessions[395:415]
    spec = NetSpec(architecture=architecture, members=2, max_epochs=15, patience=5)

    def run():
        return fit_predict(
            panel,
            STOCKS,
            train_dates=train,
            train_labels=_planted(panel, train),
            train_mask=None,
            val_dates=val,
            val_labels=_planted(panel, val),
            val_mask=None,
            test_dates=test,
            spec=spec,
        )

    first, second = run(), run()

    test_ic = daily_ic_series(first.scores, _planted(panel, test), "pearson").mean()
    assert test_ic > 0.3
    assert first.ensemble_val_ic > 0.3
    assert len(first.member_best_epoch) == 2
    np.testing.assert_array_equal(first.scores, second.scores)


def test_test_scores_are_nan_where_the_stock_has_no_close_on_the_date():
    panel = _panel(n_sessions=420)
    sessions = sorted(panel.dt.unique())
    train, val, test = sessions[260:360], sessions[365:390], sessions[395:400]
    panel = panel[~((panel.kdcode == STOCKS[3]) & (panel.dt == test[2]))]

    fit = fit_predict(
        panel,
        STOCKS,
        train_dates=train,
        train_labels=_planted(panel, train),
        train_mask=None,
        val_dates=val,
        val_labels=_planted(panel, val),
        val_mask=None,
        test_dates=test,
        spec=NetSpec(members=1, max_epochs=3, patience=2),
    )

    assert np.isnan(fit.scores[2, 3])
    assert np.isfinite(np.delete(fit.scores, 3, axis=1)).all()


def test_an_unknown_architecture_is_refused():
    panel = _panel(n_sessions=60)
    sessions = sorted(panel.dt.unique())
    with pytest.raises(ValueError, match="unknown architecture"):
        fit_predict(
            panel,
            STOCKS,
            train_dates=sessions[30:40],
            train_labels=np.zeros((10, len(STOCKS))),
            train_mask=None,
            val_dates=sessions[40:45],
            val_labels=np.zeros((5, len(STOCKS))),
            val_mask=None,
            test_dates=sessions[45:50],
            spec=NetSpec(architecture="transformer"),
        )


# --- The candidate arms on the #384 harness ---------------------------------------------

SMALL = NetSpec(members=1, max_epochs=3, patience=2)


def _context(n_sessions: int = 420) -> CandidateContext:
    panel = _panel(n_sessions=n_sessions)
    sessions = sorted(panel.dt.unique())
    train, val, test = sessions[260:360], sessions[365:390], sessions[395:415]
    rankable = np.ones((len(test), len(STOCKS)), dtype=bool)
    rankable[:, 0] = False  # one name is never tradable on the test dates
    data = {
        "train_dates": train,
        "val_dates": val,
        "test_dates": test,
        "train_labels": _planted(panel, train),
        "val_labels": _planted(panel, val),
        "train_loss_mask": np.ones((len(train), len(STOCKS)), dtype=bool),
        "val_tradable_mask": np.ones((len(val), len(STOCKS)), dtype=bool),
    }
    return CandidateContext(
        config=ExperimentConfig(),
        panel=panel.loc[panel.dt <= test[-1]].reset_index(drop=True),
        kdcodes=list(STOCKS),
        test_dates=test,
        rankable_mask=rankable,
        pit_intervals=None,
        data_loader=lambda: data,
    )


def test_both_arms_are_registered_under_stable_names():
    assert CANDIDATES["return_net_fnn"] is return_net_fnn
    assert CANDIDATES["return_net_cnn_lstm"] is return_net_cnn_lstm


@pytest.mark.parametrize("architecture", ["fnn", "cnn_lstm"])
def test_no_arm_score_moves_when_later_prices_are_scrambled(architecture):
    context = _context()
    spec = NetSpec(architecture=architecture, members=1, max_epochs=3, patience=2)

    check_no_lookahead(lambda c: _score_candidate(c, spec), context, context.test_dates[10])


def test_the_arm_keeps_the_runs_mask_and_records_its_training():
    context = _context()

    result = _score_candidate(context, SMALL)
    scores = check_result(result, context)

    assert np.isnan(scores[:, 0]).all()
    assert np.isfinite(scores[:, 1:]).all()
    assert result.controls["spec"]["architecture"] == "fnn"
    assert result.controls["input_lags_sessions"] == list(INPUT_LAGS)
    assert len(result.controls["member_best_epoch"]) == 1
    assert result.controls["train_dates"] == 100
