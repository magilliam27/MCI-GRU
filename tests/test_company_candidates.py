"""Earnings candidates read only releases public by each close, and the control is what it says (#393).

``mci_gru.evaluation.company_candidates`` scores the latest known earnings release and
the reversal-plus-momentum control. On a synthetic panel these tests pin that:

* the surprise is reported minus consensus EPS over the close two sessions before the
  release is known, and lasts ``MAX_EVENT_AGE`` sessions, no longer;
* a release after the close is not read on its own report date;
* the announcement return is the two-session return to the known session, less the
  mean of every name over the same sessions;
* neither a later release nor scrambled later prices moves an earlier score;
* the control is the mean centred rank of 1-week reversal and 12-1 momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import ExperimentConfig
from mci_gru.data.company_info import EARNINGS_COLUMNS
from mci_gru.evaluation import company_candidates as cc
from mci_gru.evaluation.baselines import momentum_scores, reversal_scores
from mci_gru.evaluation.candidates import CandidateContext, check_no_lookahead
from mci_gru.evaluation.residual_linear import centred_ranks

STOCKS = [f"S{i:02d}.N" for i in range(10)]
N_SESSIONS = 420
TEST_FROM = 300


def _panel(seed: int = 5) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    sessions = list(pd.bdate_range("2019-01-01", periods=N_SESSIONS).strftime("%Y-%m-%d"))
    returns = rng.normal(0.0003, 0.015, (N_SESSIONS, len(STOCKS)))
    close = 40.0 * np.cumprod(1.0 + returns, axis=0)
    panel = pd.DataFrame(
        [
            {"kdcode": code, "dt": dt, "open": c, "high": c, "low": c, "close": c, "volume": 1e6}
            for t, dt in enumerate(sessions)
            for code, c in zip(STOCKS, close[t], strict=True)
        ]
    )
    return panel, sessions, close


def _events(sessions: list[str]) -> pd.DataFrame:
    """Three releases per stock, spaced a quarter apart, with varied surprises."""
    rows = []
    for j, code in enumerate(STOCKS):
        for q, at in enumerate((280 + j, 310 + j, 360 + j)):
            rows.append(
                [
                    code,
                    sessions[at],
                    "after" if (j + q) % 2 else "before",
                    "2019-12-31",
                    1.0 + 0.1 * j - 0.05 * q,
                    1.0,
                    code.replace(".N", ".US"),
                ]
            )
    return pd.DataFrame(rows, columns=list(EARNINGS_COLUMNS))


@pytest.fixture()
def world(tmp_path, monkeypatch):
    panel, sessions, close = _panel()
    events = _events(sessions)
    path = tmp_path / "earnings_events.csv"
    events.to_csv(path, index=False)
    monkeypatch.setenv(cc.EARNINGS_EVENTS_ENV, str(path))
    test_dates = sessions[TEST_FROM:]
    context = CandidateContext(
        config=ExperimentConfig(),
        panel=panel,
        kdcodes=list(STOCKS),
        test_dates=test_dates,
        rankable_mask=np.ones((len(test_dates), len(STOCKS)), dtype=bool),
        pit_intervals=None,
        data_loader=dict,
    )
    return context, panel, sessions, close, events, path


def test_the_surprise_is_scaled_by_the_close_before_the_release_and_ages_out(world):
    _, panel, sessions, close, events, _ = world
    j = 3
    release = events.loc[events["kdcode"] == STOCKS[j]].iloc[1]  # session 313
    assert release["timing"] == "before"  # (3 + 1) % 2 == 0
    known = 313
    # Drop the third release (session 363) so this one stays the latest to the age limit.
    only_two = events.loc[events["report_date"] != sessions[360 + j]]
    dates = [
        sessions[known - 1],
        sessions[known],
        sessions[known + cc.MAX_EVENT_AGE],
        sessions[known + cc.MAX_EVENT_AGE + 1],
    ]
    signals = cc.earnings_signals(panel, list(STOCKS), dates, only_two)
    expected = (release["eps_actual"] - 1.0) / close[known - 2, j]
    # The day before, the previous release (283, after the close: known 284) still scores.
    assert signals.age[0, j] == known - 1 - 284
    assert signals.surprise[1, j] == pytest.approx(expected)
    assert signals.age[1, j] == 0
    assert signals.surprise[2, j] == pytest.approx(expected)
    assert signals.age[2, j] == cc.MAX_EVENT_AGE
    assert np.isnan(signals.surprise[3, j]) and signals.age[3, j] == -1


def test_a_release_after_the_close_is_not_read_on_its_report_date(world):
    _, panel, sessions, close, events, _ = world
    j = 2
    release = events.loc[events["kdcode"] == STOCKS[j]].iloc[1]  # session 312
    assert release["timing"] == "after"  # (2 + 1) % 2 == 1
    signals = cc.earnings_signals(panel, list(STOCKS), [sessions[312], sessions[313]], events)
    assert signals.age[0, j] == 312 - 282  # still the release before (282, before the open)
    assert signals.age[1, j] == 0
    assert signals.surprise[1, j] == pytest.approx((release["eps_actual"] - 1.0) / close[311, j])


def test_the_announcement_return_is_net_of_every_names_mean_move(world):
    _, panel, sessions, close, events, _ = world
    j, known = 3, 313
    signals = cc.earnings_signals(panel, list(STOCKS), [sessions[known]], events)
    two_day = close[known] / close[known - 2] - 1.0
    assert signals.announcement_return[0, j] == pytest.approx(two_day[j] - two_day.mean())


def test_a_later_release_never_moves_an_earlier_score(world):
    context, _, sessions, _, events, path = world
    cut = 330
    before = cc.earnings_surprise(context).scores
    extra = events.copy()
    extra["report_date"] = sessions[cut + 1]
    extra["eps_actual"] = 50.0
    pd.concat([events, extra]).to_csv(path, index=False)
    after = cc.earnings_surprise(context).scores
    rows = cut - TEST_FROM + 1  # test dates up to and including session `cut`
    np.testing.assert_array_equal(before[:rows], after[:rows])
    assert not np.array_equal(before[rows + 2 :], after[rows + 2 :])


@pytest.mark.parametrize(
    "candidate", [cc.earnings_surprise, cc.reversal_momentum, cc.reversal_momentum_earnings]
)
def test_no_score_moves_when_later_prices_are_scrambled(world, candidate):
    context = world[0]
    check_no_lookahead(candidate, context, context.test_dates[40])


def test_the_control_is_the_mean_centred_rank_of_reversal_and_momentum(world):
    context = world[0]
    mask = context.rankable_mask
    rev = reversal_scores(context.panel, context.kdcodes, context.test_dates, 5)
    mom = momentum_scores(context.panel, context.kdcodes, context.test_dates)
    expected = (centred_ranks(rev, mask) + centred_ranks(mom, mask)) / 2.0
    np.testing.assert_allclose(cc.reversal_momentum(context).scores, expected)


def test_the_combined_arm_weights_its_three_parts_equally(world):
    context = world[0]
    mask = context.rankable_mask
    rev = reversal_scores(context.panel, context.kdcodes, context.test_dates, 5)
    mom = momentum_scores(context.panel, context.kdcodes, context.test_dates)
    earn = cc.earnings_surprise(context).scores
    expected = (centred_ranks(rev, mask) + centred_ranks(mom, mask) + centred_ranks(earn, mask)) / 3
    np.testing.assert_allclose(cc.reversal_momentum_earnings(context).scores, expected)


def test_without_the_events_file_the_earnings_arms_refuse(world, monkeypatch):
    monkeypatch.delenv(cc.EARNINGS_EVENTS_ENV)
    with pytest.raises(ValueError, match=cc.EARNINGS_EVENTS_ENV):
        cc.earnings_surprise(world[0])
