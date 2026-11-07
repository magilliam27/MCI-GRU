"""The residual-return linear model removes market and peer moves, and finds a planted signal (#384).

``mci_gru.evaluation.residual_linear`` is the survey's first candidate. These tests
build a synthetic panel whose daily returns are a market move times each stock's beta,
plus a move shared by the stock's industry, plus a stock-specific move that partly
reverses over the following week, and pin that:

* membership follows the PIT intervals, so a name is outside the market average
  before it joins;
* betas recover the planted betas (shrunk a third of the way to one), and residual
  returns carry neither the market move nor the industry move;
* clusters recover the planted industries from return correlations alone;
* a forward label made only of market and peer moves residualises to zero;
* neither candidate's score on a date moves when later prices are scrambled;
* the fitted model finds the planted reversal: positive test IC and a positive weight
  on residual reversal.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation import residual_linear as rl
from mci_gru.evaluation.candidates import CandidateContext, check_no_lookahead, check_result
from mci_gru.evaluation.statistics import daily_ic_series

N_INDUSTRIES = 4
PER_INDUSTRY = 6
STOCKS = [f"S{i:02d}.N" for i in range(N_INDUSTRIES * PER_INDUSTRY)]
INDUSTRY = np.repeat(np.arange(N_INDUSTRIES), PER_INDUSTRY)
BETAS = np.linspace(0.6, 1.4, len(STOCKS))
LABEL_T = 5
JOINER = STOCKS[-1]
JOIN_DATE = "2018-06-01"


def _synthetic(reversal: float = 0.6, seed: int = 3):
    """Panel, PIT intervals, sessions, and every session's forward 5-session return."""
    rng = np.random.default_rng(seed)
    sessions = list(pd.bdate_range("2017-01-02", periods=900).strftime("%Y-%m-%d"))
    n, k = len(sessions), len(STOCKS)
    market = rng.normal(0.0004, 0.010, n)
    industry = rng.normal(0.0, 0.008, (n, N_INDUSTRIES))
    shock = rng.normal(0.0, 0.012, (n, k))
    own = shock.copy()
    for t in range(1, n):
        # Part of last week's stock-specific shock reverses, spread over this week.
        own[t] -= reversal * shock[max(0, t - 5) : t].sum(axis=0) / 5.0
    returns = BETAS * market[:, None] + industry[:, INDUSTRY] + own
    close = 50.0 * np.cumprod(1.0 + returns, axis=0)
    volume = rng.uniform(1e6, 2e6, (n, k))
    rows = [
        {
            "kdcode": code,
            "dt": dt,
            "open": close[t, j],
            "high": close[t, j],
            "low": close[t, j],
            "close": close[t, j],
            "volume": volume[t, j],
        }
        for t, dt in enumerate(sessions)
        for j, code in enumerate(STOCKS)
    ]
    panel = pd.DataFrame(rows)
    forward = np.full((n, k), np.nan)
    # The pipeline's label: close D+1 to close D+LABEL_T (compute_labels).
    forward[: n - LABEL_T] = close[LABEL_T:] / close[1 : n - LABEL_T + 1] - 1.0
    pit = pd.DataFrame(
        [
            {
                "kdcode": code,
                "valid_from": JOIN_DATE if code == JOINER else sessions[0],
                "valid_to": sessions[-1],
            }
            for code in STOCKS
        ]
    )
    return panel, pit, sessions, forward, market, industry


@pytest.fixture(scope="module")
def synthetic():
    return _synthetic()


def _context(synthetic, *, reversal_panel=None) -> tuple[CandidateContext, np.ndarray]:
    panel, pit, sessions, forward, _, _ = synthetic if reversal_panel is None else reversal_panel
    train = list(range(300, 640))
    val = list(range(650, 760))
    test = list(range(770, 880))
    index = {s: i for i, s in enumerate(sessions)}
    member = rl.membership(pit, STOCKS, sessions)
    labels = np.where(member, forward, np.nan)
    # Training labels end LABEL_T sessions before validation, as the pipeline cuts them.
    data = {
        "train_dates": [sessions[i] for i in train],
        "val_dates": [sessions[i] for i in val],
        "test_dates": [sessions[i] for i in test],
        "train_labels": labels[train],
        "val_labels": labels[val],
        "train_loss_mask": member[train],
        "val_tradable_mask": member[val],
        "test_labels": labels[test],
    }
    test_dates = [sessions[i] for i in test]
    context = CandidateContext(
        config=ExperimentConfig(),
        panel=panel.loc[panel["dt"] <= test_dates[-1]].reset_index(drop=True),
        kdcodes=list(STOCKS),
        test_dates=test_dates,
        rankable_mask=member[[index[d] for d in test_dates]],
        pit_intervals=pit,
        data_loader=lambda: data,
    )
    return context, labels[test]


@pytest.fixture(scope="module")
def state(synthetic):
    """The residual state with one cluster per planted industry.

    With more clusters than industries, some industries split into clusters too small to
    have peers, and those names keep their industry move by design.
    """
    panel, pit, *_ = synthetic
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rl, "N_CLUSTERS", N_INDUSTRIES)
        return rl.residual_state(panel, STOCKS, pit)


def test_membership_follows_the_pit_intervals(synthetic):
    _, pit, sessions, *_ = synthetic
    member = rl.membership(pit, STOCKS, sessions)
    joiner = STOCKS.index(JOINER)
    before = [i for i, s in enumerate(sessions) if s < JOIN_DATE]
    after = [i for i, s in enumerate(sessions) if s >= JOIN_DATE]
    assert not member[before, joiner].any()
    assert member[after, joiner].all()
    assert member[:, :joiner].all()
    assert rl.membership(None, STOCKS, sessions).all()


def test_betas_recover_the_planted_betas_shrunk_toward_one(state):
    late = len(state.sessions) - 1
    expected = (1.0 - rl.BETA_SHRINK) * BETAS + rl.BETA_SHRINK
    # The joiner's peers average its industry, which is part of every return.
    assert np.abs(state.betas[late] - expected).max() < 0.12
    first = state.betas[0]
    assert (first == 1.0).all()


def test_residual_returns_carry_neither_the_market_nor_the_industry_move(synthetic, state):
    _, _, _, _, market, industry = synthetic
    rows = np.arange(400, len(state.sessions))
    raw = state.returns[rows]
    resid = state.residuals[rows]

    def mean_abs_corr(values, factor):
        out = []
        for j in range(values.shape[1]):
            keep = np.isfinite(values[:, j])
            out.append(abs(np.corrcoef(values[keep, j], factor[keep])[0, 1]))
        return float(np.mean(out))

    assert mean_abs_corr(raw, market[rows]) > 0.4
    assert mean_abs_corr(resid, market[rows]) < 0.08
    own_industry = industry[rows][:, INDUSTRY]
    raw_ind = np.mean(
        [abs(np.corrcoef(raw[:, j], own_industry[:, j])[0, 1]) for j in range(len(STOCKS) - 1)]
    )
    resid_ind = np.mean(
        [abs(np.corrcoef(resid[:, j], own_industry[:, j])[0, 1]) for j in range(len(STOCKS) - 1)]
    )
    assert raw_ind > 0.3
    assert resid_ind < 0.12


def test_clusters_recover_the_planted_industries(state, monkeypatch):
    monkeypatch.setattr(rl, "N_CLUSTERS", N_INDUSTRIES)
    assert state.peered_share > 0.6
    late = len(state.sessions) - 1
    start = late - rl.ESTIMATION_SESSIONS
    r = state.returns[start:late]
    m = rl._row_mean(r, state.members[start:late])
    u = r - rl._betas(r, m) * m[:, None]
    labels = rl._clusters(u, state.members[late])
    for industry in range(N_INDUSTRIES):
        assert len(set(labels[industry == INDUSTRY])) == 1
    assert len(set(labels)) == N_INDUSTRIES


def test_a_label_of_market_and_peer_moves_only_residualises_to_zero(state):
    rows = np.arange(700, 720)
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.02, len(rows))
    peer = rng.normal(0, 0.02, (len(rows), 11))
    clusters = state.clusters[rows]
    labels = state.betas[rows] * market[:, None] + np.take_along_axis(peer, clusters, axis=1)
    out = rl.residual_labels(state, labels, rows)
    # The market move goes exactly; the peer move goes for every clustered name, up to the
    # small leftover of the market average including the peers' own moves.
    clustered = clusters > 0
    raw = labels - labels.mean(axis=1, keepdims=True)
    assert np.abs(out[clustered]).mean() < 0.25 * np.abs(raw[clustered]).mean()


@pytest.mark.parametrize("candidate", [rl.residual_reversal_1w, rl.residual_linear])
def test_no_score_moves_when_later_prices_are_scrambled(synthetic, candidate):
    context, _ = _context(synthetic)
    check_no_lookahead(candidate, context, context.test_dates[50])


def test_the_lookahead_check_catches_a_score_that_reads_tomorrow(synthetic):
    context, _ = _context(synthetic)

    def peeking(ctx):
        state = rl.residual_state(ctx.panel, ctx.kdcodes, ctx.pit_intervals)
        rows = rl._session_rows(state, ctx.test_dates)
        tomorrow = np.minimum(rows + 1, len(state.sessions) - 1)
        return rl.CandidateResult(state.returns[tomorrow])

    with pytest.raises(ValueError, match="change with prices after"):
        check_no_lookahead(peeking, context, context.test_dates[50])


def test_the_fitted_model_finds_the_planted_reversal(synthetic):
    context, test_labels = _context(synthetic)
    result = rl.residual_linear(context)
    scores = check_result(result, context)

    ic = float(np.nanmean(daily_ic_series(scores, test_labels, "pearson")))
    assert ic > 0.1
    weights = result.controls["coefficients"]
    assert weights["rrev_5"] > 0
    assert weights["rrev_5"] == max(weights.values())
    assert not np.isfinite(scores[~context.rankable_mask]).any()


def test_without_a_planted_reversal_the_model_finds_little(synthetic):
    context, test_labels = _context(synthetic, reversal_panel=_synthetic(reversal=0.0, seed=4))
    scores = check_result(rl.residual_linear(context), context)
    ic = float(np.nanmean(daily_ic_series(scores, test_labels, "pearson")))
    assert abs(ic) < 0.05


def test_residual_reversal_is_minus_the_last_weeks_residual_return(synthetic):
    context, _ = _context(synthetic)
    state = rl.residual_state(context.panel, STOCKS, context.pit_intervals)
    scores = check_result(rl.residual_reversal_1w(context), context)
    rows = rl._session_rows(state, context.test_dates)
    expected = -state.residuals[rows[0] - 4 : rows[0] + 1].sum(axis=0)
    member = context.rankable_mask[0]
    np.testing.assert_allclose(scores[0, member], expected[member], rtol=1e-10)


def test_centred_ranks_span_minus_half_to_half_and_leave_gaps_in_the_middle():
    values = np.array([[3.0, np.nan, 1.0, 2.0, 5.0]])
    scorable = np.array([[True, True, True, True, False]])
    out = rl.centred_ranks(values, scorable)
    np.testing.assert_allclose(out[0, :4], [0.5, 0.0, -0.5, 0.0])
    assert np.isnan(out[0, 4])


def test_without_peers_the_market_move_alone_is_removed(synthetic):
    """Peer means also carry the market move, so check market removal with peers off."""
    panel, pit, _, _, market, _ = synthetic
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rl, "MIN_PEERS", 10**6)
        state = rl.residual_state(panel, STOCKS, pit)
    assert state.peered_share == 0.0
    rows = np.arange(400, len(state.sessions))
    corr = [
        abs(np.corrcoef(state.residuals[rows, j], market[rows])[0, 1])
        for j in range(len(STOCKS) - 1)
    ]
    assert float(np.mean(corr)) < 0.08

    # With no peers, a label loses its beta times the date's equal-weight mean label.
    betas = state.betas[rows[:20]]
    labels = betas * np.linspace(-0.03, 0.03, 20)[:, None] + np.random.default_rng(1).normal(
        0.0, 0.01, betas.shape
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rl, "MIN_PEERS", 10**6)
        out = rl.residual_labels(state, labels, rows[:20])
    expected = labels - betas * labels.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)


def test_a_run_trained_on_rank_labels_is_refused(synthetic):
    context, _ = _context(synthetic)
    config = ExperimentConfig()
    config.training.label_type = "rank"
    with pytest.raises(ValueError, match="residualises return labels"):
        rl.residual_linear(dataclasses.replace(context, config=config))
