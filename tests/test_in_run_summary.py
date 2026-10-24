"""The in-run summary measures a tradable book and nets the universe (#336).

Every first-run number came from ``evaluation_summary.json``. Its Sharpe and
cumulative return treated overlapping four-session returns as daily ones, and no
number netted a benchmark. The summary now books the staggered tranche book the
label implies, reports its daily P&L, and leads with excess over the equal-weight
rankable universe. These tests pin that against cases whose answer is known:

* identical stocks, where any book's daily return is the market's;
* random scores on a rising market, where the excess must be near zero;
* small hand-worked panels for the sleeve bookkeeping, the missing-exit rule, k and
  the quintiles.
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from scipy import stats

import run_experiment
from mci_gru.config import Legs, create_config_from_dict
from mci_gru.evaluation.experiment_summary import compute_evaluation_summary
from mci_gru.evaluation.metrics import (
    SUMMARY_SCHEMA_VERSION,
    annualised_sharpe,
    evaluate_run,
)
from mci_gru.evaluation.outcomes import realized_returns, top_k_tranche_returns
from mci_gru.evaluation.statistics import (
    hansen_hodrick_mean_inference,
    hansen_hodrick_std,
    moving_block_bootstrap_ci,
)
from mci_gru.evaluation.tranche_book import tranche_book

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGS = Legs(entry_session=1, exit_session=5)


def _panel(closes: np.ndarray, kdcodes: list[str], start: str = "2020-01-01") -> pd.DataFrame:
    """Long panel from a ``(sessions, stocks)`` close matrix; NaN means no bar."""
    sessions = pd.bdate_range(start, periods=closes.shape[0]).strftime("%Y-%m-%d")
    rows = [
        {"kdcode": kdcode, "dt": session, "close": closes[t, j]}
        for t, session in enumerate(sessions)
        for j, kdcode in enumerate(kdcodes)
        if np.isfinite(closes[t, j])
    ]
    return pd.DataFrame(rows)


def _market(
    sessions: int,
    stocks: int,
    *,
    seed: int,
    drift: float = 0.0006,
    market_vol: float = 0.01,
    idio_vol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Closes and the market's daily returns; with ``idio_vol=0`` every stock is the market."""
    rng = np.random.default_rng(seed)
    market = rng.normal(drift, market_vol, sessions)
    daily = market[:, None] + rng.normal(0.0, idio_vol, (sessions, stocks))
    daily[0] = 0.0
    return 100.0 * np.cumprod(1.0 + daily, axis=0), market


def _run(
    closes: np.ndarray,
    scores: np.ndarray,
    *,
    top_k_values: list[int],
    legs: Legs = LEGS,
    tranches: int = 4,
    rankable: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """``evaluate_run`` on the first ``len(scores)`` sessions as signal dates."""
    kdcodes = [f"S{j:03d}" for j in range(closes.shape[1])]
    panel = _panel(closes, kdcodes)
    dates = sorted(panel["dt"].unique())[: len(scores)]
    if labels is None:
        labels = realized_returns(panel, dates, legs, kdcodes=kdcodes).returns
    return evaluate_run(
        scores,
        labels,
        panel=panel,
        dates=dates,
        kdcodes=kdcodes,
        rankable_mask=rankable,
        legs=legs,
        tranches=tranches,
        top_k_values=top_k_values,
        label_t=legs.exit_session,
        **kwargs,
    )


# --- Hansen-Hodrick HAC -------------------------------------------------------------


def test_hansen_hodrick_std_gives_every_lag_up_to_the_span_full_weight():
    x = np.array([0.3, -0.1, 0.4, 0.2, -0.5, 0.1, 0.6, -0.2, 0.0, 0.25])
    c = x - x.mean()
    n = x.size
    gamma = [float(c[k:] @ c[: n - k]) / n for k in range(4)]

    assert hansen_hodrick_std(x, lags=0) == pytest.approx(math.sqrt(gamma[0]))
    assert hansen_hodrick_std(x, lags=3) == pytest.approx(
        math.sqrt(gamma[0] + 2 * (gamma[1] + gamma[2] + gamma[3]))
    )


def test_hansen_hodrick_std_floors_a_negative_variance_at_zero():
    alternating = np.array([1.0, -1.0] * 20)

    assert hansen_hodrick_std(alternating, lags=1) == 0.0
    with pytest.raises(ValueError, match="lags"):
        hansen_hodrick_std(alternating, lags=-1)


def test_hansen_hodrick_inference_takes_its_lags_from_the_outcome_span():
    x = np.array([0.3, -0.1, 0.4, 0.2, -0.5, 0.1, 0.6, -0.2, 0.0, 0.25])

    result = hansen_hodrick_mean_inference(x, label_horizon=4)

    se = hansen_hodrick_std(x, lags=3) / math.sqrt(x.size)
    assert result.method == "hansen_hodrick_uniform_v1"
    assert result.lags == 3
    assert result.n_obs == x.size
    assert result.standard_error == pytest.approx(se)
    assert result.t_stat == pytest.approx(x.mean() / se)
    flat = hansen_hodrick_mean_inference(np.array([1.0, -1.0] * 20), label_horizon=2)
    assert flat.t_stat is None and flat.standard_error is None and flat.mean == 0.0


def test_the_hansen_hodrick_t_stat_holds_its_size_on_overlapping_returns():
    """Zero-mean 4-session returns sampled daily: |t| > 1.96 about 5% of the time."""
    rng = np.random.default_rng(7)
    rejections = 0
    trials = 400
    for _ in range(trials):
        daily = rng.normal(0.0, 1.0, 503)
        overlapping = np.convolve(daily, np.ones(4), mode="valid")
        t = hansen_hodrick_mean_inference(overlapping, label_horizon=4).t_stat
        rejections += abs(t) > 1.96
    assert 0.02 < rejections / trials < 0.09


# --- The tranche book ---------------------------------------------------------------


def test_the_book_rolls_each_sleeve_and_never_re_equalises_them():
    """One stock, two-session hold, two sleeves: worked out by hand."""
    closes = np.array([[100.0], [110.0], [121.0], [110.0], [100.0], [120.0], [132.0]])
    legs = Legs(entry_session=1, exit_session=3)
    panel = _panel(closes, ["A"])
    dates = sorted(panel["dt"].unique())[:4]
    realized = realized_returns(panel, dates, legs, kdcodes=["A"])
    ones = np.ones((4, 1))

    book = tranche_book(panel, ones, ones.astype(bool), realized, top_k=1, tranches=2)

    # Sleeve 0 holds the tranches of dates 0 and 2 (s1->s3, s3->s5) and is cash on s6;
    # sleeve 1 is cash until s2, then holds dates 1 and 3 (s2->s4, s4->s6).
    s0 = [0.5, 0.5 * 121 / 110, 0.5, 0.5 * 100 / 110, 0.5 * 120 / 110, 0.5 * 120 / 110]
    s1_roll = 0.5 * 100 / 121
    s1 = [0.5, 0.5, 0.5 * 110 / 121, s1_roll, s1_roll * 120 / 100, s1_roll * 132 / 100]
    value = np.add(s0, s1)
    np.testing.assert_allclose(book.daily_returns, value[1:] / value[:-1] - 1.0, rtol=1e-12)
    assert book.sessions == sorted(panel["dt"].unique())[2:7]
    assert book.tranches_opened == 4
    assert book.cumulative_return == pytest.approx(value[-1] / value[0] - 1.0)


def test_a_tranche_that_loses_its_exit_books_the_last_close_as_the_core_does():
    """A pick that stops trading mid-hold is held at its last close, then cash (E3)."""
    closes = np.array(
        [
            [100.0, 50.0],
            [100.0, 50.0],
            [110.0, 40.0],
            [120.0, 30.0],
            [130.0, np.nan],
            [140.0, np.nan],
        ]
    )
    panel = _panel(closes, ["A", "B"])
    dates = sorted(panel["dt"].unique())[:1]
    realized = realized_returns(panel, dates, LEGS, kdcodes=["A", "B"])
    scores = np.array([[1.0, 2.0]])
    mask = np.ones((1, 2), dtype=bool)

    book = tranche_book(panel, scores, mask, realized, top_k=2, tranches=1)

    core = top_k_tranche_returns(scores, mask, realized, top_k=2)[0]
    assert core == pytest.approx(0.5 * (140 / 100 - 1) + 0.5 * (30 / 50 - 1))
    assert book.cumulative_return == pytest.approx(core)
    # B's last close is session 3: held flat (cash) on sessions 4 and 5.
    value = np.array([1.0, 0.55 + 0.4, 0.6 + 0.3, 0.65 + 0.3, 0.7 + 0.3])
    np.testing.assert_allclose(book.daily_returns, value[1:] / value[:-1] - 1.0, rtol=1e-12)


def test_the_book_sharpe_matches_a_known_daily_sharpe_on_overlapping_labels():
    """Identical stocks: every book earns the market each session, so its Sharpe is known.

    The old summary annualised the overlapping four-session top-k returns as daily
    ones and reported about 1.15x (Bartlett) to 2x (naive) this value.
    """
    closes, market = _market(6000, 12, seed=3)
    scores = np.random.default_rng(4).normal(size=(5990, 12))

    summary = _run(closes, scores, top_k_values=[4])
    metrics = summary["metrics"]

    book_market = market[2:5995]  # sessions s2..s5994: first entry s1 to last exit s5994
    truth = book_market.mean() / book_market.std() * math.sqrt(252)
    assert truth > 0.5
    assert metrics["top_4_abs_sharpe"] == pytest.approx(truth, rel=0.02)
    assert metrics["universe_abs_sharpe"] == pytest.approx(truth, rel=0.02)
    assert summary["label_sessions"] == 4
    assert summary["tranches"] == 4


def test_cumulative_return_compounds_the_daily_book():
    """One sleeve, fully invested from first entry to last exit: exactly the market."""
    closes, market = _market(400, 6, seed=5, drift=0.004, market_vol=0.02)
    scores = np.random.default_rng(6).normal(size=(389, 6))

    metrics = _run(closes, scores, top_k_values=[3], tranches=1)["metrics"]

    # Tranches open on dates 0, 4, ..., 388: first entry s1, last exit s393.
    compounded = float(np.prod(1.0 + market[2:394]) - 1.0)
    assert compounded > 2.0  # summed returns would be far from this
    assert metrics["top_3_abs_cumulative_return"] == pytest.approx(compounded, rel=1e-9)
    assert metrics["universe_abs_cumulative_return"] == pytest.approx(compounded, rel=1e-9)
    assert metrics["top_3_excess_cumulative_return"] == pytest.approx(0.0, abs=1e-9)


# --- The headline: excess over the universe -----------------------------------------


def test_random_scores_on_a_rising_market_have_no_excess():
    closes, _ = _market(3000, 30, seed=8, drift=0.001, idio_vol=0.02)
    scores = np.random.default_rng(9).normal(size=(2990, 30))

    metrics = _run(closes, scores, top_k_values=[10])["metrics"]

    assert metrics["top_10_abs_period_return"] > 0.003  # the market rises ~0.4% a period
    assert abs(metrics["top_10_excess_period_return"]) < 0.0005
    assert abs(metrics["top_10_excess_t_stat"]) < 3.0
    assert metrics["top_10_abs_sharpe"] > 1.0
    assert abs(metrics["top_10_excess_sharpe"]) < 0.5


def test_scores_that_know_the_outcome_show_excess_with_a_large_t_stat():
    closes, _ = _market(800, 20, seed=10, drift=0.0, idio_vol=0.02)
    kdcodes = [f"S{j:03d}" for j in range(20)]
    panel = _panel(closes, kdcodes)
    dates = sorted(panel["dt"].unique())[:790]
    foresight = realized_returns(panel, dates, LEGS, kdcodes=kdcodes).returns

    metrics = _run(closes, foresight, top_k_values=[5])["metrics"]

    assert metrics["top_5_excess_period_return"] > 0.01
    assert metrics["top_5_excess_t_stat"] > 10.0
    assert metrics["top_5_excess_cumulative_return"] > 1.0


def test_every_k_comes_from_top_k_values_and_nets_the_universe_by_hand():
    """Six names, k of 2 and 3: the excess is the picks' mean minus all six."""
    closes, _ = _market(60, 6, seed=11, idio_vol=0.03)
    scores = np.tile(np.arange(6, dtype=float), (50, 1))  # S005 best, then S004, ...
    kdcodes = [f"S{j:03d}" for j in range(6)]
    panel = _panel(closes, kdcodes)
    dates = sorted(panel["dt"].unique())[:50]
    booked = realized_returns(panel, dates, LEGS, kdcodes=kdcodes).returns

    summary = _run(closes, scores, top_k_values=[2, 3])
    metrics = summary["metrics"]

    for k in (2, 3):
        expected = np.mean(booked[:, 6 - k :].mean(axis=1) - booked.mean(axis=1))
        assert metrics[f"top_{k}_excess_period_return"] == pytest.approx(expected, rel=1e-12)
    assert summary["top_k_values"] == [2, 3]
    assert not [key for key in metrics if key.startswith(("top_50", "top_10_"))]


def test_a_name_outside_the_rankable_set_is_never_bought():
    closes, _ = _market(40, 4, seed=12, idio_vol=0.03)
    scores = np.tile([9.0, 1.0, 2.0, 3.0], (30, 1))  # S000 always scores highest
    rankable = np.ones((30, 4), dtype=bool)
    rankable[:, 0] = False

    excluded = _run(closes, scores, top_k_values=[1], rankable=rankable)["metrics"]
    without = _run(closes[:, 1:], scores[:, 1:], top_k_values=[1])["metrics"]

    assert excluded["top_1_abs_period_return"] == pytest.approx(without["top_1_abs_period_return"])
    assert excluded["top_1_abs_sharpe"] == pytest.approx(without["top_1_abs_sharpe"])


def test_the_universe_benchmark_uses_the_same_mask_as_the_picks():
    """A soaring name outside the mask is in neither the picks nor the benchmark."""
    closes, _ = _market(40, 4, seed=24, idio_vol=0.01)
    closes[:, 0] *= np.cumprod(np.full(40, 1.05))  # S000 gains 5% a session
    scores = np.tile([0.0, 1.0, 2.0, 3.0], (30, 1))
    rankable = np.ones((30, 4), dtype=bool)
    rankable[:, 0] = False

    excluded = _run(closes, scores, top_k_values=[1], rankable=rankable)["metrics"]
    without = _run(closes[:, 1:], scores[:, 1:], top_k_values=[1])["metrics"]

    for key in (
        "universe_abs_period_return",
        "universe_abs_sharpe",
        "universe_abs_cumulative_return",
        "top_1_excess_period_return",
        "top_1_excess_sharpe",
        "top_1_excess_cumulative_return",
    ):
        assert excluded[key] == pytest.approx(without[key], rel=1e-12), key


def test_a_window_with_no_resolved_hold_reports_nan_not_zero():
    closes, _ = _market(5, 4, seed=25, idio_vol=0.01)
    scores = np.random.default_rng(26).normal(size=(3, 4))  # holds run past the panel

    metrics = _run(closes, scores, top_k_values=[2])["metrics"]

    assert metrics["book_sessions"] == 0
    for key in (
        "top_2_abs_cumulative_return",
        "top_2_excess_cumulative_return",
        "top_2_abs_sharpe",
    ):
        assert math.isnan(metrics[key]), key


def test_missing_close_events_count_only_names_that_could_be_held():
    closes, _ = _market(12, 3, seed=27, idio_vol=0.01)
    closes[4:, 0] = np.nan  # S000 stops trading after session 3
    closes[4:, 1] = np.nan  # so does S001
    scores = np.tile([1.0, 2.0, 3.0], (2, 1))
    rankable = np.ones((2, 3), dtype=bool)
    rankable[:, 1] = False  # S001 was never holdable

    summary = _run(closes, scores, top_k_values=[1], rankable=rankable)

    assert summary["missing_close_events"] == {"missing_entry": 0, "missing_exit": 2}


# --- What the summary says about itself ---------------------------------------------


def test_the_summary_labels_its_basis_and_drops_metrics_an_ic_loss_leaves_arbitrary():
    closes, _ = _market(40, 8, seed=13, idio_vol=0.02)
    scores = np.random.default_rng(14).normal(size=(30, 8))

    summary = _run(closes, scores, top_k_values=[3])

    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION == "mci_gru.evaluation_summary.v2"
    assert summary["return_basis"] == "price"
    assert summary["benchmark"] == "equal_weight_rankable_universe"
    assert summary["legs"] == {"entry_session": 1, "exit_session": 5}
    assert summary["label_t"] == 5 and summary["label_sessions"] == 4
    assert summary["excess_inference"] == {"method": "hansen_hodrick_uniform_v1", "lags": 3}
    removed = {
        "mse",
        "rmse",
        "mae",
        "hit_rate",
        "sharpe_ratio",
        "sharpe_naive",
        "sharpe_newey_west",
        "cumulative_return",
        "avg_portfolio_return",
        "top_k",
        "return_top_3",
        "sharpe_top_3",
    }
    assert not removed & set(summary["metrics"])
    json.dumps(summary)


def test_ic_and_rank_ic_are_unchanged_by_the_new_summary():
    """Control: the per-date IC means and their bootstrap CIs, recomputed independently."""
    rng = np.random.default_rng(15)
    closes, _ = _market(70, 10, seed=16, idio_vol=0.02)
    labels = rng.normal(size=(60, 10))
    scores = labels * 0.2 + rng.normal(size=(60, 10))
    labels[3, 2] = np.nan

    metrics = _run(
        closes,
        scores,
        top_k_values=[3],
        labels=labels,
        bootstrap_enabled=True,
        bootstrap_resamples=50,
        bootstrap_seed=1729,
        block_size=5,
    )["metrics"]

    def series(stat):
        out = []
        for p, r in zip(scores, labels, strict=True):
            ok = np.isfinite(r)
            out.append(stat(p[ok], r[ok])[0])
        return np.array(out)

    ic = series(stats.pearsonr)
    rank_ic = series(stats.spearmanr)
    assert metrics["avg_ic"] == pytest.approx(ic.mean(), abs=1e-12)
    assert metrics["avg_rank_ic"] == pytest.approx(rank_ic.mean(), abs=1e-12)
    assert metrics["median_rank_ic"] == pytest.approx(np.median(rank_ic), abs=1e-12)
    ci = moving_block_bootstrap_ci(ic, np.mean, 5, 50, 1729, 0.95)
    assert metrics["avg_ic_ci_lower"] == pytest.approx(ci["lower"], abs=1e-12)
    assert metrics["avg_ic_ci_upper"] == pytest.approx(ci["upper"], abs=1e-12)


def test_ic_date_coverage_counts_dates_that_could_not_be_scored():
    closes, _ = _market(30, 6, seed=17, idio_vol=0.02)
    rng = np.random.default_rng(18)
    labels = rng.normal(size=(20, 6))
    scores = rng.normal(size=(20, 6))
    scores[4] = 0.5  # constant scores: no IC that date

    metrics = _run(closes, scores, top_k_values=[2], labels=labels)["metrics"]

    assert metrics["test_dates"] == 20
    assert metrics["ic_valid_dates"] == 19
    assert metrics["rank_ic_valid_dates"] == 19
    assert metrics["ic_date_coverage"] == pytest.approx(0.95)


def test_quintiles_split_the_cross_section_evenly():
    """Twelve names in five quintiles are 2, 2, 2, 3, 3 from the bottom, not 2, 2, 2, 2, 4."""
    closes, _ = _market(20, 12, seed=19, idio_vol=0.03)
    scores = np.tile(np.arange(12, dtype=float), (10, 1))
    kdcodes = [f"S{j:03d}" for j in range(12)]
    panel = _panel(closes, kdcodes)
    dates = sorted(panel["dt"].unique())[:10]
    booked = realized_returns(panel, dates, LEGS, kdcodes=kdcodes).returns

    metrics = _run(closes, scores, top_k_values=[3])["metrics"]

    bounds = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 12)]
    for q, (lo, hi) in enumerate(bounds, start=1):
        expected = booked[:, lo:hi].mean(axis=1).mean()
        assert metrics[f"quintile_{q}_abs_period_return"] == pytest.approx(expected, rel=1e-12)
    assert metrics["long_short_abs_period_spread"] == pytest.approx(
        booked[:, 9:].mean(axis=1).mean() - booked[:, :2].mean(axis=1).mean(), rel=1e-12
    )


def test_sharpe_method_is_honoured_on_the_daily_book():
    closes, _ = _market(300, 8, seed=20, idio_vol=0.02)
    scores = np.random.default_rng(21).normal(size=(290, 8))
    kdcodes = [f"S{j:03d}" for j in range(8)]
    panel = _panel(closes, kdcodes)
    dates = sorted(panel["dt"].unique())[:290]
    realized = realized_returns(panel, dates, LEGS, kdcodes=kdcodes)
    daily = tranche_book(
        panel, scores, np.ones((290, 8), dtype=bool), realized, top_k=3, tranches=4
    ).daily_returns

    naive = _run(closes, scores, top_k_values=[3], sharpe_method="naive")
    hac = _run(closes, scores, top_k_values=[3], sharpe_method="newey_west", newey_west_lags=6)

    assert naive["metrics"]["top_3_abs_sharpe"] == pytest.approx(
        daily.mean() / daily.std() * math.sqrt(252)
    )
    assert hac["metrics"]["top_3_abs_sharpe"] == pytest.approx(
        annualised_sharpe(daily, method="newey_west", lags=6)
    )
    assert naive["sharpe_method"] == "naive" and hac["newey_west_lags"] == 6
    assert naive["metrics"]["top_3_abs_sharpe"] != hac["metrics"]["top_3_abs_sharpe"]


def test_the_config_drives_legs_tranches_and_lags():
    config = create_config_from_dict(
        {
            "model": {"label_t": 5},
            "evaluation": {"top_k_values": [2], "bootstrap_enabled": False, "tranches": 2},
        }
    )
    closes, _ = _market(40, 6, seed=22, idio_vol=0.02)
    kdcodes = [f"S{j:03d}" for j in range(6)]
    panel = _panel(closes, kdcodes)
    dates = sorted(panel["dt"].unique())[:30]
    scores = np.random.default_rng(23).normal(size=(30, 6))

    summary = compute_evaluation_summary(
        scores,
        scores,
        config,
        panel=panel,
        dates=dates,
        kdcodes=kdcodes,
        rankable_mask=np.ones((30, 6), dtype=bool),
    )

    assert summary["tranches"] == 2
    assert summary["legs"] == {"entry_session": 1, "exit_session": 5}
    assert summary["newey_west_lags"] == 3
    assert summary["benchmark"] == "equal_weight_pit_tradable_universe"
    assert "top_2_excess_period_return" in summary["metrics"]


# --- Runner wiring ------------------------------------------------------------------

RUNNER_OVERRIDES = [
    "features=base",
    "data.source=csv",
    "data.use_pit_universe=false",
    "data.train_start=2020-01-01",
    "data.train_end=2020-02-14",
    "data.val_start=2020-02-18",
    "data.val_end=2020-02-28",
    "data.test_start=2020-03-03",
    "data.test_end=2020-03-20",
    "model.his_t=3",
    "model.label_t=3",
    "model.gru_hidden_sizes=[8,4]",
    "model.hidden_size_gat1=8",
    "model.output_gat1=4",
    "model.gat_heads=1",
    "model.hidden_size_gat2=8",
    "model.num_hidden_states=4",
    "model.cross_attn_heads=1",
    "model.use_self_attention=false",
    "training.num_epochs=1",
    "training.num_models=1",
    "training.batch_size=2",
    "training.warmup_steps=0",
    "tracking.enabled=false",
    "evaluation.top_k_values=[2,6]",
]
RUNNER_STOCKS = ["A.N", "B.N", "C.N", "D.N", "E.N"]


def _write_runner_panel(path: Path, days: int = 105) -> None:
    dates = pd.bdate_range("2019-11-01", periods=days)
    rows = []
    for s_idx, stock in enumerate(RUNNER_STOCKS):
        price = 100.0 + s_idx
        for d_idx, dt in enumerate(dates):
            open_px = price * (1.0 + 0.002 * np.sin(d_idx / 4.0 + s_idx))
            close_px = open_px * (1.0 + 0.001 * (s_idx + 1) * np.cos(d_idx / 7.0 + s_idx))
            rows.append(
                {
                    "kdcode": stock,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": open_px,
                    "high": max(open_px, close_px) * 1.001,
                    "low": min(open_px, close_px) * 0.999,
                    "close": close_px,
                    "volume": 1_000_000.0 + d_idx,
                }
            )
            price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)


@contextmanager
def _restored_process_state(directory: Path):
    root = logging.getLogger()
    handlers, level, cwd = list(root.handlers), root.level, os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture(scope="module")
def runner_summary(tmp_path_factory):
    """Run the runner once on CPU, recording what it handed the summary."""
    base = tmp_path_factory.mktemp("in_run_summary")
    panel_path = base / "panel.csv"
    _write_runner_panel(panel_path)
    out = base / "out"
    out.mkdir()
    calls = []

    def spy(predictions, labels, config, **kwargs):
        calls.append({"predictions": np.array(predictions), **kwargs})
        return compute_evaluation_summary(predictions, labels, config, **kwargs)

    with initialize_config_dir(config_dir=str(REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose(
            "config",
            overrides=[
                *RUNNER_OVERRIDES,
                "graph.corr_lookback_days=20",
                f"data.filename={panel_path.as_posix()}",
            ],
        )
    patch = pytest.MonkeyPatch()
    patch.setattr(run_experiment, "compute_evaluation_summary", spy)
    patch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        with _restored_process_state(out):
            run_experiment.main.__wrapped__(cfg)
    finally:
        patch.undo()
    (call,) = calls
    summary = json.loads((out / "evaluation_summary.json").read_text())
    return call, summary, panel_path


def test_the_runner_scores_the_book_on_the_panel_it_trained_on(runner_summary):
    call, _, panel_path = runner_summary
    source = pd.read_csv(panel_path)

    assert call["kdcodes"] == RUNNER_STOCKS
    assert call["dates"][0] == "2020-03-03" and call["dates"][-1] <= "2020-03-20"
    assert call["predictions"].shape == (len(call["dates"]), len(RUNNER_STOCKS))
    assert call["rankable_mask"] is None  # no PIT masks in this run
    merged = call["panel"].merge(source, on=["kdcode", "dt"], suffixes=("", "_csv"))
    np.testing.assert_allclose(merged["close"], merged["close_csv"])


def test_the_runner_writes_the_v2_summary(runner_summary):
    _, summary, _ = runner_summary
    metrics = summary["metrics"]

    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["label_sessions"] == 2 and summary["tranches"] == 2
    assert metrics["book_sessions"] > 0
    assert math.isfinite(metrics["top_2_excess_period_return"])
    # k at least the universe holds everyone, so it cannot beat it.
    assert metrics["top_6_excess_period_return"] == pytest.approx(0.0, abs=1e-15)
    assert "hit_rate" not in metrics and "mse" not in metrics
