"""Side-module statistics are overlap-correct (#342).

Each test reproduces a Monte Carlo truth from the backtest audit's side-module
repros (project files, ``backtest/repro-math/side/``): the daily outcome is a
close ``D+1`` to close ``D+5`` return, so consecutive days share three sessions and
every daily series built from it is MA(3) under the null.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.paired_inference import (
    minimum_detectable_effect,
    paired_mean_inference,
    required_days,
    sharpe_block_bootstrap_ci,
)
from mci_gru.evaluation.selection_audit import (
    _daily_books,
    build_selection_audit,
    deflated_sharpe_ratio,
)
from mci_gru.evaluation.selection_nulls import (
    DatedScoreOutcome,
    _block_permutation,
    _induced_sources,
    run_date_blocked_permutation_null,
)

LABEL_T = 5  # close D+1 to close D+5: four sessions, three shared by neighbours
Z_SUM = 1.959964 + 0.841621  # two-sided 5%, 80% power


def _ma3(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    """A daily series of four-session sums of iid shocks: MA(3), long-run sd 4 * scale."""
    return np.convolve(rng.standard_normal(n + 3) * scale, np.ones(4), "valid")


def test_paired_hac_test_has_nominal_size_on_overlapping_days() -> None:
    """r3: Bartlett lag 4 gave SE 0.85x truth and 10% rejections at a nominal 5%."""
    rng = np.random.default_rng(2)
    n, reps = 238, 1500
    t_stats, standard_errors = [], []
    for _ in range(reps):
        result = paired_mean_inference(
            _ma3(rng, n),
            arm="a",
            control="c",
            label_horizon=LABEL_T,
            n_resamples=1,
            seed=0,
            ci_level=0.95,
        )
        t_stats.append(result.hac_t)
        standard_errors.append(result.hac_se)
    true_se = 4.0 / math.sqrt(n)
    assert 0.95 < np.mean(standard_errors) / true_se < 1.03
    assert np.mean(np.abs(np.asarray(t_stats)) > 1.96) < 0.075


def test_power_tables_use_the_long_run_sd() -> None:
    """r2: the iid sd gave an MDE of 0.0027 against a Monte Carlo truth of 0.0073."""
    rng = np.random.default_rng(1)
    n = 238
    true_mde = Z_SUM * np.std([_ma3(rng, n, scale=0.01).mean() for _ in range(5000)])
    long_run_mdes, iid_mdes, days = [], [], []
    for _ in range(300):
        result = paired_mean_inference(
            _ma3(rng, n, scale=0.01),
            arm="a",
            control="c",
            label_horizon=LABEL_T,
            n_resamples=1,
            seed=0,
            ci_level=0.95,
        )
        assert result.long_run_sd is not None
        long_run_mdes.append(minimum_detectable_effect(result.long_run_sd, n))
        iid_mdes.append(minimum_detectable_effect(result.sd_delta, n))
        days.append(required_days(result.long_run_sd, true_mde))

    assert abs(np.mean(long_run_mdes) / true_mde - 1.0) < 0.1
    assert abs(np.median(days) / n - 1.0) < 0.2
    # The iid sd the notebooks used to pass understates it by about half.
    assert np.mean(iid_mdes) < 0.6 * true_mde


def test_sharpe_of_daily_sampled_overlapping_returns_is_the_daily_sharpe() -> None:
    """Findings 1: uniform weights on 3 lags recover the daily book's annual Sharpe."""
    rng = np.random.default_rng(4)
    mu, sigma, n = 0.0005, 0.01, 200_000
    daily = rng.normal(mu, sigma, size=n + 3)
    overlapping = np.convolve(daily, np.ones(4), "valid")
    result = sharpe_block_bootstrap_ci(
        overlapping, hac_lags=3, block_size=5, n_resamples=1, seed=0, ci_level=0.95
    )
    truth = mu / sigma * math.sqrt(252)
    assert abs(result["point"] / truth - 1.0) < 0.12
    assert result["hac_lags"] == 3


# ── permutation null ──────────────────────────────────────────────────────

N_NAMES, N_DATES = 30, 60
KEYS = tuple(f"S{i:03d}" for i in range(N_NAMES))
DATES = [str(day.date()) for day in pd.bdate_range("2024-01-01", periods=N_DATES)]


def _persistent_no_skill(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """r4: a persistent score with no skill and four-session overlapping outcomes."""
    base = rng.standard_normal(N_NAMES)
    scores = base + 0.2 * rng.standard_normal((N_DATES, N_NAMES))
    shocks = rng.standard_normal((N_DATES + 4, N_NAMES))
    outcomes = shocks[1:-3] + shocks[2:-2] + shocks[3:-1] + shocks[4:]
    return scores, outcomes


def _mean_rank_ic(scores: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Mean daily Spearman IC over the date axis, vectorised over leading axes."""
    score_ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1).astype(float)
    outcome_ranks = np.argsort(np.argsort(outcomes, axis=-1), axis=-1).astype(float)
    score_ranks -= score_ranks.mean(axis=-1, keepdims=True)
    outcome_ranks -= outcome_ranks.mean(axis=-1, keepdims=True)
    daily = (score_ranks * outcome_ranks).sum(-1) / np.sqrt(
        (score_ranks**2).sum(-1) * (outcome_ranks**2).sum(-1)
    )
    return daily.mean(axis=-1)


def test_date_blocked_null_matches_the_sampling_spread_of_a_persistent_score() -> None:
    """r4: the per-date shuffle's null sd was 0.024 against a true 0.045."""
    truth_rng = np.random.default_rng(30)
    draws = [_persistent_no_skill(truth_rng) for _ in range(3000)]
    true_sd = float(
        np.std(_mean_rank_ic(np.stack([s for s, _ in draws]), np.stack([y for _, y in draws])))
    )

    rng = np.random.default_rng(3)
    null_sds = []
    for rep in range(8):
        scores, outcomes = _persistent_no_skill(rng)
        sections = [
            DatedScoreOutcome(DATES[d], KEYS, tuple(scores[d]), tuple(outcomes[d]))
            for d in range(N_DATES)
        ]
        null = run_date_blocked_permutation_null(sections, n_draws=1000, null_seed=rep, top_k=5)
        null_sds.append(np.std(null.rank_ic_means))

    assert 0.85 < np.mean(null_sds) / true_sd < 1.15
    assert null.block_dates is None and null.block_count == 1


def test_date_blocked_null_is_deterministic_and_keyed_by_block() -> None:
    rng = np.random.default_rng(5)
    scores, outcomes = _persistent_no_skill(rng)
    sections = [
        DatedScoreOutcome(DATES[d], KEYS, tuple(scores[d]), tuple(outcomes[d])) for d in range(4)
    ]
    first = run_date_blocked_permutation_null(sections, n_draws=1000, null_seed=9, top_k=5)
    again = run_date_blocked_permutation_null(sections, n_draws=1000, null_seed=9, top_k=5)
    blocked = run_date_blocked_permutation_null(
        sections, n_draws=1000, null_seed=9, top_k=5, block_dates=2
    )
    assert first == again
    assert blocked.block_count == 2 and blocked.assignment_digest != first.assignment_digest


def test_date_blocked_null_induces_a_bijection_when_names_come_and_go() -> None:
    sigma = _block_permutation(KEYS, null_seed=1, draw_id=0, block_id=0)
    assert np.array_equal(_induced_sources(KEYS, sigma, KEYS), sigma)  # full set: itself
    present = tuple(key for i, key in enumerate(KEYS) if i % 4)  # a quarter absent
    source = _induced_sources(KEYS, sigma, present)
    assert sorted(source.tolist()) == list(range(len(present)))  # still a permutation

    rng = np.random.default_rng(6)
    sections = []
    for d in range(6):
        keys = KEYS[: N_NAMES - (d % 3)]  # the last two names drop out on some dates
        sections.append(
            DatedScoreOutcome(
                DATES[d],
                keys,
                tuple(rng.standard_normal(len(keys))),
                tuple(rng.standard_normal(len(keys))),
            )
        )
    null = run_date_blocked_permutation_null(sections, n_draws=1000, null_seed=1, top_k=3)
    assert null.valid_draw_count == 1000
    assert all(-1.0 <= value <= 1.0 for value in null.rank_ic_means)


# ── deflated Sharpe on the daily book ─────────────────────────────────────

BOOK_NAMES = [f"B{i:02d}" for i in range(20)]
BOOK_SIGNALS, BOOK_K = 300, 5


def _random_score_market(
    rng: np.random.Generator, drift: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """iid daily returns for 20 names and skill-free scores on 300 signal dates."""
    dates = pd.bdate_range("2023-01-02", periods=BOOK_SIGNALS + 6).strftime("%Y-%m-%d")
    daily = rng.normal(drift, 0.01, size=(len(dates), len(BOOK_NAMES)))
    closes = 100.0 * np.cumprod(1.0 + daily, axis=0)
    market = pd.DataFrame(
        {
            "dt": np.repeat(dates, len(BOOK_NAMES)),
            "kdcode": np.tile(BOOK_NAMES, len(dates)),
            "close": closes.ravel(),
        }
    )
    predictions = pd.DataFrame(
        {
            "dt": np.repeat(dates[:BOOK_SIGNALS], len(BOOK_NAMES)),
            "kdcode": np.tile(BOOK_NAMES, BOOK_SIGNALS),
            "score": rng.standard_normal(BOOK_SIGNALS * len(BOOK_NAMES)),
        }
    )
    return predictions, market


def test_selection_audit_deflates_the_daily_book_sharpe(tmp_path) -> None:
    """r5: the overlapping label series gave about twice the Sharpe and z-stat."""
    predictions, market = _random_score_market(np.random.default_rng(12), drift=0.0005)
    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    predictions.to_csv(predictions_dir / "predictions.csv", index=False)
    market.to_csv(tmp_path / "market.csv", index=False)

    audit = build_selection_audit(
        predictions_dir=predictions_dir,
        market_data_path=tmp_path / "market.csv",
        label_t=LABEL_T,
        top_k_values=[BOOK_K],
        trial_count=1,
        bootstrap_resamples=5,
    )
    deflated = audit["deflated_sharpe"][str(BOOK_K)]

    book = _daily_books(predictions, market, label_t=LABEL_T, top_k_values=[BOOK_K])[BOOK_K]
    daily = book.daily_returns
    assert deflated["return_basis"] == "daily_tranche_book" and deflated["tranches"] == 4
    assert deflated["n_obs"] == daily.size == BOOK_SIGNALS + LABEL_T - 2
    assert deflated["annualized_sharpe"] == pytest.approx(
        np.mean(daily) / np.std(daily, ddof=1) * math.sqrt(252)
    )


def test_deflated_sharpe_has_nominal_size_on_the_daily_book() -> None:
    """r5: p <= 0.05 in 21% of no-skill samples on the overlapping label series."""
    rng = np.random.default_rng(13)
    p_values = []
    for _ in range(120):
        predictions, market = _random_score_market(rng, drift=0.0)
        book = _daily_books(predictions, market, label_t=LABEL_T, top_k_values=[BOOK_K])[BOOK_K]
        p_values.append(deflated_sharpe_ratio(book.daily_returns, trial_count=1)["p_value"])
    assert np.mean(np.asarray(p_values) <= 0.05) < 0.1
