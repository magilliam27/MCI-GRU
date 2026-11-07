"""Walk-forward evaluation pools the windows' dated test series (#342).

The merged summary used to average each evaluation key across windows: Sharpes and
IRs as ratios, CI bounds as if they were an interval, cumulative returns, and a
short final window weighted as much as a full one (math side finding C1, ``r1``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation.experiment_summary import compute_evaluation_summary
from mci_gru.evaluation.outcomes import realized_returns
from mci_gru.walkforward import (
    generate_walkforward_configs,
    pool_window_test_sets,
    pooled_evaluation_summary,
    window_test_set,
)

LABEL_T = 5
NAMES = [f"N{i:02d}" for i in range(30)]


def _market(rng: np.random.Generator, sessions: list[str]) -> pd.DataFrame:
    daily = rng.normal(0.0003, 0.01, size=(len(sessions), len(NAMES)))
    closes = 100.0 * np.cumprod(1.0 + daily, axis=0)
    return pd.DataFrame(
        {
            "dt": np.repeat(sessions, len(NAMES)),
            "kdcode": np.tile(NAMES, len(sessions)),
            "close": closes.ravel(),
        }
    )


def _window(rng, market, dates, kdcodes, skill, *, pit_mask=True):
    """Scores with the given skill on the core's booked returns; labels are those returns."""
    labels = realized_returns(market, dates, kdcodes=kdcodes).returns
    noise = rng.standard_normal(labels.shape)
    scores = skill * np.nan_to_num(labels) / 0.02 + noise
    return window_test_set(
        scores,
        labels,
        panel=market,
        dates=dates,
        kdcodes=kdcodes,
        rankable_mask=np.ones(labels.shape, dtype=bool) if pit_mask else None,
        label_t=LABEL_T,
    )


def test_pooled_summary_is_the_summary_of_every_test_date_together() -> None:
    """r1: 63 + 63 + 10-date windows; the mean of windows moved avg_ic 0.076 to 0.029."""
    rng = np.random.default_rng(0)
    sessions = pd.bdate_range("2024-01-01", periods=400).strftime("%Y-%m-%d").tolist()
    market = _market(rng, sessions)
    spans = [(10, 73, 0.1), (140, 203, 0.1), (270, 280, -0.15)]
    windows = [_window(rng, market, sessions[a:b], NAMES, skill) for a, b, skill in spans]

    pooled = pool_window_test_sets(windows)
    config = ExperimentConfig()
    summary = compute_evaluation_summary(
        pooled.predictions,
        pooled.labels,
        config,
        panel=pooled.closes,
        dates=pooled.dates,
        kdcodes=pooled.kdcodes,
        rankable_mask=pooled.rankable_mask,
    )
    per_window = [
        compute_evaluation_summary(
            w.predictions,
            w.labels,
            config,
            panel=w.closes,
            dates=w.dates,
            kdcodes=w.kdcodes,
            rankable_mask=w.rankable_mask,
        )["metrics"]
        for w in windows
    ]

    daily_ic = [
        np.corrcoef(p[np.isfinite(r)], r[np.isfinite(r)])[0, 1]
        for w in windows
        for p, r in zip(w.predictions, w.labels, strict=True)
    ]
    assert summary["metrics"]["avg_ic"] == pytest.approx(np.mean(daily_ic))
    assert summary["metrics"]["test_dates"] == 63 + 63 + 10
    mean_of_windows = np.mean([m["avg_ic"] for m in per_window])
    assert abs(summary["metrics"]["avg_ic"] - mean_of_windows) > 0.01  # the short window


def test_pooling_widens_to_the_union_of_names() -> None:
    rng = np.random.default_rng(1)
    sessions = pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d").tolist()
    market = _market(rng, sessions)
    first = _window(rng, market, sessions[0:20], NAMES[:20], 0.1)
    second = _window(rng, market, sessions[40:60], NAMES[10:], 0.1)

    pooled = pool_window_test_sets([second, first])

    assert pooled.kdcodes == NAMES
    assert pooled.dates == sessions[0:20] + sessions[40:60]
    assert np.isnan(pooled.predictions[:20, 20:]).all()  # names the first window lacked
    assert not pooled.rankable_mask[:20, 20:].any()
    assert np.array_equal(pooled.predictions[20:, 10:], second.predictions)


def test_overlapping_test_windows_are_not_pooled() -> None:
    rng = np.random.default_rng(2)
    sessions = pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    market = _market(rng, sessions)
    windows = [
        _window(rng, market, sessions[0:30], NAMES, 0.1),
        _window(rng, market, sessions[20:50], NAMES, 0.1),
    ]
    with pytest.raises(ValueError, match="overlap"):
        pool_window_test_sets(windows)


def test_one_pooled_window_books_what_the_full_panel_books() -> None:
    """The trimmed closes keep every exit: the book, not only IC, is unchanged."""
    rng = np.random.default_rng(3)
    sessions = pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d").tolist()
    market = _market(rng, sessions)
    window = _window(rng, market, sessions[20:80], NAMES, 0.1)
    config = ExperimentConfig()

    pooled = pooled_evaluation_summary(pool_window_test_sets([window]), config)["metrics"]
    direct = compute_evaluation_summary(
        window.predictions,
        window.labels,
        config,
        panel=market,
        dates=window.dates,
        kdcodes=window.kdcodes,
        rankable_mask=window.rankable_mask,
    )["metrics"]

    assert np.isfinite(direct["top_20_abs_sharpe"])
    for key, value in direct.items():
        if isinstance(value, float) and np.isfinite(value):
            assert pooled[key] == pytest.approx(value), key


def test_without_pit_a_window_ranks_only_its_own_names() -> None:
    """The first window's tail closes must not make its names rankable in the next."""
    rng = np.random.default_rng(4)
    sessions = pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d").tolist()
    market = _market(rng, sessions)
    first = _window(rng, market, sessions[0:40], NAMES, 0.1, pit_mask=False)
    second = _window(rng, market, sessions[40:80], NAMES[10:], 0.1, pit_mask=False)

    pooled = pool_window_test_sets([first, second])
    summary = pooled_evaluation_summary(pooled, ExperimentConfig())

    assert pooled.rankable_source == "SIGNAL_DATE_CLOSES"
    assert pooled.rankable_mask[:40].all()
    assert not pooled.rankable_mask[40:, :10].any()  # closes exist, but not its names
    assert pooled.rankable_mask[40:, 10:].all()
    assert summary["benchmark"] == "equal_weight_rankable_universe"
    # Left to rebuild the set from the pooled closes, the universe gains those names.
    rebuilt = compute_evaluation_summary(
        pooled.predictions,
        pooled.labels,
        ExperimentConfig(),
        panel=pooled.closes,
        dates=pooled.dates,
        kdcodes=pooled.kdcodes,
        rankable_mask=None,
    )["metrics"]
    universe = "universe_abs_period_return"
    assert summary["metrics"][universe] != pytest.approx(rebuilt[universe], abs=1e-9)


def test_windows_stepped_by_their_span_share_no_test_date() -> None:
    """Review of #342: an inclusive test_end was the next window's test_start."""
    base = ExperimentConfig()
    base = replace(
        base,
        data=replace(
            base.data, train_start="2016-01-04", test_end="2025-12-31", use_pit_universe=False
        ),
        training=replace(
            base.training,
            walkforward=replace(
                base.training.walkforward, enabled=True, test_span_months=3, step_months=3
            ),
        ),
    )
    windows = generate_walkforward_configs(base)
    assert len(windows) > 4
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert earlier.data.test_end < later.data.test_start
