"""The LightGBM candidate reads only what each split may read (#387).

``mci_gru.evaluation.lightgbm_candidate`` builds price and volume features, fits
gradient-boosted trees on a run's training dates, selects on its validation dates and
scores its test dates. These tests pin that:

* every feature at signal date ``D`` is unchanged by any bar after ``D``, and a feature
  is the stated ratio of closes counted in sessions;
* within-date percentiles live in (0, 1), respect the mask and share ties;
* the 20-day target is ``close[D+20] / close[D+1] - 1`` and keeps only training dates
  whose label exits before the first validation date, so no validation or test close
  can reach a training label;
* the fitted arm recovers a planted signal, scores only cells the test mask admits, is
  unchanged by any bar after the last test date, and is unchanged by test-period closes
  except through the test features themselves;
* malformed splits are refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation import lightgbm_candidate as lc

N_STOCKS = 40
N_SESSIONS = 420
KDCODES = [f"S{j:02d}.N" for j in range(N_STOCKS)]


def _panel(seed: int = 0, signal: float = 0.0) -> pd.DataFrame:
    """A random-walk panel; with ``signal`` the next 5 sessions partly reverse the last 5."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=N_SESSIONS).strftime("%Y-%m-%d")
    returns = rng.normal(0.0, 0.02, (N_SESSIONS, N_STOCKS))
    if signal:
        for t in range(10, N_SESSIONS):
            past = returns[t - 5 : t].sum(axis=0)
            returns[t] -= signal * (past - past.mean()) / 5.0
    closes = 50.0 * np.cumprod(1.0 + returns, axis=0)
    volumes = rng.lognormal(13.0, 0.4, (N_SESSIONS, N_STOCKS))
    rows = {
        "kdcode": np.tile(KDCODES, N_SESSIONS),
        "dt": np.repeat(dates, N_STOCKS),
        "close": closes.ravel(),
        "volume": volumes.ravel(),
    }
    return pd.DataFrame(rows)


def _sessions(panel: pd.DataFrame) -> list[str]:
    return sorted(panel["dt"].unique())


def _splits(panel: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    sessions = _sessions(panel)
    # Train, a 5-session gap, validation, a 5-session gap, test; then 20 unused sessions.
    return sessions[70:250], sessions[255:320], sessions[325:400]


def _labels(panel: pd.DataFrame, dates: list[str], horizon: int = 5) -> np.ndarray:
    return lc.resolve_labels(panel, KDCODES, dates, horizon).returns


@pytest.fixture
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "ROUND_CHECKPOINTS", (20, 60))


def _fit(panel: pd.DataFrame, target: str = "run_label", **overrides) -> lc.LightGBMFit:
    train, val, test = _splits(panel)
    kwargs = dict(
        train_dates=train,
        train_labels=_labels(panel, train),
        train_mask=None,
        val_dates=val,
        val_labels=_labels(panel, val),
        val_mask=None,
        test_dates=test,
        target=target,
        seeds=(0, 1),
        param_grid=({"num_leaves": 3, "min_data_in_leaf": 200},),
    )
    kwargs.update(overrides)
    return lc.fit_lightgbm_candidate(panel, KDCODES, **kwargs)


def _daily_ic(scores: np.ndarray, labels: np.ndarray) -> float:
    ics = []
    for s, y in zip(scores, labels, strict=True):
        ok = np.isfinite(s) & np.isfinite(y)
        ics.append(np.corrcoef(s[ok], y[ok])[0, 1])
    return float(np.mean(ics))


# --- Features -------------------------------------------------------------------------


def test_features_at_a_date_ignore_every_later_bar():
    panel = _panel()
    sessions = _sessions(panel)
    cutoff = sessions[300]
    later = panel["dt"] > cutoff
    perturbed = panel.copy()
    rng = np.random.default_rng(9)
    perturbed.loc[later, "close"] *= rng.uniform(0.5, 1.5, int(later.sum()))
    perturbed.loc[later, "volume"] *= rng.uniform(0.1, 10.0, int(later.sum()))

    dates = sessions[250:301]
    before = lc.compute_feature_panel(panel, KDCODES)
    after = lc.compute_feature_panel(perturbed, KDCODES)

    for name in before:
        np.testing.assert_allclose(
            before[name].loc[dates].to_numpy(),
            after[name].loc[dates].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=name,
        )
    # The perturbation is real: the first later date's features do move.
    assert not np.allclose(
        before["ret_1"].loc[sessions[301]].to_numpy(), after["ret_1"].loc[sessions[301]].to_numpy()
    )


def test_return_features_are_ratios_of_closes_counted_in_sessions():
    panel = _panel()
    sessions = _sessions(panel)
    features = lc.compute_feature_panel(panel, KDCODES)
    close = panel.pivot(index="dt", columns="kdcode", values="close")[KDCODES]
    at = 300
    for lag in lc.RETURN_LAGS:
        expected = close.iloc[at] / close.iloc[at - lag] - 1.0
        np.testing.assert_allclose(features[f"ret_{lag}"].loc[sessions[at]], expected, rtol=1e-12)
    expected = close.iloc[at - lc.MOMENTUM_SKIP] / close.iloc[at - lc.MOMENTUM_LOOKBACK] - 1.0
    np.testing.assert_allclose(features["mom_252_21"].loc[sessions[at]], expected, rtol=1e-12)


def test_a_missing_bar_gives_nan_rather_than_a_shifted_window():
    panel = _panel()
    sessions = _sessions(panel)
    gap = sessions[295]
    holed = panel.loc[~((panel["kdcode"] == KDCODES[0]) & (panel["dt"] == gap))]
    features = lc.compute_feature_panel(holed, KDCODES)
    # The 5-session return at 300 reads close 295, which is missing for stock 0.
    assert np.isnan(features["ret_5"].loc[sessions[300], KDCODES[0]])
    assert np.isfinite(features["ret_5"].loc[sessions[300], KDCODES[1]])


def test_percentiles_respect_the_mask_and_share_ties():
    values = np.array([[3.0, 1.0, 2.0, np.nan], [5.0, 5.0, 1.0, 7.0]])
    mask = np.array([[True, True, True, True], [True, True, True, False]])

    got = lc.cross_sectional_percentile(values, mask)

    np.testing.assert_allclose(got[0, :3], [2.5 / 3, 0.5 / 3, 1.5 / 3])
    assert np.isnan(got[0, 3])
    np.testing.assert_allclose(got[1, :3], [2.0 / 3, 2.0 / 3, 0.5 / 3])
    assert np.isnan(got[1, 3])  # masked out, though finite
    assert np.nanmin(got) > 0 and np.nanmax(got) < 1


# --- 20-day target --------------------------------------------------------------------


def test_forward_labels_are_the_run_definition_and_stop_before_the_boundary():
    panel = _panel()
    sessions = _sessions(panel)
    train, val, _ = _splits(panel)

    kept, labels = lc.forward_labels_before(panel, KDCODES, train, 20, val[0])

    boundary = sessions.index(val[0])
    assert kept == [d for d in train if sessions.index(d) + 20 < boundary]
    assert len(kept) < len(train)
    close = panel.pivot(index="dt", columns="kdcode", values="close")[KDCODES]
    i = sessions.index(kept[-1])
    np.testing.assert_allclose(labels[-1], close.iloc[i + 20] / close.iloc[i + 1] - 1.0)


def test_twenty_day_training_never_reads_a_validation_or_test_close(small_grid, monkeypatch):
    panel = _panel(signal=0.5)
    _, val, _ = _splits(panel)
    later = panel["dt"] >= val[0]
    perturbed = panel.copy()
    perturbed.loc[later, "close"] *= np.random.default_rng(3).uniform(0.5, 1.5, int(later.sum()))
    perturbed.loc[later, "volume"] *= np.random.default_rng(4).uniform(0.1, 10.0, int(later.sum()))

    seen: list[tuple[np.ndarray, np.ndarray]] = []
    train = lc._train

    def spy(params, rows, seed):
        seen.append((rows.x.copy(), rows.y.copy()))
        return train(params, rows, seed)

    monkeypatch.setattr(lc, "_train", spy)
    _fit(panel, target="forward_20d")
    base = seen[0]
    seen.clear()
    _fit(perturbed, target="forward_20d", val_labels=_labels(panel, val))
    moved = seen[0]

    # Every training input and every training target is bit-identical.
    np.testing.assert_array_equal(base[0], moved[0])
    np.testing.assert_array_equal(base[1], moved[1])


# --- The fitted arm -------------------------------------------------------------------


def test_the_arm_recovers_a_planted_reversal(small_grid):
    panel = _panel(signal=0.6)
    _, _, test = _splits(panel)

    fit = _fit(panel)

    assert fit.scores.shape == (len(test), N_STOCKS)
    assert _daily_ic(fit.scores, _labels(panel, test)) > 0.1
    assert fit.controls["chosen_rounds"] in lc.ROUND_CHECKPOINTS


def test_the_twenty_day_arm_also_recovers_it(small_grid):
    panel = _panel(signal=0.6)
    _, _, test = _splits(panel)
    fit = _fit(panel, target="forward_20d")
    assert _daily_ic(fit.scores, _labels(panel, test)) > 0.05


def test_only_cells_the_test_mask_admits_are_scored(small_grid):
    panel = _panel(signal=0.6)
    _, _, test = _splits(panel)
    mask = np.random.default_rng(5).random((len(test), N_STOCKS)) > 0.3

    fit = _fit(panel, test_mask=mask)

    assert np.all(np.isnan(fit.scores[~mask]))
    assert np.all(np.isfinite(fit.scores[mask]))


def test_bars_after_the_last_test_date_change_nothing(small_grid):
    panel = _panel(signal=0.6)
    _, _, test = _splits(panel)
    later = panel["dt"] > test[-1]
    perturbed = panel.copy()
    perturbed.loc[later, "close"] *= 3.0
    perturbed.loc[later, "volume"] *= 0.01

    np.testing.assert_array_equal(_fit(panel).scores, _fit(perturbed).scores)


def test_validation_labels_steer_selection_only(small_grid):
    panel = _panel(signal=0.6)
    train, val, _ = _splits(panel)
    grid = ({"num_leaves": 3, "min_data_in_leaf": 200}, {"num_leaves": 7, "min_data_in_leaf": 200})

    fit = _fit(panel, param_grid=grid)

    record = fit.controls["grid"]
    chosen = max(
        ((g, r) for g, row in enumerate(record) for r in row["validation_ic"]),
        key=lambda gr: record[gr[0]]["validation_ic"][gr[1]],
    )
    assert fit.controls["chosen_params"] == grid[chosen[0]]
    assert fit.controls["chosen_rounds"] == chosen[1]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"target": "forward_60d"}, "unknown target"),
        ({"test_dates": []}, "non-empty"),
    ],
)
def test_malformed_requests_are_refused(change, message):
    panel = _panel()
    with pytest.raises(ValueError, match=message):
        _fit(panel, **change)


def test_overlapping_splits_are_refused():
    panel = _panel()
    train, val, test = _splits(panel)
    with pytest.raises(ValueError, match="ordered"):
        _fit(panel, test_dates=val[-5:] + test)
