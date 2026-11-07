"""LightGBM on price and volume features, a candidate arm scored on a run's own axis (#387).

Idea 2 of the 2026-11-07 model survey. Gradient-boosted trees on an Alpha158-style set
of price and volume features, ranked within each date, fitted on the run's training
dates and stopped early on the run's validation dates. The shared candidate harness
(#384) scores the result through ``compute_evaluation_summary`` against the run's own
labels, dates, names and tradable mask, as it scores every other arm.

Information discipline, which the tests pin:

* **Features** at signal date ``D`` read closes and volumes at sessions up to and
  including ``D`` only. Sessions are the panel's own trading dates
  (``label_session_axis``), as labels count them, so a missing bar never shifts a
  window: a rolling window that holds a missing bar gives NaN, which LightGBM routes as
  missing.
* **The 5-day arm** fits on the run's own training labels (already purged so that none
  matures inside validation) and its own training loss mask.
* **The 20-day arm** fits on ``close[D+20] / close[D+1] - 1``, the run's label
  definition at a 20-session horizon, and drops every training date whose 20-day label
  exits on or after the first validation date.
* **Model selection** (a four-entry grid times six boosting-round checkpoints) reads
  only validation dates, scored by the mean daily Pearson IC of the seed ensemble
  against the run's own validation labels, which the pipeline already purged of any
  label that reads a test close.
* Test dates are only predicted.

Each input is converted to a within-date percentile, among the stocks the split's mask
admits that date, so a feature's level drifting over the years (volatility regimes,
volume growth) cannot be learned as signal. The target is ranked the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from mci_gru.data.preprocessing import label_session_axis, resolve_label_endpoints, resolve_labels
from mci_gru.evaluation.candidates import CandidateResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mci_gru.evaluation.candidates import CandidateContext

#: Horizons of the two arms, in sessions.
TARGET_HORIZONS = {"run_label": None, "forward_20d": 20}

RETURN_LAGS = (1, 5, 10, 20, 60)
MA_WINDOWS = (5, 10, 20, 60)
VOL_WINDOWS = (5, 20, 60)
RANGE_WINDOWS = (20, 60)
VOLUME_WINDOWS = (5, 20, 60)
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21

#: Fixed settings shared by every fit. Validation on all five windows (read before any
#: test score) showed boosters overfitting within tens of rounds at ordinary settings, so
#: the trees are shallow, slow, and see half the inputs per tree.
BASE_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.01,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
    "deterministic": True,
    "force_row_wise": True,
}
#: The small grid; each entry overrides ``BASE_PARAMS``.
PARAM_GRID: tuple[dict[str, Any], ...] = (
    {"num_leaves": 3, "min_data_in_leaf": 5000},
    {"num_leaves": 7, "min_data_in_leaf": 2000},
    {"num_leaves": 15, "min_data_in_leaf": 2000},
    {"num_leaves": 31, "min_data_in_leaf": 1000},
)
#: Boosting rounds at which the seed ensemble is scored on validation.
ROUND_CHECKPOINTS = (25, 50, 100, 200, 400, 800)
DEFAULT_SEEDS = tuple(range(10))
#: A validation date needs at least this many scored names to count toward its IC.
MIN_NAMES_PER_DATE = 10


@dataclass(frozen=True)
class FeatureFrame:
    """Per-date, per-stock inputs on one split's dates."""

    names: list[str]
    values: np.ndarray  # (dates, stocks, features), float64, NaN where undefined


@dataclass(frozen=True)
class LightGBMFit:
    """Scores on the test dates and what was chosen to make them."""

    scores: np.ndarray  # (test dates, stocks); NaN where no score
    controls: dict[str, Any] = field(default_factory=dict)


def _wide(panel: pd.DataFrame, kdcodes: list[str], column: str) -> pd.DataFrame:
    rows = panel.loc[panel["kdcode"].astype(str).isin(kdcodes), ["kdcode", "dt", column]].copy()
    rows["dt"] = rows["dt"].astype(str)
    rows["kdcode"] = rows["kdcode"].astype(str)
    sessions = label_session_axis(rows, kdcodes)
    wide = rows.pivot_table(index="dt", columns="kdcode", values=column, aggfunc="mean")
    return wide.reindex(index=sessions, columns=kdcodes).astype(np.float64)


def compute_feature_panel(panel: pd.DataFrame, kdcodes: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Every feature on the panel's full session axis, one wide frame per feature.

    Row ``D`` of every frame reads sessions up to and including ``D`` only.
    """
    kdcodes = [str(k) for k in kdcodes]
    close = _wide(panel, kdcodes, "close")
    volume = _wide(panel, kdcodes, "volume")
    close = close.where(close > 0)
    volume = volume.where(volume >= 0)
    ret = close / close.shift(1) - 1.0
    log_volume = np.log1p(volume)
    features: dict[str, pd.DataFrame] = {}
    for lag in RETURN_LAGS:
        features[f"ret_{lag}"] = close / close.shift(lag) - 1.0
    features["mom_252_21"] = close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1.0
    for window in MA_WINDOWS:
        features[f"ma_ratio_{window}"] = close / close.rolling(window).mean() - 1.0
    for window in VOL_WINDOWS:
        features[f"vol_{window}"] = ret.rolling(window).std()
    features["max_ret_20"] = ret.rolling(20).max()
    features["min_ret_20"] = ret.rolling(20).min()
    features["skew_60"] = ret.rolling(60).skew()
    for window in RANGE_WINDOWS:
        features[f"off_high_{window}"] = close / close.rolling(window).max() - 1.0
        features[f"off_low_{window}"] = close / close.rolling(window).min() - 1.0
    for window in VOLUME_WINDOWS:
        features[f"volume_ratio_{window}"] = log_volume - log_volume.rolling(window).mean()
    features["log_dollar_volume_20"] = np.log1p(close * volume).rolling(20).mean()
    features["pv_corr_20"] = ret.rolling(20).corr(log_volume.diff())
    return {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in features.items()}


def cross_sectional_percentile(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Within-date percentile in (0, 1) of every finite cell the mask admits; NaN elsewhere.

    ``values`` is ``(dates, stocks)``. Ties share their average rank.
    """
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    frame = pd.DataFrame(np.where(keep, values, np.nan))
    ranks = frame.rank(axis=1, method="average")
    counts = keep.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ranks.to_numpy() - 0.5) / counts
    return np.where(keep, out, np.nan)


def features_on_dates(
    feature_panel: dict[str, pd.DataFrame],
    dates: Sequence[str],
    mask: np.ndarray | None = None,
) -> FeatureFrame:
    """The features on ``dates``, each ranked within its date among the admitted stocks."""
    dates = [str(d) for d in dates]
    names = list(feature_panel)
    stacked = []
    for name in names:
        frame = feature_panel[name]
        missing = [d for d in dates if d not in frame.index]
        if missing:
            raise ValueError(f"{len(missing)} date(s) are not on the panel's axis: {missing[:3]}")
        stacked.append(cross_sectional_percentile(frame.loc[dates].to_numpy(), mask))
    return FeatureFrame(names, np.stack(stacked, axis=-1))


def forward_labels_before(
    panel: pd.DataFrame,
    kdcodes: Sequence[str],
    dates: Sequence[str],
    horizon: int,
    boundary: str,
) -> tuple[list[str], np.ndarray]:
    """``close[D+horizon] / close[D+1] - 1`` on the dates whose exit is before ``boundary``.

    Exits count sessions on the panel's own axis, as the run's labels do. A date whose
    exit is unknown (off the end of the panel) is dropped too.
    """
    kdcodes = [str(k) for k in kdcodes]
    dates = [str(d) for d in dates]
    sessions = label_session_axis(panel, kdcodes)
    _entries, exits = resolve_label_endpoints(sessions, dates, horizon)
    kept = [d for d, e in zip(dates, exits, strict=True) if e is not None and e < boundary]
    if not kept:
        return [], np.empty((0, len(kdcodes)))
    returns = resolve_labels(panel, kdcodes, kept, horizon).returns
    return kept, np.where(np.isfinite(returns), returns, np.nan)


def daily_ic(predictions: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Mean over dates of the Pearson correlation of ``predictions`` and ``labels``.

    ``groups`` gives each row's date index; dates with fewer than
    ``MIN_NAMES_PER_DATE`` rows or no spread are left out.
    """
    order = np.argsort(groups, kind="stable")
    p, y, g = predictions[order], labels[order], groups[order]
    bounds = np.flatnonzero(np.diff(g)) + 1
    ics = []
    for ps, ys in zip(np.split(p, bounds), np.split(y, bounds), strict=True):
        if len(ps) < MIN_NAMES_PER_DATE or ps.std() == 0 or ys.std() == 0:
            continue
        ics.append(float(np.corrcoef(ps, ys)[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


@dataclass(frozen=True)
class _Rows:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    n_dates: int


def _rows(features: FeatureFrame, target: np.ndarray, mask: np.ndarray | None) -> _Rows:
    keep = np.isfinite(target)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    keep &= np.isfinite(features.values).any(axis=-1)
    date_index, stock_index = np.nonzero(keep)
    return _Rows(
        features.values[date_index, stock_index],
        target[date_index, stock_index],
        date_index,
        int(np.unique(date_index).size),
    )


def _train(params: dict[str, Any], train: _Rows, seed: int) -> lgb.Booster:
    full = {**BASE_PARAMS, **params, "seed": seed}
    return lgb.train(full, lgb.Dataset(train.x, train.y), num_boost_round=max(ROUND_CHECKPOINTS))


def _zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Each date's finite scores to mean 0 and standard deviation 1; NaN stays NaN."""
    out = np.full(scores.shape, np.nan)
    for i, row in enumerate(scores):
        finite = np.isfinite(row)
        if finite.sum() > 1 and row[finite].std() > 0:
            out[i, finite] = (row[finite] - row[finite].mean()) / row[finite].std()
        elif finite.any():
            out[i, finite] = 0.0
    return out


def _zscore_groups(predictions: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros_like(predictions, dtype=np.float64)
    for g in np.unique(groups):
        rows = groups == g
        p = predictions[rows]
        out[rows] = (p - p.mean()) / p.std() if p.std() > 0 else 0.0
    return out


def fit_lightgbm_candidate(
    panel: pd.DataFrame,
    kdcodes: Sequence[str],
    *,
    train_dates: Sequence[str],
    train_labels: np.ndarray,
    train_mask: np.ndarray | None,
    val_dates: Sequence[str],
    val_labels: np.ndarray,
    val_mask: np.ndarray | None,
    test_dates: Sequence[str],
    test_mask: np.ndarray | None = None,
    target: str = "run_label",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    param_grid: Sequence[dict[str, Any]] = PARAM_GRID,
) -> LightGBMFit:
    """Fit on training dates, select on validation IC, score the test dates.

    ``train_labels`` and ``val_labels`` are the run's own (dates, stocks) labels on
    ``train_dates`` and ``val_dates``. With ``target="forward_20d"`` the training labels
    are replaced by 20-session forward returns, and training dates whose 20-session
    label would exit on or after the first validation date are dropped. Validation is
    always scored against the run's own validation labels.
    """
    if target not in TARGET_HORIZONS:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(TARGET_HORIZONS)}")
    kdcodes = [str(k) for k in kdcodes]
    train_dates = [str(d) for d in train_dates]
    val_dates = [str(d) for d in val_dates]
    test_dates = [str(d) for d in test_dates]
    if not (train_dates and val_dates and test_dates):
        raise ValueError("train, validation and test dates must all be non-empty")
    if not train_dates[-1] < val_dates[0] <= val_dates[-1] < test_dates[0]:
        raise ValueError("splits must be ordered: train < validation < test")
    train_labels = np.asarray(train_labels, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.float64)
    if train_labels.shape != (len(train_dates), len(kdcodes)):
        raise ValueError(f"train_labels shape {train_labels.shape} is not (dates, stocks)")
    if val_labels.shape != (len(val_dates), len(kdcodes)):
        raise ValueError(f"val_labels shape {val_labels.shape} is not (dates, stocks)")

    # Nothing after the last test date is ever read: features look back only, and the
    # 20-day target reads only closes before the first validation date.
    dt = panel["dt"].astype(str)
    visible = panel.loc[dt <= test_dates[-1]]
    feature_panel = compute_feature_panel(visible, kdcodes)

    horizon = TARGET_HORIZONS[target]
    fit_dates, fit_labels, fit_mask = train_dates, train_labels, train_mask
    if horizon is not None:
        fit_dates, fit_labels = forward_labels_before(
            visible.loc[visible["dt"].astype(str) < val_dates[0]],
            kdcodes,
            train_dates,
            horizon,
            val_dates[0],
        )
        if not fit_dates:
            raise ValueError(f"no training date has a {horizon}-session label before validation")
        if train_mask is not None:
            position = {d: i for i, d in enumerate(train_dates)}
            fit_mask = np.asarray(train_mask, dtype=bool)[[position[d] for d in fit_dates]]
            fit_labels = np.where(fit_mask, fit_labels, np.nan)

    train_features = features_on_dates(feature_panel, fit_dates, fit_mask)
    val_features = features_on_dates(feature_panel, val_dates, val_mask)
    test_features = features_on_dates(feature_panel, test_dates, test_mask)
    train_rows = _rows(train_features, cross_sectional_percentile(fit_labels, fit_mask), fit_mask)
    val_rows = _rows(val_features, val_labels, val_mask)
    if train_rows.y.size == 0 or val_rows.y.size == 0:
        raise ValueError("no training or validation rows survive the masks")

    flat_test = test_features.values.reshape(-1, len(test_features.names))
    has_input = np.isfinite(flat_test).any(axis=1)
    test_mask_array = None if test_mask is None else np.asarray(test_mask, dtype=bool)

    # Every grid entry trains every seed once; the seed ensemble is scored on validation
    # at each checkpoint, and the best (entry, checkpoint) on validation IC is the arm.
    grid: list[dict[str, Any]] = []
    best: tuple[float, int, int] | None = None
    boosters: list[list[lgb.Booster]] = []
    for g, params in enumerate(param_grid):
        trained = [_train(params, train_rows, int(seed)) for seed in seeds]
        boosters.append(trained)
        row: dict[str, Any] = {**params, "validation_ic": {}}
        for rounds in ROUND_CHECKPOINTS:
            ensemble = np.mean(
                [
                    _zscore_groups(b.predict(val_rows.x, num_iteration=rounds), val_rows.groups)
                    for b in trained
                ],
                axis=0,
            )
            ic = daily_ic(ensemble, val_rows.y, val_rows.groups)
            row["validation_ic"][rounds] = ic
            if np.isfinite(ic) and (best is None or ic > best[0]):
                best = (ic, g, rounds)
        grid.append(row)
    if best is None:
        raise ValueError("no grid entry gave a finite validation IC")
    best_ic, chosen_index, chosen_rounds = best

    per_seed = []
    for booster in boosters[chosen_index]:
        predictions = np.full(flat_test.shape[0], np.nan)
        predictions[has_input] = booster.predict(flat_test[has_input], num_iteration=chosen_rounds)
        scores = predictions.reshape(len(test_dates), len(kdcodes))
        if test_mask_array is not None:
            scores = np.where(test_mask_array, scores, np.nan)
        per_seed.append(_zscore_rows(scores))
    ensemble = np.mean(np.stack(per_seed), axis=0)  # NaN wherever any seed has none
    booster = boosters[chosen_index][0]

    importance = booster.feature_importance(importance_type="gain", iteration=chosen_rounds)
    controls = {
        "target": target,
        "target_horizon_sessions": horizon,
        "features": train_features.names,
        "base_params": {k: v for k, v in BASE_PARAMS.items() if k != "num_threads"},
        "grid": grid,
        "round_checkpoints": list(ROUND_CHECKPOINTS),
        "chosen_params": dict(param_grid[chosen_index]),
        "chosen_rounds": chosen_rounds,
        "chosen_validation_ic": best_ic,
        "seeds": [int(seed) for seed in seeds],
        "train_rows": int(train_rows.y.size),
        "train_dates": train_rows.n_dates,
        "train_date_range": [fit_dates[0], fit_dates[-1]],
        "validation_rows": int(val_rows.y.size),
        "validation_dates": val_rows.n_dates,
        "validation_date_range": [val_dates[0], val_dates[-1]],
        "first_seed_gain_importance": {
            name: float(value)
            for name, value in sorted(
                zip(train_features.names, importance, strict=True), key=lambda kv: -kv[1]
            )
        },
        "lightgbm_version": lgb.__version__,
    }
    return LightGBMFit(ensemble, controls)


def _candidate(context: CandidateContext, target: str) -> CandidateResult:
    data = context.data
    fit = fit_lightgbm_candidate(
        context.panel,
        context.kdcodes,
        train_dates=context.train_dates,
        train_labels=data["train_labels"],
        train_mask=data.get("train_loss_mask"),
        val_dates=context.val_dates,
        val_labels=data["val_labels"],
        val_mask=data.get("val_tradable_mask"),
        test_dates=context.test_dates,
        test_mask=context.rankable_mask,
        target=target,
        seeds=DEFAULT_SEEDS,
        param_grid=PARAM_GRID,
    )
    return CandidateResult(scores=fit.scores, controls=fit.controls)


def lightgbm_5d(context: CandidateContext) -> CandidateResult:
    """LightGBM fitted to the run's own 5-session training label."""
    return _candidate(context, "run_label")


def lightgbm_20d(context: CandidateContext) -> CandidateResult:
    """LightGBM fitted to the 20-session forward return, scored on the run's 5-session label."""
    return _candidate(context, "forward_20d")
