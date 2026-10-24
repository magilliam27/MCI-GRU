"""Statistical evaluation helpers for cross-sectional prediction experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class DatedICObservation:
    """One dated cross-sectional IC result, including invalid-date state."""

    signal_dt: str
    daily_ic: float | None
    status: str


@dataclass(frozen=True)
class HACMeanInference:
    """Serializable Newey-West inference for the mean of a dated series."""

    method: str
    n_obs: int
    mean: float | None
    lags: int
    standard_error: float | None
    t_stat: float | None
    p_value: float | None


@dataclass(frozen=True)
class HACMeanInterval:
    """Serializable uniform-weight HAC interval for the mean of a dated series."""

    method: str
    n_obs: int
    lags: int
    df: float | None
    ci_level: float
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class MovingBlockMeanInterval:
    """Serializable circular moving-block interval for a dated mean."""

    method: str
    n_obs: int
    block_size: int
    n_resamples: int
    seed: int
    ci_level: float
    lower: float | None
    upper: float | None


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}")
    return arr


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values with averaged ranks for ties, zero-based."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def cross_sectional_ic(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    method: str = "pearson",
) -> float:
    """Compute a strict cross-sectional IC without partial-set filtering."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    preds = np.asarray(predictions, dtype=np.float64)
    rets = np.asarray(true_returns, dtype=np.float64)
    if preds.ndim != 1 or rets.ndim != 1:
        raise ValueError("predictions and true_returns must be 1-D")
    if preds.shape != rets.shape:
        raise ValueError(
            f"predictions and true_returns shapes differ: {preds.shape} != {rets.shape}"
        )
    if preds.size < 2 or not np.all(np.isfinite(preds)) or not np.all(np.isfinite(rets)):
        return float("nan")
    if np.ptp(preds) == 0 or np.ptp(rets) == 0:
        return float("nan")
    if method == "spearman":
        return _corr(_average_ranks(preds), _average_ranks(rets))
    return _corr(preds, rets)


def dated_daily_ic(
    dates: Sequence[str],
    predictions: Sequence[np.ndarray],
    true_returns: Sequence[np.ndarray],
    *,
    statuses: Sequence[str] | None = None,
    method: str = "spearman",
    valid_status: str = "VALID_PRIMARY",
) -> tuple[DatedICObservation, ...]:
    """Compute strict per-date IC while retaining every date and prior status."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    date_values = tuple(str(value) for value in dates)
    prediction_rows = tuple(predictions)
    return_rows = tuple(true_returns)
    status_values = (valid_status,) * len(date_values) if statuses is None else tuple(statuses)
    lengths = {
        len(date_values),
        len(prediction_rows),
        len(return_rows),
        len(status_values),
    }
    if len(lengths) != 1:
        raise ValueError("dates, predictions, true_returns, and statuses must have equal length")

    observations: list[DatedICObservation] = []
    for signal_dt, preds_raw, rets_raw, status in zip(
        date_values,
        prediction_rows,
        return_rows,
        status_values,
        strict=True,
    ):
        if status != valid_status:
            observations.append(DatedICObservation(signal_dt, None, status))
            continue

        preds = np.asarray(preds_raw, dtype=np.float64)
        rets = np.asarray(rets_raw, dtype=np.float64)
        if preds.ndim != 1 or rets.ndim != 1:
            raise ValueError("each predictions and true_returns row must be 1-D")
        if preds.shape != rets.shape:
            raise ValueError(
                f"predictions and true_returns row shapes differ for {signal_dt}: "
                f"{preds.shape} != {rets.shape}"
            )
        if preds.size < 2:
            observations.append(
                DatedICObservation(signal_dt, None, "INVALID_IC_INSUFFICIENT_PAIRS")
            )
            continue
        if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(rets)):
            observations.append(DatedICObservation(signal_dt, None, "INVALID_IC_NONFINITE_VALUES"))
            continue
        if np.ptp(preds) == 0:
            observations.append(DatedICObservation(signal_dt, None, "INVALID_IC_CONSTANT_SCORES"))
            continue
        if np.ptp(rets) == 0:
            observations.append(DatedICObservation(signal_dt, None, "INVALID_IC_CONSTANT_OUTCOMES"))
            continue

        value = cross_sectional_ic(preds, rets, method=method)
        observations.append(DatedICObservation(signal_dt, float(value), valid_status))
    return tuple(observations)


def empirical_one_sided_p_value(
    observed_statistic: float,
    null_statistics: np.ndarray,
) -> float:
    """Return an upper-tail empirical p-value with the plus-one correction."""
    if not np.isfinite(observed_statistic):
        raise ValueError("observed_statistic must be finite")
    null_values = np.asarray(null_statistics, dtype=np.float64)
    if null_values.ndim != 1:
        raise ValueError("null_statistics must be 1-D")
    valid_null = null_values[np.isfinite(null_values)]
    if valid_null.size == 0:
        raise ValueError("null_statistics must contain at least one finite draw")
    upper_tail_count = int(np.count_nonzero(valid_null >= observed_statistic))
    return float((1 + upper_tail_count) / (1 + valid_null.size))


def daily_ic_series(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    method: str = "pearson",
) -> np.ndarray:
    """Compute one cross-sectional IC value per day."""
    preds = _as_2d(predictions)
    rets = _as_2d(true_returns)
    if preds.shape != rets.shape:
        raise ValueError(
            f"predictions and true_returns shapes differ: {preds.shape} != {rets.shape}"
        )
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    values: list[float] = []
    for p, r in zip(preds, rets, strict=True):
        mask = np.isfinite(p) & np.isfinite(r)
        if int(mask.sum()) < 2:
            continue
        p_valid = p[mask]
        r_valid = r[mask]
        if np.nanstd(p_valid) == 0 or np.nanstd(r_valid) == 0:
            continue
        if method == "spearman":
            corr = _corr(_average_ranks(p_valid), _average_ranks(r_valid))
        else:
            corr = _corr(p_valid, r_valid)
        if np.isfinite(corr):
            values.append(float(corr))
    return np.asarray(values, dtype=np.float64)


def newey_west_std(values: np.ndarray, lags: int) -> float:
    """Return the Newey-West adjusted standard deviation of a mean series."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    if lags < 0:
        raise ValueError("lags must be >= 0")

    centered = x - np.mean(x)
    n = centered.size
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        variance += 2.0 * weight * cov
    return float(np.sqrt(max(variance, 0.0)))


def hansen_hodrick_std(values: np.ndarray, lags: int) -> float:
    """Long-run standard deviation with uniform (Hansen-Hodrick) weights on lags 1..``lags``.

    The right estimator for a dated series of ``h``-session returns sampled every
    session: the overlap makes the first ``h - 1`` autocovariances real and the rest
    zero, so each gets full weight. Bartlett weights would discount them and leave the
    variance too small. The variance is floored at zero, since uniform weights do not
    guarantee a positive estimate.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if x.size <= 1:
        return 0.0

    centered = x - np.mean(x)
    n = centered.size
    variance = float(np.dot(centered, centered) / n)
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2.0 * float(np.dot(centered[lag:], centered[:-lag]) / n)
    return float(np.sqrt(max(variance, 0.0)))


def hansen_hodrick_mean_inference(
    values: np.ndarray,
    *,
    label_horizon: int,
) -> HACMeanInference:
    """Mean of a dated series of overlapping ``label_horizon``-session outcomes, HH t-stat.

    Lags are ``label_horizon - 1``, the overlap the outcomes really have. The standard
    error, t-stat and p-value are None when the estimated variance is not positive.
    """
    if label_horizon <= 0:
        raise ValueError("label_horizon must be > 0")
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be 1-D")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must contain only finite observations")

    lags = label_horizon - 1
    method = "hansen_hodrick_uniform_v1"
    if x.size == 0:
        return HACMeanInference(method, 0, None, lags, None, None, None)

    mean_value = float(np.mean(x))
    standard_error = hansen_hodrick_std(x, lags=lags) / math.sqrt(x.size)
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        return HACMeanInference(method, int(x.size), mean_value, lags, None, None, None)
    t_stat = float(mean_value / standard_error)
    return HACMeanInference(
        method=method,
        n_obs=int(x.size),
        mean=mean_value,
        lags=lags,
        standard_error=float(standard_error),
        t_stat=t_stat,
        p_value=float(math.erfc(abs(t_stat) / math.sqrt(2.0))),
    )


def overlap_hac_lags(label_horizon: int) -> int:
    """Uniform HAC lags for close ``D+1`` to close ``D+label_horizon`` outcomes.

    The outcome spans ``label_horizon - 1`` sessions, so consecutive days share
    ``label_horizon - 2`` of them and autocovariances vanish beyond that lag.
    """
    if label_horizon <= 0:
        raise ValueError("label_horizon must be > 0")
    return max(0, label_horizon - 2)


def hansen_hodrick_mean_ci(
    values: np.ndarray,
    *,
    lags: int,
    ci_level: float,
) -> HACMeanInterval:
    """Interval for the mean of a dated series whose autocovariances end at ``lags``.

    The standard error is the uniform-weight (Hansen-Hodrick) long-run one, exact for
    overlapping outcomes when ``lags`` is their overlap (#342). The critical value is
    Student-t on ``n / (2 * lags + 1) - 1`` degrees of freedom: the equivalent degrees
    of freedom of a truncated-kernel long-run variance at frequency zero, less one for
    the estimated mean. With no lags that is the t interval on ``n - 1``. A normal
    critical value leaves the interval short on a few hundred dates, since the
    long-run variance is itself estimated from few effective observations (#356).

    Non-finite values are dropped. Bounds are None when fewer than two values remain
    or the estimated variance is not positive.
    """
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be 1-D")
    x = x[np.isfinite(x)]
    method = "hansen_hodrick_t_v1"
    n = int(x.size)
    if n < 2:
        return HACMeanInterval(method, n, lags, None, ci_level, None, None)

    df = max(1.0, n / (2 * lags + 1) - 1.0)
    standard_error = hansen_hodrick_std(x, lags=lags) / math.sqrt(n)
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        return HACMeanInterval(method, n, lags, df, ci_level, None, None)
    mean_value = float(np.mean(x))
    half_width = float(stats.t.ppf(0.5 + ci_level / 2.0, df)) * standard_error
    return HACMeanInterval(
        method=method,
        n_obs=n,
        lags=lags,
        df=df,
        ci_level=ci_level,
        lower=mean_value - half_width,
        upper=mean_value + half_width,
    )


def newey_west_mean_inference(
    values: np.ndarray,
    *,
    lags: int,
    label_horizon: int,
) -> HACMeanInference:
    """Estimate a mean with overlap-aware Newey-West standard errors."""
    if label_horizon <= 0:
        raise ValueError("label_horizon must be > 0")
    minimum_lags = label_horizon - 1
    if lags < minimum_lags:
        raise ValueError(
            f"lags must be at least label_horizon - 1 ({minimum_lags}) for overlapping outcomes"
        )

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be 1-D")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must contain only finite observations")

    if x.size == 0:
        return HACMeanInference(
            method="newey_west_bartlett_v1",
            n_obs=0,
            mean=None,
            lags=lags,
            standard_error=None,
            t_stat=None,
            p_value=None,
        )

    mean_value = float(np.mean(x))
    long_run_std = newey_west_std(x, lags=lags)
    standard_error = long_run_std / math.sqrt(x.size)
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        t_stat = None
        p_value = None
        serializable_standard_error = None
    else:
        t_stat = float(mean_value / standard_error)
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
        serializable_standard_error = float(standard_error)

    return HACMeanInference(
        method="newey_west_bartlett_v1",
        n_obs=int(x.size),
        mean=mean_value,
        lags=lags,
        standard_error=serializable_standard_error,
        t_stat=t_stat,
        p_value=p_value,
    )


def newey_west_sharpe(
    returns: np.ndarray,
    periods_per_year: int = 252,
    lags: int = 0,
) -> float:
    """Annualized Sharpe using Newey-West adjusted volatility."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    vol = newey_west_std(r, lags=lags)
    if vol <= 0 or not np.isfinite(vol):
        return 0.0
    return float(np.mean(r) / vol * np.sqrt(periods_per_year))


def moving_block_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    block_size: int,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, float]:
    """Estimate a confidence interval using circular moving-block bootstrap."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"lower": float("nan"), "upper": float("nan")}
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be > 0")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n = x.size
    stats_out = np.empty(n_resamples, dtype=np.float64)
    starts = np.arange(n)
    for i in range(n_resamples):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            idx = (start + np.arange(block_size)) % n
            sample.extend(x[idx].tolist())
        stats_out[i] = float(statistic(np.asarray(sample[:n], dtype=np.float64)))

    alpha = 1.0 - ci_level
    lower, upper = np.quantile(stats_out, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"lower": float(lower), "upper": float(upper)}


def moving_block_mean_ci(
    values: np.ndarray,
    *,
    block_size: int,
    label_horizon: int,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> MovingBlockMeanInterval:
    """Estimate a dated mean interval with overlap-aware block-size validation."""
    if label_horizon <= 0:
        raise ValueError("label_horizon must be > 0")
    if block_size < label_horizon:
        raise ValueError(
            f"block_size must be at least label_horizon ({label_horizon}) for overlapping outcomes"
        )

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be 1-D")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must contain only finite observations")

    interval = moving_block_bootstrap_ci(
        x,
        statistic=lambda sample: float(np.mean(sample)),
        block_size=block_size,
        n_resamples=n_resamples,
        seed=seed,
        ci_level=ci_level,
    )
    lower = interval["lower"]
    upper = interval["upper"]
    return MovingBlockMeanInterval(
        method="circular_moving_block_percentile_v1",
        n_obs=int(x.size),
        block_size=block_size,
        n_resamples=n_resamples,
        seed=seed,
        ci_level=ci_level,
        lower=float(lower) if np.isfinite(lower) else None,
        upper=float(upper) if np.isfinite(upper) else None,
    )
