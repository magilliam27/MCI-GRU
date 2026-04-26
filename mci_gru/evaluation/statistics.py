"""Statistical evaluation helpers for cross-sectional prediction experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


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


def daily_ic_series(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    method: str = "pearson",
) -> np.ndarray:
    """Compute one cross-sectional IC value per day."""
    preds = _as_2d(predictions)
    rets = _as_2d(true_returns)
    if preds.shape != rets.shape:
        raise ValueError(f"predictions and true_returns shapes differ: {preds.shape} != {rets.shape}")
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
