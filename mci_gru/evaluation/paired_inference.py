"""Paired inference for arm-versus-control comparisons on shared test days.

Ticket 179 (Wayfinder map 157). The graph-specification ablation scored every
arm on the same test dates but compared arms through independent-looking
confidence intervals, discarding the pairing. These helpers work on the daily
difference ``delta_k(d) = IC_k(d) - IC_control(d)`` instead, using the
repository's existing overlap-aware primitives from
:mod:`mci_gru.evaluation.statistics`, and add the multiple-comparison and power
arithmetic a multi-year protocol needs. Everything here is pure; file I/O stays
in the notebook that calls it.

None of this changes the ticket-164 arbiter. It describes the same evidence
more efficiently and reports what a redesigned protocol could detect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mci_gru.evaluation.statistics import (
    moving_block_bootstrap_ci,
    moving_block_mean_ci,
    newey_west_mean_inference,
    newey_west_sharpe,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class PairedMeanInference:
    """Overlap-aware inference on the mean of one arm's daily differences."""

    arm: str
    control: str
    n_days: int
    mean_delta: float
    median_delta: float
    sd_delta: float
    win_rate: float
    hac_lags: int
    hac_se: float | None
    hac_t: float | None
    hac_p: float | None
    block_size: int
    ci_lower: float | None
    ci_upper: float | None
    top_decile_share: float


def align_daily_series(series_by_arm: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Inner-join dated series on their index, keeping only dates finite for every arm.

    Columns follow the mapping's insertion order; rows are sorted by date.
    """
    if not series_by_arm:
        raise ValueError("align_daily_series needs at least one series")
    frame = pd.concat(
        {name: pd.Series(values, dtype=float) for name, values in series_by_arm.items()},
        axis=1,
        join="inner",
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any").sort_index()
    if frame.empty:
        raise ValueError("no common dates with finite values across the supplied arms")
    return frame[list(series_by_arm.keys())]


def paired_daily_differences(frame: pd.DataFrame, control: str) -> pd.DataFrame:
    """Return ``arm - control`` per date for every non-control column."""
    if control not in frame.columns:
        raise KeyError(f"control column {control!r} is not in the aligned frame")
    others = [column for column in frame.columns if column != control]
    return frame[others].sub(frame[control], axis=0)


def tail_share(delta: np.ndarray, top_fraction: float = 0.1) -> float:
    """Share of the summed differences contributed by the top ``top_fraction`` of days.

    Returns NaN when the differences sum to zero, where a share is undefined.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    x = np.asarray(delta, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    total = float(np.sum(x))
    if total == 0.0:
        return float("nan")
    k = max(1, int(math.ceil(top_fraction * x.size - 1e-12)))
    top = np.sort(x)[::-1][:k]
    return float(np.sum(top) / total)


def paired_mean_inference(
    delta: pd.Series | np.ndarray,
    *,
    arm: str,
    control: str,
    label_horizon: int,
    n_resamples: int,
    seed: int,
    ci_level: float,
    block_size: int | None = None,
    hac_lags: int | None = None,
) -> PairedMeanInference:
    """HAC t-test and block-bootstrap interval for the mean daily difference.

    Non-finite days are dropped and ``n_days`` reports what remained. ``hac_lags``
    defaults to ``label_horizon - 1`` and ``block_size`` to ``label_horizon``, the
    same overlap-aware defaults the production evaluation resolves to.
    """
    if label_horizon <= 0:
        raise ValueError("label_horizon must be > 0")
    x = np.asarray(pd.Series(delta, dtype=float).to_numpy(), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("paired_mean_inference needs at least one finite difference")
    lags = label_horizon - 1 if hac_lags is None else int(hac_lags)
    block = label_horizon if block_size is None else int(block_size)

    hac = newey_west_mean_inference(x, lags=lags, label_horizon=label_horizon)
    interval = moving_block_mean_ci(
        x,
        block_size=block,
        label_horizon=label_horizon,
        n_resamples=n_resamples,
        seed=seed,
        ci_level=ci_level,
    )
    return PairedMeanInference(
        arm=arm,
        control=control,
        n_days=int(x.size),
        mean_delta=float(np.mean(x)),
        median_delta=float(np.median(x)),
        sd_delta=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        win_rate=float(np.mean(x > 0.0)),
        hac_lags=lags,
        hac_se=hac.standard_error,
        hac_t=hac.t_stat,
        hac_p=hac.p_value,
        block_size=block,
        ci_lower=interval.lower,
        ci_upper=interval.upper,
        top_decile_share=tail_share(x, 0.1),
    )


def bhy_adjusted_p_values(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg-Yekutieli step-up adjustment, monotone, capped at one.

    Mirrors the repeated-seed replication notebook: sorted p-values are inflated by
    ``m * c(m) / rank`` with ``c(m)`` the harmonic number, then made monotone from
    the largest rank downwards. NaN inputs stay NaN and do not count toward ``m``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(p))
    if valid_idx.size == 0:
        return out
    m = int(valid_idx.size)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    order = valid_idx[np.argsort(p[valid_idx], kind="mergesort")]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = np.minimum(p[order] * m * c_m / ranks, 1.0)
    running = 1.0
    for position in range(m - 1, -1, -1):
        running = min(running, float(adjusted[position]))
        adjusted[position] = running
    out[order] = adjusted
    return out


def _z_sum(power: float, alpha: float) -> float:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    dist = NormalDist()
    return dist.inv_cdf(1.0 - alpha / 2.0) + dist.inv_cdf(power)


def minimum_detectable_effect(
    sd: float, n_days: int, *, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Smallest mean daily difference detectable at ``power`` with ``n_days`` paired days.

    Normal approximation: ``(z_{1-alpha/2} + z_power) * sd / sqrt(n)``.
    """
    if sd <= 0.0 or not math.isfinite(sd):
        raise ValueError("sd must be a positive finite number")
    if n_days <= 0:
        raise ValueError("n_days must be > 0")
    return float(_z_sum(power, alpha) * sd / math.sqrt(n_days))


def required_days(sd: float, mde: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Paired days needed to detect ``mde`` at ``power``; inverse of the MDE formula."""
    if sd <= 0.0 or not math.isfinite(sd):
        raise ValueError("sd must be a positive finite number")
    if mde <= 0.0 or not math.isfinite(mde):
        raise ValueError("mde must be a positive finite number")
    exact = (_z_sum(power, alpha) * sd / mde) ** 2
    return int(math.ceil(round(exact, 9)))


def winsorize_rows(values: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """Clip each row to its own [lower_q, upper_q] quantiles, ignoring and keeping NaN."""
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("require 0 <= lower_q < upper_q <= 1")
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError("values must be 2-D (rows are days)")
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        low = np.nanquantile(row, lower_q)
        high = np.nanquantile(row, upper_q)
        row[finite] = np.clip(row[finite], low, high)
    return arr


def sharpe_block_bootstrap_ci(
    returns: np.ndarray,
    *,
    nw_lags: int,
    block_size: int,
    n_resamples: int,
    seed: int,
    ci_level: float,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Annualised Newey-West Sharpe with a circular moving-block bootstrap interval."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("sharpe_block_bootstrap_ci needs at least one finite return")

    def statistic(sample: np.ndarray) -> float:
        return newey_west_sharpe(sample, periods_per_year=periods_per_year, lags=nw_lags)

    point = statistic(r)
    interval = moving_block_bootstrap_ci(
        r,
        statistic=statistic,
        block_size=block_size,
        n_resamples=n_resamples,
        seed=seed,
        ci_level=ci_level,
    )
    return {
        "point": float(point),
        "lower": float(interval["lower"]),
        "upper": float(interval["upper"]),
        "n_days": int(r.size),
        "nw_lags": int(nw_lags),
        "block_size": int(block_size),
    }
