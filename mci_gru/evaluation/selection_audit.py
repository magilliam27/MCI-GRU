"""Saved-prediction model-selection audit helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mci_gru.evaluation.artifacts import write_json_artifact
from mci_gru.evaluation.portfolio import top_k_returns
from mci_gru.evaluation.prediction_report import (
    align_prediction_comparison,
    load_prediction_files,
    realized_returns_from_market_data,
)
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    moving_block_bootstrap_ci,
    newey_west_std,
)


def bhy_adjust_p_value(p_value: float, trial_count: int) -> float:
    """Apply the Benjamini-Hochberg-Yekutieli single p-value inflation."""
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    harmonic = sum(1.0 / i for i in range(1, trial_count + 1))
    return float(min(1.0, p_value * trial_count * harmonic))


def build_selection_audit(
    *,
    predictions_dir: str | Path,
    market_data_path: str | Path,
    label_t: int,
    top_k_values: list[int],
    trial_count: int,
    bootstrap_resamples: int = 500,
    bootstrap_seed: int = 123,
) -> dict[str, Any]:
    """Compute no-retraining selection evidence from saved prediction CSVs."""
    predictions = load_prediction_files(predictions_dir)
    market_data = pd.read_csv(market_data_path)
    realized = realized_returns_from_market_data(market_data, label_t=label_t)
    aligned = align_prediction_comparison(predictions, realized)
    score_matrix = _pivot(aligned, "mci_gru_score")
    return_matrix = _pivot(aligned, "realized_return")

    pearson = daily_ic_series(score_matrix, return_matrix, method="pearson")
    spearman = daily_ic_series(score_matrix, return_matrix, method="spearman")
    rank_ic_mean = _nanmean(spearman)
    nw_std = newey_west_std(spearman, lags=max(0, label_t - 1))
    t_stat = _newey_west_t_stat(rank_ic_mean, nw_std, len(spearman))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(len(spearman) - 1, 1)))
    top_k_return_map = _top_k_return_map(score_matrix, return_matrix, top_k_values)

    return {
        "schema_version": 1,
        "predictions_dir": str(Path(predictions_dir).resolve()),
        "market_data_path": str(Path(market_data_path).resolve()),
        "label_t": int(label_t),
        "trial_count": int(trial_count),
        "sample": {
            "aligned_observations": int(len(aligned)),
            "n_dates": int(aligned["dt"].nunique()),
            "n_kdcodes": int(aligned["kdcode"].nunique()),
        },
        "ic": {
            "pearson_mean": _nanmean(pearson),
            "spearman_mean": rank_ic_mean,
            "spearman_newey_west_t": t_stat,
            "spearman_p_value": p_value,
            "spearman_bootstrap_ci": moving_block_bootstrap_ci(
                spearman,
                statistic=lambda values: float(np.nanmean(values)),
                block_size=max(1, label_t),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                ci_level=0.95,
            ),
        },
        "top_k": _top_k_summary(top_k_return_map),
        "deflated_sharpe": {
            str(top_k): deflated_sharpe_ratio(returns, trial_count=trial_count)
            for top_k, returns in top_k_return_map.items()
        },
        "multiple_testing": {
            "method": "bhy_single_family_v0",
            "bhy_adjusted_p_value": bhy_adjust_p_value(p_value, trial_count),
        },
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    trial_count: int,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Estimate deflated Sharpe evidence for a return series."""
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    clean = np.asarray(returns, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return {
            "method": "bailey_lopez_de_prado_v0",
            "n_obs": int(clean.size),
            "trial_count": int(trial_count),
            "period_sharpe": None,
            "annualized_sharpe": None,
            "expected_max_sharpe": None,
            "z_stat": None,
            "p_value": None,
            "skew": None,
            "kurtosis": None,
        }

    mean_return = float(np.mean(clean))
    std_return = float(np.std(clean, ddof=1))
    if std_return <= 0.0 or not np.isfinite(std_return):
        period_sharpe = None
        annualized_sharpe = None
        z_stat = None
        p_value = None
        expected_max_sharpe = None
    else:
        period_sharpe = mean_return / std_return
        skew = _finite_or_default(stats.skew(clean, bias=False), 0.0)
        kurtosis = _finite_or_default(stats.kurtosis(clean, fisher=False, bias=False), 3.0)
        variance_term = 1.0 - skew * period_sharpe + ((kurtosis - 1.0) / 4.0) * (period_sharpe**2)
        standard_error = math.sqrt(max(variance_term, 1e-12) / max(clean.size - 1, 1))
        expected_max_sharpe = _expected_max_sharpe(trial_count, standard_error)
        z_stat = (period_sharpe - expected_max_sharpe) / standard_error
        p_value = float(stats.norm.sf(z_stat))
        annualized_sharpe = period_sharpe * math.sqrt(periods_per_year)
        return {
            "method": "bailey_lopez_de_prado_v0",
            "n_obs": int(clean.size),
            "trial_count": int(trial_count),
            "period_sharpe": float(period_sharpe),
            "annualized_sharpe": float(annualized_sharpe),
            "expected_max_sharpe": float(expected_max_sharpe),
            "z_stat": float(z_stat),
            "p_value": p_value,
            "skew": float(skew),
            "kurtosis": float(kurtosis),
        }

    return {
        "method": "bailey_lopez_de_prado_v0",
        "n_obs": int(clean.size),
        "trial_count": int(trial_count),
        "period_sharpe": period_sharpe,
        "annualized_sharpe": annualized_sharpe,
        "expected_max_sharpe": expected_max_sharpe,
        "z_stat": z_stat,
        "p_value": p_value,
        "skew": _finite_or_none(stats.skew(clean, bias=False)),
        "kurtosis": _finite_or_none(stats.kurtosis(clean, fisher=False, bias=False)),
    }


def write_selection_audit(
    audit: dict[str, Any], output_dir: str | Path, *, force: bool = False
) -> Path:
    """Write an additive selection audit JSON artifact."""
    out_dir = Path(output_dir)
    path = out_dir / "selection_audit_summary.json"
    return write_json_artifact(path, audit, force=force)


def _pivot(frame: pd.DataFrame, value_col: str) -> np.ndarray:
    wide = frame.pivot(index="dt", columns="kdcode", values=value_col).sort_index()
    return wide.to_numpy(dtype=np.float64)


def _nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else float("nan")


def _newey_west_t_stat(mean_value: float, nw_std: float, n_obs: int) -> float:
    if n_obs <= 0 or nw_std <= 0 or not np.isfinite(mean_value):
        return 0.0
    return float(mean_value / (nw_std / np.sqrt(n_obs)))


def _top_k_return_map(
    score_matrix: np.ndarray,
    return_matrix: np.ndarray,
    top_k_values: list[int],
) -> dict[int, np.ndarray]:
    return {
        int(top_k): top_k_returns(score_matrix, return_matrix, top_k=top_k)
        for top_k in top_k_values
    }


def _top_k_summary(return_map: dict[int, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for top_k, returns in return_map.items():
        summary[str(top_k)] = {
            "mean_return": _nanmean(returns),
            "n_days": int(returns.size),
        }
    return summary


def _expected_max_sharpe(trial_count: int, standard_error: float) -> float:
    if trial_count <= 1:
        return 0.0
    gamma = 0.5772156649015329
    q_1 = stats.norm.ppf(1.0 - 1.0 / trial_count)
    q_2 = stats.norm.ppf(1.0 - 1.0 / (trial_count * math.e))
    estimate = standard_error * ((1.0 - gamma) * q_1 + gamma * q_2)
    return float(estimate) if np.isfinite(estimate) else 0.0


def _finite_or_default(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else default


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
