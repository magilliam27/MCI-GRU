"""
Evaluation metrics for MCI-GRU experiments.

This module provides metrics for evaluating stock prediction models.
"""

import numpy as np

from mci_gru.evaluation.portfolio import top_k_returns
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    moving_block_bootstrap_ci,
    newey_west_sharpe,
)


def compute_metrics(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    top_k: int = 50,
    label_t: int = 1,
    bootstrap_enabled: bool = False,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    ci_level: float = 0.95,
    block_size: int | None = None,
    newey_west_lags: int | None = None,
) -> dict[str, float]:
    """
    Compute evaluation metrics for predictions.

    Args:
        predictions: Predicted scores (n_days, n_stocks)
        true_returns: Actual returns (n_days, n_stocks)
        top_k: Number of top stocks to select for portfolio metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # MSE and RMSE
    mse = np.mean((predictions - true_returns) ** 2)
    metrics["mse"] = float(mse)
    metrics["rmse"] = float(np.sqrt(mse))

    # MAE
    metrics["mae"] = float(np.mean(np.abs(predictions - true_returns)))

    # Correlation metrics
    correlations = daily_ic_series(predictions, true_returns, method="spearman")

    if len(correlations) > 0:
        metrics["avg_spearman_corr"] = float(np.mean(correlations))
        metrics["median_spearman_corr"] = float(np.median(correlations))

    # Information Coefficient (Pearson correlation)
    ic_values = daily_ic_series(predictions, true_returns, method="pearson")

    if len(ic_values) > 0:
        metrics["avg_ic"] = float(np.mean(ic_values))
        metrics["ic_ir"] = float(
            np.mean(ic_values) / (np.std(ic_values) + 1e-8)
        )  # Information Ratio
        if bootstrap_enabled:
            ci = moving_block_bootstrap_ci(
                ic_values,
                statistic=np.mean,
                block_size=block_size or max(1, label_t),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                ci_level=ci_level,
            )
            metrics["avg_ic_ci_lower"] = ci["lower"]
            metrics["avg_ic_ci_upper"] = ci["upper"]

    # Portfolio metrics (top-k selection)
    portfolio_returns = top_k_returns(predictions, true_returns, top_k=top_k)

    if len(portfolio_returns) > 0:
        nw_lags = max(0, label_t - 1) if newey_west_lags is None else newey_west_lags
        metrics["top_k"] = top_k
        metrics["avg_portfolio_return"] = float(np.mean(portfolio_returns))
        metrics["cumulative_return"] = float(np.sum(portfolio_returns))
        naive_sharpe = float(
            np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        )
        nw_sharpe = newey_west_sharpe(portfolio_returns, lags=nw_lags)
        metrics["sharpe_ratio"] = nw_sharpe if label_t > 1 else naive_sharpe
        metrics["sharpe_naive"] = naive_sharpe
        metrics["sharpe_newey_west"] = nw_sharpe
        if bootstrap_enabled:
            ci = moving_block_bootstrap_ci(
                portfolio_returns,
                statistic=np.mean,
                block_size=block_size or max(1, label_t),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + top_k,
                ci_level=ci_level,
            )
            metrics[f"top_{top_k}_return_ci_lower"] = ci["lower"]
            metrics[f"top_{top_k}_return_ci_upper"] = ci["upper"]

    return metrics


def compute_hit_rate(
    predictions: np.ndarray, true_returns: np.ndarray, threshold: float = 0.0
) -> float:
    """
    Compute directional accuracy (hit rate).

    Args:
        predictions: Predicted scores
        true_returns: Actual returns
        threshold: Threshold for positive prediction

    Returns:
        Hit rate (0 to 1)
    """
    pred_direction = predictions > threshold
    true_direction = true_returns > 0

    hits = (pred_direction == true_direction).sum()
    total = pred_direction.size

    return float(hits / total) if total > 0 else 0.0


def compute_rank_metrics(
    predictions: np.ndarray, true_returns: np.ndarray, quantiles: int = 5
) -> dict[str, float]:
    """
    Compute rank-based metrics.

    Args:
        predictions: Predicted scores (n_days, n_stocks)
        true_returns: Actual returns (n_days, n_stocks)
        quantiles: Number of quantiles for grouping

    Returns:
        Dictionary with quantile returns
    """
    metrics = {}

    quantile_returns = [[] for _ in range(quantiles)]

    for i in range(len(predictions)):
        # Rank stocks by prediction
        ranks = np.argsort(np.argsort(predictions[i]))
        n_stocks = len(ranks)
        quantile_size = n_stocks // quantiles

        for q in range(quantiles):
            start_rank = q * quantile_size
            end_rank = (q + 1) * quantile_size if q < quantiles - 1 else n_stocks

            quantile_mask = (ranks >= start_rank) & (ranks < end_rank)
            quantile_return = np.mean(true_returns[i][quantile_mask])
            quantile_returns[q].append(quantile_return)

    # Compute average return per quantile
    for q in range(quantiles):
        metrics[f"quantile_{q + 1}_return"] = float(np.mean(quantile_returns[q]))

    # Long-short spread (top quantile - bottom quantile)
    metrics["long_short_spread"] = (
        metrics[f"quantile_{quantiles}_return"] - metrics["quantile_1_return"]
    )

    return metrics


def evaluate_predictions(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    top_k_values: list[int] = None,
    label_t: int = 1,
    bootstrap_enabled: bool = False,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    ci_level: float = 0.95,
    block_size: int | None = None,
    newey_west_lags: int | None = None,
) -> dict[str, float]:
    """
    Comprehensive evaluation of predictions.

    Args:
        predictions: Predicted scores
        true_returns: Actual returns
        top_k_values: List of top-k values to evaluate

    Returns:
        Dictionary with all metrics
    """
    if top_k_values is None:
        top_k_values = [10, 20, 50, 100]

    metrics = {}

    # Basic metrics
    basic = compute_metrics(
        predictions,
        true_returns,
        label_t=label_t,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        ci_level=ci_level,
        block_size=block_size,
        newey_west_lags=newey_west_lags,
    )
    metrics.update(basic)

    # Hit rate
    metrics["hit_rate"] = compute_hit_rate(predictions, true_returns)

    # Rank metrics
    rank = compute_rank_metrics(predictions, true_returns)
    metrics.update(rank)

    # Multiple top-k evaluations
    for k in top_k_values:
        k_metrics = compute_metrics(
            predictions,
            true_returns,
            top_k=k,
            label_t=label_t,
            bootstrap_enabled=bootstrap_enabled,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
            block_size=block_size,
            newey_west_lags=newey_west_lags,
        )
        metrics[f"return_top_{k}"] = k_metrics["avg_portfolio_return"]
        metrics[f"sharpe_top_{k}"] = k_metrics["sharpe_ratio"]
        metrics[f"sharpe_top_{k}_naive"] = k_metrics["sharpe_naive"]
        metrics[f"sharpe_top_{k}_newey_west"] = k_metrics["sharpe_newey_west"]
        if f"top_{k}_return_ci_lower" in k_metrics:
            metrics[f"top_{k}_return_ci_lower"] = k_metrics[f"top_{k}_return_ci_lower"]
            metrics[f"top_{k}_return_ci_upper"] = k_metrics[f"top_{k}_return_ci_upper"]

    return metrics


def print_metrics(metrics: dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    # Group metrics by category
    loss_keys = ["mse", "rmse", "mae"]
    corr_keys = ["avg_spearman_corr", "median_spearman_corr", "avg_ic", "ic_ir"]
    portfolio_keys = [k for k in metrics if "return" in k or "sharpe" in k]

    print("\nLoss Metrics:")
    for k in loss_keys:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.6f}")

    print("\nCorrelation Metrics:")
    for k in corr_keys:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")

    print("\nPortfolio Metrics:")
    for k in portfolio_keys:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")

    if "hit_rate" in metrics:
        print(f"\nHit Rate: {metrics['hit_rate']:.4f}")

    print("=" * 60)
