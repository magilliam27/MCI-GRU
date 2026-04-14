"""
Evaluation metrics for MCI-GRU experiments.

This module provides metrics for evaluating stock prediction models.
"""

import numpy as np
from scipy import stats


def compute_metrics(
    predictions: np.ndarray, true_returns: np.ndarray, top_k: int = 50
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
    correlations = []
    for i in range(len(predictions)):
        if len(predictions[i]) > 1:
            corr, _ = stats.spearmanr(predictions[i], true_returns[i])
            if not np.isnan(corr):
                correlations.append(corr)

    if correlations:
        metrics["avg_spearman_corr"] = float(np.mean(correlations))
        metrics["median_spearman_corr"] = float(np.median(correlations))

    # Information Coefficient (Pearson correlation)
    ic_values = []
    for i in range(len(predictions)):
        if len(predictions[i]) > 1:
            corr, _ = stats.pearsonr(predictions[i], true_returns[i])
            if not np.isnan(corr):
                ic_values.append(corr)

    if ic_values:
        metrics["avg_ic"] = float(np.mean(ic_values))
        metrics["ic_ir"] = float(
            np.mean(ic_values) / (np.std(ic_values) + 1e-8)
        )  # Information Ratio

    # Portfolio metrics (top-k selection)
    portfolio_returns = []
    for i in range(len(predictions)):
        # Select top-k stocks by prediction
        top_indices = np.argsort(predictions[i])[-top_k:]
        portfolio_return = np.mean(true_returns[i][top_indices])
        portfolio_returns.append(portfolio_return)

    if portfolio_returns:
        portfolio_returns = np.array(portfolio_returns)
        metrics["top_k"] = top_k
        metrics["avg_portfolio_return"] = float(np.mean(portfolio_returns))
        metrics["cumulative_return"] = float(np.sum(portfolio_returns))
        metrics["sharpe_ratio"] = float(
            np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        )

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
    predictions: np.ndarray, true_returns: np.ndarray, top_k_values: list[int] = None
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
    basic = compute_metrics(predictions, true_returns)
    metrics.update(basic)

    # Hit rate
    metrics["hit_rate"] = compute_hit_rate(predictions, true_returns)

    # Rank metrics
    rank = compute_rank_metrics(predictions, true_returns)
    metrics.update(rank)

    # Multiple top-k evaluations
    for k in top_k_values:
        k_metrics = compute_metrics(predictions, true_returns, top_k=k)
        metrics[f"return_top_{k}"] = k_metrics["avg_portfolio_return"]
        metrics[f"sharpe_top_{k}"] = k_metrics["sharpe_ratio"]

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
