"""Saved-prediction ranking and IC evaluation metrics.

``evaluate_run`` is the in-run summary (#336): IC on the training labels, and every
portfolio number from the daily P&L of the staggered tranche book the label implies,
with the headline as excess over the equal-weight rankable universe. Returns come from
the evaluation core (#335) on the panel's session axis.

``evaluate_predictions`` and ``compute_metrics`` are the older matrix-only path, kept
for ``prediction_report`` until #342 moves it onto the core. Their Sharpe and
cumulative return treat overlapping multi-session returns as daily; do not use them
for a run's reported numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mci_gru.evaluation.outcomes import order_by_score, realized_returns
from mci_gru.evaluation.portfolio import top_k_returns
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    hansen_hodrick_mean_inference,
    moving_block_bootstrap_ci,
    newey_west_sharpe,
    newey_west_std,
)
from mci_gru.evaluation.tranche_book import (
    period_returns,
    signal_date_closes,
    tranche_book,
)

if TYPE_CHECKING:
    import pandas as pd

    from mci_gru.config import Legs

SUMMARY_SCHEMA_VERSION = "mci_gru.evaluation_summary.v2"
PERIODS_PER_YEAR = 252
QUINTILES = 5


def compute_ic_metrics(
    predictions: np.ndarray,
    true_returns: np.ndarray,
    *,
    label_t: int,
    bootstrap_enabled: bool = False,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    ci_level: float = 0.95,
    block_size: int | None = None,
) -> dict[str, float]:
    """Pearson and Spearman IC per date, their means, IRs and optional block-bootstrap CIs."""
    metrics: dict[str, float] = {}
    # Rank IC (Spearman correlation)
    rank_ic_values = daily_ic_series(predictions, true_returns, method="spearman")

    if len(rank_ic_values) > 0:
        avg_rank_ic = float(np.mean(rank_ic_values))
        median_rank_ic = float(np.median(rank_ic_values))
        metrics["avg_rank_ic"] = avg_rank_ic
        metrics["median_rank_ic"] = median_rank_ic
        metrics["rank_ic_ir"] = float(np.mean(rank_ic_values) / (np.std(rank_ic_values) + 1e-8))
        metrics["avg_spearman_corr"] = avg_rank_ic
        metrics["median_spearman_corr"] = median_rank_ic
        if bootstrap_enabled:
            ci = moving_block_bootstrap_ci(
                rank_ic_values,
                statistic=np.mean,
                block_size=block_size or max(1, label_t),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + 17,
                ci_level=ci_level,
            )
            metrics["avg_rank_ic_ci_lower"] = ci["lower"]
            metrics["avg_rank_ic_ci_upper"] = ci["upper"]

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

    return metrics


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
    valid = np.isfinite(predictions) & np.isfinite(true_returns)

    # MSE and RMSE
    mse = np.mean((predictions[valid] - true_returns[valid]) ** 2) if valid.any() else np.nan
    metrics["mse"] = float(mse)
    metrics["rmse"] = float(np.sqrt(mse))

    # MAE
    metrics["mae"] = (
        float(np.mean(np.abs(predictions[valid] - true_returns[valid])))
        if valid.any()
        else float("nan")
    )

    metrics.update(
        compute_ic_metrics(
            predictions,
            true_returns,
            label_t=label_t,
            bootstrap_enabled=bootstrap_enabled,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
            block_size=block_size,
        )
    )

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
    valid = np.isfinite(predictions) & np.isfinite(true_returns)

    hits = ((pred_direction == true_direction) & valid).sum()
    total = valid.sum()

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
        valid = np.isfinite(predictions[i]) & np.isfinite(true_returns[i])
        if valid.sum() < quantiles:
            continue
        pred_row = predictions[i][valid]
        ret_row = true_returns[i][valid]
        # Rank stocks by prediction
        ranks = np.argsort(np.argsort(pred_row))
        n_stocks = len(ranks)
        quantile_size = n_stocks // quantiles

        for q in range(quantiles):
            start_rank = q * quantile_size
            end_rank = (q + 1) * quantile_size if q < quantiles - 1 else n_stocks

            quantile_mask = (ranks >= start_rank) & (ranks < end_rank)
            quantile_return = np.mean(ret_row[quantile_mask])
            quantile_returns[q].append(quantile_return)

    # Compute average return per quantile
    for q in range(quantiles):
        metrics[f"quantile_{q + 1}_return"] = (
            float(np.mean(quantile_returns[q])) if quantile_returns[q] else float("nan")
        )

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
        metrics[f"return_top_{k}"] = k_metrics.get("avg_portfolio_return", float("nan"))
        metrics[f"sharpe_top_{k}"] = k_metrics.get("sharpe_ratio", float("nan"))
        metrics[f"sharpe_top_{k}_naive"] = k_metrics.get("sharpe_naive", float("nan"))
        metrics[f"sharpe_top_{k}_newey_west"] = k_metrics.get(
            "sharpe_newey_west",
            float("nan"),
        )
        if f"top_{k}_return_ci_lower" in k_metrics:
            metrics[f"top_{k}_return_ci_lower"] = k_metrics[f"top_{k}_return_ci_lower"]
            metrics[f"top_{k}_return_ci_upper"] = k_metrics[f"top_{k}_return_ci_upper"]

    return metrics


def annualised_sharpe(
    daily_returns: np.ndarray,
    *,
    method: str,
    lags: int,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe of a daily P&L series; NaN when it has no spread.

    ``method`` is ``evaluation.sharpe_method``: ``naive`` divides by the sample
    standard deviation, ``newey_west`` by the Bartlett long-run one with ``lags``.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    if method == "naive":
        vol = float(np.std(r))
    elif method == "newey_west":
        vol = newey_west_std(r, lags=lags)
    else:
        raise ValueError(f"sharpe_method must be 'naive' or 'newey_west', got {method!r}")
    if vol <= 0 or not np.isfinite(vol):
        return float("nan")
    return float(np.mean(r) / vol * np.sqrt(periods_per_year))


def quintile_period_returns(
    scores: np.ndarray,
    rankable_mask: np.ndarray,
    returns: np.ndarray,
    kdcodes: list[str],
    quintiles: int = QUINTILES,
) -> dict[str, float]:
    """Mean booked return per score quintile; quintile 1 lowest, sizes differ by at most one.

    Any remainder goes to the top quintiles.

    Each date ranks its rankable, scored names by the core tie rule and splits them
    evenly. A date with fewer names than quintiles, or with a name lacking a booked
    return, is skipped.
    """
    by_quintile: list[list[float]] = [[] for _ in range(quintiles)]
    for i in range(len(scores)):
        allowed = np.asarray(rankable_mask[i], dtype=bool) & np.isfinite(scores[i])
        order = order_by_score(scores[i], kdcodes)
        order = order[allowed[order]]
        if order.size < quintiles or not np.all(np.isfinite(returns[i, order])):
            continue
        # Split from the top, as top-k picks, then number from the bottom.
        for q, members in enumerate(np.array_split(order, quintiles)[::-1]):
            by_quintile[q].append(float(np.mean(returns[i, members])))
    out = {
        f"quintile_{q + 1}_abs_period_return": float(np.mean(values)) if values else float("nan")
        for q, values in enumerate(by_quintile)
    }
    out["long_short_abs_period_spread"] = (
        out[f"quintile_{quintiles}_abs_period_return"] - out["quintile_1_abs_period_return"]
    )
    return out


def evaluate_run(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    panel: pd.DataFrame,
    dates: list[str],
    kdcodes: list[str],
    rankable_mask: np.ndarray | None,
    legs: Legs,
    tranches: int,
    top_k_values: list[int],
    label_t: int,
    sharpe_method: str = "newey_west",
    newey_west_lags: int | None = None,
    bootstrap_enabled: bool = False,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    ci_level: float = 0.95,
    block_size: int | None = None,
) -> dict[str, Any]:
    """The in-run summary body: IC on ``labels``, portfolio numbers from the tranche book.

    ``predictions`` and ``labels`` are ``(dates, kdcodes)``; ``labels`` are the training
    labels the IC has always been scored against. ``panel`` (``kdcode``, ``dt``,
    ``close``) gives the booked returns and the daily P&L. ``rankable_mask`` is the PIT
    tradable mask; without one, a name is rankable on a date it printed a close.

    For every k in ``top_k_values``, ``abs`` numbers are the long-only top-k book and
    ``excess`` numbers are top-k minus the equal-weight universe on the same mask, legs
    and tranche schedule. ``*_period_return`` is the mean per signal date of the
    ``legs.hold_sessions``-session return, with a Hansen-Hodrick t-stat for the excess;
    ``*_sharpe`` and ``*_cumulative_return`` come from the book's daily P&L.
    """
    hold = legs.hold_sessions
    scores = np.asarray(predictions, dtype=np.float64)
    dates = [str(date) for date in dates]
    kdcodes = list(kdcodes)
    panel = panel.loc[panel["kdcode"].astype(str).isin(kdcodes)]
    realized = realized_returns(panel, dates, legs, kdcodes=kdcodes)
    if rankable_mask is None:
        rankable = signal_date_closes(panel, realized)
        benchmark = "equal_weight_rankable_universe"
    else:
        rankable = np.asarray(rankable_mask, dtype=bool)
        benchmark = "equal_weight_pit_tradable_universe"
    nw_lags = max(0, hold - 1) if newey_west_lags is None else newey_west_lags
    # The excess CI resamples overlapping hold-length returns, so a block never
    # shorter than the hold.
    excess_block = max(block_size or max(1, label_t), hold)

    metrics: dict[str, Any] = compute_ic_metrics(
        predictions,
        labels,
        label_t=label_t,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        ci_level=ci_level,
        block_size=block_size,
    )
    metrics["test_dates"] = len(dates)
    metrics["ic_valid_dates"] = int(daily_ic_series(predictions, labels, "pearson").size)
    metrics["rank_ic_valid_dates"] = int(daily_ic_series(predictions, labels, "spearman").size)
    metrics["ic_date_coverage"] = metrics["ic_valid_dates"] / len(dates) if dates else float("nan")

    def sharpe(book) -> float:
        return annualised_sharpe(book.daily_returns, method=sharpe_method, lags=nw_lags)

    universe_book = tranche_book(panel, scores, rankable, realized, top_k=None, tranches=tranches)
    universe_period = period_returns(scores, rankable, realized, top_k=None)
    metrics["universe_abs_period_return"] = _finite_mean(universe_period)
    metrics["universe_abs_sharpe"] = sharpe(universe_book)
    metrics["universe_abs_cumulative_return"] = universe_book.cumulative_return
    metrics["book_sessions"] = len(universe_book.sessions)

    for k in top_k_values:
        book = tranche_book(panel, scores, rankable, realized, top_k=k, tranches=tranches)
        period = period_returns(scores, rankable, realized, top_k=k)
        if book.sessions != universe_book.sessions:
            raise ValueError(f"top-{k} and universe books cover different sessions")
        excess_daily = book.daily_returns - universe_book.daily_returns
        excess_period = period - universe_period
        finite_excess = excess_period[np.isfinite(excess_period)]
        inference = hansen_hodrick_mean_inference(finite_excess, label_horizon=hold)

        metrics[f"top_{k}_abs_period_return"] = _finite_mean(period)
        metrics[f"top_{k}_abs_sharpe"] = sharpe(book)
        metrics[f"top_{k}_abs_cumulative_return"] = book.cumulative_return
        metrics[f"top_{k}_excess_period_return"] = _finite_mean(excess_period)
        metrics[f"top_{k}_excess_t_stat"] = _or_nan(inference.t_stat)
        metrics[f"top_{k}_excess_p_value"] = _or_nan(inference.p_value)
        metrics[f"top_{k}_excess_sharpe"] = annualised_sharpe(
            excess_daily, method=sharpe_method, lags=nw_lags
        )
        # Relative wealth: the top-k book's growth over the universe book's.
        metrics[f"top_{k}_excess_cumulative_return"] = float(
            (1.0 + book.cumulative_return) / (1.0 + universe_book.cumulative_return) - 1.0
        )
        metrics[f"top_{k}_unbooked_tranches"] = book.tranches_unbooked
        if bootstrap_enabled:
            ci = moving_block_bootstrap_ci(
                finite_excess,
                statistic=np.mean,
                block_size=excess_block,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + k,
                ci_level=ci_level,
            )
            metrics[f"top_{k}_excess_period_return_ci_lower"] = ci["lower"]
            metrics[f"top_{k}_excess_period_return_ci_upper"] = ci["upper"]

    metrics.update(quintile_period_returns(scores, rankable, realized.returns, kdcodes))

    events = {"missing_entry": 0, "missing_exit": 0}
    for event in realized.events:
        events[event.kind] += 1
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "label_t": label_t,
        "label_sessions": hold,
        "legs": {"entry_session": legs.entry_session, "exit_session": legs.exit_session},
        "tranches": tranches,
        "top_k_values": list(top_k_values),
        "return_basis": "price",
        "benchmark": benchmark,
        "periods_per_year": PERIODS_PER_YEAR,
        "sharpe_method": sharpe_method,
        "newey_west_lags": nw_lags,
        "excess_inference": {"method": "hansen_hodrick_uniform_v1", "lags": hold - 1},
        "excess_ci_block_size": excess_block,
        "missing_close_events": events,
        "metrics": metrics,
    }


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


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
    corr_keys = [
        "avg_ic",
        "ic_ir",
        "avg_rank_ic",
        "median_rank_ic",
        "rank_ic_ir",
        "avg_spearman_corr",
        "median_spearman_corr",
    ]
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
