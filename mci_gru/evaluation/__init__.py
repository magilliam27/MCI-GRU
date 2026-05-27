"""Evaluation, portfolio, and monitoring helpers for MCI-GRU."""

from mci_gru.evaluation.prediction_report import (
    align_prediction_comparison,
    build_markdown_report,
    compute_oos_r2_zero,
    compute_sign_metrics,
    compute_tsfm_prediction_report,
    load_prediction_files,
    load_prediction_input,
    realized_returns_from_market_data,
    write_tsfm_prediction_report,
)
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    moving_block_bootstrap_ci,
    newey_west_sharpe,
    newey_west_std,
)

__all__ = [
    "align_prediction_comparison",
    "build_markdown_report",
    "compute_oos_r2_zero",
    "compute_sign_metrics",
    "compute_tsfm_prediction_report",
    "daily_ic_series",
    "load_prediction_files",
    "load_prediction_input",
    "moving_block_bootstrap_ci",
    "newey_west_sharpe",
    "newey_west_std",
    "realized_returns_from_market_data",
    "write_tsfm_prediction_report",
]
