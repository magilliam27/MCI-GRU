"""Compatibility re-exports for metrics relocated to :mod:`mci_gru.evaluation.metrics`.

Prediction-evaluation metrics are evaluation logic; they moved out of the
training layer (WS-P) so the dependency runs one way: training -> evaluation.
"""

from mci_gru.evaluation.metrics import (
    compute_hit_rate,
    compute_metrics,
    compute_rank_metrics,
    evaluate_predictions,
    print_metrics,
)

__all__ = [
    "compute_hit_rate",
    "compute_metrics",
    "compute_rank_metrics",
    "evaluate_predictions",
    "print_metrics",
]
