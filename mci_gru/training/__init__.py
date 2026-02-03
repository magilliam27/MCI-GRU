"""
Training and evaluation for MCI-GRU experiments.

Modules:
- trainer: Training loop with dynamic graph support
- metrics: Evaluation metrics
"""

from mci_gru.training.trainer import Trainer, train_multiple_models
from mci_gru.training.metrics import compute_metrics

__all__ = [
    "Trainer",
    "train_multiple_models",
    "compute_metrics",
]
