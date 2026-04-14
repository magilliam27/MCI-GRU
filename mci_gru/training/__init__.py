"""
Training and evaluation for MCI-GRU experiments.

Modules:
- trainer: Training loop with early stopping and ensemble averaging
- metrics: Evaluation metrics
- losses: Custom loss functions (ICLoss, CombinedMSEICLoss)
"""

from mci_gru.training.losses import CombinedMSEICLoss, ICLoss
from mci_gru.training.metrics import compute_metrics
from mci_gru.training.trainer import Trainer, train_multiple_models

__all__ = [
    "Trainer",
    "train_multiple_models",
    "compute_metrics",
    "ICLoss",
    "CombinedMSEICLoss",
]
