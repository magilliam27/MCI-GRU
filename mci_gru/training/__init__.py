"""
Training and evaluation for MCI-GRU experiments.

Modules:
- trainer: Training loop with early stopping and ensemble averaging
- metrics: Evaluation metrics
- losses: Custom loss functions (ICLoss, CombinedMSEICLoss)
"""

__all__ = [
    "Trainer",
    "train_multiple_models",
    "compute_metrics",
    "evaluate_predictions",
    "ICLoss",
    "CombinedMSEICLoss",
]


def __getattr__(name):
    if name in {"CombinedMSEICLoss", "ICLoss"}:
        from mci_gru.training.losses import CombinedMSEICLoss, ICLoss

        return {"CombinedMSEICLoss": CombinedMSEICLoss, "ICLoss": ICLoss}[name]
    if name in {"compute_metrics", "evaluate_predictions"}:
        from mci_gru.training.metrics import compute_metrics, evaluate_predictions

        return {
            "compute_metrics": compute_metrics,
            "evaluate_predictions": evaluate_predictions,
        }[name]
    if name in {"Trainer", "train_multiple_models"}:
        from mci_gru.training.trainer import Trainer, train_multiple_models

        return {"Trainer": Trainer, "train_multiple_models": train_multiple_models}[name]
    raise AttributeError(name)
