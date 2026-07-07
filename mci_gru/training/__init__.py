"""
Training and evaluation for MCI-GRU experiments.

Modules:
- trainer: Training loop with early stopping and ensemble averaging
- metrics: Evaluation metrics
- losses: Custom loss functions and loss factory
"""

__all__ = [
    "Trainer",
    "train_multiple_models",
    "compute_metrics",
    "evaluate_predictions",
    "ICLoss",
    "CombinedMSEICLoss",
    "LambdaRankICLoss",
    "PortfolioICLoss",
    "SoftTopKForwardReturnLoss",
    "build_training_loss",
    "information_coefficient_sum_count",
    "mean_rank_information_coefficient",
    "rank_information_coefficient_sum_count",
]


def __getattr__(name):
    if name in {
        "CombinedMSEICLoss",
        "ICLoss",
        "LambdaRankICLoss",
        "PortfolioICLoss",
        "SoftTopKForwardReturnLoss",
        "build_training_loss",
        "information_coefficient_sum_count",
        "mean_rank_information_coefficient",
        "rank_information_coefficient_sum_count",
    }:
        from mci_gru.training.losses import (
            CombinedMSEICLoss,
            ICLoss,
            LambdaRankICLoss,
            PortfolioICLoss,
            SoftTopKForwardReturnLoss,
            build_training_loss,
            information_coefficient_sum_count,
            mean_rank_information_coefficient,
            rank_information_coefficient_sum_count,
        )

        return {
            "CombinedMSEICLoss": CombinedMSEICLoss,
            "ICLoss": ICLoss,
            "LambdaRankICLoss": LambdaRankICLoss,
            "PortfolioICLoss": PortfolioICLoss,
            "SoftTopKForwardReturnLoss": SoftTopKForwardReturnLoss,
            "build_training_loss": build_training_loss,
            "information_coefficient_sum_count": information_coefficient_sum_count,
            "mean_rank_information_coefficient": mean_rank_information_coefficient,
            "rank_information_coefficient_sum_count": rank_information_coefficient_sum_count,
        }[name]
    if name in {"compute_metrics", "evaluate_predictions"}:
        from mci_gru.training.metrics import compute_metrics, evaluate_predictions

        return {
            "compute_metrics": compute_metrics,
            "evaluate_predictions": evaluate_predictions,
        }[name]
    if name == "Trainer":
        from mci_gru.training.trainer import Trainer

        return Trainer
    if name == "train_multiple_models":
        from mci_gru.training.ensemble import train_multiple_models

        return train_multiple_models
    raise AttributeError(name)
