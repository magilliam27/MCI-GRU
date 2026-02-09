"""
Custom loss functions for MCI-GRU training.

Provides ICLoss (Information Coefficient) and CombinedMSEICLoss for
ranking-aware training when predictions are used as a ranking signal.
"""

import torch
import torch.nn as nn


class ICLoss(nn.Module):
    """
    Negative Pearson correlation loss (Information Coefficient).

    Computes the cross-sectional Pearson correlation between predictions and
    targets per sample (per day across stocks), then returns the negative mean
    so that minimizing the loss maximizes IC.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (batch_size, n_stocks)
            target: Targets of shape (batch_size, n_stocks)

        Returns:
            Scalar loss (negative mean IC across batch).
        """
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)

        cov = (pred_centered * target_centered).sum(dim=-1)
        pred_std = pred_centered.norm(dim=-1) + self.eps
        target_std = target_centered.norm(dim=-1) + self.eps

        ic = cov / (pred_std * target_std)
        return -ic.mean()


class CombinedMSEICLoss(nn.Module):
    """
    Blends MSE and negative IC: (1 - alpha) * MSE + alpha * (-IC).

    At alpha=0 this is pure MSE; at alpha=1 it is pure IC loss.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ic_loss = ICLoss(eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (batch_size, n_stocks)
            target: Targets of shape (batch_size, n_stocks)

        Returns:
            Scalar combined loss.
        """
        mse_loss = self.mse(pred, target)
        ic_loss = self.ic_loss(pred, target)
        return (1.0 - self.alpha) * mse_loss + self.alpha * ic_loss
