"""
Custom loss functions for MCI-GRU training.

Provides ICLoss (Information Coefficient) and CombinedMSEICLoss for
ranking-aware training when predictions are used as a ranking signal.
"""

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """MSE over finite target/prediction pairs only."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if not mask.any():
            return pred.sum() * 0.0
        diff = pred[mask] - target[mask]
        return torch.mean(diff * diff)


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
        values = []
        for p, t in zip(pred, target, strict=True):
            mask = torch.isfinite(p) & torch.isfinite(t)
            if int(mask.sum().item()) < 2:
                continue
            p_valid = p[mask]
            t_valid = t[mask]
            p_centered = p_valid - p_valid.mean()
            t_centered = t_valid - t_valid.mean()
            denom = p_centered.norm() * t_centered.norm() + self.eps
            values.append((p_centered * t_centered).sum() / denom)
        if not values:
            return pred.sum() * 0.0
        return -torch.stack(values).mean()


def mean_information_coefficient(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Mean cross-sectional Pearson IC per batch row (same math as ``ICLoss``, positive = good)."""
    values = []
    for p, t in zip(pred, target, strict=True):
        mask = torch.isfinite(p) & torch.isfinite(t)
        if int(mask.sum().item()) < 2:
            continue
        p_valid = p[mask]
        t_valid = t[mask]
        p_centered = p_valid - p_valid.mean()
        t_centered = t_valid - t_valid.mean()
        denom = p_centered.norm() * t_centered.norm() + eps
        values.append((p_centered * t_centered).sum() / denom)
    if not values:
        return pred.sum() * 0.0
    return torch.stack(values).mean()


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
        self.mse = MaskedMSELoss()
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
