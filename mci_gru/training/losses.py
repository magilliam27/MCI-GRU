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


def _standardize_cross_section(values: torch.Tensor, eps: float) -> torch.Tensor:
    centered = values - values.mean()
    scale = centered.pow(2).mean().sqrt()
    if (not torch.isfinite(scale)) or float(scale.detach().item()) <= eps:
        return centered
    return centered / (scale + eps)


class SoftTopKForwardReturnLoss(nn.Module):
    """
    Negative differentiable soft top-k utility over standardized forward returns.

    The kth score threshold comes from ``torch.topk`` and is detached; gradients
    still flow through each prediction's soft inclusion weight.
    """

    def __init__(self, top_k: int = 10, temperature: float = 0.25, eps: float = 1e-8):
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.top_k = top_k
        self.temperature = temperature
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (batch_size, n_stocks)
            target: Forward-return labels of shape (batch_size, n_stocks)

        Returns:
            Scalar loss (negative mean soft top-k standardized return utility).
        """
        utilities = []
        for p, t in zip(pred, target, strict=True):
            mask = torch.isfinite(p) & torch.isfinite(t)
            valid_count = int(mask.sum().item())
            if valid_count == 0:
                continue

            score_z = _standardize_cross_section(p[mask], self.eps)
            target_z = _standardize_cross_section(t[mask], self.eps)
            k_eff = min(self.top_k, valid_count)
            kth_score = torch.topk(score_z, k_eff).values[-1].detach()
            inclusion = torch.sigmoid((score_z - kth_score) / self.temperature)
            weights = inclusion / (inclusion.sum() + self.eps)
            utilities.append((weights * target_z).sum())

        if not utilities:
            return pred.sum() * 0.0
        return -torch.stack(utilities).mean()


class PortfolioICLoss(nn.Module):
    """
    Hybrid loss anchored on IC with a soft top-k forward-return utility term.

    ``(1 - weight) * ICLoss + weight * SoftTopKForwardReturnLoss``
    """

    def __init__(
        self,
        top_k: int = 10,
        weight: float = 0.25,
        temperature: float = 0.25,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not 0 <= weight <= 1:
            raise ValueError("weight must be in [0, 1]")
        self.weight = weight
        self.ic_loss = ICLoss(eps=eps)
        self.portfolio_loss = SoftTopKForwardReturnLoss(
            top_k=top_k,
            temperature=temperature,
            eps=eps,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ic_loss = self.ic_loss(pred, target)
        portfolio_loss = self.portfolio_loss(pred, target)
        return (1.0 - self.weight) * ic_loss + self.weight * portfolio_loss


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
