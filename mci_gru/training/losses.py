"""
Custom loss functions for MCI-GRU training.

Provides ICLoss (Information Coefficient) and CombinedMSEICLoss for
ranking-aware training when predictions are used as a ranking signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from mci_gru.config import TrainingConfig


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


def _zero_loss_like(pred: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def _standardize_cross_section(values: torch.Tensor, eps: float) -> torch.Tensor:
    centered = values - values.mean()
    scale = centered.pow(2).mean().sqrt()
    if (not torch.isfinite(scale)) or float(scale.detach().item()) <= eps:
        return centered
    return centered / (scale + eps)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Zero-based average ranks for a 1-D tensor; ties receive their mean rank."""
    order = torch.argsort(values)
    ranks = torch.empty(values.numel(), dtype=values.dtype, device=values.device)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _deterministic_pair_indices(
    n_items: int,
    max_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = torch.triu_indices(n_items, n_items, offset=1, device=device)
    return _cap_pair_indices(rows, cols, max_pairs)


def _cap_pair_indices(
    rows: torch.Tensor,
    cols: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rows.numel() <= max_pairs:
        return rows, cols
    positions = torch.linspace(
        0,
        rows.numel() - 1,
        steps=max_pairs,
        device=rows.device,
    ).round()
    selected = positions.to(dtype=torch.long)
    return rows[selected], cols[selected]


class LambdaRankICLoss(nn.Module):
    """
    Pairwise LambdaRankIC-style surrogate over same-date cross-sections.

    Predictions are standardized per date before the logistic ordering term.
    Pair weights use detached prediction ranks and label ranks, so gradients flow
    through score ordering but not through non-differentiable rank assignment.
    """

    def __init__(
        self,
        max_pairs_per_day: int = 4096,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if max_pairs_per_day <= 0:
            raise ValueError("max_pairs_per_day must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.max_pairs_per_day = max_pairs_per_day
        self.temperature = temperature
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for p, t in zip(pred, target, strict=True):
            mask = torch.isfinite(p) & torch.isfinite(t)
            valid_count = int(mask.sum().item())
            if valid_count < 2:
                continue

            score_z = _standardize_cross_section(p[mask], self.eps)
            label_ranks = _average_ranks(t[mask].detach())
            pred_ranks = _average_ranks(score_z.detach())
            left, right = torch.triu_indices(
                valid_count,
                valid_count,
                offset=1,
                device=score_z.device,
            )

            label_diff = label_ranks[left] - label_ranks[right]
            ordered = label_diff != 0
            if not bool(ordered.any()):
                continue

            left = left[ordered]
            right = right[ordered]
            label_diff = label_diff[ordered]
            left, right = _cap_pair_indices(left, right, self.max_pairs_per_day)
            label_diff = label_ranks[left] - label_ranks[right]
            direction = torch.sign(label_diff)
            score_diff = score_z[left] - score_z[right]

            pair_loss = F.softplus(-(direction * score_diff) / self.temperature)
            n_float = float(valid_count)
            pred_rank_sep = (pred_ranks[right] - pred_ranks[left]).abs()
            weights = (
                12.0
                * pred_rank_sep
                * label_diff.abs()
                / (n_float * (n_float * n_float - 1.0))
            ).detach()
            weight_sum = weights.sum()
            if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) <= self.eps:
                weights = (
                    12.0
                    * label_diff.abs()
                    / (n_float * (n_float * n_float - 1.0))
                ).detach()
                weight_sum = weights.sum()
            if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) <= self.eps:
                continue
            losses.append((weights * pair_loss).sum() / (weight_sum + self.eps))

        if not losses:
            return _zero_loss_like(pred)
        return torch.stack(losses).mean()


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


def information_coefficient_sum_count(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, int]:
    """Sum of rankable-row Pearson IC values and the number of rows included."""
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
        if float(denom.detach().item()) <= eps:
            continue
        values.append((p_centered * t_centered).sum() / denom)
    if not values:
        return _zero_loss_like(pred), 0
    return torch.stack(values).sum(), len(values)


def mean_rank_information_coefficient(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Mean cross-sectional Spearman Rank IC per batch row."""
    values = []
    for p, t in zip(pred, target, strict=True):
        mask = torch.isfinite(p) & torch.isfinite(t)
        if int(mask.sum().item()) < 2:
            continue
        p_ranks = _average_ranks(p[mask])
        t_ranks = _average_ranks(t[mask])
        p_centered = p_ranks - p_ranks.mean()
        t_centered = t_ranks - t_ranks.mean()
        denom = p_centered.norm() * t_centered.norm() + eps
        if float(denom.detach().item()) <= eps:
            continue
        values.append((p_centered * t_centered).sum() / denom)
    if not values:
        return _zero_loss_like(pred)
    return torch.stack(values).mean()


def rank_information_coefficient_sum_count(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, int]:
    """Sum of rankable-row Spearman Rank IC values and the number of rows included."""
    values = []
    for p, t in zip(pred, target, strict=True):
        mask = torch.isfinite(p) & torch.isfinite(t)
        if int(mask.sum().item()) < 2:
            continue
        p_ranks = _average_ranks(p[mask])
        t_ranks = _average_ranks(t[mask])
        p_centered = p_ranks - p_ranks.mean()
        t_centered = t_ranks - t_ranks.mean()
        denom = p_centered.norm() * t_centered.norm() + eps
        if float(denom.detach().item()) <= eps:
            continue
        values.append((p_centered * t_centered).sum() / denom)
    if not values:
        return _zero_loss_like(pred), 0
    return torch.stack(values).sum(), len(values)


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


def build_training_loss(training_cfg: TrainingConfig) -> tuple[nn.Module, str]:
    """Build the configured training criterion and display label."""
    if training_cfg.loss_type == "ic":
        return ICLoss(), "ic"
    if training_cfg.loss_type == "combined":
        return (
            CombinedMSEICLoss(alpha=training_cfg.ic_loss_alpha),
            f"combined (alpha={training_cfg.ic_loss_alpha})",
        )
    if training_cfg.loss_type == "portfolio_ic":
        return (
            PortfolioICLoss(
                top_k=training_cfg.portfolio_ic_top_k,
                weight=training_cfg.portfolio_ic_weight,
                temperature=training_cfg.portfolio_ic_temperature,
            ),
            (
                f"portfolio_ic (top_k={training_cfg.portfolio_ic_top_k}, "
                f"weight={training_cfg.portfolio_ic_weight}, "
                f"temperature={training_cfg.portfolio_ic_temperature})"
            ),
        )
    if training_cfg.loss_type == "lambdarank_ic":
        return (
            LambdaRankICLoss(
                max_pairs_per_day=training_cfg.lambdarank_ic_max_pairs_per_day,
                temperature=training_cfg.lambdarank_ic_temperature,
            ),
            (
                "lambdarank_ic "
                f"(max_pairs_per_day={training_cfg.lambdarank_ic_max_pairs_per_day}, "
                f"temperature={training_cfg.lambdarank_ic_temperature})"
            ),
        )
    return MaskedMSELoss(), "mse"
