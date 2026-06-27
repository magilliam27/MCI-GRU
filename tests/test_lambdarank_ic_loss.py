from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mci_gru.config import TrainingConfig
from mci_gru.training.losses import (
    LambdaRankICLoss,
    _average_ranks,
    _cap_pair_indices,
    _pair_cache_key,
    _standardize_cross_section,
    _zero_loss_like,
    build_training_loss,
    mean_rank_information_coefficient,
)


def _reference_lambdarank_ic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_pairs_per_day: int,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Old triu/filter/cap implementation used as an equivalence oracle."""
    losses = []
    for p, t in zip(pred, target, strict=True):
        mask = torch.isfinite(p) & torch.isfinite(t)
        valid_count = int(mask.sum().item())
        if valid_count < 2:
            continue

        score_z = _standardize_cross_section(p[mask], eps)
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
        left, right = _cap_pair_indices(left, right, max_pairs_per_day)
        label_diff = label_ranks[left] - label_ranks[right]
        direction = torch.sign(label_diff)
        score_diff = score_z[left] - score_z[right]

        pair_loss = F.softplus(-(direction * score_diff) / temperature)
        n_float = float(valid_count)
        pred_rank_sep = (pred_ranks[right] - pred_ranks[left]).abs()
        weights = (
            12.0
            * pred_rank_sep
            * label_diff.abs()
            / (n_float * (n_float * n_float - 1.0))
        ).detach()
        weight_sum = weights.sum()
        if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) <= eps:
            weights = (
                12.0
                * label_diff.abs()
                / (n_float * (n_float * n_float - 1.0))
            ).detach()
            weight_sum = weights.sum()
        if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) <= eps:
            continue
        losses.append((weights * pair_loss).sum() / (weight_sum + eps))

    if not losses:
        return _zero_loss_like(pred)
    return torch.stack(losses).mean()


def test_lambdarank_ic_loss_prefers_correct_full_rank_order() -> None:
    target = torch.tensor([[0.10, -0.20, 0.30, 0.00]])
    good_scores = torch.tensor([[0.20, -0.10, 0.40, 0.00]])
    reversed_scores = torch.tensor([[-0.20, 0.10, -0.40, 0.00]])

    loss = LambdaRankICLoss(max_pairs_per_day=100, temperature=1.0)

    assert loss(good_scores, target) < loss(reversed_scores, target)


def test_lambdarank_ic_loss_handles_nan_masks_and_backpropagates() -> None:
    pred = torch.tensor([[0.10, float("nan"), 0.30, 0.40, -0.20]], requires_grad=True)
    target = torch.tensor([[0.20, 1.00, float("nan"), -0.10, 0.00]])

    loss = LambdaRankICLoss(max_pairs_per_day=100, temperature=1.0)
    value = loss(pred, target)

    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_lambdarank_ic_loss_backpropagates_from_constant_rankable_scores() -> None:
    pred = torch.zeros((1, 5), requires_grad=True)
    target = torch.tensor([[0.40, -0.10, 0.20, 0.00, 0.30]])

    value = LambdaRankICLoss(max_pairs_per_day=100, temperature=1.0)(pred, target)

    assert torch.isfinite(value)
    assert value.item() > 0.0
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_lambdarank_ic_loss_caps_after_filtering_tied_label_pairs() -> None:
    pred = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    value = LambdaRankICLoss(max_pairs_per_day=2, temperature=1.0)(pred, target)

    assert torch.isfinite(value)
    assert value.item() > 0.0
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_lambdarank_ic_loss_returns_zero_like_for_unrankable_rows() -> None:
    pred = torch.tensor([[0.10, 0.20, float("nan")], [0.30, 0.40, 0.50]], requires_grad=True)
    target = torch.tensor([[float("nan"), 1.00, float("nan")], [1.00, 1.00, 1.00]])

    value = LambdaRankICLoss()(pred, target)

    assert torch.isfinite(value)
    assert value.item() == 0.0
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() == 0.0


def test_lambdarank_ic_pair_sampling_is_deterministic() -> None:
    pred = torch.tensor([[0.50, -0.20, 0.10, 0.70, -0.30, 0.00]], requires_grad=True)
    target = torch.tensor([[0.40, -0.10, 0.30, 0.20, -0.20, 0.10]])
    loss = LambdaRankICLoss(max_pairs_per_day=4, temperature=0.75)

    first = loss(pred, target)
    second = loss(pred, target)

    assert torch.isclose(first, second)


def test_lambdarank_ic_loss_reuses_cached_pair_indices() -> None:
    pred = torch.tensor([[0.50, -0.20, 0.10, 0.70, -0.30, 0.00]], requires_grad=True)
    target = torch.tensor([[0.40, -0.10, 0.30, 0.20, -0.20, 0.10]])
    loss = LambdaRankICLoss(max_pairs_per_day=4, temperature=0.75)

    first = loss(pred, target)
    cache_key = _pair_cache_key(6, pred.device)
    cached_rows, cached_cols = loss._capped_pair_index_cache[cache_key]
    second = loss(pred, target)

    assert torch.isclose(first, second)
    assert loss._capped_pair_index_cache[cache_key][0] is cached_rows
    assert loss._capped_pair_index_cache[cache_key][1] is cached_cols


@pytest.mark.parametrize("max_pairs", [2, 4, 16, 4096])
def test_lambdarank_ic_loss_matches_reference_triu_filter_cap(max_pairs: int) -> None:
    pred = torch.tensor(
        [
            [0.50, -0.20, 0.10, 0.70, -0.30, 0.00],
            [0.10, float("nan"), 0.40, -0.10, 0.20, -0.20],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [0.40, -0.10, 0.30, 0.20, -0.20, 0.10],
            [0.00, 0.30, float("nan"), 0.30, -0.10, -0.10],
            [0.50, -0.20, 0.10, 0.00, 0.30, -0.10],
        ]
    )

    optimized = LambdaRankICLoss(max_pairs_per_day=max_pairs, temperature=0.75)(pred, target)
    reference = _reference_lambdarank_ic_loss(
        pred,
        target,
        max_pairs_per_day=max_pairs,
        temperature=0.75,
    )

    assert torch.allclose(optimized, reference, atol=1e-7, rtol=1e-6)

    opt_grad = torch.autograd.grad(optimized, pred, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(reference, pred)[0]
    assert torch.allclose(opt_grad, ref_grad, atol=1e-7, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lambdarank_ic_loss_cpu_cuda_equivalence() -> None:
    torch.manual_seed(23)
    pred_cpu = torch.randn(3, 32, requires_grad=True)
    target_cpu = torch.randn(3, 32)
    target_cpu[:, ::7] = float("nan")
    loss_fn = LambdaRankICLoss(max_pairs_per_day=64, temperature=0.9)

    cpu_value = loss_fn(pred_cpu, target_cpu)
    cpu_grad = torch.autograd.grad(cpu_value, pred_cpu)[0]

    pred_cuda = pred_cpu.detach().clone().cuda().requires_grad_(True)
    target_cuda = target_cpu.cuda()
    cuda_value = loss_fn(pred_cuda, target_cuda)
    cuda_grad = torch.autograd.grad(cuda_value, pred_cuda)[0].cpu()

    assert torch.allclose(cuda_value.cpu(), cpu_value, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cuda_grad, cpu_grad, atol=1e-6, rtol=1e-6)


def test_average_ranks_handles_ties_without_cpu_scalar_loop_regression() -> None:
    ranks = _average_ranks(torch.tensor([3.0, 1.0, 1.0, 2.0, 3.0]))

    assert torch.allclose(ranks, torch.tensor([3.5, 0.5, 0.5, 2.0, 3.5]))


def test_lambdarank_ic_benchmark_script_is_checked_in() -> None:
    assert Path("scripts/benchmark_lambdarank_ic_loss.py").exists()


def test_mean_rank_information_coefficient_uses_spearman_ordering() -> None:
    pred = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    target = torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])

    value = mean_rank_information_coefficient(pred, target)

    assert torch.isclose(value, torch.tensor(0.0))


def test_build_training_loss_constructs_lambdarank_ic_loss() -> None:
    cfg = TrainingConfig(
        loss_type="lambdarank_ic",
        lambdarank_ic_max_pairs_per_day=256,
        lambdarank_ic_temperature=0.50,
    )

    criterion, label = build_training_loss(cfg)

    assert isinstance(criterion, LambdaRankICLoss)
    assert label == "lambdarank_ic (max_pairs_per_day=256, temperature=0.5)"
