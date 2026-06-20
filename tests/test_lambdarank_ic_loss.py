import torch

from mci_gru.config import TrainingConfig
from mci_gru.training.losses import (
    LambdaRankICLoss,
    build_training_loss,
    mean_rank_information_coefficient,
)


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
