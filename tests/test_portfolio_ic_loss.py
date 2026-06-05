import torch

from mci_gru.config import TrainingConfig
from mci_gru.training.losses import (
    PortfolioICLoss,
    SoftTopKForwardReturnLoss,
    build_training_loss,
)


def test_portfolio_ic_loss_backpropagates_through_predictions() -> None:
    pred = torch.tensor([[0.10, 0.20, 0.30, 0.40]], requires_grad=True)
    target = torch.tensor([[0.40, -0.10, 0.20, 0.00]])

    loss = PortfolioICLoss(top_k=2, weight=0.25, temperature=0.50)
    value = loss(pred, target)

    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_soft_topk_loss_ignores_nan_pairs_and_degrades_when_topk_exceeds_valid_count() -> None:
    pred = torch.tensor([[0.10, float("nan"), 0.30, 0.40, -0.20]], requires_grad=True)
    target = torch.tensor([[0.20, 1.00, float("nan"), -0.10, 0.00]])

    loss = SoftTopKForwardReturnLoss(top_k=10, temperature=0.25)
    value = loss(pred, target)

    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_portfolio_ic_loss_handles_constant_predictions_and_labels_without_nan() -> None:
    pred = torch.ones((2, 5), requires_grad=True)
    target = torch.ones((2, 5))

    loss = PortfolioICLoss(top_k=10, weight=0.25, temperature=0.25)
    value = loss(pred, target)

    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_soft_topk_loss_prefers_better_top10_forward_return_ordering() -> None:
    labels = torch.tensor([[2.0] * 10 + [-5.0, -5.0]])
    good_scores = torch.tensor([[12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    bad_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 11.0]])

    loss = SoftTopKForwardReturnLoss(top_k=10, temperature=0.10)

    assert loss(good_scores, labels) < loss(bad_scores, labels)


def test_build_training_loss_keeps_trainer_free_of_loss_specific_branches() -> None:
    cfg = TrainingConfig(
        loss_type="portfolio_ic",
        portfolio_ic_top_k=5,
        portfolio_ic_weight=0.40,
        portfolio_ic_temperature=0.50,
    )

    criterion, label = build_training_loss(cfg)

    assert isinstance(criterion, PortfolioICLoss)
    assert label == "portfolio_ic (top_k=5, weight=0.4, temperature=0.5)"
