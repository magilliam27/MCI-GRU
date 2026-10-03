"""Selected-checkpoint metrics: Trainer.train reports the checkpoint it saved.

A tiny synthetic model replays a fixed validation score vector per trained epoch
and records in its state dict how many epochs produced it. Each test can then
read which epoch the saved checkpoint came from, and re-score that checkpoint
through the public ``load_best_model`` and ``predict`` API, independently of the
trainer's own bookkeeping.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.training.ensemble import train_multiple_models
from mci_gru.training.losses import (
    MaskedMSELoss,
    information_coefficient_sum_count,
    rank_information_coefficient_sum_count,
)
from mci_gru.training.trainer import NoCheckpointSelectedError, Trainer, TrainingResult

KDCODES = ["AAA", "BBB", "CCC", "DDD", "EEE"]
LABELS = [0.0, 0.1, 0.2, 0.3, 0.4]
NAN_LABELS = [float("nan")] * len(LABELS)

# Validation score scripts, with their (MSE loss, Pearson IC, rank IC) against LABELS.
LOW_LOSS = [0.1, 0.0, 0.2, 0.4, 0.3]  # (0.008, 0.80, 0.80)
HIGH_IC = [1.02, 1.0, 1.2, 1.3, 1.4]  # (0.970, 0.96, 0.90)
HIGH_RANK_IC = [1.0, 1.001, 1.002, 1.003, 2.0]  # (1.102, 0.71, 1.00)
SHIFTED_LOW_LOSS = [0.4, 0.3, 0.5, 0.7, 0.6]  # (0.098, 0.80, 0.80)
CONSTANT = [0.2] * 5  # (0.020, no eligible row, no eligible row)
FAR = [1.0, 1.1, 1.2, 1.3, 1.4]  # (1.000, 1.00, 1.00)
FLAT_BUT_ORDERED = [0.0, 1e-16, 2e-16, 3e-16, 4e-16]  # (0.060, no eligible row, 1.00)
REVERSED = [0.4, 0.3, 0.2, 0.1, 0.0]  # (0.100, -1.00, -1.00)
DIVERGED = [float("nan")] * 5  # (NaN, no eligible row, no eligible row)


class EpochScriptedModel(nn.Module):
    """Scores follow a per-epoch script; the state dict records the producing epoch."""

    def __init__(self, scores_by_epoch: list[list[float]]):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("scores_by_epoch", torch.tensor(scores_by_epoch))
        self.register_buffer("trained_epochs", torch.zeros((), dtype=torch.long))

    def forward(
        self,
        time_series: torch.Tensor,
        graph_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_stocks: int,
        edge_index_sector: torch.Tensor | None = None,
        edge_weight_sector: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del graph_features, edge_index, edge_weight, n_stocks
        del edge_index_sector, edge_weight_sector, stock_mask
        if self.training:
            self.trained_epochs += 1
        scores = self.scores_by_epoch[int(self.trained_epochs) - 1] + 0.0 * self.anchor
        return scores.unsqueeze(0).expand(time_series.shape[0], -1)


class ValidationScriptedModel(EpochScriptedModel):
    """Replays the script only in validation, so a NaN epoch leaves training finite.

    Training on a NaN script would make the parameter NaN and every later epoch NaN
    with it; this variant keeps later scripted epochs reachable.
    """

    def forward(self, time_series: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.training:
            self.trained_epochs += 1
            return (0.0 * self.anchor).expand(time_series.shape[0], len(LABELS))
        return super().forward(time_series, *args, **kwargs)


def _batch(labels: list[float], date: str) -> tuple:
    n_stocks = len(labels)
    return (
        torch.zeros((1, n_stocks, 3, 2)),
        torch.tensor([labels]),
        torch.zeros((1, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        [date],
    )


def _train(
    output_path: Path,
    scores_by_epoch: list[list[float]],
    val_loader: list[tuple],
    selection_metric: str,
    minimum_selection_rows: int = 1,
    model_class: type[EpochScriptedModel] = EpochScriptedModel,
) -> tuple[Trainer, TrainingResult, list[tuple]]:
    """Train one scripted epoch per score vector; also return every epoch callback."""
    config = _config(output_path, len(scores_by_epoch), selection_metric, minimum_selection_rows)
    trainer = Trainer(
        model=model_class(scores_by_epoch),
        config=config,
        device=torch.device("cpu"),
        output_path=str(output_path),
    )
    epochs: list[tuple] = []
    result = trainer.train(
        [_batch(LABELS, "2025-01-02")],
        val_loader,
        epoch_callback=lambda *values: epochs.append(values),
    )
    return trainer, result, epochs


def _config(
    output_path: Path,
    num_epochs: int,
    selection_metric: str,
    minimum_selection_rows: int = 1,
) -> ExperimentConfig:
    return ExperimentConfig(
        training=TrainingConfig(
            loss_type="mse",
            selection_metric=selection_metric,
            minimum_selection_rows=minimum_selection_rows,
            num_epochs=num_epochs,
            num_models=1,
            batch_size=1,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="checkpoint_metrics",
        output_dir=str(output_path),
    )


def _saved_epoch(result: TrainingResult) -> int:
    return int(torch.load(result.best_model_path, weights_only=True)["trained_epochs"])


def _reported(result: TrainingResult) -> tuple[float | None, float | None, float | None]:
    return (result.best_val_loss, result.best_val_ic, result.best_val_rank_ic)


def _rescore_saved_checkpoint(
    trainer: Trainer,
    result: TrainingResult,
    val_loader: list[tuple],
) -> tuple[float, float | None, float | None]:
    """Loss, IC and rank IC of the checkpoint on disk, re-scored on one validation row."""
    trainer.load_best_model(result.best_model_path)
    dates = [batch[6][0] for batch in val_loader]
    predictions = torch.as_tensor(trainer.predict(val_loader, KDCODES, dates))
    labels = torch.cat([batch[1] for batch in val_loader])
    ic_sum, ic_rows = information_coefficient_sum_count(predictions, labels)
    rank_ic_sum, rank_ic_rows = rank_information_coefficient_sum_count(predictions, labels)
    return (
        MaskedMSELoss()(predictions, labels).item(),
        ic_sum.item() / ic_rows if ic_rows else None,
        rank_ic_sum.item() / rank_ic_rows if rank_ic_rows else None,
    )


def test_val_loss_selection_without_eligible_ic_rows_reports_ic_as_missing(
    tmp_path: Path,
) -> None:
    """With no eligible IC row, IC co-metrics are None in the result and every callback."""
    val_loader = [_batch(NAN_LABELS, "2025-01-10")]

    _, result, epochs = _train(tmp_path, [LOW_LOSS, FAR], val_loader, "val_loss")

    assert _saved_epoch(result) == 1
    assert _reported(result) == (0.0, None, None)
    assert [values[6:] for values in epochs] == [(None, None), (None, None)]


def test_co_metrics_describe_the_saved_checkpoint_not_an_earlier_epoch(tmp_path: Path) -> None:
    """An earlier epoch's IC is never reported for a later saved checkpoint without one."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    trainer, result, epochs = _train(
        tmp_path,
        [SHIFTED_LOW_LOSS, CONSTANT, FAR],
        val_loader,
        "val_loss",
    )

    assert math.isfinite(epochs[0][3]), "fixture: the first improving epoch must have an IC"
    assert _saved_epoch(result) == 2
    assert _reported(result) == pytest.approx(
        _rescore_saved_checkpoint(trainer, result, val_loader)
    )
    assert (result.best_val_ic, result.best_val_rank_ic) == (None, None)


def test_rank_ic_selection_reports_ic_as_missing_when_the_saved_epoch_has_none(
    tmp_path: Path,
) -> None:
    """Both IC metrics: the saved epoch has a rank IC but no eligible Pearson IC row."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    trainer, result, epochs = _train(
        tmp_path,
        [LOW_LOSS, FLAT_BUT_ORDERED],
        val_loader,
        "val_rank_ic",
    )

    assert math.isnan(epochs[1][3]), "fixture: the saved epoch must have no Pearson IC"
    assert epochs[1][4] == pytest.approx(1.0), "fixture: the saved epoch must have a rank IC"
    assert _saved_epoch(result) == 2
    assert _reported(result) == pytest.approx(
        _rescore_saved_checkpoint(trainer, result, val_loader)
    )
    assert result.best_val_ic is None
    assert result.best_val_rank_ic == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("selection_metric", "expected_epoch"),
    [("val_loss", 1), ("val_ic", 2), ("val_rank_ic", 3)],
)
def test_each_selection_metric_saves_its_own_best_epoch_and_reports_it(
    tmp_path: Path,
    selection_metric: str,
    expected_epoch: int,
) -> None:
    """Selection rules are unchanged and the reported metrics are the saved epoch's own."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    trainer, result, _ = _train(
        tmp_path,
        [LOW_LOSS, HIGH_IC, HIGH_RANK_IC],
        val_loader,
        selection_metric,
    )

    assert result.epochs_trained == 3
    assert _saved_epoch(result) == expected_epoch
    assert _reported(result) == pytest.approx(
        _rescore_saved_checkpoint(trainer, result, val_loader)
    )


@pytest.mark.parametrize("selection_metric", ["val_ic", "val_rank_ic"])
def test_ic_selection_still_requires_minimum_selection_rows(
    tmp_path: Path,
    selection_metric: str,
) -> None:
    """Fail-closed coverage boundary: N-1 eligible rows raise and N rows select."""
    one_eligible_row = [_batch(LABELS, "2025-01-10"), _batch(NAN_LABELS, "2025-01-13")]
    two_eligible_rows = [_batch(LABELS, "2025-01-10"), _batch(LABELS, "2025-01-13")]

    with pytest.raises(ValueError, match="insufficient validation coverage"):
        _train(
            tmp_path / "short",
            [LOW_LOSS],
            one_eligible_row,
            selection_metric,
            minimum_selection_rows=2,
        )
    _, result, _ = _train(
        tmp_path / "enough",
        [LOW_LOSS],
        two_eligible_rows,
        selection_metric,
        minimum_selection_rows=2,
    )

    assert _saved_epoch(result) == 1


def test_selection_metric_that_is_never_finite_fails_closed(tmp_path: Path) -> None:
    """A run whose selection metric is NaN on every epoch raises and saves no checkpoint."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    with pytest.raises(NoCheckpointSelectedError, match=r"val_loss.*any of 2 epoch"):
        _train(
            tmp_path,
            [DIVERGED, DIVERGED],
            val_loader,
            "val_loss",
            model_class=ValidationScriptedModel,
        )

    assert not (tmp_path / "best_model.pth").exists()


def test_first_finite_epoch_is_selected_after_non_finite_ones(tmp_path: Path) -> None:
    """A NaN epoch never improves, before or after the first finite epoch, which is saved."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    _, result, epochs = _train(
        tmp_path,
        [DIVERGED, SHIFTED_LOW_LOSS, DIVERGED],
        val_loader,
        "val_loss",
        model_class=ValidationScriptedModel,
    )

    assert math.isnan(epochs[0][2]) and math.isnan(epochs[2][2]), "fixture: NaN epochs"
    assert _saved_epoch(result) == 2
    assert result.best_val_loss == pytest.approx(0.098)


@pytest.mark.parametrize("selection_metric", ["val_ic", "val_rank_ic"])
def test_ic_selection_saves_the_first_finite_epoch_however_negative(
    tmp_path: Path,
    selection_metric: str,
) -> None:
    """IC selection starts below every finite value, so an IC of -1 is still selected."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    _, result, _ = _train(tmp_path, [REVERSED], val_loader, selection_metric)

    assert _saved_epoch(result) == 1
    assert (result.best_val_ic, result.best_val_rank_ic) == pytest.approx((-1.0, -1.0))


@pytest.mark.parametrize(
    ("selection_metric", "script"),
    [("val_loss", LOW_LOSS), ("val_ic", HIGH_IC), ("val_rank_ic", HIGH_RANK_IC)],
)
def test_selection_tie_keeps_the_earlier_epoch(
    tmp_path: Path,
    selection_metric: str,
    script: list[float],
) -> None:
    """Only a strict improvement replaces the saved checkpoint."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    _, result, _ = _train(tmp_path, [script, script], val_loader, selection_metric)

    assert _saved_epoch(result) == 1


def test_ensemble_fails_the_run_when_a_member_selects_no_checkpoint(tmp_path: Path) -> None:
    """The member's error propagates, so no member or averaged predictions are written."""
    val_loader = [_batch(LABELS, "2025-01-10")]

    with pytest.raises(NoCheckpointSelectedError, match="val_loss"):
        train_multiple_models(
            model_factory=lambda: ValidationScriptedModel([DIVERGED]),
            config=_config(tmp_path, num_epochs=1, selection_metric="val_loss"),
            train_loader=[_batch(LABELS, "2025-01-02")],
            val_loader=val_loader,
            test_loader=val_loader,
            kdcode_list=KDCODES,
            test_dates=["2025-01-10"],
            output_path=str(tmp_path),
        )

    assert not (tmp_path / "predictions_model_0").exists()
    assert not (tmp_path / "averaged_predictions").exists()
