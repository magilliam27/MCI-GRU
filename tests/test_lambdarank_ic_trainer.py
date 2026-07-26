from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.evaluation.experiment_summary import select_training_objective_value
from mci_gru.training.trainer import Trainer


class TinyRankModel(nn.Module):
    def __init__(self, n_stocks: int):
        super().__init__()
        self.scores = nn.Parameter(torch.linspace(-0.20, 0.20, n_stocks))

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
        return self.scores.unsqueeze(0).expand(time_series.shape[0], -1)


def test_trainer_can_select_checkpoint_by_val_rank_ic(tmp_path: Path) -> None:
    n_stocks = 5
    batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.tensor([[0.00, 0.10, 0.20, 0.30, 0.40], [0.00, 0.10, 0.20, 0.30, 0.40]]),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-10", "2025-01-13"],
    )
    loader = [batch]
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="lambdarank_ic",
            selection_metric="val_rank_ic",
            lambdarank_ic_max_pairs_per_day=32,
            lambdarank_ic_temperature=1.0,
            num_epochs=1,
            num_models=1,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="lambdarank_ic_smoke",
        output_dir=str(tmp_path),
    )

    result = Trainer(
        model=TinyRankModel(n_stocks=n_stocks),
        config=config,
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    ).train(loader, loader)

    assert np.isfinite(result.final_train_loss)
    assert np.isfinite(result.best_val_rank_ic)
    assert result.best_val_rank_ic > 0
    assert Path(result.best_model_path).exists()


def test_trainer_validation_rank_ic_averages_only_rankable_rows(tmp_path: Path) -> None:
    n_stocks = 4
    rankable_batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.tensor([[0.00, 0.10, 0.20, 0.30], [float("nan"), float("nan"), 0.20, float("nan")]]),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-10", "2025-01-13"],
    )
    unrankable_batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.full((2, n_stocks), float("nan")),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-14", "2025-01-15"],
    )
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="lambdarank_ic",
            selection_metric="val_rank_ic",
            num_epochs=1,
            num_models=1,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="lambdarank_ic_rankable_mean",
        output_dir=str(tmp_path),
    )

    result = Trainer(
        model=TinyRankModel(n_stocks=n_stocks),
        config=config,
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    ).train([rankable_batch], [rankable_batch, unrankable_batch])

    assert result.best_val_rank_ic == 1.0


def test_trainer_rejects_checkpoint_selection_without_eligible_rank_rows(
    tmp_path: Path,
) -> None:
    n_stocks = 4
    train_batch = (
        torch.zeros((1, n_stocks, 3, 2)),
        torch.tensor([[0.00, 0.10, 0.20, 0.30]]),
        torch.zeros((1, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-10"],
    )
    invalid_val_batch = (
        torch.zeros((1, n_stocks, 3, 2)),
        torch.full((1, n_stocks), float("nan")),
        torch.zeros((1, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-13"],
    )
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="lambdarank_ic",
            selection_metric="val_rank_ic",
            num_epochs=1,
            num_models=1,
            lr_scheduler="none",
            use_amp=False,
        ),
        output_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="insufficient validation coverage"):
        Trainer(
            model=TinyRankModel(n_stocks=n_stocks),
            config=config,
            device=torch.device("cpu"),
            output_path=str(tmp_path),
        ).train([train_batch], [invalid_val_batch])


def test_select_training_objective_respects_selection_metric() -> None:
    final_summary = {
        "mean_best_val_loss": 0.7,
        "mean_best_val_ic": 0.1,
        "mean_best_val_rank_ic": 0.2,
    }
    merged_summary = {
        "mean_best_val_loss_across_windows": 0.6,
        "mean_best_val_ic_across_windows": 0.3,
        "mean_best_val_rank_ic_across_windows": 0.4,
    }

    assert select_training_objective_value("val_loss", [final_summary], None) == 0.7
    assert select_training_objective_value("val_ic", [final_summary], None) == 0.1
    assert select_training_objective_value("val_rank_ic", [final_summary], None) == 0.2
    assert select_training_objective_value("val_loss", [final_summary], merged_summary) == 0.6
    assert select_training_objective_value("val_ic", [final_summary], merged_summary) == 0.3
    assert select_training_objective_value("val_rank_ic", [final_summary], merged_summary) == 0.4
