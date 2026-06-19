from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.training.trainer import Trainer, train_multiple_models


class TinyPortfolioModel(nn.Module):
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


def test_trainer_can_run_one_cpu_step_with_portfolio_ic(tmp_path: Path) -> None:
    n_stocks = 5
    batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.tensor([[0.20, -0.10, 0.00, 0.10, -0.20], [0.10, 0.00, -0.10, 0.20, -0.20]]),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-10", "2025-01-13"],
    )
    loader = [batch]
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="portfolio_ic",
            portfolio_ic_top_k=2,
            portfolio_ic_weight=0.25,
            portfolio_ic_temperature=0.25,
            selection_metric="val_loss",
            num_epochs=1,
            num_models=1,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="portfolio_ic_smoke",
        output_dir=str(tmp_path),
    )
    model = TinyPortfolioModel(n_stocks=n_stocks)

    result = Trainer(
        model=model,
        config=config,
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    ).train(loader, loader)

    assert np.isfinite(result.final_train_loss)
    assert np.isfinite(result.best_val_loss)
    assert Path(result.best_model_path).exists()


def test_trainer_predict_accepts_batched_test_loader(tmp_path: Path) -> None:
    n_stocks = 4
    batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.zeros((2, n_stocks)),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-14", "2025-01-15"],
    )
    config = ExperimentConfig(
        training=TrainingConfig(use_amp=False),
        experiment_name="portfolio_ic_batched_predict",
        output_dir=str(tmp_path),
    )
    model = TinyPortfolioModel(n_stocks=n_stocks)

    predictions = Trainer(
        model=model,
        config=config,
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    ).predict([batch], [f"S{i}" for i in range(n_stocks)], ["2025-01-14", "2025-01-15"])

    assert predictions.shape == (2, n_stocks)


def test_train_multiple_models_can_skip_member_prediction_and_checkpoint_exports(
    tmp_path: Path,
) -> None:
    n_stocks = 4
    train_batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.tensor([[0.20, -0.10, 0.00, 0.10], [0.10, 0.00, -0.10, 0.20]]),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-10", "2025-01-13"],
    )
    test_batch = (
        torch.zeros((1, n_stocks, 3, 2)),
        torch.zeros((1, n_stocks)),
        torch.zeros((1, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-14"],
    )
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="portfolio_ic",
            portfolio_ic_top_k=2,
            selection_metric="val_loss",
            num_epochs=1,
            num_models=2,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
            save_member_predictions=False,
            save_checkpoints=False,
        ),
        experiment_name="portfolio_ic_skip_member_preds",
        output_dir=str(tmp_path),
    )

    train_multiple_models(
        model_factory=lambda: TinyPortfolioModel(n_stocks=n_stocks),
        config=config,
        train_loader=[train_batch],
        val_loader=[train_batch],
        test_loader=[test_batch],
        kdcode_list=[f"S{i}" for i in range(n_stocks)],
        test_dates=["2025-01-14"],
        output_path=str(tmp_path),
    )

    assert not (tmp_path / "predictions_model_0").exists()
    assert not (tmp_path / "predictions_model_1").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert (tmp_path / "averaged_predictions" / "2025-01-14.csv").is_file()
