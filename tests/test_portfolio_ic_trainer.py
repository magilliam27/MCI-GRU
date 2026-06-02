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


def _write_prediction_dir(base: Path, model_id: int, rows_by_date: dict[str, list[tuple[str, float]]]) -> None:
    pred_dir = base / f"predictions_model_{model_id}"
    pred_dir.mkdir(parents=True)
    for date, rows in rows_by_date.items():
        lines = ["kdcode,dt,score"]
        lines.extend(f"{kdcode},{date},{score}" for kdcode, score in rows)
        (pred_dir / f"{date}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_train_multiple_models_reuses_complete_prediction_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    test_dates = ["2025-01-10", "2025-01-13"]
    kdcode_list = ["AAA", "BBB", "CCC"]
    _write_prediction_dir(
        tmp_path,
        0,
        {
            "2025-01-10": [("AAA", 0.10), ("BBB", 0.20)],
            "2025-01-13": [("BBB", 0.30), ("CCC", 0.40)],
        },
    )
    _write_prediction_dir(
        tmp_path,
        1,
        {
            "2025-01-10": [("AAA", 0.30), ("BBB", 0.60)],
            "2025-01-13": [("BBB", 0.70), ("CCC", 0.80)],
        },
    )
    config = ExperimentConfig(
        training=TrainingConfig(num_models=2, use_amp=False),
        experiment_name="resume_predictions",
        output_dir=str(tmp_path),
    )
    monkeypatch.setenv("MCI_GRU_RESUME_ENSEMBLE", "1")

    def model_factory():
        raise AssertionError("model_factory should not be called for complete saved predictions")

    results, averaged = train_multiple_models(
        model_factory=model_factory,
        config=config,
        train_loader=[],
        val_loader=[],
        test_loader=[],
        kdcode_list=kdcode_list,
        test_dates=test_dates,
        output_path=str(tmp_path),
    )

    assert [r.resumed_from_predictions for r in results] == [True, True]
    assert averaged.shape == (2, 3)
    np.testing.assert_allclose(averaged[0, :2], [0.20, 0.40])
    assert np.isnan(averaged[0, 2])
    np.testing.assert_allclose(averaged[1, 1:], [0.50, 0.60])
    assert (tmp_path / "averaged_predictions" / "2025-01-10.csv").exists()
    assert '"status": "OK"' in (tmp_path / "ensemble_progress.json").read_text(encoding="utf-8")
