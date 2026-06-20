import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.training.trainer import Trainer


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


class RecordingCriterion(nn.Module):
    def __init__(self, seen: list[torch.device]):
        super().__init__()
        self.seen = seen
        self.register_buffer("scale", torch.tensor(1.0))

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        device = self.scale.device
        self.seen.append(device)
        return result

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean() * self.scale


def _tiny_loader(n_stocks: int = 4):
    batch = (
        torch.zeros((2, n_stocks, 3, 2)),
        torch.tensor([[0.20, -0.10, 0.00, 0.10], [0.10, 0.00, -0.10, 0.20]]),
        torch.zeros((2, n_stocks, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-10", "2025-01-13"],
    )
    return [batch]


def test_trainer_moves_criterion_to_device(tmp_path: Path, monkeypatch) -> None:
    seen_devices: list[torch.device] = []

    def fake_build_training_loss(_training_cfg):
        return RecordingCriterion(seen_devices), "recording"

    monkeypatch.setattr("mci_gru.training.trainer.build_training_loss", fake_build_training_loss)
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="mse",
            selection_metric="val_loss",
            num_epochs=1,
            num_models=1,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="criterion_device_smoke",
        output_dir=str(tmp_path),
    )

    Trainer(
        model=TinyPortfolioModel(n_stocks=4),
        config=config,
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    ).train(_tiny_loader(), _tiny_loader())

    assert seen_devices == [torch.device("cpu")]


def test_trainer_writes_first_batch_profile_when_enabled(tmp_path: Path) -> None:
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="mse",
            selection_metric="val_loss",
            num_epochs=1,
            num_models=1,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=False,
            profile_batches=1,
        ),
        experiment_name="profile_smoke",
        output_dir=str(tmp_path),
    )

    Trainer(
        model=TinyPortfolioModel(n_stocks=4),
        config=config,
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    ).train(_tiny_loader(), _tiny_loader())

    profile_path = tmp_path / "training_step_profile.jsonl"
    rows = [json.loads(line) for line in profile_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["epoch"] == 0
    assert rows[0]["batch_index"] == 0
    assert rows[0]["device"] == "cpu"
    for key in [
        "data_wait_seconds",
        "h2d_seconds",
        "forward_loss_seconds",
        "backward_seconds",
        "optimizer_seconds",
    ]:
        assert key in rows[0]
        assert rows[0][key] >= 0.0
