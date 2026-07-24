"""Ensemble invariant: prediction = mean of independently trained models.

Protects AGENTS.md invariant 4: ``train_multiple_models`` trains N independent
models and the final prediction is the member mean. Uses tiny deterministic
CPU models so member predictions are known and distinct.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.training.ensemble import train_multiple_models

N_STOCKS = 4
KDCODES = ["AAA", "BBB", "CCC", "DDD"]
TEST_DATES = ["2025-01-10", "2025-01-13"]


class FixedScoreModel(nn.Module):
    """Emits a fixed per-stock score vector scaled by one trainable parameter."""

    def __init__(self, offset: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("base", torch.linspace(-0.2, 0.2, N_STOCKS) + offset)

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
        return (self.base * self.scale).unsqueeze(0).expand(time_series.shape[0], -1)


def _loader():
    """One batch of size 1 per test date, matching Trainer.predict's contract
    that each test batch squeezes to a single (n_stocks,) prediction row."""
    labels = [[0.20, -0.10, 0.00, 0.10], [0.10, 0.00, -0.10, 0.20]]
    return [
        (
            torch.zeros((1, N_STOCKS, 3, 2)),
            torch.tensor([labels[i]]),
            torch.zeros((1, N_STOCKS, 2)),
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            N_STOCKS,
            [TEST_DATES[i]],
        )
        for i in range(len(TEST_DATES))
    ]


def _config(tmp_path: Path, num_models: int) -> ExperimentConfig:
    return ExperimentConfig(
        training=TrainingConfig(
            loss_type="mse",
            selection_metric="val_loss",
            num_epochs=1,
            num_models=num_models,
            batch_size=2,
            learning_rate=1e-4,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="ensemble_averaging_smoke",
        output_dir=str(tmp_path),
    )


def _read_scores(pred_dir: Path, date: str) -> dict[str, float]:
    df = pd.read_csv(pred_dir / f"{date}.csv")
    return dict(zip(df["kdcode"], df["score"], strict=True))


def test_ensemble_prediction_is_mean_of_member_predictions(tmp_path: Path) -> None:
    offsets = iter([0.0, 1.0])

    def model_factory() -> nn.Module:
        return FixedScoreModel(offset=next(offsets))

    results, avg_predictions = train_multiple_models(
        model_factory=model_factory,
        config=_config(tmp_path, num_models=2),
        train_loader=_loader(),
        val_loader=_loader(),
        test_loader=_loader(),
        kdcode_list=KDCODES,
        test_dates=TEST_DATES,
        output_path=str(tmp_path),
    )

    assert len(results) == 2
    assert avg_predictions.shape == (len(TEST_DATES), N_STOCKS)

    # Averaged prediction must equal the member mean, reconstructed from the
    # per-model CSV exports (scores rounded to 5 decimals -> loose tolerance).
    for date in TEST_DATES:
        member_0 = _read_scores(tmp_path / "predictions_model_0", date)
        member_1 = _read_scores(tmp_path / "predictions_model_1", date)
        averaged = _read_scores(tmp_path / "averaged_predictions", date)
        assert set(averaged) == set(KDCODES)
        for kdcode in KDCODES:
            expected = (member_0[kdcode] + member_1[kdcode]) / 2.0
            assert abs(averaged[kdcode] - expected) < 1e-4, (
                f"{date}/{kdcode}: averaged {averaged[kdcode]} != mean {expected}"
            )

    # Members with different offsets must produce different predictions,
    # otherwise the averaging assertion above would be vacuous.
    assert member_0[KDCODES[0]] != member_1[KDCODES[0]]


def test_ensemble_writes_one_checkpoint_per_member(tmp_path: Path) -> None:
    offsets = iter([0.0, 1.0])

    def model_factory() -> nn.Module:
        return FixedScoreModel(offset=next(offsets))

    train_multiple_models(
        model_factory=model_factory,
        config=_config(tmp_path, num_models=2),
        train_loader=_loader(),
        val_loader=_loader(),
        test_loader=_loader(),
        kdcode_list=KDCODES,
        test_dates=TEST_DATES,
        output_path=str(tmp_path),
    )

    checkpoints = sorted(p.name for p in (tmp_path / "checkpoints").glob("model_*_best.pth"))
    assert checkpoints == ["model_0_best.pth", "model_1_best.pth"]

    avg_files = sorted(p.name for p in (tmp_path / "averaged_predictions").glob("*.csv"))
    assert avg_files == [f"{d}.csv" for d in TEST_DATES]


def test_ensemble_averaging_matches_numpy_mean(tmp_path: Path) -> None:
    """avg_predictions returned in-memory equals the numpy mean of member outputs."""
    offsets = iter([0.0, 1.0, 2.0])

    def model_factory() -> nn.Module:
        return FixedScoreModel(offset=next(offsets))

    _, avg_predictions = train_multiple_models(
        model_factory=model_factory,
        config=_config(tmp_path, num_models=3),
        train_loader=_loader(),
        val_loader=_loader(),
        test_loader=_loader(),
        kdcode_list=KDCODES,
        test_dates=TEST_DATES,
        output_path=str(tmp_path),
    )

    members = []
    for model_id in range(3):
        rows = [
            _read_scores(tmp_path / f"predictions_model_{model_id}", date) for date in TEST_DATES
        ]
        members.append([[row[k] for k in KDCODES] for row in rows])
    expected = np.mean(np.array(members), axis=0)
    np.testing.assert_allclose(avg_predictions, expected, atol=1e-4)
