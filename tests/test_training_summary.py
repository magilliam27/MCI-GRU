"""Training summaries average only available checkpoint metrics and count coverage.

Member level is ``build_training_summary`` (what ``run_experiment.py`` writes as
``training_summary.json``); window level is ``merge_walkforward_summary``. Missing
member slots stay null, all-unavailable means stay null, non-finite historical
values are unavailable, and every available window carries equal weight.
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.training.ensemble import train_multiple_models
from mci_gru.training.summary import build_training_summary
from mci_gru.training.trainer import TrainingResult
from mci_gru.walkforward import merge_walkforward_summary

N_STOCKS = 4
DATES = ["2025-01-10", "2025-01-13"]


def _result(loss: float | None, ic: float | None, rank_ic: float | None) -> TrainingResult:
    return TrainingResult(
        best_val_loss=loss,
        best_val_ic=ic,
        best_val_rank_ic=rank_ic,
        final_train_loss=0.0,
        epochs_trained=1,
        best_model_path="unused.pth",
    )


def _summary(*results: TrainingResult) -> dict:
    return build_training_summary(list(results), experiment_name="summary", walkforward_window=0)


def _strict_json(payload: dict) -> dict:
    """Round-trip through JSON that refuses NaN and infinities, as a strict reader would."""
    return json.loads(json.dumps(payload, allow_nan=False))


class RampModel(nn.Module):
    """Emits a fixed, non-constant score ramp so predictions are always rankable."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

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
        scores = torch.linspace(-0.2, 0.2, N_STOCKS) * self.scale
        return scores.unsqueeze(0).expand(time_series.shape[0], -1)


def _loader(labels: list[float]) -> list[tuple]:
    return [
        (
            torch.zeros((1, N_STOCKS, 3, 2)),
            torch.tensor([labels]),
            torch.zeros((1, N_STOCKS, 2)),
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            N_STOCKS,
            [date],
        )
        for date in DATES
    ]


def test_member_summary_averages_available_values_and_keeps_missing_slots() -> None:
    """Missing member slots stay null and each mean covers only the available members."""
    summary = _summary(
        _result(0.5, 0.1, 0.2),
        _result(0.6, None, None),
        _result(0.7, 0.3, 0.4),
    )

    assert summary["models_trained"] == 3
    assert summary["best_val_losses"] == [0.5, 0.6, 0.7]
    assert summary["best_val_ics"] == [0.1, None, 0.3]
    assert summary["best_val_rank_ics"] == [0.2, None, 0.4]
    assert summary["mean_best_val_loss"] == pytest.approx(0.6)
    assert summary["mean_best_val_ic"] == pytest.approx(0.2)
    assert summary["mean_best_val_rank_ic"] == pytest.approx(0.3)
    assert summary["member_coverage"] == {
        "best_val_loss": {"available": 3, "total": 3},
        "best_val_ic": {"available": 2, "total": 3},
        "best_val_rank_ic": {"available": 2, "total": 3},
    }


def test_member_summary_treats_non_finite_values_as_unavailable() -> None:
    """Historical sentinels such as -inf are unavailable rather than averaged."""
    summary = _summary(
        _result(math.inf, -math.inf, math.nan),
        _result(0.4, 0.2, -math.inf),
    )

    written = _strict_json(summary)
    assert written["best_val_losses"] == [None, 0.4]
    assert written["best_val_ics"] == [None, 0.2]
    assert written["best_val_rank_ics"] == [None, None]
    assert written["mean_best_val_loss"] == pytest.approx(0.4)
    assert written["mean_best_val_ic"] == pytest.approx(0.2)
    assert written["mean_best_val_rank_ic"] is None
    assert written["member_coverage"]["best_val_ic"] == {"available": 1, "total": 2}
    assert written["member_coverage"]["best_val_rank_ic"] == {"available": 0, "total": 2}


def test_member_summary_without_members_keeps_every_mean_missing() -> None:
    """An empty ensemble reports null means and zero coverage."""
    summary = _summary()

    assert summary["models_trained"] == 0
    assert summary["mean_best_val_loss"] is None
    assert summary["mean_best_val_ic"] is None
    assert summary["mean_best_val_rank_ic"] is None
    assert summary["member_coverage"]["best_val_loss"] == {"available": 0, "total": 0}


def test_saved_training_summary_is_null_not_inf_when_no_ic_row_is_eligible(
    tmp_path: Path,
) -> None:
    """End to end: val_loss selection with all-NaN validation labels writes null IC values."""
    config = ExperimentConfig(
        training=TrainingConfig(
            loss_type="mse",
            selection_metric="val_loss",
            num_epochs=2,
            num_models=1,
            batch_size=1,
            learning_rate=1e-4,
            lr_scheduler="none",
            use_amp=False,
        ),
        experiment_name="no_eligible_ic_rows",
        output_dir=str(tmp_path),
    )

    results, _ = train_multiple_models(
        model_factory=RampModel,
        config=config,
        train_loader=_loader([0.2, -0.1, 0.0, 0.1]),
        val_loader=_loader([float("nan")] * N_STOCKS),
        test_loader=_loader([0.2, -0.1, 0.0, 0.1]),
        kdcode_list=["AAA", "BBB", "CCC", "DDD"],
        test_dates=DATES,
        output_path=str(tmp_path),
    )
    written = _strict_json(
        build_training_summary(results, experiment_name="no_eligible_ic_rows", walkforward_window=0)
    )

    assert written["best_val_ics"] == [None]
    assert written["best_val_rank_ics"] == [None]
    assert written["mean_best_val_ic"] is None
    assert written["mean_best_val_rank_ic"] is None
    assert written["member_coverage"]["best_val_ic"] == {"available": 0, "total": 1}
    assert written["mean_best_val_loss"] == pytest.approx(0.0)


def test_walkforward_mean_weights_windows_equally_and_counts_windows() -> None:
    """Window coverage counts windows, not members, and each available window counts once."""
    windows = [
        _summary(_result(0.5, 0.1, 0.2)),
        _summary(_result(0.6, 0.3, None), _result(0.7, 0.5, None), _result(0.8, None, None)),
        _summary(_result(0.9, None, 0.6), _result(0.9, None, 0.6)),
    ]

    merged = merge_walkforward_summary(windows)

    assert merged["n_windows"] == 3
    assert merged["mean_best_val_ic_across_windows"] == pytest.approx((0.1 + 0.4) / 2)
    assert merged["mean_best_val_rank_ic_across_windows"] == pytest.approx((0.2 + 0.6) / 2)
    assert merged["mean_best_val_loss_across_windows"] == pytest.approx((0.5 + 0.7 + 0.9) / 3)
    assert merged["window_coverage"] == {
        "best_val_loss": {"available": 3, "total": 3},
        "best_val_ic": {"available": 2, "total": 3},
        "best_val_rank_ic": {"available": 2, "total": 3},
    }
    assert [w["member_coverage"]["best_val_ic"] for w in merged["windows"]] == [
        {"available": 1, "total": 1},
        {"available": 2, "total": 3},
        {"available": 0, "total": 2},
    ]


def test_walkforward_treats_historical_non_finite_window_means_as_unavailable() -> None:
    """A -inf or NaN window mean from an older run no longer poisons the aggregate."""
    historical = json.loads(
        '{"mean_best_val_loss": 0.5, "mean_best_val_ic": -Infinity, "mean_best_val_rank_ic": NaN}'
    )
    current = {"mean_best_val_loss": 0.7, "mean_best_val_ic": 0.3, "mean_best_val_rank_ic": None}

    merged = merge_walkforward_summary([historical, current])

    assert merged["mean_best_val_ic_across_windows"] == pytest.approx(0.3)
    assert merged["mean_best_val_rank_ic_across_windows"] is None
    assert merged["window_coverage"]["best_val_ic"] == {"available": 1, "total": 2}
    assert merged["window_coverage"]["best_val_rank_ic"] == {"available": 0, "total": 2}


def test_walkforward_summary_of_no_windows_stays_empty() -> None:
    """An empty walk-forward run has no aggregate at all."""
    assert merge_walkforward_summary([]) == {}
