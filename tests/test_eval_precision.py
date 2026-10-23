"""Scoring precision: training may use AMP, validation and prediction do not.

The first run predicted under CUDA autocast, so each member's scores sat on an
fp16 grid and unrelated names tied (#321, PRD #319 decision D9). The CPU tests
make the trainer believe it is on CUDA, swap the trainer's ``autocast`` for a
recorder, and read back which forward passes ran with autocast enabled. The
CUDA test checks the real numbers when a GPU is present.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import mci_gru.training.trainer as trainer_module
from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.training.trainer import Trainer

N_STOCKS = 5
LABELS = [0.0, 0.1, 0.2, 0.3, 0.4]


class AutocastRecorder:
    """Stands in for ``torch.amp.autocast``; records the ``enabled`` flag in force."""

    def __init__(self) -> None:
        self.stack: list[bool] = []

    @contextmanager
    def __call__(self, device_type: str, enabled: bool = True, **_kwargs) -> Iterator[None]:
        assert device_type == "cuda"
        self.stack.append(enabled)
        try:
            yield
        finally:
            self.stack.pop()

    @property
    def active(self) -> bool:
        return bool(self.stack) and self.stack[-1]


class PhaseRecordingModel(nn.Module):
    """Records (training?, autocast enabled?) for every forward pass."""

    def __init__(self, recorder: AutocastRecorder):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.recorder = recorder
        self.calls: list[tuple[bool, bool]] = []

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
        del time_series, edge_index, edge_weight, n_stocks
        del edge_index_sector, edge_weight_sector, stock_mask
        self.calls.append((self.training, self.recorder.active))
        return self.linear(graph_features).squeeze(-1)


def _batch(date: str) -> tuple:
    torch.manual_seed(0)
    return (
        torch.zeros((1, N_STOCKS, 3, 2)),
        torch.tensor([LABELS]),
        torch.randn((1, N_STOCKS, 2)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        N_STOCKS,
        [date],
    )


def _config(output_path: Path, use_amp: bool, eval_amp: bool | None) -> ExperimentConfig:
    training_kwargs = dict(
        loss_type="mse",
        selection_metric="val_loss",
        num_epochs=2,
        num_models=1,
        batch_size=1,
        learning_rate=1e-3,
        lr_scheduler="none",
        use_amp=use_amp,
    )
    if eval_amp is not None:
        training_kwargs["eval_amp"] = eval_amp
    return ExperimentConfig(
        training=TrainingConfig(**training_kwargs),
        experiment_name="eval_precision",
        output_dir=str(output_path),
    )


@pytest.fixture
def fake_cuda(monkeypatch) -> AutocastRecorder:
    """Route autocast to a recorder and keep batches on the CPU."""
    recorder = AutocastRecorder()
    unpack = trainer_module._unpack_loader_batch
    monkeypatch.setattr(trainer_module, "autocast", recorder)
    monkeypatch.setattr(
        trainer_module,
        "_unpack_loader_batch",
        lambda batch, device: unpack(batch, torch.device("cpu")),
    )
    monkeypatch.setattr(
        trainer_module,
        "GradScaler",
        lambda device_type, enabled=True: torch.amp.GradScaler("cpu", enabled=False),
    )
    return recorder


def _trainer_on_fake_cuda(
    tmp_path: Path, recorder: AutocastRecorder, use_amp: bool, eval_amp: bool | None
) -> tuple[Trainer, PhaseRecordingModel]:
    model = PhaseRecordingModel(recorder)
    trainer = Trainer(
        model=model,
        config=_config(tmp_path, use_amp, eval_amp),
        device=torch.device("cpu"),
        output_path=str(tmp_path),
    )
    # Only the AMP decision reads the device type; tensors stay on the CPU.
    trainer.device = torch.device("cuda")
    return trainer, model


def test_default_config_scores_in_full_precision() -> None:
    assert TrainingConfig().eval_amp is False


def test_training_keeps_amp_while_validation_runs_in_full_precision(
    tmp_path: Path, fake_cuda: AutocastRecorder
) -> None:
    trainer, model = _trainer_on_fake_cuda(tmp_path, fake_cuda, use_amp=True, eval_amp=None)

    trainer.train([_batch("2025-01-02")], [_batch("2025-01-03")])

    train_passes = {autocast for training, autocast in model.calls if training}
    validation_passes = {autocast for training, autocast in model.calls if not training}
    assert train_passes == {True}
    assert validation_passes == {False}


def test_prediction_runs_in_full_precision(tmp_path: Path, fake_cuda: AutocastRecorder) -> None:
    trainer, model = _trainer_on_fake_cuda(tmp_path, fake_cuda, use_amp=True, eval_amp=None)

    predictions = trainer.predict([_batch("2025-01-03")], ["A"] * N_STOCKS, ["2025-01-03"])

    assert predictions.shape == (1, N_STOCKS)
    assert model.calls == [(False, False)]


def test_eval_amp_opts_scoring_back_into_autocast(
    tmp_path: Path, fake_cuda: AutocastRecorder
) -> None:
    """The control: the recorder does see autocast when scoring asks for it."""
    trainer, model = _trainer_on_fake_cuda(tmp_path, fake_cuda, use_amp=True, eval_amp=True)

    trainer.train([_batch("2025-01-02")], [_batch("2025-01-03")])
    trainer.predict([_batch("2025-01-03")], ["A"] * N_STOCKS, ["2025-01-03"])

    assert {autocast for _training, autocast in model.calls} == {True}


def test_eval_amp_needs_a_cuda_device(tmp_path: Path, fake_cuda: AutocastRecorder) -> None:
    trainer, model = _trainer_on_fake_cuda(tmp_path, fake_cuda, use_amp=True, eval_amp=True)
    trainer.device = torch.device("cpu")

    trainer.predict([_batch("2025-01-03")], ["A"] * N_STOCKS, ["2025-01-03"])

    assert model.calls == [(False, False)]


class SmallNet(nn.Module):
    """Enough matmuls that fp16 rounding is visible in the scores."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, time_series, graph_features, edge_index, edge_weight, n_stocks, **_kwargs):
        return self.net(graph_features).squeeze(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_prediction_matches_fp32_to_1e6(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = SmallNet()
    reference_state = {k: v.clone() for k, v in model.state_dict().items()}
    n_stocks = 110
    batch = (
        torch.zeros((1, n_stocks, 3, 2)),
        torch.zeros((1, n_stocks)),
        torch.randn((1, n_stocks, 16)),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,)),
        n_stocks,
        ["2025-01-03"],
    )
    device = torch.device("cuda")

    def predict(use_amp: bool, eval_amp: bool | None):
        net = SmallNet()
        net.load_state_dict(reference_state)
        trainer = Trainer(
            model=net,
            config=_config(tmp_path, use_amp, eval_amp),
            device=device,
            output_path=str(tmp_path),
        )
        return trainer.predict([batch], ["A"] * n_stocks, ["2025-01-03"])

    fp32 = predict(use_amp=False, eval_amp=None)
    amp_training_run = predict(use_amp=True, eval_amp=None)
    half_precision_scoring = predict(use_amp=True, eval_amp=True)

    assert abs(amp_training_run - fp32).max() <= 1e-6
    # Control: the comparison can tell fp16 scoring apart from fp32.
    assert abs(half_precision_scoring - fp32).max() > 1e-6
