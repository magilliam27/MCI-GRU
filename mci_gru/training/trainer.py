"""
Training logic for MCI-GRU experiments.

This module provides the Trainer class that handles:
- Training loop with validation
- Early stopping
- Model checkpointing
- Inference

Graph resolution is handled upstream by the collate function (via
``GraphSchedule``), so the Trainer consumes the 9-tuple batches
(7 core tensors + optional sector ``edge_index`` / ``edge_weight``) from the loaders.
"""

import os
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mci_gru.config import ExperimentConfig
from mci_gru.training.losses import (
    CombinedMSEICLoss,
    ICLoss,
    mean_information_coefficient,
)
from mci_gru.utils.seeding import set_seed

if TYPE_CHECKING:
    from mci_gru.tracking import MLflowTrackingManager


def _unpack_loader_batch(batch, device: torch.device):
    """Move graph batch tensors to *device*; supports 7- or 9-tuple collate output."""
    if len(batch) == 7:
        time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates = batch
        edge_index_sector = None
        edge_weight_sector = None
    else:
        (
            time_series,
            labels,
            graph_features,
            edge_index,
            edge_weight,
            n_stocks,
            batch_dates,
            edge_index_sector,
            edge_weight_sector,
        ) = batch

    time_series = time_series.to(device)
    labels = labels.to(device)
    graph_features = graph_features.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    if edge_index_sector is not None:
        edge_index_sector = edge_index_sector.to(device)
    if edge_weight_sector is not None:
        edge_weight_sector = edge_weight_sector.to(device)

    return (
        time_series,
        labels,
        graph_features,
        edge_index,
        edge_weight,
        n_stocks,
        batch_dates,
        edge_index_sector,
        edge_weight_sector,
    )


@dataclass
class TrainingResult:
    best_val_loss: float
    best_val_ic: float
    final_train_loss: float
    epochs_trained: int
    best_model_path: str
    predictions: np.ndarray | None = None


def _build_lr_scheduler(
    optimizer: optim.Optimizer,
    training_cfg,
    steps_per_epoch: int,
):
    """Per-step warmup + cosine, or None when ``lr_scheduler`` is ``none``."""
    if training_cfg.lr_scheduler == "none":
        return None

    total_steps = max(1, training_cfg.num_epochs * max(1, steps_per_epoch))
    warmup_steps = min(training_cfg.warmup_steps, total_steps)
    eta_min = training_cfg.learning_rate * 0.01
    cosine_steps = max(1, total_steps - warmup_steps)

    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)


class Trainer:
    """
    Trainer for MCI-GRU models.

    Supports:
    - Standard training with validation-based early stopping
    - Multi-model training (for averaging predictions)

    Dynamic graph snapshots are resolved in the collate function; the
    Trainer receives correctly-assembled edge tensors in every batch.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device | None = None,
        output_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """
        Args:
            model: PyTorch model to train
            config: Experiment configuration
            device: Device to train on (auto-detected if None)
            output_path: Output directory override (e.g., Hydra timestamped run dir)
            checkpoint_path: Full path for best-model checkpoint file
        """
        self.model = model
        self.config = config
        self.output_path = output_path if output_path else self.config.get_output_path()
        self.checkpoint_path = checkpoint_path
        self.last_best_model_path: str | None = None

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Training state
        self.best_val_loss = float("inf")
        self.best_val_ic = float("-inf")
        self.patience_counter = 0
        self.epoch = 0

    def train(
        self,
        train_loader,
        val_loader,
        epoch_callback: Callable[..., None] | None = None,
    ) -> TrainingResult:
        """
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch_callback: Optional per-epoch callback; receives
                (epoch, train_loss, val_loss, val_ic, best_val_loss, best_val_ic).

        Returns:
            TrainingResult with training metrics
        """
        training_cfg = self.config.training

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
        if training_cfg.loss_type == "ic":
            criterion = ICLoss()
        elif training_cfg.loss_type == "combined":
            criterion = CombinedMSEICLoss(alpha=training_cfg.ic_loss_alpha)
        else:
            criterion = nn.MSELoss()

        steps_per_epoch = len(train_loader)
        scheduler = _build_lr_scheduler(optimizer, training_cfg, steps_per_epoch)

        use_amp = training_cfg.use_amp and self.device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)

        output_path = self.output_path
        os.makedirs(output_path, exist_ok=True)
        best_model_path = (
            self.checkpoint_path
            if self.checkpoint_path
            else os.path.join(output_path, "best_model.pth")
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_val_ic = float("-inf")
        self.patience_counter = 0
        final_train_loss = 0.0

        print(f"Training on {self.device}...")
        print(
            f"  Loss: {training_cfg.loss_type}"
            + (
                f" (alpha={training_cfg.ic_loss_alpha})"
                if training_cfg.loss_type == "combined"
                else ""
            )
        )
        print(f"  Selection metric: {training_cfg.selection_metric}")
        print(f"  LR scheduler: {training_cfg.lr_scheduler}")
        print(f"  AMP (CUDA): {use_amp}")
        print(f"  Max epochs: {training_cfg.num_epochs}")
        print(f"  Early stopping patience: {training_cfg.early_stopping_patience}")

        for epoch in range(training_cfg.num_epochs):
            self.epoch = epoch

            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, scaler, scheduler, use_amp
            )
            final_train_loss = train_loss

            val_loss, val_ic = self._validate(val_loader, criterion, use_amp)

            print(
                f"Epoch [{epoch + 1}/{training_cfg.num_epochs}] - Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, Val IC: {val_ic:.6f}"
            )

            improved = False
            if training_cfg.selection_metric == "val_ic":
                if val_ic > self.best_val_ic:
                    improved = True
            else:
                if val_loss < self.best_val_loss:
                    improved = True

            if improved:
                self.best_val_loss = val_loss
                self.best_val_ic = val_ic
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(
                    f"  -> New best model saved (val_loss={self.best_val_loss:.6f}, val_ic={self.best_val_ic:.6f})"
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= training_cfg.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} (patience={training_cfg.early_stopping_patience})"
                    )
                    if epoch_callback is not None:
                        epoch_callback(
                            epoch + 1,
                            train_loss,
                            val_loss,
                            val_ic,
                            self.best_val_loss,
                            self.best_val_ic,
                        )
                    break

            if epoch_callback is not None:
                epoch_callback(
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_ic,
                    self.best_val_loss,
                    self.best_val_ic,
                )

        return TrainingResult(
            best_val_loss=self.best_val_loss,
            best_val_ic=self.best_val_ic,
            final_train_loss=final_train_loss,
            epochs_trained=epoch + 1,
            best_model_path=best_model_path,
        )

    def _train_epoch(self, train_loader, optimizer, criterion, scaler, scheduler, use_amp) -> float:
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for batch in train_loader:
            (
                time_series,
                labels,
                graph_features,
                edge_index,
                edge_weight,
                n_stocks,
                _batch_dates,
                edge_index_sector,
                edge_weight_sector,
            ) = _unpack_loader_batch(batch, self.device)
            batch_size = time_series.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = self.model(
                    time_series,
                    graph_features,
                    edge_index,
                    edge_weight,
                    n_stocks,
                    edge_index_sector=edge_index_sector,
                    edge_weight_sector=edge_weight_sector,
                )
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if self.config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * batch_size
            num_samples += batch_size

        return total_loss / num_samples if num_samples > 0 else 0.0

    def _validate(self, val_loader, criterion, use_amp) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_ic = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                (
                    time_series,
                    labels,
                    graph_features,
                    edge_index,
                    edge_weight,
                    n_stocks,
                    _batch_dates,
                    edge_index_sector,
                    edge_weight_sector,
                ) = _unpack_loader_batch(batch, self.device)
                batch_size = time_series.shape[0]

                with autocast("cuda", enabled=use_amp):
                    outputs = self.model(
                        time_series,
                        graph_features,
                        edge_index,
                        edge_weight,
                        n_stocks,
                        edge_index_sector=edge_index_sector,
                        edge_weight_sector=edge_weight_sector,
                    )
                    loss = criterion(outputs, labels)
                ic = mean_information_coefficient(outputs, labels)

                total_loss += loss.item() * batch_size
                total_ic += ic.item() * batch_size
                num_samples += batch_size

        mean_loss = total_loss / num_samples if num_samples > 0 else 0.0
        mean_ic = total_ic / num_samples if num_samples > 0 else 0.0
        return mean_loss, mean_ic

    def predict(self, test_loader, kdcode_list: list[str], test_dates: list[str]) -> np.ndarray:
        """
        Args:
            test_loader: Test data loader
            kdcode_list: List of stock codes
            test_dates: List of test dates

        Returns:
            Predictions array of shape (n_dates, n_stocks)
        """
        use_amp = self.config.training.use_amp and self.device.type == "cuda"
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                (
                    time_series,
                    _,
                    graph_features,
                    edge_index,
                    edge_weight,
                    n_stocks,
                    _batch_dates,
                    edge_index_sector,
                    edge_weight_sector,
                ) = _unpack_loader_batch(batch, self.device)

                with autocast("cuda", enabled=use_amp):
                    outputs = self.model(
                        time_series,
                        graph_features,
                        edge_index,
                        edge_weight,
                        n_stocks,
                        edge_index_sector=edge_index_sector,
                        edge_weight_sector=edge_weight_sector,
                    )
                predictions = outputs.squeeze().cpu().numpy()
                all_predictions.append(predictions)

        return np.array(all_predictions)

    def save_predictions(
        self,
        predictions: np.ndarray,
        kdcode_list: list[str],
        test_dates: list[str],
        output_dir: str,
    ):
        """
        Args:
            predictions: Predictions array (n_dates, n_stocks)
            kdcode_list: Stock codes
            test_dates: Test dates
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx, date in enumerate(test_dates):
            if idx < len(predictions):
                data = [
                    [kdcode_list[i], date, round(float(predictions[idx][i]), 5)]
                    for i in range(len(kdcode_list))
                ]
                df = pd.DataFrame(columns=["kdcode", "dt", "score"], data=data)
                df.to_csv(os.path.join(output_dir, f"{date}.csv"), index=False)

    def load_best_model(self, best_model_path: str | None = None):
        if best_model_path is None:
            if self.last_best_model_path is not None:
                best_model_path = self.last_best_model_path
            elif self.checkpoint_path is not None:
                best_model_path = self.checkpoint_path
            else:
                best_model_path = os.path.join(self.output_path, "best_model.pth")

        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
            print(f"Loaded best model from {best_model_path}")
        else:
            print(f"No saved model found at {best_model_path}")


def train_multiple_models(
    model_factory,
    config: ExperimentConfig,
    train_loader,
    val_loader,
    test_loader,
    kdcode_list: list[str],
    test_dates: list[str],
    output_path: str | None = None,
    tracking_manager: Optional["MLflowTrackingManager"] = None,
) -> tuple[list[TrainingResult], np.ndarray]:
    """
    Per paper Section 4.1.2: Train num_models and average predictions.

    Graph snapshots are already baked into the data loaders via
    ``GraphSchedule``; each model simply consumes batches whose edge
    tensors reflect the correct temporal snapshot.

    Args:
        model_factory: Callable that creates a new model instance
        config: Experiment configuration
        train_loader: Training data loader (with precomputed graphs)
        val_loader: Validation data loader
        test_loader: Test data loader
        kdcode_list: Stock codes
        test_dates: List of test dates
        output_path: Optional output path override (for Hydra managed paths)
        tracking_manager: Optional MLflow tracking manager

    Returns:
        Tuple of (list of training results, averaged predictions)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_output_path = output_path if output_path else config.get_output_path()
    checkpoint_dir = os.path.join(base_output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_results = []
    all_predictions = []

    for model_id in range(config.training.num_models):
        print(f"\n{'=' * 60}")
        print(f"Training Model {model_id + 1}/{config.training.num_models}")
        print(f"{'=' * 60}")

        set_seed(config.seed + model_id)

        model = model_factory()
        model_checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_id}_best.pth")
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            output_path=base_output_path,
            checkpoint_path=model_checkpoint_path,
        )

        child_ctx = nullcontext(None)
        if tracking_manager is not None and tracking_manager.enabled:
            child_ctx = tracking_manager.create_child_run(
                run_name=f"model_{model_id}",
                tags={"run_kind": "training_child", "model_id": model_id},
            )

        with child_ctx as child_tracking:
            epoch_callback = None
            if child_tracking is not None and child_tracking.enabled:
                epoch_callback = child_tracking.log_epoch_metrics

            result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epoch_callback=epoch_callback,
            )
            all_results.append(result)

            print(
                f"Model {model_id + 1} training complete. Best val loss: {result.best_val_loss:.6f}, "
                f"best val IC: {result.best_val_ic:.6f}"
            )

            trainer.last_best_model_path = result.best_model_path
            trainer.load_best_model(result.best_model_path)
            predictions = trainer.predict(test_loader, kdcode_list, test_dates)
            all_predictions.append(predictions)

            pred_dir = os.path.join(base_output_path, f"predictions_model_{model_id}")
            trainer.save_predictions(predictions, kdcode_list, test_dates, pred_dir)

            if child_tracking is not None and child_tracking.enabled:
                child_tracking.log_metrics(
                    {
                        "best_val_loss": result.best_val_loss,
                        "best_val_ic": result.best_val_ic,
                        "final_train_loss": result.final_train_loss,
                        "epochs_trained": result.epochs_trained,
                    }
                )
                if config.tracking.log_artifacts and config.tracking.log_checkpoints:
                    child_tracking.log_artifact(
                        result.best_model_path,
                        artifact_path=f"checkpoints/model_{model_id}",
                    )
                if config.tracking.log_artifacts and config.tracking.log_predictions:
                    child_tracking.log_artifacts(
                        pred_dir,
                        artifact_path=f"predictions/model_{model_id}",
                    )

    avg_predictions = np.mean(all_predictions, axis=0)
    avg_pred_dir = os.path.join(base_output_path, "averaged_predictions")
    os.makedirs(avg_pred_dir, exist_ok=True)

    for idx, date in enumerate(test_dates):
        if idx < len(avg_predictions):
            data = [
                [kdcode_list[i], date, round(float(avg_predictions[idx][i]), 5)]
                for i in range(len(kdcode_list))
            ]
            df_pred = pd.DataFrame(columns=["kdcode", "dt", "score"], data=data)
            df_pred.to_csv(os.path.join(avg_pred_dir, f"{date}.csv"), index=False)

    print(f"\nAveraged predictions saved to {avg_pred_dir}")

    return all_results, avg_predictions
