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

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mci_gru.config import ExperimentConfig
from mci_gru.training.losses import (
    build_training_loss,
    information_coefficient_sum_count,
    rank_information_coefficient_sum_count,
)

logger = logging.getLogger(__name__)


def _unpack_loader_batch(batch, device: torch.device):
    """Move graph batch tensors to *device*; supports 7- or 9-tuple collate output."""
    non_blocking = device.type == "cuda"
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

    stock_mask = None
    if isinstance(batch_dates, dict):
        stock_mask = batch_dates.get("stock_mask")
        batch_dates = batch_dates.get("dates")

    time_series = time_series.to(device, non_blocking=non_blocking)
    labels = labels.to(device, non_blocking=non_blocking)
    graph_features = graph_features.to(device, non_blocking=non_blocking)
    edge_index = edge_index.to(device, non_blocking=non_blocking)
    edge_weight = edge_weight.to(device, non_blocking=non_blocking)
    if stock_mask is not None:
        stock_mask = stock_mask.to(device, non_blocking=non_blocking)
    if edge_index_sector is not None:
        edge_index_sector = edge_index_sector.to(device, non_blocking=non_blocking)
    if edge_weight_sector is not None:
        edge_weight_sector = edge_weight_sector.to(device, non_blocking=non_blocking)

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
        stock_mask,
    )


@dataclass
class TrainingResult:
    best_val_loss: float
    best_val_ic: float
    best_val_rank_ic: float
    final_train_loss: float
    epochs_trained: int
    best_model_path: str
    predictions: np.ndarray | None = None


def prediction_rows_for_date(
    predictions: np.ndarray,
    kdcode_list: list[str],
    date: str,
    prediction_mask: np.ndarray | None = None,
) -> list[list[object]]:
    """Build prediction CSV rows, optionally filtering to PIT-tradable stocks."""
    mask = (
        np.asarray(prediction_mask, dtype=bool)
        if prediction_mask is not None
        else np.ones(len(kdcode_list), dtype=bool)
    )
    rows: list[list[object]] = []
    for i, kdcode in enumerate(kdcode_list):
        if i >= len(predictions) or i >= len(mask) or not bool(mask[i]):
            continue
        score = float(predictions[i])
        if not np.isfinite(score):
            continue
        rows.append([kdcode, date, round(score, 5)])
    return rows


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
        self.best_val_rank_ic = float("-inf")
        self.patience_counter = 0
        self.epoch = 0
        self._profile_rows_written = 0
        self._profile_path: str | None = None

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
                (epoch, train_loss, val_loss, val_ic, val_rank_ic,
                 best_val_loss, best_val_ic, best_val_rank_ic).

        Returns:
            TrainingResult with training metrics
        """
        training_cfg = self.config.training

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
        criterion, loss_label = build_training_loss(training_cfg)
        criterion = criterion.to(self.device)

        steps_per_epoch = len(train_loader)
        scheduler = _build_lr_scheduler(optimizer, training_cfg, steps_per_epoch)

        use_amp = training_cfg.use_amp and self.device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)

        output_path = self.output_path
        os.makedirs(output_path, exist_ok=True)
        self._profile_rows_written = 0
        self._profile_path = None
        if training_cfg.profile_batches > 0:
            profile_name = "training_step_profile.jsonl"
            if self.checkpoint_path is not None:
                checkpoint_stem = os.path.basename(self.checkpoint_path).removesuffix("_best.pth")
                profile_name = f"training_step_profile_{checkpoint_stem}.jsonl"
            self._profile_path = os.path.join(output_path, profile_name)
            with open(self._profile_path, "w", encoding="utf-8"):
                pass
        best_model_path = (
            self.checkpoint_path
            if self.checkpoint_path
            else os.path.join(output_path, "best_model.pth")
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_val_ic = float("-inf")
        self.best_val_rank_ic = float("-inf")
        self.patience_counter = 0
        final_train_loss = 0.0

        logger.info(f"Training on {self.device}...")
        logger.info(f"  Loss: {loss_label}")
        logger.info(f"  Selection metric: {training_cfg.selection_metric}")
        logger.info(f"  LR scheduler: {training_cfg.lr_scheduler}")
        logger.info(f"  AMP (CUDA): {use_amp}")
        logger.info(f"  Max epochs: {training_cfg.num_epochs}")
        logger.info(f"  Early stopping patience: {training_cfg.early_stopping_patience}")

        for epoch in range(training_cfg.num_epochs):
            self.epoch = epoch

            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, scaler, scheduler, use_amp
            )
            final_train_loss = train_loss

            val_loss, val_ic, val_rank_ic = self._validate(val_loader, criterion, use_amp)

            logger.info(
                f"Epoch [{epoch + 1}/{training_cfg.num_epochs}] - Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, Val IC: {val_ic:.6f}, "
                f"Val Rank IC: {val_rank_ic:.6f}"
            )

            improved = False
            if training_cfg.selection_metric == "val_ic":
                if val_ic > self.best_val_ic:
                    improved = True
            elif training_cfg.selection_metric == "val_rank_ic":
                if val_rank_ic > self.best_val_rank_ic:
                    improved = True
            else:
                if val_loss < self.best_val_loss:
                    improved = True

            if improved:
                self.best_val_loss = val_loss
                self.best_val_ic = val_ic
                self.best_val_rank_ic = val_rank_ic
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(
                    "  -> New best model saved "
                    f"(val_loss={self.best_val_loss:.6f}, "
                    f"val_ic={self.best_val_ic:.6f}, "
                    f"val_rank_ic={self.best_val_rank_ic:.6f})"
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= training_cfg.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} (patience={training_cfg.early_stopping_patience})"
                    )
                    if epoch_callback is not None:
                        epoch_callback(
                            epoch + 1,
                            train_loss,
                            val_loss,
                            val_ic,
                            val_rank_ic,
                            self.best_val_loss,
                            self.best_val_ic,
                            self.best_val_rank_ic,
                        )
                    break

            if epoch_callback is not None:
                epoch_callback(
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_ic,
                    val_rank_ic,
                    self.best_val_loss,
                    self.best_val_ic,
                    self.best_val_rank_ic,
                )

        return TrainingResult(
            best_val_loss=self.best_val_loss,
            best_val_ic=self.best_val_ic,
            best_val_rank_ic=self.best_val_rank_ic,
            final_train_loss=final_train_loss,
            epochs_trained=epoch + 1,
            best_model_path=best_model_path,
        )

    def _sync_cuda_if_needed(self, profiling_active: bool) -> None:
        if profiling_active and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _write_profile_row(self, row: dict[str, object]) -> None:
        if self._profile_path is None:
            return
        with open(self._profile_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._profile_rows_written += 1

    def _train_epoch(self, train_loader, optimizer, criterion, scaler, scheduler, use_amp) -> float:
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        profile_limit = self.config.training.profile_batches
        profiling_active = profile_limit > 0
        batch_wait_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            batch_received = time.perf_counter()
            data_wait_seconds = batch_received - batch_wait_start

            h2d_start = time.perf_counter()
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
                stock_mask,
            ) = _unpack_loader_batch(batch, self.device)
            self._sync_cuda_if_needed(profiling_active)
            h2d_seconds = time.perf_counter() - h2d_start
            batch_size = time_series.shape[0]

            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            with autocast("cuda", enabled=use_amp):
                outputs = self.model(
                    time_series,
                    graph_features,
                    edge_index,
                    edge_weight,
                    n_stocks,
                    edge_index_sector=edge_index_sector,
                    edge_weight_sector=edge_weight_sector,
                    stock_mask=stock_mask,
                )
                loss = criterion(outputs, labels)
            self._sync_cuda_if_needed(profiling_active)
            forward_loss_seconds = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            self._sync_cuda_if_needed(profiling_active)
            backward_seconds = time.perf_counter() - backward_start

            optimizer_start = time.perf_counter()
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
            self._sync_cuda_if_needed(profiling_active)
            optimizer_seconds = time.perf_counter() - optimizer_start

            loss_value = loss.item()
            if profile_limit > 0 and self._profile_rows_written < profile_limit:
                cuda_memory_allocated = 0
                cuda_max_memory_allocated = 0
                if self.device.type == "cuda":
                    cuda_memory_allocated = int(torch.cuda.memory_allocated(self.device))
                    cuda_max_memory_allocated = int(torch.cuda.max_memory_allocated(self.device))
                self._write_profile_row(
                    {
                        "epoch": self.epoch,
                        "batch_index": batch_index,
                        "device": str(self.device),
                        "batch_size": int(batch_size),
                        "loss": float(loss_value),
                        "data_wait_seconds": data_wait_seconds,
                        "h2d_seconds": h2d_seconds,
                        "forward_loss_seconds": forward_loss_seconds,
                        "backward_seconds": backward_seconds,
                        "optimizer_seconds": optimizer_seconds,
                        "cuda_memory_allocated_bytes": cuda_memory_allocated,
                        "cuda_max_memory_allocated_bytes": cuda_max_memory_allocated,
                    }
                )

            total_loss += loss_value * batch_size
            num_samples += batch_size
            batch_wait_start = time.perf_counter()

        return total_loss / num_samples if num_samples > 0 else 0.0

    def _validate(self, val_loader, criterion, use_amp) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_ic = 0.0
        total_ic_rows = 0
        total_rank_ic = 0.0
        total_rank_ic_rows = 0
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
                    stock_mask,
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
                        stock_mask=stock_mask,
                    )
                    loss = criterion(outputs, labels)
                ic_sum, ic_count = information_coefficient_sum_count(outputs, labels)
                rank_ic_sum, rank_ic_count = rank_information_coefficient_sum_count(
                    outputs,
                    labels,
                )

                total_loss += loss.item() * batch_size
                total_ic += ic_sum.item()
                total_ic_rows += ic_count
                total_rank_ic += rank_ic_sum.item()
                total_rank_ic_rows += rank_ic_count
                num_samples += batch_size

        mean_loss = total_loss / num_samples if num_samples > 0 else 0.0
        mean_ic = total_ic / total_ic_rows if total_ic_rows > 0 else 0.0
        mean_rank_ic = total_rank_ic / total_rank_ic_rows if total_rank_ic_rows > 0 else 0.0
        return mean_loss, mean_ic, mean_rank_ic

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
                    stock_mask,
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
                        stock_mask=stock_mask,
                    )
                if stock_mask is not None:
                    outputs = outputs.masked_fill(~stock_mask, float("nan"))
                predictions = outputs.squeeze(0).cpu().numpy()
                all_predictions.append(predictions)

        return np.array(all_predictions)

    def save_predictions(
        self,
        predictions: np.ndarray,
        kdcode_list: list[str],
        test_dates: list[str],
        output_dir: str,
        prediction_masks: np.ndarray | None = None,
    ):
        """
        Args:
            predictions: Predictions array (n_dates, n_stocks)
            kdcode_list: Stock codes
            test_dates: Test dates
            output_dir: Output directory
            prediction_masks: Optional boolean ``(n_dates, n_stocks)`` tradable mask
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx, date in enumerate(test_dates):
            if idx < len(predictions):
                mask = prediction_masks[idx] if prediction_masks is not None else None
                data = prediction_rows_for_date(predictions[idx], kdcode_list, date, mask)
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
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.info(f"No saved model found at {best_model_path}")
