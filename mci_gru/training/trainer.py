"""
Training logic for MCI-GRU experiments.

This module provides the Trainer class that handles:
- Training loop with validation
- Early stopping
- Dynamic graph updates
- Model checkpointing
- Inference
"""

import os
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.graph.builder import GraphBuilder
from mci_gru.training.losses import ICLoss, CombinedMSEICLoss

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mci_gru.tracking import MLflowTrackingManager


@dataclass
class TrainingResult:
    best_val_loss: float
    final_train_loss: float
    epochs_trained: int
    best_model_path: str
    predictions: Optional[np.ndarray] = None


class Trainer:
    """
    Trainer for MCI-GRU models.
    
    Supports:
    - Standard training with validation-based early stopping
    - Dynamic graph updates during training
    - Multi-model training (for averaging predictions)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        graph_builder: Optional[GraphBuilder] = None,
        device: Optional[torch.device] = None,
        output_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            model: PyTorch model to train
            config: Experiment configuration
            graph_builder: Optional graph builder for dynamic updates
            device: Device to train on (auto-detected if None)
            output_path: Output directory override (e.g., Hydra timestamped run dir)
            checkpoint_path: Full path for best-model checkpoint file
        """
        self.model = model
        self.config = config
        self.graph_builder = graph_builder
        self.output_path = output_path if output_path else self.config.get_output_path()
        self.checkpoint_path = checkpoint_path
        self.last_best_model_path: Optional[str] = None
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Runtime state for dynamic graph (set at start of train / predict)
        self._df: Optional[pd.DataFrame] = None
        self._kdcode_list: Optional[List[str]] = None
        self._dynamic_update_count: int = 0

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
    
    def train(
        self,
        train_loader,
        val_loader,
        train_dates: Optional[List[str]] = None,
        df: Optional[pd.DataFrame] = None,
        kdcode_list: Optional[List[str]] = None,
        epoch_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ) -> TrainingResult:
        """
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            train_dates: List of training dates (for dynamic graph updates)
            df: DataFrame (for dynamic graph updates)
            kdcode_list: Stock list (for dynamic graph updates)
            
        Returns:
            TrainingResult with training metrics
        """
        training_cfg = self.config.training

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay
        )
        if training_cfg.loss_type == "ic":
            criterion = ICLoss()
        elif training_cfg.loss_type == "combined":
            criterion = CombinedMSEICLoss(alpha=training_cfg.ic_loss_alpha)
        else:
            criterion = nn.MSELoss()

        output_path = self.output_path
        os.makedirs(output_path, exist_ok=True)
        best_model_path = (
            self.checkpoint_path
            if self.checkpoint_path
            else os.path.join(output_path, 'best_model.pth')
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self._dynamic_update_count = 0
        final_train_loss = 0.0

        # Store for batch-level dynamic graph access
        self._df = df
        self._kdcode_list = kdcode_list

        dynamic = (
            self.graph_builder is not None
            and self.config.graph.update_frequency_months > 0
        )
        if self.graph_builder is not None:
            stats = self.graph_builder.get_stats()
            print(f"  Initial graph: {stats.get('n_edges', 0)} edges, "
                  f"last_update={stats.get('last_update_date')}, "
                  f"update_frequency_months={stats.get('update_frequency_months')}")

        print(f"Training on {self.device}...")
        print(f"  Loss: {training_cfg.loss_type}" + (
            f" (alpha={training_cfg.ic_loss_alpha})" if training_cfg.loss_type == "combined" else ""
        ))
        print(f"  Max epochs: {training_cfg.num_epochs}")
        print(f"  Early stopping patience: {training_cfg.early_stopping_patience}")
        if dynamic:
            print(f"  Dynamic graph: ON (update every {self.config.graph.update_frequency_months} months per batch date)")

        for epoch in range(training_cfg.num_epochs):
            self.epoch = epoch

            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            final_train_loss = train_loss

            val_loss = self._validate(val_loader, criterion)
            
            print(f"Epoch [{epoch+1}/{training_cfg.num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  -> New best model saved (val_loss={self.best_val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= training_cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (patience={training_cfg.early_stopping_patience})")
                    if epoch_callback is not None:
                        epoch_callback(epoch + 1, train_loss, val_loss, self.best_val_loss)
                    break

            if epoch_callback is not None:
                epoch_callback(epoch + 1, train_loss, val_loss, self.best_val_loss)
        
        if dynamic:
            print(f"  Dynamic graph updates applied during training: {self._dynamic_update_count}")

        return TrainingResult(
            best_val_loss=self.best_val_loss,
            final_train_loss=final_train_loss,
            epochs_trained=epoch + 1,
            best_model_path=best_model_path,
        )
    
    def _batched_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_size: int,
        num_stocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand single-graph edge tensors to cover a full batch via index shifting."""
        ei_list = [edge_index + i * num_stocks for i in range(batch_size)]
        ew_list = [edge_weight] * batch_size
        return torch.cat(ei_list, dim=1), torch.cat(ew_list, dim=0)

    def _apply_dynamic_graph(
        self,
        batch_dates: Optional[List[str]],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_stocks: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """If dynamic graph is active and dates are provided, update and return current edges."""
        dynamic = (
            self.graph_builder is not None
            and self.config.graph.update_frequency_months > 0
            and batch_dates is not None
            and self._df is not None
        )
        if not dynamic:
            return edge_index, edge_weight

        date = batch_dates[0]
        prev_stats = self.graph_builder.get_stats()
        updated_ei, updated_ew = self.graph_builder.update_if_needed(
            self._df, self._kdcode_list, date, show_progress=False
        )
        if updated_ei is not None:
            self._dynamic_update_count += 1
            new_stats = self.graph_builder.get_stats()
            print(
                f"  [graph] updated at {date}: "
                f"{prev_stats.get('n_edges', '?')} -> {new_stats.get('n_edges', '?')} edges"
            )

        ei, ew = self.graph_builder.get_current_graph()
        return self._batched_edges(ei, ew, batch_size, n_stocks)

    def _train_epoch(self, train_loader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates in train_loader:
            batch_size = time_series.shape[0]

            time_series = time_series.to(self.device)
            labels = labels.to(self.device)
            graph_features = graph_features.to(self.device)

            edge_index, edge_weight = self._apply_dynamic_graph(
                batch_dates, edge_index, edge_weight, n_stocks, batch_size
            )
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(time_series, graph_features, edge_index, edge_weight, n_stocks)
            loss = criterion(outputs, labels)
            loss.backward()

            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

            optimizer.step()

            total_loss += loss.item() * batch_size
            num_samples += batch_size

        return total_loss / num_samples if num_samples > 0 else 0.0

    def _validate(self, val_loader, criterion) -> float:
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates in val_loader:
                batch_size = time_series.shape[0]

                time_series = time_series.to(self.device)
                labels = labels.to(self.device)
                graph_features = graph_features.to(self.device)

                edge_index, edge_weight = self._apply_dynamic_graph(
                    batch_dates, edge_index, edge_weight, n_stocks, batch_size
                )
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)

                outputs = self.model(time_series, graph_features, edge_index, edge_weight, n_stocks)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * batch_size
                num_samples += batch_size

        return total_loss / num_samples if num_samples > 0 else 0.0
    
    def predict(
        self,
        test_loader,
        kdcode_list: List[str],
        test_dates: List[str]
    ) -> np.ndarray:
        """
        Args:
            test_loader: Test data loader
            kdcode_list: List of stock codes
            test_dates: List of test dates
            
        Returns:
            Predictions array of shape (n_dates, n_stocks)
        """
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for time_series, _, graph_features, edge_index, edge_weight, n_stocks, batch_dates in test_loader:
                batch_size = time_series.shape[0]

                time_series = time_series.to(self.device)
                graph_features = graph_features.to(self.device)

                edge_index, edge_weight = self._apply_dynamic_graph(
                    batch_dates, edge_index, edge_weight, n_stocks, batch_size
                )
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)

                outputs = self.model(time_series, graph_features, edge_index, edge_weight, n_stocks)
                predictions = outputs.squeeze().cpu().numpy()
                all_predictions.append(predictions)

        return np.array(all_predictions)
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        kdcode_list: List[str],
        test_dates: List[str],
        output_dir: str
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
                data = [[kdcode_list[i], date, round(float(predictions[idx][i]), 5)]
                        for i in range(len(kdcode_list))]
                df = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data)
                df.to_csv(os.path.join(output_dir, f'{date}.csv'), index=False)
    
    def load_best_model(self, best_model_path: Optional[str] = None):
        if best_model_path is None:
            if self.last_best_model_path is not None:
                best_model_path = self.last_best_model_path
            elif self.checkpoint_path is not None:
                best_model_path = self.checkpoint_path
            else:
                best_model_path = os.path.join(self.output_path, 'best_model.pth')

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
    kdcode_list: List[str],
    test_dates: List[str],
    graph_builder: Optional[GraphBuilder] = None,
    df: Optional[pd.DataFrame] = None,
    train_dates: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    tracking_manager: Optional["MLflowTrackingManager"] = None,
) -> Tuple[List[TrainingResult], np.ndarray]:
    """
    Per paper Section 4.1.2: Train num_models and average predictions.

    Args:
        model_factory: Callable that creates a new model instance
        config: Experiment configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        kdcode_list: Stock codes
        test_dates: Test dates
        graph_builder: Optional graph builder
        df: DataFrame for dynamic graph updates
        train_dates: Training dates for dynamic updates
        output_path: Optional output path override (for Hydra managed paths)
        
    Returns:
        Tuple of (list of training results, averaged predictions)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    base_output_path = output_path if output_path else config.get_output_path()
    checkpoint_dir = os.path.join(base_output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_results = []
    all_predictions = []
    
    for model_id in range(config.training.num_models):
        print(f"\n{'='*60}")
        print(f"Training Model {model_id + 1}/{config.training.num_models}")
        print(f"{'='*60}")

        model = model_factory()
        model_config = config
        model_checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_id}_best.pth")
        trainer = Trainer(
            model=model,
            config=model_config,
            graph_builder=graph_builder,
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
                train_dates=train_dates,
                df=df,
                kdcode_list=kdcode_list,
                epoch_callback=epoch_callback,
            )
            all_results.append(result)

            print(f"Model {model_id + 1} training complete. Best val loss: {result.best_val_loss:.6f}")

            trainer.last_best_model_path = result.best_model_path
            trainer.load_best_model(result.best_model_path)
            predictions = trainer.predict(test_loader, kdcode_list, test_dates)
            all_predictions.append(predictions)

            pred_dir = os.path.join(base_output_path, f'predictions_model_{model_id}')
            trainer.save_predictions(predictions, kdcode_list, test_dates, pred_dir)

            if child_tracking is not None and child_tracking.enabled:
                child_tracking.log_metrics({
                    "best_val_loss": result.best_val_loss,
                    "final_train_loss": result.final_train_loss,
                    "epochs_trained": result.epochs_trained,
                })
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
    avg_pred_dir = os.path.join(base_output_path, 'averaged_predictions')
    os.makedirs(avg_pred_dir, exist_ok=True)
    
    for idx, date in enumerate(test_dates):
        if idx < len(avg_predictions):
            data = [[kdcode_list[i], date, round(float(avg_predictions[idx][i]), 5)]
                    for i in range(len(kdcode_list))]
            df_pred = pd.DataFrame(columns=['kdcode', 'dt', 'score'], data=data)
            df_pred.to_csv(os.path.join(avg_pred_dir, f'{date}.csv'), index=False)
    
    print(f"\nAveraged predictions saved to {avg_pred_dir}")
    
    return all_results, avg_predictions
