"""
Ensemble training for MCI-GRU experiments.

Trains multiple independently seeded models and averages their predictions.
"""

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch

from mci_gru.config import ExperimentConfig
from mci_gru.training.trainer import Trainer, TrainingResult, prediction_rows_for_date
from mci_gru.utils.seeding import set_seed

if TYPE_CHECKING:
    from mci_gru.tracking import MLflowTrackingManager


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
    test_prediction_masks: np.ndarray | None = None,
) -> tuple[list[TrainingResult], np.ndarray]:
    """
    Per paper Section 4.1.2: Train num_models and average predictions.
    Each ensemble member is independently initialized with ``config.seed + model_id``.

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
        test_prediction_masks: Optional boolean tradable mask for test prediction exports

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

        model_seed = config.seed + model_id
        print(f"Model seed: {model_seed}")
        set_seed(model_seed)

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
                f"best val IC: {result.best_val_ic:.6f}, "
                f"best val Rank IC: {result.best_val_rank_ic:.6f}"
            )

            trainer.last_best_model_path = result.best_model_path
            trainer.load_best_model(result.best_model_path)
            predictions = trainer.predict(test_loader, kdcode_list, test_dates)
            all_predictions.append(predictions)

            pred_dir = os.path.join(base_output_path, f"predictions_model_{model_id}")
            trainer.save_predictions(
                predictions,
                kdcode_list,
                test_dates,
                pred_dir,
                prediction_masks=test_prediction_masks,
            )

            if child_tracking is not None and child_tracking.enabled:
                child_tracking.log_metrics(
                    {
                        "best_val_loss": result.best_val_loss,
                        "best_val_ic": result.best_val_ic,
                        "best_val_rank_ic": result.best_val_rank_ic,
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
            mask = test_prediction_masks[idx] if test_prediction_masks is not None else None
            data = prediction_rows_for_date(avg_predictions[idx], kdcode_list, date, mask)
            df_pred = pd.DataFrame(columns=["kdcode", "dt", "score"], data=data)
            df_pred.to_csv(os.path.join(avg_pred_dir, f"{date}.csv"), index=False)

    print(f"\nAveraged predictions saved to {avg_pred_dir}")

    return all_results, avg_predictions
