"""
Ensemble training for MCI-GRU experiments.

Trains multiple independently seeded models and averages their predictions.
"""

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import torch

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation.execution_provenance import (
    BACKEND_OBSERVATIONS,
    ExecutionReference,
    MemberEventLog,
)
from mci_gru.training.summary import format_checkpoint_metric
from mci_gru.training.trainer import Trainer, TrainingResult, prediction_rows_for_date
from mci_gru.utils.hashing import sha256_file
from mci_gru.utils.seeding import set_seed

if TYPE_CHECKING:
    from mci_gru.tracking import MLflowTrackingManager

logger = logging.getLogger(__name__)

_BACKEND_PROBES = {
    "cuda_available": lambda: torch.cuda.is_available(),
    "cudnn_version": lambda: torch.backends.cudnn.version(),
    "cudnn_enabled": lambda: torch.backends.cudnn.enabled,
    "cudnn_deterministic": lambda: torch.backends.cudnn.deterministic,
    "cudnn_benchmark": lambda: torch.backends.cudnn.benchmark,
    "deterministic_algorithms": lambda: torch.are_deterministic_algorithms_enabled(),
    "float32_matmul_precision": lambda: torch.get_float32_matmul_precision(),
}


def _observe(probe, value_type: type) -> dict[str, Any]:
    """Observe live state now; a failed or empty probe is explicit, never a guess."""
    try:
        value = probe()
    except Exception as exc:
        return {"status": "unavailable", "value": None, "reason": type(exc).__name__}
    if value is None:
        return {"status": "unavailable", "value": None, "reason": "not reported"}
    if type(value) is not value_type:
        return {"status": "unavailable", "value": None, "reason": "unexpected type"}
    return {"status": "observed", "value": value}


def _member_recorder(events: MemberEventLog, model_id: int, model: torch.nn.Module):
    """Translate trainer execution callbacks into this member's retained events."""

    def record(event: str, data: dict[str, Any]) -> None:
        if event == "training_started":
            parameter = next(model.parameters(), None)
            events.record(
                model_id,
                "training_started",
                {
                    **data,
                    "device": _observe(lambda: str(parameter.device), str),
                    "parameter_dtype": _observe(lambda: str(parameter.dtype), str),
                    "backend": {
                        name: _observe(_BACKEND_PROBES[name], value_type)
                        for name, value_type in BACKEND_OBSERVATIONS.items()
                    },
                },
            )
        elif event == "checkpoint_saved":
            path = Path(data["path"])
            events.record(
                model_id,
                "checkpoint_saved",
                {
                    "epoch": data["epoch"],
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                },
            )

    return record


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
    execution: ExecutionReference | None = None,
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
        execution: Optional retained execution-start reference. When given, each
            member's applied seed, observed runtime, saved and loaded checkpoint
            digests and its outcome are appended to that attempt's member events.

    Returns:
        Tuple of (list of training results, averaged predictions)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_output_path = output_path if output_path else config.get_output_path()
    checkpoint_dir = os.path.join(base_output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    events = MemberEventLog(execution) if execution is not None else None
    all_results = []
    all_predictions = []

    for model_id in range(config.training.num_models):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Model {model_id + 1}/{config.training.num_models}")
        logger.info(f"{'=' * 60}")

        model_seed = config.seed + model_id
        logger.info(f"Model seed: {model_seed}")
        set_seed(model_seed)
        if events is not None:
            # Recorded only after seeding returned; the planned seed alone is not evidence.
            events.record(
                model_id,
                "seeded",
                {"seed": model_seed, "torch_initial_seed": _observe(torch.initial_seed, int)},
            )

        try:
            model = model_factory()
            model_checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_id}_best.pth")
            trainer = Trainer(
                model=model,
                config=config,
                device=device,
                output_path=base_output_path,
                checkpoint_path=model_checkpoint_path,
            )
            execution_callback = (
                _member_recorder(events, model_id, model) if events is not None else None
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
                    execution_callback=execution_callback,
                )
                all_results.append(result)

                logger.info(
                    f"Model {model_id + 1} training complete. "
                    f"Selected-checkpoint val loss: {format_checkpoint_metric(result.best_val_loss)}, "
                    f"val IC: {format_checkpoint_metric(result.best_val_ic)}, "
                    f"val Rank IC: {format_checkpoint_metric(result.best_val_rank_ic)}"
                )

                trainer.last_best_model_path = result.best_model_path
                trainer.load_best_model(result.best_model_path)
                if events is not None:
                    events.record(
                        model_id,
                        "checkpoint_loaded",
                        {"sha256": trainer.last_loaded_checkpoint_sha256},
                    )
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
        except BaseException as exc:
            if events is not None:
                events.record(model_id, "member_failed", {"error_type": type(exc).__name__})
            raise
        if events is not None:
            events.record(model_id, "member_completed", {})

    avg_predictions = np.mean(all_predictions, axis=0)
    avg_pred_dir = os.path.join(base_output_path, "averaged_predictions")
    os.makedirs(avg_pred_dir, exist_ok=True)

    for idx, date in enumerate(test_dates):
        if idx < len(avg_predictions):
            mask = test_prediction_masks[idx] if test_prediction_masks is not None else None
            data = prediction_rows_for_date(avg_predictions[idx], kdcode_list, date, mask)
            df_pred = pd.DataFrame(columns=["kdcode", "dt", "score"], data=data)
            df_pred.to_csv(os.path.join(avg_pred_dir, f"{date}.csv"), index=False)

    logger.info(f"\nAveraged predictions saved to {avg_pred_dir}")

    return all_results, avg_predictions
