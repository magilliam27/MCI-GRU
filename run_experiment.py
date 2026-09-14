#!/usr/bin/env python
"""
MCI-GRU Experiment Runner

Main entry point for running MCI-GRU experiments with Hydra configuration.

Usage:
    # Run baseline experiment
    python run_experiment.py

    # Override lookback period
    python run_experiment.py model.his_t=5

    # Use a different experiment preset
    python run_experiment.py +experiment=with_vix

    # Sweep over multiple values
    python run_experiment.py --multirun model.his_t=5,10,15,20

    # Use Russell 1000 data
    python run_experiment.py +data=russell1000

    # Combine overrides
    python run_experiment.py +experiment=with_vix +data=russell1000 model.his_t=20
"""

import json
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mci_gru.config import create_config_from_dict
from mci_gru.data.data_manager import create_data_loaders
from mci_gru.evaluation.experiment_summary import (
    build_run_metadata,
    compute_evaluation_summary,
    select_training_objective_value,
    write_resolved_config,
)
from mci_gru.features import FeatureEngineer
from mci_gru.graph.utils import edge_feature_dim
from mci_gru.models import create_model
from mci_gru.pipeline import prepare_data, prepare_data_index_level
from mci_gru.tracking import MLflowTrackingManager
from mci_gru.training import train_multiple_models
from mci_gru.utils.seeding import set_seed
from mci_gru.walkforward import generate_walkforward_configs, merge_walkforward_summary


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig

    try:
        hydra_cfg = HydraConfig.get()
        output_path = hydra_cfg.runtime.output_dir
    except Exception:
        output_path = os.getcwd()

    logger = setup_logging(output_path, cfg.get("experiment_name", "baseline"))

    logger.info("=" * 80)
    logger.info("MCI-GRU Experiment Runner")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
    set_seed(config.seed)
    logger.info(f"\nBase random seed: {config.seed}")
    logger.info(f"Output directory: {output_path}")
    config_path = os.path.join(output_path, "config.yaml")
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to: {config_path}")

    logger.info("\nInitializing feature engineer...")
    feature_engineer = FeatureEngineer(config.features)

    window_configs = generate_walkforward_configs(config)
    use_wf_subdir = config.training.walkforward.enabled
    wf_summaries: list[dict[str, Any]] = []

    tracking_experiment_name = config.tracking.experiment_name or config.experiment_name
    tracking_run_name = (
        config.tracking.run_name or f"{config.experiment_name}-{Path(output_path).name}"
    )
    tracking_manager = MLflowTrackingManager(
        enabled=config.tracking.enabled,
        tracking_uri=config.tracking.tracking_uri,
        experiment_name=tracking_experiment_name,
        run_name=tracking_run_name,
        output_path=output_path,
        tags={
            "run_kind": "training_parent",
            "experiment_name": config.experiment_name,
            "output_path": output_path,
            "data_source": config.data.source,
            "experiment_mode": config.data.experiment_mode,
            "loss_type": config.training.loss_type,
            "label_type": config.training.label_type,
        },
    )

    try:
        objective_value = None
        if tracking_manager.enabled:
            tracking_manager.log_params(OmegaConf.to_container(cfg, resolve=True))
            mlflow_meta = tracking_manager.persist_run_metadata(
                extra_metadata={"output_path": output_path}
            )
            if mlflow_meta is not None:
                logger.info(f"MLflow run metadata saved to: {mlflow_meta}")

        for wi, cfg_w in enumerate(window_configs):
            wpath = (
                os.path.join(output_path, "walkforward", f"w{wi:03d}")
                if use_wf_subdir
                else output_path
            )
            if use_wf_subdir:
                os.makedirs(wpath, exist_ok=True)

            logger.info("\n" + "=" * 80)
            logger.info(
                "Walk-forward window %s / %s - output %s",
                wi + 1,
                len(window_configs),
                wpath,
            )
            logger.info("=" * 80)
            resolved_config_identity = write_resolved_config(cfg_w, wpath, force=True)

            window_started = perf_counter()
            timing_summary: dict[str, Any] = {
                "walkforward_window": wi,
                "output_path": wpath,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "phases": {
                    "backtest_replay_handoff_seconds": 0.0,
                },
            }

            phase_started = perf_counter()
            if config.data.experiment_mode == "index_level":
                data = prepare_data_index_level(cfg_w, feature_engineer)
            else:
                data = prepare_data(cfg_w, feature_engineer)
            timing_summary["phases"]["prepare_data_seconds"] = perf_counter() - phase_started

            metadata = build_run_metadata(
                cfg_w,
                data,
                walkforward_window=wi,
                resolved_config_identity=resolved_config_identity,
                logger=logger,
            )
            metadata_path = os.path.join(wpath, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Run metadata saved to: {metadata_path}")

            feature_reference_path = os.path.join(wpath, "feature_reference.json")
            with open(feature_reference_path, "w") as f:
                json.dump(data.get("feature_reference", {"features": {}}), f, indent=2)
            logger.info(f"Feature reference saved to: {feature_reference_path}")

            graph_data_path = os.path.join(wpath, "graph_data.pt")
            torch.save(
                {
                    "edge_index": data["edge_index"],
                    "edge_weight": data["edge_weight"],
                    "edge_index_sector": data.get("edge_index_sector"),
                    "edge_weight_sector": data.get("edge_weight_sector"),
                },
                graph_data_path,
            )
            logger.info(f"Graph data saved to: {graph_data_path}")

            logger.info("\nCreating data loaders...")
            dynamic_graph = cfg_w.graph.update_frequency_months > 0
            phase_started = perf_counter()
            train_loader, val_loader, test_loader = create_data_loaders(
                stock_features_train=data["stock_features_train"],
                x_graph_train=data["x_graph_train"],
                train_labels=data["train_labels"],
                stock_features_val=data["stock_features_val"],
                x_graph_val=data["x_graph_val"],
                val_labels=data["val_labels"],
                stock_features_test=data["stock_features_test"],
                x_graph_test=data["x_graph_test"],
                edge_index=data["edge_index"],
                edge_weight=data["edge_weight"],
                batch_size=cfg_w.training.batch_size,
                train_dates=data["train_dates"],
                val_dates=data["val_dates"],
                test_dates=data["test_dates"],
                dynamic_graph=dynamic_graph,
                graph_schedule=data.get("graph_schedule"),
                shuffle_train=cfg_w.training.shuffle_train,
                append_snapshot_age_days=cfg_w.graph.append_snapshot_age_days,
                static_graph_valid_from=data.get("graph_static_valid_from"),
                edge_index_sector=data.get("edge_index_sector"),
                edge_weight_sector=data.get("edge_weight_sector"),
                use_sector_relation=cfg_w.graph.use_sector_relation,
                train_stock_masks=data.get("train_tradable_mask"),
                val_stock_masks=data.get("val_tradable_mask"),
                test_stock_masks=data.get("test_tradable_mask"),
                dataloader_num_workers=cfg_w.training.dataloader_num_workers,
                dataloader_pin_memory=cfg_w.training.dataloader_pin_memory,
                dataloader_persistent_workers=cfg_w.training.dataloader_persistent_workers,
                dataloader_prefetch_factor=cfg_w.training.dataloader_prefetch_factor,
            )
            timing_summary["phases"]["loader_creation_seconds"] = perf_counter() - phase_started

            num_features = len(data["feature_cols"])
            edge_dim = edge_feature_dim(cfg_w.graph)
            model_cfg_dict = {
                **cfg_w.model.to_dict(),
                "edge_feature_dim": edge_dim,
                "drop_edge_p": cfg_w.graph.drop_edge_p,
                "isolate_edge_dropout_rng": cfg_w.graph.isolate_edge_dropout_rng,
                "use_sector_relation": cfg_w.graph.use_sector_relation,
            }

            def model_factory(num_features=num_features, model_cfg_dict=model_cfg_dict):
                return create_model(num_features, model_cfg_dict)

            logger.info("\n" + "=" * 80)
            logger.info("Training")
            logger.info("=" * 80)

            window_ctx = nullcontext(tracking_manager)
            if tracking_manager.enabled and use_wf_subdir:
                window_ctx = tracking_manager.create_child_run(
                    run_name=f"window_{wi}",
                    tags={"window_id": str(wi), "run_kind": "walkforward_window"},
                )

            with window_ctx as window_tracking:
                active_tracking = window_tracking if window_tracking.enabled else tracking_manager
                phase_started = perf_counter()
                results, avg_predictions = train_multiple_models(
                    model_factory=model_factory,
                    config=cfg_w,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    kdcode_list=data["kdcode_list"],
                    test_dates=data["test_dates"],
                    output_path=wpath,
                    tracking_manager=active_tracking,
                    test_prediction_masks=data.get("test_tradable_mask"),
                )
                timing_summary["phases"]["model_training_prediction_export_seconds"] = (
                    perf_counter() - phase_started
                )

                best_val_losses = [r.best_val_loss for r in results]
                best_val_ics = [r.best_val_ic for r in results]
                best_val_rank_ics = [r.best_val_rank_ic for r in results]
                training_summary = {
                    "experiment_name": cfg_w.experiment_name,
                    "models_trained": len(results),
                    "best_val_losses": best_val_losses,
                    "best_val_ics": best_val_ics,
                    "best_val_rank_ics": best_val_rank_ics,
                    "mean_best_val_loss": float(np.mean(best_val_losses))
                    if best_val_losses
                    else None,
                    "mean_best_val_ic": float(np.mean(best_val_ics)) if best_val_ics else None,
                    "mean_best_val_rank_ic": (
                        float(np.mean(best_val_rank_ics)) if best_val_rank_ics else None
                    ),
                    "walkforward_window": wi,
                }
                training_summary_path = os.path.join(wpath, "training_summary.json")
                with open(training_summary_path, "w") as f:
                    json.dump(training_summary, f, indent=2)
                logger.info(f"Training summary saved to: {training_summary_path}")

                phase_started = perf_counter()
                evaluation_summary = compute_evaluation_summary(
                    avg_predictions,
                    data["test_labels"],
                    cfg_w,
                )
                timing_summary["phases"]["evaluation_summary_seconds"] = (
                    perf_counter() - phase_started
                )
                evaluation_summary_path = os.path.join(wpath, "evaluation_summary.json")
                with open(evaluation_summary_path, "w") as f:
                    json.dump(evaluation_summary, f, indent=2)
                logger.info(f"Evaluation summary saved to: {evaluation_summary_path}")
                training_summary["evaluation"] = evaluation_summary["metrics"]
                wf_summaries.append(training_summary)

                if active_tracking.enabled:
                    active_tracking.log_metrics(
                        {
                            "models_trained": len(results),
                            "mean_best_val_loss": training_summary["mean_best_val_loss"],
                            "mean_best_val_ic": training_summary["mean_best_val_ic"],
                            "mean_best_val_rank_ic": training_summary["mean_best_val_rank_ic"],
                        },
                        prefix="training.",
                    )
                    active_tracking.log_metrics(
                        evaluation_summary["metrics"],
                        prefix="evaluation.",
                    )
                    artifact_sync_started = perf_counter()
                    if cfg_w.tracking.log_artifacts:
                        for artifact in [
                            metadata_path,
                            feature_reference_path,
                            graph_data_path,
                            training_summary_path,
                            evaluation_summary_path,
                        ]:
                            if os.path.isfile(artifact):
                                active_tracking.log_artifact(
                                    artifact, artifact_path="run_artifacts"
                                )
                        for log_path in sorted(Path(wpath).glob("training_*.log")):
                            active_tracking.log_artifact(log_path, artifact_path="logs")
                    if cfg_w.tracking.log_predictions:
                        active_tracking.log_artifacts(
                            Path(wpath) / "averaged_predictions",
                            artifact_path="predictions/averaged",
                        )
                    timing_summary["phases"]["artifact_sync_seconds"] = (
                        perf_counter() - artifact_sync_started
                    )
                else:
                    timing_summary["phases"]["artifact_sync_seconds"] = 0.0

                timing_summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                timing_summary["elapsed_seconds"] = perf_counter() - window_started
                timing_summary_path = os.path.join(wpath, "timing_summary.json")
                with open(timing_summary_path, "w") as f:
                    json.dump(timing_summary, f, indent=2)
                logger.info(f"Timing summary saved to: {timing_summary_path}")

                if (
                    active_tracking.enabled
                    and cfg_w.tracking.log_artifacts
                    and os.path.isfile(timing_summary_path)
                ):
                    active_tracking.log_artifact(
                        timing_summary_path,
                        artifact_path="run_artifacts",
                    )

        if use_wf_subdir and wf_summaries:
            merged = merge_walkforward_summary(wf_summaries)
            merged_path = os.path.join(output_path, "walkforward_summary.json")
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=2)
            logger.info("Walk-forward aggregate summary: %s", merged_path)
            objective_value = select_training_objective_value(
                config.training.selection_metric,
                wf_summaries,
                merged,
            )
        elif wf_summaries:
            objective_value = select_training_objective_value(
                config.training.selection_metric,
                wf_summaries,
                None,
            )

        logger.info("\n" + "=" * 80)
        logger.info("Experiment Complete")
        logger.info("=" * 80)
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Walk-forward windows run: {len(window_configs)}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 80)
    except Exception:
        tracking_manager.close(status="FAILED")
        raise
    else:
        tracking_manager.close(status="FINISHED")
        return objective_value


if __name__ == "__main__":
    main()
