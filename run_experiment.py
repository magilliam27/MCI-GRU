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

import hashlib
import json
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)
from mci_gru.data.data_manager import create_data_loaders
from mci_gru.features import FeatureEngineer
from mci_gru.models import create_model
from mci_gru.pipeline import prepare_data, prepare_data_index_level
from mci_gru.tracking import MLflowTrackingManager
from mci_gru.training import train_multiple_models
from mci_gru.utils.seeding import set_seed
from mci_gru.walkforward import generate_walkforward_configs, merge_walkforward_summary


def _edge_feature_dim(graph_cfg: GraphConfig) -> int:
    """Final GAT ``edge_dim`` after builder columns + optional collate snapshot age."""
    if not graph_cfg.use_multi_feature_edges:
        return 1
    n = 4
    if graph_cfg.use_lead_lag_features:
        n += 2
    if graph_cfg.append_snapshot_age_days:
        n += 1
    return n


def _data_file_fingerprint(relative_path: str, logger: logging.Logger) -> dict[str, Any]:
    """SHA-256 and stat metadata for the configured CSV path (if present)."""
    path = Path(relative_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        logger.warning("Data file not found at %s — skipping sha256", path)
        return {
            "data_file_sha256": None,
            "data_file_size_bytes": None,
            "data_file_mtime_iso": None,
        }
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    return {
        "data_file_sha256": digest.hexdigest(),
        "data_file_size_bytes": st.st_size,
        "data_file_mtime_iso": mtime,
    }


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


def dict_to_config(cfg: DictConfig) -> ExperimentConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = DataConfig(**cfg_dict.get("data", {}))
    feature_cfg = FeatureConfig(**cfg_dict.get("features", {}))
    graph_cfg = GraphConfig(**cfg_dict.get("graph", {}))
    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    training_cfg = TrainingConfig(**cfg_dict.get("training", {}))
    tracking_cfg = TrackingConfig(**cfg_dict.get("tracking", {}))

    return ExperimentConfig(
        data=data_cfg,
        features=feature_cfg,
        graph=graph_cfg,
        model=model_cfg,
        training=training_cfg,
        tracking=tracking_cfg,
        experiment_name=cfg_dict.get("experiment_name", "baseline"),
        output_dir=cfg_dict.get("output_dir", "results"),
        seed=cfg_dict.get("seed", 42),
    )


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
    config = dict_to_config(cfg)
    set_seed(config.seed)
    logger.info(f"\nRandom seed: {config.seed}")
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

            if config.data.experiment_mode == "index_level":
                data = prepare_data_index_level(cfg_w, feature_engineer)
            else:
                data = prepare_data(cfg_w, feature_engineer)

            metadata = {
                "norm_means": {k: float(v) for k, v in data["norm_means"].items()},
                "norm_stds": {k: float(v) for k, v in data["norm_stds"].items()},
                "feature_cols": data["feature_cols"],
                "kdcode_list": data["kdcode_list"],
                "his_t": cfg_w.model.his_t,
                "label_t": cfg_w.model.label_t,
                "seed": cfg_w.seed,
                "train_end": cfg_w.data.train_end,
                "data_file": cfg_w.data.filename,
                "walkforward_window": wi,
                **_data_file_fingerprint(cfg_w.data.filename, logger),
            }
            metadata_path = os.path.join(wpath, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Run metadata saved to: {metadata_path}")

            graph_data_path = os.path.join(wpath, "graph_data.pt")
            torch.save(
                {
                    "edge_index": data["edge_index"],
                    "edge_weight": data["edge_weight"],
                },
                graph_data_path,
            )
            logger.info(f"Graph data saved to: {graph_data_path}")

            logger.info("\nCreating data loaders...")
            dynamic_graph = cfg_w.graph.update_frequency_months > 0
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
                append_snapshot_age_days=cfg_w.graph.append_snapshot_age_days,
                static_graph_valid_from=data.get("graph_static_valid_from"),
                edge_index_sector=data.get("edge_index_sector"),
                edge_weight_sector=data.get("edge_weight_sector"),
                use_sector_relation=cfg_w.graph.use_sector_relation,
            )

            num_features = len(data["feature_cols"])
            edge_feature_dim = _edge_feature_dim(cfg_w.graph)
            model_cfg_dict = {
                **cfg_w.model.to_dict(),
                "edge_feature_dim": edge_feature_dim,
                "drop_edge_p": cfg_w.graph.drop_edge_p,
                "use_sector_relation": cfg_w.graph.use_sector_relation,
            }

            def model_factory():
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
                active_tracking = (
                    window_tracking if window_tracking.enabled else tracking_manager
                )
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
                )

                best_val_losses = [r.best_val_loss for r in results]
                best_val_ics = [r.best_val_ic for r in results]
                training_summary = {
                    "experiment_name": cfg_w.experiment_name,
                    "models_trained": len(results),
                    "best_val_losses": best_val_losses,
                    "best_val_ics": best_val_ics,
                    "mean_best_val_loss": float(np.mean(best_val_losses)) if best_val_losses else None,
                    "mean_best_val_ic": float(np.mean(best_val_ics)) if best_val_ics else None,
                    "walkforward_window": wi,
                }
                training_summary_path = os.path.join(wpath, "training_summary.json")
                with open(training_summary_path, "w") as f:
                    json.dump(training_summary, f, indent=2)
                logger.info(f"Training summary saved to: {training_summary_path}")
                wf_summaries.append(training_summary)

                if active_tracking.enabled:
                    active_tracking.log_metrics(
                        {
                            "models_trained": len(results),
                            "mean_best_val_loss": training_summary["mean_best_val_loss"],
                            "mean_best_val_ic": training_summary["mean_best_val_ic"],
                        },
                        prefix="training.",
                    )
                    if cfg_w.tracking.log_artifacts:
                        for artifact in [metadata_path, graph_data_path, training_summary_path]:
                            if os.path.isfile(artifact):
                                active_tracking.log_artifact(artifact, artifact_path="run_artifacts")
                        for log_path in sorted(Path(wpath).glob("training_*.log")):
                            active_tracking.log_artifact(log_path, artifact_path="logs")
                    if cfg_w.tracking.log_predictions:
                        active_tracking.log_artifacts(
                            Path(wpath) / "averaged_predictions",
                            artifact_path="predictions/averaged",
                        )

        if use_wf_subdir and wf_summaries:
            merged = merge_walkforward_summary(wf_summaries)
            merged_path = os.path.join(output_path, "walkforward_summary.json")
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=2)
            logger.info("Walk-forward aggregate summary: %s", merged_path)

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


if __name__ == "__main__":
    main()
