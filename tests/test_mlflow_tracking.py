import tempfile
from pathlib import Path
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mci_gru.config import TrackingConfig, create_config_from_dict
from mci_gru.tracking import (
    MLflowTrackingManager,
    load_run_metadata,
    load_run_metadata_from_predictions_dir,
    resolve_tracking_uri,
)
from run_experiment import dict_to_config
from tests.backtest_sp500 import setup_backtest_tracking


mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient


def test_tracking_config_defaults_disabled():
    cfg = TrackingConfig()
    assert cfg.enabled is False
    assert cfg.tracking_uri == "mlruns"
    assert cfg.log_artifacts is True
    assert cfg.log_checkpoints is True
    assert cfg.log_predictions is False


def test_config_builders_include_tracking_settings():
    config_dict = {
        "experiment_name": "mlflow_test",
        "tracking": {
            "enabled": True,
            "tracking_uri": "custom_mlruns",
            "experiment_name": "custom-exp",
            "log_predictions": True,
        },
    }

    created = create_config_from_dict(config_dict)
    assert created.tracking.enabled is True
    assert created.tracking.tracking_uri == "custom_mlruns"
    assert created.tracking.experiment_name == "custom-exp"
    assert created.tracking.log_predictions is True

    hydra_cfg = OmegaConf.create(config_dict)
    typed = dict_to_config(hydra_cfg)
    assert typed.tracking.enabled is True
    assert typed.tracking.experiment_name == "custom-exp"


def test_mlflow_manager_round_trip_logs_params_metrics_and_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        tracking_root = Path(tmp) / "mlruns"
        output_dir = Path(tmp) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = output_dir / "artifact.txt"
        artifact_path.write_text("artifact payload", encoding="utf-8")

        with MLflowTrackingManager(
            enabled=True,
            tracking_uri=str(tracking_root),
            experiment_name="unit-test-exp",
            run_name="parent-run",
            output_path=output_dir,
        ) as tracking_manager:
            tracking_manager.log_params(
                {
                    "seed": 42,
                    "nested": {"loss_type": "mse", "layers": [32, 10]},
                }
            )
            tracking_manager.log_metrics({"train_loss": 0.123}, step=1)
            tracking_manager.log_artifact(artifact_path, artifact_path="artifacts")
            metadata_path = tracking_manager.persist_run_metadata()
            run_id = tracking_manager.run_id

        assert metadata_path is not None
        assert metadata_path.exists()

        loaded_metadata = load_run_metadata(output_dir)
        assert loaded_metadata is not None
        assert loaded_metadata["run_id"] == run_id

        client = MlflowClient(tracking_uri=resolve_tracking_uri(str(tracking_root)))
        run = client.get_run(run_id)
        assert run.data.params["seed"] == "42"
        assert run.data.params["nested.loss_type"] == "mse"
        assert run.data.params["nested.layers"] == "[32, 10]"
        assert run.data.metrics["train_loss"] == pytest.approx(0.123)

        artifacts = client.list_artifacts(run_id, path="artifacts")
        assert any(item.path.endswith("artifact.txt") for item in artifacts)


def test_backtest_tracking_links_child_run_to_saved_training_run():
    with tempfile.TemporaryDirectory() as tmp:
        tracking_root = Path(tmp) / "mlruns"
        run_dir = Path(tmp) / "results" / "baseline" / "20260310_010203"
        predictions_dir = run_dir / "averaged_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        with MLflowTrackingManager(
            enabled=True,
            tracking_uri=str(tracking_root),
            experiment_name="baseline-exp",
            run_name="training-parent",
            output_path=run_dir,
        ) as training_manager:
            training_manager.persist_run_metadata()
            parent_run_id = training_manager.run_id

        linked_metadata = load_run_metadata_from_predictions_dir(predictions_dir)
        assert linked_metadata is not None
        assert linked_metadata["run_id"] == parent_run_id

        backtest_manager, loaded_metadata = setup_backtest_tracking(
            predictions_dir=str(predictions_dir),
            config={"top_k": 10, "label_t": 5},
            backtest_suffix="_tc",
        )

        assert loaded_metadata is not None
        with backtest_manager:
            child_run_id = backtest_manager.run_id
            backtest_manager.log_metrics({"ARR": 0.25}, prefix="backtest.")

        client = MlflowClient(tracking_uri=resolve_tracking_uri(str(tracking_root)))
        child_run = client.get_run(child_run_id)
        assert child_run.data.tags["mlflow.parentRunId"] == parent_run_id
        assert child_run.data.metrics["backtest.ARR"] == pytest.approx(0.25)
