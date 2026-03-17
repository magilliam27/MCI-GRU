"""
Manual MLflow tracking helpers for MCI-GRU.

This module deliberately avoids framework autologging. The training and
backtesting flows in this repository have custom lifecycle management,
custom artifacts, and downstream consumers that depend on the existing
filesystem layout. Manual logging gives us explicit control over what is
tracked and keeps MLflow additive rather than intrusive.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional


def _find_project_root() -> Path:
    """Locate the repository root via git, falling back to package ancestry."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _find_project_root()
RUN_METADATA_FILENAME = "mlflow_run.json"


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking is enabled but the 'mlflow' package is not installed. "
            "Install project requirements or disable tracking with tracking.enabled=false."
        ) from exc
    return mlflow


def resolve_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    """Resolve a tracking URI for local-first usage.

    Relative paths are resolved against the repository root and returned as
    file:// URIs so both Python and ``mlflow ui`` can read the same store.
    """
    uri = tracking_uri or "mlruns"
    if "://" in uri:
        return uri

    tracking_path = Path(uri)
    if not tracking_path.is_absolute():
        tracking_path = PROJECT_ROOT / tracking_path

    tracking_path.mkdir(parents=True, exist_ok=True)
    return tracking_path.resolve().as_uri()


def _to_serializable(value: Any) -> Any:
    """Convert dataclasses/paths into serialisable primitives."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return value


def flatten_params(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten nested mappings into MLflow parameter strings.

    MLflow params are strings, so lists and complex values are JSON-encoded.
    """
    flattened: Dict[str, str] = {}

    for raw_key, raw_value in payload.items():
        key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
        value = _to_serializable(raw_value)

        if isinstance(value, Mapping):
            flattened.update(flatten_params(value, prefix=key))
        elif isinstance(value, (list, tuple, set)):
            flattened[key] = json.dumps(list(value), default=str)
        elif value is None:
            flattened[key] = "null"
        else:
            flattened[key] = str(value)

    return flattened


def load_run_metadata(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Load persisted MLflow run metadata from a run directory."""
    metadata_path = Path(run_dir) / RUN_METADATA_FILENAME
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_metadata_from_predictions_dir(predictions_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Load tracking metadata from a predictions directory.

    The current layout is::

        results/{experiment}/{timestamp}/averaged_predictions/

    so the metadata lives one level above ``averaged_predictions/``.
    """
    return load_run_metadata(Path(predictions_dir).resolve().parent)


class MLflowTrackingManager:
    """Thin manual wrapper around an active MLflow run."""

    def __init__(
        self,
        enabled: bool = False,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ):
        self.enabled = enabled
        self.output_path = Path(output_path).resolve() if output_path else None
        self.parent_run_id = parent_run_id
        self.tracking_uri = resolve_tracking_uri(tracking_uri) if enabled else tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._mlflow = None
        self._run = None

        if not self.enabled:
            return

        if not self.experiment_name:
            raise ValueError("experiment_name is required when MLflow tracking is enabled")

        self._mlflow = _import_mlflow()
        self._mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow.set_experiment(self.experiment_name)

        start_kwargs: Dict[str, Any] = {}
        if self.run_name:
            start_kwargs["run_name"] = self.run_name
        if self.parent_run_id:
            start_kwargs["nested"] = True
            start_kwargs["parent_run_id"] = self.parent_run_id

        self._run = self._mlflow.start_run(**start_kwargs)

        if tags:
            self.log_tags(tags)

    def __enter__(self) -> "MLflowTrackingManager":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(status="FAILED" if exc else "FINISHED")
        return False

    @property
    def is_active(self) -> bool:
        return self.enabled and self._run is not None

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run is not None else None

    @property
    def experiment_id(self) -> Optional[str]:
        return self._run.info.experiment_id if self._run is not None else None

    def close(self, status: str = "FINISHED"):
        """End the active MLflow run."""
        if self.is_active:
            self._mlflow.end_run(status=status)
            self._run = None

    def log_params(self, params: Mapping[str, Any]):
        """Log nested configuration params as flat MLflow params."""
        if not self.is_active:
            return

        for key, value in flatten_params(_to_serializable(params)).items():
            self._mlflow.log_param(key, value)

    def log_tags(self, tags: Mapping[str, Any]):
        """Log run tags. All values are converted to strings."""
        if not self.is_active:
            return

        serialized: Dict[str, str] = {}
        for key, value in tags.items():
            if value is None:
                continue
            converted = _to_serializable(value)
            serialized[str(key)] = (
                json.dumps(converted, default=str)
                if isinstance(converted, (dict, list))
                else str(converted)
            )
        self._mlflow.set_tags(serialized)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """Log numeric metrics, skipping non-finite values."""
        if not self.is_active:
            return

        for key, value in metrics.items():
            if value is None or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue

            metric_key = f"{prefix}{key}" if prefix else str(key)
            self._mlflow.log_metric(metric_key, float(value), step=step)

    def log_artifact(self, path: str | Path, artifact_path: Optional[str] = None):
        """Log one artifact if it exists."""
        if not self.is_active:
            return

        artifact = Path(path)
        if artifact.exists() and artifact.is_file():
            self._mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

    def log_artifacts(self, path: str | Path, artifact_path: Optional[str] = None):
        """Log a directory tree if it exists."""
        if not self.is_active:
            return

        artifact_dir = Path(path)
        if artifact_dir.exists() and artifact_dir.is_dir():
            self._mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_path)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
    ):
        """Standard per-epoch training metrics."""
        self.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
            step=epoch,
        )

    def create_child_run(
        self,
        run_name: str,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> "MLflowTrackingManager":
        """Create a nested child run underneath the current run."""
        if not self.is_active:
            return MLflowTrackingManager(enabled=False)

        return MLflowTrackingManager(
            enabled=True,
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
            run_name=run_name,
            parent_run_id=self.run_id,
            tags=tags,
        )

    def persist_run_metadata(
        self,
        output_path: Optional[str | Path] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist enough metadata for later linked runs such as backtests."""
        if not self.is_active:
            return None

        destination_root = Path(output_path).resolve() if output_path else self.output_path
        if destination_root is None:
            raise ValueError("output_path is required to persist MLflow metadata")

        destination_root.mkdir(parents=True, exist_ok=True)
        metadata_path = destination_root / RUN_METADATA_FILENAME

        payload: Dict[str, Any] = {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "run_name": self.run_name,
        }
        if extra_metadata:
            payload.update({str(key): _to_serializable(value) for key, value in extra_metadata.items()})

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        return metadata_path
