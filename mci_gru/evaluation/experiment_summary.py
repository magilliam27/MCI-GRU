"""Experiment-level evaluation summary helpers.

Moved from ``run_experiment.py`` so the research policy they encode
(Newey-West lag defaults derived from ``label_t``, the selection-metric
key mapping) lives in the package where it is importable and unit-testable.
"""

import hashlib
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation.artifacts import write_json_artifact
from mci_gru.training import evaluate_predictions

RESOLVED_CONFIG_FILENAME = "resolved_config.json"
REDACTED_ABSOLUTE_PATH = "<ABSOLUTE_PATH>"


def _is_absolute_path_text(value: str) -> bool:
    if not value:
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _redact_absolute_paths(value: Any) -> Any:
    """Replace absolute paths with a marker, keeping every key present.

    Absolute paths carry the operator's local layout, but dropping the keys
    outright would leave a reader unable to tell a redacted setting from an
    unset one, which is exactly what this artifact exists to answer.
    """
    if isinstance(value, Path):
        return REDACTED_ABSOLUTE_PATH if value.is_absolute() else value.as_posix()
    if isinstance(value, str):
        return REDACTED_ABSOLUTE_PATH if _is_absolute_path_text(value) else value
    if isinstance(value, dict):
        return {str(key): _redact_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_absolute_paths(item) for item in value]
    return value


def write_resolved_config(
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Persist one window's complete resolved configuration and content identity.

    Refuses to overwrite an existing artifact unless *force*, matching the
    evidence-preservation contract of :func:`write_json_artifact`. The returned
    digest is taken from the bytes actually on disk, so it always describes the
    stored file rather than an in-memory approximation of it.
    """
    path = write_json_artifact(
        Path(output_dir) / RESOLVED_CONFIG_FILENAME,
        _redact_absolute_paths(asdict(config)),
        force=force,
    )
    return {
        "resolved_config_path": path.name,
        "resolved_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def data_file_fingerprint(relative_path: str, logger: logging.Logger) -> dict[str, Any]:
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


def resolved_evaluation_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    eval_cfg = config.evaluation
    return {
        "top_k_values": eval_cfg.top_k_values,
        "label_t": config.model.label_t,
        "bootstrap_enabled": eval_cfg.bootstrap_enabled,
        "bootstrap_resamples": eval_cfg.bootstrap_resamples,
        "bootstrap_seed": eval_cfg.bootstrap_seed,
        "ci_level": eval_cfg.ci_level,
        "block_size": eval_cfg.block_size or max(1, config.model.label_t),
        "newey_west_lags": eval_cfg.newey_west_lags
        if eval_cfg.newey_west_lags is not None
        else max(0, config.model.label_t - 1),
    }


def compute_evaluation_summary(
    predictions: np.ndarray,
    labels: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    metrics = evaluate_predictions(
        predictions,
        labels,
        **resolved_evaluation_kwargs(config),
    )
    return {
        "label_t": config.model.label_t,
        "top_k_values": config.evaluation.top_k_values,
        "metrics": metrics,
    }


def select_training_objective_value(
    selection_metric: str,
    wf_summaries: list[dict[str, Any]],
    merged_summary: dict[str, Any] | None,
) -> float | None:
    """Return the summary objective matching the configured checkpoint metric."""
    if selection_metric == "val_loss":
        key = "mean_best_val_loss"
    elif selection_metric == "val_rank_ic":
        key = "mean_best_val_rank_ic"
    else:
        key = "mean_best_val_ic"

    if merged_summary is not None:
        return merged_summary.get(f"{key}_across_windows")
    if wf_summaries:
        return wf_summaries[-1].get(key)
    return None
