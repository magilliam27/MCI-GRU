"""Experiment-level evaluation summary helpers.

Moved from ``run_experiment.py`` so the research policy they encode
(Newey-West lag defaults derived from ``label_t``, the selection-metric
key mapping) lives in the package where it is importable and unit-testable.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mci_gru.config import ExperimentConfig
from mci_gru.training import evaluate_predictions


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
