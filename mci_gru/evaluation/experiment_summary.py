"""Experiment-level evaluation summary helpers.

Moved from ``run_experiment.py`` so the research policy they encode
(the evaluation defaults derived from ``label_t``, the selection-metric
key mapping) lives in the package where it is importable and unit-testable.
"""

import hashlib
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation.artifacts import write_json_artifact
from mci_gru.evaluation.metrics import evaluate_run

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


def build_run_metadata(
    config: ExperimentConfig,
    data: dict[str, Any],
    *,
    walkforward_window: int,
    resolved_config_identity: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Any]:
    """One walk-forward window's ``run_metadata.json`` payload.

    Consumed identities come from the sealed preparation result; their sources
    are never reopened here. The separate legacy ``data_file_*`` keys still
    fingerprint the cwd-relative configured path at metadata time, preserving
    notebook compatibility. ``input_observations`` retains every observed read
    and use; ``data_inputs`` projects the first consumed identity per role with
    links to all of its consumed reads.
    """
    return {
        "norm_means": {k: float(v) for k, v in data["norm_means"].items()},
        "norm_stds": {k: float(v) for k, v in data["norm_stds"].items()},
        "feature_cols": data["feature_cols"],
        "kdcode_list": data["kdcode_list"],
        "his_t": config.model.his_t,
        "label_t": config.model.label_t,
        "seed": config.seed,
        "train_end": config.data.train_end,
        "data_file": config.data.filename,
        "walkforward_window": walkforward_window,
        **resolved_config_identity,
        "graph_static_valid_from": data.get("graph_static_valid_from"),
        "feature_reference_path": "feature_reference.json",
        "pit_universe_mode": data.get("pit_universe_mode"),
        "pit_breadth": data.get("pit_breadth"),
        "pit_eligibility": data.get("pit_eligibility"),
        **data_file_fingerprint(config.data.filename, logger),
        "data_inputs": data["input_observations"].data_inputs(),
        "input_observations": data["input_observations"].to_dict(),
    }


def resolved_evaluation_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    eval_cfg = config.evaluation
    label_t = config.model.label_t
    legs = eval_cfg.resolved_legs(label_t)
    return {
        "top_k_values": eval_cfg.top_k_values,
        "label_t": label_t,
        "legs": legs,
        "tranches": eval_cfg.resolved_tranches(label_t),
        "sharpe_method": eval_cfg.sharpe_method,
        "bootstrap_enabled": eval_cfg.bootstrap_enabled,
        "bootstrap_resamples": eval_cfg.bootstrap_resamples,
        "bootstrap_seed": eval_cfg.bootstrap_seed,
        "ci_level": eval_cfg.ci_level,
        "block_size": eval_cfg.block_size or max(1, label_t),
        # Lags for the Sharpe of the daily book. The label spans hold_sessions
        # close-to-close intervals (label_t - 1), not label_t.
        "newey_west_lags": eval_cfg.newey_west_lags
        if eval_cfg.newey_west_lags is not None
        else max(0, legs.hold_sessions - 1),
    }


def compute_evaluation_summary(
    predictions: np.ndarray,
    labels: np.ndarray,
    config: ExperimentConfig,
    *,
    panel: pd.DataFrame,
    dates: list[str],
    kdcodes: list[str],
    rankable_mask: np.ndarray | None,
) -> dict[str, Any]:
    """One window's ``evaluation_summary.json`` (schema v2, #336).

    IC is scored against ``labels``, the training labels; every portfolio number is
    the staggered tranche book on ``panel``'s closes. See ``evaluate_run``.
    """
    return evaluate_run(
        predictions,
        labels,
        panel=panel,
        dates=dates,
        kdcodes=kdcodes,
        rankable_mask=rankable_mask,
        **resolved_evaluation_kwargs(config),
    )


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
