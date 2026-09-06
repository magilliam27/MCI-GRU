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
from mci_gru.data.path_resolver import resolve_project_data_path
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


def data_input_identity(configured_path: str, logger: logging.Logger) -> dict[str, Any]:
    """Resolved path and content identity for one configured data input.

    Wraps :func:`data_file_fingerprint` — the single hasher — but resolves the
    configured string through :func:`resolve_project_data_path` first, so the
    digest describes the file the loaders actually open rather than one guessed
    relative to the launch directory. Recording ``resolved_path`` alongside
    ``configured_path`` is what makes a basename-fallback substitution visible
    in the artifact.
    """
    try:
        resolved = resolve_project_data_path(configured_path)
    except FileNotFoundError:
        # Today a missing input is a warning plus null fields, matching the
        # existing top-level data_file_* behaviour. Whether an unresolvable
        # configured input should instead abort the run is the data-contract
        # decision tracked separately; do not promote it here.
        logger.warning(
            "Data input not found for configured path %s — skipping sha256",
            configured_path,
        )
        return {
            "configured_path": configured_path,
            "resolved_path": None,
            "sha256": None,
            "size_bytes": None,
            "mtime_iso": None,
        }
    fingerprint = data_file_fingerprint(str(resolved), logger)
    return {
        "configured_path": configured_path,
        "resolved_path": str(resolved),
        "sha256": fingerprint["data_file_sha256"],
        "size_bytes": fingerprint["data_file_size_bytes"],
        "mtime_iso": fingerprint["data_file_mtime_iso"],
    }


def configured_data_inputs(config: ExperimentConfig) -> list[tuple[str, str]]:
    """``(config field, configured path)`` for every raw input this run reads.

    The gates mirror the loaders one-for-one, so an input that the run never
    opens is never recorded: ``index_level`` mode reads ``data.index_filename``
    through :class:`~mci_gru.data.data_manager.DataManager.load_index_series`
    and never touches ``data.filename``; the PIT universe is read only when
    ``data.use_pit_universe``; the sector map only when
    ``graph.use_sector_relation``; and the deprecated regime CSV only when
    ``features.include_global_regime``. An unset or null path yields no entry.
    """
    entries: list[tuple[str, str]] = []
    if config.data.experiment_mode == "index_level":
        if config.data.index_filename:
            entries.append(("data.index_filename", config.data.index_filename))
    elif config.data.source == "csv" and config.data.filename:
        entries.append(("data.filename", config.data.filename))
    if config.data.use_pit_universe and config.data.pit_universe_csv:
        entries.append(("data.pit_universe_csv", config.data.pit_universe_csv))
    if config.graph.use_sector_relation and config.graph.sector_map_csv:
        entries.append(("graph.sector_map_csv", config.graph.sector_map_csv))
    if config.features.include_global_regime and config.features.regime_inputs_csv:
        entries.append(("features.regime_inputs_csv", config.features.regime_inputs_csv))
    return entries


def data_inputs_identity(
    config: ExperimentConfig,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """Identity of every configured data input this run consumes.

    ``data.pit_universe_csv`` is opened with a bare ``pd.read_csv`` rather than
    through the resolver, so its ``configured_path`` is the string that read
    used and ``resolved_path`` is where the resolver finds the same name; a
    disagreement between the two is itself the finding.
    """
    return {
        field: data_input_identity(path, logger) for field, path in configured_data_inputs(config)
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

    Pure in *config*, *data* and *resolved_config_identity*, so the artifact a
    run writes is observable without running one. The top-level ``data_file_*``
    keys keep their existing cwd-relative meaning — the notebook generators
    read ``data_file_sha256`` — and ``data_inputs`` adds the resolved identity
    of every input the window consumed.
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
        **data_file_fingerprint(config.data.filename, logger),
        "data_inputs": data_inputs_identity(config, logger),
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
