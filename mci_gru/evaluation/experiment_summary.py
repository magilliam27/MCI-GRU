"""Experiment-level evaluation summary helpers.

Moved from ``run_experiment.py`` so the research policy they encode
(Newey-West lag defaults derived from ``label_t``, the selection-metric
key mapping) lives in the package where it is importable and unit-testable.
"""

import hashlib
import logging
from collections.abc import Callable, Sequence
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
LINKED_SCORE_TIE_TOLERANCE = 1e-6


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


def linked_pairs(edge_index: Any) -> np.ndarray:
    """Unordered pairs of distinct stocks joined by an edge, as a sorted ``(P, 2)`` array.

    Direction, duplicates and self-loops are dropped: a pair is linked once whatever
    the edge list repeats.
    """
    if edge_index is None:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edges.shape}")
    lo = np.minimum(edges[0], edges[1])
    hi = np.maximum(edges[0], edges[1])
    distinct = lo != hi
    if not distinct.any():
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.stack([lo[distinct], hi[distinct]], axis=1), axis=0)


def correlation_edges_for_date(data: dict[str, Any]) -> Callable[[str], Any]:
    """The correlation ``edge_index`` the test loader gives the model on each date.

    A dynamic run reads its ``GraphSchedule`` snapshot, as the collate does; a
    static run uses the one graph throughout.
    """
    schedule = data.get("graph_schedule")
    if schedule is not None:
        return lambda date: schedule.get_graph_for_date(date)[0]
    edge_index = data["edge_index"]
    return lambda date: edge_index


def count_linked_score_ties(
    scores: np.ndarray,
    pairs_by_date: Sequence[np.ndarray],
    tolerance: float = LINKED_SCORE_TIE_TOLERANCE,
) -> tuple[list[int], list[int]]:
    """Per date, the linked pairs with both scores finite, and those within ``tolerance``.

    A non-finite score (a masked, non-tradable stock) takes its pairs out of both counts.
    """
    scores = np.asarray(scores, dtype=np.float64)
    evaluated: list[int] = []
    tied: list[int] = []
    for row, pairs in zip(scores, pairs_by_date, strict=True):
        a, b = row[pairs[:, 0]], row[pairs[:, 1]]
        finite = np.isfinite(a) & np.isfinite(b)
        evaluated.append(int(finite.sum()))
        tied.append(int((finite & (np.abs(a - b) <= tolerance)).sum()))
    return evaluated, tied


def linked_score_tie_summary(
    avg_predictions: np.ndarray,
    member_predictions: Sequence[np.ndarray],
    test_dates: Sequence[str],
    edge_index_for_date: Callable[[str], Any],
    *,
    eval_autocast: bool,
    tolerance: float = LINKED_SCORE_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Count correlation-linked pairs whose test scores tie, for the ensemble and each member.

    Linked names that share a neighbourhood scored identically through the GAT v1
    layers (#319), and nothing reported it. ``eval_autocast`` records whether the
    scores were produced under fp16 autocast, where unrelated names also collide.
    """
    dates = [str(d) for d in test_dates]
    pairs_by_date = [linked_pairs(edge_index_for_date(d)) for d in dates]

    def by_date(counts: list[int]) -> dict[str, Any]:
        return {"per_date": dict(zip(dates, counts, strict=True)), "total": int(sum(counts))}

    evaluated, ensemble_tied = count_linked_score_ties(avg_predictions, pairs_by_date, tolerance)
    members = []
    for model_id, scores in enumerate(member_predictions):
        _, tied = count_linked_score_ties(scores, pairs_by_date, tolerance)
        members.append({"model_id": model_id, **by_date(tied)})
    return {
        "graph": "correlation",
        "tolerance": tolerance,
        "scores_dtype": str(np.asarray(avg_predictions).dtype),
        "eval_autocast": bool(eval_autocast),
        "linked_pairs": by_date(evaluated),
        "ensemble": by_date(ensemble_tied),
        "members": members,
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
