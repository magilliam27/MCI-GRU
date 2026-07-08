"""Upload allowlist for Colab saved-prediction recovery runs.

The recovery notebooks upload row-level run directories to Drive after training.
This helper keeps that upload narrow by default: averaged ensemble predictions
and provenance metadata are durable evidence; per-model prediction CSVs and
checkpoints are opt-in archival payloads.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

ROOT_METADATA_FILES = frozenset(
    {
        "config.yaml",
        "run_metadata.json",
        "training_summary.json",
        "evaluation_summary.json",
        "timing_summary.json",
        "feature_reference.json",
        "run_summary.json",
    }
)
HYDRA_METADATA_FILES = frozenset({"config.yaml", "hydra.yaml", "overrides.yaml"})
CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth"})


def _relative_path(path: Path, run_dir: Path) -> Path:
    try:
        return path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"{path} is not under run_dir {run_dir}") from exc


def should_upload_recovery_artifact(
    path: Path | str,
    run_dir: Path | str,
    *,
    upload_checkpoints: bool = False,
    upload_per_model_predictions: bool = False,
) -> bool:
    """Return whether ``path`` should be mirrored to Drive for a row run.

    Default behavior intentionally excludes broad recursive CSV matches. In
    particular, ``predictions_model_*/*.csv`` files are excluded unless the
    caller explicitly opts into per-model archival.
    """

    file_path = Path(path)
    root = Path(run_dir)
    if not file_path.is_file():
        return False

    rel = _relative_path(file_path, root)
    parts = rel.parts
    if not parts:
        return False

    top = parts[0]
    suffix = file_path.suffix.lower()

    if top == "averaged_predictions":
        return True

    if top.startswith("predictions_model_"):
        return upload_per_model_predictions and suffix == ".csv"

    if top == "checkpoints":
        return upload_checkpoints and suffix in CHECKPOINT_SUFFIXES

    if top == ".hydra" and len(parts) == 2:
        return parts[1] in HYDRA_METADATA_FILES

    if len(parts) == 1:
        name = parts[0]
        return name in ROOT_METADATA_FILES or (name.startswith("training_") and suffix == ".log")

    return False


def iter_recovery_upload_files(
    run_dir: Path | str,
    *,
    upload_checkpoints: bool = False,
    upload_per_model_predictions: bool = False,
) -> Iterable[Path]:
    """Yield row-run files that should be uploaded, sorted by relative path."""

    root = Path(run_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for file_path in files:
        if should_upload_recovery_artifact(
            file_path,
            root,
            upload_checkpoints=upload_checkpoints,
            upload_per_model_predictions=upload_per_model_predictions,
        ):
            yield file_path


def count_averaged_prediction_csvs(run_dir: Path | str) -> int:
    """Count durable ensemble prediction CSVs for heartbeat/row summaries."""

    return len(list((Path(run_dir) / "averaged_predictions").glob("*.csv")))
