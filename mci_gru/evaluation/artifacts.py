from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

_DROP = object()
_RESEARCH_BUNDLE_FILES = (
    "protocol.json",
    "date_evidence.csv",
    "result.json",
    "report.md",
)


def to_jsonable(value: Any) -> Any:
    """Convert NumPy and non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize *payload* as strict, deterministic UTF-8 JSON with a final LF."""
    text = json.dumps(
        to_jsonable(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def canonical_csv_bytes(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> bytes:
    """Serialize rows with a fixed schema, stable ordering, and stable numbers."""
    column_names = tuple(str(column) for column in columns)
    if not column_names:
        raise ValueError("CSV columns must not be empty")
    if len(set(column_names)) != len(column_names):
        raise ValueError("CSV columns must be unique")

    serialized_rows: list[tuple[str, ...]] = []
    allowed = set(column_names)
    for row in rows:
        extra = {str(key) for key in row} - allowed
        if extra:
            raise ValueError(f"CSV row contains undeclared columns: {sorted(extra)}")
        serialized_rows.append(tuple(_csv_value(row.get(column)) for column in column_names))
    serialized_rows.sort()

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(column_names)
    writer.writerows(serialized_rows)
    return stream.getvalue().encode("utf-8")


def build_research_study_id(
    *,
    research_semantics_version: str,
    protocol: Mapping[str, Any],
    input_hashes: Mapping[str, Any],
    code_identity: Mapping[str, Any],
) -> str:
    """Return a semantic study ID that is independent of operational roots."""
    if not research_semantics_version.strip():
        raise ValueError("research_semantics_version must not be empty")
    identity = {
        "code_identity": _semantic_payload(code_identity),
        "input_hashes": _semantic_payload(input_hashes),
        "protocol": _semantic_payload(protocol),
        "research_semantics_version": research_semantics_version,
        "schema": "mci_gru.selection_research_identity.v1",
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def write_selection_research_bundle(
    output_root: str | Path,
    *,
    research_semantics_version: str,
    protocol: Mapping[str, Any],
    input_hashes: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    date_evidence: Iterable[Mapping[str, Any]],
    date_evidence_columns: Sequence[str],
    result: Mapping[str, Any],
    report: str,
) -> dict[str, str | Path]:
    """Write one immutable five-file selection-research evidence bundle.

    The final directory name is the semantic study ID. Operational absolute
    paths are removed from canonical payloads and redacted from the report.
    The four evidence artifacts are staged first; ``manifest.json`` is staged
    last and hashes those four exact byte streams.
    """
    canonical_protocol = _semantic_payload(protocol)
    canonical_inputs = _semantic_payload(input_hashes)
    canonical_code = _semantic_payload(code_identity)
    canonical_result = _semantic_payload(result)
    study_id = build_research_study_id(
        research_semantics_version=research_semantics_version,
        protocol=protocol,
        input_hashes=input_hashes,
        code_identity=code_identity,
    )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    bundle_dir = root / study_id

    absolute_paths = _collect_absolute_paths(
        protocol,
        input_hashes,
        code_identity,
        result,
        root.resolve(),
    )
    leaf_bytes = {
        "protocol.json": canonical_json_bytes(canonical_protocol),
        "date_evidence.csv": canonical_csv_bytes(date_evidence, date_evidence_columns),
        "result.json": canonical_json_bytes(canonical_result),
        "report.md": _canonical_markdown_bytes(report, absolute_paths),
    }
    manifest = {
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in leaf_bytes.items()
        },
        "code_identity": canonical_code,
        "input_hashes": canonical_inputs,
        "protocol_sha256": hashlib.sha256(leaf_bytes["protocol.json"]).hexdigest(),
        "research_semantics_version": research_semantics_version,
        "schema": "mci_gru.selection_research_manifest.v1",
        "study_id": study_id,
    }
    expected_bytes = {
        **leaf_bytes,
        "manifest.json": canonical_json_bytes(manifest),
    }
    if bundle_dir.exists():
        return _verify_existing_research_bundle(
            bundle_dir,
            study_id=study_id,
            expected_bytes=expected_bytes,
        )

    staging_dir = root / f".{study_id}.staging-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        for name in _RESEARCH_BUNDLE_FILES:
            (staging_dir / name).write_bytes(leaf_bytes[name])
        (staging_dir / "manifest.json").write_bytes(expected_bytes["manifest.json"])
        if bundle_dir.exists():
            verified = _verify_existing_research_bundle(
                bundle_dir,
                study_id=study_id,
                expected_bytes=expected_bytes,
            )
            shutil.rmtree(staging_dir)
            return verified
        staging_dir.rename(bundle_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return _research_bundle_paths(bundle_dir, study_id)


def _verify_existing_research_bundle(
    bundle_dir: Path,
    *,
    study_id: str,
    expected_bytes: Mapping[str, bytes],
) -> dict[str, str | Path]:
    actual_names = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    if actual_names != set(expected_bytes):
        raise FileExistsError(
            f"Existing research bundle does not match canonical study {study_id}: file set differs"
        )
    mismatches = [
        name
        for name, expected in expected_bytes.items()
        if (bundle_dir / name).read_bytes() != expected
    ]
    if mismatches:
        raise FileExistsError(
            f"Existing research bundle does not match canonical study {study_id}: "
            f"byte mismatch in {sorted(mismatches)}"
        )
    return _research_bundle_paths(bundle_dir, study_id)


def _research_bundle_paths(
    bundle_dir: Path,
    study_id: str,
) -> dict[str, str | Path]:
    return {
        "study_id": study_id,
        "bundle_dir": bundle_dir,
        "protocol": bundle_dir / "protocol.json",
        "date_evidence": bundle_dir / "date_evidence.csv",
        "result": bundle_dir / "result.json",
        "report": bundle_dir / "report.md",
        "manifest": bundle_dir / "manifest.json",
    }


def write_json_artifact(path: str | Path, payload: Any, *, force: bool = False) -> Path:
    """Write a strict JSON artifact, refusing to overwrite existing evidence by default."""
    artifact_path = Path(path)
    if artifact_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(to_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return artifact_path


def _semantic_payload(value: Any) -> Any:
    normalized = _remove_absolute_paths(value)
    return None if normalized is _DROP else to_jsonable(normalized)


def _remove_absolute_paths(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _remove_absolute_paths(value.item())
    if isinstance(value, np.ndarray):
        return _remove_absolute_paths(value.tolist())
    if isinstance(value, Path):
        return _DROP if value.is_absolute() else value.as_posix()
    if isinstance(value, str):
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        return _DROP if _is_absolute_path_text(normalized) else normalized
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_absolute_path_text(key_text):
                continue
            normalized = _remove_absolute_paths(item)
            if normalized is not _DROP:
                output[key_text] = normalized
        return output
    if isinstance(value, (list, tuple)):
        items = [_remove_absolute_paths(item) for item in value]
        return [item for item in items if item is not _DROP]
    if isinstance(value, (set, frozenset)):
        items = [_remove_absolute_paths(item) for item in value]
        return sorted((item for item in items if item is not _DROP), key=repr)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return 0.0 if value == 0.0 else value
    return value


def _csv_value(value: Any) -> str:
    normalized = _remove_absolute_paths(value)
    if normalized is _DROP or normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, int):
        return str(normalized)
    if isinstance(normalized, float):
        if not math.isfinite(normalized):
            return ""
        if normalized == 0.0:
            return "0"
        return format(normalized, ".17g")
    if isinstance(normalized, Decimal):
        if not normalized.is_finite():
            return ""
        return format(normalized, "f")
    if isinstance(normalized, (date, datetime)):
        return normalized.isoformat()
    if isinstance(normalized, (dict, list)):
        return canonical_json_bytes(to_jsonable(normalized)).decode("utf-8").removesuffix("\n")
    return str(normalized).replace("\r\n", "\n").replace("\r", "\n")


def _is_absolute_path_text(value: str) -> bool:
    if not value:
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _collect_absolute_paths(*values: Any) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Path):
            if value.is_absolute():
                found.add(str(value))
            return
        if isinstance(value, str):
            if _is_absolute_path_text(value):
                found.add(value)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    for item in values:
        visit(item)
    return found


def _canonical_markdown_bytes(report: str, absolute_paths: set[str]) -> bytes:
    text = report.replace("\r\n", "\n").replace("\r", "\n")
    replacements: set[str] = set()
    for path in absolute_paths:
        replacements.add(path)
        replacements.add(path.replace("\\", "/"))
        replacements.add(path.replace("/", "\\"))
    for path in sorted(replacements, key=len, reverse=True):
        if path:
            text = text.replace(path, "<ABSOLUTE_PATH>")
    return (text.rstrip("\n") + "\n").encode("utf-8")
