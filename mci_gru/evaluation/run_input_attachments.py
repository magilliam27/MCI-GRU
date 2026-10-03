"""Attach a window's exact input declarations and observed reads to its saved run.

A window's attachment retains, under ``input_attachments/``:

- each consumed package's manifest, byte-exact, named by its SHA-256;
- the window's frozen ``input_observations`` record (#191 / #206);
- ``record.json``, which binds those files by digest and states which declared
  package file each required input role was expected to read.

The required roles are declared independently of what was observed, so an
input that was enabled but never read, or read but never declared, is
visible rather than silently absent. Raw package files are referenced by their
declared identity, never copied.

``read_run_inputs`` reads only the retained files: a relocated result verifies
without its original data, manifests or Git checkout. Missing or inconsistent
evidence makes the attachment ``incomplete``. Execution status (#144) and
preservation status (#207) are carried exactly as supplied; nothing here
observes them, and an attachment never proves preservation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mci_gru.data.input_manifest import InputManifestError, read_input_manifest
from mci_gru.evaluation.artifacts import canonical_json_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mci_gru.data.input_manifest import ManifestSnapshot
    from mci_gru.data.input_observations import InputObservations

ATTACHMENT_DIR = "input_attachments"
RECORD_NAME = "record.json"
OBSERVATIONS_NAME = "input_observations.json"
RECORD_SCHEMA = "mci_gru.run_input_attachments.v1"
OBSERVATIONS_SCHEMA = "mci_gru.input_observations.v1"
EXECUTION_STATUSES = ("unknown", "incomplete", "failed", "complete")
PRESERVATION_STATUSES = ("unproven", "partial", "complete")


@dataclass(frozen=True)
class RoleBinding:
    """One declared package file that a required input role is expected to read."""

    role: str
    manifest_sha256: str
    path: str


@dataclass(frozen=True)
class AttachmentReference:
    """A retained attachment record and the digest that anchors it."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class RunInputs:
    """What a retained attachment proves, and what it does not."""

    status: str
    packages: tuple[dict[str, Any], ...]
    roles: tuple[dict[str, Any], ...]
    problems: list[str]
    execution: dict[str, Any]
    preservation: dict[str, Any]


def attach_run_inputs(
    window_dir: str | Path,
    *,
    manifests: Sequence[ManifestSnapshot],
    observations: InputObservations,
    required: Sequence[RoleBinding],
    execution: dict[str, Any] | None = None,
    preservation: dict[str, Any] | None = None,
) -> AttachmentReference:
    """Retain one window's declarations and observed reads; never overwrite.

    *execution* and *preservation* are supplied by their owners and recorded
    unchanged; each defaults to its weakest status. Inconsistent evidence is
    still retained, so the reader can report it; only a malformed declaration
    is refused here.
    """
    window = Path(window_dir)
    execution = {"status": "unknown"} if execution is None else dict(execution)
    preservation = {"status": "unproven"} if preservation is None else dict(preservation)
    _validate_status(execution, EXECUTION_STATUSES, "execution")
    _validate_status(preservation, PRESERVATION_STATUSES, "preservation")
    by_digest = {snapshot.sha256: snapshot for snapshot in manifests}
    if len(by_digest) != len(manifests):
        raise ValueError("A manifest is attached more than once")
    for binding in required:
        if not isinstance(binding.role, str) or not binding.role:
            raise ValueError("A required role needs a name")
        if binding.manifest_sha256 not in by_digest:
            raise ValueError(f"Required role {binding.role} names a manifest that is not attached")
    if len(set(required)) != len(required):
        raise ValueError("A required role binding is declared more than once")

    directory = window / ATTACHMENT_DIR
    packages = []
    for snapshot in manifests:
        relative = f"{ATTACHMENT_DIR}/manifests/{snapshot.sha256}.json"
        _write_new(window / relative, snapshot.raw_bytes)
        packages.append(
            {
                "package_id": snapshot.manifest.package_id,
                "package_revision": snapshot.manifest.package_revision,
                "manifest_sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
                "path": relative,
            }
        )
    observed = canonical_json_bytes(observations.to_dict())
    _write_new(directory / OBSERVATIONS_NAME, observed)
    record = {
        "schema": RECORD_SCHEMA,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "input_observations": {
            "path": f"{ATTACHMENT_DIR}/{OBSERVATIONS_NAME}",
            "sha256": hashlib.sha256(observed).hexdigest(),
        },
        "required": [
            {"role": binding.role, "manifest_sha256": binding.manifest_sha256, "path": binding.path}
            for binding in required
        ],
        "execution": execution,
        "preservation": preservation,
    }
    payload = canonical_json_bytes(record)
    path = directory / RECORD_NAME
    _write_new(path, payload)
    return AttachmentReference(path, hashlib.sha256(payload).hexdigest())


def read_run_inputs(reference: AttachmentReference) -> RunInputs:
    """Verify a retained attachment from its own files, wherever it now lives."""
    try:
        payload = reference.path.read_bytes()
    except OSError as exc:
        raise ValueError("Input attachment record missing or unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("Input attachment record digest mismatch")
    try:
        record = json.loads(payload)
        if record["schema"] != RECORD_SCHEMA:
            raise ValueError("unsupported schema")
        _validate_status(record["execution"], EXECUTION_STATUSES, "execution")
        _validate_status(record["preservation"], PRESERVATION_STATUSES, "preservation")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid input attachment record: {exc}") from exc
    window = reference.path.parent.parent
    problems: list[str] = []

    declarations: dict[str, ManifestSnapshot] = {}
    packages = []
    for entry in record["packages"]:
        digest = entry["manifest_sha256"]
        label = f"{entry['package_id']} {entry['package_revision']}"
        package = {**entry, "raw_bytes": None}
        packages.append(package)
        path = window / entry["path"]
        if not path.is_file():
            problems.append(f"manifest snapshot missing: {label}")
            continue
        try:
            snapshot = read_input_manifest(path, expected_sha256=digest)
        except (OSError, InputManifestError) as exc:
            problems.append(f"manifest snapshot altered: {label} ({exc})")
            continue
        package["raw_bytes"] = snapshot.raw_bytes
        declarations[digest] = snapshot

    observations = _read_observations(window, record["input_observations"], problems)
    roles = _link_roles(record["required"], declarations, observations, problems)
    return RunInputs(
        "incomplete" if problems else "complete",
        tuple(packages),
        roles,
        problems,
        record["execution"],
        record["preservation"],
    )


def _read_observations(
    window: Path, entry: dict[str, Any], problems: list[str]
) -> dict[str, Any] | None:
    path = window / entry["path"]
    try:
        content = path.read_bytes()
    except OSError:
        problems.append("input observations missing")
        return None
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        problems.append("input observations differ from the attachment record")
        return None
    observations = json.loads(content)
    if observations.get("schema") != OBSERVATIONS_SCHEMA:
        problems.append("input observations have an unsupported schema")
        return None
    return observations


def _link_roles(
    required: list[dict[str, Any]],
    declarations: dict[str, ManifestSnapshot],
    observations: dict[str, Any] | None,
    problems: list[str],
) -> tuple[dict[str, Any], ...]:
    """Match every consumed read to a declared file, and every declaration to a read."""
    events = (
        {}
        if observations is None
        else {event["observation_id"]: event for event in observations["observations"]}
    )
    uses = [] if observations is None else observations["uses"]
    roles = []
    for binding in required:
        declared = _declared_file(binding, declarations, problems)
        roles.append({**binding, "declared": declared, "observation_ids": []})
    if observations is None:
        return tuple(roles)
    required_roles = {binding["role"] for binding in required}
    for use in uses:
        role = use["role"]
        if role not in required_roles:
            problems.append(f"consumed role {role} is not declared")
            continue
        event = events.get(use["observation_id"])
        identity = {} if event is None else event.get("identity", {})
        matches = [
            entry
            for entry in roles
            if entry["role"] == role
            and entry["declared"] is not None
            and identity.get("sha256") == entry["declared"]["sha256"]
            and identity.get("manifest_sha256", entry["manifest_sha256"])
            == entry["manifest_sha256"]
        ]
        if not matches:
            problems.append(
                f"observed input differs for role {role}: observed SHA-256 "
                f"{identity.get('sha256')}, size {identity.get('size_bytes')}"
            )
            continue
        for entry in matches:
            entry["observation_ids"].append(use["observation_id"])
    for entry in roles:
        if not entry["observation_ids"]:
            problems.append(f"required role {entry['role']} was not consumed from {entry['path']}")
    return tuple(roles)


def _declared_file(
    binding: dict[str, Any], declarations: dict[str, ManifestSnapshot], problems: list[str]
) -> dict[str, Any] | None:
    snapshot = declarations.get(binding["manifest_sha256"])
    if snapshot is None:
        problems.append(f"required role {binding['role']} has no verified manifest")
        return None
    for file in snapshot.manifest.files:
        if file.path == binding["path"]:
            return {"sha256": file.sha256, "size_bytes": file.size_bytes}
    problems.append(f"required role {binding['role']} names a file its manifest lacks")
    return None


def _validate_status(value: Any, allowed: tuple[str, ...], label: str) -> None:
    if not isinstance(value, dict) or value.get("status") not in allowed:
        raise ValueError(f"{label} status must be one of {', '.join(allowed)}")


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
