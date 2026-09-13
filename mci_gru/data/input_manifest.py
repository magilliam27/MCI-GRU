"""Read, verify and publish exact declarations of complete input packages.

Schema version 1 requires package_id, package_revision, files, provenance and
required_extensions. Each file carries path, purpose, sha256 and size_bytes.
Only metadata objects are optional descriptive extensions; unknown core fields,
other schema versions and nonempty required_extensions are rejected. Provenance
records source, acquisition_mode, acquired_at, producing_command and an argv list
in producing_arguments, using null plus explanatory unknowns when unavailable.
Callers must supply credential-free provenance and descriptions, not locators.

Publication sorts files by path and uses canonical_json_bytes: sorted object keys,
compact strict UTF-8 JSON and one final LF. Readers retain original bytes even
when their formatting is not canonical. The exact-byte digest is returned outside
the document. New declarations require a new package revision and output path;
the caller supplies the revision rather than this module maintaining a registry.

The supplied inventory is authoritative; this module never discovers unrelated
files, repairs sources, retrieves data or proves remote preservation. Verification
describes the observed files, not a lock against later external modification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mci_gru.evaluation.artifacts import canonical_json_bytes
from mci_gru.utils.hashing import sha256_file

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


class InputManifestError(ValueError):
    """An input declaration cannot be accepted."""


class ManifestDigestMismatchError(InputManifestError):
    """The observed manifest byte stream differs from the requested identity."""

    def __init__(self, path: Path, expected_sha256: str, observed_sha256: str) -> None:
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256
        super().__init__(
            f"Manifest SHA-256 mismatch at {path}: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )


@dataclass(frozen=True)
class InputFileRecord:
    """Expected identity of one package-relative source file."""

    path: str
    purpose: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputFileSpec:
    """One file in the caller's complete source inventory."""

    path: str
    purpose: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputManifest:
    """A versioned package declaration, separate from its storage location."""

    schema_version: int
    package_id: str
    package_revision: str
    files: tuple[InputFileRecord, ...]
    provenance: dict[str, Any]
    required_extensions: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestSnapshot:
    """A parsed declaration bound to its exact retained byte stream."""

    manifest: InputManifest
    raw_bytes: bytes
    sha256: str

    @property
    def size_bytes(self) -> int:
        """Return the length of the retained byte stream."""
        return len(self.raw_bytes)


@dataclass(frozen=True)
class VerifiedInputFile:
    """A declaration bound to the exact source path and observed file identity."""

    record: InputFileRecord
    resolved_path: Path
    sha256: str
    size_bytes: int


class InputFileMismatchError(InputManifestError):
    """An observed source file differs from its declared identity."""

    def __init__(self, observation: VerifiedInputFile) -> None:
        self.expected_sha256 = observation.record.sha256
        self.observed_sha256 = observation.sha256
        self.expected_size_bytes = observation.record.size_bytes
        self.observed_size_bytes = observation.size_bytes
        super().__init__(
            f"File identity mismatch at {observation.resolved_path}: "
            f"expected SHA-256 {self.expected_sha256}, size {self.expected_size_bytes}; "
            f"observed SHA-256 {self.observed_sha256}, size {self.observed_size_bytes}"
        )


@dataclass(frozen=True)
class PackageVerification:
    """A manifest snapshot together with its verified source bindings."""

    snapshot: ManifestSnapshot
    files: tuple[VerifiedInputFile, ...]


def _validate_relative_path(path: str) -> None:
    """Accept portable, unambiguous package-relative file names."""
    if not isinstance(path, str) or not path:
        raise InputManifestError(f"Unsafe package path: {path!r}")
    for part in path.split("/"):
        if (
            not part
            or part in {".", ".."}
            or part.endswith((".", " "))
            or re.search(r'[<>:"\\|?*\x00-\x1f]', part)
            or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
        ):
            raise InputManifestError(f"Unsafe package path: {path!r}")


def _validate_inventory(paths: Collection[str]) -> None:
    for path in paths:
        _validate_relative_path(path)
    folded = {path.casefold() for path in paths}
    if len(folded) != len(paths):
        raise InputManifestError("Duplicate destinations in package inventory")
    for path in folded:
        components = path.split("/")
        if any("/".join(components[:index]) in folded for index in range(1, len(components))):
            raise InputManifestError("Colliding file and directory paths in package inventory")


def _source_path(package_root: Path, relative: str) -> Path:
    _validate_relative_path(relative)
    path = (package_root / relative).resolve()
    if not path.is_relative_to(package_root.resolve()):
        raise InputManifestError(f"Source path resolves outside package root: {path}")
    if not path.is_file():
        raise InputManifestError(f"Missing source file: {path}")
    return path


def _nonempty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InputManifestError(f"{field_name} must be a nonempty string")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputManifestError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _validate_json_types(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputManifestError("JSON object keys must be strings")
            _validate_json_types(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_types(item)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise InputManifestError(f"Unsupported JSON value type: {type(value).__name__}")


def _validate_payload(payload: dict[str, Any]) -> None:
    # Reject non-finite numbers and invalid Unicode before any canonicalization.
    try:
        json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise InputManifestError(f"Invalid JSON data: {error}") from error
    _validate_json_types(payload)
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise InputManifestError("Unsupported schema_version; supported version is 1")
    if payload["required_extensions"] != []:
        raise InputManifestError("Unsupported required_extensions; version 1 supports none")
    _nonempty_string(payload["package_id"], "package_id")
    _nonempty_string(payload["package_revision"], "package_revision")
    if not isinstance(payload["files"], list) or not payload["files"]:
        raise InputManifestError("files must be a nonempty list")
    provenance = payload["provenance"]
    required_provenance = {
        "source",
        "acquisition_mode",
        "acquired_at",
        "producing_command",
        "producing_arguments",
        "unknowns",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise InputManifestError(
            "provenance must explicitly record known and unknown acquisition facts"
        )
    for name in ("source", "acquisition_mode", "acquired_at", "producing_command"):
        if provenance[name] is not None:
            _nonempty_string(provenance[name], f"provenance.{name}")
    arguments = provenance["producing_arguments"]
    if arguments is not None and (
        not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments)
    ):
        raise InputManifestError("provenance.producing_arguments must be an argument list or null")
    unknowns = provenance["unknowns"]
    if not isinstance(unknowns, list):
        raise InputManifestError("provenance.unknowns must be a list")
    for unknown in unknowns:
        _nonempty_string(unknown, "provenance.unknowns entry")
    if any(value is None for value in provenance.values()) and not unknowns:
        raise InputManifestError("provenance.unknowns must explain unavailable acquisition facts")
    for item in [payload, *payload["files"]]:
        if not isinstance(item, dict):
            raise InputManifestError("file records must be objects")
        if not isinstance(item.get("metadata", {}), dict):
            raise InputManifestError("metadata must be an object")
    for record in payload["files"]:
        _nonempty_string(record["purpose"], "purpose")
        digest = record["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise InputManifestError("sha256 must contain 64 lowercase hexadecimal characters")
        size = record["size_bytes"]
        if type(size) is not int or size < 0:
            raise InputManifestError("size_bytes must be a nonnegative integer")


def read_input_manifest(path: Path, *, expected_sha256: str | None = None) -> ManifestSnapshot:
    """Read an input manifest while retaining its original bytes and digest."""
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ManifestDigestMismatchError(path, expected_sha256, sha256)
    try:
        payload = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
        _validate_payload(payload)
        manifest = InputManifest(
            **{
                **payload,
                "files": tuple(InputFileRecord(**record) for record in payload["files"]),
                "required_extensions": tuple(payload["required_extensions"]),
            }
        )
        _validate_inventory([record.path for record in manifest.files])
    except (ValueError, TypeError, KeyError) as error:
        raise InputManifestError(f"Invalid manifest at {path}: {error}") from error
    return ManifestSnapshot(manifest=manifest, raw_bytes=raw_bytes, sha256=sha256)


def validate_input_package(
    snapshot: ManifestSnapshot, package_root: Path, *, expected_paths: Collection[str]
) -> PackageVerification:
    """Verify a declared package at an explicit root without modifying sources."""
    declared = [record.path for record in snapshot.manifest.files]
    _validate_inventory(declared)
    _validate_inventory(expected_paths)
    if set(declared) != set(expected_paths):
        raise InputManifestError(
            f"Package inventory mismatch: missing declarations "
            f"{sorted(set(expected_paths) - set(declared))}; unexpected declarations "
            f"{sorted(set(declared) - set(expected_paths))}"
        )
    observations = []
    for record in snapshot.manifest.files:
        path = _source_path(package_root, record.path)
        observation = VerifiedInputFile(record, path, sha256_file(path), path.stat().st_size)
        if observation.sha256 != record.sha256 or observation.size_bytes != record.size_bytes:
            raise InputFileMismatchError(observation)
        observations.append(observation)
    return PackageVerification(snapshot, tuple(observations))


def write_input_manifest(
    output_path: Path,
    package_root: Path,
    *,
    package_id: str,
    package_revision: str,
    files: Collection[InputFileSpec],
    provenance: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    expected_files: Collection[InputFileRecord] | None = None,
) -> ManifestSnapshot:
    """Publish a new manifest, leaving sources and existing outputs unchanged.

    ``files`` is the caller's explicit complete inventory. For a historical
    backfill, ``expected_files`` must independently supply the entire retained
    hash/size inventory. Publication requires an existing output parent outside
    package_root on a filesystem supporting atomic hard-link creation. Storage
    failures stop without a partial final manifest; no overwrite fallback exists.
    """
    if output_path.resolve().is_relative_to(package_root.resolve()):
        raise InputManifestError("Manifest output must be outside the source root")
    _validate_inventory([spec.path for spec in files])
    try:
        records = []
        for spec in sorted(files, key=lambda spec: spec.path):
            source_path = _source_path(package_root, spec.path)
            records.append(
                InputFileRecord(
                    spec.path,
                    spec.purpose,
                    sha256_file(source_path),
                    source_path.stat().st_size,
                    spec.metadata,
                )
            )
    except OSError as error:
        raise InputManifestError(f"Cannot read source file: {error}") from error
    manifest = InputManifest(
        1,
        package_id,
        package_revision,
        tuple(records),
        provenance,
        (),
        {} if metadata is None else metadata,
    )
    payload = asdict(manifest)
    payload["files"] = list(payload["files"])
    payload["required_extensions"] = []
    _validate_payload(payload)
    if expected_files is not None:
        _validate_inventory([record.path for record in expected_files])
        expected_by_path = {record.path: record for record in expected_files}
        if set(expected_by_path) != {record.path for record in records}:
            raise InputManifestError(
                "Historical inventory differs from the complete source specification"
            )
        for record in records:
            expected = expected_by_path[record.path]
            if record.sha256 != expected.sha256 or record.size_bytes != expected.size_bytes:
                raise InputFileMismatchError(
                    VerifiedInputFile(
                        expected,
                        (package_root / record.path).resolve(),
                        record.sha256,
                        record.size_bytes,
                    )
                )
    raw_bytes = canonical_json_bytes(payload)
    snapshot = ManifestSnapshot(manifest, raw_bytes, hashlib.sha256(raw_bytes).hexdigest())
    validate_input_package(snapshot, package_root, expected_paths=[spec.path for spec in files])
    staged_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=".input-manifest-", delete=False
        ) as handle:
            staged_path = Path(handle.name)
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-filesystem hard-link publication is atomic and never replaces a destination.
        os.link(staged_path, output_path)
    except FileExistsError as error:
        raise InputManifestError(f"Manifest output already exists: {output_path}") from error
    except OSError as error:
        raise InputManifestError(f"Cannot publish manifest at {output_path}: {error}") from error
    finally:
        if staged_path is not None:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                # Publication already has an unambiguous outcome. Cleanup must not
                # turn a completed publish into failure or obscure the original error.
                logger.warning("Could not remove manifest staging file %s", staged_path)
    return snapshot
