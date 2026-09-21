"""Retain exact auxiliary observations in versioned input-manifest packages."""

from __future__ import annotations

import base64
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd

from mci_gru.data.input_manifest import (
    InputFileSpec,
    InputManifestError,
    ManifestSnapshot,
    read_input_manifest,
    write_input_manifest,
)
from mci_gru.data.input_observations import (
    InputObservationContext,
    InputObservationError,
    ObservationId,
)
from mci_gru.evaluation.artifacts import canonical_json_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


class InputSnapshotError(InputObservationError):
    """A snapshot failure with safe structured facts for preparation reporting."""

    def __init__(
        self,
        *,
        role: str,
        source: str,
        request: dict[str, Any],
        stage: str,
        reason: str,
        **identity: Any,
    ) -> None:
        self.facts = {
            "role": role,
            "source": source,
            "request": request,
            "stage": stage,
            "reason": reason,
            "reason_code": reason,
            "required": True,
            **identity,
        }
        super().__init__(f"Required auxiliary input {role}: {stage}/{reason}")


@dataclass(frozen=True)
class SnapshotReference:
    """The selected snapshot location and expected manifest identity."""

    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class Snapshot:
    """An original observation and its retained acquisition facts."""

    data: bytes | pd.Series | pd.DataFrame
    provenance: dict[str, Any]
    manifest: ManifestSnapshot


@dataclass(frozen=True)
class ObservedInput:
    """A raw observation whose use is linked after the loader accepts it."""

    data: bytes | pd.Series | pd.DataFrame
    observation_id: ObservationId
    record: dict[str, Any]


class InputSnapshots:
    """Select capture or offline replay and contribute to one observation context."""

    def __init__(
        self,
        *,
        mode: str = "source",
        directory: Path | None = None,
        references: dict[str, list[SnapshotReference]] | None = None,
        input_observations: InputObservationContext | None = None,
    ) -> None:
        if mode not in {"source", "capture", "replay"}:
            raise ValueError("Snapshot mode must be source, capture, or replay")
        if mode == "capture" and directory is None:
            raise ValueError("Snapshot capture requires an output directory")
        self.mode = mode
        self.directory = directory
        self.references = {key: list(values) for key, values in (references or {}).items()}
        self.input_observations = (
            input_observations if input_observations is not None else InputObservationContext()
        )
        self._positions: dict[str, int] = {}

    def observe(
        self,
        *,
        role: str,
        source: str,
        request: dict[str, Any],
        acquire: Callable[[], bytes | pd.Series | pd.DataFrame],
    ) -> ObservedInput:
        """Observe original provider output before application transformations."""
        key = hashlib.sha256(
            canonical_json_bytes({"role": role, "source": source, "request": request})
        ).hexdigest()
        observed_at = datetime.now(timezone.utc).isoformat()
        stage = "read"
        known_identity = {}
        try:
            if self.mode == "replay":
                position = self._positions.get(key, 0)
                candidates = self.references.get(key, [])
                if position >= len(candidates):
                    raise InputSnapshotError(
                        role=role,
                        source=source,
                        request=request,
                        stage="integrity",
                        reason="missing_snapshot",
                    )
                reference = candidates[position]
                snapshot = load_snapshot(reference, role=role, source=source, request=request)
                self._positions[key] = position + 1
                data = snapshot.data
                acquired_at = snapshot.provenance["acquired_at"]
            else:
                data = acquire()
                acquired_at = datetime.now(timezone.utc).isoformat()
                # Freeze the observed SDK value before any publication or caller
                # code can mutate its buffer. The loader consumes this exact copy.
                stage = "validate"
                observed_codec, observed_content = _encode(data)
                known_identity = {
                    "sha256": hashlib.sha256(observed_content).hexdigest(),
                    "size_bytes": len(observed_content),
                }
                data = _decode(observed_codec, observed_content)
                reference = None
                if self.mode == "capture":
                    stage = "retain"
                    revision = uuid4().hex
                    reference = save_snapshot(
                        self.directory / revision,
                        data,
                        package_id=key,
                        package_revision=revision,
                        role=role,
                        source=source,
                        request=request,
                        acquired_at=acquired_at,
                    )
                    self.references.setdefault(key, []).append(reference)
            stage = "validate"
            codec, content = _encode(data)
            identity = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
            if reference is not None:
                identity.update(
                    manifest_path=str(reference.manifest_path),
                    manifest_sha256=reference.manifest_sha256,
                    snapshot_key=key,
                )
            record = {
                "source_kind": source,
                "role": role,
                "request": request,
                "stage": "acquire",
                "outcome": "success",
                "observed_at": observed_at,
                "acquired_at": acquired_at,
                "mode": self.mode,
                "codec": codec,
                "identity": identity,
            }
            observation_id = self.input_observations.record_observation(record)
            return ObservedInput(data, observation_id, record)
        except Exception as error:
            failure = (
                error
                if isinstance(error, InputSnapshotError)
                else InputSnapshotError(
                    role=role,
                    source=source,
                    request=request,
                    stage=stage,
                    reason={
                        "read": "acquisition_failed",
                        "validate": "unsupported_observation",
                        "retain": "snapshot_publication_failed",
                    }[stage],
                    error_code=type(error).__name__,
                    **({"identity": known_identity} if known_identity else {}),
                )
            )
            if isinstance(error, InputObservationError) and not isinstance(
                error, InputSnapshotError
            ):
                facts = {
                    **failure.facts,
                    "stage": "integrity",
                    "reason": "observation_integrity_failed",
                    "reason_code": "observation_integrity_failed",
                }
                failure = error
                failure.facts = facts
            self.input_observations.record_observation(
                {
                    "source_kind": source,
                    "role": role,
                    "request": request,
                    "stage": failure.facts["stage"],
                    "outcome": "error",
                    "observed_at": observed_at,
                    "mode": self.mode,
                    "failure": failure.facts,
                }
            )
            raise failure from None

    def use(self, observation: ObservedInput, role: str) -> None:
        """Link an accepted original observation to its actual downstream role."""
        self.input_observations.record_use(observation.observation_id, role)

    @contextmanager
    def accepted(self, *observations: ObservedInput, role: str | None = None) -> Iterator[None]:
        """Associate transformation success or rejection with its original inputs."""
        try:
            yield
        except Exception as error:
            for observation in observations:
                record = observation.record
                failure = InputSnapshotError(
                    role=record["role"],
                    source=record["source_kind"],
                    request=record["request"],
                    stage="parse",
                    reason="invalid_provider_observation",
                    error_code=type(error).__name__,
                    identity=record["identity"],
                    source_observation_id=observation.observation_id.index,
                )
                self.input_observations.record_observation(
                    {
                        "source_kind": record["source_kind"],
                        "role": record["role"],
                        "request": record["request"],
                        "identity": record["identity"],
                        "stage": "parse",
                        "outcome": "error",
                        "error_code": type(error).__name__,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "source_observation_id": observation.observation_id.index,
                        "failure": failure.facts,
                    }
                )
            if len(observations) > 1:
                failure = InputSnapshotError(
                    role=role or "auxiliary_inputs",
                    source=observations[0].record["source_kind"],
                    request={"inputs": [item.record["request"] for item in observations]},
                    stage="parse",
                    reason="invalid_provider_observation",
                    error_code=type(error).__name__,
                    identity={"inputs": [item.record["identity"] for item in observations]},
                    source_observation_ids=[item.observation_id.index for item in observations],
                )
            raise failure from None
        else:
            for observation in observations:
                self.use(observation, observation.record["role"])

    def read_file(
        self,
        path: Path | None,
        *,
        role: str,
        configured_path: str,
        parse: Callable[[bytes], Any],
        parser: dict[str, Any],
    ) -> Any:
        """Retain the existing native reader's buffer, or parse its offline copy.

        Capture contributes a publication event associated by original content,
        role and parser request. It neither repeats acquisition nor claims the
        native event ID, which the shared reader creates after its callback.
        """
        request = {"configured_path": configured_path, "parser": parser}
        if self.mode == "replay":

            def unavailable() -> bytes:
                raise AssertionError("Offline file replay attempted acquisition")

            observation = self.observe(
                role=role, source="file", request=request, acquire=unavailable
            )
            try:
                result = parse(observation.data)
            except Exception as error:
                self.input_observations.record_observation(
                    {
                        "source_kind": "file",
                        "role": role,
                        "configured_path": configured_path,
                        "stage": "integrity"
                        if isinstance(error, InputObservationError)
                        else "parse",
                        "outcome": "error",
                        "error_code": type(error).__name__,
                        "request": request,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "mode": "replay",
                        "source_observation_id": observation.observation_id.index,
                        "identity": {
                            "sha256": hashlib.sha256(observation.data).hexdigest(),
                            "size_bytes": len(observation.data),
                        },
                    }
                )
                raise
            self.use(observation, role)
            return result
        publication = None

        def parse_original(content: bytes) -> Any:
            nonlocal publication
            if self.mode == "capture":
                key = hashlib.sha256(
                    canonical_json_bytes({"role": role, "source": "file", "request": request})
                ).hexdigest()
                revision = uuid4().hex
                acquired_at = datetime.now(timezone.utc).isoformat()
                try:
                    reference = save_snapshot(
                        self.directory / revision,
                        content,
                        package_id=key,
                        package_revision=revision,
                        role=role,
                        source="file",
                        request=request,
                        acquired_at=acquired_at,
                    )
                except Exception as error:
                    raise InputSnapshotError(
                        role=role,
                        source="file",
                        request=request,
                        stage="retain",
                        reason="snapshot_publication_failed",
                        error_code=type(error).__name__,
                    ) from None
                self.references.setdefault(key, []).append(reference)
                publication = self.input_observations.record_observation(
                    {
                        "source_kind": "snapshot_publication",
                        "stage": "retain",
                        "outcome": "success",
                        "role": role,
                        "request": request,
                        "observed_at": acquired_at,
                        "acquired_at": acquired_at,
                        "mode": "capture",
                        "codec": "bytes",
                        "relationship": {
                            "kind": "retains_parser_input",
                            "configured_path": configured_path,
                        },
                        "identity": {
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "size_bytes": len(content),
                            "manifest_path": str(reference.manifest_path),
                            "manifest_sha256": reference.manifest_sha256,
                            "snapshot_key": key,
                        },
                    }
                )
            return parse(content)

        result = self.input_observations.read_file(
            path, role=role, configured_path=configured_path, parse=parse_original, parser=parser
        )
        if publication is not None:
            self.input_observations.record_use(publication, role, configured_path)
        return result


def _scalar(value: Any) -> Any:
    if value is pd.NA:
        return {"type": "NA"}
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        return {"type": "float", "hex": value.hex()}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_scalar(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_scalar(item) for item in value]}
    if isinstance(value, dict):
        return {
            "type": "mapping",
            "items": [[_scalar(key), _scalar(item)] for key, item in value.items()],
        }
    raise ValueError("Unsupported SDK scalar type")


def _restore_scalar(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value["type"] == "NA":
        return pd.NA
    if value["type"] == "float":
        return float.fromhex(value["hex"])
    if value["type"] == "tuple":
        return tuple(_restore_scalar(item) for item in value["items"])
    if value["type"] == "list":
        return [_restore_scalar(item) for item in value["items"]]
    if value["type"] == "mapping":
        return {_restore_scalar(key): _restore_scalar(item) for key, item in value["items"]}
    raise ValueError("Unsupported SDK scalar encoding")


def _array(values: Any) -> dict[str, Any]:
    if isinstance(values.dtype, np.dtype) and values.dtype.kind in "biufcmM":
        array = np.asarray(values)
        return {
            "dtype": array.dtype.str,
            "bytes": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if str(values.dtype) not in {"object", "Int64", "Float64", "boolean", "string"}:
        raise ValueError("Unsupported SDK array type")
    return {"dtype": str(values.dtype), "values": [_scalar(value) for value in values.tolist()]}


def _restore_array(payload: dict[str, Any]) -> Any:
    if "bytes" in payload:
        dtype = np.dtype(payload["dtype"])
        if dtype.kind not in "biufcmM":
            raise ValueError("Unsupported SDK binary dtype")
        return np.frombuffer(base64.b64decode(payload["bytes"], validate=True), dtype=dtype).copy()
    if payload["dtype"] not in {"object", "Int64", "Float64", "boolean", "string"}:
        raise ValueError("Unsupported SDK array type")
    return pd.array([_restore_scalar(value) for value in payload["values"]], dtype=payload["dtype"])


def _axis(index: pd.Index) -> dict[str, Any]:
    if isinstance(index, pd.MultiIndex):
        return {
            "type": "multi",
            "levels": [_axis(level) for level in index.levels],
            "codes": [code.tolist() for code in index.codes],
            "names": [_scalar(name) for name in index.names],
        }
    if isinstance(index, pd.DatetimeIndex):
        return {
            "type": "datetime",
            "array": _array(index.asi8),
            "dtype": str(index.dtype),
            "name": _scalar(index.name),
            "freq": index.freqstr,
        }
    if isinstance(index, pd.RangeIndex):
        return {
            "type": "range",
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
            "name": _scalar(index.name),
        }
    return {"type": "index", "array": _array(index), "name": _scalar(index.name)}


def _restore_axis(payload: dict[str, Any]) -> pd.Index:
    kind = payload["type"]
    if kind == "multi":
        return pd.MultiIndex(
            levels=[_restore_axis(level) for level in payload["levels"]],
            codes=payload["codes"],
            names=[_restore_scalar(name) for name in payload["names"]],
        )
    name = _restore_scalar(payload["name"])
    if kind == "datetime":
        return pd.DatetimeIndex(
            _restore_array(payload["array"]),
            dtype=payload["dtype"],
            name=name,
            freq=payload["freq"],
        )
    if kind == "range":
        return pd.RangeIndex(payload["start"], payload["stop"], payload["step"], name=name)
    if kind == "index":
        return pd.Index(_restore_array(payload["array"]), name=name, tupleize_cols=False)
    raise ValueError("Unsupported SDK index encoding")


def _encode(data: bytes | pd.Series | pd.DataFrame) -> tuple[str, bytes]:
    if isinstance(data, bytes):
        return "bytes", data
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Unsupported SDK observation type")
    payload = {"index": _axis(data.index), "attrs": _scalar(data.attrs)}
    if isinstance(data, pd.Series):
        payload.update(kind="series", name=_scalar(data.name), array=_array(data))
    else:
        payload.update(
            kind="frame",
            columns=_axis(data.columns),
            arrays=[_array(data.iloc[:, i]) for i in range(data.shape[1])],
        )
    return "pandas.v1", canonical_json_bytes(payload)


def _decode(codec: str, content: bytes) -> bytes | pd.Series | pd.DataFrame:
    if codec == "bytes":
        return content
    if codec != "pandas.v1":
        raise ValueError("Unsupported observation codec")
    payload = json.loads(content)
    index = _restore_axis(payload["index"])
    if payload["kind"] == "series":
        data = pd.Series(
            _restore_array(payload["array"]), index=index, name=_restore_scalar(payload["name"])
        )
    elif payload["kind"] == "frame":
        data = pd.DataFrame(
            {i: _restore_array(array) for i, array in enumerate(payload["arrays"])}, index=index
        )
        data.columns = _restore_axis(payload["columns"])
    else:
        raise ValueError("Unsupported SDK observation encoding")
    data.attrs = _restore_scalar(payload["attrs"])
    return data


def save_snapshot(
    destination: Path,
    data: bytes | pd.Series | pd.DataFrame,
    *,
    package_id: str,
    package_revision: str,
    role: str,
    source: str,
    request: dict[str, Any],
    acquired_at: str,
) -> SnapshotReference:
    """Publish a new exact-byte snapshot without replacing an existing package."""
    codec, content = _encode(data)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / "files"
    root.mkdir()
    (root / "observations.bin").write_bytes(content)
    descriptor = {
        "schema_version": 1,
        "codec": codec,
        "provenance": {
            "role": role,
            "source": source,
            "request": request,
            "acquired_at": acquired_at,
        },
    }
    (root / "snapshot.json").write_bytes(canonical_json_bytes(descriptor))
    manifest_path = destination / "manifest.json"
    manifest = write_input_manifest(
        manifest_path,
        root,
        package_id=package_id,
        package_revision=package_revision,
        files=[
            InputFileSpec("observations.bin", "original auxiliary observation"),
            InputFileSpec("snapshot.json", "snapshot decoding and acquisition contract"),
        ],
        provenance={
            "source": source,
            "acquisition_mode": "exact_observation_snapshot",
            "acquired_at": acquired_at,
            "producing_command": None,
            "producing_arguments": None,
            "unknowns": ["Caller process command and arguments are not captured"],
        },
    )
    return SnapshotReference(manifest_path, manifest.sha256)


def load_snapshot(
    reference: SnapshotReference,
    *,
    role: str,
    source: str,
    request: dict[str, Any],
) -> Snapshot:
    """Reload the explicitly selected retained observation."""
    try:
        manifest = read_input_manifest(
            reference.manifest_path, expected_sha256=reference.manifest_sha256
        )
        root = (reference.manifest_path.parent / "files").resolve()
        if {record.path for record in manifest.manifest.files} != {
            "observations.bin",
            "snapshot.json",
        }:
            raise InputManifestError("Snapshot inventory mismatch")
        contents = {}
        for record in manifest.manifest.files:
            path = (root / record.path).resolve()
            if not path.is_relative_to(root):
                raise InputManifestError("Snapshot member escapes its package")
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != record.sha256 or len(content) != record.size_bytes:
                raise InputSnapshotError(
                    role=role,
                    source=source,
                    request=request,
                    stage="integrity",
                    reason="member_identity_mismatch",
                    expected_sha256=record.sha256,
                    observed_sha256=digest,
                    expected_size_bytes=record.size_bytes,
                    observed_size_bytes=len(content),
                )
            contents[record.path] = content
        descriptor = json.loads(contents["snapshot.json"])
        if (
            set(descriptor) != {"schema_version", "codec", "provenance"}
            or type(descriptor["schema_version"]) is not int
            or descriptor["schema_version"] != 1
        ):
            raise ValueError("Unsupported snapshot descriptor")
        if any(
            descriptor["provenance"][key] != value
            for key, value in {
                "role": role,
                "source": source,
                "request": request,
            }.items()
        ):
            raise InputSnapshotError(
                role=role,
                source=source,
                request=request,
                stage="integrity",
                reason="selection_mismatch",
            )
        return Snapshot(
            _decode(descriptor["codec"], contents["observations.bin"]),
            descriptor["provenance"],
            manifest,
        )
    except (OSError, InputManifestError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, InputSnapshotError):
            raise
        identity = {
            name: getattr(error, name)
            for name in ("expected_sha256", "observed_sha256")
            if hasattr(error, name)
        }
        raise InputSnapshotError(
            role=role,
            source=source,
            request=request,
            stage="integrity",
            reason=type(error).__name__,
            **identity,
        ) from None
