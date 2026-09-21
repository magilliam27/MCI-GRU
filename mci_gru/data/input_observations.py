"""Per-preparation observations of the content actually supplied to parsers.

This carrier records facts; it does not select sources, retain packages, or
decide whether a source failure permits training to continue.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
from pandas.io.common import infer_compression

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

Parsed = TypeVar("Parsed")


class InputObservationError(ValueError):
    """Observation integrity failed; ordinary source fallbacks must not catch it."""


@dataclass(frozen=True)
class ObservationId:
    """A context-local event reference, not a content or path identity."""

    context: object
    index: int


@dataclass(frozen=True)
class InputUse:
    observation_id: int
    role: str
    configured_path: str | None


@dataclass(frozen=True)
class InputObservations:
    """Immutable serialized observations and their downstream uses."""

    records: tuple[str, ...]
    uses: tuple[InputUse, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-ready value, without reading any source."""
        return {
            "schema": "mci_gru.input_observations.v1",
            "observations": [
                dict(json.loads(record), observation_id=i) for i, record in enumerate(self.records)
            ],
            "uses": [
                {
                    "observation_id": use.observation_id,
                    "role": use.role,
                    "configured_path": use.configured_path,
                }
                for use in self.uses
            ],
        }

    def data_inputs(self) -> dict[str, dict[str, Any]]:
        """First consumed identity per role, with links to every consumed read."""
        inputs: dict[str, dict[str, Any]] = {}
        for use in self.uses:
            record = json.loads(self.records[use.observation_id])
            if use.role not in inputs:
                inputs[use.role] = {
                    **record.get("identity", {}),
                    "configured_path": use.configured_path,
                    "observation_ids": [],
                }
            inputs[use.role]["observation_ids"].append(use.observation_id)
        return inputs


class InputObservationContext:
    """Append observations and consumption links, then seal one window."""

    def __init__(self) -> None:
        self._identity = object()
        self._records: list[str] = []
        self._uses: list[InputUse] = []
        self._frozen: InputObservations | None = None

    def record_observation(self, record: Mapping[str, Any]) -> ObservationId:
        """Copy JSON facts at contribution time; nested caller values are not retained."""
        if self._frozen is not None:
            raise InputObservationError("Input observation context is sealed")
        try:
            value = dict(record)
            if value.get("outcome") not in ("success", "empty", "error") or any(
                not isinstance(value.get(key), str) or not value[key]
                for key in ("source_kind", "stage", "observed_at")
            ):
                raise InputObservationError("Invalid input observation fields")
            encoded = json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise InputObservationError("Invalid input observation record") from exc
        reference = ObservationId(self._identity, len(self._records))
        self._records.append(encoded)
        return reference

    def record_use(
        self, observation_id: ObservationId, role: str, configured_path: str | None = None
    ) -> None:
        """Link a successful observation to a role without overwriting earlier uses."""
        if self._frozen is not None:
            raise InputObservationError("Input observation context is sealed")
        if observation_id.context is not self._identity or not 0 <= observation_id.index < len(
            self._records
        ):
            raise InputObservationError("Unknown input observation")
        record = json.loads(self._records[observation_id.index])
        if record["outcome"] != "success":
            raise InputObservationError("Only successful observations can supply an input")
        self._uses.append(InputUse(observation_id.index, role, configured_path))

    def freeze(self) -> InputObservations:
        """Idempotently seal this window; later contributions are rejected."""
        if self._frozen is None:
            self._frozen = InputObservations(tuple(self._records), tuple(self._uses))
        return self._frozen

    def read_csv(self, path: str | Path, *, role: str, configured_path: str) -> pd.DataFrame:
        """Parse the same immutable bytes whose identity is recorded."""
        compression = infer_compression(str(path), "infer")
        return self.read_file(
            path,
            role=role,
            configured_path=configured_path,
            parse=lambda content: pd.read_csv(BytesIO(content), compression=compression),
            parser={
                "name": "pandas.read_csv",
                "options": {"compression": compression},
            },
        )

    def read_file(
        self,
        path: str | Path,
        *,
        role: str,
        configured_path: str,
        parse: Callable[[bytes], Parsed],
        parser: Mapping[str, Any],
    ) -> Parsed:
        """Bind an application parser's input to one observed byte identity."""
        if self._frozen is not None:
            raise InputObservationError("Input observation context is sealed")
        resolved = Path(path).absolute()
        record = {
            "source_kind": "file",
            "role": role,
            "configured_path": configured_path,
            "parser": dict(parser),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "identity": {"resolved_path": str(resolved)},
        }
        try:
            with resolved.open("rb") as handle:
                content = handle.read()
                stat = os.fstat(handle.fileno())
        except Exception as exc:
            self._record_failure(record, exc, "read")
            raise
        record.update(
            stage="parse",
            outcome="success",
            identity={
                "resolved_path": str(resolved),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            },
        )
        try:
            result = parse(content)
        except Exception as exc:
            self._record_failure(record, exc, "parse")
            raise
        reference = self.record_observation(record)
        self.record_use(reference, role, configured_path)
        return result

    def _record_failure(self, record: dict[str, Any], exc: Exception, stage: str) -> None:
        self.record_observation(
            {
                **record,
                "stage": "integrity" if isinstance(exc, InputObservationError) else stage,
                "outcome": "error",
                "error_code": type(exc).__name__,
            }
        )
