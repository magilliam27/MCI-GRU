from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from cockpit._compat import StrEnum
from cockpit.models import WorkstreamStatus

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

DECISION_REGISTRY_PATH = "docs/agents/cockpit/workstream-decisions.json"
FORMAT_VERSION = 1


class SurfaceDisposition(StrEnum):
    CANONICAL = "canonical"
    PARKED = "parked"
    ARCHIVE = "archive"
    STALE = "stale"


@dataclass(frozen=True)
class WorkstreamDecision:
    status: WorkstreamStatus
    canonical_surface: str
    reason: str
    next_action: str
    last_reviewed: date


@dataclass(frozen=True)
class SurfaceDecision:
    workstreams: tuple[str, ...]
    disposition: SurfaceDisposition
    reason: str
    next_action: str
    last_reviewed: date


@dataclass(frozen=True)
class DecisionRegistry:
    workstreams: dict[str, WorkstreamDecision] = field(default_factory=dict)
    surfaces: dict[str, SurfaceDecision] = field(default_factory=dict)

    def is_reviewed(self, workstream: str, branch: str) -> bool:
        surface = self.surfaces.get(branch)
        return surface is not None and workstream in surface.workstreams


def read_registry_workstream_names(repo_root: Path) -> set[str]:
    path = repo_root / DECISION_REGISTRY_PATH
    if not path.exists():
        return set()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(raw, dict):
        return set()
    workstreams = raw.get("workstreams")
    if not isinstance(workstreams, dict):
        return set()
    return set(workstreams)


def load_decision_registry(
    repo_root: Path,
    *,
    known_workstreams: Collection[str],
) -> DecisionRegistry:
    path = repo_root / DECISION_REGISTRY_PATH
    if not path.exists():
        return DecisionRegistry()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {DECISION_REGISTRY_PATH}: {exc.msg}") from exc

    root = _object(raw, "decision registry")
    _keys(root, {"format_version", "workstreams", "surfaces"}, "decision registry")
    version = root.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported cockpit decision registry format_version: {version}")

    known = set(known_workstreams)
    workstreams = _parse_workstreams(root.get("workstreams"), known)
    surfaces = _parse_surfaces(root.get("surfaces"), known)
    return DecisionRegistry(workstreams=workstreams, surfaces=surfaces)


def _parse_workstreams(
    raw: object,
    known_workstreams: set[str],
) -> dict[str, WorkstreamDecision]:
    entries = _object(raw, "workstreams")
    parsed: dict[str, WorkstreamDecision] = {}
    for name, value in entries.items():
        if name not in known_workstreams:
            raise ValueError(f"Unknown workstream in cockpit decision registry: {name}")
        item = _object(value, f"workstreams.{name}")
        _keys(
            item,
            {"status", "canonical_surface", "reason", "next_action", "last_reviewed"},
            f"workstreams.{name}",
        )
        status = _enum(WorkstreamStatus, item.get("status"), f"workstreams.{name}.status")
        if status == WorkstreamStatus.NEEDS_USER_DECISION:
            raise ValueError(
                f"workstreams.{name}.status cannot persist needs-user-decision; "
                "remove the entry until a decision is made"
            )
        parsed[name] = WorkstreamDecision(
            status=status,
            canonical_surface=_text(
                item.get("canonical_surface"), f"workstreams.{name}.canonical_surface"
            ),
            reason=_text(item.get("reason"), f"workstreams.{name}.reason"),
            next_action=_text(item.get("next_action"), f"workstreams.{name}.next_action"),
            last_reviewed=_date(item.get("last_reviewed"), f"workstreams.{name}.last_reviewed"),
        )
    return parsed


def _parse_surfaces(
    raw: object,
    known_workstreams: set[str],
) -> dict[str, SurfaceDecision]:
    entries = _object(raw, "surfaces")
    parsed: dict[str, SurfaceDecision] = {}
    for branch, value in entries.items():
        if branch != branch.strip() or branch.startswith(("origin/", "remotes/", "refs/")):
            raise ValueError(
                f"Surface keys must use normalized branch names without remote prefixes: {branch}"
            )
        item = _object(value, f"surfaces.{branch}")
        _keys(
            item,
            {"workstreams", "disposition", "reason", "next_action", "last_reviewed"},
            f"surfaces.{branch}",
        )
        workstreams = _string_list(item.get("workstreams"), f"surfaces.{branch}.workstreams")
        unknown = sorted(set(workstreams) - known_workstreams)
        if unknown:
            raise ValueError(
                f"Unknown workstream in surfaces.{branch}.workstreams: {', '.join(unknown)}"
            )
        parsed[branch] = SurfaceDecision(
            workstreams=workstreams,
            disposition=_enum(
                SurfaceDisposition,
                item.get("disposition"),
                f"surfaces.{branch}.disposition",
            ),
            reason=_text(item.get("reason"), f"surfaces.{branch}.reason"),
            next_action=_text(item.get("next_action"), f"surfaces.{branch}.next_action"),
            last_reviewed=_date(item.get("last_reviewed"), f"surfaces.{branch}.last_reviewed"),
        )
    return parsed


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _keys(value: dict[str, object], allowed: set[str], label: str) -> None:
    missing = sorted(allowed - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {', '.join(unknown)}")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{label} must be a non-empty array of strings")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicates")
    return normalized


def _date(value: object, label: str) -> date:
    text = _text(value, label)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD") from exc


def _enum(enum_type, value: object, label: str):
    text = _text(value, label)
    try:
        return enum_type(text)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{label} must be one of: {choices}") from exc
