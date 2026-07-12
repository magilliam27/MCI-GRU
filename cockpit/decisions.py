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
FORMAT_VERSION = 2
# Backward-compatible parsing: version 1 files (which have no ``workstream_aliases``
# section) remain valid, version 2 adds the optional alias section, and any other
# version is rejected. The rejection contract test targets version 3.
SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2})


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
    aliases: dict[str, str] = field(default_factory=dict)

    def is_reviewed(self, workstream: str, branch: str) -> bool:
        surface = self.surfaces.get(branch)
        return surface is not None and workstream in surface.workstreams


def read_registry_workstream_names(repo_root: Path) -> set[str]:
    """Peek at the registry's ``workstreams`` keys without validating the file.

    This breaks the chicken-and-egg between building the known-workstream set and
    loading the registry: the caller needs the registry's declared names to seed
    the known set, but ``load_decision_registry`` needs a known set to validate
    against. This helper performs no validation; it only reads the keys so that
    ``load_decision_registry`` can still fully validate afterward. It returns an
    empty set whenever the file is missing, the JSON is invalid, or the root or
    ``workstreams`` section is absent or not an object.
    """
    path = repo_root / DECISION_REGISTRY_PATH
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, dict):
        return set()
    workstreams = raw.get("workstreams")
    if not isinstance(workstreams, dict):
        return set()
    return {name for name in workstreams if isinstance(name, str)}


def read_registry_aliases(repo_root: Path) -> dict[str, str]:
    """Peek at the registry's ``workstream_aliases`` map without validating.

    This mirrors :func:`read_registry_workstream_names`: it lets a source read the
    token-to-canonical-name alias map without triggering the full validated load.
    It returns an empty mapping whenever the file is missing, the JSON is invalid,
    the root or ``workstream_aliases`` section is absent or not an object, or an
    individual entry has a non-string or empty token/canonical name. Whitespace is
    stripped from both sides so callers see the same normalized keys the validated
    parser would produce.
    """
    path = repo_root / DECISION_REGISTRY_PATH
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    aliases = raw.get("workstream_aliases")
    if not isinstance(aliases, dict):
        return {}
    parsed: dict[str, str] = {}
    for token, canonical in aliases.items():
        if not isinstance(token, str) or not token.strip():
            continue
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        parsed[token.strip()] = canonical.strip()
    return parsed


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
    _keys(
        root,
        {"format_version", "workstreams", "surfaces"},
        "decision registry",
        optional={"workstream_aliases"},
    )
    version = root.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported cockpit decision registry format_version: {version}")

    known = set(known_workstreams)
    workstreams = _parse_workstreams(root.get("workstreams"), known)
    surfaces = _parse_surfaces(root.get("surfaces"), known)
    aliases = _parse_aliases(root.get("workstream_aliases"))
    return DecisionRegistry(workstreams=workstreams, surfaces=surfaces, aliases=aliases)


def _parse_aliases(raw: object) -> dict[str, str]:
    """Validate the optional ``workstream_aliases`` section.

    The section is a mapping of a non-empty token (or full slug) to a non-empty
    canonical workstream name. It is optional: an absent section (``None``) is
    valid and yields an empty map, which keeps version 1 files backward
    compatible. A present-but-malformed section (non-object, non-string or empty
    value, or duplicate normalized tokens) raises ``ValueError``.
    """
    if raw is None:
        return {}
    entries = _object(raw, "workstream_aliases")
    parsed: dict[str, str] = {}
    for token, value in entries.items():
        normalized = token.strip()
        if not normalized:
            raise ValueError("workstream_aliases token must be a non-empty string")
        if normalized in parsed:
            raise ValueError(f"workstream_aliases has a duplicate token: {normalized}")
        parsed[normalized] = _text(value, f"workstream_aliases.{normalized}")
    return parsed


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


def _keys(
    value: dict[str, object],
    allowed: set[str],
    label: str,
    *,
    optional: Collection[str] = (),
) -> None:
    optional_set = set(optional)
    missing = sorted(allowed - set(value))
    unknown = sorted(set(value) - allowed - optional_set)
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
