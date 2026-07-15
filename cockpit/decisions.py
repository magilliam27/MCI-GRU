from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Literal

from cockpit.models import Confidence, SurfaceDisposition, WorkstreamStatus

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from cockpit.models import AutoDecisionSet

DECISION_REGISTRY_PATH = "docs/agents/cockpit/workstream-decisions.json"
FORMAT_VERSION = 2
# Backward-compatible parsing: version 1 files (which have no ``workstream_aliases``
# section) remain valid, version 2 adds the optional alias section, and any other
# version is rejected. The rejection contract test targets version 3.
SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2})
CONTINUING_WORKSTREAM_STATUSES = frozenset(
    {
        WorkstreamStatus.ACTIVE,
        WorkstreamStatus.BLOCKED,
        WorkstreamStatus.LOCAL_ONLY,
        WorkstreamStatus.READY_FOR_AGENT,
    }
)


@dataclass(frozen=True)
class WorkstreamDecision:
    status: WorkstreamStatus
    canonical_surface: str
    reason: str
    next_action: str
    last_reviewed: date
    provenance: Literal["override", "auto"] = "override"


@dataclass(frozen=True)
class SurfaceDecision:
    workstreams: tuple[str, ...]
    disposition: SurfaceDisposition
    reason: str
    next_action: str
    last_reviewed: date
    provenance: Literal["override", "auto"] = "override"
    confidence: Confidence | None = None
    association_basis: str = "explicit-surface"


@dataclass(frozen=True)
class DecisionRegistry:
    workstreams: dict[str, WorkstreamDecision] = field(default_factory=dict)
    surfaces: dict[str, SurfaceDecision] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    def is_reviewed(self, workstream: str, branch: str) -> bool:
        surface = self.surfaces.get(branch)
        return surface is not None and workstream in surface.workstreams


def overlay_auto_decisions(
    registry: DecisionRegistry,
    auto: AutoDecisionSet,
) -> DecisionRegistry:
    auto_surfaces = {
        branch: SurfaceDecision(
            workstreams=decision.workstreams,
            disposition=decision.disposition,
            reason=f"Auto rule {decision.rule}: {decision.evidence}",
            next_action=_auto_surface_next_action(decision.disposition),
            last_reviewed=decision.last_reviewed,
            provenance="auto",
            confidence=decision.confidence,
            association_basis=decision.association_basis,
        )
        for branch, decision in auto.surfaces.items()
    }
    explicit_canonical_surfaces: dict[str, str] = {}
    for branch, decision in registry.surfaces.items():
        if decision.disposition != SurfaceDisposition.CANONICAL:
            continue
        for workstream in decision.workstreams:
            existing = explicit_canonical_surfaces.get(workstream)
            if existing is not None and existing != branch:
                raise ValueError(
                    "Multiple explicit canonical surfaces for workstream "
                    f"{workstream}: {existing}, {branch}"
                )
            explicit_canonical_surfaces[workstream] = branch
    explicit_canonical_workstreams = set(explicit_canonical_surfaces)
    for branch, decision in tuple(auto_surfaces.items()):
        if (
            branch not in registry.surfaces
            and decision.disposition == SurfaceDisposition.CANONICAL
            and explicit_canonical_workstreams.intersection(decision.workstreams)
        ):
            workstreams = sorted(explicit_canonical_workstreams.intersection(decision.workstreams))
            auto_surfaces[branch] = replace(
                decision,
                disposition=SurfaceDisposition.STALE,
                reason=(
                    "Generated canonical choice suppressed by an explicit canonical surface "
                    f"for {', '.join(workstreams)}."
                ),
                next_action="Follow the explicit canonical surface recorded in the registry.",
            )
    auto_workstreams = {
        name: WorkstreamDecision(
            status=decision.status,
            canonical_surface=decision.canonical_surface,
            reason=f"Auto rule {decision.rule}: {decision.evidence}",
            next_action=_auto_workstream_next_action(decision.status),
            last_reviewed=decision.last_reviewed,
            provenance="auto",
        )
        for name, decision in auto.workstreams.items()
    }
    effective_workstreams = {**auto_workstreams, **registry.workstreams}
    for workstream, branch in explicit_canonical_surfaces.items():
        if workstream in effective_workstreams:
            effective_workstreams[workstream] = replace(
                effective_workstreams[workstream], canonical_surface=branch
            )
    effective_surfaces = {**auto_surfaces, **registry.surfaces}
    _validate_effective_canonical_surfaces(
        effective_workstreams,
        effective_surfaces,
        registry.workstreams,
        registry.surfaces,
    )
    return DecisionRegistry(
        workstreams=effective_workstreams,
        surfaces=effective_surfaces,
        aliases=dict(registry.aliases),
    )


def _validate_effective_canonical_surfaces(
    workstreams: dict[str, WorkstreamDecision],
    surfaces: dict[str, SurfaceDecision],
    explicit_workstreams: dict[str, WorkstreamDecision],
    explicit_surfaces: dict[str, SurfaceDecision],
) -> None:
    for workstream, decision in workstreams.items():
        if decision.status not in CONTINUING_WORKSTREAM_STATUSES:
            continue
        has_explicit_surface = any(
            workstream in surface.workstreams for surface in explicit_surfaces.values()
        )
        associated = {
            branch: surface
            for branch, surface in surfaces.items()
            if workstream in surface.workstreams
        }
        if not associated:
            continue
        canonical = sorted(
            branch
            for branch, surface in associated.items()
            if surface.disposition == SurfaceDisposition.CANONICAL
        )
        explicit_workstream = explicit_workstreams.get(workstream)
        has_authoritative_external_route = bool(
            explicit_workstream is not None
            and explicit_workstream.canonical_surface.strip()
            and explicit_workstream.canonical_surface not in surfaces
        )
        if has_explicit_surface and not canonical and not has_authoritative_external_route:
            raise ValueError(
                f"No effective canonical surface for continuing workstream {workstream}"
            )
        if (
            not has_authoritative_external_route
            and len(canonical) == 1
            and decision.canonical_surface != canonical[0]
        ):
            workstreams[workstream] = replace(
                decision,
                canonical_surface=canonical[0],
            )


def _auto_surface_next_action(disposition: SurfaceDisposition) -> str:
    if disposition == SurfaceDisposition.CANONICAL:
        return "Continue from this generated canonical surface."
    if disposition == SurfaceDisposition.ARCHIVE:
        return "Retain as archive-labelled evidence; remove only with separate approval."
    if disposition == SurfaceDisposition.STALE:
        return "Review before resuming; remove only with separate approval."
    return "Keep parked until an explicit override resumes it."


def _auto_workstream_next_action(status: WorkstreamStatus) -> str:
    if status == WorkstreamStatus.ACTIVE:
        return "Continue from the generated canonical surface."
    if status == WorkstreamStatus.DONE:
        return "Keep completion evidence; reopen only when new live evidence appears."
    if status == WorkstreamStatus.STALE:
        return "Review stale evidence before resuming."
    return "Follow the generated workstream status."


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
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read {DECISION_REGISTRY_PATH}") from exc
    return parse_decision_registry_text(payload, known_workstreams=known_workstreams)


def parse_decision_registry_text(
    payload: str,
    *,
    known_workstreams: Collection[str],
    admit_historical_workstreams: bool = False,
) -> DecisionRegistry:
    """Parse a registry payload from a file or committed Git evidence.

    Current registry loads keep the default strict known-workstream boundary.
    A committed historical snapshot may opt into names declared by that snapshot
    so removed overrides remain available for lifecycle comparison.
    """
    try:
        raw = json.loads(payload)
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
    if admit_historical_workstreams:
        known.update(_historical_workstream_names(root))
    workstreams = _parse_workstreams(root.get("workstreams"), known)
    surfaces = _parse_surfaces(root.get("surfaces"), known)
    aliases = _parse_aliases(root.get("workstream_aliases"))
    return DecisionRegistry(workstreams=workstreams, surfaces=surfaces, aliases=aliases)


def _historical_workstream_names(root: dict[str, object]) -> set[str]:
    names = set(_object(root.get("workstreams"), "workstreams"))
    surfaces = _object(root.get("surfaces"), "surfaces")
    for branch, value in surfaces.items():
        item = _object(value, f"surfaces.{branch}")
        names.update(_string_list(item.get("workstreams"), f"surfaces.{branch}.workstreams"))
    return names


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
