from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cockpit.models import SurfaceDisposition, WorkstreamStatus

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from datetime import date


@dataclass(frozen=True)
class WorkstreamOverrideCommand:
    workstream: str
    status: WorkstreamStatus
    reason: str


@dataclass(frozen=True)
class SurfaceOverrideCommand:
    branch: str
    disposition: SurfaceDisposition
    workstream: str
    reason: str


@dataclass(frozen=True)
class ClearWorkstreamOverrideCommand:
    workstream: str


@dataclass(frozen=True)
class ClearSurfaceOverrideCommand:
    branch: str


OverrideCommand = (
    WorkstreamOverrideCommand
    | SurfaceOverrideCommand
    | ClearWorkstreamOverrideCommand
    | ClearSurfaceOverrideCommand
)


@dataclass(frozen=True)
class OverrideApplication:
    registry: dict[str, object]
    processed_command_ids: tuple[str, ...]
    applied: bool


_WORKSTREAM_OVERRIDE = re.compile(
    r'^/cockpit override workstream "([^"\r\n]+)" status ([a-z-]+) '
    r'reason "([^"\r\n]+)"$'
)
_SURFACE_OVERRIDE = re.compile(
    r'^/cockpit override surface "([^"\r\n]+)" disposition ([a-z-]+) '
    r'workstream "([^"\r\n]+)" reason "([^"\r\n]+)"$'
)
_CLEAR_WORKSTREAM_OVERRIDE = re.compile(r'^/cockpit clear-override workstream "([^"\r\n]+)"$')
_CLEAR_SURFACE_OVERRIDE = re.compile(r'^/cockpit clear-override surface "([^"\r\n]+)"$')


def parse_override_command(
    text: str,
    *,
    known_workstreams: Collection[str],
    known_branches: Collection[str],
) -> (
    WorkstreamOverrideCommand
    | SurfaceOverrideCommand
    | ClearWorkstreamOverrideCommand
    | ClearSurfaceOverrideCommand
):
    """Parse and validate one structured cockpit override command."""
    clear_workstream_match = _CLEAR_WORKSTREAM_OVERRIDE.fullmatch(text)
    if clear_workstream_match is not None:
        workstream = clear_workstream_match.group(1)
        if workstream not in known_workstreams:
            raise ValueError(f"Unknown workstream: {workstream}")
        return ClearWorkstreamOverrideCommand(workstream=workstream)

    clear_surface_match = _CLEAR_SURFACE_OVERRIDE.fullmatch(text)
    if clear_surface_match is not None:
        branch = clear_surface_match.group(1)
        _validate_branch(branch)
        if branch not in known_branches:
            raise ValueError(f"Unknown branch: {branch}")
        return ClearSurfaceOverrideCommand(branch=branch)

    surface_match = _SURFACE_OVERRIDE.fullmatch(text)
    if surface_match is not None:
        branch, raw_disposition, workstream, reason = surface_match.groups()
        _validate_branch(branch)
        if branch not in known_branches:
            raise ValueError(f"Unknown branch: {branch}")
        if workstream not in known_workstreams:
            raise ValueError(f"Unknown workstream: {workstream}")
        try:
            disposition = SurfaceDisposition(raw_disposition)
        except ValueError as exc:
            raise ValueError(f"Unknown surface disposition: {raw_disposition}") from exc
        return SurfaceOverrideCommand(
            branch=branch,
            disposition=disposition,
            workstream=workstream,
            reason=_reason(reason),
        )

    match = _WORKSTREAM_OVERRIDE.fullmatch(text)
    if match is None:
        raise ValueError("Malformed cockpit override command")
    workstream, raw_status, reason = match.groups()
    if workstream not in known_workstreams:
        raise ValueError(f"Unknown workstream: {workstream}")
    try:
        status = WorkstreamStatus(raw_status)
    except ValueError as exc:
        raise ValueError(f"Unknown workstream status: {raw_status}") from exc
    if status == WorkstreamStatus.NEEDS_USER_DECISION:
        raise ValueError("Workstream status cannot be needs-user-decision")
    return WorkstreamOverrideCommand(
        workstream=workstream,
        status=status,
        reason=_reason(reason),
    )


def _reason(value: str) -> str:
    if not value.strip():
        raise ValueError("reason must be a non-empty string")
    if value != value.strip():
        raise ValueError("reason must not have leading or trailing whitespace")
    return value


def _validate_branch(branch: str) -> None:
    if branch != branch.strip() or branch.startswith(("origin/", "remotes/", "refs/")):
        raise ValueError("Surface target must use a normalized branch name")


def apply_override_command(
    registry: Mapping[str, object],
    command: OverrideCommand,
    *,
    command_id: str,
    applied_on: date,
    canonical_surfaces: Mapping[str, str],
    processed_command_ids: Collection[str] = (),
) -> OverrideApplication:
    """Apply one validated command to a copied JSON-compatible registry."""
    if not command_id.strip():
        raise ValueError("command_id must be a non-empty string")
    updated = deepcopy(dict(registry))
    receipt_ids = tuple(sorted(set(processed_command_ids)))
    if command_id in receipt_ids:
        return OverrideApplication(
            registry=updated,
            processed_command_ids=receipt_ids,
            applied=False,
        )
    if isinstance(command, WorkstreamOverrideCommand):
        workstreams = updated.get("workstreams")
        if not isinstance(workstreams, dict):
            raise ValueError("decision registry workstreams must be a JSON object")
        existing = workstreams.get(command.workstream)
        existing_canonical = (
            existing.get("canonical_surface") if isinstance(existing, dict) else None
        )
        canonical_surface = (
            existing_canonical
            if isinstance(existing_canonical, str) and existing_canonical.strip()
            else canonical_surfaces.get(command.workstream)
        )
        if not isinstance(canonical_surface, str) or not canonical_surface.strip():
            raise ValueError(f"Missing canonical surface for workstream: {command.workstream}")
        workstreams[command.workstream] = {
            "status": command.status.value,
            "canonical_surface": canonical_surface.strip(),
            "reason": command.reason,
            "next_action": command.reason,
            "last_reviewed": applied_on.isoformat(),
        }
    elif isinstance(command, SurfaceOverrideCommand):
        surfaces = updated.get("surfaces")
        if not isinstance(surfaces, dict):
            raise ValueError("decision registry surfaces must be a JSON object")
        if command.disposition == SurfaceDisposition.CANONICAL:
            for branch, existing in surfaces.items():
                if branch == command.branch or not isinstance(existing, dict):
                    continue
                if existing.get(
                    "disposition"
                ) == SurfaceDisposition.CANONICAL.value and command.workstream in existing.get(
                    "workstreams", ()
                ):
                    raise ValueError(
                        f"Workstream {command.workstream} already has explicit "
                        f"canonical surface {branch}; clear it before selecting "
                        f"{command.branch}"
                    )
        surfaces[command.branch] = {
            "workstreams": [command.workstream],
            "disposition": command.disposition.value,
            "reason": command.reason,
            "next_action": command.reason,
            "last_reviewed": applied_on.isoformat(),
        }
    elif isinstance(command, ClearWorkstreamOverrideCommand):
        workstreams = updated.get("workstreams")
        if not isinstance(workstreams, dict):
            raise ValueError("decision registry workstreams must be a JSON object")
        if command.workstream not in workstreams:
            raise ValueError(f"No explicit workstream override exists: {command.workstream}")
        del workstreams[command.workstream]
    elif isinstance(command, ClearSurfaceOverrideCommand):
        surfaces = updated.get("surfaces")
        if not isinstance(surfaces, dict):
            raise ValueError("decision registry surfaces must be a JSON object")
        if command.branch not in surfaces:
            raise ValueError(f"No explicit surface override exists: {command.branch}")
        del surfaces[command.branch]
    else:
        raise TypeError(f"Unsupported cockpit override command: {type(command).__name__}")
    receipt_ids = tuple(sorted({*receipt_ids, command_id}))
    return OverrideApplication(
        registry=updated,
        processed_command_ids=receipt_ids,
        applied=True,
    )
