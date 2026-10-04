"""Declare, from the effective configuration, which file each required input role reads.

The required roles come from the configuration alone, never from what was
observed, so an input that was enabled but never read stays visible (#208).
Each role is then bound to one declared package file, or left undeclared:

- a selected file lying under ``data.input_package_root`` binds to its entry in
  the pinned ``data.input_package_manifest``;
- an auxiliary input in capture or replay mode binds to the snapshot package
  that retained its original bytes;
- anything else, such as a live provider read in source mode, is required but
  has no declared package, which an attachment reports as incomplete.

:func:`attach_window_inputs` retains the result beside one window attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mci_gru.data.auxiliary_quality import REGIME_INPUT_ROLES
from mci_gru.data.input_manifest import read_input_manifest
from mci_gru.data.path_resolver import PROJECT_ROOT, resolve_project_data_path
from mci_gru.data.pit import CESSATION_EVENTS_ROLE
from mci_gru.evaluation.run_input_attachments import (
    ATTACHMENT_DIR,
    RoleBinding,
    attach_run_inputs,
    read_run_inputs,
)

if TYPE_CHECKING:
    import logging

    from mci_gru.config import ExperimentConfig
    from mci_gru.data.input_manifest import ManifestSnapshot
    from mci_gru.data.input_observations import InputObservations

SNAPSHOT_FILE = "observations.bin"
# Roles whose reads pass through the auxiliary snapshot store.
SNAPSHOT_MODES = ("capture", "replay")


@dataclass(frozen=True)
class RequiredRole:
    """One input role the configuration requires, and where a file role reads from.

    ``configured_path`` is ``None`` for a provider read. ``basename_fallback``
    mirrors the loader's own resolution for the role.
    """

    role: str
    configured_path: str | None = None
    basename_fallback: bool = False
    auxiliary: bool = False


@dataclass(frozen=True)
class InputDeclarations:
    manifests: tuple[ManifestSnapshot, ...]
    required: tuple[RoleBinding, ...]


def required_input_roles(config: ExperimentConfig) -> tuple[RequiredRole, ...]:
    """Every input role this configuration's preparation must read."""
    data, features = config.data, config.features
    roles: list[RequiredRole] = []
    if data.experiment_mode == "index_level":
        if data.index_filename:
            roles.append(RequiredRole("data.index_filename", data.index_filename))
        else:
            roles.append(RequiredRole("fred.close", auxiliary=True))
    else:
        roles.append(RequiredRole("data.filename", data.filename))
        if data.use_pit_universe:
            roles.append(RequiredRole("data.pit_universe_csv", data.pit_universe_csv))
            if data.pit_cessation_events_csv:
                roles.append(RequiredRole(CESSATION_EVENTS_ROLE, data.pit_cessation_events_csv))
        if config.graph.use_sector_relation and config.graph.sector_map_csv:
            roles.append(RequiredRole("graph.sector_map_csv", config.graph.sector_map_csv))
    if features.include_vix:
        if data.auxiliary_sources.get("vix", "file") == "lseg":
            roles.append(RequiredRole("lseg.vix", auxiliary=True))
        else:
            roles.append(
                RequiredRole(
                    "implicit.vix_csv", "vix_data.csv", basename_fallback=True, auxiliary=True
                )
            )
    if features.include_credit_spread:
        roles += [
            RequiredRole(f"fred.{name}", auxiliary=True) for name in ("ig_spread", "hy_spread")
        ]
    if features.include_global_regime:
        if features.regime_inputs_csv:
            roles.append(
                RequiredRole(
                    "features.regime_inputs_csv",
                    features.regime_inputs_csv,
                    basename_fallback=True,
                    auxiliary=True,
                )
            )
        else:
            roles += [
                RequiredRole(f"fred.{column}", auxiliary=True) for column in REGIME_INPUT_ROLES
            ]
    return tuple(roles)


def declare_window_inputs(
    config: ExperimentConfig, observations: InputObservations
) -> InputDeclarations:
    """Bind each required role to a declared package file, or leave it undeclared.

    The declared package is read only when a selected file lies under its root;
    then a manifest that is missing or differs from its pinned digest raises,
    because the configuration names an identity that is not there.
    """
    data = config.data
    manifests: dict[str, ManifestSnapshot] = {}
    package_root = (
        _declared_location(data.input_package_root) if data.input_package_manifest else None
    )
    package = None

    retained = _retained_snapshots(observations)
    snapshots = data.auxiliary_snapshot_mode in SNAPSHOT_MODES
    required: list[RoleBinding] = []
    for entry in required_input_roles(config):
        if entry.auxiliary and snapshots:
            packages = retained.get(entry.role, {})
            for digest, path in packages.items():
                if digest not in manifests:
                    manifests[digest] = read_input_manifest(path, expected_sha256=digest)
                required.append(RoleBinding(entry.role, digest, SNAPSHOT_FILE))
            if not packages:
                required.append(RoleBinding(entry.role, None, None))
            continue
        location = _file_location(entry)
        if (
            package_root is not None
            and location is not None
            and location.is_relative_to(package_root)
        ):
            if package is None:
                package = read_input_manifest(
                    resolve_project_data_path(
                        data.input_package_manifest, allow_basename_fallback=False
                    ),
                    expected_sha256=data.input_package_manifest_sha256,
                )
                manifests[package.sha256] = package
            relative = location.relative_to(package_root).as_posix()
            required.append(RoleBinding(entry.role, package.sha256, relative))
        else:
            required.append(RoleBinding(entry.role, None, None))
    return InputDeclarations(tuple(manifests.values()), tuple(required))


def attach_window_inputs(
    config: ExperimentConfig,
    observations: InputObservations,
    window_dir: str | Path,
    *,
    attempt_id: str,
    execution_start: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Retain one attempt's declarations and reads; return its reference and status.

    The attempt has only started, so execution is ``unknown`` and points at the
    start record (#144) exactly as ``run_metadata.json`` does, relative to the
    window directory; preservation is never proven here (#207). Incomplete
    evidence is logged and retained, never raised: admission gates the run. A
    configuration that names a missing or altered package raises; the runner
    records that as a failed attachment and carries on.
    """
    declarations = declare_window_inputs(config, observations)
    directory = Path(window_dir) / ATTACHMENT_DIR / attempt_id
    reference = attach_run_inputs(
        directory,
        manifests=declarations.manifests,
        observations=observations,
        required=declarations.required,
        execution={"status": "unknown", "start": dict(execution_start)},
    )
    inputs = read_run_inputs(reference)
    if inputs.problems:
        logger.warning("Input attachment is %s: %s", inputs.status, "; ".join(inputs.problems))
    return {
        "path": reference.path.relative_to(Path(window_dir)).as_posix(),
        "sha256": reference.sha256,
        "status": inputs.status,
    }


def _retained_snapshots(observations: InputObservations) -> dict[str, dict[str, Path]]:
    """Snapshot manifests behind each consumed role, keyed by manifest digest."""
    recorded = observations.to_dict()
    events = {event["observation_id"]: event for event in recorded["observations"]}
    retained: dict[str, dict[str, Path]] = {}
    for use in recorded["uses"]:
        identity = events.get(use["observation_id"], {}).get("identity", {})
        if identity.get("manifest_sha256") and identity.get("manifest_path"):
            retained.setdefault(use["role"], {})[identity["manifest_sha256"]] = Path(
                identity["manifest_path"]
            )
    return retained


def _file_location(entry: RequiredRole) -> Path | None:
    if entry.configured_path is None:
        return None
    if entry.basename_fallback:
        try:
            return resolve_project_data_path(entry.configured_path).resolve()
        except FileNotFoundError:
            return None
    return _declared_location(entry.configured_path)


def _declared_location(configured: str) -> Path:
    """The configured path itself, else relative to the project root, as selected reads resolve."""
    candidate = Path(configured)
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / configured).resolve()
