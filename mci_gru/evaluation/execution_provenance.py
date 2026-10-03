"""Capture an execution-start snapshot, record what each member actually ran,
bind each window's outputs in a terminal receipt, and inspect all of it without
live observations.

This module does not start training or collect data inputs. A run first writes
a plan naming the windows it expects; each window's start record names that
plan, member events are chained to the start, and a terminal receipt written
only after the window's outputs exist binds the start, events and outputs. A run
receipt binds every window's terminal receipt. Callers retain the returned plan
and receipt digests alongside the artifacts as their trust anchors.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from mci_gru.evaluation.artifacts import canonical_json_bytes
from mci_gru.utils.hashing import sha256_file

SCHEMA = "mci_gru.execution_start.v1"
PLAN_SCHEMA = "mci_gru.execution_plan.v1"
WINDOW_RECEIPT_SCHEMA = "mci_gru.execution_window_receipt.v1"
RUN_RECEIPT_SCHEMA = "mci_gru.execution_run_receipt.v1"
EVIDENCE_DIR = "execution_provenance"
SOURCE_ROOTS = (
    "run_experiment.py",
    "mci_gru",
    "configs",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
)
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".txt", ".lock"}
_CREDENTIAL_ASSIGNMENT = re.compile(rb"""(?im)["']?\b([\w-]+)["']?\s*[:=]\s*([^\r\n,}]+)""")
_CREDENTIAL_URI = re.compile(rb"[a-z]+://[^/\s:]+:[^/@\s]+@", re.IGNORECASE)
# A PEM/PGP private-key armour header. Written so this pattern's own source
# text does not match it, which would otherwise exclude this module.
_PRIVATE_KEY_HEADER = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")
# An OmegaConf environment or node reference with no inline default: the value
# names where a credential comes from and is not itself a credential.
_REFERENCE_ONLY = re.compile(r"\$\{(?:oc\.env:[A-Za-z_]\w*|[A-Za-z_][\w.]*)\}")
_TRAILING_COMMENT = re.compile(r"\s+#.*$")


@dataclass(frozen=True)
class ExecutionReference:
    """A retained artifact location and the digest established when capturing it."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class CapturedExecution:
    """Decoded evidence; mutating this returned object cannot change the artifact."""

    record: dict[str, Any]
    resolved_config_bytes: bytes
    source_files: dict[str, bytes]


def capture_execution_start(
    repo_root: str | Path,
    resolved_config_path: str | Path,
    *,
    resolved_config_sha256: str,
    output_dir: str | Path,
    window_id: str | None = None,
    plan: ExecutionReference | None = None,
) -> ExecutionReference:
    """Retain source, existing redacted config and observations before preparation.

    Each call creates a distinct incomplete attempt. No existing artifact or
    source file is overwritten. The source scope is the training entry point,
    package, configuration and dependency declarations, including untracked code.
    When *plan* is given the attempt names that run plan, which must expect
    *window_id*.
    """
    if window_id is not None and (not isinstance(window_id, str) or not window_id):
        raise ValueError("Invalid window identifier")
    plan_link = None
    if plan is not None:
        plan_record = _read_plan(plan)
        if window_id not in {window["window_id"] for window in plan_record["windows"]}:
            raise ValueError("The plan does not expect this window")
        plan_link = {"run_id": plan_record["run_id"], "sha256": plan.sha256}
    root = Path(repo_root).resolve()
    captured_at = datetime.now(timezone.utc).isoformat()
    config_bytes = Path(resolved_config_path).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != resolved_config_sha256:
        raise ValueError("Resolved config digest mismatch")
    _validate_config(config_bytes)
    sources = {}
    problems = []
    for name in SOURCE_ROOTS:
        path = root / name
        if name in {"run_experiment.py", "mci_gru"} and not path.exists():
            problems.append({"path": name, "reason": "required source root missing"})
        pending = [path]
        while pending:
            candidate = pending.pop()
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
                problems.append({"path": relative, "reason": "source link excluded"})
                continue
            if candidate.is_dir():
                try:
                    pending.extend(sorted(candidate.iterdir()))
                except OSError as exc:
                    problems.append({"path": relative, "reason": type(exc).__name__})
            elif candidate.is_file() and candidate.suffix in SOURCE_SUFFIXES:
                relative = candidate.relative_to(root).as_posix()
                try:
                    content = candidate.read_bytes()
                except OSError as exc:
                    problems.append({"path": relative, "reason": type(exc).__name__})
                    continue
                if candidate.stem.lower() in {"credentials", "secrets"} or _contains_credentials(
                    content, candidate.suffix
                ):
                    problems.append(
                        {"path": relative, "reason": "credential-bearing source excluded"}
                    )
                    continue
                sources[relative] = _blob(content)
    commit = _git_observation(root, "rev-parse", "HEAD")
    status = _git_observation(root, "status", "--porcelain", "--", *SOURCE_ROOTS)
    if status["status"] == "observed":
        status["value"] = bool(status["value"])
    diff = _git_observation(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        *SOURCE_ROOTS,
        hash_output=True,
    )
    versions = {"python": {"status": "observed", "value": platform.python_version()}}
    for package in ("numpy", "pandas", "scipy", "torch", "torch-geometric"):
        try:
            versions[package] = {"status": "observed", "value": metadata.version(package)}
        except (metadata.PackageNotFoundError, OSError, ValueError) as exc:
            versions[package] = {
                "status": "unavailable",
                "value": None,
                "reason": type(exc).__name__,
            }
    hashes = {name: blob["sha256"] for name, blob in sources.items()}
    try:
        platform_observation = {"status": "observed", "value": platform.platform()}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        platform_observation = {
            "status": "unavailable",
            "value": None,
            "reason": type(exc).__name__,
        }
    record = {
        "schema": SCHEMA,
        "attempt_id": uuid.uuid4().hex,
        "window_id": window_id,
        "plan": plan_link,
        "captured_at_utc": captured_at,
        "execution_status": "incomplete",
        "resolved_config": _blob(config_bytes),
        "source_files": sources,
        "source_capture": {
            "status": "partial" if problems else "complete",
            "roots": list(SOURCE_ROOTS),
            "problems": problems,
        },
        "code_identity": {
            "schema": "mci_gru.selection_research_code_identity.v1",
            "git_commit": commit["value"],
            "git_dirty": bool(status["value"]) if status["status"] == "observed" else None,
            "git_dirty_diff_sha256": diff["value"],
            "git_observations": {"commit": commit, "status": status, "diff": diff},
            "source_hashes": hashes,
            "working_tree_source_sha256": hashlib.sha256(canonical_json_bytes(hashes)).hexdigest(),
            "runtime_versions": {
                name: observation["value"] for name, observation in versions.items()
            },
        },
        "environment": {
            "runtime_versions": versions,
            "platform": platform_observation["value"],
            "platform_observation": platform_observation,
            "python_implementation": sys.implementation.name,
            "colab_runtime_label": {"status": "unknown", "value": None, "reason": "not observed"},
        },
    }
    payload = canonical_json_bytes(record)
    destination = Path(output_dir) / "execution_provenance" / (record["attempt_id"] + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    return ExecutionReference(destination, hashlib.sha256(payload).hexdigest())


def read_execution_provenance(reference: ExecutionReference) -> CapturedExecution:
    """Read only the captured artifact; never query source paths, Git or packages."""
    try:
        payload = reference.path.read_bytes()
    except OSError as exc:
        raise ValueError("Execution evidence missing or unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("Execution evidence digest mismatch")
    record = json.loads(payload, parse_constant=_invalid_constant)
    if not isinstance(record, dict):
        raise ValueError("Invalid execution evidence object")
    if record.get("schema") != SCHEMA:
        raise ValueError("Unsupported execution evidence schema")
    try:
        _validate_metadata(record)
        if record["execution_status"] != "incomplete":
            raise ValueError("Start evidence cannot establish completion")
        if datetime.fromisoformat(record["captured_at_utc"]).tzinfo is None:
            raise ValueError("Timestamp must identify its time zone")
        config = _decode_blob(record["resolved_config"])
        _validate_config(config)
        sources = {}
        for name, blob in record["source_files"].items():
            _validate_source_path(name)
            sources[name] = _decode_blob(blob)
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
        identity = record["code_identity"]
        if hashes != identity["source_hashes"]:
            raise ValueError("Source inventory mismatch")
        if (
            hashlib.sha256(canonical_json_bytes(hashes)).hexdigest()
            != identity["working_tree_source_sha256"]
        ):
            raise ValueError("Source tree digest mismatch")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid execution evidence: " + str(exc)) from exc
    return CapturedExecution(record, config, sources)


MEMBER_EVENT_SCHEMA = "mci_gru.execution_member_event.v1"
MEMBER_EVENTS = (
    "seeded",
    "training_started",
    "checkpoint_saved",
    "checkpoint_loaded",
    "member_completed",
    "member_failed",
)
BACKEND_OBSERVATIONS = {
    "cuda_available": bool,
    "cudnn_version": int,
    "cudnn_enabled": bool,
    "cudnn_deterministic": bool,
    "cudnn_benchmark": bool,
    "deterministic_algorithms": bool,
    "float32_matmul_precision": str,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MemberExecution:
    """What one attempt's retained member events establish, and what they do not.

    ``status`` is ``complete`` only when every planned member left complete
    evidence; ``failed`` when a member recorded a failure; otherwise
    ``incomplete``. Each member lists the problems that keep it from complete.
    """

    status: str
    expected_members: int
    members: tuple[dict[str, Any], ...]


class MemberEventLog:
    """Append-only, hash-chained member events kept beside one start record.

    The start record stays immutable. Each event names the digest of the event
    before it, or the start digest for the first, so an edit, deletion or
    reorder is detectable. Events that never arrive, as when the process is
    killed, leave the member unfinished rather than complete.
    """

    def __init__(self, reference: ExecutionReference) -> None:
        start = read_execution_provenance(reference)
        self._attempt_id = start.record["attempt_id"]
        self._path = _member_events_path(reference.path)
        self._previous = reference.sha256
        self._sequence = 0
        try:
            with self._path.open("xb"):
                pass
        except FileExistsError as exc:
            raise ValueError("Member events already recorded for this attempt") from exc

    def record(self, model_id: int, event: str, data: dict[str, Any]) -> None:
        """Durably append one event for ensemble member *model_id*."""
        if event not in MEMBER_EVENTS:
            raise ValueError(f"Unknown member event: {event}")
        line = canonical_json_bytes(
            {
                "schema": MEMBER_EVENT_SCHEMA,
                "attempt_id": self._attempt_id,
                "sequence": self._sequence,
                "previous_sha256": self._previous,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_id": model_id,
                "event": event,
                "data": data,
            }
        )
        with self._path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._previous = hashlib.sha256(line).hexdigest()
        self._sequence += 1


def read_member_execution(reference: ExecutionReference) -> MemberExecution:
    """Read retained member events against the retained start and its config.

    The planned member count and seeds come from the retained resolved config,
    never from the live one; they are the denominator, not evidence of a run.
    """
    start = read_execution_provenance(reference)
    try:
        config = json.loads(start.resolved_config_bytes)
        expected = config["training"]["num_models"]
        base_seed = config["seed"]
        if type(expected) is not int or expected < 1 or type(base_seed) is not int:
            raise TypeError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Retained config does not state the planned members") from exc
    try:
        payload = _member_events_path(reference.path).read_bytes()
    except FileNotFoundError:
        payload = b""
    except OSError as exc:
        raise ValueError("Invalid member events: unreadable") from exc
    lines = payload.splitlines(keepends=True)
    # A final line without its terminator is a write the process never finished.
    truncated = bool(lines) and not lines[-1].endswith(b"\n")
    if truncated:
        lines = lines[:-1]
    states: list[dict[str, Any]] = []
    previous = reference.sha256
    try:
        for sequence, line in enumerate(lines):
            entry = json.loads(line, parse_constant=_invalid_constant)
            if not isinstance(entry, dict) or canonical_json_bytes(entry) != line:
                raise ValueError("non-canonical event")
            if (
                entry["schema"] != MEMBER_EVENT_SCHEMA
                or entry["attempt_id"] != start.record["attempt_id"]
                or entry["sequence"] != sequence
                or entry["previous_sha256"] != previous
            ):
                raise ValueError("event does not continue this attempt's chain")
            if datetime.fromisoformat(entry["recorded_at_utc"]).tzinfo is None:
                raise ValueError("timestamp must identify its time zone")
            _apply_member_event(states, entry["model_id"], entry["event"], entry["data"])
            previous = hashlib.sha256(line).hexdigest()
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid member events: " + str(exc)) from exc
    members = tuple(_member_report(state, base_seed) for state in states)
    if any(member["status"] == "failed" for member in members):
        status = "failed"
    elif (
        not truncated
        and len(members) == expected
        and all(member["status"] == "complete" for member in members)
    ):
        status = "complete"
    else:
        status = "incomplete"
    return MemberExecution(status, expected, members)


@dataclass(frozen=True)
class RunExecution:
    """What a run's retained plan, starts, events and receipts establish.

    ``status`` is ``complete`` only when a run receipt binds a complete terminal
    receipt for every window the plan expected; ``failed`` when a window's
    members recorded a failure; otherwise ``incomplete``. Each window lists the
    problems that keep it from complete.
    """

    status: str
    run_id: str
    experiment_mode: str
    expected_windows: int
    windows: tuple[dict[str, Any], ...]
    problems: list[str]
    run_receipt_sha256: str | None


def write_execution_plan(
    output_dir: str | Path, *, experiment_mode: str, window_dirs: list[str | Path]
) -> ExecutionReference:
    """Record, before any window starts, which windows this run expects.

    Window *i* is ``walkforward_window=i`` with ``window_id=str(i)``; its output
    directory is kept relative to *output_dir* so the run can be relocated.
    """
    root = Path(output_dir)
    if not isinstance(experiment_mode, str) or not experiment_mode or not window_dirs:
        raise ValueError("A plan needs an experiment mode and at least one window")
    record = {
        "schema": PLAN_SCHEMA,
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": experiment_mode,
        "windows": [
            {
                "walkforward_window": index,
                "window_id": str(index),
                "output_dir": _relative_to(root, Path(window_dir)),
            }
            for index, window_dir in enumerate(window_dirs)
        ],
    }
    return _write_new(root / EVIDENCE_DIR / f"{record['run_id']}.plan.json", record)


def write_window_receipt(
    start: ExecutionReference,
    *,
    plan: ExecutionReference,
    walkforward_window: int,
    artifacts: list[str],
) -> ExecutionReference:
    """Bind one finished window's start, member events and outputs.

    Written only after the window's outputs exist, so its absence is what an
    unfinished window looks like. *artifacts* name files or directories relative
    to the window's output directory; every file is bound by digest.
    """
    plan_record = _read_plan(plan)
    captured = read_execution_provenance(start)
    if type(walkforward_window) is not int or not (
        0 <= walkforward_window < len(plan_record["windows"])
    ):
        raise ValueError("The plan does not expect this window")
    expected = plan_record["windows"][walkforward_window]
    window_dir = start.path.parent.parent
    if (
        captured.record["window_id"] != expected["window_id"]
        or captured.record["plan"] != {"run_id": plan_record["run_id"], "sha256": plan.sha256}
        or not _same_directory(_plan_root(plan) / expected["output_dir"], window_dir)
    ):
        raise ValueError("The start does not belong to this plan window")
    events = _member_events_path(start.path)
    if not events.is_file():
        raise ValueError("No member events for this attempt")
    record = {
        "schema": WINDOW_RECEIPT_SCHEMA,
        "attempt_id": captured.record["attempt_id"],
        "run_id": plan_record["run_id"],
        "plan_sha256": plan.sha256,
        "walkforward_window": walkforward_window,
        "window_id": expected["window_id"],
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": {"path": start.path.name, "sha256": start.sha256},
        "member_events": {"path": events.name, "sha256": sha256_file(events)},
        "artifacts": _bind_artifacts(window_dir, artifacts),
    }
    return _write_new(start.path.with_name(f"{record['attempt_id']}.receipt.json"), record)


def write_run_receipt(
    plan: ExecutionReference,
    window_receipts: list[ExecutionReference],
    *,
    artifacts: list[str] = (),
) -> ExecutionReference:
    """Bind one terminal receipt per expected window, and run-level outputs."""
    plan_record = _read_plan(plan)
    root = _plan_root(plan)
    if len(window_receipts) != len(plan_record["windows"]):
        raise ValueError("A run receipt needs one terminal receipt per expected window")
    windows = []
    for index, receipt in enumerate(window_receipts):
        record = _read_record(receipt, WINDOW_RECEIPT_SCHEMA, "Terminal receipt")
        if record["plan_sha256"] != plan.sha256 or record["walkforward_window"] != index:
            raise ValueError("The terminal receipt does not belong to this plan window")
        windows.append(
            {
                "walkforward_window": index,
                "path": _relative_to(root, receipt.path),
                "sha256": receipt.sha256,
            }
        )
    record = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_id": plan_record["run_id"],
        "plan_sha256": plan.sha256,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "windows": windows,
        "artifacts": _bind_artifacts(root, list(artifacts)),
    }
    return _write_new(plan.path.with_name(f"{record['run_id']}.run_receipt.json"), record)


def read_execution_run(plan: ExecutionReference) -> RunExecution:
    """Read a run back from its retained plan without observing anything live.

    A window, event or receipt that is missing, unreadable or does not match
    what binds it is reported as a problem; it never reads as completion.
    """
    plan_record = _read_plan(plan)
    root = _plan_root(plan)
    problems: list[str] = []
    anchors: dict[int, dict[str, Any]] = {}
    run_receipt_sha256 = None
    run_receipt = plan.path.with_name(f"{plan_record['run_id']}.run_receipt.json")
    run_receipt_digest = _file_sha256(run_receipt)
    if not run_receipt.exists():
        problems.append("no run receipt")
    elif run_receipt_digest is None:
        problems.append("run receipt invalid: not a readable file")
    else:
        try:
            reference = ExecutionReference(run_receipt, run_receipt_digest)
            record = _read_record(reference, RUN_RECEIPT_SCHEMA, "Run receipt")
            if (
                record["run_id"] != plan_record["run_id"]
                or record["plan_sha256"] != plan.sha256
                or [entry["walkforward_window"] for entry in record["windows"]]
                != list(range(len(plan_record["windows"])))
            ):
                raise ValueError("does not bind this plan's windows")
            for entry in record["windows"]:
                _validate_relative_path(entry["path"])
                if not _SHA256.fullmatch(entry["sha256"]):
                    raise ValueError("invalid terminal receipt digest")
            anchors = {entry["walkforward_window"]: entry for entry in record["windows"]}
            problems.extend(_artifact_problems(root, record["artifacts"]))
            run_receipt_sha256 = reference.sha256
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            problems.append(f"run receipt invalid: {exc}")
    windows = tuple(
        _read_window(root, plan, plan_record["run_id"], window, anchors.get(index))
        for index, window in enumerate(plan_record["windows"])
    )
    if any(window["status"] == "failed" for window in windows):
        status = "failed"
    elif not problems and all(window["status"] == "complete" for window in windows):
        status = "complete"
    else:
        status = "incomplete"
    return RunExecution(
        status,
        plan_record["run_id"],
        plan_record["experiment_mode"],
        len(plan_record["windows"]),
        windows,
        problems,
        run_receipt_sha256,
    )


def _read_window(
    root: Path,
    plan: ExecutionReference,
    run_id: str,
    window: dict[str, Any],
    anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    index = window["walkforward_window"]
    window_dir = root / window["output_dir"]
    evidence = window_dir / EVIDENCE_DIR
    problems: list[str] = []
    report: dict[str, Any] = {
        "walkforward_window": index,
        "window_id": window["window_id"],
        "output_dir": window["output_dir"],
        "status": "incomplete",
        "attempt_id": None,
        "start_sha256": None,
        "receipt_sha256": None,
        "members": None,
        "problems": problems,
    }
    receipt = None
    if anchor is not None:
        path = root / anchor["path"]
        digest = _file_sha256(path)
        if not path.is_file():
            problems.append("terminal receipt missing")
        elif digest is None:
            problems.append("terminal receipt unreadable")
        elif digest != anchor["sha256"]:
            problems.append("terminal receipt differs from the run receipt")
        else:
            receipt = _window_receipt(ExecutionReference(path, anchor["sha256"]), problems)
    else:
        found = []
        for path in sorted(evidence.glob("*.receipt.json")):
            digest = _file_sha256(path)
            if digest is None:
                continue
            candidate = _window_receipt(ExecutionReference(path, digest), [])
            if (
                candidate is not None
                and candidate[1]["plan_sha256"] == plan.sha256
                and candidate[1]["walkforward_window"] == index
            ):
                found.append(candidate)
        if len(found) > 1:
            problems.append("more than one terminal receipt")
        elif found:
            receipt = found[0]
        else:
            problems.append("no terminal receipt")
    if receipt is not None:
        reference, record = receipt
        report["receipt_sha256"] = reference.sha256
        start = ExecutionReference(evidence / record["start"]["path"], record["start"]["sha256"])
    else:
        starts = _plan_starts(evidence, run_id, plan.sha256, window["window_id"])
        if len(starts) > 1:
            problems.append("more than one execution start")
        elif not starts:
            problems.append("no execution start")
        start = starts[0] if len(starts) == 1 else None
    if start is None:
        return report
    try:
        captured = read_execution_provenance(start)
    except ValueError as exc:
        problems.append(f"execution start invalid: {exc}")
        return report
    report["attempt_id"] = captured.record["attempt_id"]
    report["start_sha256"] = start.sha256
    if captured.record["window_id"] != window["window_id"] or captured.record["plan"] != {
        "run_id": run_id,
        "sha256": plan.sha256,
    }:
        problems.append("execution start belongs to another plan window")
    try:
        members = read_member_execution(start)
    except ValueError as exc:
        problems.append(str(exc))
        members = None
    report["members"] = members
    if members is not None and members.status != "complete":
        problems.append(f"members {members.status}")
    if receipt is not None:
        record = receipt[1]
        if (
            record["attempt_id"] != captured.record["attempt_id"]
            or record["window_id"] != window["window_id"]
        ):
            problems.append("terminal receipt belongs to another attempt")
        events = _member_events_path(start.path)
        if _file_sha256(events) != record["member_events"]["sha256"]:
            problems.append("member events differ from the terminal receipt")
        problems.extend(_artifact_problems(window_dir, record["artifacts"]))
        if "run_metadata.json" not in record["artifacts"]:
            problems.append("terminal receipt does not bind the run metadata")
        else:
            try:
                metadata = json.loads((window_dir / "run_metadata.json").read_bytes())
                linked = (
                    metadata["execution_start"]
                    == {
                        "path": f"{EVIDENCE_DIR}/{start.path.name}",
                        "sha256": start.sha256,
                    }
                    and metadata["walkforward_window"] == index
                )
            except (OSError, KeyError, TypeError, ValueError):
                linked = False
            if not linked:
                problems.append("metadata does not reference this start")
        for member in members.members if members is not None else ():
            name = f"checkpoints/model_{member['model_id']}_best.pth"
            if record["artifacts"].get(name) != member["checkpoint"]["loaded_sha256"]:
                problems.append(
                    f"checkpoint differs from the one member {member['model_id']} loaded"
                )
    if members is not None and members.status == "failed":
        report["status"] = "failed"
    elif receipt is not None and not problems:
        report["status"] = "complete"
    return report


def _window_receipt(
    reference: ExecutionReference, problems: list[str]
) -> tuple[ExecutionReference, dict[str, Any]] | None:
    try:
        record = _read_record(reference, WINDOW_RECEIPT_SCHEMA, "Terminal receipt")
        if not re.fullmatch(r"[0-9a-f]{32}", record["attempt_id"]):
            raise ValueError("invalid attempt identifier")
        if not _SHA256.fullmatch(record["plan_sha256"]):
            raise ValueError("invalid plan digest")
        if type(record["walkforward_window"]) is not int or type(record["window_id"]) is not str:
            raise ValueError("invalid window")
        for key in ("start", "member_events"):
            if not re.fullmatch(r"[0-9a-f]{32}(\.events\.jsonl|\.json)", record[key]["path"]):
                raise ValueError(f"invalid {key} path")
            if not _SHA256.fullmatch(record[key]["sha256"]):
                raise ValueError(f"invalid {key} digest")
        if not isinstance(record["artifacts"], dict):
            raise ValueError("invalid artifacts")
        for name, digest in record["artifacts"].items():
            _validate_relative_path(name)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError("invalid artifact digest")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        problems.append(f"terminal receipt invalid: {exc}")
        return None
    return reference, record


def _plan_starts(
    evidence: Path, run_id: str, plan_sha256: str, window_id: str
) -> list[ExecutionReference]:
    starts = []
    for path in sorted(evidence.glob("*.json")):
        digest = _file_sha256(path)
        if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name) or digest is None:
            continue
        reference = ExecutionReference(path, digest)
        try:
            record = read_execution_provenance(reference).record
        except ValueError:
            continue
        if record["window_id"] == window_id and record["plan"] == {
            "run_id": run_id,
            "sha256": plan_sha256,
        }:
            starts.append(reference)
    return starts


def _read_plan(reference: ExecutionReference) -> dict[str, Any]:
    record = _read_record(reference, PLAN_SCHEMA, "Execution plan")
    try:
        if not re.fullmatch(r"[0-9a-f]{32}", record["run_id"]):
            raise ValueError("invalid run identifier")
        if not isinstance(record["experiment_mode"], str) or not record["experiment_mode"]:
            raise ValueError("invalid experiment mode")
        if datetime.fromisoformat(record["created_at_utc"]).tzinfo is None:
            raise ValueError("timestamp must identify its time zone")
        windows = record["windows"]
        if not isinstance(windows, list) or not windows:
            raise ValueError("no expected windows")
        for index, window in enumerate(windows):
            if set(window) != {"walkforward_window", "window_id", "output_dir"}:
                raise ValueError("invalid window entry")
            if window["walkforward_window"] != index or window["window_id"] != str(index):
                raise ValueError("windows out of order")
            if window["output_dir"] != ".":
                _validate_relative_path(window["output_dir"])
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid execution plan: " + str(exc)) from exc
    return record


def _read_record(reference: ExecutionReference, schema: str, label: str) -> dict[str, Any]:
    try:
        payload = reference.path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} missing or unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"{label} digest mismatch")
    record = json.loads(payload, parse_constant=_invalid_constant)
    if not isinstance(record, dict) or canonical_json_bytes(record) != payload:
        raise ValueError(f"{label} is not canonical")
    if record.get("schema") != schema:
        raise ValueError(f"Unsupported {label.lower()} schema")
    return record


def _write_new(path: Path, record: dict[str, Any]) -> ExecutionReference:
    payload = canonical_json_bytes(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return ExecutionReference(path, hashlib.sha256(payload).hexdigest())


def _bind_artifacts(directory: Path, names: list[str]) -> dict[str, str]:
    bound = {}
    for name in names:
        path = directory / name
        if path.is_dir():
            files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            files = [path]
        else:
            raise ValueError(f"Artifact missing: {name}")
        for file in files:
            relative = file.relative_to(directory).as_posix()
            if relative.split("/")[0] == EVIDENCE_DIR:
                raise ValueError("Execution evidence cannot bind itself")
            bound[relative] = sha256_file(file)
    return bound


def _artifact_problems(directory: Path, artifacts: dict[str, str]) -> list[str]:
    problems = []
    for name, digest in sorted(artifacts.items()):
        _validate_relative_path(name)
        path = directory / name
        current = _file_sha256(path)
        if not path.is_file():
            problems.append(f"artifact missing: {name}")
        elif current is None:
            problems.append(f"artifact unreadable: {name}")
        elif current != digest:
            problems.append(f"artifact differs: {name}")
    return problems


def _file_sha256(path: Path) -> str | None:
    """Digest a regular file; a directory or unreadable file has no digest."""
    if not path.is_file():
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def _plan_root(plan: ExecutionReference) -> Path:
    return plan.path.parent.parent


def _relative_to(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Evidence must stay inside the run's output directory") from exc
    return relative or "."


def _same_directory(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _member_events_path(start_path: Path) -> Path:
    return start_path.with_name(start_path.stem + ".events.jsonl")


def _apply_member_event(states: list[dict[str, Any]], model_id: Any, event: str, data: Any) -> None:
    if type(model_id) is not int or not isinstance(data, dict):
        raise ValueError("invalid event shape")
    if model_id == len(states):
        if states and states[-1]["stage"] != "completed":
            raise ValueError("a member started before the previous one finished")
        states.append(
            {
                "model_id": model_id,
                "stage": "new",
                "applied_seed": None,
                "torch_initial_seed": None,
                "runtime": None,
                "saved_sha256": None,
                "loaded": False,
                "loaded_sha256": None,
                "error_type": None,
            }
        )
    elif model_id != len(states) - 1:
        raise ValueError("member events out of order")
    state = states[model_id]
    stage = state["stage"]
    if event == "seeded" and stage == "new":
        if type(data["seed"]) is not int:
            raise ValueError("invalid applied seed")
        _observation_value(data["torch_initial_seed"], int)
        state.update(
            stage="seeded",
            applied_seed=data["seed"],
            torch_initial_seed=data["torch_initial_seed"],
        )
    elif event == "training_started" and stage == "seeded":
        if type(data["amp_requested"]) is not bool or type(data["amp_effective"]) is not bool:
            raise ValueError("invalid precision record")
        _observation_value(data["device"], str)
        _observation_value(data["parameter_dtype"], str)
        if set(data["backend"]) != set(BACKEND_OBSERVATIONS):
            raise ValueError("invalid backend inventory")
        for name, value_type in BACKEND_OBSERVATIONS.items():
            _observation_value(data["backend"][name], value_type)
        state.update(stage="started", runtime=data)
    elif event == "checkpoint_saved" and stage == "started" and not state["loaded"]:
        if (
            type(data["epoch"]) is not int
            or data["epoch"] < 1
            or type(data["size_bytes"]) is not int
            or not _SHA256.fullmatch(data["sha256"])
        ):
            raise ValueError("invalid checkpoint record")
        state["saved_sha256"] = data["sha256"]
    elif event == "checkpoint_loaded" and stage == "started" and not state["loaded"]:
        if data["sha256"] is not None and not _SHA256.fullmatch(data["sha256"]):
            raise ValueError("invalid checkpoint record")
        state.update(loaded=True, loaded_sha256=data["sha256"])
    elif event == "member_completed" and stage == "started" and state["loaded"]:
        state["stage"] = "completed"
    elif event == "member_failed" and stage in {"seeded", "started"}:
        if not isinstance(data["error_type"], str) or not data["error_type"]:
            raise ValueError("invalid failure record")
        state.update(stage="failed", error_type=data["error_type"])
    else:
        raise ValueError(f"unexpected {event} event")


def _member_report(state: dict[str, Any], base_seed: int) -> dict[str, Any]:
    planned_seed = base_seed + state["model_id"]
    problems = []
    if state["stage"] != "failed":
        if state["stage"] != "completed":
            problems.append("member did not finish")
        if state["applied_seed"] != planned_seed:
            problems.append("applied seed differs from the plan")
        seed_observation = state["torch_initial_seed"]
        if seed_observation is None or seed_observation != {
            "status": "observed",
            "value": state["applied_seed"],
        }:
            problems.append("seed application not observed")
        runtime = state["runtime"]
        if runtime is not None and runtime["device"]["status"] != "observed":
            problems.append("device not observed")
        if state["saved_sha256"] is None:
            problems.append("no checkpoint saved in this attempt")
        if state["loaded"] and state["loaded_sha256"] is None:
            problems.append("checkpoint missing at load")
        elif state["loaded"] and state["loaded_sha256"] != state["saved_sha256"]:
            problems.append("loaded checkpoint differs from the last one saved")
    if state["stage"] == "failed":
        status = "failed"
    else:
        status = "incomplete" if problems else "complete"
    return {
        "model_id": state["model_id"],
        "status": status,
        "planned_seed": planned_seed,
        "applied_seed": state["applied_seed"],
        "torch_initial_seed": state["torch_initial_seed"],
        "runtime": state["runtime"],
        "checkpoint": {
            "saved_sha256": state["saved_sha256"],
            "loaded_sha256": state["loaded_sha256"],
        },
        "error_type": state["error_type"],
        "problems": problems,
    }


def _validate_source_path(name: str) -> None:
    try:
        _validate_relative_path(name)
    except ValueError:
        raise ValueError("Unsafe source path") from None
    if name.split("/")[0] not in SOURCE_ROOTS:
        raise ValueError("Unsafe source path")


def _validate_relative_path(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("Unsafe relative path")


def _observation_value(observation: dict[str, Any], value_type: type) -> Any:
    status, value = observation["status"], observation["value"]
    if status == "observed":
        if type(value) is not value_type or (value_type is str and not value):
            raise ValueError("Invalid observed value")
    elif status in {"unavailable", "unknown"}:
        if (
            value is not None
            or not isinstance(observation["reason"], str)
            or not observation["reason"]
        ):
            raise ValueError("Invalid unavailable observation")
    else:
        raise ValueError("Invalid observation status")
    return value


def _validate_metadata(record: dict[str, Any]) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", record["attempt_id"]):
        raise ValueError("Invalid attempt identifier")
    if record["window_id"] is not None and (
        not isinstance(record["window_id"], str) or not record["window_id"]
    ):
        raise ValueError("Invalid window identifier")
    plan = record["plan"]
    if plan is not None and (
        set(plan) != {"run_id", "sha256"}
        or not isinstance(plan["run_id"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", plan["run_id"])
        or not isinstance(plan["sha256"], str)
        or not _SHA256.fullmatch(plan["sha256"])
    ):
        raise ValueError("Invalid plan link")
    capture = record["source_capture"]
    if capture["roots"] != list(SOURCE_ROOTS) or not isinstance(capture["problems"], list):
        raise ValueError("Invalid source capture scope")
    if capture["status"] != ("partial" if capture["problems"] else "complete"):
        raise ValueError("Invalid source capture status")
    for problem in capture["problems"]:
        _validate_source_path(problem["path"])
        if not isinstance(problem["reason"], str) or not problem["reason"]:
            raise ValueError("Invalid source capture problem")
    identity = record["code_identity"]
    if identity["schema"] != "mci_gru.selection_research_code_identity.v1":
        raise ValueError("Invalid code identity schema")
    observations = identity["git_observations"]
    commit = _observation_value(observations["commit"], str)
    dirty = _observation_value(observations["status"], bool)
    diff = _observation_value(observations["diff"], str)
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("Invalid Git commit")
    if diff is not None and not re.fullmatch(r"[0-9a-f]{64}", diff):
        raise ValueError("Invalid Git diff digest")
    if (identity["git_commit"], identity["git_dirty"], identity["git_dirty_diff_sha256"]) != (
        commit,
        dirty,
        diff,
    ):
        raise ValueError("Inconsistent Git observations")
    environment = record["environment"]
    versions = environment["runtime_versions"]
    if set(versions) != {"python", "numpy", "pandas", "scipy", "torch", "torch-geometric"}:
        raise ValueError("Invalid runtime version inventory")
    version_values = {name: _observation_value(value, str) for name, value in versions.items()}
    if identity["runtime_versions"] != version_values:
        raise ValueError("Inconsistent runtime versions")
    if environment["platform"] != _observation_value(environment["platform_observation"], str):
        raise ValueError("Inconsistent platform observation")
    if (
        not isinstance(environment["python_implementation"], str)
        or not environment["python_implementation"]
    ):
        raise ValueError("Invalid Python implementation")
    colab = environment["colab_runtime_label"]
    _observation_value(colab, str)
    if colab["status"] != "unknown":
        raise ValueError("Start evidence cannot establish a Colab runtime label")


def _blob(content: bytes) -> dict[str, Any]:
    return {
        "content_base64": base64.b64encode(content).decode("ascii"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _decode_blob(blob: dict[str, Any]) -> bytes:
    content = base64.b64decode(blob["content_base64"], validate=True)
    if len(content) != blob["size_bytes"] or hashlib.sha256(content).hexdigest() != blob["sha256"]:
        raise ValueError("Retained byte content mismatch")
    return content


def _sensitive_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized.endswith(("apikey", "password", "passwd", "secret", "token", "privatekey"))


def _secret_value(value: Any) -> bool:
    # A switch such as ``use_api_key: false`` names a credential but holds none.
    if isinstance(value, bool):
        return False
    return value not in (None, "", "<REDACTED>", "[REDACTED]", "null", "None", "~")


def _contains_credentials(content: bytes, suffix: str = "") -> bool:
    if _CREDENTIAL_URI.search(content) or _PRIVATE_KEY_HEADER.search(content):
        return True
    if suffix == ".py":
        try:
            tree = ast.parse(content)
        except (SyntaxError, UnicodeError, ValueError):
            # Unparseable Python cannot be screened reliably; exclude it.
            return True
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Assign):
                pairs = [(target, node.value) for target in node.targets]
            elif isinstance(node, ast.AnnAssign):
                pairs = [(node.target, node.value)]
            elif isinstance(node, ast.Dict):
                pairs = list(zip(node.keys, node.values, strict=True))
            elif isinstance(node, ast.keyword):
                pairs = [(node.arg, node.value)]
            for key, value in pairs:
                if isinstance(key, ast.Subscript):
                    key = key.slice
                name = (
                    key.id
                    if isinstance(key, ast.Name)
                    else key.attr
                    if isinstance(key, ast.Attribute)
                    else key.value
                    if isinstance(key, ast.Constant)
                    else key
                )
                if (
                    isinstance(name, str)
                    and _sensitive_name(name)
                    and isinstance(value, ast.Constant)
                    and _secret_value(value.value)
                ):
                    return True
        return False
    return any(
        _sensitive_name(match[1].decode("utf-8", errors="replace"))
        and _secret_value(match[2].decode("utf-8", errors="replace").strip().strip("\"'"))
        and not _reference_only(content, match.start(2))
        for match in _CREDENTIAL_ASSIGNMENT.finditer(content)
    )


def _reference_only(content: bytes, start: int) -> bool:
    end = content.find(b"\n", start)
    value = content[start : end if end >= 0 else len(content)].decode("utf-8", errors="replace")
    value = _TRAILING_COMMENT.sub("", value).strip().strip("\"'")
    return _REFERENCE_ONLY.fullmatch(value) is not None


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


def _validate_config(content: bytes) -> None:
    try:
        payload = json.loads(content, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid resolved config JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Resolved config must be a credential-free object")
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if any(_sensitive_name(key) and _secret_value(item) for key, item in value.items()):
                raise ValueError("Resolved config contains credentials")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            if _CREDENTIAL_URI.search(value.encode()) or _PRIVATE_KEY_HEADER.search(value.encode()):
                raise ValueError("Resolved config contains credentials")
            if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
                raise ValueError("Resolved config contains an unredacted absolute path")


def _git_observation(root: Path, *args: str, hash_output: bool = False) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "value": None, "reason": type(exc).__name__}
    if result.returncode != 0:
        return {
            "status": "unavailable",
            "value": None,
            "reason": "git command failed",
            "returncode": result.returncode,
        }
    value = (
        hashlib.sha256(result.stdout).hexdigest()
        if hash_output
        else result.stdout.decode("utf-8", errors="replace").strip()
    )
    return {"status": "observed", "value": value}
