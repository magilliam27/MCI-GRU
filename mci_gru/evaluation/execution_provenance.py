"""Capture an execution-start snapshot and inspect it without live observations.

This module does not start training, collect data inputs, or declare completion.
Callers retain the returned digest alongside the artifact as its trust anchor.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
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

SCHEMA = "mci_gru.execution_start.v1"
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
) -> ExecutionReference:
    """Retain source, existing redacted config and observations before preparation.

    Each call creates a distinct incomplete attempt. No existing artifact or
    source file is overwritten. The source scope is the training entry point,
    package, configuration and dependency declarations, including untracked code.
    """
    if window_id is not None and (not isinstance(window_id, str) or not window_id):
        raise ValueError("Invalid window identifier")
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


def _validate_source_path(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or name.split("/")[0] not in SOURCE_ROOTS
    ):
        raise ValueError("Unsafe source path")


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
    return value not in (None, "", "<REDACTED>", "[REDACTED]", "null", "None", "~")


def _contains_credentials(content: bytes, suffix: str = "") -> bool:
    if _CREDENTIAL_URI.search(content) or b"PRIVATE KEY-----" in content:
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
        for match in _CREDENTIAL_ASSIGNMENT.finditer(content)
    )


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
            if _CREDENTIAL_URI.search(value.encode()) or "PRIVATE KEY-----" in value:
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
