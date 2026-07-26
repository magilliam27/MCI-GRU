from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mci_gru.evaluation.artifacts import write_json_artifact

CORE_ARTIFACTS = [
    "run_metadata.json",
    "training_summary.json",
    "evaluation_summary.json",
]

CONFIG_CANDIDATES = [
    "config.yaml",
    "config.yml",
    "config.json",
    "resolved_config.json",
    "resolved_config.yaml",
    ".hydra/config.yaml",
]
DATA_FINGERPRINT_CANDIDATES = [
    "pit_universe.csv",
    "pit_universe.parquet",
    "market_data.csv",
    "stock_data.csv",
    "processed_data.csv",
    "processed_data.parquet",
]
NORMALIZATION_CANDIDATES = [
    "normalization_stats.json",
    "normalization_reference.json",
    "normalizer.json",
    "scaler.pkl",
]
GRAPH_CANDIDATES = [
    "graph_data.pt",
    "graph_metadata.json",
    "graph_policy.json",
]
CHECKPOINT_PATTERNS = ["*.pt", "*.pth", "*.ckpt"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path, pattern: str = "*.csv") -> dict[str, Any]:
    root = Path(path)
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    return _describe_file_collection(root, files)


def describe_artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not path.exists():
        return {"exists": False, "path": str(resolved)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def build_run_manifest(
    run_dir: str | Path,
    *,
    selection_rule: str | None = None,
    sibling_trial_ids: list[str] | None = None,
    command: str | None = None,
    feature_lag_policy: str | None = None,
    normalization_reference: str | None = None,
    graph_policy: str | None = None,
    mlflow_run_id: str | None = None,
    seed_policy: str | None = None,
    paper_trade_eligible: bool | None = None,
    repo_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    predictions_dir = root / "averaged_predictions"
    metadata = _load_json(root / "run_metadata.json")
    config = _config_summary(root, metadata)
    graph = _graph_summary(root, graph_policy, metadata)
    checkpoints = _describe_artifact_set(root, CHECKPOINT_PATTERNS, exclude_names={"graph_data.pt"})
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root.resolve()),
        "selection_rule": selection_rule,
        "sibling_trial_ids": sibling_trial_ids or [],
        "provenance": {
            "command": command or _metadata_lookup(metadata, "command", "run_command", "argv"),
            "git": _git_state(Path(repo_dir) if repo_dir is not None else root, metadata=metadata),
            "environment": _environment_summary(),
        },
        "config": config,
        "data_fingerprints": {
            "artifacts": _describe_candidates(root, DATA_FINGERPRINT_CANDIDATES),
            "pit_file_hash": _first_existing_hash(
                root, ["pit_universe.csv", "pit_universe.parquet"]
            ),
            "metadata_data_source": _metadata_lookup(metadata, "data_source", "source", "data"),
        },
        "feature_lag_policy": feature_lag_policy
        or _metadata_lookup(metadata, "feature_lag_policy", "feature_policy"),
        "normalization_reference": {
            "declared": normalization_reference,
            "artifacts": _describe_candidates(root, NORMALIZATION_CANDIDATES),
        },
        "graph": graph,
        "checkpoints": checkpoints,
        "mlflow_run_id": mlflow_run_id or _metadata_lookup(metadata, "mlflow_run_id", "run_id"),
        "seed_policy": seed_policy or _metadata_lookup(metadata, "seed_policy", "seed", "seeds"),
        "paper_trade_eligible": bool(paper_trade_eligible)
        if paper_trade_eligible is not None
        else False,
        "paper_trade_eligibility_declared": paper_trade_eligible is not None,
        "paper_trade_eligibility_inputs": {
            "has_checkpoint": checkpoints["file_count"] > 0,
            "has_frozen_graph": graph["artifact"]["exists"],
            "has_config": config["artifact"]["exists"]
            or config["metadata_config_sha256"] is not None,
        },
        "artifacts": {name: describe_artifact(root / name) for name in CORE_ARTIFACTS},
        "prediction_artifact": (
            sha256_directory(predictions_dir)
            if predictions_dir.exists()
            else {"exists": False, "path": str(predictions_dir.resolve())}
        ),
    }


def validate_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    missing = [name for name in CORE_ARTIFACTS if not (root / name).is_file()]
    predictions_dir = root / "averaged_predictions"
    if not predictions_dir.is_dir():
        missing.append("averaged_predictions")
    elif not any(predictions_dir.glob("*.csv")):
        missing.append("averaged_predictions/*.csv")
    return {
        "schema_version": 1,
        "run_dir": str(root.resolve()),
        "status": "OK" if not missing else "FAILED",
        "missing_artifacts": missing,
    }


def write_run_manifest(
    run_dir: str | Path,
    *,
    selection_rule: str | None = None,
    sibling_trial_ids: list[str] | None = None,
    command: str | None = None,
    feature_lag_policy: str | None = None,
    normalization_reference: str | None = None,
    graph_policy: str | None = None,
    mlflow_run_id: str | None = None,
    seed_policy: str | None = None,
    paper_trade_eligible: bool | None = None,
    repo_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    root = Path(run_dir)
    manifest = build_run_manifest(
        root,
        selection_rule=selection_rule,
        sibling_trial_ids=sibling_trial_ids,
        command=command,
        feature_lag_policy=feature_lag_policy,
        normalization_reference=normalization_reference,
        graph_policy=graph_policy,
        mlflow_run_id=mlflow_run_id,
        seed_policy=seed_policy,
        paper_trade_eligible=paper_trade_eligible,
        repo_dir=repo_dir,
    )
    validation = validate_run_bundle(root)
    manifest_path = root / "run_manifest.json"
    validation_path = root / "artifact_validation.json"
    if not force:
        for path in (manifest_path, validation_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    write_json_artifact(manifest_path, manifest, force=force)
    write_json_artifact(validation_path, validation, force=force)
    return {"manifest": manifest_path, "validation": validation_path}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _metadata_lookup(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    config = metadata.get("config")
    if isinstance(config, dict):
        for key in keys:
            if key in config:
                return config[key]
    return None


def _config_summary(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    artifact = _first_existing_artifact(root, CONFIG_CANDIDATES)
    metadata_config = metadata.get("config")
    return {
        "artifact": artifact,
        "metadata_config_sha256": _stable_json_hash(metadata_config)
        if metadata_config is not None
        else None,
    }


def _graph_summary(
    root: Path, graph_policy: str | None, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "policy": graph_policy or _metadata_lookup(metadata, "graph_policy", "graph"),
        "artifact": _first_existing_artifact(root, GRAPH_CANDIDATES),
        "artifacts": _describe_candidates(root, GRAPH_CANDIDATES),
    }


def _describe_candidates(root: Path, candidates: list[str]) -> dict[str, Any]:
    return {name: describe_artifact(root / name) for name in candidates}


def _first_existing_artifact(root: Path, candidates: list[str]) -> dict[str, Any]:
    for name in candidates:
        artifact = describe_artifact(root / name)
        if artifact["exists"]:
            artifact["name"] = name
            return artifact
    return {"exists": False, "candidates": candidates}


def _first_existing_hash(root: Path, candidates: list[str]) -> str | None:
    for name in candidates:
        path = root / name
        if path.is_file():
            return sha256_file(path)
    return None


def _describe_artifact_set(
    root: Path,
    patterns: list[str],
    *,
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude_names or set()
    files: list[Path] = []
    for pattern in patterns:
        files.extend(
            path for path in root.rglob(pattern) if path.is_file() and path.name not in excluded
        )
    unique_files = sorted(set(files))
    return _describe_file_collection(root, unique_files)


def _describe_file_collection(root: Path, files: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    entries = []
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        file_hash = sha256_file(file_path)
        entries.append(
            {
                "path": rel,
                "sha256": file_hash,
                "size_bytes": file_path.stat().st_size,
            }
        )
        digest.update(rel.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(path: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    git_metadata = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status_short = run_git("status", "--short")
    commit = run_git("rev-parse", "HEAD") or git_metadata.get("commit")
    branch = run_git("branch", "--show-current") or git_metadata.get("branch")
    dirty = bool(status_short) if status_short is not None else git_metadata.get("dirty")
    return {
        "repo_dir": str(path.resolve()),
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty) if dirty is not None else None,
        "status_short": status_short
        if status_short is not None
        else git_metadata.get("status_short"),
    }


def _environment_summary() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
