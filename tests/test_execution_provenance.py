"""Execution evidence remains inspectable without the live checkout or runtime."""

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata

import pytest

from mci_gru.config import create_config_from_dict
from mci_gru.evaluation.artifacts import canonical_json_bytes
from mci_gru.evaluation.execution_provenance import (
    capture_execution_start,
    read_execution_provenance,
)
from mci_gru.evaluation.experiment_summary import write_resolved_config


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mci_gru").mkdir()
    (repo / "run_experiment.py").write_bytes(b"SEED = 3\n")
    for args in [
        ("init",),
        ("add", "."),
        (
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ),
    ]:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    (repo / "run_experiment.py").write_bytes(b"SEED = 73\n")
    (repo / "mci_gru" / "new_model.py").write_bytes(b"WIDTH = 17\r\n")
    config = create_config_from_dict({"seed": 73, "data": {"filename": str(repo / "prices.csv")}})
    config_dir = tmp_path / "config"
    identity = write_resolved_config(config, config_dir)
    config_path = config_dir / identity["resolved_config_path"]
    return repo, config, config_path, identity["resolved_config_sha256"], commit


def test_retained_readback_survives_live_source_config_and_environment_changes(
    tmp_path, monkeypatch
):
    repo, config, config_path, config_digest, commit = _fixture(tmp_path)
    original_config = config_path.read_bytes()
    original_python = ".".join(map(str, sys.version_info[:3]))
    original_torch = metadata.version("torch")
    before = datetime.now(timezone.utc)
    reference = capture_execution_start(
        repo,
        config_path,
        resolved_config_sha256=config_digest,
        output_dir=tmp_path / "evidence",
        window_id="w000",
    )
    after = datetime.now(timezone.utc)
    (repo / "run_experiment.py").write_bytes(b"SEED = 999\n")
    (repo / "mci_gru" / "new_model.py").unlink()
    config.seed = 999
    config_path.write_text('{"seed":999}', encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise AssertionError("Read-back must not observe the live runtime")

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(metadata, "version", unavailable)
    retained = read_execution_provenance(reference)

    assert retained.source_files == {
        "run_experiment.py": b"SEED = 73\n",
        "mci_gru/new_model.py": b"WIDTH = 17\r\n",
    }
    assert retained.resolved_config_bytes == original_config
    assert json.loads(retained.resolved_config_bytes)["seed"] == 73
    assert json.loads(retained.resolved_config_bytes)["data"]["filename"] == "<ABSOLUTE_PATH>"
    assert retained.record["resolved_config"]["sha256"] == config_digest
    assert hashlib.sha256(retained.resolved_config_bytes).hexdigest() == config_digest
    assert retained.record["code_identity"]["git_commit"] == commit
    assert retained.record["code_identity"]["git_dirty"] is True
    assert retained.record["environment"]["runtime_versions"]["python"]["value"] == original_python
    assert retained.record["environment"]["runtime_versions"]["torch"]["value"] == original_torch
    assert before <= datetime.fromisoformat(retained.record["captured_at_utc"]) <= after
    assert retained.record["window_id"] == "w000"
    assert retained.record["execution_status"] == "incomplete"


def test_source_scope_preserves_training_code_and_excludes_credentials(tmp_path):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    (repo / ".env").write_text("FRED_API_KEY=fixture-credential", encoding="utf-8")
    (repo / "mci_gru" / "credentials.py").write_text(
        'TOKEN = "fixture-credential"', encoding="utf-8"
    )
    (repo / "mci_gru" / "model.py").write_text(
        'API_KEY = "fixture-inline-secret"', encoding="utf-8"
    )
    (repo / "configs").mkdir()
    (repo / "configs" / "train.yaml").write_bytes(b"seed: 73\n")
    (repo / "mci_gru" / "pipeline.py").write_bytes(b"LABEL_HORIZON = 5\n")
    reference = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(reference)
    assert retained.source_files["configs/train.yaml"] == b"seed: 73\n"
    assert retained.source_files["mci_gru/pipeline.py"] == b"LABEL_HORIZON = 5\n"
    assert "mci_gru/credentials.py" not in retained.source_files
    assert "mci_gru/model.py" not in retained.source_files
    assert all(
        b"fixture-credential" not in content and b"fixture-inline-secret" not in content
        for content in retained.source_files.values()
    )
    assert retained.record["source_capture"]["status"] == "partial"
    excluded = {row["path"] for row in retained.record["source_capture"]["problems"]}
    assert {"mci_gru/credentials.py", "mci_gru/model.py"} <= excluded


@pytest.mark.parametrize("damage", ["missing", "changed", "unsupported"])
def test_reader_rejects_missing_changed_or_unsupported_evidence(tmp_path, damage):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    reference = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    if damage == "missing":
        reference.path.unlink()
    elif damage == "changed":
        reference.path.write_bytes(reference.path.read_bytes() + b" ")
    else:
        record = json.loads(reference.path.read_bytes())
        record["schema"] = "mci_gru.execution_start.v999"
        payload = json.dumps(record).encode()
        reference.path.write_bytes(payload)
        reference = replace(reference, sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="missing|digest|Unsupported"):
        read_execution_provenance(reference)


def test_capture_rejects_changed_config_without_modifying_sources_or_prior_attempt(tmp_path):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    output = tmp_path / "evidence"
    reference = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=output
    )
    original = reference.path.read_bytes()
    config_path.write_bytes(b'{"seed":999}')
    with pytest.raises(ValueError, match="config.*digest"):
        capture_execution_start(repo, config_path, resolved_config_sha256=digest, output_dir=output)
    assert config_path.read_bytes() == b'{"seed":999}'
    assert (repo / "run_experiment.py").read_bytes() == b"SEED = 73\n"
    assert reference.path.read_bytes() == original
    assert len(list(output.rglob("*.json"))) == 1


@pytest.mark.parametrize(
    "payload",
    [
        b'{"tracking":{"password":"fixture-secret"}}',
        b'{"fred_api_key":"fixture-secret"}',
        rb'{"pass\u0077ord":"fixture-secret"}',
        b'{"tracking":{"uri":"https://user:fixture-secret@server/"}}',
        b'{"data":{"filename":"/private/market.csv"}}',
        b'{"seed":NaN}',
        b"[]",
    ],
)
def test_capture_refuses_unsafe_or_invalid_resolved_config(tmp_path, payload):
    repo, _, config_path, _, _ = _fixture(tmp_path)
    config_path.write_bytes(payload)
    with pytest.raises(ValueError, match="config"):
        capture_execution_start(
            repo,
            config_path,
            resolved_config_sha256=hashlib.sha256(payload).hexdigest(),
            output_dir=tmp_path / "evidence",
        )
    assert config_path.read_bytes() == payload
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("git_failure", ["absent", "nonzero"])
def test_unavailable_observations_are_retained_as_unknown_not_clean(
    tmp_path, monkeypatch, git_failure
):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    original_version = metadata.version

    def unavailable_git(args, **kwargs):
        if git_failure == "absent":
            raise FileNotFoundError("git unavailable")
        return subprocess.CompletedProcess(args, 128, b"", b"fatal: fixture unavailable")

    def missing_torch(package):
        if package == "torch":
            raise metadata.PackageNotFoundError(package)
        return original_version(package)

    monkeypatch.setattr(subprocess, "run", unavailable_git)
    monkeypatch.setattr(metadata, "version", missing_torch)
    reference = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(reference)
    assert retained.record["code_identity"]["git_commit"] is None
    assert retained.record["code_identity"]["git_dirty"] is None
    assert retained.record["code_identity"]["git_observations"]["commit"]["status"] == "unavailable"
    assert retained.record["code_identity"]["git_observations"]["commit"]["reason"]
    assert retained.record["environment"]["runtime_versions"]["torch"] == {
        "status": "unavailable",
        "value": None,
        "reason": "PackageNotFoundError",
    }
    assert retained.record["environment"]["colab_runtime_label"]["status"] == "unknown"
    assert retained.source_files["run_experiment.py"] == b"SEED = 73\n"
    assert retained.record["execution_status"] == "incomplete"


def test_unreadable_source_is_explicit_partial_evidence_and_not_a_silent_omission(
    tmp_path, monkeypatch
):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    denied = repo / "mci_gru" / "new_model.py"
    path_type = type(denied)
    original_read = path_type.read_bytes

    def read_except_denied(path):
        if path == denied:
            raise PermissionError("sensitive error message must not be retained")
        return original_read(path)

    monkeypatch.setattr(path_type, "read_bytes", read_except_denied)
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(ref)
    assert "mci_gru/new_model.py" not in retained.source_files
    assert retained.record["source_capture"]["status"] == "partial"
    assert {"path": "mci_gru/new_model.py", "reason": "PermissionError"} in retained.record[
        "source_capture"
    ]["problems"]
    assert b"sensitive error message" not in ref.path.read_bytes()


@pytest.mark.parametrize(
    "damage",
    [
        "blob",
        "blob_digest",
        "size",
        "source_inventory",
        "source_digest",
        "path",
        "shape",
        "status",
        "timestamp",
    ],
)
def test_reader_rejects_invalid_record_semantics_even_with_a_matching_outer_digest(
    tmp_path, damage
):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    reference = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    record = json.loads(reference.path.read_bytes())
    if damage == "blob":
        record["resolved_config"]["content_base64"] = "e30="
    elif damage == "size":
        record["resolved_config"]["size_bytes"] += 1
    elif damage == "blob_digest":
        record["resolved_config"]["sha256"] = "0" * 64
    elif damage == "source_digest":
        record["code_identity"]["working_tree_source_sha256"] = "0" * 64
    elif damage == "source_inventory":
        del record["source_files"]["mci_gru/new_model.py"]
    elif damage == "path":
        record["source_files"]["../escape.py"] = record["source_files"].pop("run_experiment.py")
        hashes = {name: blob["sha256"] for name, blob in record["source_files"].items()}
        record["code_identity"]["source_hashes"] = hashes
        record["code_identity"]["working_tree_source_sha256"] = hashlib.sha256(
            canonical_json_bytes(hashes)
        ).hexdigest()
    elif damage == "shape":
        record = []
    elif damage == "status":
        record["execution_status"] = "completed"
    else:
        record["captured_at_utc"] = "yesterday"
    payload = json.dumps(record).encode()
    reference.path.write_bytes(payload)
    reference = replace(reference, sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="Invalid"):
        read_execution_provenance(reference)


@pytest.mark.parametrize("real_link", [False, True])
def test_capture_does_not_follow_source_links_outside_the_declared_tree(
    tmp_path, monkeypatch, real_link
):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    external = tmp_path / "outside.py"
    external.write_bytes(b"PRIVATE_SOURCE = 17\n")
    link = repo / "mci_gru" / "linked.py"
    if real_link:
        try:
            link.symlink_to(external)
        except OSError as exc:
            pytest.skip(f"Host cannot create a symlink: {type(exc).__name__}")
    else:
        link.write_bytes(external.read_bytes())
        original_is_symlink = type(link).is_symlink
        monkeypatch.setattr(
            type(link), "is_symlink", lambda path: path == link or original_is_symlink(path)
        )
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(ref)
    assert "mci_gru/linked.py" not in retained.source_files
    assert retained.record["source_capture"]["status"] == "partial"
    assert any(
        p["path"] == "mci_gru/linked.py" for p in retained.record["source_capture"]["problems"]
    )
    assert external.read_bytes() == b"PRIVATE_SOURCE = 17\n"


def test_missing_training_root_is_not_reported_as_a_complete_source_snapshot(tmp_path):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    (repo / "run_experiment.py").unlink()
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(ref)
    assert retained.record["source_capture"]["status"] == "partial"
    assert {
        "path": "run_experiment.py",
        "reason": "required source root missing",
    } in retained.record["source_capture"]["problems"]


def test_package_observation_errors_are_explicit_and_labels_do_not_shift(tmp_path, monkeypatch):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    observed = {
        "numpy": "1.25.1",
        "pandas": "2.1.2",
        "scipy": "1.12.3",
        "torch": "2.7.4",
        "torch-geometric": "2.5.6",
    }

    def version(package):
        if package == "scipy":
            raise ValueError("broken metadata with fixture-secret")
        return observed[package]

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(platform, "platform", lambda: "fixture-platform")
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    environment = read_execution_provenance(ref).record["environment"]
    for package, value in observed.items():
        if package == "scipy":
            assert environment["runtime_versions"][package] == {
                "status": "unavailable",
                "value": None,
                "reason": "ValueError",
            }
        else:
            assert environment["runtime_versions"][package] == {
                "status": "observed",
                "value": value,
            }
    assert environment["platform"] == "fixture-platform"
    assert environment["python_implementation"] == sys.implementation.name
    assert b"fixture-secret" not in ref.path.read_bytes()


def test_each_capture_is_a_distinct_incomplete_attempt_and_readback_does_not_write(tmp_path):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    refs = [
        capture_execution_start(
            repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
        )
        for _ in range(2)
    ]
    assert refs[0].path != refs[1].path
    first_bytes = refs[0].path.read_bytes()
    first_mtime = refs[0].path.stat().st_mtime_ns
    first = read_execution_provenance(refs[0])
    second = read_execution_provenance(refs[1])
    assert first.record["attempt_id"] != second.record["attempt_id"]
    assert first.record["source_capture"]["status"] == "complete"
    assert first.record["execution_status"] == second.record["execution_status"] == "incomplete"
    first.record["execution_status"] = "completed"
    assert read_execution_provenance(refs[0]).record["execution_status"] == "incomplete"
    assert refs[0].path.read_bytes() == first_bytes
    assert refs[0].path.stat().st_mtime_ns == first_mtime


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("provider.py", b'FRED_API_KEY = "fixture-secret"'),
        ("provider.py", rb'config = {"pass\u0077ord": "fixture-secret"}'),
        ("provider.py", b'client = connect(client_secret="fixture-secret")'),
        ("provider.py", b'os.environ["FRED_API_KEY"] = "fixture-secret"'),
        ("provider.py", b'settings["password"] = "fixture-secret"'),
        ("provider.yaml", b"password: fixture-secret"),
        ("provider.toml", b'fred_api_key = "fixture-secret"'),
    ],
)
def test_source_credential_formats_are_excluded_with_explicit_partial_evidence(
    tmp_path, name, content
):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    (repo / "mci_gru" / name).write_bytes(content)
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(ref)
    assert f"mci_gru/{name}" not in retained.source_files
    assert retained.record["source_capture"]["status"] == "partial"
    assert b"fixture-secret" not in b"".join(retained.source_files.values())


def test_runtime_credential_lookup_and_redacted_config_are_preserved(tmp_path):
    repo, _, config_path, _, _ = _fixture(tmp_path)
    content = b'import os\nFRED_API_KEY = os.getenv("FRED_API_KEY")\n'
    (repo / "mci_gru" / "provider.py").write_bytes(content)
    config_bytes = b'{"password":"<REDACTED>","fred_api_key":null}'
    config_path.write_bytes(config_bytes)
    ref = capture_execution_start(
        repo,
        config_path,
        resolved_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        output_dir=tmp_path / "evidence",
    )
    retained = read_execution_provenance(ref)
    assert retained.source_files["mci_gru/provider.py"] == content
    assert retained.resolved_config_bytes == config_bytes


def test_platform_observation_failure_still_retains_an_incomplete_attempt(tmp_path, monkeypatch):
    repo, _, config_path, digest, _ = _fixture(tmp_path)

    def unavailable():
        raise subprocess.SubprocessError("sensitive observation error")

    monkeypatch.setattr(platform, "platform", unavailable)
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(ref)
    assert retained.record["environment"]["platform"] is None
    assert retained.record["environment"]["platform_observation"] == {
        "status": "unavailable",
        "value": None,
        "reason": "SubprocessError",
    }
    assert retained.record["execution_status"] == "incomplete"
    assert b"sensitive observation error" not in ref.path.read_bytes()


def test_code_identity_keeps_existing_v1_fields_with_execution_time_observations(tmp_path):
    repo, _, config_path, digest, commit = _fixture(tmp_path)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    retained = read_execution_provenance(ref)
    identity = retained.record["code_identity"]
    assert identity["schema"] == "mci_gru.selection_research_code_identity.v1"
    assert identity["git_commit"] == commit
    assert identity["git_dirty_diff_sha256"] == hashlib.sha256(diff).hexdigest()
    assert identity["runtime_versions"] == {
        name: observation["value"]
        for name, observation in retained.record["environment"]["runtime_versions"].items()
    }
    assert identity["git_observations"]["status"] == {"status": "observed", "value": True}
    assert identity["git_observations"]["diff"]["value"] == identity["git_dirty_diff_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("attempt_id", None, True),
        ("window_id", None, True),
        ("environment", None, True),
        ("source_capture", None, True),
        ("code_identity.git_observations", None, True),
        ("attempt_id", "", False),
        ("window_id", 17, False),
        ("source_capture.roots", [], False),
        ("source_capture.status", "partial", False),
        ("source_capture.problems", [{"path": "../escape.py", "reason": "failed"}], False),
        ("code_identity.schema", "future", False),
        ("code_identity.git_observations.commit.status", "invented", False),
        ("code_identity.git_observations.commit.value", None, False),
        ("code_identity.git_observations.status.value", "true", False),
        ("code_identity.git_observations.diff.value", "bad", False),
        ("code_identity.git_dirty", False, False),
        ("code_identity.git_dirty_diff_sha256", None, False),
        ("environment.runtime_versions.torch", None, True),
        ("environment.runtime_versions.torch", {"status": "unavailable", "value": "2.7"}, False),
        ("environment.runtime_versions.torch", {"status": "unavailable", "value": None}, False),
        ("environment.platform", "changed", False),
        ("environment.python_implementation", "", False),
        ("environment.colab_runtime_label", {"status": "observed", "value": "G4"}, False),
        ("code_identity.runtime_versions", {}, False),
    ],
)
def test_reader_requires_valid_attempt_and_observation_metadata(tmp_path, field, value, remove):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    ref = capture_execution_start(
        repo, config_path, resolved_config_sha256=digest, output_dir=tmp_path / "evidence"
    )
    record = json.loads(ref.path.read_bytes())
    target = record
    *parents, key = field.split(".")
    for parent in parents:
        target = target[parent]
    if remove:
        del target[key]
    else:
        target[key] = value
    payload = json.dumps(record).encode()
    ref.path.write_bytes(payload)
    ref = replace(ref, sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="Invalid"):
        read_execution_provenance(ref)


@pytest.mark.parametrize("window_id", ["", 17])
def test_invalid_window_identifier_is_rejected_before_output(tmp_path, window_id):
    repo, _, config_path, digest, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="window identifier"):
        capture_execution_start(
            repo,
            config_path,
            resolved_config_sha256=digest,
            output_dir=tmp_path / "evidence",
            window_id=window_id,
        )
    assert not (tmp_path / "evidence").exists()
