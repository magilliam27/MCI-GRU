"""Behavioral checks at the approved input-manifest reader boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mci_gru.data.input_manifest import (
    InputFileSpec,
    InputManifestError,
    read_input_manifest,
    validate_input_package,
    write_input_manifest,
)

RAW_MANIFEST = (
    b'{ "schema_version": 1, "package_id": "tiny", "package_revision": "r1",\n'
    b'  "files": [{"path": "observations.bin", "purpose": "source observations",'
    b' "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",'
    b' "size_bytes": 3}],\n'
    b'  "provenance": {"source": "synthetic", "acquisition_mode": "fixture",'
    b' "acquired_at": null, "producing_command": null, "producing_arguments": null,'
    b' "unknowns": ["No historical acquisition"]}, "required_extensions": [] }\n'
)
MANIFEST_SHA256 = "b0a2b3ce4a03d64eb8515c3365fee86d1a318e1abda6a38acc9e16551c142846"


def test_selected_manifest_matches_retained_ten_file_historical_inventory() -> None:
    # Expected identities are transcribed from the independently retained historical
    # MANIFEST.txt (SHA-256 2a35394d...e502), not recomputed from this manifest.
    prefix = "sp500_pit_gics_top10_mcap_monthly_20160104_20260731"
    expected = {
        f"constituents/{prefix}_all_metadata_snapshots.csv": (
            "31673ec69e5306e6d923623dc857c5be9dda805eabd7643648dbaacca90cec90",
            5034067,
        ),
        f"constituents/{prefix}_changes.csv": (
            "142a062e6cb4f00c12403f821e6b6756a888cc3e7e191ec0b9b54f7c0ef74e00",
            19832,
        ),
        f"constituents/{prefix}_current_members.csv": (
            "f5e6b150a8d33e7f11c772bd65e73ef7c6a3c602e1d73b6e7472a8fea12e5b91",
            20819,
        ),
        f"constituents/{prefix}_meta.json": (
            "710ad681209f29d076f7f5235e2366d3f8c49e391ca98464acf4fc0c268a2663",
            77461,
        ),
        f"constituents/{prefix}_pit_universe.csv": (
            "15721bb3c8d17a16901c02d352799cfa7b38bbb65b6f5a7e362c7d6424703200",
            16058,
        ),
        f"constituents/{prefix}_snapshots.csv": (
            "995414e997dd2c3dd251fe5d07d37ff563efabdbcde6f4108a91c0eab0fb8b84",
            1267167,
        ),
        f"constituents/{prefix}_sp500_membership_intervals.csv": (
            "5d7f819183a0e1e84e79a79e58ddb1e26fbbe6f732d2be59521c2e3959d4a5f6",
            23006,
        ),
        f"market/{prefix}_lseg_20150101_20260731.csv": (
            "d64c4d041ef4c1632ed76e1456885ffe8301a477c8e27c4a73805f94ff97aeb4",
            40035626,
        ),
        f"market/{prefix}_lseg_20150101_20260731.meta.json": (
            "3f8f97fb6ec221ccc50525f71af7d0d9634378b96c493538ec786c61feda1c96",
            943,
        ),
        f"market/{prefix}_lseg_20150101_20260731_coverage.csv": (
            "05f2072b77fdac5a0de18bf989d6ca71adf0a94a9232c36072b5cf9b0ef06cb5",
            7140,
        ),
    }
    path = Path(__file__).resolve().parents[1] / "data" / "manifests" / f"{prefix}.r1.json"
    snapshot = read_input_manifest(
        path, expected_sha256="1e2043dcb4f00a88de128985cc94423c122e480034c16ec741710a2460653ecf"
    )

    assert snapshot.manifest.package_id == prefix
    assert snapshot.manifest.package_revision == "r1"
    assert {
        item.path: (item.sha256, item.size_bytes) for item in snapshot.manifest.files
    } == expected
    assert sum(item.size_bytes for item in snapshot.manifest.files) == 46502119
    assert snapshot.manifest.provenance["producing_command"] is None
    assert snapshot.manifest.provenance["acquisition_mode"] is None
    assert snapshot.manifest.provenance["unknowns"]


def _make_sources(root: Path, raw_bytes: bytes = RAW_MANIFEST) -> Path:
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(raw_bytes)
    (root / "observations.bin").write_bytes(b"abc")
    # An old timestamp makes even an identical-byte rewrite observable.
    os.utime(manifest_path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    return manifest_path


def _source_state(root: Path) -> dict[str, tuple[bytes, int, int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_mode,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("expected_sha256", [None, MANIFEST_SHA256])
def test_reader_retains_exact_manifest_and_leaves_sources_unchanged(
    tmp_path: Path, expected_sha256: str | None
) -> None:
    manifest_path = _make_sources(tmp_path)
    before = _source_state(tmp_path)

    snapshot = read_input_manifest(manifest_path, expected_sha256=expected_sha256)

    assert snapshot.raw_bytes == RAW_MANIFEST
    assert snapshot.sha256 == MANIFEST_SHA256
    assert snapshot.size_bytes == 463
    assert snapshot.manifest.package_id == "tiny"
    assert snapshot.manifest.package_revision == "r1"
    record = snapshot.manifest.files[0]
    assert (record.path, record.purpose, record.sha256, record.size_bytes) == (
        "observations.bin",
        "source observations",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        3,
    )
    assert _source_state(tmp_path) == before


def test_reader_rejects_digest_mismatch_without_changing_sources(tmp_path: Path) -> None:
    manifest_path = _make_sources(tmp_path)
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="SHA-256 mismatch") as failure:
            read_input_manifest(manifest_path, expected_sha256="0" * 64)

        assert failure.value.expected_sha256 == "0" * 64
        assert failure.value.observed_sha256 == MANIFEST_SHA256
        assert str(manifest_path) in str(failure.value)
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize("raw_bytes", [b"{broken", b"\xff", b"[]", b"null"])
def test_reader_rejects_invalid_documents_without_changing_sources(
    tmp_path: Path, raw_bytes: bytes
) -> None:
    manifest_path = _make_sources(tmp_path, raw_bytes)
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="Invalid manifest"):
            read_input_manifest(manifest_path)
    finally:
        assert _source_state(tmp_path) == before


def test_package_verification_returns_the_exact_source_binding(tmp_path: Path) -> None:
    manifest_path = _make_sources(tmp_path)
    snapshot = read_input_manifest(manifest_path)
    before = _source_state(tmp_path)

    verified = validate_input_package(snapshot, tmp_path, expected_paths=["observations.bin"])

    assert verified.snapshot == snapshot
    assert len(verified.files) == 1
    observation = verified.files[0]
    assert observation.resolved_path == (tmp_path / "observations.bin").resolve()
    assert observation.record == snapshot.manifest.files[0]
    assert observation.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert observation.size_bytes == 3
    assert _source_state(tmp_path) == before


@pytest.mark.parametrize("altered_bytes", [b"abd", b"abc\n"])
def test_package_verification_rejects_altered_bytes_without_repair(
    tmp_path: Path, altered_bytes: bytes
) -> None:
    manifest_path = _make_sources(tmp_path)
    snapshot = read_input_manifest(manifest_path)
    (tmp_path / "observations.bin").write_bytes(altered_bytes)
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="File identity mismatch") as failure:
            validate_input_package(snapshot, tmp_path, expected_paths=["observations.bin"])

        assert failure.value.expected_sha256 == snapshot.manifest.files[0].sha256
        assert failure.value.observed_sha256 != failure.value.expected_sha256
        assert failure.value.expected_size_bytes == 3
        assert failure.value.observed_size_bytes == len(altered_bytes)
        assert "observations.bin" in str(failure.value)
    finally:
        assert _source_state(tmp_path) == before


def test_package_verification_checks_size_even_when_hash_matches(tmp_path: Path) -> None:
    manifest_path = _make_sources(
        tmp_path, RAW_MANIFEST.replace(b'"size_bytes": 3', b'"size_bytes": 4')
    )
    snapshot = read_input_manifest(manifest_path)
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="File identity mismatch") as failure:
            validate_input_package(snapshot, tmp_path, expected_paths=["observations.bin"])

        assert failure.value.expected_sha256 == failure.value.observed_sha256
        assert (failure.value.expected_size_bytes, failure.value.observed_size_bytes) == (4, 3)
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize(
    "expected_paths", [[], ["observations.bin", "omitted.bin"], ["observations.bin"] * 2]
)
def test_package_requires_an_independent_complete_unique_inventory(
    tmp_path: Path, expected_paths: list[str]
) -> None:
    manifest_path = _make_sources(tmp_path)
    snapshot = read_input_manifest(manifest_path)
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="inventory"):
            validate_input_package(snapshot, tmp_path, expected_paths=expected_paths)
    finally:
        assert _source_state(tmp_path) == before


def test_package_rejects_duplicate_declarations_without_changing_sources(tmp_path: Path) -> None:
    payload = json.loads(RAW_MANIFEST)
    payload["files"] *= 2
    manifest_path = _make_sources(tmp_path, json.dumps(payload).encode())
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="[Dd]uplicate"):
            snapshot = read_input_manifest(manifest_path)
            validate_input_package(snapshot, tmp_path, expected_paths=["observations.bin"])
    finally:
        assert _source_state(tmp_path) == before


def test_missing_source_is_reported_without_basename_fallback_or_repair(tmp_path: Path) -> None:
    manifest_path = _make_sources(tmp_path)
    snapshot = read_input_manifest(manifest_path)
    (tmp_path / "observations.bin").unlink()
    (tmp_path / "fallback").mkdir()
    (tmp_path / "fallback" / "observations.bin").write_bytes(b"abc")
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="Missing source.*observations.bin"):
            validate_input_package(snapshot, tmp_path, expected_paths=["observations.bin"])
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        "/outside.bin",
        "../outside.bin",
        "a/../outside.bin",
        "a//b",
        "./a",
        "C:/outside.bin",
        "C:outside.bin",
        "a\\b",
        "a/",
        "a\x00b",
        "a:b",
        "CON.txt",
        "COM¹.txt",
        "LPT²",
        "NUL .txt",
        "CONIN$",
        "CONOUT$",
        "a/NUL",
        "a. /b",
        "a./b",
        "a /b",
    ],
)
def test_reader_rejects_unsafe_package_paths(tmp_path: Path, unsafe_path: str) -> None:
    payload = json.loads(RAW_MANIFEST)
    payload["files"][0]["path"] = unsafe_path
    manifest_path = _make_sources(tmp_path, json.dumps(payload).encode())
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="[Uu]nsafe.*path"):
            read_input_manifest(manifest_path)
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize("other_path", ["OBSERVATIONS.bin", "observations.bin/child"])
def test_reader_rejects_colliding_destinations(tmp_path: Path, other_path: str) -> None:
    payload = json.loads(RAW_MANIFEST)
    payload["files"].append({**payload["files"][0], "path": other_path})
    manifest_path = _make_sources(tmp_path, json.dumps(payload).encode())
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="[Cc]olliding|[Dd]uplicate"):
            read_input_manifest(manifest_path)
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize("outside_root", [False, True])
def test_package_links_must_resolve_inside_explicit_root(
    tmp_path: Path, outside_root: bool
) -> None:
    root = tmp_path / "package"
    root.mkdir()
    target = (tmp_path if outside_root else root) / "target"
    target.mkdir()
    (target / "observations.bin").write_bytes(b"abc")
    link = root / "linked"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=True
        )
    else:
        link.symlink_to(target, target_is_directory=True)
    payload = json.loads(RAW_MANIFEST)
    payload["files"][0]["path"] = "linked/observations.bin"
    manifest_path = _make_sources(root, json.dumps(payload).encode())
    snapshot = read_input_manifest(manifest_path)
    before = _source_state(tmp_path)

    try:
        if outside_root:
            with pytest.raises(InputManifestError, match="outside.*root"):
                validate_input_package(snapshot, root, expected_paths=["linked/observations.bin"])
        else:
            result = validate_input_package(
                snapshot, root, expected_paths=["linked/observations.bin"]
            )
            assert result.files[0].resolved_path == (target / "observations.bin").resolve()
            assert result.files[0].sha256 == snapshot.manifest.files[0].sha256
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("required_extensions", ["delta-decoding"]),
        ("required_extensions", ""),
        ("package_id", ""),
        ("package_revision", False),
        ("files", []),
        ("provenance", {}),
        ("metadata", []),
        ("file.sha256", "z" * 64),
        ("file.sha256", "BA" * 32),
        ("file.size_bytes", -1),
        ("file.size_bytes", True),
        ("file.size_bytes", 3.0),
        ("file.purpose", ""),
        ("file.metadata", []),
    ],
)
def test_reader_rejects_incompatible_core_semantics(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(RAW_MANIFEST)
    if field.startswith("file."):
        payload["files"][0][field.removeprefix("file.")] = value
    else:
        payload[field] = value
    manifest_path = _make_sources(tmp_path, json.dumps(payload).encode())
    before = _source_state(tmp_path)

    try:
        with pytest.raises(InputManifestError, match="Invalid manifest"):
            read_input_manifest(manifest_path)
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize(
    "raw_bytes",
    [
        RAW_MANIFEST.replace(b'"schema_version": 1', b'"schema_version": 2, "schema_version": 1'),
        RAW_MANIFEST.replace(
            b'"required_extensions": []', b'"required_extensions": [], "metadata": {"x": NaN}'
        ),
        RAW_MANIFEST.replace(
            b'"required_extensions": []', b'"required_extensions": [], "metadata": {"x": 1e999}'
        ),
        RAW_MANIFEST.replace(
            b'"required_extensions": []', b'"required_extensions": [], "metadata": {"x": "\\ud800"}'
        ),
        RAW_MANIFEST.replace(b'"files": [', b'"files": [null,'),
        RAW_MANIFEST.replace(b'"files": [', b'"files": [17,'),
    ],
)
def test_reader_rejects_ambiguous_or_nonportable_json(tmp_path: Path, raw_bytes: bytes) -> None:
    manifest_path = _make_sources(tmp_path, raw_bytes)
    before = _source_state(tmp_path)
    try:
        with pytest.raises(InputManifestError, match="Invalid manifest"):
            read_input_manifest(manifest_path)
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", 42),
        ("acquired_at", ""),
        ("producing_arguments", "unknown"),
        ("unknowns", []),
        ("unknowns", "unknown"),
        ("unknowns", [None]),
        ("unexpected_required_fact", "value"),
    ],
)
def test_reader_requires_explicit_typed_provenance(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(RAW_MANIFEST)
    payload["provenance"][field] = value
    manifest_path = _make_sources(tmp_path, json.dumps(payload).encode())
    before = _source_state(tmp_path)
    try:
        with pytest.raises(InputManifestError, match="Invalid manifest"):
            read_input_manifest(manifest_path)
    finally:
        assert _source_state(tmp_path) == before


CANONICAL_TWO_FILES = (
    b'{"files":[{"metadata":{},"path":"empty.bin","purpose":"empty observations",'
    b'"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size_bytes":0},'
    b'{"metadata":{},"path":"observations.bin","purpose":"source observations",'
    b'"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","size_bytes":3}],'
    b'"metadata":{},"package_id":"tiny","package_revision":"r1","provenance":'
    b'{"acquired_at":null,"acquisition_mode":"fixture","producing_arguments":null,'
    b'"producing_command":null,"source":"synthetic","unknowns":["No historical acquisition"]},'
    b'"required_extensions":[],"schema_version":1}\n'
)


def test_writer_publishes_deterministic_exact_bytes_without_changing_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    (root / "empty.bin").write_bytes(b"")
    files = [
        InputFileSpec("observations.bin", "source observations"),
        InputFileSpec("empty.bin", "empty observations"),
    ]
    before = _source_state(root)

    for index, order in enumerate([files, list(reversed(files))]):
        output = tmp_path / f"copy-{index}.json"
        snapshot = write_input_manifest(
            output,
            root,
            package_id="tiny",
            package_revision="r1",
            files=order,
            provenance=json.loads(RAW_MANIFEST)["provenance"],
        )
        assert snapshot.raw_bytes == CANONICAL_TWO_FILES
        assert output.read_bytes() == CANONICAL_TWO_FILES
        assert snapshot.sha256 == read_input_manifest(output).sha256
        verified = validate_input_package(
            snapshot, root, expected_paths=["empty.bin", "observations.bin"]
        )
        assert [(item.record.path, item.size_bytes) for item in verified.files] == [
            ("empty.bin", 0),
            ("observations.bin", 3),
        ]
    assert _source_state(root) == before


def test_writer_never_overwrites_an_existing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    output = tmp_path / "existing.json"
    output.write_bytes(b"existing declaration")
    before = _source_state(tmp_path)
    try:
        with pytest.raises(InputManifestError, match="already exists"):
            write_input_manifest(
                output,
                root,
                package_id="tiny",
                package_revision="r2",
                files=[InputFileSpec("observations.bin", "source observations")],
                provenance=json.loads(RAW_MANIFEST)["provenance"],
            )
    finally:
        assert _source_state(tmp_path) == before


def test_publication_failure_leaves_no_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    output = tmp_path / "new.json"
    before = _source_state(tmp_path)
    observed = []

    def storage_failure(descriptor: int) -> None:
        # Inject an OS storage failure, leaving the package and all real file I/O intact.
        observed.append(descriptor)
        assert not output.exists()
        raise OSError("simulated storage failure")

    monkeypatch.setattr(os, "fsync", storage_failure)
    try:
        with pytest.raises(InputManifestError, match="publish"):
            write_input_manifest(
                output,
                root,
                package_id="tiny",
                package_revision="r1",
                files=[InputFileSpec("observations.bin", "source observations")],
                provenance=json.loads(RAW_MANIFEST)["provenance"],
            )
        assert observed
    finally:
        assert _source_state(tmp_path) == before


def test_writer_requires_output_outside_source_root(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    before = _source_state(tmp_path)
    try:
        with pytest.raises(InputManifestError, match="outside.*source"):
            write_input_manifest(
                root / "new.json",
                root,
                package_id="tiny",
                package_revision="r1",
                files=[InputFileSpec("observations.bin", "source observations")],
                provenance=json.loads(RAW_MANIFEST)["provenance"],
            )
    finally:
        assert _source_state(tmp_path) == before


@pytest.mark.parametrize("condition", ["match", "altered", "omitted", "duplicate"])
def test_writer_checks_historical_identity_before_publication(
    tmp_path: Path, condition: str
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    manifest_path = _make_sources(root)
    expected = read_input_manifest(manifest_path).manifest.files
    if condition == "altered":
        (root / "observations.bin").write_bytes(b"abd")
    if condition == "omitted":
        expected = ()
    if condition == "duplicate":
        expected *= 2
    output = tmp_path / "historical.json"
    before = _source_state(root)

    try:
        if condition == "match":
            result = write_input_manifest(
                output,
                root,
                package_id="tiny",
                package_revision="r1",
                files=[InputFileSpec("observations.bin", "source observations")],
                provenance=json.loads(RAW_MANIFEST)["provenance"],
                expected_files=expected,
            )
            assert result.manifest.files == expected
        else:
            with pytest.raises(InputManifestError, match="identity mismatch|inventory"):
                write_input_manifest(
                    output,
                    root,
                    package_id="tiny",
                    package_revision="r1",
                    files=[InputFileSpec("observations.bin", "source observations")],
                    provenance=json.loads(RAW_MANIFEST)["provenance"],
                    expected_files=expected,
                )
            assert not output.exists()
    finally:
        assert _source_state(root) == before


@pytest.mark.parametrize("condition", ["missing", "unsafe", "duplicate", "empty", "metadata"])
def test_invalid_source_specification_cannot_publish(tmp_path: Path, condition: str) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    files = [InputFileSpec("observations.bin", "source observations")]
    metadata = {}
    if condition in {"missing", "unsafe"}:
        files = [
            InputFileSpec("missing.bin" if condition == "missing" else "../other.bin", "source")
        ]
    if condition == "duplicate":
        files *= 2
    if condition == "empty":
        files = []
    if condition == "metadata":
        metadata = []
    output = tmp_path / "invalid.json"
    before = _source_state(tmp_path)
    try:
        with pytest.raises(InputManifestError):
            write_input_manifest(
                output,
                root,
                package_id="tiny",
                package_revision="r1",
                files=files,
                provenance=json.loads(RAW_MANIFEST)["provenance"],
                metadata=metadata,
            )
    finally:
        assert _source_state(tmp_path) == before


def test_cli_publishes_the_same_manifest_and_refuses_replacement(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    (root / "empty.bin").write_bytes(b"")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "package_id": "tiny",
                "package_revision": "r1",
                "files": [
                    {"path": "observations.bin", "purpose": "source observations"},
                    {"path": "empty.bin", "purpose": "empty observations"},
                ],
                "provenance": json.loads(RAW_MANIFEST)["provenance"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "published.json"
    command = [
        sys.executable,
        "-B",
        "-m",
        "scripts.data.write_input_manifest",
        "--spec",
        str(spec),
        "--package-root",
        str(root),
        "--output",
        str(output),
    ]
    before = _source_state(root)

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == CANONICAL_TWO_FILES
    assert json.loads(result.stdout)["sha256"] == read_input_manifest(output).sha256
    published = _source_state(tmp_path)
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert _source_state(root) == before
    assert _source_state(tmp_path) == published


@pytest.mark.parametrize("condition", ["duplicate_keys", "altered_expected", "invalid_root"])
def test_cli_rejects_invalid_specifications_before_publication(
    tmp_path: Path, condition: str
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    original = _make_sources(root)
    payload = {
        "package_id": "tiny",
        "package_revision": "r1",
        "files": [{"path": "observations.bin", "purpose": "source observations"}],
        "provenance": json.loads(RAW_MANIFEST)["provenance"],
    }
    if condition == "altered_expected":
        payload["expected_files"] = json.loads(original.read_bytes())["files"]
        payload["expected_files"][0]["sha256"] = "0" * 64
    raw = json.dumps(payload)
    if condition == "duplicate_keys":
        raw = raw.replace('"package_id": "tiny"', '"package_id": "different", "package_id": "tiny"')
    if condition == "invalid_root":
        raw = "null"
    spec = tmp_path / "spec.json"
    spec.write_text(raw, encoding="utf-8")
    output = tmp_path / "invalid.json"
    before = _source_state(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.data.write_input_manifest",
            "--spec",
            str(spec),
            "--package-root",
            str(root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr
    assert _source_state(tmp_path) == before


def test_extensible_package_revisions_preserve_metadata_and_earlier_bytes(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    files = []
    for index in range(12):
        relative = f"custom-{index:02d}.bin"
        (root / relative).write_bytes(b"abc")
        files.append(
            InputFileSpec(
                relative,
                "auxiliary observations",
                {
                    "optional_future_description": ["sample", {"units": "custom"}],
                    "sha256": "descriptive only",
                },
            )
        )
    provenance = {
        "source": "synthetic",
        "acquisition_mode": "fixture",
        "acquired_at": "2026-01-01",
        "producing_command": "generate fixture",
        "producing_arguments": [],
        "unknowns": [],
    }
    metadata = {"future_description": {"format": "illustrative", "scale": 1.5}, "label": "é"}
    before = _source_state(root)
    first = write_input_manifest(
        tmp_path / "r1.json",
        root,
        package_id="custom",
        package_revision="r1",
        files=files,
        provenance=provenance,
        metadata=metadata,
    )
    earlier = _source_state(tmp_path)
    second = write_input_manifest(
        tmp_path / "r2.json",
        root,
        package_id="custom",
        package_revision="r2",
        files=files,
        provenance=provenance,
        metadata={**metadata, "new_description": True},
    )
    loaded = read_input_manifest(tmp_path / "r1.json", expected_sha256=first.sha256)
    verified = validate_input_package(loaded, root, expected_paths=[item.path for item in files])

    assert len(verified.files) == 12
    assert loaded.manifest.metadata == metadata
    assert loaded.manifest.files[0].metadata["optional_future_description"] == [
        "sample",
        {"units": "custom"},
    ]
    assert (
        loaded.manifest.files[0].sha256
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert b'"label":"\xc3\xa9"' in loaded.raw_bytes
    assert first.sha256 != second.sha256
    assert _source_state(tmp_path)["r1.json"] == earlier["r1.json"]
    assert _source_state(root) == before


@pytest.mark.parametrize(
    "metadata", [{"x": float("nan")}, {"x": (1, 2)}, {"x": b"bytes"}, {1: "value"}]
)
def test_writer_rejects_metadata_that_would_need_lossy_json_conversion(
    tmp_path: Path, metadata: dict
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    before = _source_state(tmp_path)
    try:
        with pytest.raises(InputManifestError):
            write_input_manifest(
                tmp_path / "invalid.json",
                root,
                package_id="tiny",
                package_revision="r1",
                files=[InputFileSpec("observations.bin", "source observations")],
                provenance=json.loads(RAW_MANIFEST)["provenance"],
                metadata=metadata,
            )
    finally:
        assert _source_state(tmp_path) == before


def test_cleanup_failure_after_publication_keeps_a_successful_complete_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    _make_sources(root)
    output = tmp_path / "complete.json"
    before = _source_state(root)

    def denied_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError("injected staging cleanup denial")

    with monkeypatch.context() as patches:
        patches.setattr(Path, "unlink", denied_cleanup)
        result = write_input_manifest(
            output,
            root,
            package_id="tiny",
            package_revision="r1",
            files=[InputFileSpec("observations.bin", "source observations")],
            provenance=json.loads(RAW_MANIFEST)["provenance"],
        )

    assert read_input_manifest(output, expected_sha256=result.sha256).raw_bytes == result.raw_bytes
    assert _source_state(root) == before


def test_git_checkout_preserves_manifest_identity_with_windows_line_endings(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--template=", str(repository)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repository), "config", "core.autocrlf", "true"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "core.safecrlf", "false"], check=True)
    attributes = Path(__file__).resolve().parents[1] / ".gitattributes"
    if attributes.exists():
        (repository / ".gitattributes").write_bytes(attributes.read_bytes())
    manifest_path = repository / "data" / "manifests" / "fixture.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(RAW_MANIFEST)
    subprocess.run(["git", "-C", str(repository), "add", "."], capture_output=True, check=True)
    manifest_path.unlink()
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "checkout-index",
            "--force",
            "--",
            "data/manifests/fixture.json",
        ],
        capture_output=True,
        check=True,
    )

    snapshot = read_input_manifest(manifest_path, expected_sha256=MANIFEST_SHA256)
    assert snapshot.raw_bytes == RAW_MANIFEST
