"""Saved-run input attachment: exact declarations linked to observed reads (#208)."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from mci_gru.data.input_manifest import (
    InputFileSpec,
    ManifestSnapshot,
    read_input_manifest,
    write_input_manifest,
)
from mci_gru.data.input_observations import InputObservationContext, InputObservations
from mci_gru.data.input_snapshots import InputSnapshots, SnapshotReference, save_snapshot
from mci_gru.evaluation.artifacts import canonical_json_bytes
from mci_gru.evaluation.run_input_attachments import (
    AttachmentReference,
    RoleBinding,
    attach_run_inputs,
    read_run_inputs,
)

PRICES = b"dt,kdcode,close\n2020-01-02,AAA,10.5\n2020-01-02,BBB,20.25\n"
UNIVERSE = b"dt,kdcode\n2020-01-02,AAA\n2020-01-02,BBB\n"
REGIME = b"dt,value\n2020-01-02,1.75\n2020-01-03,\n"
REQUEST = {"series_id": "DGS10", "observation_start": "2019-01-01"}
PROVENANCE = {
    "source": "synthetic fixture",
    "acquisition_mode": "fixture",
    "acquired_at": "2020-02-01T00:00:00+00:00",
    "producing_command": None,
    "producing_arguments": None,
    "unknowns": ["Fixture packages have no producing command"],
}


@dataclass
class Fixture:
    root: Path
    window: Path
    stock: ManifestSnapshot
    regime: ManifestSnapshot
    observations: InputObservations
    required: list[RoleBinding]


def _noncanonical(path: Path) -> ManifestSnapshot:
    """Rewrite a published manifest as valid but deliberately noncanonical bytes."""
    payload = json.loads(path.read_bytes())
    path.write_bytes(json.dumps(payload, indent=3, sort_keys=False).encode() + b"\r\n")
    snapshot = read_input_manifest(path)
    assert snapshot.raw_bytes != canonical_json_bytes(payload)
    return snapshot


def _stock_package(root: Path, *, revision: str = "r1") -> tuple[Path, ManifestSnapshot]:
    package = root / "packages" / f"stock-{revision}"
    (package / "market").mkdir(parents=True)
    (package / "constituents").mkdir()
    (package / "market" / "prices.csv").write_bytes(PRICES)
    (package / "constituents" / "universe.csv").write_bytes(UNIVERSE)
    declared = root / "declared"
    declared.mkdir(exist_ok=True)
    write_input_manifest(
        declared / f"stock.{revision}.json",
        package,
        package_id="synthetic-stock",
        package_revision=revision,
        files=[
            InputFileSpec("market/prices.csv", "price panel"),
            InputFileSpec("constituents/universe.csv", "point-in-time universe"),
        ],
        provenance=PROVENANCE,
    )
    return package, _noncanonical(declared / f"stock.{revision}.json")


def _regime_package(root: Path, revision: str) -> ManifestSnapshot:
    reference = save_snapshot(
        root / "packages" / f"regime-{revision}",
        REGIME,
        package_id="synthetic-regime",
        package_revision=revision,
        role="fred.DGS10",
        source="fred",
        request=REQUEST,
        acquired_at="2020-02-01T00:00:00+00:00",
    )
    return _noncanonical(reference.manifest_path)


def _snapshot_key() -> str:
    return hashlib.sha256(
        canonical_json_bytes({"role": "fred.DGS10", "source": "fred", "request": REQUEST})
    ).hexdigest()


def _observe(
    root: Path,
    package: Path,
    regime: ManifestSnapshot,
    *,
    vix: ManifestSnapshot | None = None,
) -> InputObservations:
    """Read through the landed carriers, as the loaders do."""
    context = InputObservationContext()
    context.read_csv(
        package / "market" / "prices.csv", role="data.filename", configured_path="prices.csv"
    )
    context.read_csv(
        package / "constituents" / "universe.csv",
        role="data.pit_universe_csv",
        configured_path="universe.csv",
    )
    manifest_path = root / "packages" / f"regime-{regime.manifest.package_revision}"
    snapshots = InputSnapshots(
        mode="replay",
        references={
            _snapshot_key(): [SnapshotReference(manifest_path / "manifest.json", regime.sha256)]
        },
        input_observations=context,
    )

    def never():
        raise AssertionError("replay must not acquire")

    observed = snapshots.observe(role="fred.DGS10", source="fred", request=REQUEST, acquire=never)
    snapshots.use(observed, "regime")
    return context.freeze()


def _fixture(tmp_path: Path) -> Fixture:
    package, stock = _stock_package(tmp_path)
    regime = _regime_package(tmp_path, "r1")
    observations = _observe(tmp_path, package, regime)
    required = [
        RoleBinding("data.filename", stock.sha256, "market/prices.csv"),
        RoleBinding("data.pit_universe_csv", stock.sha256, "constituents/universe.csv"),
        RoleBinding("regime", regime.sha256, "observations.bin"),
    ]
    return Fixture(tmp_path, tmp_path / "run" / "window", stock, regime, observations, required)


def _attach(fixture: Fixture, **overrides) -> AttachmentReference:
    arguments = {
        "manifests": [fixture.stock, fixture.regime],
        "observations": fixture.observations,
        "required": fixture.required,
        **overrides,
    }
    return attach_run_inputs(fixture.window, **arguments)


def _relocate(fixture: Fixture, reference: AttachmentReference) -> AttachmentReference:
    """Move the saved run and make every original source unavailable."""
    moved = fixture.root / "elsewhere"
    shutil.move(fixture.root / "run", moved)
    shutil.rmtree(fixture.root / "packages")
    shutil.rmtree(fixture.root / "declared")
    return AttachmentReference(
        moved / "window" / reference.path.relative_to(fixture.window), reference.sha256
    )


def test_relocated_attachment_recovers_exact_declarations_and_their_reads(tmp_path):
    fixture = _fixture(tmp_path)
    reference = _relocate(fixture, _attach(fixture))

    inputs = read_run_inputs(reference)

    assert (inputs.status, inputs.problems) == ("complete", [])
    packages = {package["package_id"]: package for package in inputs.packages}
    for snapshot in (fixture.stock, fixture.regime):
        package = packages[snapshot.manifest.package_id]
        assert package["raw_bytes"] == snapshot.raw_bytes
        assert package["manifest_sha256"] == hashlib.sha256(snapshot.raw_bytes).hexdigest()
        assert package["package_revision"] == snapshot.manifest.package_revision
    links = {role["role"]: role for role in inputs.roles}
    assert links["data.filename"]["observation_ids"] == [0]
    assert links["data.pit_universe_csv"]["observation_ids"] == [1]
    assert links["regime"]["observation_ids"] == [2]
    assert links["data.filename"]["declared"] == {
        "sha256": hashlib.sha256(PRICES).hexdigest(),
        "size_bytes": len(PRICES),
    }
    assert inputs.execution == {"status": "unknown"}
    assert inputs.preservation == {"status": "unproven"}
    # Raw packages are referenced by identity, never copied into the run.
    saved = {path.name for path in (reference.path.parent.parent).rglob("*")}
    assert not {"prices.csv", "universe.csv", "observations.bin"} & saved


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        ("delete_stock_manifest", "manifest snapshot missing: synthetic-stock r1"),
        ("edit_stock_manifest", "manifest snapshot altered: synthetic-stock r1"),
        ("canonicalize_stock_manifest", "manifest snapshot altered: synthetic-stock r1"),
        ("delete_observations", "input observations missing"),
        ("edit_observations", "input observations differ from the attachment record"),
    ],
)
def test_missing_or_altered_retained_evidence_is_never_complete(tmp_path, damage, problem):
    fixture = _fixture(tmp_path)
    reference = _relocate(fixture, _attach(fixture))
    attachments = reference.path.parent
    manifest = attachments / "manifests" / f"{fixture.stock.sha256[:16]}.json"
    observations = attachments / "input_observations.json"
    if damage == "delete_stock_manifest":
        manifest.unlink()
    elif damage == "edit_stock_manifest":
        manifest.write_bytes(manifest.read_bytes().replace(b"price panel", b"price panes"))
    elif damage == "canonicalize_stock_manifest":
        manifest.write_bytes(canonical_json_bytes(json.loads(manifest.read_bytes())))
    elif damage == "delete_observations":
        observations.unlink()
    elif damage == "edit_observations":
        observations.write_bytes(observations.read_bytes().replace(b"prices.csv", b"prices.CSV"))

    inputs = read_run_inputs(reference)

    assert inputs.status == "incomplete"
    assert any(entry.startswith(problem) for entry in inputs.problems), inputs.problems


def test_a_same_size_read_with_different_bytes_is_not_the_declared_input(tmp_path):
    package, stock = _stock_package(tmp_path)
    regime = _regime_package(tmp_path, "r1")
    prices = package / "market" / "prices.csv"
    prices.write_bytes(PRICES.replace(b"10.5", b"10.6"))
    fixture = Fixture(
        tmp_path,
        tmp_path / "run" / "window",
        stock,
        regime,
        _observe(tmp_path, package, regime),
        [
            RoleBinding("data.filename", stock.sha256, "market/prices.csv"),
            RoleBinding("data.pit_universe_csv", stock.sha256, "constituents/universe.csv"),
            RoleBinding("regime", regime.sha256, "observations.bin"),
        ],
    )

    inputs = read_run_inputs(_relocate(fixture, _attach(fixture)))

    assert inputs.status == "incomplete"
    assert inputs.problems == [
        "observed input differs for role data.filename: observed SHA-256 "
        f"{hashlib.sha256(PRICES.replace(b'10.5', b'10.6')).hexdigest()}, size {len(PRICES)}",
        "required role data.filename was not consumed from market/prices.csv",
    ]


def test_a_required_role_that_was_never_read_is_explicit(tmp_path):
    fixture = _fixture(tmp_path)
    required = [*fixture.required, RoleBinding("vix", fixture.regime.sha256, "observations.bin")]

    inputs = read_run_inputs(_relocate(fixture, _attach(fixture, required=required)))

    assert inputs.status == "incomplete"
    assert inputs.problems == ["required role vix was not consumed from observations.bin"]


def test_a_read_with_no_declaration_is_explicit_even_when_every_declared_file_matches(tmp_path):
    fixture = _fixture(tmp_path)
    required = [binding for binding in fixture.required if binding.role != "regime"]

    inputs = read_run_inputs(
        _relocate(fixture, _attach(fixture, manifests=[fixture.stock], required=required))
    )

    assert inputs.status == "incomplete"
    assert inputs.problems == ["consumed role regime is not declared"]


def test_identical_bytes_from_another_package_revision_are_not_the_declared_read(tmp_path):
    fixture = _fixture(tmp_path)
    other = _regime_package(tmp_path, "r2")
    assert other.manifest.files == fixture.regime.manifest.files
    required = [
        *fixture.required[:2],
        RoleBinding("regime", other.sha256, "observations.bin"),
    ]

    inputs = read_run_inputs(
        _relocate(fixture, _attach(fixture, manifests=[fixture.stock, other], required=required))
    )

    assert inputs.status == "incomplete"
    assert inputs.problems[0].startswith("observed input differs for role regime")


def test_a_declaration_naming_a_file_its_manifest_lacks_is_explicit(tmp_path):
    fixture = _fixture(tmp_path)
    required = [
        RoleBinding("data.filename", fixture.stock.sha256, "market/prices_v2.csv"),
        *fixture.required[1:],
    ]

    inputs = read_run_inputs(_relocate(fixture, _attach(fixture, required=required)))

    assert inputs.status == "incomplete"
    assert "required role data.filename names a file its manifest lacks" in inputs.problems


def test_a_required_read_with_no_declared_package_is_linked_but_never_complete(tmp_path):
    _, stock = _stock_package(tmp_path)
    regime = _regime_package(tmp_path, "r1")
    context = InputObservationContext()
    live = InputSnapshots(mode="source", input_observations=context)
    observed = live.observe(
        role="fred.DGS10", source="fred", request=REQUEST, acquire=lambda: REGIME
    )
    live.use(observed, "regime")
    assert "manifest_sha256" not in observed.record["identity"]
    fixture = Fixture(
        tmp_path,
        tmp_path / "run" / "window",
        stock,
        regime,
        context.freeze(),
        [RoleBinding("regime", None, None), RoleBinding("credit", None, None)],
    )

    inputs = read_run_inputs(_relocate(fixture, _attach(fixture, manifests=[])))

    assert inputs.status == "incomplete"
    assert inputs.problems == [
        "required role regime has no declared package",
        "required role credit has no declared package",
        "required role credit was not consumed",
    ]
    links = {role["role"]: role["observation_ids"] for role in inputs.roles}
    assert links == {"regime": [0], "credit": []}


def test_a_required_role_needs_both_a_manifest_and_a_path(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="needs both a manifest and a path"):
        _attach(fixture, required=[RoleBinding("regime", fixture.regime.sha256, None)])
    assert not fixture.window.exists()


def test_supplied_execution_and_preservation_status_are_carried_unchanged(tmp_path):
    fixture = _fixture(tmp_path)
    execution = {
        "status": "incomplete",
        "start": {"path": "execution_provenance/" + "a" * 32 + ".json", "sha256": "b" * 64},
    }
    preservation = {"status": "partial", "detail": "local copy verified; Drive copy unverified"}

    inputs = read_run_inputs(
        _relocate(fixture, _attach(fixture, execution=execution, preservation=preservation))
    )

    assert inputs.status == "complete"
    assert inputs.execution == execution
    assert inputs.preservation == preservation


@pytest.mark.parametrize(
    "override",
    [
        {"execution": {"status": "done"}},
        {"preservation": {"status": "assumed"}},
        {"preservation": {}},
    ],
)
def test_an_unknown_status_is_refused(tmp_path, override):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="status must be one of"):
        _attach(fixture, **override)
    assert not fixture.window.exists()


def test_a_required_role_must_name_an_attached_manifest(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="names a manifest that is not attached"):
        _attach(fixture, manifests=[fixture.stock])
    assert not fixture.window.exists()


def test_an_existing_attachment_is_never_replaced(tmp_path):
    fixture = _fixture(tmp_path)
    reference = _attach(fixture)
    before = reference.path.read_bytes()
    with pytest.raises(FileExistsError):
        _attach(fixture)
    assert reference.path.read_bytes() == before


def test_an_attachment_record_that_differs_from_its_anchor_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    reference = _attach(fixture)
    record = json.loads(reference.path.read_bytes())
    record["preservation"] = {"status": "complete"}
    reference.path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="digest mismatch"):
        read_run_inputs(reference)
