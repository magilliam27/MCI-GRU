"""Tests for the first admitted run's Colab helpers and notebook (#187, #278).

``scripts/first_run_colab.py`` decides what the Colab run reads and runs: the recipe
overrides, the prerequisites an admitted run needs, and the staged input bytes. Each
guard here is paired with the case it must refuse.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict
from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from scripts import first_run_colab as frc
from scripts import nb_lib

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = REPO_ROOT / frc.RECIPE_PATH
DRIVE_MANIFEST_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "first_run" / "drive_MANIFEST.txt"
NOTEBOOK = REPO_ROOT / "notebooks" / "first_run_colab.ipynb"


def _load_generator():
    """Load the generator the way it runs, with scripts/ on sys.path for nb_lib."""
    scripts_dir = str(REPO_ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "gen_first_run_colab_nb", REPO_ROOT / "scripts" / "gen_first_run_colab_nb.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_dir)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Recipe parsing and run overrides
# ---------------------------------------------------------------------------


def test_recipe_overrides_are_read_from_the_recipe_document():
    overrides = frc.recipe_overrides(RECIPE.read_text(encoding="utf-8"))
    # Whichever data config the recipe names, it must declare a package the notebook stages.
    assert overrides[0].startswith("data=gics_top10_110_2016")
    assert frc.selected_package(REPO_ROOT) in frc.PUBLISHED_PACKAGES.values()
    assert "training.num_models=20" in overrides
    assert "training.num_epochs=100" in overrides
    assert "features.include_global_regime=true" in overrides
    assert frc.recipe_slug(RECIPE.read_text(encoding="utf-8")).startswith(
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only"
    )


def test_recipe_parsing_refuses_a_missing_block_or_a_key_set_twice():
    with pytest.raises(ValueError, match="Hydra Overrides"):
        frc.recipe_overrides("# no block here\n")
    twice = "## Hydra Overrides\n\n```text\nseed=1\nseed=2\n```\n"
    with pytest.raises(ValueError, match="more than once"):
        frc.recipe_overrides(twice)
    with pytest.raises(ValueError, match="Recipe slug"):
        frc.recipe_slug("nothing")


def test_smoke_replaces_the_budget_in_place_and_full_keeps_it():
    recipe = frc.recipe_overrides(RECIPE.read_text(encoding="utf-8"))
    run_dir = Path("/content/mci_gru_runs/tag")
    full = frc.build_run_overrides(recipe, run_dir=run_dir, experiment_name="x", smoke=False)
    smoke = frc.build_run_overrides(recipe, run_dir=run_dir, experiment_name="x", smoke=True)
    for overrides in (full, smoke):
        keys = [override.split("=", 1)[0] for override in overrides]
        assert len(keys) == len(set(keys)), "Hydra refuses a key given twice"
        assert f"hydra.run.dir={run_dir}" in overrides
        assert frc.override_value(overrides, "data.auxiliary_snapshot_mode") == "capture"
        assert frc.override_value(overrides, "tracking.enabled") == "false"
    assert frc.override_value(full, "training.num_models") == "20"
    assert frc.override_value(full, "training.num_epochs") == "100"
    assert frc.override_value(smoke, "training.num_models") == "1"
    assert frc.override_value(smoke, "training.num_epochs") == "2"
    # Everything except the budget and the run's own location is the recipe, unchanged.
    budget = set(frc.SMOKE_BUDGET)
    assert [o for o in smoke if o in recipe] == [
        o for o in recipe if o.split("=", 1)[0] not in budget
    ]


def test_capture_is_added_only_when_the_recipe_does_not_select_it():
    without = ["data=gics_top10_110_2016", "training.num_models=20", "training.num_epochs=100"]
    with_capture = [*without, "data.auxiliary_snapshot_mode=capture"]
    run_dir = Path("/r")
    added = frc.build_run_overrides(without, run_dir=run_dir, experiment_name="x", smoke=False)
    assert "data.auxiliary_snapshot_mode=capture" in added
    assert "data.auxiliary_snapshot_directory=/r/input_snapshots" in added
    kept = frc.build_run_overrides(with_capture, run_dir=run_dir, experiment_name="x", smoke=False)
    assert kept.count("data.auxiliary_snapshot_mode=capture") == 1
    assert frc.override_value(kept, "data.auxiliary_snapshot_directory") is None


def test_replace_override_refuses_a_key_the_recipe_does_not_set():
    assert frc.replace_override(["a=1", "b=2"], "b", "3") == ["a=1", "b=3"]
    with pytest.raises(KeyError):
        frc.replace_override(["a=1"], "b", "3")


@pytest.mark.parametrize("smoke", [False, True])
def test_run_overrides_compose_into_the_typed_config(smoke):
    """The notebook's overrides go through Hydra the way run_experiment.py takes them."""
    recipe = frc.recipe_overrides(RECIPE.read_text(encoding="utf-8"))
    overrides = frc.build_run_overrides(
        recipe, run_dir=Path("/tmp/first-run"), experiment_name="slug", smoke=smoke
    )
    job_overrides = [o for o in overrides if not o.startswith("hydra.")]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=job_overrides)
    config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
    assert config.training.num_models == (1 if smoke else 20)
    assert config.data.auxiliary_snapshot_mode == "capture"
    assert config.features.include_global_regime is True
    assert config.features.regime_market_csv == frc.EODHD_MARKET_RELATIVE
    assert config.experiment_name == "slug"
    assert config.tracking.enabled is False


# ---------------------------------------------------------------------------
# Prerequisites from open pull requests
# ---------------------------------------------------------------------------

_BASE_RECIPE = [
    "data=gics_top10_110_2016",
    "training.num_models=20",
    "training.num_epochs=100",
]
_PR_274_LINE = "data.auxiliary_snapshot_mode=capture"
_PR_275_LINES = [
    "model.market_latent_mode=data_dependent",
    "model.cross_section_block=residual",
    "model.gru_attn_layer_widths=per_layer",
]


def _fake_repo(tmp_path, *, pr270=True, pr274_capture=True, pr274_role=True, pr275=True):
    lines = list(_BASE_RECIPE)
    if pr274_capture:
        lines.append(_PR_274_LINE)
    if pr275:
        lines += _PR_275_LINES
    recipe = tmp_path / frc.RECIPE_PATH
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        "Recipe slug:\n\n```text\nslug\n```\n\n## Hydra Overrides\n\n```text\n"
        + "\n".join(lines)
        + "\n```\n",
        encoding="utf-8",
    )
    data = tmp_path / "configs" / "data" / "gics_top10_110_2016.yaml"
    data.parent.mkdir(parents=True)
    package = (
        f"input_package_manifest: {frc.PACKAGE_MANIFEST}\n"
        f"input_package_manifest_sha256: {frc.PACKAGE_MANIFEST_SHA256}\n"
        f"input_package_root: {frc.PACKAGE_ROOT}\n"
    )
    data.write_text("source: csv\n" + (package if pr270 else ""), encoding="utf-8")
    evaluation = tmp_path / "mci_gru" / "evaluation"
    evaluation.mkdir(parents=True)
    if pr270:
        (evaluation / "run_input_attachments.py").write_text("", encoding="utf-8")
    (evaluation / "run_input_declarations.py").write_text(
        "from mci_gru.data.data_manager import MARKET_FILE_ROLE\n" if pr274_role else "",
        encoding="utf-8",
    )
    return tmp_path


def _missing(repo):
    return [
        (p.pull_request, p.description.split(" (")[0])
        for p in frc.admitted_run_prerequisites(repo)
        if not p.present
    ]


def test_every_prerequisite_present_reads_as_ready(tmp_path):
    assert _missing(_fake_repo(tmp_path)) == []


@pytest.mark.parametrize(
    ("missing", "expected_pr"),
    [
        ({"pr270": False}, ["#270", frc.PACKAGE_PIN_PREREQUISITE]),
        ({"pr274_capture": False}, ["#274"]),
        ({"pr274_role": False}, ["#274"]),
        ({"pr275": False}, ["#275"]),
    ],
)
def test_each_missing_prerequisite_is_named_by_its_pull_request(tmp_path, missing, expected_pr):
    names = _missing(_fake_repo(tmp_path, **missing))
    assert [pr for pr, _ in names] == expected_pr, names


def test_one_of_three_model_pins_is_not_enough_for_275(tmp_path):
    repo = _fake_repo(tmp_path)
    recipe = repo / frc.RECIPE_PATH
    recipe.write_text(
        recipe.read_text(encoding="utf-8").replace("model.gru_attn_layer_widths=per_layer\n", ""),
        encoding="utf-8",
    )
    assert [pr for pr, _ in _missing(repo)] == ["#275"]


def test_a_digest_without_a_pin_is_named_as_the_pin_not_as_270(tmp_path):
    repo = _fake_repo(tmp_path)
    data = repo / "configs" / "data" / "gics_top10_110_2016.yaml"
    data.write_text(
        data.read_text(encoding="utf-8").replace(frc.PACKAGE_MANIFEST_SHA256, "ab" * 32),
        encoding="utf-8",
    )
    assert [pr for pr, _ in _missing(repo)] == [frc.PACKAGE_PIN_PREREQUISITE]


def test_270_needs_the_package_declaration_in_the_data_config(tmp_path):
    repo = _fake_repo(tmp_path)
    data = repo / "configs" / "data" / "gics_top10_110_2016.yaml"
    data.write_text("source: csv\n", encoding="utf-8")
    assert [pr for pr, _ in _missing(repo)] == ["#270", frc.PACKAGE_PIN_PREREQUISITE]


def _declare_package(repo, package):
    data = repo / "configs" / "data" / "gics_top10_110_2016.yaml"
    data.write_text(
        "source: csv\n"
        f"input_package_manifest: {package.manifest}\n"
        f"input_package_manifest_sha256: {package.manifest_sha256}\n"
        f"input_package_root: {frc.PACKAGE_ROOT}\n",
        encoding="utf-8",
    )


_OTHER_PACKAGE = frc.PublishedPackage(
    "data/manifests/other_package.r1.json", "ab" * 32, "/content/drive/other", "manifest_copy"
)


@pytest.mark.parametrize("package", [frc.LSEG_PACKAGE, _OTHER_PACKAGE])
def test_the_recipe_data_config_chooses_the_package_to_stage(tmp_path, monkeypatch, package):
    monkeypatch.setattr(
        frc,
        "PUBLISHED_PACKAGES",
        {**frc.PUBLISHED_PACKAGES, _OTHER_PACKAGE.manifest_sha256: _OTHER_PACKAGE},
    )
    repo = _fake_repo(tmp_path)
    _declare_package(repo, package)
    assert frc.selected_package(repo) == package
    assert _missing(repo) == []


def test_a_package_the_notebook_does_not_know_is_refused(tmp_path):
    repo = _fake_repo(tmp_path)
    unknown = frc.PublishedPackage("data/manifests/x.r1.json", "1" * 64, "/x", "manifest_copy")
    _declare_package(repo, unknown)
    with pytest.raises(frc.StagingError, match="does not declare a package this notebook knows"):
        frc.selected_package(repo)
    assert [pr for pr, _ in _missing(repo)] == [frc.PACKAGE_PIN_PREREQUISITE]


def test_a_known_digest_under_another_manifest_path_is_refused(tmp_path):
    repo = _fake_repo(tmp_path)
    _declare_package(
        repo,
        frc.PublishedPackage(
            _OTHER_PACKAGE.manifest, frc.LSEG_PACKAGE.manifest_sha256, "/x", "manifest_copy"
        ),
    )
    with pytest.raises(frc.StagingError):
        frc.selected_package(repo)


@pytest.mark.parametrize("package", list(frc.PUBLISHED_PACKAGES.values()))
def test_published_package_pins_match_their_committed_manifests(package):
    """A manifest not yet on this branch is checked once the commit that adds it lands."""
    assert frc.PUBLISHED_PACKAGES[package.manifest_sha256] is package
    manifest = REPO_ROOT / package.manifest
    if not manifest.exists():
        pytest.skip(f"{package.manifest} is not committed on this branch")
    assert _sha(manifest.read_bytes()) == package.manifest_sha256


def test_every_data_config_naming_a_published_digest_names_its_manifest():
    for path in sorted((REPO_ROOT / "configs" / "data").glob("*.yaml")):
        config = OmegaConf.to_container(OmegaConf.load(path))
        package = frc.PUBLISHED_PACKAGES.get(config.get("input_package_manifest_sha256"))
        if package is not None:
            assert frc.declared_package(config) == package, path.name


def test_prerequisite_check_runs_on_this_checkout():
    prerequisites = frc.admitted_run_prerequisites(REPO_ROOT)
    assert {p.pull_request for p in prerequisites} == {
        "#270",
        "#274",
        "#275",
        frc.PACKAGE_PIN_PREREQUISITE,
    }


def test_full_mode_cli_refuses_when_a_prerequisite_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(frc, "REPO_DIR", _fake_repo(tmp_path, pr275=False))
    assert frc.main(["prerequisites", "--mode", "full"]) == 1
    assert "#275" in capsys.readouterr().out
    assert frc.main(["prerequisites", "--mode", "smoke"]) == 0


@pytest.mark.parametrize("mode", ["full", "smoke"])
def test_neither_mode_runs_without_a_pin_for_the_declared_package(
    tmp_path, monkeypatch, capsys, mode
):
    repo = _fake_repo(tmp_path)
    unknown = frc.PublishedPackage("data/manifests/x.r1.json", "ab" * 32, "/x", "manifest_copy")
    _declare_package(repo, unknown)
    monkeypatch.setattr(frc, "REPO_DIR", repo)
    assert frc.main(["prerequisites", "--mode", mode]) == 1
    assert "neither mode can stage it" in capsys.readouterr().out


@pytest.mark.parametrize("drive_override", ["", "/elsewhere"])
def test_staging_hands_the_selected_package_to_stage_package(tmp_path, monkeypatch, drive_override):
    monkeypatch.setattr(
        frc,
        "PUBLISHED_PACKAGES",
        {**frc.PUBLISHED_PACKAGES, _OTHER_PACKAGE.manifest_sha256: _OTHER_PACKAGE},
    )
    repo = _fake_repo(tmp_path)
    _declare_package(repo, _OTHER_PACKAGE)
    monkeypatch.setattr(frc, "REPO_DIR", repo)
    calls = []

    class _Staged:
        def summary(self):
            return {"files": []}

    def fake_stage_package(repo_dir, drive_dir, **pins):
        calls.append((repo_dir, drive_dir, pins))
        return _Staged()

    monkeypatch.setattr(frc, "stage_package", fake_stage_package)
    monkeypatch.setattr(frc, "stage_eodhd_market_file", lambda *_: "present")
    frc._stage_inputs(drive_override, "/eodhd.csv")
    assert calls == [
        (
            repo,
            Path(drive_override or _OTHER_PACKAGE.drive_dir),
            {
                "manifest": _OTHER_PACKAGE.manifest,
                "manifest_sha256": _OTHER_PACKAGE.manifest_sha256,
                "drive_inventory": _OTHER_PACKAGE.drive_inventory,
            },
        )
    ]


# ---------------------------------------------------------------------------
# Staging single files
# ---------------------------------------------------------------------------


def test_stage_verified_file_copies_and_then_reuses(tmp_path):
    data = b"dt,close\n2008-01-02,1447.16\n"
    source = tmp_path / "drive" / "f.csv"
    source.parent.mkdir()
    source.write_bytes(data)
    target = tmp_path / "repo" / "data" / "f.csv"
    pins = {"sha256": _sha(data), "size": len(data)}
    assert frc.stage_verified_file(source, target, **pins) == "copied"
    assert target.read_bytes() == data
    assert not target.with_name("f.csv.partial").exists()
    assert frc.stage_verified_file(source, target, **pins) == "reused"


def test_a_wrong_drive_copy_is_never_copied(tmp_path):
    source = tmp_path / "f.csv"
    source.write_bytes(b"other bytes")
    target = tmp_path / "out" / "f.csv"
    with pytest.raises(frc.StagingError, match="Drive copy"):
        frc.stage_verified_file(source, target, sha256=_sha(b"pinned"), size=6)
    assert not target.exists()


def test_a_file_in_place_with_other_bytes_is_never_replaced(tmp_path):
    source = tmp_path / "f.csv"
    source.write_bytes(b"pinned")
    target = tmp_path / "out" / "f.csv"
    target.parent.mkdir()
    target.write_bytes(b"pinneX")
    with pytest.raises(frc.StagingError, match="already in place"):
        frc.stage_verified_file(source, target, sha256=_sha(b"pinned"), size=6)
    assert target.read_bytes() == b"pinneX"


def test_a_missing_drive_file_names_its_path(tmp_path):
    with pytest.raises(frc.StagingError, match="Missing on Drive"):
        frc.stage_verified_file(
            tmp_path / "absent.csv", tmp_path / "t.csv", sha256=_sha(b"x"), size=1
        )


def test_a_corrupt_copy_is_refused_and_never_published(tmp_path, monkeypatch):
    source = tmp_path / "f.csv"
    source.write_bytes(b"pinned")
    target = tmp_path / "out" / "f.csv"

    def bad_copy(src, dst):
        Path(dst).write_bytes(b"pinneX")

    monkeypatch.setattr(frc.shutil, "copyfile", bad_copy)
    with pytest.raises(frc.StagingError, match="The copy"):
        frc.stage_verified_file(source, target, sha256=_sha(b"pinned"), size=6)
    assert not target.exists()


def test_eodhd_pins_match_the_notebook_library():
    assert frc.EODHD_MARKET_RELATIVE == nb_lib.EODHD_MARKET_RELATIVE
    assert frc.EODHD_MARKET_SHA256 == nb_lib.EODHD_MARKET_SHA256
    assert frc.EODHD_MARKET_SIZE == nb_lib.EODHD_MARKET_SIZE
    assert frc.EODHD_MARKET_DRIVE_PATH == nb_lib.EODHD_MARKET_DRIVE_PATH
    with_momentum = OmegaConf.load(REPO_ROOT / "configs" / "features" / "with_momentum.yaml")
    assert with_momentum.regime_market_csv == frc.EODHD_MARKET_RELATIVE


# ---------------------------------------------------------------------------
# Staging the package
# ---------------------------------------------------------------------------


def test_the_drive_inventory_is_the_one_the_committed_r1_manifest_records():
    """The Drive MANIFEST.txt, downloaded 2026-10-04, agrees with r1 byte for byte."""
    raw = DRIVE_MANIFEST_FIXTURE.read_bytes()
    r1 = json.loads((REPO_ROOT / frc.PACKAGE_MANIFEST).read_bytes())
    assert _sha(raw) == r1["metadata"]["historical_inventory"]["sha256"]
    assert len(raw) == r1["metadata"]["historical_inventory"]["size_bytes"]
    assert frc.parse_historical_inventory(raw) == {
        record["path"]: (record["sha256"], record["size_bytes"]) for record in r1["files"]
    }
    assert _sha((REPO_ROOT / frc.PACKAGE_MANIFEST).read_bytes()) == frc.PACKAGE_MANIFEST_SHA256


@pytest.mark.parametrize(
    "line",
    ["abc 12 market/x.csv", f"{'a' * 64} twelve market/x.csv", f"{'a' * 64} 12"],
)
def test_malformed_inventory_lines_stop(line):
    with pytest.raises(frc.StagingError):
        frc.parse_historical_inventory(f"# header\n{line}\n".encode())


def test_an_inventory_listing_a_path_twice_stops():
    line = f"{'a' * 64}  12  market/x.csv\n"
    with pytest.raises(frc.StagingError, match="twice"):
        frc.parse_historical_inventory((line + line).encode())


def _package(tmp_path, *, inventory_text=None, record_inventory_of=None):
    """A synthetic Drive package, its r1-style manifest in a repo, and the pins."""
    files = {
        "constituents/u_pit_universe.csv": b"kdcode,valid_from,valid_to\nA,2016-01-04,\n",
        "market/u_lseg.csv": b"kdcode,dt,close\nA,2016-01-04,10\n",
    }
    drive = tmp_path / "drive"
    for path, data in files.items():
        (drive / path).parent.mkdir(parents=True, exist_ok=True)
        (drive / path).write_bytes(data)
    if inventory_text is None:
        inventory_text = "# sha256  bytes  path\n" + "".join(
            f"{_sha(data)}  {len(data)}  {path}\n" for path, data in sorted(files.items())
        )
    inventory = inventory_text.encode()
    (drive / frc.HISTORICAL_INVENTORY_NAME).write_bytes(inventory)
    recorded = record_inventory_of if record_inventory_of is not None else inventory
    repo = tmp_path / "repo"
    (repo / "data" / "manifests").mkdir(parents=True)
    snapshot = write_input_manifest(
        repo / "data" / "manifests" / "u.r1.json",
        drive,
        package_id="u",
        package_revision="r1",
        files=[InputFileSpec(path, "test file") for path in files],
        provenance={
            "source": "test",
            "acquisition_mode": None,
            "acquired_at": None,
            "producing_command": None,
            "producing_arguments": None,
            "unknowns": ["synthetic"],
        },
        metadata={"historical_inventory": {"sha256": _sha(recorded), "size_bytes": len(recorded)}},
    )
    pins = {"manifest": "data/manifests/u.r1.json", "manifest_sha256": snapshot.sha256}
    return repo, drive, files, pins


def test_stage_package_copies_and_verifies_every_file(tmp_path):
    repo, drive, files, pins = _package(tmp_path)
    staged = frc.stage_package(repo, drive, **pins)
    for path, data in files.items():
        assert (repo / frc.PACKAGE_ROOT / path).read_bytes() == data
    summary = staged.summary()
    assert summary["package_id"] == "u"
    assert {item["path"] for item in summary["files"]} == set(files)
    assert {item["staging"] for item in summary["files"]} == {"copied"}
    again = frc.stage_package(repo, drive, **pins)
    assert {item["staging"] for item in again.summary()["files"]} == {"reused"}


def test_a_drive_inventory_other_than_the_recorded_one_stops(tmp_path):
    repo, drive, _, pins = _package(tmp_path, record_inventory_of=b"some other inventory\n")
    with pytest.raises(frc.StagingError, match="not the inventory the manifest vouches for"):
        frc.stage_package(repo, drive, **pins)
    assert not (repo / frc.PACKAGE_ROOT).exists()


def test_an_inventory_that_disagrees_with_r1_stops_even_with_the_recorded_digest(tmp_path):
    """r1 records this inventory's digest, but the inventory lists a file r1 lacks."""
    extra = f"{'b' * 64}  3  market/extra.csv\n"
    _, _, files, _ = _package(tmp_path / "probe")
    text = "".join(f"{_sha(d)}  {len(d)}  {p}\n" for p, d in sorted(files.items())) + extra
    repo, drive, _, pins = _package(tmp_path / "real", inventory_text=text)
    with pytest.raises(frc.StagingError, match="disagree"):
        frc.stage_package(repo, drive, **pins)
    assert not (repo / frc.PACKAGE_ROOT).exists()


def test_a_corrupt_drive_package_file_stops_before_it_is_copied(tmp_path):
    repo, drive, _, pins = _package(tmp_path)
    (drive / "market" / "u_lseg.csv").write_bytes(b"kdcode,dt,close\nA,2016-01-04,11\n")
    with pytest.raises(frc.StagingError, match="Drive copy"):
        frc.stage_package(repo, drive, **pins)
    assert not (repo / frc.PACKAGE_ROOT / "market" / "u_lseg.csv").exists()


def _drive_with_manifest_copy(tmp_path):
    """A package whose Drive folder carries a copy of its manifest, as the EODHD package does."""
    repo, drive, files, pins = _package(tmp_path)
    (drive / frc.HISTORICAL_INVENTORY_NAME).unlink()
    manifest = repo / pins["manifest"]
    (drive / manifest.name).write_bytes(manifest.read_bytes())
    return repo, drive, files, {**pins, "drive_inventory": "manifest_copy"}


def test_a_drive_copy_proven_by_its_manifest_copy_is_staged(tmp_path):
    repo, drive, files, pins = _drive_with_manifest_copy(tmp_path)
    staged = frc.stage_package(repo, drive, **pins)
    for path, data in files.items():
        assert (repo / frc.PACKAGE_ROOT / path).read_bytes() == data
    assert staged.summary()["drive_inventory_sha256"] == pins["manifest_sha256"]


def test_a_drive_manifest_copy_with_other_bytes_stops(tmp_path):
    repo, drive, _, pins = _drive_with_manifest_copy(tmp_path)
    copy = drive / Path(pins["manifest"]).name
    copy.write_bytes(copy.read_bytes() + b"\n")
    with pytest.raises(frc.StagingError, match="not the inventory the manifest vouches for"):
        frc.stage_package(repo, drive, **pins)
    assert not (repo / frc.PACKAGE_ROOT).exists()


def test_a_drive_folder_without_its_manifest_copy_stops(tmp_path):
    repo, drive, _, pins = _drive_with_manifest_copy(tmp_path)
    (drive / Path(pins["manifest"]).name).unlink()
    with pytest.raises(frc.StagingError, match="Missing on Drive"):
        frc.stage_package(repo, drive, **pins)


def test_a_manifest_with_other_bytes_stops(tmp_path):
    repo, drive, _, pins = _package(tmp_path)
    with pytest.raises(Exception, match="(?i)digest|sha"):
        frc.stage_package(repo, drive, manifest=pins["manifest"], manifest_sha256="0" * 64)


# ---------------------------------------------------------------------------
# Environment pins
# ---------------------------------------------------------------------------


def test_lock_pins_read_the_lock_and_allow_only_a_local_suffix():
    pins = frc.lock_pins((REPO_ROOT / "requirements.lock").read_text(encoding="utf-8"))
    assert pins["torch"] == "2.12.1"
    assert pins["pandas"] == "3.0.3"
    assert pins["torch-geometric"] == "2.8.0"
    installed = {"torch": "2.12.1+cu130", "pandas": "3.0.2", "torch-geometric": "2.8.0"}
    problems = frc.lock_mismatches(
        {k: pins[k] for k in installed} | {"fredapi": "0.5.2"}, installed.get
    )
    assert problems == [
        "fredapi: not installed (lock pins 0.5.2)",
        "pandas: 3.0.2 installed, lock pins 3.0.3",
    ]


# ---------------------------------------------------------------------------
# Copying to Drive
# ---------------------------------------------------------------------------


def test_sync_copies_new_and_changed_files_and_never_deletes(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    (local / "checkpoints").mkdir(parents=True)
    (local / "checkpoints" / "m0.pth").write_bytes(b"one")
    (drive / "kept_on_drive.txt").parent.mkdir(parents=True)
    (drive / "kept_on_drive.txt").write_bytes(b"x")
    assert frc.sync_to_drive(local, drive) == 1
    assert frc.sync_to_drive(local, drive) == 0
    (local / "checkpoints" / "m0.pth").write_bytes(b"one more")
    assert frc.sync_to_drive(local, drive) == 1
    assert (drive / "checkpoints" / "m0.pth").read_bytes() == b"one more"
    assert (drive / "kept_on_drive.txt").read_bytes() == b"x"


@pytest.mark.parametrize("exit_code", [0, 3])
def test_run_with_drive_sync_copies_outputs_and_reports_the_exit_code(tmp_path, exit_code):
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    script = (
        "import pathlib, sys\n"
        f"pathlib.Path({str(local)!r}, 'out.txt').write_text('done')\n"
        "print('training line')\n"
        f"sys.exit({exit_code})\n"
    )
    lines = []
    returncode = frc.run_with_drive_sync(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        local_dir=local,
        drive_dir=drive,
        every_seconds=0,
        echo=lines.append,
    )
    assert returncode == exit_code
    assert lines == ["training line"]
    assert (drive / "out.txt").read_text() == "done"
    heartbeat = json.loads((drive / "colab_heartbeat.json").read_text())
    assert heartbeat["state"] == "finished"
    assert heartbeat["returncode"] == exit_code


# ---------------------------------------------------------------------------
# The generated notebook
# ---------------------------------------------------------------------------


def _notebook_sources() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_committed_notebook_matches_its_generator(tmp_path, monkeypatch):
    generator = _load_generator()
    monkeypatch.setattr(generator, "NOTEBOOK_PATH", tmp_path / "nb.ipynb")
    generator.main()
    assert (tmp_path / "nb.ipynb").read_bytes() == NOTEBOOK.read_bytes()


def test_notebook_reads_the_fred_key_from_colab_secrets_only():
    sources = _notebook_sources()
    assert 'userdata.get("FRED_API_KEY")' in sources
    assert "MY_FRED_KEY" not in sources
    assert "print(os.environ[" not in sources


def test_notebook_installs_the_lock_and_names_the_pinned_drive_paths():
    sources = _notebook_sources()
    assert '"-r", str(REPO_DIR / "requirements.lock")' in sources
    assert '"--no-deps", "-e"' in sources
    assert 'PACKAGE_DRIVE_DIR = ""' in sources
    assert f'EODHD_DRIVE_PATH = "{frc.EODHD_MARKET_DRIVE_PATH}"' in sources
    assert 'RUN_MODE = "full"' in sources


def test_notebook_runs_the_repository_in_the_lock_qualified_python():
    """Colab's kernel Python moves (3.13 on 2026-10-10); the lock is qualified for one 3.12."""
    sources = _notebook_sources()
    qualified = re.findall(
        r'python-version: "([0-9.]+)"',
        (REPO_ROOT / ".github" / "workflows" / "lock-stack-linux.yml").read_text(encoding="utf-8"),
    )
    assert set(qualified) == {"3.12.11"}
    assert f'PYTHON_VERSION = "{qualified[0]}"' in sources
    assert 'UV + ["venv", "--python", PYTHON_VERSION, str(VENV_DIR)]' in sources
    assert 'PIP = UV + ["pip", "install", "--python", PY]' in sources
    assert sources.index('os.environ.pop("PYTHONPATH", None)') < sources.index(
        'UV + ["venv", "--python", PYTHON_VERSION'
    )
    invocations = re.findall(r'([\w.]+), "scripts/first_run_colab.py"', sources)
    assert invocations == ["PY"] * 7


def test_the_run_folder_lists_installed_packages_without_pip():
    lines = frc.installed_distributions().splitlines()
    assert lines == sorted(lines, key=str.lower)
    assert f"PyYAML=={yaml.__version__}" in lines


def test_notebook_holds_no_recipe_values_of_its_own():
    """The recipe is read at the cloned commit; a copy in the notebook would drift."""
    sources = _notebook_sources()
    for override in frc.recipe_overrides(RECIPE.read_text(encoding="utf-8")):
        assert override not in sources, override


def test_cli_help_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "first_run_colab.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "check-env" in result.stdout


# ---------------------------------------------------------------------------
# Tracked package files and line endings
# ---------------------------------------------------------------------------


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _git_repo_with(tmp_path, files):
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for path, data in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def test_the_tracked_package_sidecars_are_the_pinned_bytes_with_crlf_endings():
    """On a Linux clone the two tracked sidecars hold LF bytes; their CRLF form is the pin."""
    r1 = json.loads((REPO_ROOT / frc.PACKAGE_MANIFEST).read_bytes())
    tracked = set(_git(REPO_ROOT, "ls-files", frc.PACKAGE_ROOT).splitlines())
    in_package = [r for r in r1["files"] if f"{frc.PACKAGE_ROOT}/{r['path']}" in tracked]
    assert len(in_package) == 2, in_package
    for record in in_package:
        blob = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{frc.PACKAGE_ROOT}/{record['path']}"],
            check=True,
            capture_output=True,
        ).stdout
        crlf = blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        assert (_sha(crlf), len(crlf)) == (record["sha256"], record["size_bytes"])


def test_tracked_lf_files_are_realigned_to_their_crlf_pins_and_git_stays_clean(tmp_path):
    lf = b'{\n  "a": 1\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    repo = _git_repo_with(tmp_path, {"data/raw/market/x.meta.json": lf, "README": b"r\n"})
    inventory = {
        "market/x.meta.json": (_sha(crlf), len(crlf)),
        "market/untracked.csv": ("0" * 64, 1),
    }
    assert frc.align_tracked_line_endings(repo, inventory) == ["market/x.meta.json"]
    assert (repo / "data/raw/market/x.meta.json").read_bytes() == crlf
    assert _git(repo, "status", "--porcelain") == ""
    # A second pass finds nothing to do.
    assert frc.align_tracked_line_endings(repo, inventory) == []
    # A file that reverts to LF is realigned again without a second attributes rule.
    (repo / "data/raw/market/x.meta.json").write_bytes(lf)
    assert frc.align_tracked_line_endings(repo, inventory) == ["market/x.meta.json"]
    rules = (repo / ".git" / "info" / "attributes").read_text().splitlines()
    assert rules.count("/data/raw/market/x.meta.json text eol=crlf") == 1


def test_a_tracked_file_with_other_content_is_not_realigned(tmp_path):
    lf = b'{\n  "a": 1\n}\n'
    repo = _git_repo_with(tmp_path, {"data/raw/market/x.meta.json": lf})
    other = b'{\r\n  "a": 2\r\n}\r\n'
    with pytest.raises(frc.StagingError, match="not the package's"):
        frc.align_tracked_line_endings(repo, {"market/x.meta.json": (_sha(other), len(other))})
    assert (repo / "data/raw/market/x.meta.json").read_bytes() == lf
    assert (
        not (repo / ".git" / "info" / "attributes").exists()
        or "x.meta.json" not in (repo / ".git" / "info" / "attributes").read_text()
    )


def test_stage_package_in_a_clone_realigns_a_tracked_sidecar(tmp_path):
    lf = b'{"pull": 1}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    data = b"kdcode,dt,close\nA,2016-01-04,10\n"
    drive = tmp_path / "drive"
    (drive / "market").mkdir(parents=True)
    (drive / "market" / "p.meta.json").write_bytes(crlf)
    (drive / "market" / "p.csv").write_bytes(data)
    inventory = (
        f"{_sha(data)}  {len(data)}  market/p.csv\n{_sha(crlf)}  {len(crlf)}  market/p.meta.json\n"
    )
    (drive / frc.HISTORICAL_INVENTORY_NAME).write_bytes(inventory.encode())
    repo = _git_repo_with(
        tmp_path, {"data/raw/market/p.meta.json": lf, "data/manifests/.keep": b""}
    )
    snapshot = write_input_manifest(
        repo / "data" / "manifests" / "p.r1.json",
        drive,
        package_id="p",
        package_revision="r1",
        files=[
            InputFileSpec("market/p.csv", "panel"),
            InputFileSpec("market/p.meta.json", "sidecar"),
        ],
        provenance={
            "source": "test",
            "acquisition_mode": None,
            "acquired_at": None,
            "producing_command": None,
            "producing_arguments": None,
            "unknowns": ["synthetic"],
        },
        metadata={
            "historical_inventory": {
                "sha256": _sha(inventory.encode()),
                "size_bytes": len(inventory.encode()),
            }
        },
    )
    staged = frc.stage_package(
        repo, drive, manifest="data/manifests/p.r1.json", manifest_sha256=snapshot.sha256
    )
    outcomes = {item["path"]: item["staging"] for item in staged.summary()["files"]}
    assert outcomes == {"market/p.csv": "copied", "market/p.meta.json": "aligned"}
    # The staged CSV is untracked here; in the repository *.csv is ignored.
    assert _git(repo, "status", "--porcelain", "--", "data/raw/market/p.meta.json") == ""


# ---------------------------------------------------------------------------
# Copy failures and the run command
# ---------------------------------------------------------------------------


def test_a_failed_drive_copy_is_reported_and_does_not_stop_the_run(tmp_path, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()

    def broken(*_args, **_kwargs):
        raise OSError("Transport endpoint is not connected")

    monkeypatch.setattr(frc, "sync_to_drive", broken)
    lines = []
    returncode = frc.run_with_drive_sync(
        [sys.executable, "-c", "import time; print('epoch 1'); time.sleep(0.3); print('epoch 2')"],
        cwd=tmp_path,
        local_dir=local,
        drive_dir=drive,
        every_seconds=0.05,
        echo=lines.append,
    )
    assert returncode == 0
    assert "epoch 1" in lines and "epoch 2" in lines
    assert any(line.startswith("[drive copy] failed") for line in lines)


def test_sync_skips_capture_staging_files(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    (local / "input_snapshots").mkdir(parents=True)
    (local / "input_snapshots" / ".input-manifest-abc").write_bytes(b"tmp")
    (local / "input_snapshots" / "manifest.json").write_bytes(b"{}")
    frc.sync_to_drive(local, drive)
    assert (drive / "input_snapshots" / "manifest.json").exists()
    assert not (drive / "input_snapshots" / ".input-manifest-abc").exists()


SECRET = "fixture-fred-key-0123456789"


@pytest.fixture
def run_repo(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path / "repo")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    monkeypatch.setattr(frc, "REPO_DIR", repo)
    monkeypatch.setattr(frc, "_stage_inputs", lambda *_: {"package": "stub"})
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    return repo


def _run(tmp_path, mode="full", tag="t1"):
    return frc.main(
        [
            "run", "--mode", mode,
            "--local-root", str(tmp_path / "local"),
            "--drive-root", str(tmp_path / "drive"),
            "--run-tag", tag,
            "--tag-file", str(tmp_path / "tag.txt"),
        ]
    )  # fmt: skip


def test_a_full_run_refuses_a_missing_prerequisite(tmp_path, run_repo):
    recipe = run_repo / frc.RECIPE_PATH
    recipe.write_text(
        recipe.read_text(encoding="utf-8").replace("model.cross_section_block=residual\n", ""),
        encoding="utf-8",
    )
    _git(run_repo, "commit", "-qam", "drop a pin")
    assert _run(tmp_path) == 1
    assert not (tmp_path / "local").exists()


def test_a_run_refuses_without_the_fred_key(tmp_path, run_repo, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", " ")
    assert _run(tmp_path, mode="smoke") == 1
    assert not (tmp_path / "local").exists()


def test_a_full_run_refuses_source_changes_including_untracked_files(tmp_path, run_repo):
    (run_repo / "mci_gru" / "evaluation" / "stray.py").write_text("x = 1\n", encoding="utf-8")
    assert _run(tmp_path) == 1
    assert not (tmp_path / "local").exists()


def test_a_run_refuses_to_reuse_a_folder(tmp_path, run_repo):
    (tmp_path / "drive" / "t1").mkdir(parents=True)
    assert _run(tmp_path) == 1
    assert not (tmp_path / "local").exists()


def test_a_run_records_itself_without_the_key(tmp_path, run_repo, monkeypatch):
    # The fake checkout has no run_experiment.py, so training exits non-zero at once;
    # everything before and after it is real.
    real_run = frc.subprocess.run

    def run_without_pip(command, *args, **kwargs):
        assert "pip" not in command, command
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(frc.subprocess, "run", run_without_pip)
    returncode = _run(tmp_path)
    assert returncode != 0
    record_path = tmp_path / "drive" / "t1" / "colab_run_record.json"
    record = json.loads(record_path.read_text())
    assert record["mode"] == "full"
    assert record["returncode"] == returncode
    assert record["staging"] == {"package": "stub"}
    assert record["fred_api_key_set"] is True
    assert "tracking.enabled=false" in record["overrides"]
    assert (tmp_path / "tag.txt").read_text().strip() == "t1"
    freeze = (tmp_path / "drive" / "t1" / "pip_freeze.txt").read_text().splitlines()
    assert f"PyYAML=={yaml.__version__}" in freeze
    for path in (tmp_path / "drive").rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(errors="ignore"), path


# ---------------------------------------------------------------------------
# The model-fix rerun's arms (#326)
# ---------------------------------------------------------------------------

#: The recipe as #325 leaves it: the fixed inputs pinned, so the levels arm can undo them.
_FIXED_RECIPE_LINE = "features.base_inputs=relative"


def _recipe_with_fixed_inputs() -> list[str]:
    recipe = frc.recipe_overrides(RECIPE.read_text(encoding="utf-8"))
    if frc.override_value(recipe, "features.base_inputs") is None:
        recipe.append(_FIXED_RECIPE_LINE)
    return recipe


def test_the_recipe_arm_is_the_recipe():
    recipe = _recipe_with_fixed_inputs()
    assert frc.apply_arm(recipe, "recipe") == recipe
    assert frc.arm_experiment_name("slug", "recipe") == "slug"


def test_each_control_arm_differs_from_the_recipe_only_by_its_change():
    recipe = _recipe_with_fixed_inputs()
    zeroed = frc.apply_arm(recipe, "graph_zeroed")
    assert zeroed == [*recipe, "+experiment=graph_zeroed"]
    levels = frc.apply_arm(recipe, "levels_inputs")
    assert [o for o in levels if o not in recipe] == ["features.base_inputs=levels"]
    assert [o for o in recipe if o not in levels] == [_FIXED_RECIPE_LINE]
    assert len(levels) == len(recipe)
    names = {frc.arm_experiment_name("slug", arm) for arm in frc.RUN_ARMS}
    assert len(names) == len(frc.RUN_ARMS), "an arm must not file its results under the recipe id"


def test_an_arm_that_would_change_nothing_is_refused():
    with pytest.raises(
        KeyError, match="changes features.base_inputs, which the recipe does not set"
    ):
        frc.apply_arm(["seed=1729"], "levels_inputs")
    with pytest.raises(ValueError, match="changes nothing"):
        frc.apply_arm(["features.base_inputs=levels"], "levels_inputs")
    with pytest.raises(ValueError, match="adds experiment, which the recipe already sets"):
        frc.apply_arm(["+experiment=baseline"], "graph_zeroed")


@pytest.mark.parametrize("arm", sorted(frc.RUN_ARMS))
def test_every_arm_composes_into_the_typed_config(arm):
    """Each arm goes through Hydra the way run_experiment.py takes it."""
    recipe = _recipe_with_fixed_inputs()
    overrides = frc.build_run_overrides(
        recipe,
        run_dir=Path("/tmp/rerun"),
        experiment_name=frc.arm_experiment_name("slug", arm),
        smoke=False,
        arm=arm,
    )
    job_overrides = [o for o in overrides if not o.startswith("hydra.")]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=job_overrides)
    config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
    assert config.graph.zero_edges is (arm == "graph_zeroed")
    assert config.features.base_inputs == ("levels" if arm == "levels_inputs" else "relative")
    assert config.experiment_name == frc.arm_experiment_name("slug", arm)
    # Same seed, budget and data whichever arm runs.
    assert config.seed == 1729
    assert config.training.num_models == 20
    assert config.training.num_epochs == 100
    assert config.features.include_global_regime is True


def _run_arm(tmp_path, arm, mode="full"):
    return frc.main(
        [
            "run", "--mode", mode, "--arm", arm,
            "--local-root", str(tmp_path / "local"),
            "--drive-root", str(tmp_path / "drive"),
            "--tag-file", str(tmp_path / "tag.txt"),
        ]
    )  # fmt: skip


def test_a_control_arm_records_itself_and_is_never_an_admitted_run_candidate(tmp_path, run_repo):
    returncode = _run_arm(tmp_path, "graph_zeroed")
    tag = (tmp_path / "tag.txt").read_text().strip()
    assert re.fullmatch(r"\d{8}T\d{6}Z-full-graph-zeroed-[0-9a-f]{8}", tag), tag
    record = json.loads((tmp_path / "drive" / tag / "colab_run_record.json").read_text())
    assert record["returncode"] == returncode
    assert record["arm"] == "graph_zeroed"
    assert "not an admitted-run candidate" in record["evidence"]
    assert "+experiment=graph_zeroed" in record["overrides"]
    assert "experiment_name=slug__arm-graph-zeroed" in record["overrides"]


def test_an_arm_the_recipe_cannot_take_is_refused_before_anything_is_written(
    tmp_path, run_repo, capsys
):
    # The fake recipe predates #325, so it does not pin the fixed inputs to undo.
    assert _run_arm(tmp_path, "levels_inputs") == 1
    assert "which the recipe does not set" in capsys.readouterr().out
    assert not (tmp_path / "local").exists()
    assert not (tmp_path / "drive").exists()


# GPU precision check -------------------------------------------------------

_JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_eval_precision" name="test_prediction_runs_in_full_precision"/>
<testcase classname="tests.test_eval_precision" name="test_cuda_prediction_matches_fp32_to_1e6">{}</testcase>
</testsuite></testsuites>"""


@pytest.mark.parametrize(
    ("body", "outcome"),
    [
        ("", "passed"),
        ('<skipped message="needs a CUDA device"/>', "skipped"),
        ('<failure message="x"/>', "failed"),
        ('<error message="x"/>', "failed"),
    ],
)
def test_junit_outcome_reads_the_cuda_test(body, outcome):
    assert frc.junit_outcome(_JUNIT.format(body), frc.GPU_PRECISION_TEST) == outcome
    assert frc.junit_outcome(_JUNIT.format(body), "test_not_there") == "missing"


@pytest.mark.parametrize(
    ("body", "pytest_exit", "expected"),
    [("", 0, 0), ('<skipped message="needs a CUDA device"/>', 0, 1), ("", 1, 1)],
)
def test_the_gpu_check_passes_only_when_the_cuda_test_ran_and_passed(
    tmp_path, monkeypatch, body, pytest_exit, expected
):
    def fake_pytest(command, *args, **kwargs):
        assert frc.GPU_PRECISION_TEST_FILE in command
        junit = next(c.split("=", 1)[1] for c in command if c.startswith("--junitxml="))
        Path(junit).write_text(_JUNIT.format(body), encoding="utf-8")
        return subprocess.CompletedProcess(command, pytest_exit, stdout="log line\n", stderr="")

    monkeypatch.setattr(frc.subprocess, "run", fake_pytest)
    assert frc.main(["gpu-check", "--out-dir", str(tmp_path / "gpu")]) == expected
    assert (tmp_path / "gpu" / "eval_precision_gpu.log").read_text() == "log line\n"


def test_the_gpu_check_fails_when_pytest_writes_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        frc.subprocess,
        "run",
        lambda command, *a, **k: subprocess.CompletedProcess(command, 0, "", ""),
    )
    assert frc.main(["gpu-check", "--out-dir", str(tmp_path / "gpu")]) == 1


# Arm comparison ------------------------------------------------------------

_DATES = [f"2025-01-{day:02d}" for day in range(2, 22)]


_FAKE_RECIPE = ["seed=1729", "training.num_models=20", _FIXED_RECIPE_LINE, "tracking.enabled=false"]


def _fake_package(staging="reused"):
    return {
        "package_id": "p",
        "package_revision": "r2",
        "manifest_sha256": "m" * 64,
        "files": [
            {"path": "data/raw/a.csv", "sha256": "a" * 64, "size_bytes": 3, "staging": staging}
        ],
    }


def _fake_arm_run(
    root,
    arm,
    ic,
    *,
    commit="c0ffee",
    ties=None,
    summary_ic=None,
    mode="full",
    overrides=None,
    package=None,
    observations=(b"fred",),
):
    """A finished run folder as the launcher and run_experiment.py leave it."""
    import numpy as np
    import pandas as pd

    run_dir = root / arm
    window = run_dir / "outputs"
    window.mkdir(parents=True)
    ic = np.asarray(ic, dtype=float)
    rank_ic = ic / 2
    pd.DataFrame({"dt": _DATES, "ic": ic, "rank_ic": rank_ic}).to_csv(
        window / "daily_ic.csv", index=False, float_format="%.17g"
    )
    summary = {
        "label_t": 5,
        "top_k_values": [50],
        "metrics": {
            "avg_ic": float(np.nanmean(ic)) if summary_ic is None else summary_ic,
            "avg_rank_ic": float(np.nanmean(rank_ic)),
            "avg_ic_ci_lower": -0.1,
            "avg_ic_ci_upper": 0.1,
        },
    }
    if ties is not None:
        summary["linked_score_ties"] = ties
    (window / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for index, data in enumerate(observations):
        files = run_dir / "input_snapshots" / f"obs{index}" / "files"
        files.mkdir(parents=True)
        (files / "observations.bin").write_bytes(data)
    if overrides is None:
        overrides = frc.apply_arm(_FAKE_RECIPE, arm)
    frc.write_json(
        run_dir / "colab_run_record.json",
        {
            "arm": arm,
            "mode": mode,
            "run_tag": f"tag-{arm}",
            "commit": commit,
            "returncode": 0,
            "staging": {"package": package or _fake_package()},
            "overrides": [
                *overrides,
                f"hydra.run.dir=/content/mci_gru_runs/tag-{arm}",
                f"experiment_name=slug-{arm}",
            ],
        },
    )
    return run_dir


def _ties(per_date_tied, per_date_pairs):
    return {
        "tolerance": 1e-6,
        "linked_pairs": {"per_date": per_date_pairs, "total": sum(per_date_pairs.values())},
        "ensemble": {"per_date": per_date_tied, "total": sum(per_date_tied.values())},
        "members": [{"model_id": 0, "per_date": per_date_tied, "total": 7}],
    }


def test_compare_pairs_the_reference_against_each_arm_on_common_dates(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.05, len(_DATES))
    recipe_ic = base + 0.01
    recipe_ic[4] = np.nan  # a day the recipe arm could not score
    runs = [
        _fake_arm_run(
            tmp_path, "recipe", recipe_ic,
            ties=_ties({_DATES[0]: 0, _DATES[1]: 1}, {_DATES[0]: 26, _DATES[1]: 24}),
        ),
        _fake_arm_run(tmp_path, "graph_zeroed", base, ties=_ties({}, {})),
        _fake_arm_run(tmp_path, "levels_inputs", base + 0.02),
    ]  # fmt: skip
    out = tmp_path / "comparison"
    argv = ["compare", "--out-dir", str(out), "--resamples", "200"]
    for run_dir in runs:
        argv += ["--run", f"{run_dir.name}={run_dir}"]
    assert frc.main(argv) == 0

    comparison = json.loads((out / "arm_comparison.json").read_text())
    assert [a["arm"] for a in comparison["arms"]] == ["recipe", "graph_zeroed", "levels_inputs"]
    recipe = comparison["arms"][0]
    assert recipe["n_dates"] == len(_DATES) - 1
    assert recipe["experiment_name"] == "slug-recipe"
    assert recipe["linked_score_ties"]["ensemble_tied_total"] == 1
    assert recipe["linked_score_ties"]["ensemble_tied_share"] == pytest.approx(1 / 50)
    assert recipe["linked_score_ties"]["dates_with_an_ensemble_tie"] == 1
    assert comparison["arms"][1]["linked_score_ties"]["ensemble_tied_share"] is None
    assert comparison["arms"][2]["linked_score_ties"] is None

    paired = {(p["metric"], p["control"]): p for p in comparison["paired_daily_difference"]}
    assert set(paired) == {
        (metric, control)
        for metric in ("ic", "rank_ic")
        for control in ("graph_zeroed", "levels_inputs")
    }
    keep = np.isfinite(recipe_ic)
    vs_zeroed = paired[("ic", "graph_zeroed")]
    assert vs_zeroed["arm"] == "recipe"
    assert vs_zeroed["n_days"] == vs_zeroed["n_common_dates"] == len(_DATES) - 1
    assert vs_zeroed["mean_delta"] == pytest.approx(0.01, abs=1e-12)
    assert paired[("ic", "levels_inputs")]["mean_delta"] == pytest.approx(-0.01, abs=1e-12)
    assert paired[("rank_ic", "graph_zeroed")]["mean_delta"] == pytest.approx(
        float(np.mean((recipe_ic[keep] - base[keep]) / 2)), abs=1e-12
    )
    assert vs_zeroed["hac_lags"] == 4
    assert (out / "daily_ic_by_arm.csv").exists()


def test_compare_refuses_a_dated_series_its_summary_does_not_average(tmp_path, capsys):
    ic = [0.01] * len(_DATES)
    runs = [
        _fake_arm_run(tmp_path, "recipe", ic, summary_ic=0.02),
        _fake_arm_run(tmp_path, "graph_zeroed", ic),
    ]
    argv = ["compare", "--out-dir", str(tmp_path / "out")]
    for run_dir in runs:
        argv += ["--run", f"{run_dir.name}={run_dir}"]
    assert frc.main(argv) == 1
    assert "differs from the summary" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_compare_refuses_arms_from_different_commits(tmp_path, capsys):
    ic = [0.01] * len(_DATES)
    runs = [
        _fake_arm_run(tmp_path, "recipe", ic),
        _fake_arm_run(tmp_path, "graph_zeroed", ic, commit="beef"),
    ]
    argv = ["compare", "--out-dir", str(tmp_path / "out")]
    for run_dir in runs:
        argv += ["--run", f"{run_dir.name}={run_dir}"]
    assert frc.main(argv) == 1
    assert "ran commit" in capsys.readouterr().out


def test_compare_refuses_a_folder_labelled_as_another_arm(tmp_path, capsys):
    ic = [0.01] * len(_DATES)
    recipe = _fake_arm_run(tmp_path, "recipe", ic)
    zeroed = _fake_arm_run(tmp_path, "graph_zeroed", ic)
    argv = ["compare", "--out-dir", str(tmp_path / "out"), "--run", f"recipe={zeroed}"]
    argv += ["--run", f"graph_zeroed={recipe}"]
    assert frc.main(argv) == 1
    assert "is arm 'graph_zeroed', not 'recipe'" in capsys.readouterr().out


def test_the_notebook_runs_one_recipe_arm_unless_told_otherwise():
    sources = _notebook_sources()
    assert 'RUN_ARMS = ["recipe"]' in sources
    assert '"--arm", arm' in sources
    assert sources.index('"gpu-check"') < sources.index('"run",')


def _compare(tmp_path, runs, out="out"):
    argv = ["compare", "--out-dir", str(tmp_path / out), "--resamples", "50"]
    for arm, run_dir in runs:
        argv += ["--run", f"{arm}={run_dir}"]
    return frc.main(argv)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "smoke"}, "ran mode"),
        ({"overrides": [*_FAKE_RECIPE, "+experiment=graph_zeroed", "graph.drop_edge_p=0.0"]}, "overrides other than"),
        ({"overrides": [s.replace("20", "1") for s in _FAKE_RECIPE] + ["+experiment=graph_zeroed"]}, "overrides other than"),
        ({"package": {**_fake_package(), "manifest_sha256": "n" * 64}}, "another input package"),
    ],
)  # fmt: skip
def test_compare_refuses_a_control_that_differs_by_more_than_its_arm(
    tmp_path, capsys, change, message
):
    ic = [0.01] * len(_DATES)
    recipe = _fake_arm_run(tmp_path, "recipe", ic)
    zeroed = _fake_arm_run(tmp_path, "graph_zeroed", ic, **change)
    assert _compare(tmp_path, [("recipe", recipe), ("graph_zeroed", zeroed)]) == 1
    assert message in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_compare_accepts_the_same_package_bytes_however_they_were_staged(tmp_path):
    ic = [0.01, 0.02] * (len(_DATES) // 2)
    recipe = _fake_arm_run(tmp_path, "recipe", ic, package=_fake_package("copied"))
    zeroed = _fake_arm_run(tmp_path, "graph_zeroed", ic, package=_fake_package("reused"))
    assert _compare(tmp_path, [("recipe", recipe), ("graph_zeroed", zeroed)]) == 0


def test_compare_refuses_an_arm_given_twice_or_a_reference_other_than_the_recipe(tmp_path, capsys):
    ic = [0.01] * len(_DATES)
    recipe = _fake_arm_run(tmp_path, "recipe", ic)
    zeroed = _fake_arm_run(tmp_path, "graph_zeroed", ic)
    assert _compare(tmp_path, [("recipe", recipe), ("recipe", recipe)]) == 1
    assert "each arm may be given once" in capsys.readouterr().out
    assert _compare(tmp_path, [("graph_zeroed", zeroed), ("recipe", recipe)]) == 1
    assert "must be 'recipe'" in capsys.readouterr().out


def test_compare_flags_auxiliary_inputs_that_differ_from_the_recipe_arms(tmp_path):
    ic = [0.01, 0.03] * (len(_DATES) // 2)
    recipe = _fake_arm_run(tmp_path, "recipe", ic, observations=(b"fred-a", b"index"))
    same = _fake_arm_run(tmp_path, "graph_zeroed", ic, observations=(b"index", b"fred-a"))
    revised = _fake_arm_run(tmp_path, "levels_inputs", ic, observations=(b"fred-b", b"index"))
    runs = [("recipe", recipe), ("graph_zeroed", same), ("levels_inputs", revised)]
    assert _compare(tmp_path, runs) == 0
    arms = json.loads((tmp_path / "out" / "arm_comparison.json").read_text())["arms"]
    assert [a["auxiliary_inputs_match_reference"] for a in arms] == [True, True, False]
    assert [a["auxiliary_observations"] for a in arms] == [2, 2, 2]


def test_compare_writes_strict_json_and_never_writes_over_a_folder(tmp_path, capsys):
    ic = [0.01] * len(_DATES)  # identical arms: every difference is 0, so some stats are NaN
    recipe = _fake_arm_run(tmp_path, "recipe", ic)
    zeroed = _fake_arm_run(tmp_path, "graph_zeroed", ic)
    runs = [("recipe", recipe), ("graph_zeroed", zeroed)]
    assert _compare(tmp_path, runs) == 0
    text = (tmp_path / "out" / "arm_comparison.json").read_text()
    json.loads(text, parse_constant=lambda token: pytest.fail(f"non-strict JSON {token}"))
    assert json.loads(text)["paired_daily_difference"][0]["top_decile_share"] is None
    assert _compare(tmp_path, runs) == 1
    assert "already exists" in capsys.readouterr().out


def test_check_arms_refuses_before_training_an_arm_the_recipe_cannot_take(
    tmp_path, run_repo, capsys
):
    # The fake recipe predates #325: levels_inputs has nothing to undo.
    assert frc.main(["check-arms", "recipe", "graph_zeroed"]) == 0
    assert "ok      graph_zeroed: +experiment=graph_zeroed" in capsys.readouterr().out
    assert frc.main(["check-arms", "recipe", "graph_zeroed", "levels_inputs"]) == 1
    out = capsys.readouterr().out
    assert "REFUSED 'levels_inputs'" in out
    assert frc.main(["check-arms", "recipe", "recipe"]) == 1
    assert frc.main(["check-arms", "no_such_arm"]) == 1


def test_the_notebook_checks_every_arm_before_any_trains():
    sources = _notebook_sources()
    assert sources.index('"check-arms", *RUN_ARMS') < sources.index('"run",')
    assert "for arm, tag in RUN_TAGS.items():" in sources
