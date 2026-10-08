"""Tests for the first admitted run's Colab helpers and notebook (#187, #278).

``scripts/first_run_colab.py`` decides what the Colab run reads and runs: the recipe
overrides, the prerequisites an admitted run needs, and the staged input bytes. Each
guard here is paired with the case it must refuse.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
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
    assert overrides[0] == "data=gics_top10_110_2016"
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
        ({"pr270": False}, "#270"),
        ({"pr274_capture": False}, "#274"),
        ({"pr274_role": False}, "#274"),
        ({"pr275": False}, "#275"),
    ],
)
def test_each_missing_prerequisite_is_named_by_its_pull_request(tmp_path, missing, expected_pr):
    names = _missing(_fake_repo(tmp_path, **missing))
    assert [pr for pr, _ in names] == [expected_pr], names


def test_one_of_three_model_pins_is_not_enough_for_275(tmp_path):
    repo = _fake_repo(tmp_path)
    recipe = repo / frc.RECIPE_PATH
    recipe.write_text(
        recipe.read_text(encoding="utf-8").replace("model.gru_attn_layer_widths=per_layer\n", ""),
        encoding="utf-8",
    )
    assert [pr for pr, _ in _missing(repo)] == ["#275"]


def test_270_needs_the_manifest_this_notebook_stages_not_just_the_key(tmp_path):
    repo = _fake_repo(tmp_path)
    data = repo / "configs" / "data" / "gics_top10_110_2016.yaml"
    data.write_text(
        data.read_text(encoding="utf-8").replace(frc.PACKAGE_MANIFEST_SHA256, "0" * 64),
        encoding="utf-8",
    )
    assert [pr for pr, _ in _missing(repo)] == ["#270"]


def test_prerequisite_check_runs_on_this_checkout():
    prerequisites = frc.admitted_run_prerequisites(REPO_ROOT)
    assert {p.pull_request for p in prerequisites} == {"#270", "#274", "#275"}


def test_full_mode_cli_refuses_when_a_prerequisite_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(frc, "REPO_DIR", _fake_repo(tmp_path, pr275=False))
    assert frc.main(["prerequisites", "--mode", "full"]) == 1
    assert "#275" in capsys.readouterr().out
    assert frc.main(["prerequisites", "--mode", "smoke"]) == 0


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
    with pytest.raises(frc.StagingError, match="not the inventory the r1 manifest records"):
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
    assert f'PACKAGE_DRIVE_DIR = "{frc.PACKAGE_DRIVE_DIR}"' in sources
    assert f'EODHD_DRIVE_PATH = "{frc.EODHD_MARKET_DRIVE_PATH}"' in sources
    assert 'RUN_MODE = "full"' in sources


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


def test_a_run_records_itself_without_the_key(tmp_path, run_repo):
    # The fake checkout has no run_experiment.py, so training exits non-zero at once;
    # everything before and after it is real.
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
    assert (tmp_path / "drive" / "t1" / "pip_freeze.txt").exists()
    for path in (tmp_path / "drive").rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(errors="ignore"), path
