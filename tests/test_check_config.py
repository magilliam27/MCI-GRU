"""Contract tests for scripts/check_config.py.

The pre-flight config checker must only report things it actually verified. It
previously read `evaluate_sp500.py`, which is not in the tree, so one section
always fell into its except branch and emitted a permanent bogus warning; it
probed `LSEG_API_KEY`, which gates nothing; and it scanned a hard-coded list of
legacy CSV names that no current data config points at. See issue 146.

Every assertion here is paired with a case in which it must fail, so that a
regression cannot pass silently.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

from omegaconf import OmegaConf

import scripts.check_config as check_config_module
from scripts.check_config import (
    FALLBACK_DATA_GROUP,
    check_config,
    find_configured_csv,
    selected_data_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_config.py"

GOOD_DATES = textwrap.dedent(
    """
    train_start: "2019-01-01"
    train_end: "2023-12-31"
    val_start: "2024-01-08"
    val_end: "2024-12-31"
    test_start: "2025-01-08"
    test_end: "2025-12-31"
    """
).strip()


def _config_tree(tmp_path: Path, data_group: str, data_body: str | None, csv: str | None = None):
    """Build a synthetic Hydra config tree and return its root."""
    (tmp_path / "configs" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "features").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "config.yaml").write_text(
        f"defaults:\n  - data: {data_group}\n  - features: with_momentum\n  - _self_\n"
        "hydra:\n  run:\n    dir: outputs/x\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "features" / "with_momentum.yaml").write_text(
        "x: 1\n", encoding="utf-8"
    )
    if data_body is not None:
        (tmp_path / "configs" / "data" / f"{data_group}.yaml").write_text(
            data_body, encoding="utf-8"
        )
    if csv is not None:
        target = tmp_path / csv
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("dt,kdcode,close\n", encoding="utf-8")
    return tmp_path


def _run(tree: Path) -> tuple[str, int]:
    """Run the real script as a subprocess with cwd inside the synthetic tree."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tree,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return proc.stdout + proc.stderr, proc.returncode


# --- The data config actually selected, not a hard-coded guess ---------------


def test_selected_data_config_follows_the_defaults_list():
    cfg = OmegaConf.create({"defaults": [{"data": "gics_top10_110_2016"}, "_self_"]})
    assert selected_data_config(cfg) == "configs/data/gics_top10_110_2016.yaml"

    # Control: a different selection must produce a different path, so this
    # cannot be satisfied by returning a constant.
    other = OmegaConf.create({"defaults": [{"data": "csv_sp500"}]})
    assert selected_data_config(other) == "configs/data/csv_sp500.yaml"


def test_selected_data_config_falls_back_when_defaults_name_no_data_group():
    for cfg in (
        OmegaConf.create({}),
        OmegaConf.create({"defaults": []}),
        OmegaConf.create({"defaults": ["_self_", {"features": "with_momentum"}]}),
    ):
        assert selected_data_config(cfg) == f"configs/data/{FALLBACK_DATA_GROUP}.yaml"


def test_repository_default_config_resolves_to_a_data_config_that_exists():
    """The committed base config must name a data config that is really there."""
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    assert (REPO_ROOT / selected_data_config(cfg)).exists()


def test_fallback_data_group_names_a_config_that_exists():
    """The fallback fires only when defaults name no data group, so a typo in it
    would otherwise sit unnoticed until exactly that edge case."""
    assert (REPO_ROOT / f"configs/data/{FALLBACK_DATA_GROUP}.yaml").exists()


def test_the_base_default_is_a_csv_source_not_lseg():
    """The base experiment must not select an LSEG config.

    `configs/data/sp500.yaml` was the default and is `source: lseg`, so a bare
    run selected a live-LSEG config for a data path this project does not use.
    """
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    data_cfg = OmegaConf.load(REPO_ROOT / selected_data_config(cfg))
    assert data_cfg.source == "csv", f"base default is source={data_cfg.source}"

    # Control: the config this replaced really is the lseg one, so the
    # assertion above is discriminating rather than trivially true.
    assert OmegaConf.load(REPO_ROOT / "configs/data/sp500.yaml").source == "lseg"


# --- CSV presence is answered against the configured filename ----------------


def test_find_configured_csv_locates_a_present_file_and_reports_an_absent_one(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mine.csv").write_text("dt\n", encoding="utf-8")

    assert find_configured_csv("data/mine.csv") is not None
    # Control: the same call for a file that does not exist must return None.
    assert find_configured_csv("data/absent.csv") is None


# --- The three defects this ticket fixes, each guarded ------------------------


def _literal_open_paths(source: str) -> list[str]:
    """Return every string literal the source passes as open()'s first argument."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if (
            name == "open"
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append(node.args[0].value)
    return found


def test_the_script_opens_no_path_that_is_absent_from_the_tree():
    """The original defect: reading a file that is not in the repository."""
    missing = [
        p
        for p in _literal_open_paths(SCRIPT.read_text(encoding="utf-8"))
        if not (REPO_ROOT / p).exists()
    ]
    assert not missing, f"check_config.py opens paths that do not exist: {missing}"


def test_the_absent_path_detector_actually_detects(tmp_path):
    """Control: without this, the test above passes for a script that opens nothing."""
    bad = 'with open("evaluate_sp500.py", encoding="utf-8") as f:\n    content = f.read()\n'
    assert _literal_open_paths(bad) == ["evaluate_sp500.py"]
    assert not (REPO_ROOT / "evaluate_sp500.py").exists(), (
        "evaluate_sp500.py is back in the tree; this ticket's premise has changed"
    )


def test_the_lseg_branch_probes_the_import_rather_than_an_env_var():
    source = SCRIPT.read_text(encoding="utf-8")
    # LSEG_API_KEY gates nothing; LSEGLoader.connect() imports refinitiv.data.
    assert "LSEG_API_KEY" not in source
    assert 'find_spec("refinitiv.data")' in source


def test_absent_refinitiv_package_is_a_warning_not_a_validation_error(monkeypatch):
    """`find_spec` on a dotted name imports the parent and raises when it is absent.

    Letting that escape turned "the optional LSEG SDK is not installed" into a
    hard error. CI has no `refinitiv` package and caught this; a developer
    machine that has it installed never reaches this path, so it is forced.
    """

    def raise_missing_parent(name):
        raise ModuleNotFoundError(f"No module named 'refinitiv' (probing {name})")

    monkeypatch.setattr(check_config_module.importlib.util, "find_spec", raise_missing_parent)
    assert check_config_module.refinitiv_data_available() is False

    # Control: the same helper must still report True when the spec resolves.
    monkeypatch.setattr(check_config_module.importlib.util, "find_spec", lambda name: object())
    assert check_config_module.refinitiv_data_available() is True


def test_run_emits_no_unreadable_file_warning_for_any_data_source(tmp_path):
    """A source the script cannot inspect must not produce a bogus warning."""
    for index, (source, filename) in enumerate(
        [("csv", "data/mine.csv"), ("lseg", "null"), ("parquet", "x.parquet")]
    ):
        tree = _config_tree(
            tmp_path / f"case{index}",
            "some_group",
            f"universe: sp500\nsource: {source}\nfilename: {filename}\n{GOOD_DATES}\n",
        )
        out, _ = _run(tree)
        assert "Could not check" not in out, out
        assert "No known data files" not in out, out


def test_data_file_warning_is_gated_on_the_csv_source(tmp_path):
    lseg = _config_tree(
        tmp_path / "lseg",
        "g",
        f"universe: sp500\nsource: lseg\nfilename: null\n{GOOD_DATES}\n",
    )
    out, code = _run(lseg)
    assert "CSV file not found" not in out, out
    assert code == 0

    # Control: the identical tree on the csv source, with the file missing,
    # must warn -- otherwise the gate above proves nothing.
    csv = _config_tree(
        tmp_path / "csv",
        "g",
        f"universe: sp500\nsource: csv\nfilename: data/mine.csv\n{GOOD_DATES}\n",
    )
    out, _ = _run(csv)
    assert "CSV file not found at configured path: data/mine.csv" in out, out


def test_missing_selected_data_config_is_a_hard_error(tmp_path):
    tree = _config_tree(tmp_path / "ghost", "ghost_group", None)
    out, code = _run(tree)
    assert "configs/data/ghost_group.yaml NOT FOUND" in out, out
    assert code == 1

    # Control: the same tree with that config present must succeed.
    present = _config_tree(
        tmp_path / "present",
        "ghost_group",
        f"universe: sp500\nsource: lseg\nfilename: null\n{GOOD_DATES}\n",
    )
    out, code = _run(present)
    assert "NOT FOUND" not in out, out
    assert code == 0


def test_out_of_order_dates_still_fail(tmp_path):
    """The checker must retain a way to fail, or none of the above means anything."""
    bad = GOOD_DATES.replace('val_start: "2024-01-08"', 'val_start: "2018-01-08"')
    tree = _config_tree(
        tmp_path / "dates", "g", f"universe: sp500\nsource: lseg\nfilename: null\n{bad}\n"
    )
    out, code = _run(tree)
    assert "Dates are not in chronological order" in out, out
    assert code == 1


def test_importing_the_module_does_not_rebind_stdout():
    """Importing must not clobber the caller's stdout (it would break capture).

    This has to run in a fresh interpreter: this test module imports
    `scripts.check_config` at the top, so by the time any test body runs the
    import has already happened and an in-process check could never observe it.
    Reported over stderr, which the module never rebinds.
    """
    probe = (
        "import sys; before = sys.stdout; "
        "import scripts.check_config; "
        "sys.stderr.write('REBOUND' if sys.stdout is not before else 'INTACT')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert "INTACT" in proc.stderr, proc.stderr
    assert callable(check_config)
