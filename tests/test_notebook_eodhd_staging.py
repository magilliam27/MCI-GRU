"""EODHD S&P 500 staging in the regime-enabled Colab notebooks (#278).

``configs/features/with_momentum.yaml`` points ``features.regime_market_csv`` at
an EODHD index file under the gitignored ``data/raw``, so a Colab checkout has
to copy it from Drive before any run that enables global regime features. The
helper source comes from ``scripts/nb_lib.py``; these tests pin that every
regime-enabled generator embeds it, gates the call on the notebook's own regime
toggle, stages before the first training run, and that the helper itself copies,
verifies, and refuses to copy or replace mismatched bytes.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
STAGING_CALL = "stage_eodhd_market_file(REPO_DIR)"
STAGING_FIRST_LINE = "# EODHD S&P 500 index file for the regime market input"


def _load_nb_lib():
    spec = importlib.util.spec_from_file_location("nb_lib", ROOT / "scripts" / "nb_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb_lib = _load_nb_lib()

FULL_VARIANT = {"enabled": True, "name": "full", "overrides": []}
NO_REGIME_VARIANT = {
    "enabled": True,
    "name": "no_regime",
    "overrides": [
        "features.include_global_regime=false",
        "features.regime_strict=false",
        "features.regime_include_subsequent_returns=false",
    ],
}

# notebook -> (toggle namespaces with whether the call must stage, tokens tying
# those toggles to the runs' regime overrides)
NOTEBOOKS: dict[str, tuple[list[tuple[dict, bool]], list[str]]] = {
    "graph_specification_ablation_colab.ipynb": (
        [({}, True)],
        ["features=with_momentum", "features.include_global_regime=true"],
    ),
    "long_history_pit_eval_colab.ipynb": (
        [({}, True)],
        ["features=with_momentum", "features.include_global_regime=true"],
    ),
    "sp500_pit_gics_top10_baseline_colab.ipynb": (
        [({}, True)],
        ["features=with_momentum", "features.include_global_regime=true"],
    ),
    "rolling_temporal_backtest_colab.ipynb": (
        [
            ({"REGIME_INPUTS_CSV": ""}, True),
            ({"REGIME_INPUTS_CSV": "data/raw/regime/legacy.csv"}, False),
        ],
        [
            "features.include_global_regime=true",
            "BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')",
        ],
    ),
    "performance_proof_tests_colab.ipynb": (
        [
            ({"REGIME_INPUTS_CSV": "", "MODEL_VARIANTS": [FULL_VARIANT, NO_REGIME_VARIANT]}, True),
            (
                {
                    "REGIME_INPUTS_CSV": "data/raw/regime/legacy.csv",
                    "MODEL_VARIANTS": [FULL_VARIANT, NO_REGIME_VARIANT],
                },
                False,
            ),
            (
                {
                    "REGIME_INPUTS_CSV": "",
                    "MODEL_VARIANTS": [{**FULL_VARIANT, "enabled": False}, NO_REGIME_VARIANT],
                },
                False,
            ),
        ],
        [
            "features.include_global_regime=true",
            "'features.include_global_regime=false',",
            "BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')",
        ],
    ),
    "pit_masked_panel_2022_2025_colab.ipynb": (
        [({"USE_GLOBAL_REGIME": True}, True), ({"USE_GLOBAL_REGIME": False}, False)],
        ["features.include_global_regime={str(USE_GLOBAL_REGIME).lower()}"],
    ),
    "pit_universe_validation_colab.ipynb": (
        [
            ({"REGIME_STRICT": True, "REGIME_INPUTS_CSV": ""}, True),
            ({"REGIME_STRICT": False, "REGIME_INPUTS_CSV": ""}, False),
            ({"REGIME_STRICT": True, "REGIME_INPUTS_CSV": "data/raw/regime/legacy.csv"}, False),
        ],
        [
            "features.include_global_regime={str(REGIME_STRICT).lower()}",
            "BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')",
        ],
    ),
    "pit_repeated_seed_replication_colab.ipynb": (
        [({"USE_STATIC_REGIME_INPUTS": False}, True), ({"USE_STATIC_REGIME_INPUTS": True}, False)],
        [
            "features.include_global_regime=true",
            "BASE_OVERRIDES.append(f'features.regime_inputs_csv={STATIC_REGIME_INPUTS_RELATIVE_PATH}')",
            "if not USE_STATIC_REGIME_INPUTS:\n        return None, '', {}",
        ],
    ),
}

# Toggles that stage are run against a checkout that reads the file and one from
# before #276 that does not; toggles that skip get no config at all, which also
# shows the config is only read once the toggle asks for staging.
TOGGLE_CASES = [
    pytest.param(
        name, toggles, checkout, id=f"{name.removesuffix('_colab.ipynb')}-{index}-{checkout}"
    )
    for name, (cases, _) in NOTEBOOKS.items()
    for index, (toggles, stages) in enumerate(cases)
    for checkout in (("reads", "pre276") if stages else ("toggle-off",))
]


def _code_cells(name: str) -> list[str]:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _staging_block(name: str) -> tuple[int, int, str]:
    """(cell index, offset of the call, source from the helper through its call)."""
    cells = _code_cells(name)
    defining = [i for i, source in enumerate(cells) if "def stage_eodhd_market_file(" in source]
    assert len(defining) == 1, (name, defining)
    index = defining[0]
    source = cells[index]
    start = source.index(STAGING_FIRST_LINE)
    call = source.index(STAGING_CALL, start)
    assert source.count(STAGING_CALL) == 1, name
    # The block runs on through the indented lines and else/elif around the call.
    head, _, rest = source[start:].partition(STAGING_CALL)
    block = head + STAGING_CALL
    for line in rest.split("\n")[1:]:
        if not line.startswith((" ", "else:", "elif ")):
            break
        block += "\n" + line
    return index, call, block


def _module_literals(source: str) -> dict:
    literals = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                literals[target.id] = node.value.value
    return literals


def _first(cells: list[str], token: str) -> tuple[float, float]:
    for index, source in enumerate(cells):
        if token in source:
            return index, source.index(token)
    return float("inf"), float("inf")


def test_helper_pins_the_file_with_momentum_reads() -> None:
    features = yaml.safe_load((CONFIG_DIR / "features" / "with_momentum.yaml").read_text())
    assert features["regime_market_csv"] == nb_lib.EODHD_MARKET_RELATIVE
    # The file sits in its own folder beside the LSEG package, not inside it.
    assert nb_lib.EODHD_MARKET_DRIVE_PATH.endswith(
        nb_lib.EODHD_MARKET_RELATIVE.removeprefix("data/raw/market")
    )
    assert "2026-10-02-110-universe-2016" not in nb_lib.EODHD_MARKET_DRIVE_PATH


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_notebook_defines_the_pinned_staging_helper(name: str) -> None:
    _, _, block = _staging_block(name)
    literals = _module_literals(block)
    assert literals["EODHD_MARKET_SHA256"] == nb_lib.EODHD_MARKET_SHA256
    assert literals["EODHD_MARKET_SIZE"] == nb_lib.EODHD_MARKET_SIZE
    assert literals["EODHD_MARKET_RELATIVE"] == nb_lib.EODHD_MARKET_RELATIVE
    assert nb_lib.eodhd_market_staging_source() in block
    assert f'_EodhdPath("{nb_lib.EODHD_MARKET_DRIVE_PATH}")' in block


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_notebook_stages_after_clone_and_before_the_first_training_run(name: str) -> None:
    cells = _code_cells(name)
    index, call, _ = _staging_block(name)
    assert _first(cells, "drive.mount(") < (index, call)
    assert _first(cells, "REPO_DIR = ") < (index, call)
    assert min(_first(cells, "'clone'"), _first(cells, '"clone"')) < (index, call)
    training = _first(cells, "run_experiment.py")
    assert training != (float("inf"), float("inf")), name
    assert (index, call) < training


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_notebook_gate_variables_drive_the_runs_regime_overrides(name: str) -> None:
    combined = "\n".join(_code_cells(name))
    for token in NOTEBOOKS[name][1]:
        assert token in combined, (name, token)
    if not any("regime_inputs_csv" in token for token in NOTEBOOKS[name][1]):
        assert "regime_inputs_csv" not in combined, name


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_notebook_markdown_names_the_drive_path(name: str) -> None:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert f"`{nb_lib.EODHD_MARKET_DRIVE_PATH}`" in markdown


def _write_features_config(repo: Path, *, market_csv_line: str | None) -> None:
    """The real with_momentum config, with its regime_market_csv line replaced."""
    lines = (CONFIG_DIR / "features" / "with_momentum.yaml").read_text().splitlines()
    kept = [line for line in lines if not line.startswith("regime_market_csv:")]
    assert len(kept) == len(lines) - 1
    if market_csv_line is not None:
        kept.append(market_csv_line)
    path = repo / "configs" / "features" / "with_momentum.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


@pytest.mark.parametrize(("name", "toggles", "checkout"), TOGGLE_CASES)
def test_notebook_staging_call_follows_its_regime_toggle_and_checkout(
    name: str, toggles: dict, checkout: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    if Path(nb_lib.EODHD_MARKET_DRIVE_PATH).exists():
        pytest.skip("the real Drive file is mounted here, so a missing-file stop cannot be shown")
    _, _, block = _staging_block(name)
    namespace = {"REPO_DIR": tmp_path, **toggles}
    if checkout == "reads":
        _write_features_config(
            tmp_path, market_csv_line=f"regime_market_csv: {nb_lib.EODHD_MARKET_RELATIVE}"
        )
        with pytest.raises(FileNotFoundError, match="not on Drive at"):
            exec(compile(block, name, "exec"), namespace)
    else:
        if checkout == "pre276":
            _write_features_config(tmp_path, market_csv_line=None)
        exec(compile(block, name, "exec"), namespace)
        printed = capsys.readouterr().out
        assert printed.startswith("Skipped EODHD S&P 500 staging: "), printed
        assert printed.count("\n") == 1, printed
        if checkout == "pre276":
            assert "does not set regime_market_csv" in printed
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "name", ["rolling_temporal_backtest_colab.ipynb", "performance_proof_tests_colab.ipynb"]
)
def test_legacy_regime_csv_runs_clear_the_unread_market_file(name: str) -> None:
    combined = "\n".join(_code_cells(name))
    assert (
        "if REGIME_INPUTS_CSV:\n"
        "    BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')\n"
        "    # The legacy CSV supplies the market variable, so the EODHD file is not read.\n"
        "    BASE_OVERRIDES.append('features.regime_market_csv=null')\n"
    ) in combined


def test_null_market_file_override_composes_beside_a_legacy_regime_csv() -> None:
    overrides = [
        "features=with_momentum",
        "features.include_global_regime=true",
        "features.regime_inputs_csv=data/raw/regime/legacy.csv",
        "features.regime_market_csv=null",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    features = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True)).features
    assert features.regime_inputs_csv == "data/raw/regime/legacy.csv"
    assert features.regime_market_csv is None


# -- the helper itself --------------------------------------------------------

PAYLOAD = b"dt,close\n2010-01-04,1132.99\n2010-01-05,1136.52\n"


def _helper(monkeypatch: pytest.MonkeyPatch, payload: bytes = PAYLOAD) -> dict:
    namespace: dict = {}
    exec(nb_lib.eodhd_market_staging_source(), namespace)
    monkeypatch.setitem(namespace, "EODHD_MARKET_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(namespace, "EODHD_MARKET_SIZE", len(payload))
    return namespace


def _drive_file(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "drive" / "sp500_index.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def test_helper_source_defines_and_does_not_call_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _helper(monkeypatch)
    assert callable(namespace["stage_eodhd_market_file"])
    assert STAGING_CALL not in nb_lib.eodhd_market_staging_source()


def _is_read(monkeypatch: pytest.MonkeyPatch, repo: Path) -> bool:
    return _helper(monkeypatch)["eodhd_market_file_is_read"](repo)


def test_helper_finds_the_file_read_by_this_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _is_read(monkeypatch, ROOT) is True


def test_helper_reports_a_checkout_without_the_key_as_not_reading_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_features_config(tmp_path, market_csv_line=None)
    assert _is_read(monkeypatch, tmp_path) is False


def test_helper_reports_a_null_market_file_as_not_reading_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_features_config(tmp_path, market_csv_line="regime_market_csv: null")
    assert _is_read(monkeypatch, tmp_path) is False


def test_helper_reports_the_pinned_market_file_as_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_features_config(
        tmp_path, market_csv_line=f"regime_market_csv: {nb_lib.EODHD_MARKET_RELATIVE}"
    )
    assert _is_read(monkeypatch, tmp_path) is True


def test_helper_refuses_a_checkout_naming_another_market_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other = "data/raw/market/eodhd_sp500_2010_20270101/sp500_index.csv"
    _write_features_config(tmp_path, market_csv_line=f"regime_market_csv: {other}")
    with pytest.raises(RuntimeError, match="names regime_market_csv") as caught:
        _is_read(monkeypatch, tmp_path)
    assert other in str(caught.value)
    assert nb_lib.EODHD_MARKET_RELATIVE in str(caught.value)


def test_helper_copies_and_verifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stage = _helper(monkeypatch)["stage_eodhd_market_file"]
    drive = _drive_file(tmp_path, PAYLOAD)
    repo = tmp_path / "repo"

    target = stage(repo, drive_path=drive)

    assert target == repo / nb_lib.EODHD_MARKET_RELATIVE
    assert target.read_bytes() == PAYLOAD
    # A second run finds the verified file in place and leaves it alone.
    drive.unlink()
    assert stage(repo, drive_path=drive) == target
    assert target.read_bytes() == PAYLOAD


def test_helper_names_a_missing_drive_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stage = _helper(monkeypatch)["stage_eodhd_market_file"]
    missing = tmp_path / "drive" / "sp500_index.csv"
    repo = tmp_path / "repo"

    with pytest.raises(FileNotFoundError, match="not on Drive at") as caught:
        stage(repo, drive_path=missing)

    assert str(missing) in str(caught.value)
    assert not (repo / nb_lib.EODHD_MARKET_RELATIVE).exists()


def test_helper_refuses_a_drive_file_with_other_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = _helper(monkeypatch)["stage_eodhd_market_file"]
    drive = _drive_file(tmp_path, PAYLOAD.replace(b"1132.99", b"1132.98"))
    repo = tmp_path / "repo"

    with pytest.raises(RuntimeError, match="does not match the pinned pull"):
        stage(repo, drive_path=drive)

    assert not (repo / nb_lib.EODHD_MARKET_RELATIVE).exists()


def test_helper_never_replaces_a_mismatched_file_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = _helper(monkeypatch)["stage_eodhd_market_file"]
    drive = _drive_file(tmp_path, PAYLOAD)
    repo = tmp_path / "repo"
    target = repo / nb_lib.EODHD_MARKET_RELATIVE
    target.parent.mkdir(parents=True)
    stale = b"dt,close\n2010-01-04,1.0\n"
    target.write_bytes(stale)

    with pytest.raises(RuntimeError, match="does not match the pinned pull"):
        stage(repo, drive_path=drive)

    assert target.read_bytes() == stale


def test_helper_checks_the_copy_after_writing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = _helper(monkeypatch)
    drive = _drive_file(tmp_path, PAYLOAD)
    repo = tmp_path / "repo"

    class _TruncatingShutil:
        @staticmethod
        def copyfile(source, target):
            Path(target).write_bytes(Path(source).read_bytes()[:-1])

    monkeypatch.setitem(namespace, "_eodhd_shutil", _TruncatingShutil)

    with pytest.raises(RuntimeError, match="does not match the pinned pull"):
        namespace["stage_eodhd_market_file"](repo, drive_path=drive)
