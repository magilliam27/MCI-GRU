import ast
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
NOTEBOOK_PATH = Path("notebooks/pit_universe_validation_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_pit_universe_validation_nb.py")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def test_pit_notebook_includes_survivorship_controls() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "GENERATE_PIT_UNIVERSE",
        "export_sp500_joiner_leaver_pit.py",
        "_pit_universe.csv",
        "PIT_UNIVERSE_CSV",
        "data.use_pit_universe=true",
        "data.pit_universe_csv=",
        "data.filter_stocks_per_split=true",
        "data.pit_universe_mode=masked_panel",
        "+data.filter_stocks_per_split=true",
        "kdcode",
        "valid_from",
        "valid_to",
        "2026-05-13",
        "str.split('^'",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator

    assert "row_availability_fallback" not in combined
    assert "row_availability_fallback" not in generator
    assert "+data.use_pit_universe" not in combined
    assert "+data.pit_universe_csv" not in combined


def test_pit_notebook_writes_comparison_artifacts() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    expected_outputs = [
        "pit_universe_validation_manifest.json",
        "pit_training_results.csv",
        "pit_backtest_results_raw.csv",
        "pit_vs_baseline_decision_table.csv",
        "pit_pooled_daily_significance.csv",
        "pit_universe_validation_summary.md",
    ]

    for output_name in expected_outputs:
        assert output_name in combined
        assert output_name in generator


def test_pit_notebook_preserves_frozen_recipe_scope() -> None:
    combined = "\n".join(_cell_sources())

    assert "static-threshold-shuffle__pure-ic-returns-5d-val-ic" in combined
    assert "BASE_SEEDS = [1729, 2718, 3141]" in combined
    assert "TOP_K_VALUES = [15, 20]" in combined
    assert "COST_SCENARIOS" in combined
    assert "full" in combined
    assert "no_regime" in combined


def test_pit_notebook_code_cells_parse() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]

    assert code_cells
    for source in code_cells:
        ast.parse(source)


def _notebook_literal(name: str):
    """The literal assigned to ``name`` in the notebook's code cells."""
    for source in _cell_sources():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {NOTEBOOK_PATH}")


UNIVERSE_CONTROLS = _notebook_literal("UNIVERSE_CONTROLS")
PROOF_WINDOWS = _notebook_literal("PROOF_WINDOWS")


def test_universe_control_composition_guard_finds_the_pit_controls_and_windows() -> None:
    """The composition test below is not vacuous: PIT controls and data configs are found."""
    pit_controls = {control["name"] for control in UNIVERSE_CONTROLS if control["requires_pit"]}
    assert {"pit_universe", "pit_plus_per_split"} <= pit_controls
    assert {window["data_config"] for window in PROOF_WINDOWS} >= {"temporal_2016"}


@pytest.mark.parametrize("data_config", sorted({w["data_config"] for w in PROOF_WINDOWS}))
@pytest.mark.parametrize("control", UNIVERSE_CONTROLS, ids=lambda control: control["name"])
def test_universe_control_overrides_compose_against_the_window_data_config(
    control: dict, data_config: str
) -> None:
    """Every control's overrides compose with the notebook's own data configs (issue 248).

    ``+data.use_pit_universe`` failed here, because ``temporal_*`` already declares it.
    """
    overrides = [
        f"data={data_config}",
        *(item.format(pit_csv="pit_universe.csv") for item in control["overrides"]),
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    data = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True)).data

    assert data.use_pit_universe is control["requires_pit"], control["name"]
    if control["requires_pit"]:
        # masked_panel is also the default, so only the override itself shows the
        # control names its PIT mode.
        assert "++data.pit_universe_mode=masked_panel" in control["overrides"], control["name"]
        assert data.pit_universe_csv == "pit_universe.csv"
        assert data.pit_universe_mode == "masked_panel"
