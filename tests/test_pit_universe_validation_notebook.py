import ast
import json
from pathlib import Path

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
        "+data.use_pit_universe=true",
        "+data.pit_universe_csv=",
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
