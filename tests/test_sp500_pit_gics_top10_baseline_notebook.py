import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_lseg

NOTEBOOK_PATH = Path("notebooks/sp500_pit_gics_top10_baseline_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_sp500_pit_gics_top10_baseline_nb.py")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _code_cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_reduced_pit_baseline_notebook_pins_multiyear_windows_and_data_gate() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "YEARS = [2022, 2023, 2024]",
        "REQUIRE_APPLES_TO_APPLES_SELECTOR_HISTORY = True",
        "'train_start': '2016-01-01'",
        "'train_end': '2020-12-31'",
        "'val_start': '2021-01-08'",
        "'val_end': '2021-12-31'",
        "'test_start': '2022-01-08'",
        "'test_end': '2022-12-31'",
        "'train_start': '2017-01-01'",
        "'train_end': '2021-12-31'",
        "'val_start': '2022-01-08'",
        "'test_start': '2023-01-08'",
        "'train_start': '2018-01-01'",
        "'train_end': '2022-12-31'",
        "'val_start': '2023-01-08'",
        "'test_start': '2024-01-08'",
        "selector snapshots begin after required train_start",
        "not apples-to-apples",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_reduced_pit_baseline_notebook_uses_extended_selector_bundle() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_",
        "lseg_20150101_20260622.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_reduced_pit_baseline_notebook_preserves_frozen_recipe_and_pit_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1",
        "NUM_MODELS = 20",
        "NUM_EPOCHS = 100",
        "EARLY_STOPPING_PATIENCE = 15",
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}",
        "data.pit_breadth_policy=error",
        "training.loss_type=ic",
        "training.label_type=returns",
        "training.selection_metric=val_ic",
        "training.shuffle_train=true",
        "model.label_t=5",
        "graph.update_frequency_months=0",
        "graph.top_k=0",
        "graph.use_multi_feature_edges=true",
        "graph.drop_edge_p=0.1",
        "features.include_global_regime=true",
        "features.regime_strict=true",
        "features.regime_include_subsequent_returns=false",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_reduced_pit_baseline_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
