import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_fred

NOTEBOOK_PATH = Path("notebooks/pit_masked_panel_2022_2025_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_pit_masked_panel_2022_2025_nb.py")


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


def _notebook_function(name: str):
    for source in _code_cell_sources():
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace: dict[str, object] = {}
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        filename=f"<notebook:{name}>",
                        mode="exec",
                    ),
                    namespace,
                )
                return namespace[name]
    raise AssertionError(f"{name} function not found in notebook")


def test_pit_masked_panel_notebook_uses_frozen_default_recipe() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1",
        "SMOKE_MODE = False",
        "USE_GLOBAL_REGIME = True",
        "NUM_MODELS = 1 if SMOKE_MODE else 20",
        "EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 15",
        "LEARNING_RATE = '5e-5'",
        "'features=with_momentum'",
        "'features.include_weekly_momentum=true'",
        "'features.momentum_blend_mode=static'",
        "features.include_global_regime={str(USE_GLOBAL_REGIME).lower()}",
        "'features.regime_include_subsequent_returns=false'",
        "'graph.update_frequency_months=0'",
        "'graph.corr_lookback_days=252'",
        "'graph.top_k=0'",
        "'graph.top_k_metric=corr'",
        "'graph.use_multi_feature_edges=true'",
        "'graph.append_snapshot_age_days=false'",
        "'graph.use_lead_lag_features=false'",
        "'training.shuffle_train=true'",
        "training.loss_type={MODEL_RECIPE['loss_type']}",
        "training.label_type={MODEL_RECIPE['label_type']}",
        "training.selection_metric={MODEL_RECIPE['selection_metric']}",
        "model.label_t={MODEL_RECIPE['label_t']}",
        "graph.drop_edge_p={MODEL_RECIPE['drop_edge_p']}",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_pit_masked_panel_summary_text_reports_full_run_budget() -> None:
    build_run_mode_summary_lines = _notebook_function("build_run_mode_summary_lines")

    text = "\n".join(
        build_run_mode_summary_lines(
            smoke_mode=False,
            num_models=20,
            num_epochs=100,
            early_stopping_patience=15,
            use_global_regime=True,
            fred_api_key_set=True,
        )
    )

    assert "Run mode: `full`" in text
    assert "Model count: `20`" in text
    assert "Epoch cap: `100`" in text
    assert "Early stopping patience: `15`" in text
    assert "Global regime enabled: `True`" in text
    assert "FRED_API_KEY present: `True`" in text
    assert "docs/DEFAULT_EXPERIMENT_RECIPE.md" in text
    assert "one-epoch smoke runs" not in text
    assert "mechanics evidence" not in text


def test_pit_masked_panel_summary_text_keeps_smoke_caveat() -> None:
    build_run_mode_summary_lines = _notebook_function("build_run_mode_summary_lines")

    text = "\n".join(
        build_run_mode_summary_lines(
            smoke_mode=True,
            num_models=1,
            num_epochs=1,
            early_stopping_patience=2,
            use_global_regime=False,
            fred_api_key_set=False,
        )
    )

    assert "Run mode: `smoke`" in text
    assert "Model count: `1`" in text
    assert "Epoch cap: `1`" in text
    assert "Early stopping patience: `2`" in text
    assert "Global regime enabled: `False`" in text
    assert "FRED_API_KEY present: `False`" in text
    assert "one-epoch smoke runs are mechanics evidence" in text
    assert "not model-performance evidence" in text


def test_pit_masked_panel_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
