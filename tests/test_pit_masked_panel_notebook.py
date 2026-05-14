import ast
import json
from pathlib import Path

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


def test_pit_masked_panel_notebook_uses_frozen_default_recipe() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
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


def test_pit_masked_panel_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
