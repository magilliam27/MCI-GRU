import ast
import importlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.requires_data

PRESETS = {
    "long_history_his_t_21": 21,
    "long_history_his_t_63": 63,
    "long_history_his_t_126": 126,
}
NOTEBOOK_PATH = Path("notebooks/long_history_pit_eval_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_long_history_pit_eval_nb.py")


def _load_preset(name: str):
    return OmegaConf.load(Path("configs/experiment") / f"{name}.yaml")


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


def test_long_history_presets_pin_frozen_recipe_semantics() -> None:
    for name, his_t in PRESETS.items():
        cfg = _load_preset(name)

        assert cfg.experiment_name == name
        assert cfg.model.his_t == his_t
        assert cfg.model.label_t == 5
        assert cfg.model.temporal_encoder == "gru_attn"
        assert cfg.training.num_models == 20
        assert cfg.training.num_epochs == 100
        assert cfg.training.early_stopping_patience == 15
        assert cfg.training.learning_rate == 5e-5
        assert cfg.training.lr_scheduler == "cosine"
        assert cfg.training.loss_type == "ic"
        assert cfg.training.label_type == "returns"
        assert cfg.training.selection_metric == "val_ic"
        assert cfg.training.shuffle_train is True
        assert cfg.graph.update_frequency_months == 0
        assert cfg.graph.corr_lookback_days == 252
        assert cfg.graph.top_k == 0
        assert cfg.graph.top_k_metric == "corr"
        assert cfg.graph.use_multi_feature_edges is True
        assert cfg.graph.append_snapshot_age_days is False
        assert cfg.graph.use_lead_lag_features is False
        assert cfg.graph.drop_edge_p == 0.1
        assert cfg.features.include_momentum is True
        assert cfg.features.include_weekly_momentum is True
        assert cfg.features.momentum_blend_mode == "static"
        assert cfg.features.include_global_regime is True
        assert cfg.features.regime_strict is True
        assert cfg.features.regime_include_subsequent_returns is False


def test_long_history_presets_do_not_make_252_first_pass() -> None:
    preset_names = {
        path.stem for path in Path("configs/experiment").glob("long_history_his_t_*.yaml")
    }

    assert set(PRESETS) <= preset_names
    assert "long_history_his_t_252" not in preset_names


def test_long_history_colab_notebook_is_generated_and_pit_scoped() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Long-History PIT Evaluation",
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        "BRANCH = 'codex/pit-universe-validation'",
        "REQUIRED_DATA_CONFIG_FIELDS",
        "pit_universe_mode",
        "pit_min_scoreable_stocks",
        "pit_breadth_policy",
        "does not support true PIT masked-panel fields",
        "HIS_T_VALUES = [10, 21, 63, 126]",
        "INCLUDE_HIS_T_252 = False",
        "YEARS = [2022, 2023, 2024, 2025]",
        "MAX_JOBS = None",
        "RUN_TRAINING = True",
        "RUN_BACKTESTS = True",
        "is_complete_training_run",
        "training_summary.json",
        "averaged_predictions",
        "Ignoring incomplete run dir",
        "latest_run_dir",
        "long_history_pit_eval_manifest.json",
        "training_results_interim.csv",
        "backtest_results_interim.csv",
        "grouped_his_t_summary.csv",
        "long_history_decision_table.csv",
        "Mechanics smoke rows are wiring evidence, not model-performance evidence.",
        "model.his_t={job['his_t']}",
        "data.pit_universe_mode=masked_panel",
        "his_t=10",
        "his_t=21",
        "his_t=63",
        "his_t=126",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_long_history_colab_writes_missing_pit_temporal_presets() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "def write_pit_experiment_presets() -> None:",
        "write_pit_experiment_presets()",
        "Wrote PIT preset:",
        "# Auto-written by long_history_pit_eval_colab.ipynb.",
        "pit_temporal_2022.yaml",
        "pit_temporal_2025.yaml",
        "pit_universe_mode: masked_panel",
        "pit_breadth_policy:",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_long_history_colab_setup_survives_existing_generated_presets() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "def clear_notebook_generated_repo_files() -> None:",
        "NOTEBOOK_GENERATED_REPO_FILES",
        "Unlinked generated repo file before branch checkout:",
        "configs/experiment/pit_temporal_2022.yaml",
        "configs/experiment/pit_temporal_2025.yaml",
        "'checkout', '-B', BRANCH, f'origin/{BRANCH}'",
        "'pull', '--ff-only', 'origin', BRANCH",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_long_history_colab_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)


def test_training_metrics_import_is_not_blocked_by_prediction_report_cycle() -> None:
    metrics = importlib.import_module("mci_gru.training.metrics")

    assert hasattr(metrics, "evaluate_predictions")


def test_long_history_docs_use_temporal_universe_set_for_non_pit_smoke() -> None:
    guide = Path("docs/CONFIGURATION_GUIDE.md").read_text(encoding="utf-8")

    assert "data=temporal_2019" in guide
    assert "sp500_2019_universe_data_through_2026.csv" in guide
    assert "non-PIT anchored historical snapshot universe" in guide
