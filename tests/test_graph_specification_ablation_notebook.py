"""Contract tests for the graph-specification ablation harness (ticket 166).

The generated notebook is the runnable form of the issue-164 protocol: five
screening arms, the twin-exclusion hygiene rule, screening at 3 seeds x 20
epochs with recipe semantics preserved, frozen-recipe confirmation for A0, A1,
and any arm separating from A0, and the pooled-daily-IC arbiter with the
April composite and per-span density/isolation disclosed alongside.

The load-bearing test composes every arm's overrides through Hydra against the
real ``configs/`` tree and constructs the typed ``ExperimentConfig`` — the
cheap way to catch an arm that would otherwise fail forty minutes into a GPU
run. A green run here proves structure and composition only; it is not
live-run evidence (see docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md).
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "graph_specification_ablation_colab.ipynb"
GENERATOR_PATH = ROOT / "scripts" / "gen_graph_specification_ablation_nb.py"
CONFIG_DIR = ROOT / "configs"

EXPECTED_ARM_KEYS = ["A0_zeroed", "A1_shipped", "A2_thr05", "A3_topk20", "A4_sector_only"]
TWIN_OVERRIDE = '+graph.exclude_edge_pairs=[["GOOG.OQ","GOOGL.OQ"]]'


@pytest.fixture(scope="module")
def generator():
    scripts_dir = str(ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "gen_graph_specification_ablation_nb", GENERATOR_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_dir)


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


def _compose_experiment_config(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


# ── structure ─────────────────────────────────────────────────────────────


def test_notebook_code_cells_parse_and_request_gpu() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"

    code_cells = _code_cell_sources()
    assert code_cells
    for source in code_cells:
        ast.parse(source)


def test_notebook_defines_the_five_protocol_arms() -> None:
    combined = "\n".join(_cell_sources())
    generator_text = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        *EXPECTED_ARM_KEYS,
        "+experiment=graph_zeroed",
        "+experiment=graph_thr05",
        "+experiment=graph_topk20_static",
        "+experiment=graph_sector_only",
        "graph.judge_value=0.8",
        "graph.top_k=0",
    ]
    for token in required_tokens:
        assert token in combined, token
        assert token in generator_text, token


def test_notebook_pins_recipe_semantics_held_fixed_keys_and_twin_exclusion() -> None:
    combined = "\n".join(_cell_sources())
    generator_text = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "data=gics_top10_110_2016",
        "features=with_momentum",
        "features.include_global_regime=true",
        "features.regime_include_subsequent_returns=false",
        "training.loss_type=ic",
        "training.label_type=returns",
        "training.selection_metric=val_ic",
        "training.shuffle_train=true",
        "model.label_t=5",
        "graph.corr_lookback_days=252",
        "graph.update_frequency_months=0",
        "graph.use_multi_feature_edges=true",
        "graph.append_snapshot_age_days=false",
        "graph.use_lead_lag_features=false",
        "graph.drop_edge_p=0.1",
        TWIN_OVERRIDE,
    ]
    for token in required_tokens:
        assert token in combined, token
        assert token in generator_text, token


def test_notebook_screen_and_confirmation_budgets() -> None:
    combined = "\n".join(_cell_sources())
    generator_text = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "SCREEN_NUM_MODELS = 3",
        "SCREEN_NUM_EPOCHS = 20",
        "CONFIRM_NUM_MODELS = 20",
        "CONFIRM_NUM_EPOCHS = 100",
        "EARLY_STOPPING_PATIENCE = 15",
        "BOOTSTRAP_RESAMPLES = 1000",
        "BASE_SEED = 1729",
        "SMOKE_MODE = False",
    ]
    for token in required_tokens:
        assert token in combined, token
        assert token in generator_text, token


def test_notebook_reports_arbiter_composite_and_disclosures() -> None:
    combined = "\n".join(_cell_sources())

    required_tokens = [
        # Arbiter: test-span pooled daily IC with CI, from the production
        # evaluation summary, reported per calendar year.
        "avg_ic_ci_lower",
        "avg_ic_ci_upper",
        "per_year",
        # April composite, computed alongside and labelled non-authoritative.
        "DECISION_SCORE_WEIGHTS",
        "avg_spearman_corr",
        "sharpe_top_20_newey_west",
        "return_top_20",
        "not the arbiter",
        # Per-span adjacency disclosure and the twin hygiene check.
        "isolated_fraction",
        "twin_edge_count",
        # Promotion rule.
        "PROMOTED_ARMS",
        "ALWAYS_CONFIRMED_ARMS",
    ]
    for token in required_tokens:
        assert token in combined, token


# ── composition: every arm must construct a valid ExperimentConfig ────────


def test_every_arm_composes_a_valid_experiment_config_at_both_budgets(generator) -> None:
    assert [arm["key"] for arm in generator.ARMS] == EXPECTED_ARM_KEYS

    budgets = [
        (generator.SCREEN_NUM_MODELS, generator.SCREEN_NUM_EPOCHS),
        (generator.CONFIRM_NUM_MODELS, generator.CONFIRM_NUM_EPOCHS),
    ]
    for arm in generator.ARMS:
        for num_models, num_epochs in budgets:
            overrides = generator.arm_overrides(arm, num_models, num_epochs)
            cfg = _compose_experiment_config(overrides)

            # The hygiene rule survives Hydra composition into the typed config.
            assert cfg.graph.exclude_edge_pairs == [["GOOG.OQ", "GOOGL.OQ"]], arm["key"]
            # Recipe semantics preserved at every budget.
            assert cfg.training.loss_type == "ic"
            assert cfg.training.label_type == "returns"
            assert cfg.training.selection_metric == "val_ic"
            assert cfg.training.num_models == num_models
            assert cfg.training.num_epochs == num_epochs
            assert cfg.training.early_stopping_patience == 15
            # Held-fixed graph invariants.
            assert cfg.graph.corr_lookback_days == 252
            assert cfg.graph.update_frequency_months == 0
            assert cfg.graph.use_multi_feature_edges is True
            assert cfg.graph.drop_edge_p == 0.1
            assert cfg.seed == 1729


def test_arm_specifications_match_the_protocol(generator) -> None:
    by_key = {arm["key"]: arm for arm in generator.ARMS}

    def cfg_for(key: str):
        arm = by_key[key]
        return _compose_experiment_config(
            generator.arm_overrides(arm, generator.SCREEN_NUM_MODELS, generator.SCREEN_NUM_EPOCHS)
        )

    a0 = cfg_for("A0_zeroed")
    assert a0.graph.zero_edges is True
    assert a0.graph.use_sector_relation is False

    a1 = cfg_for("A1_shipped")
    assert a1.graph.zero_edges is False
    assert a1.graph.judge_value == 0.8
    assert a1.graph.top_k == 0

    a2 = cfg_for("A2_thr05")
    assert a2.graph.zero_edges is False
    assert a2.graph.judge_value == 0.5
    assert a2.graph.top_k == 0

    a3 = cfg_for("A3_topk20")
    assert a3.graph.zero_edges is False
    assert a3.graph.top_k == 20
    assert a3.graph.top_k_metric == "corr"

    a4 = cfg_for("A4_sector_only")
    assert a4.graph.zero_edges is True
    assert a4.graph.use_sector_relation is True
    assert a4.graph.sector_map_csv is not None
    assert a4.graph.sector_map_csv.endswith("all_metadata_snapshots.csv")
