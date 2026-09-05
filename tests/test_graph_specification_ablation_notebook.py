"""Contract tests for the graph-specification ablation harness.

The generated notebook is the runnable form of the multi-year protocol
pre-registered on ticket 181 and built out in ticket 183: six arms (the
ticket-166 five plus the scalar-edge top-K arm A3s), six folds (four core plus
the 2021 stress and 2025 bridge folds run alongside), every arm confirmed
unconditionally at 20 x 100 in every fold with no promotion gate, the isolated
edge-dropout RNG on across all arms, and a paired arbiter against the zeroed
control.

The load-bearing tests compose every (fold, arm) pair's overrides through Hydra
against the real ``configs/`` tree and construct the typed ``ExperimentConfig``
-- the cheap way to catch a job that would otherwise fail forty minutes into a
GPU run. A green run here proves structure and composition only; it is not
live-run evidence (see docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md).
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import create_config_from_dict
from mci_gru.graph.utils import edge_feature_dim

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "graph_specification_ablation_colab.ipynb"
GENERATOR_PATH = ROOT / "scripts" / "gen_graph_specification_ablation_nb.py"
CONFIG_DIR = ROOT / "configs"
BRIDGE_DATA_CONFIG = CONFIG_DIR / "data" / "gics_top10_110_2016.yaml"

EXPECTED_ARM_KEYS = [
    "A0_zeroed",
    "A1_shipped",
    "A2_thr05",
    "A3_topk20",
    "A3s_topk20_scalar",
    "A4_sector_only",
]
EXPECTED_CORE_FOLD_KEYS = ["F2022", "F2023", "F2024", "F2025"]
EXPECTED_ALONGSIDE_FOLD_KEYS = ["F2021_stress", "F2025_bridge"]
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


# -- structure ------------------------------------------------------------


def test_notebook_code_cells_parse_and_request_gpu() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"

    code_cells = _code_cell_sources()
    assert code_cells
    for source in code_cells:
        ast.parse(source)

    # The runtime preflight, per the ablation-notebook pattern: hard-refuse
    # T4/CPU runtimes and sample GPU utilisation for the run record.
    combined = "\n".join(_cell_sources())
    for token in [
        "REQUIRE_G4_L4_GPU = True",
        "G4/L4-class Colab runtime, not T4/CPU",
        "scripts/monitor_gpu_util.py",
    ]:
        assert token in combined, token


def test_notebook_regenerates_byte_identically(generator) -> None:
    assert generator.render() == NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_notebook_cell_order_is_the_protocol_order() -> None:
    headings = [source for source in _cell_sources() if source.startswith("## ")]
    assert headings == [
        "## 1. Setup",
        "## 2. FRED Key And Data Staging",
        "## 3. Protocol: Arms, Folds, Recipe Semantics, Budgets",
        "## 4. Jobs And Manifest",
        "## 5. Train The Jobs",
        "## 6. Collect The Arbiter Metrics",
        "## 7. Disclosure: Twin Check, Per-Span Density / Isolation, Graph Staleness",
        "## 8. Disclosure: Pooled Daily IC Per Year",
        "## 9. Paired Inference Against The Control",
        "## 10. Disclosure: Ensemble Scale, Sharpe Intervals, Basket Returns, April Composite",
        "## 11. Sanity Summary",
    ]


def test_notebook_defines_the_six_protocol_arms(generator) -> None:
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
        # The edge-width pin is rendered per arm, so the key rides the embedded
        # composer as a prefix; its value is asserted through composition below.
        "graph.use_multi_feature_edges=",
    ]
    for token in required_tokens:
        assert token in combined, token
        assert token in generator_text, token

    by_key = {arm["key"]: arm for arm in generator.ARMS}
    assert "graph.use_multi_feature_edges=false" in generator.held_fixed_overrides(
        by_key["A3s_topk20_scalar"]
    )
    for key in EXPECTED_ARM_KEYS:
        if key == "A3s_topk20_scalar":
            continue
        assert "graph.use_multi_feature_edges=true" in generator.held_fixed_overrides(
            by_key[key]
        ), key


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
        "graph.append_snapshot_age_days=false",
        "graph.use_lead_lag_features=false",
        "graph.drop_edge_p=0.1",
        "graph.isolate_edge_dropout_rng=true",
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
        "SMOKE_MODE = False",
    ]
    for token in required_tokens:
        assert token in combined, token
        assert token in generator_text, token


# -- folds ----------------------------------------------------------------


def test_fold_table_is_the_pre_registered_shape(generator) -> None:
    """Rolling five-year train, validation Y-1, test Y, 22-day gaps (ticket 181 s2)."""
    by_key = {fold["key"]: fold for fold in generator.FOLDS}
    assert [fold["key"] for fold in generator.FOLDS] == [
        *EXPECTED_CORE_FOLD_KEYS,
        *EXPECTED_ALONGSIDE_FOLD_KEYS,
    ]
    assert generator.CORE_FOLD_KEYS == EXPECTED_CORE_FOLD_KEYS
    assert generator.ALONGSIDE_FOLD_KEYS == EXPECTED_ALONGSIDE_FOLD_KEYS

    expected_core = {
        "F2022": ("2016-01-04", "2020-12-31", "2021-01-22", "2022-01-22"),
        "F2023": ("2017-01-01", "2021-12-31", "2022-01-22", "2023-01-22"),
        "F2024": ("2018-01-01", "2022-12-31", "2023-01-22", "2024-01-22"),
        "F2025": ("2019-01-01", "2023-12-31", "2024-01-22", "2025-01-22"),
    }
    for key, (train_start, train_end, val_start, test_start) in expected_core.items():
        fold = by_key[key]
        assert fold["pool"] == "core", key
        assert (fold["train_start"], fold["train_end"]) == (train_start, train_end), key
        assert (fold["val_start"], fold["val_end"]) == (val_start, f"{val_start[:4]}-12-31"), key
        assert (fold["test_start"], fold["test_end"]) == (test_start, f"{test_start[:4]}-12-31"), (
            key
        )
        assert fold["test_year"] == int(test_start[:4]), key

    # The 2021 stress fold: four training years, the COVID year as validation.
    stress = by_key["F2021_stress"]
    assert stress["pool"] == "stress"
    assert (stress["train_start"], stress["train_end"]) == ("2016-01-04", "2019-12-31")
    assert (stress["val_start"], stress["val_end"]) == ("2020-01-22", "2020-12-31")
    assert (stress["test_start"], stress["test_end"]) == ("2021-01-22", "2021-12-31")

    assert by_key["F2025_bridge"]["pool"] == "bridge"


def test_bridge_fold_reproduces_the_ticket_167_data_config_verbatim(generator) -> None:
    """The bridge fold is the one check that the new harness reproduces the old fold."""
    data_cfg = yaml.safe_load(BRIDGE_DATA_CONFIG.read_text(encoding="utf-8"))
    bridge = next(fold for fold in generator.FOLDS if fold["key"] == "F2025_bridge")
    for boundary in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
        assert bridge[boundary] == data_cfg[boundary], boundary


def test_alongside_folds_are_never_pooled_into_the_primary(generator) -> None:
    assert set(generator.CORE_FOLD_KEYS).isdisjoint(generator.ALONGSIDE_FOLD_KEYS)
    assert len(generator.CORE_FOLD_KEYS) == 4
    combined = "\n".join(_cell_sources())
    assert "CORE_FOLD_KEYS" in combined
    assert "never pooled" in combined


def test_base_seed_is_one_per_fold_and_shared_across_arms(generator) -> None:
    assert generator.BASE_SEED_ORIGIN == 1729
    assert generator.BASE_SEED_FOLD_STRIDE == 1000
    seeds = {fold["key"]: generator.fold_seed(fold) for fold in generator.FOLDS}
    assert seeds["F2022"] == 1729
    assert seeds["F2023"] == 2729
    assert seeds["F2024"] == 3729
    assert seeds["F2025"] == 4729
    # One seed per fold, and no two folds share one.
    assert len(set(seeds.values())) == len(generator.FOLDS)
    for fold in generator.FOLDS:
        expected = generator.BASE_SEED_ORIGIN + generator.BASE_SEED_FOLD_STRIDE * fold["index"]
        assert generator.fold_seed(fold) == expected, fold["key"]


def test_folds_pass_data_overrides_and_never_walkforward(generator) -> None:
    # The property is that no job composes a walkforward override: the fold
    # shape is fixed by data.* boundaries, not resolved by that path.
    for fold in generator.FOLDS:
        for arm in generator.ARMS:
            overrides = generator.arm_overrides(arm, fold, 1, 1)
            assert not [key for key in overrides if key.startswith("training.walkforward")], (
                fold["key"],
                arm["key"],
            )

    for fold in generator.FOLDS:
        overrides = generator.fold_overrides(fold)
        assert overrides == [
            f"data.train_start={fold['train_start']}",
            f"data.train_end={fold['train_end']}",
            f"data.val_start={fold['val_start']}",
            f"data.val_end={fold['val_end']}",
            f"data.test_start={fold['test_start']}",
            f"data.test_end={fold['test_end']}",
        ], fold["key"]


# -- composition: every (fold, arm) must construct a valid ExperimentConfig -


def test_every_fold_and_arm_composes_a_valid_experiment_config(generator) -> None:
    assert [arm["key"] for arm in generator.ARMS] == EXPECTED_ARM_KEYS

    budgets = [
        (generator.SCREEN_NUM_MODELS, generator.SCREEN_NUM_EPOCHS),
        (generator.CONFIRM_NUM_MODELS, generator.CONFIRM_NUM_EPOCHS),
    ]
    for fold in generator.FOLDS:
        for arm in generator.ARMS:
            for num_models, num_epochs in budgets:
                overrides = generator.arm_overrides(arm, fold, num_models, num_epochs)
                cfg = _compose_experiment_config(overrides)
                where = f"{fold['key']}/{arm['key']}"

                # The fold's splits survive Hydra composition into the typed config.
                assert cfg.data.train_start == fold["train_start"], where
                assert cfg.data.train_end == fold["train_end"], where
                assert cfg.data.val_start == fold["val_start"], where
                assert cfg.data.val_end == fold["val_end"], where
                assert cfg.data.test_start == fold["test_start"], where
                assert cfg.data.test_end == fold["test_end"], where
                # One base seed per fold, shared by every arm in that fold.
                assert cfg.seed == generator.fold_seed(fold), where

                # The hygiene rule survives composition.
                assert cfg.graph.exclude_edge_pairs == [["GOOG.OQ", "GOOGL.OQ"]], where
                # Recipe semantics preserved at every budget.
                assert cfg.training.loss_type == "ic", where
                assert cfg.training.label_type == "returns", where
                assert cfg.training.selection_metric == "val_ic", where
                assert cfg.training.num_models == num_models, where
                assert cfg.training.num_epochs == num_epochs, where
                assert cfg.training.early_stopping_patience == 15, where
                # Held-fixed graph invariants.
                assert cfg.graph.corr_lookback_days == 252, where
                assert cfg.graph.update_frequency_months == 0, where
                assert cfg.graph.drop_edge_p == 0.1, where
                assert cfg.graph.isolate_edge_dropout_rng is True, where


def test_a3s_arm_is_the_only_scalar_edge_width(generator) -> None:
    """A3s carries one edge channel; every other arm carries four (ticket 181 s3)."""
    by_key = {arm["key"]: arm for arm in generator.ARMS}
    fold = next(f for f in generator.FOLDS if f["key"] == "F2022")

    widths = {}
    for key, arm in by_key.items():
        cfg = _compose_experiment_config(
            generator.arm_overrides(arm, fold, generator.CONFIRM_NUM_MODELS, 1)
        )
        widths[key] = edge_feature_dim(cfg.graph)

    assert widths["A3s_topk20_scalar"] == 1
    assert all(width == 4 for key, width in widths.items() if key != "A3s_topk20_scalar"), widths

    a3s = _compose_experiment_config(
        generator.arm_overrides(by_key["A3s_topk20_scalar"], fold, 1, 1)
    )
    assert a3s.graph.use_multi_feature_edges is False
    assert a3s.graph.top_k == 20
    assert a3s.graph.top_k_metric == "corr"
    assert a3s.graph.zero_edges is False

    a3 = _compose_experiment_config(generator.arm_overrides(by_key["A3_topk20"], fold, 1, 1))
    assert a3.graph.use_multi_feature_edges is True
    assert a3.graph.top_k == 20


def test_arm_specifications_match_the_protocol(generator) -> None:
    by_key = {arm["key"]: arm for arm in generator.ARMS}
    fold = next(f for f in generator.FOLDS if f["key"] == "F2022")

    def cfg_for(key: str):
        return _compose_experiment_config(
            generator.arm_overrides(by_key[key], fold, generator.SCREEN_NUM_MODELS, 1)
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


# -- no promotion gate ----------------------------------------------------


def test_no_promotion_gate_survives_anywhere(generator) -> None:
    """Ticket 181 s8: every arm confirms unconditionally; the screen decides nothing."""
    combined = "\n".join(_cell_sources())
    generator_text = GENERATOR_PATH.read_text(encoding="utf-8")
    for token in ["ALWAYS_CONFIRMED_ARMS", "PROMOTED_ARMS", "separated_from_A0", "promotion.json"]:
        assert token not in combined, token
        assert token not in generator_text, token
    assert "no promotion gate" in combined


def test_confirm_runs_every_arm_in_every_fold(generator) -> None:
    assert generator.confirm_arm_keys() == EXPECTED_ARM_KEYS
    assert len(generator.confirm_arm_keys()) == len(generator.ARMS) == 6


def test_promotion_rule_string_is_data_driven(generator) -> None:
    """It went stale in the ticket-166 harness as prose; it is rendered from data now."""
    text = generator.promotion_rule_text(
        generator.ARMS, "screen", generator.SCREEN_NUM_MODELS, generator.SCREEN_NUM_EPOCHS
    )
    for arm in generator.ARMS:
        assert arm["key"] in text, arm["key"]
    assert "screen" in text
    assert f"{generator.CONFIRM_NUM_MODELS}x{generator.CONFIRM_NUM_EPOCHS}" in text

    # Drop an arm and the string must follow; that is what "data-driven" buys.
    fewer = generator.promotion_rule_text(
        generator.ARMS[:-1], "screen", generator.SCREEN_NUM_MODELS, generator.SCREEN_NUM_EPOCHS
    )
    assert fewer != text
    assert generator.ARMS[-1]["key"] not in fewer

    # So must the stage.
    other_stage = generator.promotion_rule_text(
        generator.ARMS, "confirm", generator.CONFIRM_NUM_MODELS, generator.CONFIRM_NUM_EPOCHS
    )
    assert other_stage != text


# -- outcome map ----------------------------------------------------------


def test_outcome_branch_implements_the_pre_registered_map(generator) -> None:
    """Ticket 181 s9, per arm, against the 0.005 line."""
    assert generator.EFFECT_WORTH_ACTING_ON == 0.005
    assert generator.SIGN_CONSISTENT_FOLDS_REQUIRED == 3

    cleared = generator.outcome_branch(
        bhy_p=0.01, mean_delta=0.006, sign_consistent_folds=3, realised_mde=0.004
    )
    assert cleared == "cleared"

    # Branch 1 needs all three conditions.
    assert (
        generator.outcome_branch(
            bhy_p=0.20, mean_delta=0.006, sign_consistent_folds=4, realised_mde=0.004
        )
        != "cleared"
    )
    assert (
        generator.outcome_branch(
            bhy_p=0.01, mean_delta=-0.006, sign_consistent_folds=4, realised_mde=0.004
        )
        != "cleared"
    )
    assert (
        generator.outcome_branch(
            bhy_p=0.01, mean_delta=0.006, sign_consistent_folds=2, realised_mde=0.004
        )
        != "cleared"
    )

    assert (
        generator.outcome_branch(
            bhy_p=0.90, mean_delta=-0.001, sign_consistent_folds=1, realised_mde=0.004
        )
        == "does_not_earn_its_edges"
    )
    assert (
        generator.outcome_branch(
            bhy_p=0.90, mean_delta=-0.001, sign_consistent_folds=1, realised_mde=0.007
        )
        == "undecidable_at_this_universe_and_horizon"
    )
    # The 0.005 line is inclusive on branch 2, per the ruling's "MDE <= 0.005".
    assert (
        generator.outcome_branch(
            bhy_p=0.90, mean_delta=-0.001, sign_consistent_folds=1, realised_mde=0.005
        )
        == "does_not_earn_its_edges"
    )
    # A missing p-value cannot clear.
    assert (
        generator.outcome_branch(
            bhy_p=float("nan"), mean_delta=0.006, sign_consistent_folds=4, realised_mde=0.004
        )
        != "cleared"
    )


# -- paired inference cell -------------------------------------------------


def test_paired_inference_cell_mirrors_the_pre_registered_arbiter(generator) -> None:
    combined = "\n".join(_cell_sources())

    assert generator.CONTROL_ARM == "A0_zeroed"
    assert [key for key in EXPECTED_ARM_KEYS if key != "A0_zeroed"] == generator.COMPARISON_ARMS
    assert generator.BHY_CONTRAST_COUNT == len(generator.COMPARISON_ARMS) == 5
    assert (generator.LABEL_T, generator.BLOCK_SIZE, generator.HAC_LAGS) == (5, 5, 4)
    assert generator.N_RESAMPLES == 1000
    assert generator.BOOTSTRAP_SEED == 1729
    assert generator.CI_LEVEL == 0.95
    assert (generator.POWER, generator.ALPHA) == (0.8, 0.05)

    for token in [
        "from mci_gru.evaluation.paired_inference import",
        "align_daily_series",
        "paired_daily_differences",
        "paired_mean_inference",
        "bhy_adjusted_p_values",
        "minimum_detectable_effect",
        # Primary, per fold and pooled over the core folds only.
        "paired_per_fold",
        "paired_pooled_core",
        # Pre-registered secondaries.
        "spearman",
        "per_model_ic",
        "sign_test",
        # Realised MDE against the 0.005 line, with the per-arm branch.
        "realised_mde",
        "outcome_branch",
        "EFFECT_WORTH_ACTING_ON",
        # Arm-versus-arm, descriptive and unadjusted.
        "descriptive",
    ]:
        assert token in combined, token


def test_paired_cell_pools_only_the_core_folds(generator) -> None:
    """The pooled primary iterates the guarded core-fold list and nothing else.

    Asserted structurally rather than by searching the cell for a loop header:
    the cell has several fold loops, so a text search still matched after the
    pooling loop alone was switched to every present fold (mutation M8 on
    ticket 183's pull request).
    """
    paired_cell = next(source for source in _code_cell_sources() if "paired_pooled_core" in source)
    tree = ast.parse(paired_cell)

    # POOLED_FOLD_KEYS is exactly list(CORE_FOLD_KEYS) -- read off the assignment
    # node, not by substring: `list(CORE_FOLD_KEYS) + list(ALONGSIDE_FOLD_KEYS)`
    # contains the substring and would otherwise pass (mutation M8b).
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "POOLED_FOLD_KEYS"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call), ast.dump(assignment.value)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "list"
    assert [ast.dump(arg) for arg in assignment.value.args] == [
        ast.dump(ast.Name(id="CORE_FOLD_KEYS", ctx=ast.Load()))
    ], ast.dump(assignment.value)

    # And the cell refuses to run if an alongside fold ever reaches that list.
    assert "set(POOLED_FOLD_KEYS) & set(ALONGSIDE_FOLD_KEYS)" in paired_cell

    pooled = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "pooled_deltas"
    )
    iterated = {
        node.iter.id
        for node in ast.walk(pooled)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
    }
    assert iterated == {"POOLED_FOLD_KEYS"}, iterated
    assert "FOLD_KEYS_PRESENT" not in ast.dump(pooled)
    assert "ALONGSIDE_FOLD_KEYS" not in ast.dump(pooled)

    # And the pooled table is built only through that function.
    pooled_rows = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "pooled_rows"
    )
    called = {
        node.func.id
        for node in ast.walk(pooled_rows)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "pooled_deltas" in called
    assert not [
        node
        for node in ast.walk(pooled_rows)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and "FOLD" in node.iter.id
    ]


# -- disclosures ----------------------------------------------------------


def test_notebook_reports_arbiter_composite_and_disclosures() -> None:
    combined = "\n".join(_cell_sources())

    required_tokens = [
        # Arbiter: test-span pooled daily IC with CI, per calendar year.
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
        # Ticket-183 disclosures.
        "staleness_jaccard",
        "ensemble_scale",
        "rank_average",
        "sharpe_block_bootstrap_ci",
        "paired_basket_return",
    ]
    for token in required_tokens:
        assert token in combined, token


def test_density_disclosure_spans_follow_the_fold(generator) -> None:
    """Ticket-166 pinned one hard-coded span table; spans are per fold now."""
    combined = "\n".join(_cell_sources())
    assert "def fold_spans(" in combined
    assert '"2016-01-04", "2023-12-31"' not in combined


# -- resume ---------------------------------------------------------------


def test_resume_is_keyed_on_fold_arm_and_stage(generator) -> None:
    combined = "\n".join(_cell_sources())
    assert "RESUME_COMPLETED_TRAINING = True" in combined
    assert "evaluation_summary.json" in combined

    fold = next(f for f in generator.FOLDS if f["key"] == "F2023")
    arm = next(a for a in generator.ARMS if a["key"] == "A3s_topk20_scalar")
    name = generator.job_name("confirm", fold, arm)
    assert "confirm" in name
    assert fold["key"] in name
    assert arm["key"] in name
    assert str(generator.fold_seed(fold)) in name

    # Every (fold, arm, stage) triple gets its own resume key.
    names = {
        generator.job_name(stage, f, a)
        for stage in ("screen", "confirm")
        for f in generator.FOLDS
        for a in generator.ARMS
    }
    assert len(names) == 2 * len(generator.FOLDS) * len(generator.ARMS)


def test_smoke_tag_is_not_an_input() -> None:
    """The smoke stage proves wiring; nothing downstream may read its artifacts."""
    code = "\n".join(_code_cell_sources())
    run_tag_cell = next(source for source in _code_cell_sources() if "RUN_TAG = " in source)
    assert "SMOKE_MODE" not in run_tag_cell.split("RUN_TAG = ")[1].split("\n")[0]
    assert "smoke" not in code.split("STAGES_FOR_ANALYSIS = ")[1].split("\n")[0].lower()
