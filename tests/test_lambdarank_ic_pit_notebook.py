import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/lambdarank_ic_pit_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_lambdarank_ic_pit_nb.py")


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


def test_lambdarank_ic_notebook_pins_branch_and_grid_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "LambdaRankIC Pairwise Rank IC PIT Grid",
        "LOSS_PATH_DECISION_2026-06-04.md",
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        'BRANCH = "codex/lambdarank-ic-colab"',
        "SMOKE_MODE = True",
        "SMOKE_YEARS = [2025]",
        "FULL_YEARS = [2022, 2023, 2024, 2025]",
        "SMOKE_BASE_SEEDS = [314159]",
        "FULL_BASE_SEEDS = [314159, 271828, 161803]",
        "NUM_MODELS = 1 if SMOKE_MODE else 20",
        "NUM_EPOCHS = 1 if SMOKE_MODE else 100",
        "EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 15",
        "EXPECTED_JOB_COUNT = len(YEARS) * len(BASE_SEEDS) * len(OBJECTIVE_VARIANTS)",
        "EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS",
        "assert EXPECTED_JOB_COUNT == (3 if SMOKE_MODE else 36)",
        "assert EXPECTED_TOTAL_MODELS == (3 if SMOKE_MODE else 720)",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_lambdarank_ic_notebook_emits_three_objective_variants() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "'pure_ic_baseline'",
        "'portfolio_ic_hybrid'",
        "'lambdarank_ic_candidate'",
        "'loss_type': 'ic'",
        "'selection_metric': 'val_ic'",
        "'loss_type': 'portfolio_ic'",
        "'selection_metric': 'val_loss'",
        "'portfolio_ic_top_k': 10",
        "'portfolio_ic_weight': 0.25",
        "'portfolio_ic_temperature': 0.25",
        "'loss_type': 'lambdarank_ic'",
        "'selection_metric': 'val_rank_ic'",
        "'lambdarank_ic_max_pairs_per_day': 4096",
        "'lambdarank_ic_temperature': 1.0",
        "training.loss_type={variant['loss_type']}",
        "training.selection_metric={variant['selection_metric']}",
        "training.lambdarank_ic_max_pairs_per_day",
        "training.lambdarank_ic_temperature",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_lambdarank_ic_notebook_preserves_pit_recipe_and_fails_fast() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        'TrainingConfig(loss_type="lambdarank_ic", selection_metric="val_rank_ic")',
        "build_training_loss(probe_cfg)",
        "features.include_global_regime=true",
        "features.regime_include_subsequent_returns=false",
        "graph.update_frequency_months=0",
        "graph.drop_edge_p=0.1",
        "training.label_type=returns",
        "training.shuffle_train=true",
        "model.label_t=5",
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks=450",
        "data.pit_breadth_policy=error",
        "lambdarank_ic_pit_manifest.json",
        "lambdarank_ic_pit_training_results.json",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_lambdarank_ic_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
