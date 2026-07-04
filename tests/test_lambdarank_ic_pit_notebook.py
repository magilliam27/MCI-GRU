import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/lambdarank_ic_pit_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_lambdarank_ic_pit_nb.py")

# Setup-cell text now emitted by nb_lib.colab_setup_cell rather than being
# duplicated in the generator source; assert these against the generated
# notebook only (the notebook is the real contract).
SHARED_SETUP_TOKENS = {
    'BRANCH = "codex/colab-gpu-utilization-hardening-20260620"',
    "G4/L4-class Colab runtime",
    "not T4/CPU",
    'BLOCKED_GPU_NAMES = ("T4",)',
    "ALLOWED_GPU_MARKERS = (",
    "STRICT_GPU_MARKERS: list[str] = []",
}


def _assert_tokens(required_tokens: list[str]) -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")
    for token in required_tokens:
        assert token in combined
        if token not in SHARED_SETUP_TOKENS:
            assert token in generator


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
    required_tokens = [
        "LambdaRankIC Pairwise Rank IC PIT Grid",
        "LOSS_PATH_DECISION_2026-06-04.md",
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        'BRANCH = "codex/colab-gpu-utilization-hardening-20260620"',
        "SMOKE_MODE = False",
        "SCREEN_MODE = True",
        "SMOKE_YEARS = [2022]",
        "SCREEN_YEARS = [2022]",
        "FULL_YEARS = [2022, 2023, 2024, 2025]",
        "SMOKE_BASE_SEEDS = [314159]",
        "SCREEN_BASE_SEEDS = [314159]",
        "FULL_BASE_SEEDS = [314159, 271828, 161803]",
        "SCREEN_NUM_MODELS = 1",
        "SCREEN_NUM_EPOCHS = 40",
        "SCREEN_EARLY_STOPPING_PATIENCE = 8",
        "EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS",
        "expected_jobs_by_mode = len(SCREEN_PAIR_CAPS) if SCREEN_MODE else (3 if SMOKE_MODE else 36)",
        "expected_models_by_mode = len(SCREEN_PAIR_CAPS) if SCREEN_MODE else (3 if SMOKE_MODE else 720)",
    ]

    _assert_tokens(required_tokens)


def test_lambdarank_ic_notebook_emits_three_objective_variants() -> None:
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

    _assert_tokens(required_tokens)


def test_lambdarank_ic_notebook_has_lower_pair_screen_contract() -> None:
    required_tokens = [
        'BRANCH = "codex/colab-gpu-utilization-hardening-20260620"',
        "SCREEN_MODE = True",
        "SCREEN_YEARS = [2022]",
        "SCREEN_BASE_SEEDS = [314159]",
        "SCREEN_PAIR_CAPS = [512, 1024, 2048, 4096]",
        "SCREEN_NUM_MODELS = 1",
        "SCREEN_NUM_EPOCHS = 40",
        "SCREEN_EARLY_STOPPING_PATIENCE = 8",
        "for max_pairs_per_day in variant_pair_caps:",
        "'lambdarank_ic_pair_cap_screen'",
        '"screen_mode": SCREEN_MODE',
        '"pair_caps": PAIR_CAPS',
        '"budget_mode": BUDGET_MODE',
        "time.perf_counter()",
        '"elapsed_seconds": round(elapsed_seconds, 3)',
        '"elapsed_seconds",',
    ]

    _assert_tokens(required_tokens)


def test_lambdarank_ic_notebook_has_colab_reliability_contract() -> None:
    required_tokens = [
        "G4/L4-class Colab runtime",
        "not T4/CPU",
        'BLOCKED_GPU_NAMES = ("T4",)',
        "ALLOWED_GPU_MARKERS = (",
        "STRICT_GPU_MARKERS: list[str] = []",
        "Refusing to reuse existing run root",
        'HEARTBEAT_PATH = RUN_ROOT / "heartbeat.json"',
        'GPU_UTIL_PATH = RUN_ROOT / "gpu_util.csv"',
        "def write_heartbeat(",
        'status: str = "RUNNING"',
        "training_results.csv",
        "training_results.json",
        "scripts/monitor_gpu_util.py",
        "google.colab.runtime.unassign()",
        "Runtime > Disconnect and delete runtime",
    ]

    _assert_tokens(required_tokens)


def test_lambdarank_ic_notebook_preserves_pit_recipe_and_fails_fast() -> None:
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

    _assert_tokens(required_tokens)


def test_lambdarank_ic_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
