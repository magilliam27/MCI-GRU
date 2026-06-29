import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/portfolio_ic_pit_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_portfolio_ic_pit_nb.py")


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


def test_portfolio_ic_notebook_pins_smoke_and_full_grid_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Portfolio-IC Hybrid Loss PIT Grid",
        "Machine Learning Meets Markowitz",
        "docs/DEFAULT_EXPERIMENT_RECIPE.md",
        'BRANCH = "codex/colab-gpu-utilization-hardening-20260620"',
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1",
        "SMOKE_MODE = True",
        "SMOKE_YEARS = [2025]",
        "FULL_YEARS = [2022, 2023, 2024, 2025]",
        "SMOKE_BASE_SEEDS = [314159]",
        "FULL_BASE_SEEDS = [314159, 271828, 161803]",
        "NUM_MODELS = 1 if SMOKE_MODE else 20",
        "NUM_EPOCHS = 1 if SMOKE_MODE else 100",
        "EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 15",
        "'pure_ic_baseline'",
        "'portfolio_ic_hybrid'",
        "'loss_type': 'ic'",
        "'selection_metric': 'val_ic'",
        "'loss_type': 'portfolio_ic'",
        "'selection_metric': 'val_loss'",
        "'portfolio_ic_top_k': 10",
        "'portfolio_ic_weight': 0.25",
        "'portfolio_ic_temperature': 0.25",
        "EXPECTED_JOB_COUNT = len(YEARS) * len(BASE_SEEDS) * len(OBJECTIVE_VARIANTS)",
        "EXPECTED_TOTAL_MODELS = EXPECTED_JOB_COUNT * NUM_MODELS",
        "assert EXPECTED_JOB_COUNT == (2 if SMOKE_MODE else 24)",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_portfolio_ic_notebook_has_non_t4_gpu_gate_and_sampler() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "G4/L4-class Colab runtime",
        "not T4/CPU",
        'BLOCKED_GPU_NAMES = ("T4",)',
        "ALLOWED_GPU_MARKERS = (",
        "STRICT_GPU_MARKERS: list[str] = []",
        'GPU_UTIL_PATH = RUN_ROOT / "gpu_util.csv"',
        "scripts/monitor_gpu_util.py",
        "google.colab.runtime.unassign()",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_portfolio_ic_notebook_emits_expected_loss_overrides() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "training.loss_type={variant['loss_type']}",
        "training.selection_metric={variant['selection_metric']}",
        "training.portfolio_ic_top_k={variant['portfolio_ic_top_k']}",
        "training.portfolio_ic_weight={variant['portfolio_ic_weight']}",
        "training.portfolio_ic_temperature={variant['portfolio_ic_temperature']}",
        "features.include_global_regime=true",
        "features.regime_include_subsequent_returns=false",
        "graph.drop_edge_p=0.1",
        "training.shuffle_train=true",
        "model.label_t=5",
        "data.pit_universe_mode=masked_panel",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_portfolio_ic_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
