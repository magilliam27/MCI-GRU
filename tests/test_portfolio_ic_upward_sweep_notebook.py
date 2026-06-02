import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/portfolio_ic_upward_sweep_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_portfolio_ic_upward_sweep_nb.py")
RUNNER_PATH = Path("scripts/run_portfolio_ic_upward_sweep.py")


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


def test_upward_sweep_notebook_uses_visible_foreground_runner() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Portfolio-IC Upward Weight Sweep",
        "no detached process and no hidden kernel launch",
        "AUTO_UNASSIGN_ON_FINISH = True",
        "RESUME_RUN_ROOT = \"/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_upward_sweep/20260601_013922_static_regime_full\"",
        "run_portfolio_ic_upward_sweep.py",
        "--resume",
        "subprocess.run(cmd, cwd=REPO_DIR, text=True, check=True)",
        "runtime.unassign()",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator

    forbidden_tokens = [
        "subprocess.Popen",
        "capture_output=True)\n        finally",
        "run_path",
        "kernel",
    ]
    foreground_cell = _code_cell_sources()[-1]
    for token in forbidden_tokens:
        assert token not in foreground_cell


def test_upward_sweep_runner_pins_full_grid_and_resume_contract() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    required_tokens = [
        "DEFAULT_YEARS = [2022, 2023, 2024, 2025]",
        "DEFAULT_BASE_SEEDS = [314159, 271828, 161803]",
        "DEFAULT_WEIGHTS = [0.75, 1.0]",
        "NUM_MODELS = 20",
        "NUM_EPOCHS = 100",
        "EARLY_STOPPING_PATIENCE = 15",
        "MCI_GRU_RESUME_ENSEMBLE",
        "hydra.run.dir={run_dir.as_posix()}",
        "best_existing_run_dir",
        "Refusing runtime GPU",
    ]

    for token in required_tokens:
        assert token in runner


def test_upward_sweep_notebook_code_cells_parse() -> None:
    for source in _code_cell_sources():
        ast.parse(source)
