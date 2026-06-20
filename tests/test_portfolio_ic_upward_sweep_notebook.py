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


def test_upward_sweep_has_generator_runner_and_branch_pin() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Portfolio-IC Upward Weight Sweep",
        "codex/colab-gpu-utilization-hardening-20260620",
        "run_portfolio_ic_upward_sweep.py",
        "RESUME_RUN_ROOT",
        "AUTO_UNASSIGN_ON_FINISH",
        "heartbeat.json",
        "training_results.csv",
        "training_results.json",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator
        assert token in runner


def test_upward_sweep_rejects_t4_and_records_gpu_evidence() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "nvidia-smi",
        "Refusing runtime GPU",
        "G4/L4-class Colab runtime",
        "not T4/CPU",
        "ALLOWED_GPU_MARKERS",
        "'L4'",
        "'RTX PRO'",
        "'BLACKWELL'",
        "gpu_name",
        "gpu_util.csv",
        "monitor_gpu_util.py",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator
        assert token in runner


def test_upward_sweep_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
