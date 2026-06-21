import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/lambdarank_ic_full_tranche_colab.ipynb")


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


def test_full_tranche_notebook_pins_narrow_full_recipe_scope() -> None:
    combined = "\n".join(_cell_sources())

    required_tokens = [
        "LambdaRankIC Full-Recipe Confirmation Tranche",
        'BRANCH = "codex/colab-gpu-utilization-hardening-20260620"',
        "SMOKE_MODE = False",
        "SCREEN_MODE = False",
        "MAX_JOBS = 3",
        "FULL_PAIR_CAPS = [512]",
        "FULL_NUM_MODELS = 20",
        "FULL_NUM_EPOCHS = 100",
        "FULL_EARLY_STOPPING_PATIENCE = 15",
        '"actual_job_count": ACTUAL_JOB_COUNT',
        '"actual_total_models": ACTUAL_TOTAL_MODELS',
        "lambdarank_ic_full_tranche",
        "lambdarank_ic_full_tranche_manifest.json",
    ]

    for token in required_tokens:
        assert token in combined


def test_full_tranche_notebook_keeps_three_candidate_variants() -> None:
    combined = "\n".join(_cell_sources())

    required_tokens = [
        "'pure_ic_baseline'",
        "'portfolio_ic_hybrid'",
        "'lambdarank_ic_candidate'",
        "'loss_type': 'ic'",
        "'loss_type': 'portfolio_ic'",
        "'loss_type': 'lambdarank_ic'",
        "'selection_metric': 'val_rank_ic'",
        "training.lambdarank_ic_max_pairs_per_day",
    ]

    for token in required_tokens:
        assert token in combined


def test_full_tranche_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
