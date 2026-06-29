import ast
import json
from pathlib import Path

NOTEBOOKS = [
    (
        Path("notebooks/volatility_targeting_pit_colab.ipynb"),
        Path("scripts/gen_volatility_targeting_pit_nb.py"),
    ),
    (
        Path("notebooks/volatility_targeting_full_pit_colab.ipynb"),
        Path("scripts/gen_volatility_targeting_pit_nb.py"),
    ),
]


def _cell_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _code_cell_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_volatility_pit_notebooks_have_real_gpu_gate() -> None:
    required_tokens = [
        "G4/L4-class Colab runtime, not T4/CPU",
        "nvidia-smi",
        "ALLOWED_GPU_MARKERS = (",
        'BLOCKED_GPU_NAMES = ("T4",)',
        "STRICT_GPU_MARKERS: list[str] = []",
        "GPU_UTIL_PATH",
        "scripts/monitor_gpu_util.py",
    ]

    for notebook_path, generator_path in NOTEBOOKS:
        combined = "\n".join(_cell_sources(notebook_path))
        generator = generator_path.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in combined
            assert token in generator


def test_volatility_pit_code_cells_parse() -> None:
    for notebook_path, _generator_path in NOTEBOOKS:
        code_cells = _code_cell_sources(notebook_path)
        assert code_cells
        for source in code_cells:
            ast.parse(source)
