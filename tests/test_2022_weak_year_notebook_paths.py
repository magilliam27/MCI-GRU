import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/2022_weak_year_investigation.ipynb")
GENERATOR_PATH = Path("scripts/gen_2022_weak_year_investigation_nb.py")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def test_2022_notebook_uses_discovered_drive_run_folder() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    expected = "performance_proof_missing_grid"
    stale = "performance_proof_tests"

    assert expected in combined
    assert expected in generator
    assert stale not in combined
    assert stale not in generator


def test_2022_notebook_setup_fails_before_loader_when_artifacts_missing() -> None:
    setup_source = next(source for source in _cell_sources() if "REQUIRED_ARTIFACT_FILES" in source)

    assert "Missing weak-year artifacts" in setup_source
    assert "raise FileNotFoundError" in setup_source
