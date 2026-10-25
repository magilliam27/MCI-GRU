"""Every notebook that replays on a PIT file names the export's cutoff (#342).

The daily replay refuses open-ended PIT rows (blank ``valid_to``) without
``--pit_export_cutoff`` (#339), as training refuses them without
``data.pit_export_cutoff``. A notebook that passes the PIT file to either without
its cutoff stops on any export with open-ended rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _code(notebook: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


PIT_NOTEBOOKS = [nb for nb in NOTEBOOKS if "--pit_universe_csv" in _code(nb)]


def test_the_scan_finds_the_pit_replay_notebooks() -> None:
    assert {nb.name for nb in PIT_NOTEBOOKS} >= {
        "long_history_pit_eval_colab.ipynb",
        "pit_masked_panel_2022_2025_colab.ipynb",
        "pit_repeated_seed_replication_colab.ipynb",
        "sp500_pit_gics_top10_baseline_colab.ipynb",
    }


@pytest.mark.parametrize("notebook", PIT_NOTEBOOKS, ids=lambda nb: nb.stem)
def test_pit_replay_and_training_carry_the_export_cutoff(notebook: Path) -> None:
    code = _code(notebook)
    assert "PIT_EXPORT_CUTOFF = " in code
    assert "'--pit_export_cutoff'" in code or '"--pit_export_cutoff"' in code
    assert "data.pit_export_cutoff={PIT_EXPORT_CUTOFF}" in code
