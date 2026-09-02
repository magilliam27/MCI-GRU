"""Contract test for the paired re-analysis notebook (ticket 179).

Proves structure, constants, and byte-identical regeneration. It is not run
evidence: the notebook's results exist only once it has executed on Colab with
Drive-backed artifacts.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "graph_paired_reanalysis_colab.ipynb"
GENERATOR_PATH = REPO_ROOT / "scripts" / "gen_graph_paired_reanalysis_nb.py"


@pytest.fixture(scope="module")
def generator():
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("gen_graph_paired_reanalysis_nb", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_sources() -> list[str]:
    return ["".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def _all_source() -> str:
    return "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])


def test_notebook_exists_and_every_code_cell_parses() -> None:
    assert NOTEBOOK_PATH.is_file()
    sources = _code_sources()
    assert len(sources) >= 9
    for source in sources:
        ast.parse(source)


def test_notebook_is_a_cpu_notebook() -> None:
    notebook = _notebook()
    assert "accelerator" not in notebook["metadata"]
    setup = _code_sources()[0]
    assert "REQUIRE_G4_L4_GPU = False" in setup
    assert "drive.mount" in setup


def test_notebook_regenerates_byte_identically(generator) -> None:
    assert generator.render() == NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_notebook_pins_the_ticket_branch_from_the_generator_constant(generator) -> None:
    setup = _code_sources()[0]
    assert f'BRANCH = "{generator.NOTEBOOK_BRANCH}"' in setup


def test_notebook_targets_the_ablation_run_and_its_arms(generator) -> None:
    source = _all_source()
    assert generator.RUN_TAG == "20260901_015032"
    assert f"RUN_TAG = {generator.RUN_TAG!r}" in source
    assert generator.CONTROL_ARM == "A0_zeroed"
    assert generator.COMPARISON_ARMS == ["A1_shipped", "A2_thr05", "A3_topk20", "A4_sector_only"]
    for arm in [generator.CONTROL_ARM, *generator.COMPARISON_ARMS]:
        assert arm in source
    assert generator.STAGES == ["confirm", "screen"]
    assert "20260901_014022" not in "".join(_code_sources()), "smoke run tag must not be an input"


def test_notebook_uses_the_arbiter_label_and_overlap_aware_settings(generator) -> None:
    source = "".join(_code_sources())
    assert "realized_returns_from_market_data(market, label_t=LABEL_T)" in source
    assert "close_pivot.shift(-LABEL_T) / close_pivot - 1" in source  # section-8 variant, alongside
    assert (generator.LABEL_T, generator.BLOCK_SIZE, generator.HAC_LAGS) == (5, 5, 4)
    assert generator.N_RESAMPLES == 1000 and generator.BOOTSTRAP_SEED == 1729
    assert generator.CI_LEVEL == 0.95
    for name, value in [
        ("LABEL_T", 5),
        ("BLOCK_SIZE", 5),
        ("HAC_LAGS", 4),
        ("N_RESAMPLES", 1000),
        ("BOOTSTRAP_SEED", 1729),
    ]:
        assert f"{name} = {value}" in source


def test_notebook_calls_the_paired_inference_module(generator) -> None:
    source = "".join(_code_sources())
    for name in [
        "align_daily_series",
        "paired_daily_differences",
        "paired_mean_inference",
        "bhy_adjusted_p_values",
        "minimum_detectable_effect",
        "required_days",
        "sharpe_block_bootstrap_ci",
        "tail_share",
        "winsorize_rows",
    ]:
        assert f"{name}(" in source, name
    import mci_gru.evaluation.paired_inference as module

    for name in [
        "align_daily_series",
        "paired_daily_differences",
        "paired_mean_inference",
        "bhy_adjusted_p_values",
        "minimum_detectable_effect",
        "required_days",
        "sharpe_block_bootstrap_ci",
        "tail_share",
        "winsorize_rows",
    ]:
        assert callable(getattr(module, name))


def test_notebook_writes_the_reanalysis_artifacts() -> None:
    source = "".join(_code_sources())
    assert 'OUT_DIR = RUN_ROOT / "reanalysis"' in source
    for artifact in [
        "arbiter_reconciliation.csv",
        "paired_inference.csv",
        "power.csv",
        "delta_distribution.csv",
        "largest_abs_delta_days.csv",
        "sharpe_intervals.csv",
        "paired_portfolio_returns.csv",
        "seed_paired_per_model_ic.csv",
        "ensemble_scale_audit.csv",
        "reanalysis_summary.md",
    ]:
        assert artifact in source, artifact


def test_notebook_verifies_the_panel_digest_before_use(generator) -> None:
    source = "".join(_code_sources())
    assert generator.PANEL_SHA256 in source
    assert "if panel_sha != PANEL_SHA256:" in source
