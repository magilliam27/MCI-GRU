import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/ablation_evaluation_loop_colab.ipynb")
FULL_FACTORIAL_NOTEBOOK_PATH = Path("notebooks/full_feature_factorial_ablation_colab.ipynb")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _full_factorial_cell_sources() -> list[str]:
    notebook = json.loads(FULL_FACTORIAL_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def test_ablation_colab_notebook_requires_gpu_before_loop() -> None:
    sources = _cell_sources()
    combined = "\n".join(sources)

    assert "REQUIRE_GPU = True" in combined
    assert "torch.cuda.is_available()" in combined
    assert "subprocess.run(" in combined
    assert "Child CUDA available:" in combined
    assert "raise RuntimeError(msg)" in combined


def test_ablation_colab_notebook_runs_child_python_unbuffered() -> None:
    sources = _cell_sources()
    helper_source = next(source for source in sources if "def run_ablation" in source)

    assert "sys.executable," in helper_source
    assert "'-u'," in helper_source


def test_ablation_colab_notebook_requests_gpu_runtime_metadata() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["metadata"]["accelerator"] == "GPU"


def test_ablation_colab_notebook_has_finalist_confirmation_section() -> None:
    sources = _cell_sources()
    combined = "\n".join(sources)

    assert "## 10. Finalist Confirmation" in combined
    assert "RUN_FINALIST_CONFIRMATION = False" in combined
    assert "FINALIST_CONFIRMATION_ROOT" in combined
    assert "finalist_confirmation" in combined
    assert "'modern_defaults'" in combined
    assert "'topk10_abs_graph'" in combined
    assert "'topk20_abs_graph'" in combined
    assert "'topk30_abs_graph'" in combined
    assert "'topk20_pos_graph'" in combined
    assert "CONFIRMATION_NUM_MODELS = 3" in combined
    assert "CONFIRMATION_NUM_EPOCHS = 20" in combined


def test_ablation_colab_notebook_has_final_confirmation_signal_check() -> None:
    sources = _cell_sources()
    combined = "\n".join(sources)

    assert "## 11. Final Confirmation: Signal Survival Check" in combined
    assert "RUN_FINAL_CONFIRMATION = False" in combined
    assert "FINAL_CONFIRMATION_ROOT" in combined
    assert "final_confirmation" in combined
    assert "FINAL_CONFIRMATION_PRIMARY_RUNS" in combined
    assert "'modern_defaults'" in combined
    assert "'topk30_abs_graph'" in combined
    assert "INCLUDE_RETURN_CHALLENGER = True" in combined
    assert "'topk10_abs_graph'" in combined
    assert "training.walkforward.enabled=true" in combined
    assert "FINAL_CONFIRMATION_MAX_WINDOWS = 5" in combined
    assert "FINAL_CONFIRMATION_NUM_MODELS = 20" in combined
    assert "FINAL_CONFIRMATION_NUM_EPOCHS = 100" in combined
    assert "FINAL_CONFIRMATION_EARLY_STOPPING_PATIENCE = 15" in combined
    assert "training.early_stopping_patience={FINAL_CONFIRMATION_EARLY_STOPPING_PATIENCE}" in combined
    assert "FINAL_CONFIRMATION_BOOTSTRAP_RESAMPLES = 1000" in combined
    assert "ic_ci_pass" in combined
    assert "top20_ci_pass" in combined
    assert "continuation_recommendation" in combined


def test_ablation_colab_notebook_has_2026_holdout_confirmation() -> None:
    sources = _cell_sources()
    combined = "\n".join(sources)

    assert "## 12. 2026 Holdout Confirmation" in combined
    assert "RUN_2026_HOLDOUT = False" in combined
    assert "HOLDOUT_2026_ROOT" in combined
    assert "holdout_2026" in combined
    assert "PREFERRED_2026_DATA_FILE" in combined
    assert "sp500_2019_universe_data_through_2026.csv" in combined
    assert "HOLDOUT_TRAIN_END = '2024-12-31'" in combined
    assert "HOLDOUT_VAL_START = '2025-01-08'" in combined
    assert "HOLDOUT_VAL_END = '2025-12-31'" in combined
    assert "HOLDOUT_TEST_START = '2026-01-08'" in combined
    assert "HOLDOUT_TEST_END = '2026-04-30'" in combined
    assert "HOLDOUT_2026_RUNS" in combined
    assert "HOLDOUT_2026_NUM_MODELS = 20" in combined
    assert "HOLDOUT_2026_NUM_EPOCHS = 100" in combined
    assert "HOLDOUT_2026_EARLY_STOPPING_PATIENCE = 15" in combined
    assert "training.early_stopping_patience={HOLDOUT_2026_EARLY_STOPPING_PATIENCE}" in combined
    assert "'modern_defaults'" in combined
    assert "'topk30_abs_graph'" in combined
    assert "HOLDOUT_INCLUDE_RETURN_CHALLENGER = False" in combined
    assert "'topk10_abs_graph'" in combined
    assert "training.walkforward.enabled=false" in combined
    assert "holdout_2026_recommendation" in combined
    assert "holdout_ic_ci_pass" in combined
    assert "holdout_top20_ci_pass" in combined


def test_full_factorial_notebook_recovers_only_failed_regime_runs() -> None:
    sources = _full_factorial_cell_sources()
    combined = "\n".join(sources)

    assert "RECOVERY_RUN_TAG = '20260429_024346'" in combined
    assert "RUN_ONLY_FAILED_REGIME_RUNS = True" in combined
    assert "SKIP_COMPLETED_RUNS = True" in combined
    assert "run_queue = ABLATIONS" in combined
    assert "if RUN_ONLY_FAILED_REGIME_RUNS:" in combined
    assert "'regime_current_only', 'regime_with_forward_context'" in combined
    assert "ablation.get('regime_factor') in failed_regime_factors" in combined
    assert "for ablation in run_queue:" in combined
