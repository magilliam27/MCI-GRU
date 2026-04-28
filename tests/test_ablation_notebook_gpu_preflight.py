import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/ablation_evaluation_loop_colab.ipynb")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
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
