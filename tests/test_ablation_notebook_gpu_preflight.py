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
