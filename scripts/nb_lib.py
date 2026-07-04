"""Shared library for the notebook generators (``scripts/gen_*_nb.py``).

WS-L of the rearchitecture plan. The generators are invoked as
``python scripts/gen_*_nb.py`` from the repo root, so ``scripts/`` is
``sys.path[0]`` and a plain ``import nb_lib`` resolves without packaging.

Two historical cell-source conventions exist among the generators and both
are preserved so regenerated notebooks stay byte-identical:

- ``md``/``code``: ``dedent(text).strip().splitlines(keepends=True)`` —
  the last source line carries no trailing newline.
- ``md_lines``/``code_lines``: ``[line + "\\n" for line in ...splitlines()]`` —
  every source line, including the last, ends with a newline.
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Canonical Colab GPU notebook metadata (most generators).
COLAB_GPU_METADATA: dict = {
    "accelerator": "GPU",
    "colab": {"provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

#: Colab GPU metadata whose kernelspec omits the "language" key
#: (gen_long_history_pit_eval_nb, gen_sp500_pit_gics_top10_baseline_nb).
COLAB_GPU_METADATA_BARE_KERNEL: dict = {
    "accelerator": "GPU",
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

#: Plain local-kernel metadata with a pinned language_info version
#: (gen_performance_proof_nb, gen_temporal_rolling_backtest_nb,
#: gen_train_test_nb, gen_promising_backtest_nb).
LOCAL_PY310_METADATA: dict = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10.0"},
}


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def md_lines(text: str) -> dict:
    text = textwrap.dedent(text).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code_lines(text: str) -> dict:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


def build_notebook(cells: list[dict], *, metadata: dict | None = None) -> dict:
    """nbformat-4 shell. Single definition replacing per-file copies."""
    return {
        "cells": cells,
        "metadata": COLAB_GPU_METADATA if metadata is None else metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(
    cells: list[dict],
    out_path: Path,
    *,
    metadata: dict | None = None,
    indent: int = 1,
    trailing_newline: bool = False,
) -> None:
    """Serialize the notebook with the exact per-generator dump conventions."""
    payload = json.dumps(build_notebook(cells, metadata=metadata), indent=indent)
    if trailing_newline:
        payload += "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    print(f"Wrote {out_path}")
