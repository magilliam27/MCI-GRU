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
#: (gen_performance_proof_nb, gen_temporal_rolling_backtest_nb).
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


# Canonical Colab setup-cell source (reference: the historical inline cell in
# the retired scripts/gen_lambdarank_ic_pit_nb.py, at tag archive/pre-cleanup-2026-09). Deliberately a raw string: the "\n"
# inside the RuntimeError message must land in the notebook as a literal
# backslash-n, exactly as the generators emitted it. The §…§ placeholders are
# substituted by colab_setup_cell().
_COLAB_SETUP_TEMPLATE = r"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
BRANCH = "§BRANCH§"
REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
REQUIRE_G4_L4_GPU = §REQUIRE_GPU§
BLOCKED_GPU_NAMES = §BLOCKED§
ALLOWED_GPU_MARKERS = (
§ALLOWED§)
§STRICT§

def detect_gpu_name() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed. Expected G4/L4-class Colab runtime, not T4/CPU.\n"
            + proc.stderr
        )
    gpu_name = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    if not gpu_name:
        raise RuntimeError("nvidia-smi did not report a GPU name.")
    upper_gpu = gpu_name.upper()
    if any(blocked in upper_gpu for blocked in BLOCKED_GPU_NAMES):
        raise RuntimeError(
            f"Expected G4/L4-class Colab runtime, not T4/CPU. Visible GPU: {gpu_name}"
        )
    if not any(marker in upper_gpu for marker in ALLOWED_GPU_MARKERS):
        raise RuntimeError(
            f"Refusing runtime GPU {gpu_name}; allowed markers are {ALLOWED_GPU_MARKERS}."
        )
    if STRICT_GPU_MARKERS and not any(marker in upper_gpu for marker in STRICT_GPU_MARKERS):
        raise RuntimeError(
            f"GPU {gpu_name} does not match STRICT_GPU_MARKERS={STRICT_GPU_MARKERS}."
        )
    return gpu_name

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
    else:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[§PIP_EXTRAS§]"], check=True)

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print("Repo:", REPO_DIR)
print("Branch:", BRANCH)
subprocess.run(["git", "rev-parse", "HEAD"], check=False)
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    GPU_NAME = detect_gpu_name()
    print("GPU:", GPU_NAME)
elif REQUIRE_G4_L4_GPU:
    raise RuntimeError(
        "Expected G4/L4-class Colab runtime, not T4/CPU. "
        "Switch Runtime -> Change runtime type -> G4 GPU before training."
    )
"""


#: Directory (relative to the repo root) currently holding the backtest
#: engines. WS-C flips this single value to "scripts" when the engines move.
BACKTEST_ENGINE_DIR = "scripts"


def backtest_engine_path_expr(
    engine: str,
    *,
    repo_var: str = "REPO_DIR",
    quote: str = "'",
    split_path: bool = True,
) -> str:
    """Source-text expression for the backtest engine script path, spliced
    into generated notebook cells (e.g. ``str(REPO_DIR / 'tests' /
    'backtest_sp500.py')``). This is the single place WS-C edits when the
    engines move from ``tests/`` to ``scripts/``.

    ``engine`` is ``"backtest_sp500"`` or ``"backtest_sp500_daily"``.
    ``quote`` and ``split_path`` preserve the historical per-generator
    formatting so regenerated notebooks stay byte-identical.
    """
    if split_path:
        return f"str({repo_var} / {quote}{BACKTEST_ENGINE_DIR}{quote} / {quote}{engine}.py{quote})"
    return f"str({repo_var} / {quote}{BACKTEST_ENGINE_DIR}/{engine}.py{quote})"


def _tuple_literal(values: tuple[str, ...]) -> str:
    if not values:
        return "()"
    if len(values) == 1:
        return f'("{values[0]}",)'
    return "(" + ", ".join(f'"{value}"' for value in values) + ")"


def colab_setup_cell(
    *,
    branch: str,
    pip_extras: str = "dev,tracking,fred",
    require_gpu: bool = True,
    blocked_gpu_names: tuple[str, ...] = ("T4",),
    allowed_gpu_markers: tuple[str, ...] = (
        "G4",
        "L4",
        "A100",
        "H100",
        "V100",
        "RTX PRO",
        "BLACKWELL",
    ),
    strict_gpu_markers: tuple[str, ...] = (),
    extra_setup_source: str = "",
) -> dict:
    """Returns the standard Colab setup *code cell* (repo clone, pip install,
    GPU preflight). ``extra_setup_source`` is appended verbatim so generators
    keep their experiment-specific probe blocks."""
    allowed_block = "".join(f'    "{marker}",\n' for marker in allowed_gpu_markers)
    strict_line = (
        "STRICT_GPU_MARKERS: list[str] = []"
        if not strict_gpu_markers
        else (
            "STRICT_GPU_MARKERS: list[str] = ["
            + ", ".join(f'"{marker}"' for marker in strict_gpu_markers)
            + "]"
        )
    )
    source = (
        _COLAB_SETUP_TEMPLATE.replace("§BRANCH§", branch)
        .replace("§REQUIRE_GPU§", str(require_gpu))
        .replace("§BLOCKED§", _tuple_literal(blocked_gpu_names))
        .replace("§ALLOWED§", allowed_block)
        .replace("§STRICT§", strict_line)
        .replace("§PIP_EXTRAS§", pip_extras)
    )
    if extra_setup_source:
        source = source.rstrip("\n") + "\n\n" + textwrap.dedent(extra_setup_source).strip("\n")
    return code(source)


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


#: The EODHD S&P 500 index file the regime market input reads (#276). Its path is
#: ``features.regime_market_csv`` in ``configs/features/with_momentum.yaml``;
#: ``data/raw`` is gitignored, so a Colab checkout has to stage it from Drive.
#: The SHA-256 and size are the 2026-09-19 pull's, as recorded in that pull's
#: ``inventory.json``. ``scripts/first_run_colab.py`` pins the same values.
EODHD_MARKET_RELATIVE = "data/raw/market/eodhd_sp500_2010_20260919/sp500_index.csv"
EODHD_MARKET_SHA256 = "9f5cfaf5e9f057065b83619b01568164e1f61a45df9453c66269fb54f080f5ac"
EODHD_MARKET_SIZE = 351815
#: The Drive copy sits in its own folder beside the preserved LSEG package, so
#: that package's verified ten-file inventory stays as it was restored (#207).
EODHD_MARKET_DRIVE_PATH = (
    "/content/drive/MyDrive/MCI_GRU_shared/preservation/eodhd_sp500_2010_20260919/sp500_index.csv"
)

_EODHD_STAGING_TEMPLATE = """
# EODHD S&P 500 index file for the regime market input (#276, #278).
# configs/features/with_momentum.yaml reads it from EODHD_MARKET_RELATIVE, and
# data/raw is gitignored, so it is copied here from Drive. A wrong hash stops
# the notebook, and a file already in place with other bytes is never replaced.
import hashlib as _eodhd_hashlib
import shutil as _eodhd_shutil
from pathlib import Path as _EodhdPath

import yaml as _eodhd_yaml

EODHD_MARKET_RELATIVE = "§RELATIVE§"
EODHD_MARKET_SHA256 = "§SHA256§"
EODHD_MARKET_SIZE = §SIZE§
EODHD_MARKET_DRIVE_PATH = _EodhdPath("§DRIVE§")


def _eodhd_sha256(path):
    digest = _eodhd_hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _eodhd_check(path):
    size = path.stat().st_size
    sha256 = _eodhd_sha256(path)
    if size != EODHD_MARKET_SIZE or sha256 != EODHD_MARKET_SHA256:
        raise RuntimeError(
            f"EODHD S&P 500 file {path} does not match the pinned pull: "
            f"expected {EODHD_MARKET_SIZE} bytes, sha256 {EODHD_MARKET_SHA256}; "
            f"found {size} bytes, sha256 {sha256}. Nothing was copied or replaced."
        )


def stage_eodhd_market_file(repo_dir, drive_path=EODHD_MARKET_DRIVE_PATH):
    target = _EodhdPath(repo_dir) / EODHD_MARKET_RELATIVE
    if target.exists():
        _eodhd_check(target)
        print("EODHD S&P 500 file already staged and verified:", target)
        return target
    source = _EodhdPath(drive_path)
    if not source.exists():
        raise FileNotFoundError(
            "The regime market input needs the EODHD S&P 500 file, and it is not on Drive at "
            f"{source}. Copy data/raw/market/eodhd_sp500_2010_20260919/ from the Windows "
            "checkout to that Drive folder, or pass its Drive path to stage_eodhd_market_file()."
        )
    _eodhd_check(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    _eodhd_shutil.copyfile(source, target)
    _eodhd_check(target)
    print("Staged and verified EODHD S&P 500 file:", target)
    return target


def eodhd_market_file_is_read(repo_dir):
    # True when the checkout's with_momentum features config reads the pinned file.
    # A checkout from before #276 has no regime_market_csv key (or null), so it
    # never reads it; a different path means the pin above is wrong.
    config_path = _EodhdPath(repo_dir) / "configs" / "features" / "with_momentum.yaml"
    with open(config_path, encoding="utf-8") as handle:
        configured = (_eodhd_yaml.safe_load(handle) or {}).get("regime_market_csv")
    if configured is None:
        return False
    if configured != EODHD_MARKET_RELATIVE:
        raise RuntimeError(
            f"{config_path} names regime_market_csv {configured!r}, but this notebook pins "
            f"{EODHD_MARKET_RELATIVE!r}. Update the pinned EODHD file before running."
        )
    return True
"""


def eodhd_market_staging_source(*, call: str = "") -> str:
    """Source text that defines ``stage_eodhd_market_file(repo_dir)`` in a notebook.

    The generated function copies the pinned EODHD S&P 500 file from Drive to
    ``EODHD_MARKET_RELATIVE`` under the checkout and checks its size and SHA-256
    before and after the copy. It also defines ``eodhd_market_file_is_read(repo_dir)``,
    which says whether the checkout's ``with_momentum`` config reads that file.
    ``call`` is appended verbatim, so a generator can gate the call on its own
    regime toggle and the checkout, e.g. ``"if USE_GLOBAL_REGIME and
    eodhd_market_file_is_read(REPO_DIR):\\n    stage_eodhd_market_file(REPO_DIR)"``.
    """
    source = (
        _EODHD_STAGING_TEMPLATE.replace("§RELATIVE§", EODHD_MARKET_RELATIVE)
        .replace("§SHA256§", EODHD_MARKET_SHA256)
        .replace("§SIZE§", str(EODHD_MARKET_SIZE))
        .replace("§DRIVE§", EODHD_MARKET_DRIVE_PATH)
        .strip("\n")
    )
    if call:
        source += "\n\n" + textwrap.dedent(call).strip("\n")
    return source
