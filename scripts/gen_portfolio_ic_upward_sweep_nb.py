"""Generate the foreground Colab notebook for the 0.75 / 1.0 Portfolio-IC sweep."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path("notebooks/portfolio_ic_upward_sweep_colab.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Portfolio-IC Upward Weight Sweep

        Foreground Colab launcher for `portfolio_ic_weight=0.75` and `1.0`.
        This notebook intentionally uses the visible Colab cell as the control
        plane: no detached process and no hidden kernel launch.

        The runner is model-resumable. It reuses complete `predictions_model_N`
        folders, recovers predictions from existing `model_N_best.pth`
        checkpoints when possible, and only trains missing ensemble members.
        It writes `heartbeat.json`, `training_results.csv`,
        `training_results.json`, and `gpu_util.csv` in the run root summaries.
        """
    ),
    md("## 1. Setup"),
    code(
        r"""
        import os
        import shutil
        import subprocess
        import sys
        from datetime import UTC, datetime
        from pathlib import Path

        try:
            from google.colab import drive, runtime, userdata

            IN_COLAB = True
        except ImportError:
            drive = None
            runtime = None
            userdata = None
            IN_COLAB = False

        REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
        BRANCH = "codex/colab-gpu-utilization-hardening-20260620"
        REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()

        if IN_COLAB:
            drive.mount("/content/drive")
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
                subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)], check=True)

        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        print("Repo:", REPO_DIR)
        print("Branch:", BRANCH)
        subprocess.run(["git", "rev-parse", "HEAD"], check=False)
        """
    ),
    md("## 2. Inputs And GPU Gate"),
    code(
        r"""
        if IN_COLAB and not os.environ.get("FRED_API_KEY"):
            try:
                secret = userdata.get("FRED_API_KEY") if userdata is not None else None
                if secret:
                    os.environ["FRED_API_KEY"] = secret
                    print("FRED_API_KEY loaded from Colab Secrets.")
            except Exception as exc:
                print("Could not read FRED_API_KEY from Colab Secrets:", exc)

        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is required for the current regime-enabled preset.")

        drive_data_dir = Path("/content/drive/MyDrive/MCI_GRU_shared/data") if IN_COLAB else REPO_DIR / "data/raw/market"
        drive_market_csv = drive_data_dir / "sp500_pit_union_lseg_20150101_20260513.csv"
        drive_pit_csv = drive_data_dir / "sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
        drive_regime_csv = (
            Path("/content/drive/MyDrive/MCI-GRU-Ablations/pit_repeated_seed_replication/20260520_183538/inputs/pit_repeated_seed_regime_inputs_20260520_183538.csv")
            if IN_COLAB
            else REPO_DIR / "data/raw/regime/pit_repeated_seed_regime_inputs_20260520_183538.csv"
        )

        repo_market_csv = REPO_DIR / "data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv"
        repo_pit_csv = REPO_DIR / "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
        repo_regime_csv = REPO_DIR / "data/raw/regime/pit_repeated_seed_regime_inputs_20260520_183538.csv"

        for source, dest in [
            (drive_market_csv, repo_market_csv),
            (drive_pit_csv, repo_pit_csv),
            (drive_regime_csv, repo_regime_csv),
        ]:
            if not source.exists():
                raise FileNotFoundError(f"Missing required input: {source}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print("Prepared:", dest)

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip().splitlines()[0]
        print("GPU:", gpu)
        upper_gpu = gpu.upper()
        ALLOWED_GPU_MARKERS = ('G4', 'L4', 'A100', 'H100', 'V100', 'RTX PRO', 'BLACKWELL')
        BLOCKED_GPU_NAMES = ("T4",)
        STRICT_GPU_MARKERS: list[str] = []
        if any(marker in upper_gpu for marker in BLOCKED_GPU_NAMES) or not any(
            marker in upper_gpu for marker in ALLOWED_GPU_MARKERS
        ):
            raise RuntimeError(
                f"Refusing runtime GPU {gpu}; expected G4/L4-class Colab runtime, not T4/CPU."
            )
        if STRICT_GPU_MARKERS and not any(marker in upper_gpu for marker in STRICT_GPU_MARKERS):
            raise RuntimeError(f"GPU {gpu} does not match STRICT_GPU_MARKERS={STRICT_GPU_MARKERS}.")
        gpu_name = gpu
        print("GPU gate passed.")
        """
    ),
    md("## 3. Foreground Resumable Sweep"),
    code(
        r"""
        AUTO_UNASSIGN_ON_FINISH = True
        RESUME_RUN_ROOT = "/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_upward_sweep/20260601_013922_static_regime_full"

        if RESUME_RUN_ROOT:
            RUN_ROOT = Path(RESUME_RUN_ROOT)
        else:
            run_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_static_regime_full")
            RUN_ROOT = Path("/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_upward_sweep") / run_tag

        cmd = [
            sys.executable,
            "-u",
            str(REPO_DIR / "scripts" / "run_portfolio_ic_upward_sweep.py"),
            "--run-root",
            str(RUN_ROOT),
            "--repo-dir",
            str(REPO_DIR),
            "--resume",
        ]
        print("Foreground command:")
        print(" ".join(cmd))
        print("Run root:", RUN_ROOT)
        print("Runner writes heartbeat.json, training_results.csv, training_results.json, gpu_util.csv.")
        print("GPU monitor helper:", REPO_DIR / "scripts" / "monitor_gpu_util.py")

        try:
            subprocess.run(cmd, cwd=REPO_DIR, text=True, check=True)
        finally:
            if IN_COLAB and AUTO_UNASSIGN_ON_FINISH:
                print("Releasing Colab runtime.")
                runtime.unassign()
        """
    ),
]


def build_notebook(notebook_cells: list[dict]) -> dict:
    return {
        "cells": notebook_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(cells), indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
