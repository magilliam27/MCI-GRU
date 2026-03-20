"""Generate notebooks/train_test_backtest_workflow.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "train_test_backtest_workflow.ipynb"


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(text: str):
    lines = text.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


cells = []

cells.append(
    md(
        r"""
# MCI-GRU: Train → Test metrics → Backtest → Download

Workflow aligned with `Seed_test (1).ipynb`:

1. **Setup** — repo root, optional Colab clone/install, **`FRED_API_KEY` (required)** for live regime series
2. **Data** — market panel CSV only (path or Colab upload under `data/raw/market/`); regime inputs come from FRED, not a separate file
3. **Train** — `run_experiment.py` (Hydra); outputs under `results/<experiment_name>/<timestamp>/`
4. **Inspect** — `training_summary.json`, `run_metadata.json`, `averaged_predictions/`
5. **Backtest** — `tests/backtest_sp500.py --auto_save`
6. **Package** — zip the run directory (+ backtest folder); Colab `files.download`

**Label horizon:** Training uses `model.label_t` (e.g. 21). Backtest uses `--label_t` for the return / holding window (default **5** in the script). Set `--label_t` to match the horizon you want to simulate (they need not equal `model.label_t`, but you should know what each means).
"""
    )
)

cells.append(md("## 1. Environment and repository root"))

cells.append(
    code(
        r"""
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from google.colab import files as colab_files  # type: ignore
    IN_COLAB = True
except ImportError:
    colab_files = None
    IN_COLAB = False

COLAB_REPO_DIR = Path("/content/MCI-GRU")
MANUAL_REPO_ROOT = None  # e.g. Path(r"C:\Users\you\MCI-GRU")


def resolve_repo_root() -> Path:
    if MANUAL_REPO_ROOT is not None:
        return Path(MANUAL_REPO_ROOT).resolve()
    cwd = Path.cwd().resolve()
    for p in (cwd, cwd.parent, cwd.parent.parent):
        if (p / "run_experiment.py").is_file():
            return p
    if IN_COLAB and COLAB_REPO_DIR.is_dir():
        return COLAB_REPO_DIR.resolve()
    raise FileNotFoundError(
        "Could not find run_experiment.py. Set MANUAL_REPO_ROOT or run from repo / clone on Colab."
    )


REPO_ROOT = resolve_repo_root()
os.chdir(REPO_ROOT)
print("REPO_ROOT =", REPO_ROOT)
print("IN_COLAB =", IN_COLAB)
"""
    )
)

cells.append(md("### 1b. Colab: clone / install (no-op locally)"))

cells.append(
    code(
        r"""
if IN_COLAB:
    REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
    BRANCH = "main"

    if not COLAB_REPO_DIR.exists():
        subprocess.run(
            ["git", "clone", "-b", BRANCH, REPO_URL, str(COLAB_REPO_DIR)], check=True
        )
    else:
        subprocess.run(["git", "-C", str(COLAB_REPO_DIR), "fetch", "origin"], check=False)
        subprocess.run(["git", "-C", str(COLAB_REPO_DIR), "checkout", BRANCH], check=False)
        subprocess.run(["git", "-C", str(COLAB_REPO_DIR), "pull", "origin", BRANCH], check=False)

    os.chdir(COLAB_REPO_DIR)
    REPO_ROOT = COLAB_REPO_DIR.resolve()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )
    print("Colab ready:", REPO_ROOT)
else:
    print("Local: skipped clone/install.")
"""
    )
)

cells.append(md("### 1c. `FRED_API_KEY` (required for regime features)"))

cells.append(
    code(
        r"""
# Paste your key here, OR set the environment variable before starting Jupyter/Colab.
MY_FRED_KEY = ""  # e.g. "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

if MY_FRED_KEY.strip():
    os.environ["FRED_API_KEY"] = MY_FRED_KEY.strip()

if not os.environ.get("FRED_API_KEY"):
    raise ValueError(
        "Set MY_FRED_KEY in this cell or export FRED_API_KEY. "
        "It is required to load regime inputs (yields, oil, VIX, SP500, copper) from FRED."
    )
print("FRED_API_KEY is set (length", len(os.environ["FRED_API_KEY"]), ").")
"""
    )
)

cells.append(
    md(
        r"""
## 2. Market data CSV (and optional Colab upload)

Regime inputs are **not** read from a file: `features=with_momentum` with `include_global_regime=true` leaves `regime_inputs_csv` null, so `load_regime_inputs` pulls macro series from the **FRED API**.

To use a prebuilt regime CSV instead, add e.g. `features.regime_inputs_csv=data/raw/market/your_file.csv` (full column contract in `docs/REGIME_DATA_CONTRACT.md`).
"""
    )
)

cells.append(
    code(
        r"""
MARKET_CSV = REPO_ROOT / "data" / "raw" / "market" / "sp500_2019_universe_data_through_2026.csv"
MARKET_CSV.parent.mkdir(parents=True, exist_ok=True)

if IN_COLAB and colab_files is not None:
    if not MARKET_CSV.is_file():
        print("Upload market universe CSV (any filename; saved as MARKET_CSV)…")
        up = colab_files.upload()
        if not up:
            raise RuntimeError("Market CSV upload cancelled or empty")
        MARKET_CSV.write_bytes(next(iter(up.values())))

if not MARKET_CSV.is_file():
    raise FileNotFoundError(f"Market CSV missing: {MARKET_CSV}")

print("MARKET_CSV OK:", MARKET_CSV)
"""
    )
)

cells.append(md("## 3. Hydra configuration (Seed_test `BASE`-aligned)"))

cells.append(
    code(
        r"""
# Mirrors Seed_test (1).ipynb BASE: with_momentum + dynamic momentum + global regime.
# Regime: FRED only — do NOT set features.regime_inputs_csv (stays null from with_momentum.yaml).
EXPERIMENT_NAME = "notebook_train_backtest"
SEED = 7
NUM_MODELS = 10
HIS_T = 60
LABEL_T = 21

rel_market = MARKET_CSV.relative_to(REPO_ROOT).as_posix()

HYDRA_OVERRIDES = [
    f"experiment_name={EXPERIMENT_NAME}",
    f"seed={SEED}",
    # Swap default data pack (must NOT use +data — that adds a 2nd data config and Hydra errors:
    # "Multiple values for data. To override a value use 'override data: temporal_2019'")
    "data=temporal_2019",
    "data.source=csv",
    f"data.filename={rel_market}",
    "features=with_momentum",
    "features.include_weekly_momentum=false",
    "features.momentum_encoding=binary",
    "features.momentum_blend_mode=dynamic",
    "features.momentum_dynamic_correction_fast_weight=0.15",
    "features.momentum_dynamic_rebound_fast_weight=0.7",
    "features.momentum_dynamic_lookback_periods=0",
    "features.momentum_dynamic_min_history=252",
    "features.momentum_dynamic_min_state_observations=3",
    "features.include_global_regime=true",
    "features.regime_enforce_lag_days=0",
    f"model.his_t={HIS_T}",
    f"model.label_t={LABEL_T}",
    "model.use_multi_scale=false",
    "model.use_self_attention=false",
    "model.activation=elu",
    "model.latent_init_scale=0.02",
    "training.label_type=rank",
    f"training.num_models={NUM_MODELS}",
    "training.num_epochs=100",
    "training.early_stopping_patience=15",
]

print(f"Hydra overrides ({len(HYDRA_OVERRIDES)} args):")
for o in HYDRA_OVERRIDES:
    print(" ", o)
"""
    )
)

cells.append(md("## 4. Train"))

cells.append(
    code(
        r"""
RUN_TRAIN = True


def run_training():
    cmd = [sys.executable, str(REPO_ROOT / "run_experiment.py"), *HYDRA_OVERRIDES]
    print("CMD:", " ".join(cmd[:3]), "... +", len(HYDRA_OVERRIDES), "overrides")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Training exit code {proc.returncode}. Full subprocess output is printed above."
        )


if RUN_TRAIN:
    run_training()
else:
    print("Skipped training (RUN_TRAIN=False).")
"""
    )
)

cells.append(md("## 5. Latest run → val/test summary"))

cells.append(
    code(
        r"""
RESULTS_ROOT = REPO_ROOT / "results"


def is_hydra_timestamp_dir(name: str) -> bool:
    return bool(re.fullmatch(r"\d{8}_\d{6}", name))


def find_latest_run(results_root: Path, experiment_name: str) -> Path:
    base = results_root / experiment_name
    if not base.is_dir():
        raise FileNotFoundError(base)
    candidates = [p for p in base.iterdir() if p.is_dir() and is_hydra_timestamp_dir(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No runs under {base}")
    return sorted(candidates)[-1]


LATEST_RUN = find_latest_run(RESULTS_ROOT, EXPERIMENT_NAME)
PRED_DIR = LATEST_RUN / "averaged_predictions"

print("LATEST_RUN:", LATEST_RUN)
print("PRED_DIR:", PRED_DIR)

ts = LATEST_RUN / "training_summary.json"
if ts.is_file():
    with open(ts, encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2)[:6000])
else:
    print("No training_summary.json")

rm = LATEST_RUN / "run_metadata.json"
if rm.is_file():
    with open(rm, encoding="utf-8") as f:
        meta = json.load(f)
    print("run_metadata keys:", sorted(meta.keys()))
"""
    )
)

cells.append(md("## 6. Backtest"))

cells.append(
    code(
        r"""
BACKTEST_LABEL_T = 5
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
TOP_K = 10
BACKTEST_SUFFIX = ""

rel_market_bt = MARKET_CSV.relative_to(REPO_ROOT).as_posix()

bt_cmd = [
    sys.executable,
    str(REPO_ROOT / "tests" / "backtest_sp500.py"),
    "--predictions_dir",
    str(PRED_DIR),
    "--data_file",
    rel_market_bt,
    "--test_start",
    TEST_START,
    "--test_end",
    TEST_END,
    "--top_k",
    str(TOP_K),
    "--label_t",
    str(BACKTEST_LABEL_T),
    "--auto_save",
    "--plot",
]
if BACKTEST_SUFFIX:
    bt_cmd.extend(["--backtest_suffix", BACKTEST_SUFFIX])

print("Backtest:", " ".join(bt_cmd))
proc = subprocess.run(bt_cmd, cwd=str(REPO_ROOT))
if proc.returncode != 0:
    raise RuntimeError(f"Backtest exit code {proc.returncode}")

bt_name = "backtest" + (BACKTEST_SUFFIX or "")
BACKTEST_DIR = LATEST_RUN / bt_name
print("BACKTEST_DIR:", BACKTEST_DIR)
metrics_file = BACKTEST_DIR / "backtest_metrics.json"
if metrics_file.is_file():
    with open(metrics_file, encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2)[:4000])
"""
    )
)

cells.append(md("## 7. Zip + download"))

cells.append(
    code(
        r"""
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_base = REPO_ROOT / "results" / f"{EXPERIMENT_NAME}_bundle_{stamp}"
archive_path = shutil.make_archive(
    str(zip_base), "zip", root_dir=str(LATEST_RUN.parent), base_dir=LATEST_RUN.name
)
print("Archive:", archive_path)

if IN_COLAB and colab_files is not None:
    colab_files.download(archive_path)
else:
    print("Local: open the .zip above or copy from:", LATEST_RUN)
"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT)
