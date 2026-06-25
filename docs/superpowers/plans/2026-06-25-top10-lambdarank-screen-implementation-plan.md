# Top-10 PIT LambdaRankIC Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a test-guarded Colab launcher for a 2022 complete-pair LambdaRankIC screen on the reduced PIT S&P 500 GICS top-10 universe.

**Architecture:** Add a new notebook contract test, a focused notebook generator, and the generated notebook. The generator should copy the reliable patterns from the reduced top-10 baseline notebook and LambdaRankIC notebook without modifying the existing pure-IC baseline, full-PIT LambdaRankIC launcher, core loss, trainer, metrics, or base config.

**Tech Stack:** Python standard library (`json`, `pathlib`, `textwrap`, `subprocess` inside generated notebook), pytest, AST parsing, existing `run_experiment.py`, existing `tests/backtest_sp500_daily.py`, Colab Drive artifacts.

---

## File Structure

- Create `tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py`: notebook/generator contract tests for the reduced top-10 LambdaRankIC screen launcher.
- Create `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`: generator that writes the Colab notebook JSON.
- Create `notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb`: generated notebook artifact.
- Keep unchanged: `scripts/gen_sp500_pit_gics_top10_baseline_nb.py`, `notebooks/sp500_pit_gics_top10_baseline_colab.ipynb`, `scripts/gen_lambdarank_ic_pit_nb.py`, `notebooks/lambdarank_ic_pit_colab.ipynb`, `mci_gru/training/losses.py`, `mci_gru/training/trainer.py`, `mci_gru/training/metrics.py`, `mci_gru/config.py`, `configs/config.yaml`.

## Guardrails

- Do not stage or alter the pre-existing dirty `AGENTS.md` change or untracked research docs.
- Do not use the static current top-10 files.
- Do not lower PIT discipline by using row-filter, stayer-only, or complete-stock filtering.
- Do not silently reuse the full-S&P LambdaRankIC `data.pit_min_scoreable_stocks=450`; this launcher must use `100`.
- Do not launch live Colab in this implementation step. Live execution requires a separate explicit approval after local validation.

---

### Task 1: Add The Failing Notebook Contract Test

**Files:**
- Create: `tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py`
- Will fail until Task 2 creates: `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`
- Will fail until Task 2 generates: `notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py`:

```python
import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb")
GENERATOR_PATH = Path("scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py")


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _code_cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_top10_lambdarank_screen_pins_year_budget_and_complete_pair_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "Top-10 PIT LambdaRankIC Screen",
        "docs/superpowers/specs/2026-06-25-top10-lambdarank-screen-design.md",
        "YEARS = [2022]",
        "BASE_SEEDS = [314159]",
        "NUM_MODELS = 1",
        "NUM_EPOCHS = 40",
        "EARLY_STOPPING_PATIENCE = 8",
        "PAIR_CAP = 8192",
        "COMPLETE_PAIR_COUNT_110 = 5995",
        "assert PAIR_CAP >= COMPLETE_PAIR_COUNT_110",
        "training.loss_type=lambdarank_ic",
        "training.selection_metric=val_rank_ic",
        "training.lambdarank_ic_max_pairs_per_day=8192",
        "training.lambdarank_ic_temperature=1.0",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_uses_reduced_2016_start_bundle_and_masked_panel() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_",
        "lseg_20150101_20260622.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json",
        "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_lseg_20150101_20260622.meta.json",
        "data.use_pit_universe=true",
        "data.pit_universe_mode=masked_panel",
        "data.pit_min_scoreable_stocks=100",
        "data.pit_breadth_policy=error",
        "selector_start == '2016-01-04'",
        "\"snapshot_dates\": 127",
        "\"snapshot_min_selected\": 110",
        "\"snapshot_max_selected\": 110",
        "\"pit_union_kdcodes\": 205",
        "\"missing_identifiers\": []",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_preserves_recipe_and_colab_runtime_contract() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1",
        "graph.update_frequency_months=0",
        "graph.top_k=0",
        "graph.top_k_metric=corr",
        "graph.use_multi_feature_edges=true",
        "graph.drop_edge_p=0.1",
        "features=with_momentum",
        "features.include_global_regime=true",
        "features.regime_strict=true",
        "features.regime_include_subsequent_returns=false",
        "G4/L4-class Colab runtime",
        "not T4/CPU",
        "BLOCKED_GPU_NAMES = (\"T4\",)",
        "ALLOWED_GPU_MARKERS = (",
        "FRED_API_KEY is required",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_writes_drive_truth_artifacts_and_backtests() -> None:
    combined = "\n".join(_cell_sources())
    generator = GENERATOR_PATH.read_text(encoding="utf-8")

    required_tokens = [
        "sp500_gics_top10_lambdarank_ic_screen",
        "HEARTBEAT_PATH = DRIVE_RUN_ROOT / \"heartbeat.json\"",
        "data_audit.json",
        "lambdarank_ic_sp500_pit_gics_top10_screen_manifest.json",
        "training_results.csv",
        "training_results.json",
        "backtest_results.csv",
        "backtest_results.json",
        "run_summary.json",
        "logs",
        "summaries",
        "artifacts",
        "tests/backtest_sp500_daily.py",
        "--pit_universe_csv",
        "--top_k",
        "rank-drop",
        "write_heartbeat(\"FAILED\", \"failed\"",
    ]

    for token in required_tokens:
        assert token in combined
        assert token in generator


def test_top10_lambdarank_screen_notebook_code_cells_parse() -> None:
    code_cells = _code_cell_sources()

    assert code_cells
    for source in code_cells:
        ast.parse(source)
```

- [ ] **Step 2: Run the focused test and verify it fails for the expected reason**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sp500_pit_gics_top10_lambdarank_ic_notebook.py -v --basetemp .tmp_pytest\pytest -p no:cacheprovider
```

Expected: fail with `FileNotFoundError` for `notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb` or `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`.

- [ ] **Step 3: Commit the red test**

Run:

```powershell
git add tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py
git commit -m "Add top10 LambdaRankIC notebook contract"
```

Expected: commit succeeds and stages only the new test file.

---

### Task 2: Add The Notebook Generator

**Files:**
- Create: `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`
- Test: `tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py`

- [ ] **Step 1: Create the generator file**

Create `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`.

Use the same lightweight JSON notebook style as the existing generators:

```python
"""Generate a Colab launcher for reduced PIT GICS top-10 LambdaRankIC screen runs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
```

Add markdown and code cells with these exact responsibilities:

- title and operating instructions;
- environment setup, repo clone/switch, GPU detection, FRED secret loading;
- run constants, helper functions, heartbeat writer, run-stream wrapper;
- data staging and audit;
- Hydra override construction, manifest write, training subprocess, saved-prediction backtest subprocess;
- result aggregation and final summary.

Use these exact constants in the generated notebook setup cell:

```python
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
BRANCH = "codex/top10-lambdarank-screen-20260625"
REPO_DIR = Path("/content/MCI-GRU") if IN_COLAB else Path.cwd()
REQUIRE_G4_L4_GPU = True
BLOCKED_GPU_NAMES = ("T4",)
ALLOWED_GPU_MARKERS = (
    "G4",
    "L4",
    "A100",
    "H100",
    "RTX PRO 6000",
    "BLACKWELL",
)
STRICT_GPU_MARKERS: list[str] = []
```

Use these exact screen constants:

```python
YEARS = [2022]
BASE_SEEDS = [314159]
NUM_MODELS = 1
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
PAIR_CAP = 8192
COMPLETE_PAIR_COUNT_110 = 5995
assert PAIR_CAP >= COMPLETE_PAIR_COUNT_110
TOP_K = 10
SPREAD_BPS = 10
SLIPPAGE_BPS = 5
MIN_RANK_DROP = 30
PIT_MIN_SCOREABLE_STOCKS = 100
```

Use these exact reduced bundle names:

```python
MARKET_FILENAME = (
    "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_"
    "lseg_20150101_20260622.csv"
)
PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_pit_universe.csv"
SNAPSHOT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_snapshots.csv"
MARKET_META_FILENAME = MARKET_FILENAME.replace(".csv", ".meta.json")
PIT_META_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260622_meta.json"
```

Use this exact 2022 window:

```python
PIT_WINDOWS = {
    2022: {
        "experiment_name": "sp500_pit_gics_top10_lambdarank_ic_screen_2022",
        "train_start": "2016-01-01",
        "train_end": "2020-12-31",
        "val_start": "2021-01-08",
        "val_end": "2021-12-31",
        "test_start": "2022-01-08",
        "test_end": "2022-12-31",
    }
}
```

Use this run-root and artifact layout:

```python
DRIVE_RUN_ROOT = (
    Path("/content/drive/MyDrive/MCI-GRU-Ablations/sp500_gics_top10_lambdarank_ic_screen") / RUN_TAG
    if IN_COLAB
    else Path.cwd() / "drive_outputs" / "sp500_gics_top10_lambdarank_ic_screen" / RUN_TAG
)
LOCAL_RUN_ROOT = (
    Path("/content/mci_gru_runs/sp500_gics_top10_lambdarank_ic_screen") / RUN_TAG
    if IN_COLAB
    else Path.cwd() / "results" / "sp500_gics_top10_lambdarank_ic_screen" / RUN_TAG
)
LOG_DIR = DRIVE_RUN_ROOT / "logs"
SUMMARY_DIR = DRIVE_RUN_ROOT / "summaries"
ARTIFACT_DIR = DRIVE_RUN_ROOT / "artifacts"
HEARTBEAT_PATH = DRIVE_RUN_ROOT / "heartbeat.json"
MANIFEST_FILENAME = "lambdarank_ic_sp500_pit_gics_top10_screen_manifest.json"
```

Use this base override function in the generated notebook:

```python
def base_overrides(year: int, year_root: Path, repo_pit_csv: Path) -> list[str]:
    window = PIT_WINDOWS[year]
    return [
        f"experiment_name={window['experiment_name']}",
        f"output_dir={year_root.as_posix()}",
        "seed=314159",
        "data.source=csv",
        f"data.filename=data/raw/market/{MARKET_FILENAME}",
        f"data.train_start={window['train_start']}",
        f"data.train_end={window['train_end']}",
        f"data.val_start={window['val_start']}",
        f"data.val_end={window['val_end']}",
        f"data.test_start={window['test_start']}",
        f"data.test_end={window['test_end']}",
        "data.use_pit_universe=true",
        f"data.pit_universe_csv={repo_pit_csv.relative_to(REPO_DIR).as_posix()}",
        "data.pit_universe_mode=masked_panel",
        f"data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}",
        "data.pit_breadth_policy=error",
        f"training.num_models={NUM_MODELS}",
        f"training.num_epochs={NUM_EPOCHS}",
        f"training.early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        "training.learning_rate=5e-5",
        "training.lr_scheduler=cosine",
        "training.loss_type=lambdarank_ic",
        "training.selection_metric=val_rank_ic",
        f"training.lambdarank_ic_max_pairs_per_day={PAIR_CAP}",
        "training.lambdarank_ic_temperature=1.0",
        "training.label_type=returns",
        "training.shuffle_train=true",
        "model.label_t=5",
        "graph.judge_value=0.8",
        "graph.update_frequency_months=0",
        "graph.corr_lookback_days=252",
        "graph.top_k=0",
        "graph.top_k_metric=corr",
        "graph.use_multi_feature_edges=true",
        "graph.append_snapshot_age_days=false",
        "graph.use_lead_lag_features=false",
        "graph.drop_edge_p=0.1",
        "features=with_momentum",
        "features.include_momentum=true",
        "features.include_weekly_momentum=true",
        "features.momentum_encoding=binary",
        "features.momentum_blend_mode=static",
        "features.momentum_blend_fast_weight=0.5",
        "features.include_global_regime=true",
        "features.regime_strict=true",
        "features.regime_enforce_lag_days=0",
        "features.regime_include_subsequent_returns=false",
        "features.regime_change_months=12",
        "features.regime_norm_months=120",
        "features.regime_exclusion_months=1",
        "features.regime_similarity_quantile=0.2",
        "features.regime_min_history_months=24",
        "tracking.enabled=false",
        "tracking.log_artifacts=false",
        "tracking.log_checkpoints=false",
        "tracking.log_predictions=false",
    ]
```

- [ ] **Step 2: Implement generated data audit checks**

Inside the notebook's data audit cell, implement these checks:

```python
selector_start = str(pd.to_datetime(snapshots["as_of_date"]).min().date())
assert selector_start == "2016-01-04", selector_start

snapshot_counts = snapshots.groupby("as_of_date")["kdcode"].nunique()
sector_counts = snapshots.groupby(["as_of_date", "gics_sector"])["kdcode"].nunique()

data_audit = {
    "selector_start": selector_start,
    "snapshot_dates": int(snapshot_counts.shape[0]),
    "snapshot_min_selected": int(snapshot_counts.min()),
    "snapshot_max_selected": int(snapshot_counts.max()),
    "bad_sector_cells": int((sector_counts != 10).sum()),
    "pit_union_kdcodes": int(pit_preview["kdcode"].nunique()),
    "missing_identifiers": market_meta.get("missing_identifiers", []),
    "complete_pair_count_110": COMPLETE_PAIR_COUNT_110,
    "pair_cap": PAIR_CAP,
}
assert data_audit["snapshot_dates"] == 127, data_audit
assert data_audit["snapshot_min_selected"] == 110, data_audit
assert data_audit["snapshot_max_selected"] == 110, data_audit
assert data_audit["bad_sector_cells"] == 0, data_audit
assert data_audit["pit_union_kdcodes"] == 205, data_audit
assert data_audit["missing_identifiers"] == [], data_audit
write_json(DRIVE_RUN_ROOT / "data_audit.json", data_audit)
```

- [ ] **Step 3: Implement training and saved-prediction backtest subprocesses**

The generated notebook training cell should run:

```python
train_cmd = [
    sys.executable,
    "run_experiment.py",
    *overrides,
]
```

The backtest subprocess should run after training using the averaged predictions and same PIT CSV:

```python
backtest_cmd = [
    sys.executable,
    "tests/backtest_sp500_daily.py",
    "--predictions_dir",
    str(year_root / "averaged_predictions"),
    "--pit_universe_csv",
    str(repo_pit_csv),
    "--top_k",
    str(TOP_K),
    "--spread_bps",
    str(SPREAD_BPS),
    "--slippage_bps",
    str(SLIPPAGE_BPS),
    "--min_rank_drop",
    str(MIN_RANK_DROP),
    "--output_dir",
    str(year_root / "saved_prediction_backtest_top10_rankdrop"),
]
```

If `tests/backtest_sp500_daily.py` uses underscore argument names instead of hyphen names, inspect its parser and use the accepted spelling. Keep `--pit_universe_csv`, `--top_k`, and the reduced PIT CSV path.

- [ ] **Step 4: Generate the notebook from the generator**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\gen_sp500_pit_gics_top10_lambdarank_ic_nb.py
```

Expected: output includes `Wrote notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb`.

- [ ] **Step 5: Commit the generator and notebook**

Run:

```powershell
git add scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb
git commit -m "Add top10 LambdaRankIC screen notebook"
```

Expected: commit succeeds and includes only the generator and generated notebook.

---

### Task 3: Turn The Contract Test Green

**Files:**
- Modify: `scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py`
- Modify: `notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb`
- Test: `tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py`

- [ ] **Step 1: Run the new contract test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sp500_pit_gics_top10_lambdarank_ic_notebook.py -v --basetemp .tmp_pytest\pytest -p no:cacheprovider
```

Expected: all tests pass. If a token assertion fails, update the generator and regenerate the notebook rather than editing the notebook directly.

- [ ] **Step 2: Run syntax verification for the generator**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\gen_sp500_pit_gics_top10_lambdarank_ic_nb.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [ ] **Step 3: Run ruff on touched Python files**

Run:

```powershell
.\.venv\Scripts\ruff.exe check scripts\gen_sp500_pit_gics_top10_lambdarank_ic_nb.py tests\test_sp500_pit_gics_top10_lambdarank_ic_notebook.py
```

Expected: output includes `All checks passed!`.

- [ ] **Step 4: Commit contract fixes if needed**

If Steps 1-3 required edits after Task 2's commit, run:

```powershell
git add tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb
git commit -m "Stabilize top10 LambdaRankIC notebook contract"
```

Expected: commit succeeds. If no edits were needed, skip this commit.

---

### Task 4: Run Focused Regression Verification

**Files:**
- Read-only verification across existing LambdaRankIC, top-10, and saved-prediction tests.

- [ ] **Step 1: Run the focused regression suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sp500_pit_gics_top10_lambdarank_ic_notebook.py tests\test_lambdarank_ic_loss.py tests\test_lambdarank_ic_config.py tests\test_lambdarank_ic_trainer.py tests\test_sp500_pit_gics_top10_baseline_notebook.py tests\test_sp500_pit_gics_top10_mcap_export.py tests\test_sp500_gics_top10_mcap_export.py tests\test_pit_saved_prediction_backtests.py -v --basetemp .tmp_pytest\pytest -p no:cacheprovider
```

Expected: all selected tests pass. If a failure is unrelated to the touched files, record it exactly and do not broaden scope until the failure is understood.

- [ ] **Step 2: Check the final diff**

Run:

```powershell
git diff --stat
git diff --name-only
git status --short --branch
```

Expected: only these implementation files are changed or staged:

```text
tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py
scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py
notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb
```

The pre-existing dirty paths may still appear unstaged:

```text
AGENTS.md
docs/research/current/MCI_GRU_PROGRAM_MAP_2026-06-19.md
docs/research/current/MCI_GRU_RESEARCH_OPPORTUNITY_SCAN_2026-06-19.md
```

- [ ] **Step 3: Commit final verification cleanup if needed**

If Step 1 or Step 2 required code or test changes, run:

```powershell
git add tests/test_sp500_pit_gics_top10_lambdarank_ic_notebook.py scripts/gen_sp500_pit_gics_top10_lambdarank_ic_nb.py notebooks/sp500_pit_gics_top10_lambdarank_ic_colab.ipynb
git commit -m "Verify top10 LambdaRankIC screen launcher"
```

Expected: commit succeeds. If no changes were needed, skip this commit.

---

### Task 5: Final Review And Handoff

**Files:**
- Read: `docs/superpowers/specs/2026-06-25-top10-lambdarank-screen-design.md`
- Read: `docs/superpowers/plans/2026-06-25-top10-lambdarank-screen-implementation-plan.md`
- Read: final `git status`

- [ ] **Step 1: Compare implementation against the design**

Check the implementation against these design requirements:

```text
new generator exists
new notebook is generated from it
2022 screen only
reduced 2016-start PIT GICS top-10 filenames
masked_panel
pit_min_scoreable_stocks=100
lambdarank_ic
val_rank_ic
pair cap 8192
seed 314159
1 model
40 epochs
patience 8
G4/L4 non-T4 runtime gate
FRED_API_KEY gate
heartbeat
data audit
manifest
training results
backtest results
run summary
no core loss/trainer/config changes
no existing baseline notebook changes
```

Expected: every line is satisfied by the new test, generator, or notebook.

- [ ] **Step 2: Record verification evidence in the final response**

Report these exact items:

```text
branch name
commits created
files changed
pytest command and result
py_compile command and result
ruff command and result
whether live Colab was launched
remaining dirty unrelated files
```

Expected: live Colab was not launched during local implementation.

- [ ] **Step 3: Ask for live-run approval**

End by asking whether to launch the validated notebook in visible Colab on a non-T4 runtime.

Expected: do not start Colab until the user explicitly approves the live execution step.
