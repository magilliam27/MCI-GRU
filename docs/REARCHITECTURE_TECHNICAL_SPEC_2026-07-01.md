# MCI-GRU Repo Rearchitecture — Technical Specification

> Companion to `docs/ARCHITECTURE_REVIEW.md` (ML/research posture review) and
> the software-architecture chat review that preceded this spec. This document
> is the execution plan: exact target file layout, exact signatures, exact
> call sites that must be updated, phase sequencing, and acceptance criteria.
>
> **Non-goal:** no model math, training semantics, no-lookahead behavior, or
> config defaults change in this spec. Every workstream here is a structural
> move — same behavior, different file organization. Any workstream that
> can't be done behavior-preserving is flagged explicitly.

## 0. Principles

1. **Behavior-preserving first.** Every split keeps the exact same public
   import paths working (via `__init__.py` re-exports or shim modules) unless
   a call site is explicitly listed as "must change."
2. **One workstream = one PR = one green test run.** `.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest` and `ruff check .` must pass before merging each workstream, not just at the end.
3. **Phase gates.** Phase 2 (core pipeline decomposition) does not start until
   Phase 0 and Phase 1 are merged and green — Phase 2 touches no-lookahead-
   sensitive code and needs a stable base to bisect against if something
   regresses.
4. **No new abstractions without a second caller.** Don't introduce a shared
   module for logic that only one place uses (see model-split helper
   analysis — no `_common.py` needed today).
5. **Rollback unit = one workstream.** Each workstream's PR should be
   revertable independently without breaking the others.

---

## Phase 0 — Housekeeping (parallelizable, zero behavior risk)

These five workstreams don't touch training/inference math and can be done
in any order, in parallel, by different people.

### WS-A: Dependency manifest consolidation + lockfile

**Current state:** `pyproject.toml` and `requirements.txt` duplicate ~12 core
packages with inconsistent optional-vs-required treatment (`mlflow`,
`fredapi` are "main" in `requirements.txt`, extras in `pyproject.toml`). No
lockfile exists anywhere (confirmed: zero `uv.lock`/`poetry.lock`/
`Pipfile.lock`/`requirements-lock.txt`). Documented workflow throughout
`AGENTS.md`, CI, and `README.md` is **pip + venv + setuptools** — `uv` is
mentioned exactly once, as an unactioned suggestion in
`docs/ARCHITECTURE_REVIEW.md:249`.

**Target:**
- `pyproject.toml` remains the single source of truth for dependency ranges.
- Delete `requirements.txt`, or regenerate it as a derived artifact
  (`pip-compile pyproject.toml -o requirements.txt`) so it can't drift.
- Add `pip-tools` to `[dev]` extras; generate `requirements.lock` (pinned,
  transitive-closed) via `pip-compile --extra=dev --extra=fred -o requirements.lock pyproject.toml`.
- CI installs from the lock in the `test` job; a separate lightweight CI
  check (or pre-commit hook) fails if `requirements.lock` is stale relative
  to `pyproject.toml`.

**Rationale for pip-tools over `uv`/Poetry:** pip-tools requires zero build-
backend migration and zero AGENTS.md/CI/README rewrite beyond one new lock
file and one new install line — least disruption given the current
documented `pip install -e ".[dev]"` workflow.

**Steps:**
1. Add `pip-tools` to `pyproject.toml` `[dev]` extras.
2. Generate `requirements.lock`.
3. Update `.github/workflows/ci.yml` install step to `pip install -r requirements.lock` (or `pip-sync`).
4. Delete `requirements.txt` (or replace with a comment pointing to `pyproject.toml` + `requirements.lock`).
5. Update `AGENTS.md`/`README.md` install instructions.

**Risk:** low. Verification: CI green, `pip install -e ".[dev,fred]"` still works locally on Windows venv.

---

### WS-B: Relocate `mci_gru/cockpit/` to top-level `cockpit/`

**Current state:** ~1,504 lines of agent/GitHub-ops tooling (`gh` CLI
subprocess automation, git topology evidence, markdown rendering) living
inside the ML research package namespace. Verified **zero coupling** in
either direction with `pipeline.py`/`training/` — this is a pure move, not a
decoupling exercise.

**Target layout:**
```
cockpit/
├── __init__.py       (same 8 re-exports: CockpitReport, Decision, GitHubAction, RunColor, Workstream, WorkstreamStatus, render_cockpit_packet, render_workstream_register)
├── _compat.py
├── models.py
├── git.py
├── evidence.py
├── github.py
├── render.py
└── runner.py
```

**Exact call sites requiring an import-line change (23 lines, 11 files):**

| File:Line | Change |
|---|---|
| `mci_gru/cockpit/__init__.py:1,9` | → `cockpit/__init__.py`, imports become intra-package (`from cockpit.models import ...`) |
| `mci_gru/cockpit/models.py:6` | `from mci_gru.cockpit._compat import StrEnum` → `from cockpit._compat import StrEnum` |
| `mci_gru/cockpit/evidence.py:8-9` | `mci_gru.cockpit.git`/`.models` → `cockpit.git`/`.models` |
| `mci_gru/cockpit/github.py:7` | `mci_gru.cockpit.git` → `cockpit.git` |
| `mci_gru/cockpit/render.py:8` | `mci_gru.cockpit.models` (TYPE_CHECKING) → `cockpit.models` |
| `mci_gru/cockpit/runner.py:7-10,21,27` | 5 internal cockpit imports → `cockpit.*` |
| `scripts/refresh_cockpit.py:12` | `from mci_gru.cockpit.runner import ...` → `from cockpit.runner import ...` |
| `tests/test_cockpit_models.py:7` | same pattern |
| `tests/test_cockpit_render.py:5,13` | same pattern |
| `tests/test_cockpit_runner.py:8-9,171,195` | import lines + 2 `monkeypatch.setattr("mci_gru.cockpit.*")` strings → `cockpit.*` |
| `tests/test_cockpit_github.py:9,100` | import + 1 `monkeypatch.setattr` string |

**`pyproject.toml` changes required:**
```toml
[tool.setuptools.packages.find]
include = ["mci_gru*", "cockpit*"]

[tool.ruff.lint.isort]
known-first-party = ["mci_gru", "cockpit"]
```

**Docs to update (Python import paths, not just CLI refs):** `docs/superpowers/plans/2026-06-20-mci-gru-cockpit-agent-implementation-plan.md` (18 lines) and `...-github-sync-implementation-plan.md` (4 lines). CLI-only docs (`docs/agents/cockpit/RUNBOOK.md`, `docs/index.md`) need no change — they invoke `scripts/refresh_cockpit.py`, unaffected.

**Risk:** low, purely mechanical. Verification: `tests/test_cockpit_*.py` (5 files) green, `scripts/refresh_cockpit.py --help` runs.

---

### WS-C: Relocate backtest CLIs out of `tests/`

**Current state:** `tests/backtest_sp500.py` (2,989 lines) and
`tests/backtest_sp500_daily.py` (2,124 lines) are argparse CLI scripts with
no `def test_*`/`class Test*` — not collected by pytest
(`python_files = ["test_*.py"]`), but imported via `sys.path` hacks by real
tests and invoked via subprocess by orchestration scripts and 11 notebook
generators.

**Target layout:**
```
mci_gru/evaluation/backtest_engine.py   # core: load_stock_data, load_predictions,
                                         # simulate_trading_strategy*, evaluate,
                                         # save_backtest_results, setup_backtest_tracking
scripts/backtest_sp500.py               # thin CLI wrapper (argparse + call into backtest_engine)
scripts/backtest_sp500_daily.py         # thin CLI wrapper (daily-only subset)
```

**Every call site requiring a change (exhaustive, from grep):**

| File:Line | Old | New |
|---|---|---|
| `tests/test_backtest_fairness.py:296-300` | `sys.path.insert(...)` + `import backtest_sp500 as bp` | `from mci_gru.evaluation import backtest_engine as bp` |
| `tests/test_mlflow_tracking.py:19` | `from tests.backtest_sp500 import setup_backtest_tracking` | `from mci_gru.evaluation.backtest_engine import setup_backtest_tracking` |
| `scripts/scratch/scratch_backtest.py:17-23` | `from tests.backtest_sp500 import (...)` | `from mci_gru.evaluation.backtest_engine import (...)` |
| `paper_trade/scripts/compare_regime.py:42` | `PROJECT_ROOT / "tests" / "backtest_sp500.py"` | `PROJECT_ROOT / "scripts" / "backtest_sp500.py"` |
| `scripts/run_pit_saved_prediction_backtests.py:342` | `repo_dir / "tests" / "backtest_sp500_daily.py"` | `repo_dir / "scripts" / "backtest_sp500_daily.py"` |
| `scripts/run_pit_repeated_seed_backtest_sensitivity.py:459` | same | same |
| `tests/test_pit_saved_prediction_backtests.py:26,42` | fixture path + assertion | update path string |
| `tests/test_pit_repeated_seed_backtest_sensitivity.py:111,165` | fixture path + command-string assertion | update path string |
| 11× `scripts/gen_*_nb.py` (see list below) | embedded `tests/backtest_sp500*.py` path string in generated notebook code | `scripts/backtest_sp500*.py` |
| `README.md:585`, `docs/MLFLOW_TRACKING.md:71,81` | `python tests/backtest_sp500.py` / `python -m tests.backtest_sp500` | `python scripts/backtest_sp500.py` |

Notebook generators to update: `gen_pit_masked_panel_2022_2025_nb.py:575`, `gen_promising_backtest_nb.py:268`, `gen_volatility_targeting_repeated_seed_nb.py:496`, `gen_volatility_targeting_ablation_nb.py:572`, `gen_long_history_pit_eval_nb.py:652`, `gen_train_test_nb.py:328`, `gen_sp500_pit_gics_top10_baseline_nb.py:433`, `gen_performance_proof_nb.py:493`, `gen_pit_universe_validation_nb.py:708`, `gen_pit_repeated_seed_replication_nb.py:987`, `gen_temporal_rolling_backtest_nb.py:427`.

**Migration safety net:** leave `tests/backtest_sp500.py` and
`tests/backtest_sp500_daily.py` as deprecated one-line re-export shims
(`from scripts.backtest_sp500 import *`) for one release cycle before
deleting, so any external/Colab reference isn't broken mid-transition.

**Risk:** medium — largest surface area of any Phase 0 workstream (25+ call
sites). Do this one with a dedicated PR and full-repo grep verification
before merging (`rg "backtest_sp500"` should show zero remaining `tests/`
paths outside the deprecated shims).

---

### WS-D: Activate pytest markers

**Current state:** `pyproject.toml:91-95` declares `slow`, `requires_data`,
`requires_fred`, `requires_lseg` markers. Zero usage anywhere
(`grep -r "@pytest.mark.requires" tests/` → 0 hits). `tests/conftest.py:8-17`
does nodeid-substring auto-skip instead, which misses several files that
touch FRED/LSEG without those strings in their filename.

**Target:** tag the following files (found via grep for FRED/LSEG/real-CSV
signals):

| File | Marker | Reason |
|---|---|---|
| `tests/test_regime_features.py` | `requires_fred` | exercises `DataManager.load_regime_inputs()` FRED path |
| `tests/test_pit_masked_panel_notebook.py` | `requires_fred` | asserts FRED_API_KEY notebook contract |
| `tests/test_volatility_targeting_ablation_notebook.py` | `requires_fred` | same |
| `tests/test_volatility_targeting_repeated_seed_notebook.py` | `requires_fred` | same |
| `tests/test_pit_repeated_seed_replication_notebook.py` | `requires_lseg`, `requires_fred` | hardcoded LSEG union CSV path + FRED env vars |
| `tests/test_sp500_pit_gics_top10_baseline_notebook.py` | `requires_lseg` | references LSEG-sourced CSV filename |
| `tests/test_ablation_notebook_gpu_preflight.py` | `requires_data` | asserts real production CSV filename |
| `tests/test_long_history_issue23.py` | `requires_data` | same |
| `tests/test_pit_alias_coverage_audit.py` | `requires_lseg` | imports LSEG alias audit tooling |

**Then** update `.github/workflows/ci.yml` test step from
`-m "not requires_data and not requires_lseg"` (currently a no-op filter) to
verify it actually excludes the newly-tagged files, and add `-m "not slow"`
once any test is tagged `slow`.

**Risk:** low. Verification: `pytest tests/ -m "not requires_data and not requires_lseg" --collect-only` shows the 9 files excluded; full local run with credentials still passes untagged.

---

### WS-E: Enforce docs source-of-truth policy mechanically

**Current state:** `docs/agents/domain.md` declares a SOT hierarchy but
`docs/research/README.md` itself admits root-level dated reports "not yet
moved" to `research/{current,archive}`. PIT evidence is scattered across
root `docs/*.md`, `docs/research/current/`, `docs/audits/`, and 10+
`docs/handoffs/*.md`.

**Target:** a small CI/lint script (`scripts/check_docs_sot.py`) that fails
if a new `docs/*.md` file matching a dated-report naming pattern
(`^[A-Z0-9_]+_\d{4}-\d{2}-\d{2}\.md$`) is added directly under `docs/`
instead of `docs/research/{current,archive}/`. Run it as a CI lint step
alongside `ruff check .`. This doesn't move the existing ~24 root-level
reports (that's a separate one-time cleanup, out of scope here) — it just
stops the bleeding for new files.

**Risk:** near-zero (additive CI check only, doesn't move files).

---

## Phase 1 — Mechanical file splits (behavior-preserving, moderate mechanical risk)

Gate: Phase 0 merged and green. These three workstreams are independent of
each other and can run in parallel, but each depends on nothing in Phase 2.

### WS-F: Split `mci_gru/models/mci_gru.py` (827 lines → 6 files)

**Target layout:**
```
mci_gru/models/
├── __init__.py       (same 11-name __all__, re-exports from submodules below)
├── temporal.py        # AttentionResetGRUCell, ImprovedGRU, GRUWithAttention,
│                       # CausalTransformerEncoder, MultiScaleTemporalEncoder,
│                       # _transformer_nhead_for_d_model
├── graph.py            # GATBlock, GATLayer = GATBlock, GATLayer_1 = GATBlock,
│                       # _make_activation
├── attention.py         # SelfAttention
├── latent.py            # MarketLatentStateLearner
├── trunk.py             # StockPredictionModel, _make_output_activation,
│                       # _maybe_ln_ch, _maybe_drop, _apply_edge_dropout
└── factory.py           # create_model
```

**Dependency graph (verified, no cross-cycles):**
```
temporal.py   → (leaf; CausalTransformerEncoder uses _transformer_nhead_for_d_model, same file)
graph.py      → (leaf)
attention.py  → (leaf)
latent.py     → (leaf)
trunk.py      → temporal, graph, attention, latent
factory.py    → trunk
```
No helper is used by more than one target file — **no `_common.py` needed.**

**Compatibility shim:** keep `mci_gru/models/mci_gru.py` as a re-export shim
(`from mci_gru.models.temporal import *; from mci_gru.models.graph import *; ...`)
because 4 test files import directly from this path:
`tests/test_dynamic_graph_updates.py:639,657` (`GATBlock`),
`tests/test_pit_masked_panel.py:24` (`SelfAttention`),
`tests/test_mci_gru_phase2.py:6-7` (`CausalTransformerEncoder`, `MultiScaleTemporalEncoder`, `GRUWithAttention`, `ImprovedGRU`, `MarketLatentStateLearner`, `create_model`).

**`__init__.py` re-exports (must stay identical):** `StockPredictionModel`, `ImprovedGRU`, `GRUWithAttention`, `MultiScaleTemporalEncoder`, `AttentionResetGRUCell`, `GATBlock`, `GATLayer`, `GATLayer_1`, `SelfAttention`, `MarketLatentStateLearner`, `create_model`. Consumers verified unaffected: `run_experiment.py:58`, `paper_trade/scripts/infer.py:37`, `paper_trade/scripts/compare_regime.py:32`, `scripts/verify_baseline.py:99`, `tests/test_phase3_graph_and_walkforward.py:19`.

**Steps:**
1. Create the 6 new files, moving classes verbatim (no logic changes).
2. Update `models/__init__.py` to import from new locations.
3. Turn `models/mci_gru.py` into the compat shim.
4. Run full test suite; specifically the 4 files with direct submodule imports.
5. (Optional follow-up, not required for this PR) migrate those 4 test files to import from `mci_gru.models` or the new submodules directly, then delete the shim in a later cleanup PR.

**Risk:** low-medium. Pure move; risk is import-path typos, not logic. Verification: `ruff check .` (catches unused/missing imports) + full pytest.

---

### WS-G: Split `mci_gru/graph/builder.py` (450 lines → 3 files)

**Target layout:**
```
mci_gru/graph/
├── __init__.py        (same __all__: GraphBuilder, GraphSchedule)
├── schedule.py          # GraphSchedule (verbatim move, lines 17-66)
├── correlation.py        # pure math: _daily_returns_pivot, compute_correlation_matrix,
│                       # build_edges, _select_edges_threshold, _select_edges_topk,
│                       # _lead_lag_columns — becomes free functions taking config values
│                       # as explicit params instead of `self`
├── builder.py            # GraphBuilder orchestrator: __init__, build_graph,
│                       # precompute_snapshots, get_update_dates, get_stats,
│                       # + legacy lazy trio (should_update, update_if_needed, get_current_graph)
└── utils.py             # unchanged (edge_feature_dim)
```

**Classification (from source audit):**
- **Pure math, no `self` mutation** → `correlation.py`: `_daily_returns_pivot`, `compute_correlation_matrix`, `build_edges`, `_select_edges_threshold`, `_select_edges_topk`, `_lead_lag_columns`.
- **Mutates `self` state** (`correlation_matrix`, `last_update_date`, `current_edge_index`, `current_edge_weight`) → stays on `GraphBuilder`: `build_graph`, `precompute_snapshots`.
- **Legacy lazy-update API**, used only by `tests/test_dynamic_graph_updates.py` (5 call sites) and `scripts/verify_baseline.py:275-281` → stays on `GraphBuilder`, not deleted (still load-bearing for that test file and script), just co-located with a comment marking it legacy.

**Call sites to preserve via `__init__.py` re-export:** `mci_gru/pipeline.py:35,536,540`, `mci_gru/data/data_manager.py:28` (`GraphSchedule`, TYPE_CHECKING), `tests/test_dynamic_graph_updates.py:26`, `tests/test_phase3_graph_and_walkforward.py:18`, `scripts/verify_baseline.py:238,266,271`.

**Steps:**
1. Move `GraphSchedule` verbatim to `schedule.py`.
2. Extract pure-math methods to free functions in `correlation.py` (signature change: drop `self`, add explicit `judge_value`/`top_k`/`top_k_metric`/`lead_lag_days` params).
3. `GraphBuilder` methods (`build_edges` etc.) become thin wrappers calling the `correlation.py` free functions with `self.<field>` as args — preserves the existing `GraphBuilder.build_edges(...)` call site in `mci_gru/pipeline.py` and elsewhere without any caller change.
4. Re-export `GraphBuilder`, `GraphSchedule` from `graph/__init__.py` exactly as today.
5. Run `tests/test_dynamic_graph_updates.py` and `tests/test_phase3_graph_and_walkforward.py` — these exercise both the math and the legacy lazy path.

**Risk:** medium. The pure-math extraction changes internal call signatures (self-methods → free functions) even though `GraphBuilder`'s own public methods are untouched — care needed that `build_edges`'s wrapper passes identical args in identical order (e.g. `top_k_metric` string constants, `lead_lag_days` list identity).

---

### WS-H: Split `mci_gru/training/trainer.py` (738 lines → 2 files)

**Target layout:**
```
mci_gru/training/
├── __init__.py        (same lazy-loaded __all__)
├── trainer.py            # Trainer class only: __init__, train, predict,
│                       # save_predictions, load_best_model, _train_epoch,
│                       # _validate, _sync_cuda_if_needed, _write_profile_row
│                       # + TrainingResult dataclass, prediction_rows_for_date
├── ensemble.py            # train_multiple_models (ensemble loop, per-model
│                       # seeding, MLflow child-run orchestration, averaging,
│                       # averaged-CSV export)
└── losses.py / metrics.py  (unchanged)
```

**Verified boundary:** `train_multiple_models` only touches `Trainer`'s
**public** surface — `Trainer(...)`, `.train()`, `.load_best_model()`,
`.predict()`, `.save_predictions()`, and one public attribute write
(`trainer.last_best_model_path = ...`). No private-method or private-state
reach-through. This means the split is a clean interface boundary, not a
false decomposition.

**MLflow calls that move to `ensemble.py`:** `tracking_manager.create_child_run(...)` (per-member child run), `child_tracking.log_epoch_metrics` (passed as `epoch_callback` into `trainer.train()`), `child_tracking.log_metrics({...})` (best_val_loss/ic/rank_ic/final_train_loss/epochs_trained), `child_tracking.log_artifact(...)` / `log_artifacts(...)` (checkpoint + prediction CSVs, gated by `config.tracking.log_artifacts`).

**Call sites to preserve:**
- `mci_gru/training/__init__.py` lazy-exports `Trainer` and `train_multiple_models` — update the lazy-loader's source module for `train_multiple_models` to `ensemble.py`, re-export `TrainingResult`/`prediction_rows_for_date` from `trainer.py` unchanged.
- `run_experiment.py:61` imports both names from `mci_gru.training` — unaffected by the split as long as `__init__.py` re-exports correctly.
- `tests/test_lambdarank_ic_trainer.py:8`, `tests/test_portfolio_ic_trainer.py:9` import `Trainer` directly from `mci_gru.training.trainer` — unaffected (class stays there).
- `tests/test_pit_masked_panel.py:27` imports `prediction_rows_for_date` from `mci_gru.training.trainer` — unaffected (stays there).

**Risk:** low. Verified clean interface boundary means this is closer to a pure file move than the model/graph splits.

---

## Phase 2 — Core pipeline decomposition (higher risk, no-lookahead-sensitive)

Gate: Phase 0 + Phase 1 merged and green. Do not start until then — this
phase touches the exact code paths guarded by the repo's no-lookahead
invariant, so bisecting against a clean base matters if a regression shows
up in backtest results.

### WS-I: Extract `mci_gru/data/transforms.py` (shared normalize/impute primitives)

**Why first in Phase 2:** WS-J (pipeline staging) and the paper-trade
dedup both depend on this module existing — do it once, use it twice.

**Current duplication (verified near-identical logic in two places):**

| Logic | `mci_gru/pipeline.py` | `paper_trade/scripts/infer.py` |
|---|---|---|
| Per-day cross-sectional NaN impute | lines 345-357 | lines 182-192 |
| 3σ clip + z-score | `_apply_normalisation` 202-217 | lines 194-199 |
| Single-date graph-feature fill | via `generate_graph_features` | lines 220-228 |
| Window/tensor indexing | `_stock_feature_row_slice` 220-233 + `_build_tensors` | lines 179-180, 201-218 (manual pivot) |

**New module, exact signatures:**

```python
# mci_gru/data/transforms.py

def impute_feature_nans_by_day(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Per-day cross-sectional mean fill, then zero-fill remaining NaNs."""

def compute_zscore_norm_stats(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_end: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train-period-only mean/std. Moved verbatim from pipeline._compute_norm_stats."""

def normalize_features_zscore(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    means: Mapping[str, float],
    stds: Mapping[str, float],
    *,
    clip_sigma: float = 3.0,
    default_mean: float = 0.0,
    default_std: float = 1.0,
) -> pd.DataFrame:
    """3-sigma clip + z-score. Moved from pipeline._apply_normalisation;
    default_mean/default_std params reconcile pipeline's `.get(col, 0.0/1.0)`
    fallback with infer.py's direct `means[col]` indexing — infer.py callers
    pass no default (KeyError on missing col, matching current behavior)."""

def build_single_date_tensors(
    df_norm: pd.DataFrame,
    kdcode_list: Sequence[str],
    feature_cols: Sequence[str],
    his_t: int,
    target_date: str,
    *,
    use_polars: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (time_series (1,S,his_t,F), graph_features (1,S,F)) for one date.
    Single-date special case of generate_time_series_features/generate_graph_features."""
```

**Note on the z-score fallback discrepancy:** pipeline.py's `_apply_normalisation` uses `.get(col, 0.0/1.0)` (silently defaults missing stats to no-op), while `infer.py` indexes `means[col]`/`stds[col]` directly (raises `KeyError` on a missing column). This is an existing behavioral difference between training and inference paths — **flag, don't silently unify.** The shared function takes explicit `default_mean`/`default_std` params so each caller keeps its current fallback behavior; unifying the fallback policy itself is out of scope for this spec (raise as a separate research question if desired).

**Steps:**
1. Create `mci_gru/data/transforms.py` with the four functions above, moving logic verbatim (only signature shape changes, not computation).
2. Point `pipeline.py:90-105,202-217,345-357` at the new module (replace bodies with imports + calls).
3. Point `pipeline.py:599-602,616-617` (index-level path) at the same functions.
4. Write direct unit tests for the four functions in isolation (currently untestable without full `ExperimentConfig` + I/O) — this is a net-new testability win, add to `tests/test_preprocessing_vectorised.py` or a new `tests/test_transforms.py`.
5. Do **not** touch `infer.py` yet — that's WS-K, done after WS-J to avoid two moving pieces at once.

**Risk:** medium. This is copy-then-verify: run the full pytest suite plus a real (or synthetic-CSV) `run_experiment.py` smoke run before/after and diff `run_metadata.json` norm stats + a sample of `averaged_predictions/*.csv` to confirm bit-for-bit (or float-tolerance) identical output.

---

### WS-J: Decompose `prepare_data()` into staged functions

**Current state:** `prepare_data()` (lines 324-573, ~250 lines) is a single
function orchestrating 14 sequential concerns with ~18 branch axes (PIT
mode × normalization method × universe filter × dynamic graph × sector ×
label type). Ten private helpers already exist (`_load_auxiliary_data`,
`_compute_norm_stats`, `_build_feature_reference`, `_apply_pit_universe`,
`_filter_to_masked_pit_panel`, `_audit_pit_breadth`, `_pit_mask_summary`,
`_apply_normalisation`, `_stock_feature_row_slice`, `_build_tensors`) but
orchestration, branching, and 20+ `print` calls remain inline in the
function body.

**Key coupling hazards identified (must be preserved exactly, not "cleaned up"):**

1. **Three DataFrames coexist with different roles**: raw engineered `df` (used for labels, PIT masks, and the correlation graph), `df_norm` (normalized, pre-filter), `df_filtered` (normalized + universe-filtered, used for windows/tensors). A staged refactor must carry all three explicitly (e.g. via a `PipelineFrames` dataclass), not collapse them into one variable — collapsing was flagged by the exploration pass as the single biggest risk of silently changing behavior.
2. **`df_for_labels` has different semantics per mode**: unnormalized `df` in stock-panel mode, but normalized `df_filtered` in `prepare_data_index_level`. Don't assume one meaning.
3. **Normalization fit scope under masked PIT**: `_compute_norm_stats` fits on a PIT-row-filtered *subset* but applies to the *full* `df_filled` panel. Reordering filter-vs-fit in a refactor is the most direct way to introduce lookahead — this must be covered by a regression test before refactoring, not after.
4. **`data_manager.kdcode_list` mutation**: `prepare_data` line 411 and `DataManager.filter_complete_stocks{,_per_split}` (lines 478/515) mutate `self.kdcode_list` on the shared `DataManager` instance as a side effect, while `kdcode_list` is also threaded through explicitly as a return value. The staged refactor should treat the explicit return value as canonical and the `DataManager` mutation as a legacy side-effect to preserve (not rely on) — do not remove the mutation, since it's unclear what else might read `data_manager.kdcode_list` post-call (`combined_collate_fn` does not, per WS-I research, but grep before removing).
5. **`label_type` two-step under PIT**: `_build_tensors` forces `label_type="returns"` for masked-PIT mode; rank relabeling happens later, after mask application, via `apply_rank_labels`. Non-PIT mode ranks inside `_build_tensors` directly. These are genuinely different code paths, not an accidental duplication — keep both.

**Target stage functions (signatures per exploration pass, `mci_gru/pipeline.py` reorganized internally — no new files needed for this workstream, just internal restructuring):**

```python
@dataclass
class PipelineFrames:
    """Carries the three DataFrame variants through the staged pipeline."""
    raw: pd.DataFrame        # post-engineer, pre-impute (labels, PIT masks, graph)
    normalized: pd.DataFrame  # pre-universe-filter
    filtered: pd.DataFrame    # post-universe-filter (windows/tensors)

def load_raw_data(config: ExperimentConfig) -> tuple[DataManager, pd.DataFrame]: ...
def load_auxiliary_data(data_manager, config) -> tuple[pd.DataFrame | None, ...]: ...  # existing _load_auxiliary_data, renamed public
def engineer_features(df, feature_engineer, vix_df, credit_df, regime_df) -> tuple[pd.DataFrame, list[str]]: ...

@dataclass(frozen=True)
class PitContext:
    intervals: pd.DataFrame | None
    masked_panel: bool
    csv_path: str | None

def resolve_pit_context(config: ExperimentConfig) -> PitContext: ...

@dataclass(frozen=True)
class NormFit:
    means: dict[str, float]
    stds: dict[str, float]
    rank_gauss_reference: dict[str, np.ndarray] | None

def fit_normalisation(df_filled, feature_cols, train_end, mode, pit: PitContext) -> NormFit: ...
def select_universe(df_norm, data_manager, config, pit: PitContext) -> tuple[pd.DataFrame, list[str]]: ...

@dataclass
class TensorBundle:
    train_dates: list[str]; val_dates: list[str]; test_dates: list[str]
    stock_features_train: np.ndarray  # (T_tr, S, his_t, F)
    stock_features_val: np.ndarray
    stock_features_test: np.ndarray
    x_graph_train: np.ndarray         # (T_tr, S, F)
    x_graph_val: np.ndarray
    x_graph_test: np.ndarray
    train_labels: np.ndarray          # (T_tr, S)
    val_labels: np.ndarray
    test_labels: np.ndarray

def build_tensors(frames: PipelineFrames, kdcode_list, feature_cols, dates, his_t, label_t, label_type, *, use_polars=False, fill_missing_labels=True) -> TensorBundle: ...
def apply_pit_masks_to_tensors(tensors: TensorBundle, frames: PipelineFrames, kdcode_list, pit: PitContext, his_t, label_t, label_type, min_scoreable, breadth_policy) -> tuple[TensorBundle, dict]: ...

@dataclass(frozen=True)
class GraphArtifacts:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    graph_schedule: GraphSchedule | None
    edge_index_sector: torch.Tensor | None
    edge_weight_sector: torch.Tensor | None

def build_correlation_graph(frames: PipelineFrames, kdcode_list, graph_config, train_start, test_end) -> GraphArtifacts: ...

def prepare_data(config: ExperimentConfig, feature_engineer: FeatureEngineer) -> dict[str, Any]:
    """Unchanged public signature and return dict shape — now a ~40-line
    orchestrator calling the stage functions above in sequence."""
```

`prepare_data`'s external signature, and every key in its returned dict
(`kdcode_list`, `train/val/test_dates`, `stock_features_*`, `x_graph_*`,
`*_labels`, PIT mask keys, `edge_index`, `edge_weight`, `feature_cols`,
`graph_schedule`, `df`, `norm_means`, `norm_stds`,
`graph_static_valid_from`, `edge_index_sector`, `edge_weight_sector`,
`rank_gauss_reference`, `feature_reference`, `pit_breadth`,
`pit_universe_mode`) **do not change** — this is purely an internal
decomposition. `run_experiment.py` and every test consuming
`prepare_data()`'s output needs zero changes.

**`prepare_data_index_level`** shares most stages but skips impute/PIT/
universe/sector and uses S=1/empty-graph — branch early on an `index_level`
flag inside the shared stage functions rather than maintaining a fully
parallel second implementation, where the exploration pass confirmed the
logic is genuinely shared (impute, normalize) vs genuinely different
(no PIT, no graph).

**Verification protocol (mandatory before merging, given the no-lookahead
sensitivity):**
1. Run `run_experiment.py` with a fixed seed and a small synthetic/CSV
   dataset before the refactor; save `run_metadata.json` (norm stats,
   feature list) and `averaged_predictions/*.csv`.
2. Run the identical config after the refactor.
3. Diff both artifacts — norm means/stds must match to float precision,
   predictions must match bit-for-bit (same seed, same data, same code path
   just reorganized).
4. Repeat for at least one masked-PIT config and one dynamic-graph config,
   since those are the highest-branch-count paths.
5. Full pytest suite green, especially `tests/test_pit_masked_panel.py`,
   `tests/test_dynamic_graph_updates.py`, `tests/test_phase3_graph_and_walkforward.py`,
   and any test with "no_lookahead" in its name.

**Risk:** highest in this spec. This is the only workstream where I'd
recommend a dedicated reviewer pass focused solely on the PIT/normalization
ordering before merge, independent of the automated diff above.

---

### WS-K: Point `paper_trade/scripts/infer.py` at `mci_gru/data/transforms.py`

**Depends on:** WS-I (transforms module must exist first).

**Current state:** `infer.py::prepare_inference_data()` (lines 128-237)
reimplements per-day NaN impute, 3σ clip+z-score, and single-date tensor
construction inline (~110 lines), instead of sharing code with
`pipeline.py`.

**Exact line replacements:**

| File | Current lines | Replace with |
|---|---|---|
| `paper_trade/scripts/infer.py` | 182-192 | `df = impute_feature_nans_by_day(df, feature_cols)` |
| `paper_trade/scripts/infer.py` | 194-199 | `df = normalize_features_zscore(df, feature_cols, means, stds)` |
| `paper_trade/scripts/infer.py` | 201-228 | `time_series, graph_features = build_single_date_tensors(df, kdcode_list, feature_cols, his_t, target_date)` |

**Stays local to `infer.py` (genuinely inference-specific, not shared):**
CSV load via `pd.read_csv` (not `DataManager.load()`), saved-metadata-driven
`kdcode_list`/`means`/`stds` (not computed), single-`target_date` resolution
logic, `return_observed_features` drift-CSV support, and
`prepare_inference_regime_df` (FRED/LSEG regime load with inference-horizon
clamp — has no training-time analog).

**Also fix while here:** `paper_trade/scripts/compare_regime.py::load_models()`
(lines 52-74) duplicates the checkpoint+graph load loop from
`infer.py::run_inference()` — extract a shared `load_frozen_model_and_graph(model_dir, device) -> (model, edge_index, edge_weight)` helper in `paper_trade/scripts/infer.py` (or a new `paper_trade/scripts/_shared.py`) and have `compare_regime.py` call it.

**Verification:** run `paper_trade/scripts/run_nightly.py` (or `infer.py`
standalone) against a fixed frozen checkpoint before/after; diff output
CSVs and drift artifacts for exact match. This is the AGENTS.md-invariant
path ("Paper-trade inference does not use `GraphBuilder`") — confirm the
grep-verified zero-`GraphBuilder`-import invariant still holds after the
change (it will, since `transforms.py` has no graph dependency).

**Risk:** low-medium, isolated to `paper_trade/`. Does not touch training.

---

## Sequencing Summary

```
Phase 0 (parallel, any order)          Phase 1 (parallel, after Phase 0)     Phase 2 (sequential, after Phase 1)
────────────────────────────────       ──────────────────────────────       ─────────────────────────────────
WS-A  deps/lockfile                    WS-F  models split                   WS-I  transforms.py extraction
WS-B  cockpit relocation               WS-G  graph/builder split                    ↓
WS-C  backtest relocation              WS-H  trainer/ensemble split          WS-J  pipeline.py staging
WS-D  pytest markers                                                                ↓
WS-E  docs SOT check                                                         WS-K  infer.py adopts transforms.py
```

Each phase requires: `ruff check .` clean, full `pytest tests/ -v --basetemp .tmp_pytest\pytest` green, and (Phase 2 only) the before/after artifact diff protocol in WS-J passing, before the next phase starts.

---

## Acceptance Criteria (definition of done, whole spec)

1. Every import path listed as "must keep working" in this spec still
   resolves — verified by full pytest pass, not just `ruff`.
2. No file in `mci_gru/` core package exceeds ~400 lines post-split (current
   worst offenders: `mci_gru.py` 827 → 6 files, `config.py` 929 unchanged —
   out of scope, see note below, `data_manager.py` 883 unchanged — out of
   scope, `trainer.py` 738 → 2 files, `pipeline.py` 673 → same file, reduced
   inline body, `builder.py` 450 → 3 files).
3. `tests/` contains zero non-`test_*.py` files over ~200 lines (backtest
   engines relocated to `mci_gru/evaluation/`).
4. CI's `requires_data`/`requires_lseg`/`requires_fred` marker filters
   actually exclude tests (verified via `--collect-only` diff).
5. `paper_trade/scripts/infer.py` shares normalize/impute/window logic with
   `mci_gru/pipeline.py` (zero duplicated inline implementations).
6. Single dependency manifest (`pyproject.toml`) + one lock file, zero
   drift between `requirements.txt` and `pyproject.toml` (file deleted).
7. All behavior-preserving claims backed by the WS-J artifact-diff protocol
   for at least: default config, masked-PIT config, dynamic-graph config.

**Explicitly out of scope for this spec** (noted for a future spec, not
attempted here): `config.py`'s 155-field/9-dataclass surface and
`data_manager.py`'s 883 lines (including the 248-line `load_regime_inputs`)
are flagged in the review as sprawl but are **not** included in this
rearchitecture pass — they don't have the same "one god-function doing 14
things" shape as `pipeline.py`; they're wide-but-flat (many small validated
fields / many small loader methods), which is a different problem
(arguably acceptable sprawl for a research config surface) than the
depth-first coupling this spec targets. Revisit separately if config
sprawl becomes a velocity problem.

---

## Appendix: `mci_gru/config.py` Field Inventory (ground truth, unchanged by this spec)

155 fields across 9 dataclasses, provided here so any future config-focused
spec starts from verified counts rather than re-deriving them.

| Dataclass | Lines | Field count |
|---|---|---|
| `DataConfig` | 14-97 | 18 |
| `FeatureConfig` | 100-268 | 44 |
| `GraphConfig` | 271-343 | 13 |
| `ModelConfig` | 346-455 | 23 |
| `WalkforwardConfig` | 458-480 | 7 |
| `TrainingConfig` | 483-619 | 25 |
| `EvaluationConfig` | 622-652 | 8 |
| `TrackingConfig` | 655-676 | 7 |
| `ExperimentConfig` | 679-837 | 10 (7 sub-configs + `experiment_name`/`output_dir`/`seed`) |

Every class except `TrackingConfig` has `__post_init__` validation. Cross-
config validation (`_validate_embargo`, lines 714-742) lives only on
`ExperimentConfig`, correctly centralized as the one genuinely cross-cutting
check.
