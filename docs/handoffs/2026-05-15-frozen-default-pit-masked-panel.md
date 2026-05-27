# Frozen Default PIT Masked-Panel Handoff

Last updated: 2026-05-15

## Resume Here

- Branch `codex/pit-universe-validation` is pushed and in sync with
  `origin/codex/pit-universe-validation` at commit
  `7644eeb Use frozen default recipe for PIT masked-panel notebook`.
- The immediate next action is to run the updated
  `notebooks/pit_masked_panel_2022_2025_colab.ipynb` full test from the pushed
  branch, after confirming `FRED_API_KEY` is available in Colab.
- Start continuation by reading `docs/DEFAULT_EXPERIMENT_RECIPE.md` and the
  config cells generated from `scripts/gen_pit_masked_panel_2022_2025_nb.py`.
- The working tree still has unrelated dirty files from other work; do not
  stage, revert, or summarize them as part of this change without checking with
  the user.

## Current Objective

Move from the successful PIT masked-panel smoke validation to a full
2022-2025 PIT validation run using the newer frozen default recipe from the
previous ablation/proof notebooks.

## What Changed

- Added `docs/DEFAULT_EXPERIMENT_RECIPE.md` as the canonical default recipe
  reference for production-style confirmation notebooks and PIT validation runs.
- Updated repo guidance so new agents and users see the frozen recipe first:
  `AGENTS.md`, `README.md`, `docs/index.md`,
  `docs/CONFIGURATION_GUIDE.md`, `docs/ARCHITECTURE.md`,
  `docs/NOTEBOOK_BEST_PRACTICES.md`,
  `docs/MODERN_DEFAULTS_HANDOFF_2026.md`, and
  `docs/FULL_FEATURE_FACTORIAL_ABLATION.md`.
- Updated `scripts/gen_pit_masked_panel_2022_2025_nb.py` and regenerated
  `notebooks/pit_masked_panel_2022_2025_colab.ipynb`.
- Added `tests/test_pit_masked_panel_notebook.py` to lock the notebook and
  generator to the frozen recipe tokens and verify notebook code cells parse.

## Key Decisions

- Canonical recipe slug:
  `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`.
- Full-run defaults now use `SMOKE_MODE=False`, `USE_GLOBAL_REGIME=True`,
  `NUM_MODELS=20`, `NUM_EPOCHS=100`, `EARLY_STOPPING_PATIENCE=15`,
  `BATCH_SIZE=32`, `learning_rate=5e-5`, and `seed=1729`.
- Modeling defaults are pure IC loss, raw 5-day return labels,
  `selection_metric=val_ic`, shuffled static threshold graph,
  multi-feature edges, and `graph.drop_edge_p=0.1`.
- Regime defaults are strict current-only global regime features:
  `features.include_global_regime=true`,
  `features.regime_include_subsequent_returns=false`, and FRED-backed loading.
  A full run requires `FRED_API_KEY`.
- The previous Colab smoke artifact is only a mechanics check. Its one-epoch
  backtest results are not evidence of model quality.

## Important Files

- `docs/DEFAULT_EXPERIMENT_RECIPE.md`: canonical frozen recipe.
- `AGENTS.md`: Codex-first table of contents now points agents to the frozen
  recipe.
- `scripts/gen_pit_masked_panel_2022_2025_nb.py`: source of truth for the
  Colab notebook.
- `notebooks/pit_masked_panel_2022_2025_colab.ipynb`: user-facing notebook to
  run next.
- `tests/test_pit_masked_panel_notebook.py`: regression coverage for recipe
  tokens and notebook parseability.
- `docs/handoffs/2026-05-14-pit-masked-panel-colab.md`: previous handoff with
  the smoke-run context before the frozen-default update.

## Verification

- `git status --short --branch`
  - Branch is `codex/pit-universe-validation...origin/codex/pit-universe-validation`.
  - Only unrelated dirty files remain outside the pushed commit.
- `git show --stat --oneline --decorate --no-renames HEAD`
  - Verified latest commit is `7644eeb` and contains the 12 intended files.
- `.venv\Scripts\python.exe -m pytest tests\test_pit_masked_panel_notebook.py -v -p no:cacheprovider`
  - Result: `2 passed in 0.07s`.
- `.venv\Scripts\ruff.exe check scripts\gen_pit_masked_panel_2022_2025_nb.py tests\test_pit_masked_panel_notebook.py`
  - Result: `All checks passed!`.
- `git diff --check HEAD~1..HEAD -- <intended files>`
  - Result: exit code `0`, no output.

## Data/Experiment State

- Reviewed Drive smoke folder:
  `MCI-GRU-Ablations/pit_masked_panel_2022_2025/20260514_033256`.
- Smoke run state:
  - `SMOKE_MODE=True`
  - `USE_GLOBAL_REGIME=False`
  - FRED not set
  - Training completed for 2022, 2023, 2024, and 2025 in `masked_panel` mode.
- Smoke breadth checks were clean:
  - `scoreable_count.below_threshold=0` across train/val/test split rows.
  - Prediction files matched expected counts:
    2022 `237/237`, 2023 `237/237`, 2024 `239/239`, 2025 `238/238`.
  - Missing predictions and mismatches were `0`.
- Known non-blocking smoke warning:
  PyTorch warned about a non-writable NumPy tensor from
  `mci_gru/data/data_manager.py` around the `sliding_window_view` path. No
  downstream mutation issue was observed in the smoke run.

## Open Risks

- The full frozen-default Colab run has not been executed yet after this
  notebook update.
- Full default mode will fail early if `USE_GLOBAL_REGIME=True`,
  `REGIME_STRICT=True`, and `FRED_API_KEY` is missing.
- Full-run runtime and Drive storage pressure are still unmeasured for the
  20-model, 100-epoch, 2022-2025 run.
- Current working tree includes unrelated local changes:
  `docs/REGIME_DATA_CONTRACT.md`, `scripts/colab_regime_reconcile.py`,
  `skills/research-paper-to-mci-gru/*`, `tests/test_regime_features.py`,
  `.codex_tmp/`, PIT data metadata sidecars, older handoffs, and
  `docs/research-paper-evaluations/`.

## Next Actions

1. In Colab, pull or reclone branch `codex/pit-universe-validation` and confirm
   the notebook shows commit `7644eeb` or newer.
2. Add `FRED_API_KEY` as a Colab Secret before running the full defaults.
3. Run `notebooks/pit_masked_panel_2022_2025_colab.ipynb` with defaults:
   `SMOKE_MODE=False` and `USE_GLOBAL_REGIME=True`.
4. Inspect the new Drive summary artifacts first:
   `pit_masked_panel_2022_2025_summary.md`,
   `pit_breadth_summary.csv`, `prediction_count_checks.csv`, and
   `backtest_results.csv`.
5. If breadth and prediction checks stay clean, compare performance against the
   previous ablation/proof baselines and decide whether to open or update a PR.

## Commands Run

- `Get-Date -Format yyyy-MM-dd`
- `git status --short --branch`
- `git log --oneline -3`
- `git show --stat --oneline --decorate --no-renames HEAD`
- `git show --name-only --format=short --no-renames HEAD`
- `rg -n "static-threshold-shuffle|MODEL_RECIPE|NUM_MODELS|NUM_EPOCHS|EARLY_STOPPING_PATIENCE|FRED_API_KEY|drop_edge_p|selection_metric" scripts\gen_pit_masked_panel_2022_2025_nb.py docs\DEFAULT_EXPERIMENT_RECIPE.md tests\test_pit_masked_panel_notebook.py`
- `.venv\Scripts\python.exe -m pytest tests\test_pit_masked_panel_notebook.py -v -p no:cacheprovider`
- `.venv\Scripts\ruff.exe check scripts\gen_pit_masked_panel_2022_2025_nb.py tests\test_pit_masked_panel_notebook.py`
- `git diff --check HEAD~1..HEAD -- <intended files>`

## Do Not Do

- Do not revert masked-panel behavior to complete-stock filtering or
  continuous-member filtering.
- Do not treat `results/`, `outputs/`, `*.pth`, or `*.pt` as source of truth.
- Do not stage or push unrelated dirty files without explicit user direction.
- Do not quote the one-epoch smoke backtest as model-performance evidence.

## References

- Pushed commit:
  `7644eeb Use frozen default recipe for PIT masked-panel notebook`.
- Previous smoke handoff:
  `docs/handoffs/2026-05-14-pit-masked-panel-colab.md`.
- Smoke Drive summary file:
  `https://drive.google.com/file/d/1a30yWKjlc4nTLaa7LzSxm7gAiDlkjhAo`.
- Smoke prediction count checks:
  `https://drive.google.com/file/d/1kKpyXdlNnaNqyMpn3rzwT0BV5umRXdmy/view?usp=drivesdk`.
