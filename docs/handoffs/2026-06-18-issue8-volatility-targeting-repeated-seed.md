# Issue 8 Volatility Targeting Repeated Seed Handoff

Last updated: 2026-06-18

## Resume Here

- Start from branch `codex/issue8-vol-repeated-seed`, commit `9b5d201`.
- Open the branch-backed Colab notebook:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/issue8-vol-repeated-seed/notebooks/volatility_targeting_ablation_colab.ipynb`
- Current blocker: setup cell failed before training at `drive.mount("/content/drive")` with `ValueError: mount failed`.
- The failed G4 runtime was manually disconnected/deleted; the visible state after cleanup was `Reconnect G4 High-RAM Click to connect`.
- Next useful fix is notebook Drive staging resilience before relaunching the expensive repeated-seed sweep.

## Current Objective

Continue Issue #8 volatility-targeting diagnosis by running the documented component/clip ablations as a repeated-seed panel. Compare every candidate against same-year, same-seed `baseline_vol`, not a pooled baseline.

## What Changed

- Updated `scripts/gen_volatility_targeting_ablation_nb.py` and regenerated `notebooks/volatility_targeting_ablation_colab.ipynb`.
- Added default `RUN_STAGE = "repeated_seed_validation"`.
- Added narrow repeated-seed matrix:
  - years `[2022, 2023, 2024, 2025]`
  - seeds `[314159, 271828, 161803]`
  - variants `baseline_vol`, `vt_full_clip_0p25_4p0`, `vt_clip_0p50_2p0`, `vt_scale_only`, `vt_ewm_only`
- Replaced removed boolean overrides with canonical `features.volatility_targeting_components=[...]`.
- Changed delta logic to group by `["year", "seed"]`.
- Added repeated-seed summary CSV outputs by variant and by year/variant.
- Updated `tests/test_volatility_targeting_ablation_notebook.py` to lock the new contract.

## Key Decisions

- Use `volatility_targeting_components`, not old booleans such as `volatility_targeting_include_scale`.
- Publish a scoped branch because Colab GitHub URLs read remote code; the older `codex/pit-universe-validation` ref does not support the component-list API.
- Do not run on T4/CPU. The Colab runtime was set to G4 High-RAM before setup was attempted.
- Treat the temporary Drive upload `issue8_volatility_targeting_repeated_seed_validation_20260618.ipynb` / file id `1guG5FBvIRhtC6oMFj2Heal_TkTyNgoRj` as stale: it was uploaded before the notebook branch constant was changed. Prefer the GitHub-backed URL above.

## Important Files

- `scripts/gen_volatility_targeting_ablation_nb.py`: source of truth for notebook edits.
- `notebooks/volatility_targeting_ablation_colab.ipynb`: generated Colab notebook.
- `tests/test_volatility_targeting_ablation_notebook.py`: notebook contract test.
- `tests/test_volatility_targeting_features.py`: validates component-list feature/config behavior.
- `docs/workflows/COLAB_PLAYWRIGHT_MCP_GUIDE.md`: Colab control-plane and cleanup workflow.
- `docs/NOTEBOOK_BEST_PRACTICES.md`: Drive staging and output conventions.
- `docs/handoffs/2026-05-28-issue8-volatility-targeting-ablation.md`: prior Issue #8 experiment state and single-seed evidence.
- `docs/research/current/ISSUE8_VOL_TARGETING_CAUSE_ANALYSIS_2026-05-27.md`: causal hypothesis and next tests.

## Verification

- `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests\test_volatility_targeting_ablation_notebook.py tests\test_volatility_targeting_features.py -q`
  - Result: `10 passed`.
- `C:\Users\magil\MCI-GRU\.venv\Scripts\ruff.exe check scripts\gen_volatility_targeting_ablation_nb.py tests\test_volatility_targeting_ablation_notebook.py`
  - Result: `All checks passed!`
- Hydra parse smoke:
  - Parsed `features.volatility_targeting_components=[scale]` and `features.volatility_target_scale_clip=[0.50,2.0]`.
  - Result printed `['scale'] [0.5, 2.0]`.
- Git:
  - Branch created and pushed: `codex/issue8-vol-repeated-seed`.
  - Commit: `9b5d201 Add Issue 8 repeated-seed ablation notebook`.
  - `git status --short --branch` after push showed branch tracking `origin/codex/issue8-vol-repeated-seed`; no tracked modifications before this handoff file was added.
- Colab manual evidence:
  - Chrome opened the GitHub-backed notebook while signed in as the user.
  - Runtime type was changed from T4 to `G4 GPU`; toolbar showed `Connect G4 High-RAM`.
  - Running setup triggered expected GitHub `Run anyway` warning; accepted it.
  - Setup connected to a GPU backend and failed at Drive mount: `ValueError: mount failed`.
  - Runtime was manually disconnected/deleted; toolbar showed `Reconnect G4 High-RAM`.

## Open Risks

- No experiment/backtest rows completed in Colab during this turn.
- The notebook still depends on DriveFS mount for input data and output paths; this failed before clone/install reached completion.
- The current notebook writes full repeated-seed training as 60 full-budget jobs. Confirm runtime budget before relaunching.
- FRED secret/data cell was not reached; `FRED_API_KEY` and data file availability remain unverified for this run.
- The handoff file itself is currently an uncommitted local note unless the next agent stages/commits it.

## Next Actions

1. Patch the notebook generator to handle Drive mount failure before relaunch:
   - minimally try `drive.mount("/content/drive", force_remount=True, timeout_ms=120000)`;
   - preferably add a Drive API fallback or clear failure message following `docs/NOTEBOOK_BEST_PRACTICES.md`.
2. Regenerate `notebooks/volatility_targeting_ablation_colab.ipynb`, run the same focused tests and ruff, commit, and push to `codex/issue8-vol-repeated-seed`.
3. Reopen the GitHub-backed Colab URL, set G4/L4-class runtime, run setup only, and confirm:
   - non-T4 `nvidia-smi` output,
   - cloned branch `codex/issue8-vol-repeated-seed`,
   - no DriveFS failure.
4. Run the FRED/PIT input cell and matrix build cell; record the run root under `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/<RUN_TAG>`.
5. Launch the training/backtest cell only after setup/data preflight passes; monitor Drive artifacts (`issue8_vol_targeting_ablation_results.csv`, deltas CSV, seed summary CSVs) as the source of truth.

## Data/Experiment State

- Intended Drive run root pattern:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/<RUN_TAG>`
- Intended result artifacts:
  - `issue8_vol_targeting_ablation_manifest.json`
  - `issue8_vol_targeting_ablation_results.csv`
  - `issue8_vol_targeting_ablation_deltas_vs_baseline.csv`
  - `issue8_vol_targeting_repeated_seed_summary_by_variant.csv`
  - `issue8_vol_targeting_repeated_seed_summary_by_year_variant.csv`
  - `issue8_vol_targeting_ablation_summary.md`

## Do Not Do

- Do not compare candidates to pooled baselines; use same-year, same-seed `baseline_vol`.
- Do not run the experiment on T4 or CPU.
- Do not restore old per-component boolean config knobs.
- Do not treat the stale Drive-uploaded notebook as the runnable source.
- Do not leave a failed setup runtime connected; cleanup is manual if the final cell never runs.
