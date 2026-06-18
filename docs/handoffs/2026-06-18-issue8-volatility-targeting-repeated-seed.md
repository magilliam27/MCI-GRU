# Issue 8 Volatility Targeting Repeated Seed Handoff

Last updated: 2026-06-18

## Resume Here

- Start from branch `codex/issue8-vol-repeated-seed`, latest pushed handoff commit `3643e0e`.
- Open the branch-backed Colab notebook:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/issue8-vol-repeated-seed/notebooks/volatility_targeting_ablation_colab.ipynb`
- Current Colab run is live in Chrome on the branch-backed notebook.
- Setup succeeded after granting the standard Drive OAuth prompt:
  - Drive mounted at `/content/drive`.
  - GPU printed `NVIDIA RTX PRO 6000 Blackwell Server Edition` from a `G4 GPU` runtime.
  - Repo printed `/content/MCI-GRU`.
  - Branch printed `codex/issue8-vol-repeated-seed`.
- FRED/PIT preflight succeeded after granting Colab Secrets access for `FRED_API_KEY`.
- Build cell created Drive run root:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/20260618_183044`
- Training/backtest cell was launched at about 2:32 PM ET on 2026-06-18 and started:
  `Training: issue8_ablate_baseline_vol_2022_seed314159`

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
  - Handoff commit: `3643e0e Document Issue 8 repeated-seed Colab blocker`.
- Colab manual evidence, first failed attempt:
  - Chrome opened the GitHub-backed notebook while signed in as the user.
  - Runtime type was changed from T4 to `G4 GPU`; toolbar showed `Connect G4 High-RAM`.
  - Running setup triggered expected GitHub `Run anyway` warning; accepted it.
  - Setup connected to a GPU backend and failed at Drive mount: `ValueError: mount failed`.
  - Runtime was manually disconnected/deleted; toolbar showed `Reconnect G4 High-RAM`.
- Colab manual evidence, retry:
  - In-app browser was on stale Drive notebook file id `1guG5FBvIRhtC6oMFj2Heal_TkTyNgoRj`, whose setup cell still had `BRANCH = "codex/pit-universe-validation"`.
  - Navigated to the branch-backed URL because it correctly had `BRANCH = "codex/issue8-vol-repeated-seed"`.
  - In-app runtime picker still showed `T4 GPU` and would not actuate the G4 radio through the browser helper; Chrome fallback was used with the visible signed-in Colab UI.
  - Chrome runtime dialog showed `G4 GPU` selected.
  - Setup mounted Drive and printed GPU/repo/branch evidence above.
  - FRED/PIT cell printed:
    - `FRED_API_KEY loaded from Colab Secrets.`
    - market CSV `/content/MCI-GRU/data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv` hash prefix `84e1f3f2b79a7982`.
    - PIT CSV `/content/MCI-GRU/data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv` hash prefix `a9ef692b83a9575c`.
  - Build cell printed manifest path:
    `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/20260618_183044/issue8_vol_targeting_ablation_manifest.json`.
  - Training/backtest cell is running; no return code had printed at this handoff update.

## Open Risks

- No experiment/backtest rows had completed at this handoff update.
- The current notebook writes full repeated-seed training as 60 full-budget jobs; expect a long run unless prior artifacts are reused.
- The training cell captures each subprocess output and prints `Training return code` only after a job completes, so the visible notebook may be quiet during long training.
- The in-app browser tab still points to the stale Drive upload unless manually redirected; use the branch-backed URL for source truth.
- This handoff update is a local working-tree edit until staged/committed/pushed.

## Next Actions

1. Keep the Chrome Colab tab alive and monitor the running training/backtest cell.
2. Source-of-truth run root:
   `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/20260618_183044`
3. Check for first completed rows in:
   - `issue8_vol_targeting_ablation_results.csv`
   - `issue8_vol_targeting_ablation_deltas_vs_baseline.csv`
   - `issue8_vol_targeting_repeated_seed_summary_by_variant.csv`
   - `issue8_vol_targeting_repeated_seed_summary_by_year_variant.csv`
4. If the cell fails, inspect the logged stdout/stderr files under:
   `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/20260618_183044/logs/`
5. Once enough rows exist, compare each candidate against same-year, same-seed `baseline_vol`.

## Data/Experiment State

- Intended Drive run root pattern:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/<RUN_TAG>`
- Active Drive run root:
  `/content/drive/MyDrive/MCI-GRU-Ablations/volatility_targeting_issue8_ablation/20260618_183044`
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
