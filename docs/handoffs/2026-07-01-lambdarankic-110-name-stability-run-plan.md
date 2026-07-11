# LambdaRankIC 110-Name Stability Run Plan Handoff

Last updated: 2026-07-01

## Resume Here

- Start with a Drive/artifact inventory, not training: recover the missing 110-name `averaged_predictions` for LambdaRankIC base seeds `161803` and `271828` across 2022, 2023, 2024, and 2025, and place or link them under one canonical campaign root.
- Current decision point: 2024 is the stress year. LambdaRankIC seed `161803` produced 40.22% net return with 120 trades and 1.20% cumulative cost; seed `271828` produced 15.88% net return with 362 trades and 3.62% cumulative cost. The weak seed is not explained by costs alone because gross return before costs was only about 20.14%.
- Immediate next move: run only no-retraining diagnostics on existing or regenerated saved predictions, then decide whether any additional Colab training matrix is justified. Any Colab saved-prediction, backtest, or training run requires explicit user approval.

## Current Objective

Plan the minimum additional 110-name PIT GICS top-10-per-sector LambdaRankIC stability campaign needed to verify base-seed stability, rank stability, and the 2024 churn issue before LambdaRankIC can be considered for default status.

Pure IC remains the current/default recipe. LambdaRankIC remains experimental unless robust saved-prediction and repeated-base-seed evidence justifies promotion or a narrower hybrid role.

## What Changed

- Added and updated the planning handoff: `docs/handoffs/2026-07-01-lambdarankic-110-name-stability-run-plan.md`.
- Added `scripts/colab_recovery_upload_filter.py`, a reusable upload allowlist for Colab saved-prediction recovery runs.
- Added `scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py`, the exact Colab launcher used for the safe relaunch attempt. It is G4-only, Drive API based, skips recovered rows, disables backtests, and uploads only averaged predictions plus provenance by default.
- Added `tests/test_colab_recovery_upload_filter.py`, a focused regression test proving per-model CSVs and broad recursive CSV matches are excluded by default.
- Added `docs/handoffs/2026-07-02-lambdarankic-g4-recovery-relaunch-patch.md`, a pasteable Colab patch/runbook for the safe relaunch.
- No new training, backtests, GitHub changes, pushes, or expensive local jobs were launched. A patched Colab launcher cell was attempted after explicit user approval, but it stopped at Google/Drive auth before Drive heartbeat or any new training row.
- The requested local handoffs `docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md` and `docs/handoffs/2026-07-01-lambdarankic-stability-diagnostics-coordination.md` were not present in this checkout; this note uses the prompt's current evidence summary plus local docs/code/memory evidence.

## Key Decisions

- Treat base seeds and ensemble member seeds as separate axes. A 20-model ensemble is one base-seed run with model seeds `config.seed + model_id`; it is not 20 independent repeated base seeds.
- Prioritize saved-prediction recovery or regeneration before more training. The current goal is to explain the 2024 split between base seeds, not to expand the random grid reflexively.
- Keep the comparable universe fixed: true PIT masked panel, 110 active names from top 10 per GICS sector, 5-day raw return labels, and no-lookahead train/validation/test splits.
- Lock pair cap from the run manifest before comparing seeds. Do not compare `pair_cap=4096` against `pair_cap=8192` as if it were only a seed difference.
- Ignore older pair-cap-1024 same-seed CSVs from the broader PIT universe for this decision. They are not the same 110-name top-10 loss-matrix universe.
- Use 2024 as the stress year for rank-stability and pair-cap diagnostics before multiplying all-year training.

## Important Files

- `AGENTS.md`: repo invariants; pure IC frozen recipe remains default, PIT masked panel breadth must not be collapsed.
- `docs/ARCHITECTURE.md`: data flow, masked-panel behavior, ensemble averaging, and `averaged_predictions/` contract.
- `docs/DEFAULT_EXPERIMENT_RECIPE.md`: production-style default recipe; LambdaRankIC must be an explicit experimental override.
- `docs/research/README.md`: handoffs are operational continuity notes, not research evidence by themselves.
- `docs/handoffs/2026-06-22-sp500-gics-top10-multiyear-baseline.md`: reduced 110-name PIT top-10 baseline context and selector-history caveats.
- `scripts/gen_lambdarank_ic_pit_nb.py`: current LambdaRankIC notebook generator, seed/pair-cap constants, heartbeat, manifest, and training result contract.
- `scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py`: Colab-only relaunch script pasted into the visible notebook as a base64 launcher cell. Do not run locally.
- `scripts/colab_recovery_upload_filter.py`: incident guard for future recovery cells; uploads `averaged_predictions/**` and provenance metadata by default while excluding `predictions_model_*/*.csv`.
- `scripts/run_pit_saved_prediction_backtests.py`: saved-prediction replay wrapper for cost-aware rank-drop-gated backtests.
- `tests/backtest_sp500_daily.py`: top-k daily backtest, transaction-cost model, rank-drop gate, trade journal, daily holdings, and metrics artifacts.
- `tests/test_colab_recovery_upload_filter.py`: regression coverage for the upload allowlist.
- `docs/handoffs/2026-07-02-lambdarankic-g4-recovery-relaunch-patch.md`: self-contained recovery-cell patch and resume-matrix guard.
- `mci_gru/training/trainer.py`: confirms `config.seed` is the base seed and ensemble members use `config.seed + model_id`.
- `mci_gru/config.py` and `mci_gru/training/losses.py`: LambdaRankIC config and loss surfaces.

## Data/Experiment State

- Completed recovered matrix: 28/28 backtest rows across pure IC, LambdaRankIC, and Portfolio-IC weight50 for the 110-name PIT top-10 loss matrix.
- Current repeated-base-seed 110-name LambdaRankIC training coverage from the active investigation:
  - `161803`: years 2022, 2023, 2024, 2025.
  - `271828`: years 2022, 2023, 2024, 2025.
  - `314159`: older 110-name top-10 LambdaRankIC seed exists for 2022, 2023, 2024, 2025.
- Visible current repeated-seed CSV predictions for `161803` and `271828` were not found in Drive, even though Colab logs showed they existed and recovered backtests loaded them.
- Older 110-name top-10 LambdaRankIC saved-prediction CSV folders were found only for seed `314159`.
- Current recovered backtests used `min_rank_drop=30` and transaction costs of 10 bps bid/ask round-trip plus 5 bps slippage per trade.

## Live Drive Inventory

Drive search approved by the user and performed on 2026-07-01. No Drive files were mutated.

- Current repeated-seed campaign root found:
  - `sp500_gics_top10_loss_comparison_repeated_seeds`: `https://drive.google.com/drive/folders/1lhL-tnUoShh8ImNdTED_sRBOf_dqcOim`
  - Completed run `20260629_011839`: `https://drive.google.com/drive/folders/1Co5Vd2dOSMrHUN5x_OzbJpkjJFocSHMo`
- Completed run `20260629_011839` contains `heartbeat.json`, `manifest.json`, `run_summary.json`, `logs/`, `summaries/`, and `artifacts/`. Manifest/heartbeat evidence shows L4 runtime, 20 models, 100 epochs, patience 15, frozen recipe, years 2022-2025, and `lambdarank_pair_cap=8192`.
- Current LambdaRankIC `summaries/completed_rows/lambdarank_ic/<year>/seed*.json` files exist for seeds `161803` and `271828` across 2022, 2023, 2024, and 2025. Exact training and backtest logs also exist for all eight rows.
- Those current row JSONs record `predictions_dir` under `/content/mci_gru_runs/.../averaged_predictions`, i.e. Colab runtime-local paths, not Drive paths.
- Exact Drive searches for `top10_lambdarank_ic_<year>_seed<seed>` for seeds `161803` and `271828` across all four years returned only training/backtest logs, not run-name folders or `averaged_predictions` folders.
- The current run's `artifacts/local_run_root` copy exists, but its copied `training/` subtree listed only `portfolio_ic_weight50`; no LambdaRankIC training subtree or prediction folders were found there.
- User-supplied folder `https://drive.google.com/drive/folders/1hEh23PUtfD2104gxuVF_fKDJshYzFg8v` is a real 2025 `averaged_predictions` CSV folder, but its parent path resolves to `top10_portfolio_ic_weight50_2025_seed161803/20260630_090258/averaged_predictions`. Treat it as Portfolio-IC weight50 evidence, not one of the missing LambdaRankIC prediction folders.
- After the user approved rerunning the missing predictions on 2026-07-01, the exact prior Colab notebook was found at `https://colab.research.google.com/drive/1uRIERr_OFE8HC79FV6GHhR1isZ3AjFIt` (`sp500_top10_loss_seed_matrix_resume_20260629_011839.ipynb`). Its saved metadata and prior heartbeat are L4, but the live Colab UI was changed to `G4 GPU` before any recovery run was launched. Do not run that notebook unchanged for recovery: it skips rows when `summaries/completed_rows/lambdarank_ic/<year>/seed*.json` already exists, which is true for the eight missing-prediction rows.
- Live recovery launched on 2026-07-01 after explicit user approval:
  - Run family: `lambdarankic_110_name_missing_predictions_g4`
  - Run tag: `20260701_185554`
  - Drive run root: `https://drive.google.com/drive/folders/1mO5dqZ6QMIRMDQHmrbd30so2ui7HeT5V`
  - Heartbeat: `https://drive.google.com/file/d/1x1L1IGLFqA6-oxU2Jnn9xtnkI6K5vK2R/view`
  - Runtime proof from Colab cell output: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, i.e. G4-class; L4/T4 are blocked by the in-cell `nvidia-smi` gate.
  - Scope: only the missing LambdaRankIC rows for seeds `271828` and `161803` across 2022-2025, `pair_cap=8192`, no backtests, no new seeds. DriveFS mount failed twice, so the live recovery uses Google Drive API upload/download rather than `drive.mount`.
- Live recovery incident investigated on 2026-07-02 after the user reported a hang/burning compute:
  - Immediate mitigation complete: the connected Colab GPU runtime was manually disconnected and deleted. The toolbar changed from a connected Python GPU backend to `Reconnect G4 High-RAM`.
  - Latest heartbeat before disconnect still said `RUNNING`, phase `train_03_2023_seed271828`, `returncode=0`, updated `2026-07-02T01:25:25Z`.
  - Training did not hang. The third-row log ends with `Experiment Complete`, saved `averaged_predictions`, `training_summary.json`, `evaluation_summary.json`, and `timing_summary.json` at `2026-07-02T01:25:22Z`.
  - The hang/waste was the post-training Drive API upload loop. The notebook upload filter allowed every `.csv` anywhere under the run directory, so it uploaded every `predictions_model_*/*.csv` file, not just `averaged_predictions/*.csv`.
  - `summaries/training_rows.json` has only two manifest-complete rows: 2022 seed `271828` and 2022 seed `161803`. Each row records `prediction_csv_count=246` but `uploaded_file_count=5177`, confirming the per-model CSV explosion.
  - Row 3, 2023 seed `271828`, is not in `training_rows.json`, but it is likely salvageable: Drive run folder `https://drive.google.com/drive/folders/1Xtli0wHhLEeLxtVuaRJqSf-9DlZ4_DCy`, `averaged_predictions` folder `https://drive.google.com/drive/folders/1i2eiUSi0CGpatkzZ64hqAz04v71ob6FV`, visible CSV range `2023-01-09.csv` through `2023-12-29.csv`.
  - The same row's `evaluation_summary.json` exists at `https://drive.google.com/file/d/14z1TZjQLgYedMXn3AOjBZdFsPGSaYZS4/view`; top-line row metrics include `avg_rank_ic=0.04027808165325881`, `avg_ic=0.032401191248697225`, and `return_top_10=0.002066922028983739`.
  - The runtime was stopped mid-upload after `predictions_model_16` appeared. Do not treat per-model folder completeness as a blocker for saved-prediction diagnostics unless ensemble-dispersion diagnostics are explicitly approved.
  - No fourth training log exists in the run `logs/` folder, so there is no evidence that 2023 seed `161803` or any 2024/2025 row started.
- Safe relaunch attempted on 2026-07-02 after the user explicitly approved launching again if the issue was fixed:
  - A new patched launcher cell was inserted into the visible Colab notebook as Cell 2. The local source passed `py_compile` before paste and the in-notebook G4 gate printed `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition`.
  - The launcher blocks `L4`/`T4`, sets `RUN_BACKTESTS = False`, filters out recovered 2022 rows, checks the 2023 seed `271828` Drive CSV count before skipping it, and uses the narrow upload allowlist.
  - The run did not reach Drive heartbeat or training. It stopped in `google.colab.auth.authenticate_user()` after Colab's in-notebook credentials prompt; the external `accounts.google.com` OAuth tab was blocked by browser policy, so the agent could not complete account authorization.
  - The current execution was interrupted before any new row training started, then `Runtime > Disconnect and delete runtime` was confirmed. The toolbar returned to `Reconnect G4 High-RAM`, so compute was stopped.
  - Resume path: the user must complete the Google/Drive OAuth prompt manually in the visible browser session, then rerun only the patched Cell 2 launcher. Do not rerun the older broad-upload recovery cell.
- Earlier sibling repeated-seed roots `20260629_005700` and `20260629_005322` contain only heartbeat/logs/summaries at the root level and no `artifacts/` folder.
- Older seed `314159` saved-prediction controls are Drive-visible:
  - 2022-2024 root `sp500_gics_top10_lambdarank_ic_full/20260626_172316`: `https://drive.google.com/drive/folders/1xAovLZMDFgGcVCOVxmqpYApk1YBWaJc7`
  - 2022 `averaged_predictions`: `https://drive.google.com/drive/folders/1IJ62jNdpLbFW4Kuc9l68LkG3NmTmS9bd`
  - 2023 `averaged_predictions`: `https://drive.google.com/drive/folders/1tp-BEvU2yPMxnE_c6ul3o1gVi-ZyeF3n`
  - 2024 `averaged_predictions`: `https://drive.google.com/drive/folders/1KHK3TSjtjz4Ft-XTU5DkmVtgcklgKTwt`
  - 2025 run `20260627_012707`: `https://drive.google.com/drive/folders/1myeVj1kbiVEOxt0CLTQf_-9-u3enc4wE`
  - 2025 `averaged_predictions`: `https://drive.google.com/drive/folders/1PIV6uuwKDBKAGYMIRsvepCD7cgo9CgO3`
- Broader-universe `lambdarank_ic_pairs1024_*` folders exist for seeds `161803` and `271828`, but they remain outside the comparable 110-name top-10 loss-matrix universe.

Interpretation:

- The current `161803`/`271828` evidence is sufficient to confirm training/backtest completion and runtime config, but not sufficient for cross-seed rank-stability diagnostics that require same-date CSV prediction surfaces.
- At launch, the 12-row known-seed matrix was four Drive-visible `314159` prediction folders plus eight missing current prediction folders for seeds `161803` and `271828`. After the incident run, the current visible recovery state is two manifest-complete 2022 rows plus one likely salvageable 2023 seed `271828` averaged-prediction folder.
- Before training, make one last recovery/provenance check from Drive-visible folders, any checkpoint mirror, or any unlisted artifact copy. If those are gone, the minimum approved runtime work becomes targeted regeneration/retraining of only unrecovered current LambdaRankIC seed-year prediction surfaces, not expansion to new exploratory seeds.
- After the 2026-07-02 incident, do not relaunch the same recovery cell unchanged. First patch the upload contract so it uploads only `averaged_predictions/**` plus root-level metadata/log summaries by default. Treat 2022 seeds `271828` and `161803` as recovered. Treat 2023 seed `271828` as salvageable pending a cheap Drive CSV count/provenance check. The remaining likely training rows are 2023 seed `161803`, 2024 seeds `271828` and `161803`, and 2025 seeds `271828` and `161803`.

## No-Retraining Diagnostics First

Run these only after explicit user approval if they require Colab, Drive mutation, or backtest execution.

1. Artifact inventory and recovery:
   - Build a `saved_prediction_inventory.csv` with one row per `(year, base_seed, pair_cap, run_tag)`.
   - Columns: `year`, `base_seed`, `pair_cap`, `loss_type`, `selection_metric`, `num_models`, `train_start`, `val_start`, `test_start`, `training_run_dir`, `checkpoint_dir`, `averaged_predictions_dir`, `prediction_csv_count`, `first_prediction_date`, `last_prediction_date`, `backtest_dir`, `status`, `notes`.
   - If `averaged_predictions` exist, copy or link them into the canonical campaign root.
   - If checkpoints/training outputs exist but CSVs are missing, regenerate `averaged_predictions` from the existing checkpoints before any retraining. This is still approval-required if done in Colab.

2. Rank and boundary stability:
   - For each year and seed pair, compute same-date cross-seed Spearman and Kendall rank correlation over the PIT-active tradable names.
   - Compute Top-10, Top-20, and Top-30 Jaccard agreement by date.
   - Compute top-10 boundary churn: ranks 8-12 membership, #10/#11 score margin, daily additions/removals, names crossing the boundary, and next-day return contribution of boundary swaps.
   - For 2024, compare `161803` vs `271828` on dates where `271828` churns but `161803` holds stable.

3. Rank-drop gate and cost sensitivity:
   - Baseline: gate `min_rank_drop=30`, spread `10`, slippage `5`.
   - Decompose effects: no cost/no gate, cost/no gate, no cost/gate30, cost/gate30.
   - Gate sweep on saved predictions: no gate, `10`, `20`, `30`, `40`, `60` with the baseline cost model.
   - Cost sweep: spread/slippage `(0,0)`, `(5,2)`, `(10,5)`, `(20,10)` at gate30.
   - Output whether 2024 is a gross-ranking failure, a boundary-hysteresis failure, a cost/turnover failure, or a mixed signal.

4. Ensemble dispersion only if needed:
   - Do not confuse per-model predictions with base seeds.
   - Request per-model predictions only if averaged predictions show unexplained seed instability and the run artifacts can expose model-level outputs cheaply.

## Minimum Additional Run Matrix

The cheapest useful matrix is not new training. It is a canonical saved-prediction/replay matrix for the three known base seeds:

| Phase | Runs | Purpose | Requires approval |
| --- | ---: | --- | --- |
| A. Recover/regenerate saved predictions | 4 existing `314159` prediction folders plus 8 missing current folders: seeds `161803`, `271828` x years `2022-2025` | Establish one comparable 110-name LambdaRankIC prediction surface | Yes, if Colab/Drive execution is needed |
| B. Replay diagnostics on saved predictions | Same 12 seed-year rows across rank/cost/gate sensitivity scenarios | Diagnose 2024 churn and seed stability without new training | Yes |
| C. Targeted same-seed reproducibility check | 1 training rerun: `year=2024`, `base_seed=271828`, same pair cap and config | Only if artifacts are unrecoverable or 2024 looks like a one-off runtime anomaly | Yes |
| D. Targeted pair-cap stress | 2 training runs: `year=2024`, seeds `161803` and `271828`, cap `8192` if their current cap is lower | Test whether pair sampling undercoverage drives the 2024 split | Yes |
| E. Promotion-confidence seed expansion | 8 training runs: new base seeds `112358` and `141421` x years `2022-2025`, at the selected pair cap | Only if A/B pass or are promising enough to consider default promotion | Yes |

Recommended base-seed policy:

- Use confirmed seeds `314159`, `161803`, and `271828` for the first decision read.
- Do not add more seeds if the three-seed diagnostics clearly fail stability gates.
- Add `112358` and `141421` only if LambdaRankIC looks promotable or borderline after saved-prediction diagnostics.
- Rerun an existing seed only as a reproducibility check, not as a substitute for new base-seed evidence.

Recommended pair-cap policy:

- First, read manifests and normalize interpretation around the actual cap used by each existing run.
- For 110 names, complete same-day pair count is about 5,995 when all names are valid. `pair_cap=8192` is effectively all-pairs for that case; `pair_cap=4096` is a sampled but substantial subset.
- Do not run a broad pair-cap sweep yet. If current bad 2024 rows used a lower cap, test `8192` only on 2024 seeds `161803` and `271828` before extending all years.
- Do not use the broader-universe `pair_cap=1024` artifacts as evidence for the 110-name top-10 decision.

## Expected Drive Folder Contract

Use one campaign root, for example:

`/content/drive/MyDrive/MCI-GRU-Ablations/lambdarankic_110_name_stability/20260701_<run_tag>/`

Expected files and folders:

- `heartbeat.json`: current phase, current job, GPU/runtime evidence, completed/expected rows, last artifact written, error state.
- `lambdarankic_110_name_stability_manifest.json`: code commit, branch, data CSV paths and hashes when available, PIT universe CSV, selector artifact, years, base seeds, pair caps, model count, epoch budget, patience, loss config, selection metric, cost/rank-gate scenarios.
- `summaries/saved_prediction_inventory.csv`: canonical map of all recovered/regenerated prediction folders.
- `summaries/training_results.csv` and `summaries/training_results.json`: one row per training job when training is actually run.
- `summaries/backtest_results.csv` and `summaries/backtest_results.json`: one row per `(year, seed, pair_cap, scenario)`.
- `summaries/rank_stability_summary.csv`: per-year and per-seed-pair rank agreement aggregates.
- `summaries/cross_seed_rank_correlation.csv`: daily Spearman/Kendall by seed pair.
- `summaries/cross_seed_jaccard.csv`: daily Top-10/Top-20/Top-30 Jaccard by seed pair.
- `summaries/top10_boundary_churn.csv`: ranks 8-12 churn, #10/#11 margins, additions/removals, next-day contribution.
- `summaries/drive_artifact_inventory.csv`: one row per discovered Drive artifact or missing expected artifact, including URL, file/folder type, seed, year, run tag, and notes.
- `summaries/rank_drop_cost_sensitivity.csv`: gate and cost sensitivity grid.
- `summaries/decision_gate_report.md`: short final readout against the gates below.
- `predictions/<year>/seed<seed>/pair_cap_<cap>/averaged_predictions/*.csv`: recovered or regenerated averaged predictions.
- `predictions/<year>/seed<seed>/pair_cap_<cap>/per_model_predictions/`: optional; only if ensemble-dispersion diagnostics are approved.
- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/backtest_metrics.json`
- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/backtest_results.csv`
- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/daily_returns.csv`
- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/daily_holdings.csv`
- `backtests/<scenario>/<year>/seed<seed>/pair_cap_<cap>/trade_journal.csv`

For the restarted missing-prediction recovery, the default upload allowlist must be narrower than the incident cell:

- Always upload: `averaged_predictions/**`, `config.yaml`, `run_metadata.json`, `training_summary.json`, `evaluation_summary.json`, `timing_summary.json`, `feature_reference.json`, `training_*.log`, heartbeat, manifest, and row summary files.
- Do not upload by default: `predictions_model_*/*.csv`, checkpoints, `graph_data.pt`, or any broad recursive `.csv` match outside `averaged_predictions/`.
- Add upload heartbeat fields: `upload_phase`, `upload_current_folder`, `uploaded_file_count`, `expected_averaged_prediction_csv_count`, `last_uploaded_file`, and `last_upload_utc`.
- Append a completed row immediately after `averaged_predictions` and required metadata are uploaded; make per-model prediction archival a separate, explicitly approved step.
- Include a `finally` guard that disconnects the runtime on success or failure, but do not rely on `runtime.unassign()` alone; the notebook operator should still confirm the toolbar shows a reconnect state.

## Decision Gates

Pre-register these before running new training so 2024 tuning does not become hindsight fitting.

Promote LambdaRankIC candidate only if:

- The 12-row known-seed saved-prediction matrix is complete or faithfully regenerated, and the promotion-confidence matrix reaches at least five base seeds if default promotion is being considered.
- 2024 no longer shows a seed-specific churn cliff like `271828` versus `161803`, or the cause is fixed by a pre-declared cap/filter that also holds in 2022, 2023, and 2025.
- Net results after baseline costs and gate30 are competitive with pure IC across years, not just in average return.
- Worst seed-year and worst drawdown are not materially worse than pure IC.
- Turnover and total transaction cost are not materially higher than pure IC unless accompanied by robust net-return improvement and stable rank agreement.
- Cross-seed Top-10 agreement and rank correlations are high enough that the model is not effectively selecting a different portfolio for each base seed.
- Validation Rank IC improvements map to out-of-sample backtest behavior; otherwise `val_rank_ic` is not a sufficient selector for default status.

Keep LambdaRankIC experimental if:

- Saved predictions cannot be recovered or regenerated cleanly enough to audit provenance.
- 2024 remains dominated by unstable top-10 boundary churn, weak gross return, or seed-specific degradation.
- Performance depends on one base seed, one rank-drop gate, one cost assumption, or one pair cap.
- New base seeds broaden dispersion instead of narrowing it.
- LambdaRankIC improves Rank IC but does not improve deployable top-10 net behavior under realistic churn controls.

Consider LambdaRankIC as a second-stage or hybrid ranker if:

- Full-list Rank IC and Top-20/Top-30 agreement improve, but raw Top-10 boundary stability remains fragile.
- A pure-IC/LambdaRankIC rank blend, top-30 reranker, or hysteresis filter reduces churn without sacrificing 2023/2024 upside.
- The hybrid rule is validated across all years and base seeds, not tuned only to 2024 seed `271828`.

## Verification

- Read local source-of-truth docs: `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/DEFAULT_EXPERIMENT_RECIPE.md`, `docs/research/README.md`, and `docs/handoffs/2026-06-22-sp500-gics-top10-multiyear-baseline.md`.
- Checked that requested handoffs `docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md` and `docs/handoffs/2026-07-01-lambdarankic-stability-diagnostics-coordination.md` are absent in this checkout.
- Checked current code surfaces with `rg`:
  - `mci_gru/training/trainer.py` confirms `model_seed = config.seed + model_id`.
  - `scripts/gen_lambdarank_ic_pit_nb.py` contains LambdaRankIC seed, pair-cap, heartbeat, manifest, and result-writing logic.
  - `scripts/run_pit_saved_prediction_backtests.py` and `tests/backtest_sp500_daily.py` contain the cost-aware rank-drop replay contract and emitted backtest artifacts.
- Performed Drive read-only inventory after user approval:
  - Found current repeated-seed campaign root, completed run root, manifest/heartbeat/run-summary files, logs, compact summary rows, and current `artifacts/local_run_root`.
  - Confirmed current exact-name Drive searches for `top10_lambdarank_ic_<year>_seed<161803|271828>` only surfaced logs, not prediction folders.
  - Confirmed older seed `314159` has Drive-visible `averaged_predictions` CSV folders for 2022, 2023, 2024, and 2025.
- Ran `git status --short`; it emitted only warnings about the user global git ignore path before this file was added, and no status entries were shown.
- Post-write `git status --short` shows this handoff as the only status entry, with the same global-ignore warning.
- Post-write section scan found the required handoff sections and no `TBD`, `TODO`, or `PLACEHOLDER` residue.
- Investigated the 2026-07-02 Colab hang:
  - Verified the live toolbar still showed a connected GPU backend, then disconnected/deleted the runtime and observed `Reconnect G4 High-RAM`.
  - Fetched current Drive heartbeat, `training_rows.json`, logs, and row folders from `lambdarankic_110_name_missing_predictions_g4/20260701_185554`.
  - Confirmed 2022 seeds `271828` and `161803` are manifest-complete, 2023 seed `271828` training completed and uploaded averaged predictions, and the apparent hang was the unbounded per-model CSV upload loop.
- Added and locally checked the upload allowlist:
  - `python -m py_compile scripts/colab_recovery_upload_filter.py tests/test_colab_recovery_upload_filter.py` passed under system Python 3.13.3.
  - `python -m py_compile scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py scripts/colab_recovery_upload_filter.py tests/test_colab_recovery_upload_filter.py` passed under system Python 3.13.3.
  - A direct stdlib assertion harness calling all three test functions passed.
  - Repo venv commands could not run because `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\ruff.exe` are absent in this worktree.
- Added the self-contained Colab relaunch patch:
  - It replaces the broad `.csv` upload filter with the narrow allowlist.
  - It adds upload heartbeat progress every 25 files.
  - It filters out 2022 seeds `271828`/`161803`, and skips 2023 seed `271828` only if Drive API count confirms 246 averaged prediction CSVs.
- Attempted the fixed Colab relaunch after user approval:
  - Inserted the patched launcher as Cell 2 in the visible notebook.
  - Confirmed the in-cell GPU gate printed `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
  - Accepted Colab's notebook credential prompt, but the run blocked inside Google/Drive auth and the external `accounts.google.com` flow was not accessible to the agent.
  - Interrupted the cell before Drive heartbeat or training, then disconnected/deleted the runtime and confirmed `Reconnect G4 High-RAM`.
- Not run: repo pytest/Ruff via the preferred venv, new training rows, saved-prediction replay, backtests, GitHub mutation. No Drive heartbeat/run root was created by the auth-blocked relaunch attempt.

## Open Risks

- The prompt's current recovered all-years handoff is not present locally, so exact Drive IDs and run tags for the 2026-06-30/2026-07-01 artifacts still need live Drive verification in the controlling thread.
- The current repeated-seed manifest/heartbeat shows `lambdarank_pair_cap=8192`; if any row was resumed from another source, preserve row-level manifest checking before comparing seed differences.
- Missing `averaged_predictions` for `161803` and `271828` now look more like Drive artifact loss or runtime-local-only output than a search-term issue. Still check any live Colab runtime/checkpoint mirror before retraining.
- A rank-drop gate or hysteresis rule tuned on 2024 alone may overfit the stress year.
- Colab/Drive execution is the correct venue for any further diagnostics that require runtime work; do not use the user's local PC for expensive testing.

## Next Actions

1. Done: read-only Drive inventory found four usable seed `314159` prediction folders and no Drive-visible current `161803`/`271828` prediction folders.
2. Done: final Drive search plus Colab recovery checks did not find live-runtime copies of the eight missing current `averaged_predictions` folders.
3. Done: stopped the hanging/burning Colab runtime. Root cause is post-training Drive upload scope, not LambdaRankIC training.
4. Done: local guard and relaunch patch added. `scripts/colab_recovery_upload_filter.py` provides the narrow upload allowlist, `scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py` is the exact Colab launcher, and `docs/handoffs/2026-07-02-lambdarankic-g4-recovery-relaunch-patch.md` has the patch/runbook.
5. Blocked pending user OAuth: the patched relaunch passed the G4 gate but stopped in Colab/Google Drive auth before Drive heartbeat or training. User must manually complete Google/Drive OAuth, then rerun only patched Cell 2. Do not rerun 2022 rows. Verify whether 2023 seed `271828` has a complete averaged CSV set before deciding whether to rerun it.
6. Approval already given for the fixed relaunch, but runtime work should resume only after the OAuth blocker is cleared: targeted Colab regeneration/retraining only for unrecovered current rows, with `pair_cap=8192`, strict G4-class runtime proof, no backtests, no new seeds, and mandatory Drive persistence of `averaged_predictions`, heartbeat, manifest, logs, and row summaries.
7. Approval required: run saved-prediction-only rank stability, boundary churn, and rank-drop/cost sensitivity diagnostics for seeds `314159`, `161803`, and `271828` across 2022-2025.
8. Approval required only if diagnostics are inconclusive: run the single 2024 same-seed reproducibility check for `271828`.
9. Approval required only if 2024 churn appears pair-cap-related despite the manifest's `8192` cap: run a targeted pair-cap or pair-sampling reproducibility stress on 2024 seeds `161803` and `271828`.
10. Approval required only if LambdaRankIC still looks promotable: add base seeds `112358` and `141421` across all four years at the selected pair cap.

## Commands Run

- `git status --short`
- `Test-Path docs/handoffs/2026-06-30-sp500-top10-loss-backtest-all-years-recovered.md`
- `Test-Path docs/handoffs/2026-07-01-lambdarankic-stability-diagnostics-coordination.md`
- `rg --files docs/handoffs`
- `Get-Content -Raw AGENTS.md`
- `Get-Content -Raw docs/ARCHITECTURE.md`
- `Get-Content -Raw docs/DEFAULT_EXPERIMENT_RECIPE.md`
- `Get-Content -Raw docs/research/README.md`
- `Get-Content -Raw docs/handoffs/2026-06-22-sp500-gics-top10-multiyear-baseline.md`
- `rg -n "FULL_BASE_SEEDS|SCREEN_BASE_SEEDS|SMOKE_BASE_SEEDS|FULL_PAIR_CAPS|SCREEN_PAIR_CAPS|heartbeat|averaged_predictions|backtest_results|training_results|manifest|lambdarank_ic_max_pairs_per_day" scripts/gen_lambdarank_ic_pit_nb.py`
- `rg -n "min_rank_drop|spread|slippage|top_k|backtest_suffix|transaction_cost|rank_drop|predictions_dir" scripts/run_pit_saved_prediction_backtests.py tests/backtest_sp500_daily.py`
- `rg -n "model_seed|config.seed \+ model_id|train_multiple_models|set_seed" mci_gru/training/trainer.py mci_gru/config.py`
- `rg -n "lambdarank_ic|LambdaRankIC|val_rank_ic|pair_cap|lambdarank_ic_max_pairs_per_day|selection_metric" mci_gru/config.py mci_gru/training/losses.py mci_gru/training/trainer.py scripts/gen_lambdarank_ic_pit_nb.py`
- `git diff --check -- docs/handoffs/2026-07-01-lambdarankic-110-name-stability-run-plan.md`
- `rg -n "^## (Resume Here|Current Objective|What Changed|Key Decisions|Important Files|Verification|Open Risks|Next Actions)|TBD|TODO|PLACEHOLDER" docs/handoffs/2026-07-01-lambdarankic-110-name-stability-run-plan.md`
- `Get-Content -Raw docs/handoffs/2026-07-01-lambdarankic-110-name-stability-run-plan.md`
- Google Drive connector: searched and listed current repeated-seed campaign roots, current exact run names for seeds `161803`/`271828`, and older seed `314159` prediction folders.
- Chrome/Colab UI: verified a connected GPU backend, opened connection options, selected `Disconnect and delete runtime`, confirmed `Yes`, and observed `Reconnect G4 High-RAM`.
- Google Drive connector: fetched the live recovery heartbeat, training rows, row 3 training log, row 3 evaluation summary, and listed the run/log/training folders for `lambdarankic_110_name_missing_predictions_g4/20260701_185554`.
- `python -m py_compile scripts/colab_recovery_upload_filter.py tests/test_colab_recovery_upload_filter.py`
- `python -m py_compile scripts/colab_lambdarankic_110_name_missing_predictions_relaunch.py scripts/colab_recovery_upload_filter.py tests/test_colab_recovery_upload_filter.py`
- `python -c "<direct stdlib assertion harness for tests/test_colab_recovery_upload_filter.py>"`
- `Select-String -Path docs/handoffs/2026-07-02-lambdarankic-g4-recovery-relaunch-patch.md,scripts/colab_recovery_upload_filter.py,tests/test_colab_recovery_upload_filter.py -Pattern '[ \t]+$'`
- Chrome/Colab UI: pasted the patched relaunch launcher into Cell 2, verified `NVIDIA RTX PRO 6000 Blackwell Server Edition`, hit Colab/Google Drive auth, interrupted before any new training, selected `Disconnect and delete runtime`, confirmed `Yes`, and observed `Reconnect G4 High-RAM`.

## User Preferences

- Plan only; do not execute training, backtests, Colab, or expensive local jobs without a later explicit approval.
- If testing is needed later, use Colab, not the user's local PC.
- Do not mutate GitHub or push anything.
- Preserve PIT masked-panel and no-lookahead invariants.
- Do not conflate ensemble-member seeds with repeated/base seeds.
- Keep pure IC as the default until LambdaRankIC earns promotion with robust evidence.

## Do Not Do

- Do not launch new training before recovering or regenerating saved predictions.
- Do not treat missing Drive-visible prediction CSVs as proof that the models must be retrained.
- Do not use broader PIT-universe pair-cap-1024 artifacts as 110-name evidence.
- Do not change defaults, paper-trade behavior, or GitHub state from this planning thread.
- Do not tune a churn filter on 2024 and then call it general without cross-year and cross-seed validation.

## References

- Prompt current evidence summary from thread `019f18fa-98ec-7710-a1b2-d30dbbaa42d2`.
- Local memory summary: prior LambdaRankIC seed-gate and 110-name inventory separated base seeds from ensemble members and found missing 110-name saved-prediction surfaces for current repeated seeds.
- `docs/research/current/SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23.md` may be useful in a later evidence report, but this handoff did not promote anything into research evidence.
