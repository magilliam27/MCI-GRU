# Colab Portfolio-IC Runner Issues Handoff

Last updated: 2026-06-01

## Resume Here

- Continue monitoring the visible Colab GitHub notebook:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/portfolio-ic-hybrid-testing/notebooks/portfolio_ic_upward_sweep_colab.ipynb`
- The run is launched from the real Colab UI, not a hidden kernel. Do not restart it unless Drive artifacts show it stopped or failed.
- First check Drive artifacts:
  - Heartbeat:
    `https://drive.google.com/file/d/1KPLxyljCv-wEnd4KkTism4n8jDjNCWAc/view`
  - Job 7 run dir folder:
    `https://drive.google.com/drive/folders/1RCQIM-HKvw-ueFtw9ifIe2gp6Uio0qUU`
  - Ensemble progress:
    `https://drive.google.com/file/d/14HMfrIPIzlkdkx9Yoijap-x2GRbAFSfQ/view`
- Current confirmed state:
  - Runtime GPU gate passed on `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
  - `heartbeat.json` showed `status=RUNNING`, `job_index=7`, `current_job=portfolio_ic_weight75_2024_seed314159`.
  - `ensemble_progress.json` existed and later showed `status=RUNNING`, `completed_models=3`, `last_model_id=2`, `last_model_status=reused_predictions`.

## Current Objective

Resume the Portfolio-IC upward weight sweep for `portfolio_ic_weight=0.75` and `1.0` without burning unnecessary Colab resources. The current run must use G4-class or better non-T4 hardware and must release the Colab runtime when the foreground notebook finishes or fails.

## What Changed

- Implemented and pushed commit `0581cf1 Add resumable Portfolio-IC Colab runner` on branch `codex/portfolio-ic-hybrid-testing`.
- Added model-level resume support so a job interruption resumes from existing `predictions_model_N` folders or model checkpoints instead of restarting the whole 20-model job.
- Added a foreground Colab launcher notebook that runs via visible cell execution and calls `runtime.unassign()` in a `finally` block.
- Added a script-level GPU gate that rejects T4 and accepts non-T4 G4-class or better devices.
- Relaunched the notebook from the real Colab UI after selecting `G4 GPU` in `Change runtime type`.

## Key Decisions

- Treat `G4 GPU` as a hard runtime preflight for these full runs. A T4 runtime is not acceptable unless the user explicitly asks for a cheap proof run.
- Use visible Colab notebook execution as the control plane. The earlier detached or hidden-kernel approach made failures opaque and left the visible notebook stale on an error.
- Use Drive artifacts as truth when Colab cell output does not stream child subprocess logs.
- Keep the run root fixed to the existing partial run:
  `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_upward_sweep/20260601_013922_static_regime_full`
- Resume job 7 in the best partial run dir:
  `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_upward_sweep/20260601_013922_static_regime_full/training_runs/portfolio_ic_weight75_2024_seed314159/20260601_043243`

## Important Files

- `mci_gru/training/trainer.py` - model-level resume logic and `ensemble_progress.json` writes.
- `run_experiment.py` - records resumed-from-predictions and resumed-from-checkpoint counts.
- `scripts/run_portfolio_ic_upward_sweep.py` - foreground resumable sweep runner, GPU gate, heartbeat, job grid.
- `scripts/gen_portfolio_ic_upward_sweep_nb.py` - notebook generator.
- `notebooks/portfolio_ic_upward_sweep_colab.ipynb` - visible Colab launcher with auto-unassign.
- `tests/test_portfolio_ic_trainer.py` - resume behavior coverage.
- `tests/test_portfolio_ic_upward_sweep_notebook.py` - notebook contract coverage.
- `docs/workflows/COLAB_PLAYWRIGHT_MCP_GUIDE.md` - update target for agent-facing Colab operating guidance.
- `docs/NOTEBOOK_BEST_PRACTICES.md` - update target for durable notebook design guidance.

## Verification

- Worktree branch: `codex/portfolio-ic-hybrid-testing`.
- Latest pushed commit: `0581cf1 Add resumable Portfolio-IC Colab runner`.
- Current worktree status check with `git -c safe.directory=... status --short` printed only global-ignore permission warnings and no changed files.
- Earlier focused verification in this thread:
  - `python -m pytest tests\test_portfolio_ic_trainer.py tests\test_portfolio_ic_upward_sweep_notebook.py -v --tb=short --basetemp .tmp_pytest\portfolio_ic_resume`
  - Result recorded in-thread: `5 passed`.
  - `ruff check` on touched files passed.
- Colab UI verification:
  - Toolbar showed `Connect G4 High-RAM` before launch.
  - Runtime later showed `G4 (Python 3)`.
  - Input/GPU gate output showed `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition` and `GPU gate passed.`
- Drive verification:
  - `heartbeat.json` modified at `2026-06-02T00:17:30Z`.
  - Heartbeat content showed job 7 running on the Blackwell GPU.
  - Job 7 run dir had fresh updates to `config.yaml`, `run_metadata.json`, `feature_reference.json`, and `graph_data.pt`.
  - `ensemble_progress.json` was created at `2026-06-02T00:21:26Z`, then modified at `2026-06-02T00:24:45Z` and showed model 2 reused from predictions.

## Open Risks

- Colab cell output is not currently streaming the runner's `Jobs:` and `Starting:` lines, even though Drive artifacts prove the runner is active.
- `ensemble_progress.json` had only reached `completed_models=3` when this handoff was written. Confirm it advances to the already-existing prediction dirs and then to new model training.
- `runtime.unassign()` is in the final foreground cell. If an earlier setup cell fails before the final cell starts, automatic release will not run and the runtime must be manually disconnected/deleted.
- GitHub-hosted notebooks can require `Run anyway`, Drive OAuth, and per-notebook Colab Secret access. The FRED secret prompt caused an initial error; after granting access, the input/GPU cell had to be rerun.
- The stale Drive notebook tab still exists and may show the old error. It is not the active control plane.

## Data/Experiment State

- Run root:
  `/content/drive/MyDrive/MCI-GRU-Ablations/portfolio_ic_hybrid_upward_sweep/20260601_013922_static_regime_full`
- Job grid: 24 jobs total:
  - weights: `0.75`, `1.0`
  - years: `2022`, `2023`, `2024`, `2025`
  - seeds: `314159`, `271828`, `161803`
  - order: all `0.75` jobs by year/seed, then all `1.0` jobs by year/seed.
- Completed before resume: 6 OK jobs, all `portfolio_ic_weight75` for 2022 and 2023 across the 3 seeds.
- Current job: `portfolio_ic_weight75_2024_seed314159`.
- Existing partials for current job:
  - `20260601_043243`: predictions through model 16 and checkpoints through model 17; chosen as resume dir.
  - `20260601_180451`: predictions through model 14 and checkpoints through model 15.
  - `20260601_193021`: predictions through model 3 and checkpoints through model 4.

## User Preferences

- Do not use T4 for this sweep.
- Prefer real Colab UI operation over hidden kernel launches.
- Avoid wasting Colab resources. Confirm automatic shutdown and manually release the runtime if setup fails before the final cell can run.
- Report exact artifacts and timestamps instead of vague "still running" reassurance.

## Do Not Do

- Do not rerun all 24 jobs from scratch unless the user explicitly chooses that.
- Do not treat the stale Drive notebook tab as authoritative.
- Do not paste or expose the `FRED_API_KEY`; only refer to the secret by name.
- Do not assume closing the local PC or browser stops Colab. Only Colab-side disconnect/delete/unassign stops resource use.

## Next Actions

1. Refresh `ensemble_progress.json`; expect it to advance through reused predictions for models 0-16 and then recover/train models 17-19.
2. If progress stalls, inspect the visible Colab cell and Drive run dir before restarting anything.
3. After job 7 completes, verify `training_results.csv/json` updates and the runner advances to job 8.
4. When all jobs complete or on failure, confirm `runtime.unassign()` ran or manually disconnect/delete the runtime.
5. Update Colab guidance docs with the lessons from this incident, especially G4 preflight, visible-control-plane preference, prompt handling, Drive heartbeat/progress checks, and setup-cell failure cleanup.

## References

- Active GitHub Colab notebook:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/portfolio-ic-hybrid-testing/notebooks/portfolio_ic_upward_sweep_colab.ipynb`
- Stale Drive notebook:
  `https://colab.research.google.com/drive/1rPc_w6Ixc5HK9IM86LxlJVU7NEnT06Cp`
- Summary folder:
  `https://drive.google.com/drive/folders/11mxjM0S2xzycLr6Rxr3QSeH-DajcQSTK`
