# Notebook Best Practices

Last updated: 2026-06-18

This guide documents the notebook conventions used for MCI-GRU Colab experiment
notebooks. Follow it when creating or modifying notebooks under `notebooks/`.

## Purpose

Experiment notebooks should be reproducible launchers and analysis harnesses, not
places where core modeling logic lives. Keep reusable behavior in the Python
package and use notebooks to:

- mount Google Drive,
- clone or update the repo,
- stage Drive data into the local Colab VM,
- select local data paths and experiment settings,
- define a small, explicit run matrix,
- call `run_experiment.py` with Hydra overrides,
- collect standardized outputs,
- export decision tables, plots, logs, and a Markdown summary back to Drive.

Treat Google Drive as durable storage, not as the notebook's active working
filesystem. Mounted Drive is convenient but quota-limited; notebooks should do
high-volume reads and writes on the Colab VM and copy compact artifacts to Drive
at well-defined checkpoints.

## Agent Operation

When an agent needs to operate a notebook in Colab, default to the
`chrome:control-chrome` workflow in
`docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`. Chrome control is the preferred
surface for the visible Colab UI because it can use the user's existing Chrome
profile and logged-in Google, Drive, and Colab state.

Use Playwright MCP only as a legacy fallback when Chrome control is unavailable,
blocked, or explicitly requested. Keep long full-preset runs in the visible
notebook UI, and use Drive artifacts as the durable source of truth when cell
output does not stream.

After any live Colab attempt, complete the mandatory Run Review in
`docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md`. Separate control-plane proof,
local notebook checks, and Drive artifact evidence from actual full-run
validation; a killed proof run or passing contract test is not proof that a
remote Colab training run succeeded.

## Drive Path Discovery

Before creating or repairing a notebook that reads existing Drive artifacts, use
the Google Drive skill/connector to ground the artifact location. Do not infer
paths from memory, run tags, sibling notebook conventions, or local
`drive_outputs/` mirrors.

Required discovery flow:

- Search Drive for one or two unique artifact names, such as a decision table
  CSV, summary report, manifest, or run zip.
- List the candidate run folder and confirm it contains the required files.
- Read folder metadata and walk parents until the durable root folder is clear.
- Translate only the verified Drive folder into the Colab mount path, usually
  `/content/drive/MyDrive/<root>/<experiment>/<run_tag>`.
- Record the confirmed path in the notebook markdown and setup cell.
- Keep `*_DIR` and `*_ZIP` environment overrides for moved or copied artifacts.
- If required artifacts are missing, fail in the setup cell with the searched
  candidate folders instead of allowing a later loader cell to raise a generic
  `FileNotFoundError`.

When a user reports a notebook cannot find Drive files, start with Drive
discovery again. The correct fix is usually the real Drive folder, not another
guessed fallback.

## Required Structure

Use the same high-level sections as the ablation notebooks:

1. Title and intent.
2. Mount Drive, clone repo, and install dependencies.
3. Configuration cell with all user-editable knobs.
4. Data archive discovery and local VM staging.
5. Data availability check against local paths.
6. Matrix definition.
7. Run, collect, and score helpers.
8. Matrix execution using local scratch outputs.
9. Main-effect and interaction analysis.
10. Visualization.
11. Failed-run inspection.
12. Compact artifact sync and summary report export.

Keep markdown cells short and operational. A notebook should tell the runner what
the next cell does and what output to expect.

## Configuration

Put all run-time knobs in one early code cell:

- data file candidates,
- date splits,
- model budget,
- batch size and learning rate,
- seed or seed list,
- strict regime settings,
- optional feature flags,
- output root and run tag.
- local scratch root,
- Drive export root.

Use full-budget defaults only for final confirmation notebooks:

```text
training.num_models = 20
training.num_epochs = 100
training.early_stopping_patience = 15
evaluation.bootstrap_resamples = 1000
```

For scouting notebooks, keep the quick-run knobs visible and label them clearly.

Full current-preset Colab notebooks should include a hard runtime preflight:

- require the user or agent to select `G4 GPU` in the Colab UI,
- confirm the visible toolbar/runtime label is not `T4`,
- run an in-notebook `nvidia-smi` gate before training,
- stop before launching if the runtime is CPU, T4, or otherwise below the
  notebook-specific requirement.

Use separate roots for work and persistence:

```text
LOCAL_WORK_ROOT = "/content/mci_gru_work/<experiment_name>/<run_tag>"
DRIVE_EXPORT_ROOT = "/content/drive/MyDrive/MCI-GRU-Ablations/<experiment_name>/<run_tag>"
```

During execution, write logs, predictions, intermediate CSVs, and temporary
files under `LOCAL_WORK_ROOT`. Sync only selected final artifacts to
`DRIVE_EXPORT_ROOT`.

## Google Drive Quota Safety

Colab notebooks can hit Google Drive operation or bandwidth quotas, especially
when reading many small files, repeatedly scanning mounted directories, writing
large log streams, or using a popular shared Drive file. Failures may surface as
`Input/output error` from mounted Drive paths.

Design notebooks around this rule:

```text
Drive is the source and sink. /content is the workspace.
```

Required practices:

- Keep datasets on Drive as a small number of archives such as `.zip` or
  `.tar.gz`.
- Copy the selected archive from Drive to `/content` once at notebook startup.
- Unarchive into a local VM directory such as `/content/mci_gru_data/<run_tag>`.
- Point `data.csv_path`, `REGIME_INPUTS_CSV`, and other experiment inputs at
  local `/content` paths.
- Write per-run logs, checkpoints, predictions, and temporary outputs locally.
- Copy only compact final artifacts back to Drive after runs finish.
- Avoid loops that call `os.listdir`, `glob`, `read_csv`, `torch.load`, or
  similar file operations directly against many mounted Drive files.
- Avoid writing every stdout/stderr line directly to Drive during training.

If a Drive quota failure occurs, do not immediately rerun the same notebook
against mounted Drive paths. First stage the needed data locally, reduce the
number of Drive file operations, or wait for quota reset if the file itself is
quota-limited.

## Data and Lookahead Safety

Preserve the repository invariants:

- Train, validation, and test dates must respect the label embargo.
- Normalization and graph construction must use train-period cutoffs.
- Dynamic graph runs should use `GraphSchedule` through `graph.update_frequency_months`.
- Do not use rank-label portfolio-return metrics as raw-return metrics unless the
  evaluation scale has been explicitly audited.

For regime features, prefer strict failure over silent zero-filled features in
ablation notebooks:

```text
features.regime_strict=true
```

If FRED access is unavailable, set `REGIME_INPUTS_CSV` to a file that follows
`docs/REGIME_DATA_CONTRACT.md`.

For Colab data setup, prefer this flow:

```text
Drive archive -> local /content archive copy -> local extracted data -> Hydra local paths
```

Do not train directly from a directory of many small files in mounted Drive. If a
shared source file is popular or quota-limited, make a private Drive copy first
and use that private copy as the notebook input.

## Matrix Design

Keep matrices small enough to answer the stated question.

- Use broad factorial notebooks only for discovery.
- Use focused confirmation notebooks after a report identifies winners.
- Make each factor explicit in the result rows, not only embedded in the run name.
- Put diagnostic controls behind flags so reruns can focus on primary candidates.
- Avoid mixing label scales in one decision score unless the summary clearly
  separates them.

Run names should be deterministic, readable, and path-safe:

```text
static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1
```

## Hydra Overrides

Build overrides as lists and compose them from named factor blocks. Avoid inline
string construction scattered across cells.

Every notebook should include these baseline overrides unless the experiment is
specifically testing them:

```text
seed=1729
data.source=csv
features=with_momentum
features.include_weekly_momentum=true
features.momentum_blend_mode=static
features.include_global_regime=true
features.regime_strict=true
features.regime_include_subsequent_returns=false
tracking.enabled=true
tracking.log_predictions=false
graph.use_multi_feature_edges=true
graph.update_frequency_months=0
graph.corr_lookback_days=252
graph.top_k=0
graph.top_k_metric=corr
graph.drop_edge_p=0.1
training.loss_type=ic
training.label_type=returns
training.selection_metric=val_ic
training.shuffle_train=true
model.label_t=5
```

For graph experiments, always record:

- `graph.update_frequency_months`,
- `graph.corr_lookback_days`,
- `graph.top_k`,
- `graph.top_k_metric`,
- `graph.use_multi_feature_edges`,
- edge feature additions such as snapshot age or lead-lag,
- `training.shuffle_train`.

## Outputs

Each notebook run folder should contain:

- manifest JSON,
- raw results CSV,
- decision table CSV,
- interim decision table CSV,
- HTML decision table,
- main-effect CSVs,
- interaction CSVs where useful,
- metric bar plot PNG,
- stdout and stderr logs per run,
- Markdown summary report.

Write outputs to Google Drive under:

```text
/content/drive/MyDrive/MCI-GRU-Ablations/<experiment_name>/<run_tag>
```

The full run should execute under a local scratch folder first, for example:

```text
/content/mci_gru_work/<experiment_name>/<run_tag>
```

The Drive folder should receive final artifacts only. Prefer copying these
outputs in one sync/export cell after matrix execution:

- manifest JSON,
- raw and decision CSVs,
- plots,
- failed-run log tails or compressed logs,
- Markdown summary report,
- a compressed archive of detailed logs when needed.

The manifest should include the run matrix, baseline overrides, filters, budget
settings, local work root, Drive export root, and data archive used for staging.

Long full-preset notebooks should also write small durable progress artifacts to
Drive while the heavy work runs:

- `heartbeat.json` for job-level phase, status, current job, GPU, and timestamp,
- `ensemble_progress.json` for per-model progress and resume state,
- `training_results.csv` and `training_results.json` for completed jobs.

Use these files as the source of truth when Colab cell output does not stream.

## Scoring and Reporting

Use the shared decision-score pattern from the existing ablation notebooks, but
state its limitations in the summary report. Always report the underlying
metrics next to the score:

- average IC,
- IC confidence interval lower bound,
- average Spearman correlation,
- top-20 return,
- top-20 return confidence interval lower bound,
- top-20 Newey-West Sharpe,
- training mean best validation IC where available.

Decision scores help sort candidates; they are not a substitute for reading the
metric columns and failed-run logs.

## Failure Handling

Never hide failed runs.

- Keep `stdout.log` and `stderr.log` for every run.
- Add a final failed-run inspection cell that prints log tails.
- Keep failed rows in the raw and decision tables with `status=FAILED`.
- Treat recurring failures as experiment findings, not notebook noise.
- For expensive ensembles, make jobs resumable at the model level with
  checkpoints or prediction folders. Repeated failure at the same job can be a
  runtime interruption plus an atomic job design, not necessarily a model-code
  bug.
- If logs are large, keep full logs locally during execution and export
  compressed logs or relevant tails to Drive.
- Classify Drive `Input/output error` failures separately from model or data
  failures; they usually indicate a notebook I/O pattern or Drive quota issue,
  not an experiment result.
- If Drive OAuth, `Run anyway`, or Colab Secrets access prompts interrupt setup,
  grant the required access, rerun the setup/gate cell, and re-check the runtime
  before training. Do not assume final cleanup ran if the notebook failed before
  the launcher cell.

## Runtime Cleanup

Full-run Colab notebooks should launch long work from a visible foreground final
cell, not from a hidden kernel or detached background process. Wrap that launcher
with `try` / `finally` and call `google.colab.runtime.unassign()` in the final
block so the runtime releases after completion or failure.

This auto-unassign path only runs if execution reaches the final cell. If an
earlier setup, OAuth, secret-access, or GPU-gate cell fails, manually use
`Runtime > Disconnect and delete runtime` before leaving the run unattended.

## Review Checklist

Before pushing a notebook:

- Open it as JSON or run `ConvertFrom-Json` to verify it is valid.
- Confirm the first setup cell clones the intended branch.
- Confirm existing Drive artifact paths were verified with the Google Drive
  skill/connector rather than guessed from local paths or prior notebooks.
- Confirm output paths include a timestamped run tag.
- Confirm data is staged from Drive to local `/content` before training.
- Confirm Hydra data paths point at local `/content` files, not mounted Drive
  files, during execution.
- Confirm detailed logs and intermediate outputs are written locally first.
- Confirm the final Drive sync exports compact artifacts rather than streaming
  every intermediate file to Drive.
- Confirm every factor has row metadata.
- Confirm strict regime behavior is intentional.
- Confirm no secrets are committed.
- Confirm GitHub-hosted notebook prompts are expected: `Run anyway`, Drive
  OAuth, and Colab Secrets per-notebook access where needed.
- Confirm full current-preset notebooks hard-gate on both visible `G4 GPU`
  runtime selection and in-notebook GPU evidence.
- Confirm long expensive ensembles write heartbeat, per-model progress, and
  completed-job result artifacts to Drive.
- Confirm foreground final-cell cleanup calls `runtime.unassign()` and documents
  the manual cleanup path for earlier setup failures.
- Confirm summary report links or paths match generated artifacts.
- Run a quick syntax/JSON validation locally if the notebook was edited by hand.
