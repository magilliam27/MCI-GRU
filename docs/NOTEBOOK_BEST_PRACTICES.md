# Notebook Best Practices

Last updated: 2026-05-01

This guide documents the notebook conventions used for MCI-GRU Colab experiment
notebooks. Follow it when creating or modifying notebooks under `notebooks/`.

## Purpose

Experiment notebooks should be reproducible launchers and analysis harnesses, not
places where core modeling logic lives. Keep reusable behavior in the Python
package and use notebooks to:

- mount Google Drive,
- clone or update the repo,
- select data and experiment settings,
- define a small, explicit run matrix,
- call `run_experiment.py` with Hydra overrides,
- collect standardized outputs,
- export decision tables, plots, logs, and a Markdown summary.

## Required Structure

Use the same high-level sections as the ablation notebooks:

1. Title and intent.
2. Mount Drive, clone repo, and install dependencies.
3. Configuration cell with all user-editable knobs.
4. Data availability check.
5. Matrix definition.
6. Run, collect, and score helpers.
7. Matrix execution.
8. Main-effect and interaction analysis.
9. Visualization.
10. Failed-run inspection.
11. Summary report export.

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

Use full-budget defaults only for final confirmation notebooks:

```text
training.num_models = 20
training.num_epochs = 100
training.early_stopping_patience = 15
evaluation.bootstrap_resamples = 1000
```

For scouting notebooks, keep the quick-run knobs visible and label them clearly.

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
static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__seed-42
```

## Hydra Overrides

Build overrides as lists and compose them from named factor blocks. Avoid inline
string construction scattered across cells.

Every notebook should include these baseline overrides unless the experiment is
specifically testing them:

```text
data.source=csv
features=with_momentum
tracking.enabled=true
tracking.log_predictions=false
graph.use_multi_feature_edges=true
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

The manifest should include the run matrix, baseline overrides, filters, and
budget settings.

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

## Review Checklist

Before pushing a notebook:

- Open it as JSON or run `ConvertFrom-Json` to verify it is valid.
- Confirm the first setup cell clones the intended branch.
- Confirm output paths include a timestamped run tag.
- Confirm every factor has row metadata.
- Confirm strict regime behavior is intentional.
- Confirm no secrets are committed.
- Confirm summary report links or paths match generated artifacts.
- Run a quick syntax/JSON validation locally if the notebook was edited by hand.
