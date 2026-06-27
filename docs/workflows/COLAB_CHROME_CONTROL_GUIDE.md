# Colab Chrome Control Agent Guide

Last updated: 2026-06-18

Use this guide when an agent needs to operate a Google Colab notebook from the
logged-in Windows/Codex environment. The default browser surface is now the
`chrome:control-chrome` skill, because it can claim or open tabs in the user's
existing Chrome profile and preserve Google, Drive, Colab, and extension state.

This is a reusable workflow doc. Notebook-specific handoffs can point here and
then list only the branch, Colab URL, cell order, expected outputs, and artifact
paths for that run.

## Default Scope

- Use `chrome:control-chrome` for automated Colab work by default.
- Use the visible Colab notebook UI as the control plane for full-preset or
  costly training runs.
- Prefer purpose-built connectors for non-browser work, such as Google Drive
  artifact discovery. Use Chrome control for the Colab UI itself.
- Claim an existing Colab tab when it is already open on the right notebook and
  branch. Open a new Colab tab only when no suitable tab exists.
- Click through routine notebook, Drive, and runtime prompts that are directly
  required for the user-requested run.
- Stop and ask the user before entering passwords, OTPs, passkeys, account
  recovery details, or unrelated Google security settings.
- Never paste or repeat API keys, account credentials, tokens, or private
  secrets in docs, reviews, screenshots, or final messages.
- Closing Chrome, releasing a tab, or ending the local agent is not proof that a
  Colab cloud runtime stopped. Only a Colab-side disconnected/deleted runtime
  state counts.
- Avoid hidden kernel launches or detached background processes when the user
  needs traceable progress, prompt handling, and runtime cleanup.

Use Playwright MCP only as a documented legacy fallback when Chrome control is
unavailable, blocked, or explicitly requested. Record why the fallback was used.

## Chrome Control Bootstrap

Before controlling Chrome, read the `chrome:control-chrome` skill instructions
for the current plugin version. If the browser-control JavaScript tool is not
already exposed, use tool discovery for:

```text
node_repl js
```

Follow the skill bootstrap exactly. In particular:

- import the plugin's `scripts/browser-client.mjs` by absolute path,
- bind the Chrome extension browser surface,
- read the complete `browser.documentation()` output before acting,
- use only the Node REPL browser-client route for Chrome extension control.

Do not use external browser-control tools or the in-app browser for this Colab
surface unless the user explicitly chooses a fallback.

## Tab Preflight

Start every fresh browser attempt by listing existing Chrome tabs:

```js
await browser.user.openTabs()
```

Then:

1. Choose a matching Colab tab by visible title, URL, recency, and tab group.
2. Claim that exact tab with `browser.user.claimTab(tab)`.
3. Reuse the returned controllable tab for the in-skill `tab.playwright` API,
   DOM snapshots, screenshots, clicks, typing, and content reads.
4. If no suitable Colab tab exists, open the notebook-specific Colab URL using
   the browser API documented by `browser.documentation()`.

Do not guess tab ids. Do not reload a tab that is already on the right URL
unless reload is intentional and safe. When a tab is left for the user or a
later agent, finalize Chrome browser work with that tab marked as a handoff or
deliverable according to the Chrome skill.

## Preflight Checklist

1. Read the notebook-specific handoff or issue first.
2. Record the target branch and Colab URL.
3. Confirm whether this is a proof run or a full training run.
4. Confirm the runtime target. Current MCI-GRU full-preset training runs should
   use `G4 GPU`; do not accept `T4` unless the user explicitly asks for a
   cheaper proof.
5. Check whether the notebook must be fresh from GitHub. Colab GitHub URLs read
   the remote branch, not local unpushed edits.
6. Use Chrome control to list and claim an existing matching tab, or open the
   notebook-specific Colab URL.

For the repeated-seed PIT proof notebook, the known-good URL was:

```text
https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_repeated_seed_replication_colab.ipynb
```

The hardwired PIT input commit for that notebook was:

```text
2119773 Hardwire repeated-seed Colab inputs
```

Treat those as historical examples unless a current handoff names that exact
notebook again.

## Runtime Setup

Use the Colab UI, not local assumptions:

1. Open `Runtime`.
2. Choose `Change runtime type`.
3. Select `G4 GPU`.
4. Save.
5. Confirm the toolbar or runtime status shows `G4`, not `T4`, before running
   setup or training cells.
6. Run the notebook's GPU gate and confirm `nvidia-smi` reports a non-T4
   G4-class or better device before launching a full current-preset job.

For one proven MCI-GRU run, Colab reported `G4 (Python 3)` and the setup cell saw
CUDA with an `NVIDIA RTX PRO 6000 Blackwell Server Edition` GPU. Treat the exact
GPU model as evidence for that run, not as a permanent requirement.

If a notebook opens already connected to `T4`, stop before rerunning cells and
switch the runtime to `G4 GPU`. Runtime changes can disconnect the notebook, so
reconnect and rerun setup from the top after saving the runtime change.

## Runtime Matrix

Use this matrix before launching cells, and record the accepted runtime in the
run review:

| Run type | Accepted runtime evidence |
| --- | --- |
| Full preset | Visible `G4 GPU` selection plus in-notebook GPU evidence showing a non-`T4` G4-class or better CUDA device. Reject `T4` and CPU. |
| Screen/proof | Same `G4 GPU` expectation if training is launched, unless the user explicitly approved a cheap proof. Stop after the requested proof point. |
| Replay-only | CPU is acceptable when replaying saved predictions or reading existing artifacts without training. Do not claim new training validation. |
| User-approved cheap proof | `T4` or CPU is acceptable only when the user explicitly approved it. Label the result as cheap proof, not full-preset evidence. |

Do not report an exact GPU model as an expected requirement unless the notebook
printed that model during the current attempt.

## Secret Preflight

Regime-enabled current-preset runs need `FRED_API_KEY`.

1. Prefer the Colab Secrets panel.
2. Add or update a secret named exactly `FRED_API_KEY`.
3. Enable notebook access for the target notebook if Colab asks.
4. Rerun the notebook's key/config cell and confirm it prints that
   `FRED_API_KEY` is loaded or set.

Never commit, screenshot, paste into markdown, save back to GitHub, or repeat
the raw API key in final messages or reviews. If the user gives the key during a
run, use it only to populate the Colab secret or the current ephemeral runtime,
then refer to it as `FRED_API_KEY` or a redacted key in notes.

## Prompt Handling

Expected prompts the agent may accept for the target notebook:

- `Run anyway` for a GitHub-hosted notebook warning.
- `Connect to Google Drive` when the notebook mounts Drive.
- Google OAuth `Continue` for Drive access requested by the Colab notebook.
- Colab Secrets per-notebook access prompts for an existing secret such as
  `FRED_API_KEY`.
- Runtime reconnection prompts when they are clearly tied to the notebook run.

Stop and ask the user for:

- password entry
- OTP or SMS code
- passkey enrollment or passkey use
- account recovery prompts
- security settings not directly tied to Colab execution
- unexpected account chooser ambiguity
- missing or expired API/secret access that cannot be resolved from the
  existing notebook/session

If a setup cell errors because Drive OAuth or Colab Secrets access was not yet
granted, grant the prompt, rerun the setup/gate cell, and re-check the visible
runtime state. If the error happened before the final cleanup cell could start,
assume auto-release did not run and manually disconnect/delete the runtime when
stopping the attempt.

## Notebook Freshness

Before running expensive cells, inspect the setup and data-input cells in the
live Colab UI.

For the repeated-seed PIT notebook, Cell 5 had to use explicit paths and avoid
recursive Drive discovery. The expected paths for that historical run were:

```text
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json
```

If the live Colab cell still has blank paths or recursive `root.glob(...)`
searches when the handoff expects fixed paths, stop. The notebook is probably
stale relative to the branch or commit. Do not let it hang on Drive discovery.

## Standard Run Pattern

For a proof run, execute only far enough to verify that training starts:

1. Run the setup cell: mount Drive, clone or update repo, install dependencies,
   check GPU and config fields.
2. Run the data-input cell and verify row counts, date range, stock count, and
   relevant hashes if the notebook prints them.
3. Run the runtime/key/config cell.
4. Run the job-matrix/static-input cell.
5. Run helper-definition cells.
6. Start the final training launcher cell.
7. Watch until the first training job begins and prints the expected run tag or
   command.
8. Stop the runtime if this was only a proof.

For the repeated-seed PIT notebook, the proven historical cell order was:

```text
Cell 3   setup, Drive mount, clone, install, GPU/PIT config checks
Cell 5   PIT data inputs
Cell 7   FRED/runtime configuration
Cell 9   job matrix and static regime inputs
Cell 11  training/backtest helpers
Cell 13  execute training and backtests
```

The proof succeeded when Cell 13 output included:

```text
Training: pit_seed_314159_replication_2022
Attempt: 1/3
```

For a full run, continue monitoring after this point according to the
notebook-specific handoff. Watch for prompts, disconnects, Drive write failures,
and first-job errors.

## Failure Taxonomy

Use this taxonomy in the run review instead of collapsing every Colab problem
into a generic automation failure:

- Profile lock: Chrome cannot attach to the user's intended profile. Stop before
  switching browser surfaces, and record the selected fallback reason.
- Target/page/context closed: the claimed tab or automation context disappeared.
  Re-list tabs, reclaim the correct Colab tab, and avoid claiming success from
  stale output.
- OAuth prompt/tab close: Drive, Colab, or Google auth interrupted setup or
  closed a popup/tab. Handle only expected prompts, then rerun the setup/gate
  cell and re-check runtime state.
- DriveFS/quota: mounted Drive returns I/O errors, stalls, or quota symptoms.
  Treat Drive artifacts or Drive API reads as the durable truth, and avoid
  rerunning the same mounted-Drive pattern blindly.
- Stale GitHub branch: the live notebook differs from the expected branch or
  commit. Stop before expensive cells; Colab GitHub URLs read the remote branch,
  not local unpushed edits.
- GPU gate failure: visible runtime, toolbar, or `nvidia-smi` evidence does not
  meet the run type in the runtime matrix. Stop or get explicit cheap-proof
  approval before launching training.
- Interrupted atomic job: a long all-or-nothing ensemble restarts after
  disconnect or interruption. Record it separately from model-code failure and
  prefer model-level checkpoints or resumable prediction folders.

## Long-Run Monitoring

When Colab output does not stream child-process logs, treat Drive artifacts as
the source of truth:

- `heartbeat.json` for job-level phase, status, current job, runtime GPU, and
  last update time.
- `ensemble_progress.json` for model-level resume progress inside an expensive
  ensemble job.
- `training_results.csv` and `training_results.json` for completed jobs and
  final statuses.

Repeated failure or restart at the same job is not automatically a model-code
failure. In the Portfolio-IC upward sweep, job 7 repeatedly restarted because a
long atomic 20-model job was interrupted. Expensive ensembles should write
per-model checkpoints or prediction folders and resume from them before
retraining the whole job.

## Full-Run Cleanup

Full-run notebooks should call `google.colab.runtime.unassign()` from the
foreground final launcher cell, normally in a `finally` block. That cleanup only
runs if execution reaches the final cell. If an earlier setup, OAuth, secret, or
GPU-gate cell fails, manually use `Runtime > Disconnect and delete runtime`
after collecting the needed evidence.

When ending Chrome browser work, finalize Chrome tabs according to the
`chrome:control-chrome` skill:

- keep a live Colab tab as `handoff` if a later turn should continue from it,
- keep it as `deliverable` if the user needs the open page,
- otherwise omit it so agent-created intermediate tabs can close.

## Stopping A Proof Run

When the goal is to prove automation rather than finish training:

1. Use Colab: `Runtime > Disconnect and delete runtime`.
2. Confirm the deletion prompt with `Yes`.
3. Keep watching until the UI shows a disconnected state.
4. Record the visible status, such as `Reconnect G4 High-RAM Click to connect`.

Do not rely on local process cleanup as a stop condition. A local browser,
extension, or agent disconnect can leave a Colab runtime running in the cloud.

## Legacy Playwright MCP Fallback

The earlier Playwright MCP guide is superseded. Use it only when:

- the Chrome skill is not installed, cannot bootstrap, or cannot claim/open tabs,
- the user explicitly requests the old Playwright MCP route, or
- a notebook-specific handoff requires reproducing an old Playwright incident.

If fallback is necessary, record the reason before acting. Do not broad-kill
local Chrome or Node processes. Only inspect or stop processes when you have
profile-specific evidence and the action is necessary for the chosen fallback.

## Run Review

After any live Colab attempt, success, failure, or stopped proof, write a short
run review. This is mandatory evidence hygiene; do not claim Colab success from
browser cleanup, local tests, notebook contract checks, or a killed proof run.

Use this structure:

```md
# Colab Chrome Control Run Review

## Scope
- Branch:
- Notebook URL:
- Goal: proof run or full run
- Claimed tab: title, URL, and whether it was reused or opened
- Surface: chrome:control-chrome or fallback, with fallback reason if any

## Runtime And Evidence
- Runtime accepted from matrix:
- Visible runtime/GPU evidence:
- In-notebook GPU gate evidence:
- Drive artifact root:
- Drive artifact evidence checked:

## Prompts And Control
- Prompts handled:
- Prompts that required user input:
- Cells or phases run:

## Notebook And Git State
- Cell freshness checked: yes/no
- Remote branch/commit evidence:

## Outcome
- Status: succeeded, failed, blocked, or stopped proof
- Failure taxonomy category, if any:
- Cleanup state: disconnected/deleted, auto-unassigned, still running, or handed off
- Residual risk:

## Recommended Follow-up
- ...
```

## Do Not Do

- Do not enter credentials or recovery information for the user.
- Do not inspect Chrome cookies, local storage, profiles, passwords, or session
  stores.
- Do not save notebook changes back to GitHub from Colab if any cell contains
  secrets.
- Do not start an expensive full training run when the user asked only to prove
  automation.
- Do not leave a proof run running after seeing the first training command.
- Do not treat a killed proof run as model-validation evidence.
- Do not stage, commit, or revert unrelated dirty worktree files while operating
  Colab.

## References

- `docs/NOTEBOOK_BEST_PRACTICES.md`
- `docs/DEFAULT_EXPERIMENT_RECIPE.md`
- `docs/handoffs/2026-05-22-colab-mcp-training-run.md`
