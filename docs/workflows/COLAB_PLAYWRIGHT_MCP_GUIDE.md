# Colab Playwright MCP Agent Guide

Last updated: 2026-05-26

Use this guide when an agent needs to operate a Google Colab notebook through
Playwright MCP from the logged-in Windows/Codex environment. It is intended for
MCI-GRU training notebooks where the agent should mount Drive, select the right
runtime, start a model run, monitor the first outputs, and stop cleanly when the
goal is only to prove automation.

This is a reusable workflow doc. Notebook-specific handoffs can point here and
then list only the branch, Colab URL, cell order, expected outputs, and artifact
paths for that run.

## Default Scope

- Use Playwright MCP when the bundled Chrome plugin or native bridge is not
  usable.
- Prefer the existing authenticated MCP Chrome profile so Google and Drive auth
  can be reused.
- Click through routine notebook, Drive, and runtime prompts that are directly
  required for the user-requested run.
- Stop and ask the user before entering passwords, OTPs, passkeys, account
  recovery details, or unrelated Google security settings.
- Never paste or repeat API keys, account credentials, tokens, or private
  secrets in docs, reviews, screenshots, or final messages.
- Closing Chrome, killing MCP, or ending the local agent is not proof that a
  Colab cloud runtime stopped. Only a Colab-side disconnected/deleted runtime
  state counts.

## Tool Discovery

If Playwright MCP tools are not already available, use tool discovery first:

```text
tool_search query: playwright browser snapshot tab click navigate
```

The exact tool names can vary by MCP exposure, but the required capabilities
are:

- list browser tabs
- navigate or open a URL
- inspect the accessibility snapshot
- click visible controls
- type into focused fields when necessary
- wait for output or UI changes

Start every fresh browser attempt with a tab-list call, not with blind
navigation.

```text
browser_tabs {"action":"list"}
```

## Preflight Checklist

1. Read the notebook-specific handoff or issue first.
2. Record the target branch and Colab URL.
3. Confirm whether this is a proof run or a full training run.
4. Confirm the runtime target. Current MCI-GRU training runs should use
   `G4 GPU`; do not accept `T4` for full current-preset runs unless the user
   explicitly asks for a cheaper proof.
5. Check whether the notebook must be fresh from GitHub. Colab GitHub URLs read
   the remote branch, not local unpushed edits.
6. Run `browser_tabs {"action":"list"}`.
7. Reuse an existing Colab tab if it is on the right notebook and branch.
8. If no suitable tab exists, navigate to the notebook-specific Colab URL.

For the current repeated-seed PIT proof notebook, the known-good URL is:

```text
https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_repeated_seed_replication_colab.ipynb
```

The hardwired PIT input commit for that notebook is:

```text
2119773 Hardwire repeated-seed Colab inputs
```

## Runtime Setup

Use the Colab UI, not local assumptions:

1. Open `Runtime`.
2. Choose `Change runtime type`.
3. Select `G4 GPU`.
4. Save.
5. Confirm the toolbar or runtime status shows `G4`, not `T4`, before running
   setup or training cells.

For the MCI-GRU proof run, Colab reported `G4 (Python 3)` and the setup cell saw
CUDA with an `NVIDIA RTX PRO 6000 Blackwell Server Edition` GPU. Treat the exact
GPU model as evidence for that run, not as a permanent requirement.

If a notebook opens already connected to `T4`, stop before rerunning cells and
switch the runtime to `G4 GPU`. Runtime changes can disconnect the notebook, so
reconnect and rerun setup from the top after saving the runtime change.

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

If the notebook needs FRED access, prefer an existing Colab secret or notebook
runtime configuration named `FRED_API_KEY`. Do not copy the secret into a guide
or review.

## Notebook Freshness

Before running expensive cells, inspect the setup and data-input cells in the
live Colab UI.

For the repeated-seed PIT notebook, Cell 5 must use explicit paths and must not
perform recursive Drive discovery. The expected paths are:

```text
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json
```

If the live Colab cell still has blank paths or recursive `root.glob(...)`
searches, stop. The notebook is probably stale relative to the branch or
commit. Do not let it hang on Drive discovery.

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

For the repeated-seed PIT notebook, the proven cell order was:

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

## Stopping A Proof Run

When the goal is to prove automation rather than finish training:

1. Use Colab: `Runtime > Disconnect and delete runtime`.
2. Confirm the deletion prompt with `Yes`.
3. Keep watching until the UI shows a disconnected state.
4. Record the visible status, such as `Reconnect G4 High-RAM Click to connect`.

Do not rely on local process cleanup as a stop condition. A local browser crash
or MCP disconnect can leave a Colab runtime running in the cloud.

## Profile Lock Recovery

If the first tab-list call fails with a profile ownership error like:

```text
Browser is already in use for C:\Users\magil\.codex\playwright-mcp-chrome-profile, use --isolated to run multiple instances of the same browser
```

do not run notebook steps. Inspect only the profile-specific owners:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*playwright-mcp-chrome-profile*' } |
  Select-Object ProcessId,Name,CommandLine
```

If normal shell access is denied, record that and request or use an approved
elevated inspection. Do not broad-kill all `chrome.exe` or `node.exe` processes.

When stale owners are clearly tied to the MCP profile, stop only those processes:

```powershell
$profilePattern='*playwright-mcp-chrome-profile*'
$targets = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'node.exe' -or $_.Name -eq 'chrome.exe') -and
    $_.CommandLine -like $profilePattern
}
$targets | Select-Object ProcessId,Name
$targets | ForEach-Object {
    try {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
    } catch {
        Write-Output "Already exited: $($_.ProcessId) $($_.Name)"
    }
}
Start-Sleep -Seconds 2
Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'node.exe' -or $_.Name -eq 'chrome.exe') -and
    $_.CommandLine -like $profilePattern
} | Select-Object ProcessId,Name,CommandLine
```

After cleanup, retry from a fresh agent or fresh MCP session if the current MCP
transport closes.

`--isolated` can avoid a profile lock, but it usually loses the logged-in
Google/Drive state. Use it only as a diagnostic unless the user is present for
auth.

## Target-Closed Errors

There are two different `Target page, context or browser has been closed`
patterns.

Fatal initialization failure:

- Happens on the first `browser_tabs {"action":"list"}` call.
- No tab list is returned.
- The error may mention `async initializeServer`.
- Treat this as an MCP/browser startup failure, not a notebook failure.
- Stop the attempt, inspect for profile-specific `node.exe` or `chrome.exe`
  owners, clean only those owners if appropriate, then retry in a fresh session.

Recoverable OAuth-tab close:

- Happens after clicking Google OAuth `Continue`.
- The OAuth popup closes itself and Playwright reports a closed target.
- Run `browser_tabs {"action":"list"}` again.
- If the Colab tab is still present, continue from the notebook.

Record which case occurred. Do not conflate them in reviews.

## Review Template

When a rehearsal fails or a subagent tests the guide, write a short review with
this structure:

```md
# Colab Playwright MCP Run Review

## Scope
- Notebook URL:
- Branch:
- Goal: proof run or full run

## What Worked
- ...

## Blocker
- Exact tool/UI error:
- Where it occurred:

## Notebook State
- Cell freshness checked: yes/no
- Runtime selected:
- Cells run:

## Runtime State
- Connected/disconnected/deleted:
- Evidence:

## Recommended Guide Update
- ...
```

## Do Not Do

- Do not enter credentials or recovery information for the user.
- Do not save notebook changes back to GitHub from Colab if any cell contains
  secrets.
- Do not start an expensive full training run when the user asked only to prove
  automation.
- Do not leave a proof run running after seeing the first training command.
- Do not treat a killed proof run as model-validation evidence.
- Do not broadly kill local Chrome or Node processes without profile-specific
  evidence.
- Do not stage, commit, or revert unrelated dirty worktree files while operating
  Colab.

## References

- `docs/NOTEBOOK_BEST_PRACTICES.md`
- `docs/DEFAULT_EXPERIMENT_RECIPE.md`
- `docs/handoffs/2026-05-22-colab-mcp-training-run.md`
