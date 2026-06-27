# Colab MCP Training Runbook Review 3

Fresh-agent rehearsal date: 2026-05-22

## Scope

This review tested the fresh-agent preflight in
`docs/handoffs/2026-05-22-colab-mcp-training-run.md` for the MCI-GRU Colab
Playwright MCP workflow. The goal was to resume or reopen
`notebooks/pit_repeated_seed_replication_colab.ipynb`, inspect Cell 5 before
execution, start Cell 13 only far enough to prove the first training job
launches, then immediately disconnect and delete the runtime.

The rehearsal did not reach Colab. It stopped at the local Playwright MCP
profile-lock preflight.

## What Worked

- The handoff and prior review gave a clear target notebook, branch, Colab URL,
  runtime type, cell order, and stop condition.
- The handoff's first required MCP call was followed exactly:
  `browser_tabs {"action":"list"}`.
- The handoff's Windows diagnostic command for profile locks was attempted
  exactly after the MCP profile-lock error appeared.
- No broad process kill was attempted.
- No notebook cells were run, no credentials were entered, and no API keys were
  inspected or repeated.

## Blocker

The first Playwright MCP call returned:

```text
Browser is already in use for C:\Users\magil\.codex\playwright-mcp-chrome-profile, use --isolated to run multiple instances of the same browser
```

The required process-identification command then failed:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*playwright-mcp-chrome-profile*' } | Select-Object ProcessId,Name,CommandLine
```

with:

```text
Get-CimInstance : Access denied
```

Because the process query was permission-denied, this agent could not identify
stale profile-specific MCP or Chrome processes safely. It therefore stopped the
browser attempt rather than using broad `chrome.exe` or `node.exe` cleanup.

## Old Session Resume

Old-session resume did not work. The agent could not list tabs or inspect the
previous Colab page because the shared Playwright MCP Chrome profile was locked
before any tab information was available.

## Cell 5 Status

Cell 5 was not inspected in Colab. The rehearsal never reached the notebook, so
this review cannot independently verify whether the live GitHub/Colab notebook
contains the fixed paths:

- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv`
- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json`

## Runtime Status

No new Colab runtime was started by this rehearsal. No training command was
launched. Since the agent never reached Colab, it could not verify the current
cloud runtime state through `Runtime > Disconnect and delete runtime > Yes`.

## Recommended Runbook Edit

Add a specific fallback for the permission-denied profile-lock diagnostic:

1. If `browser_tabs {"action":"list"}` fails with the shared-profile lock, run
   the documented `Get-CimInstance Win32_Process ... playwright-mcp-chrome-profile`
   command.
2. If that command returns `Access denied`, stop the rehearsal immediately and
   report a local process-inspection blocker.
3. Tell the next operator to restart Codex/MCP or manually close only
   Playwright MCP/Chrome processes that are visibly tied to
   `C:\Users\magil\.codex\playwright-mcp-chrome-profile`.
4. Do not try `--isolated` for the main proof unless the user is present for
   Google/Drive auth, because it likely loses the authenticated Colab session.
5. Do not use broad process kills, and do not treat closing local browser
   processes as proof that Colab cloud training stopped.

## Bottom Line

Iteration 3 reproduced the same fresh-agent takeover blocker, with an added
detail: in this environment the documented process-inspection command itself can
be permission-denied. The runbook should make that an explicit stop condition so
future agents do not drift into unsafe process cleanup or unrelated browser
routes.
