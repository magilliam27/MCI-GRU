# Colab MCP Training Runbook Review 6

Fresh-agent rehearsal date: 2026-05-22

## Scope

This review tested iteration 6 of the fresh-agent rehearsal for
`docs/handoffs/2026-05-22-colab-mcp-training-run.md`. The intended goal was to
resume an existing Colab tab if possible, verify that Cell 5 in the live GitHub
Colab notebook reflects commit `2119773`, run cells `3`, `5`, `7`, `9`, `11`,
then start Cell `13` only far enough to prove the first training job begins,
and immediately disconnect/delete the runtime.

The rehearsal stopped before reaching Colab because the first required
Playwright MCP preflight call failed during browser initialization.

## What Worked

- The runbook and prior reviews were read before browser work.
- The first Playwright MCP call followed the runbook exactly:
  `browser_tabs {"action":"list"}`.
- No notebook cells were run.
- No credentials, passwords, OTPs, passkeys, recovery details, or API keys were
  entered or exposed.
- No broad `chrome.exe` or `node.exe` process kill was attempted.
- A profile-specific process inspection was attempted afterward as a diagnostic,
  but not used for cleanup.

## Blocker

The first Playwright MCP call returned:

```text
Error: async initializeServer: Target page, context or browser has been closed
```

The browser log showed Playwright MCP launching Chrome with the shared profile:

```text
--user-data-dir=C:\Users\magil\.codex\playwright-mcp-chrome-profile --remote-debugging-pipe about:blank
```

It then immediately closed the launched process during initialization:

```text
<launched> pid=13696
<gracefully close start>
<kill>
taskkill stderr: ERROR: The process "13696" not found.
<process did exit: exitCode=0, signal=null>
```

This was not the previous explicit shared-profile lock message. It was a fresh
Playwright MCP initialization/target-closed failure before tab enumeration.

The follow-up profile-specific inspection command:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*playwright-mcp-chrome-profile*' } | Select-Object ProcessId,Name,CommandLine
```

failed from the normal shell with:

```text
Get-CimInstance : Access denied
```

## Old Session Resume

Old-session resume did not occur. The tab list could not be retrieved because
Playwright MCP failed before returning any browser tabs.

## Cell 5 Status

Cell 5 was not inspected in Colab. This rehearsal cannot independently verify
whether the live GitHub Colab page reflects commit `2119773` or whether it
contains the fixed paths:

- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv`
- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json`

No stale Cell 5 behavior was observed in this iteration because Colab was never
reached.

## Runtime Status

No Colab runtime was connected, created, or deleted by this rehearsal. Cell 13
was not reached, and no training command was launched.

Because the browser failed before tab enumeration, this iteration could not
verify Colab's cloud runtime state through
`Runtime > Disconnect and delete runtime > Yes`.

## Recommended Runbook Edit

Add a distinct stop condition for the non-lock Playwright MCP initialization
failure:

1. If `browser_tabs {"action":"list"}` fails with
   `Target page, context or browser has been closed`, stop the rehearsal before
   trying notebook steps.
2. Record the Chrome launch line and the launched PID/close sequence from the
   MCP error.
3. Run only the profile-specific process inspection command if needed for
   diagnostics. If normal `Get-CimInstance` returns `Access denied`, record that
   explicitly and leave cleanup to the main thread.
4. Do not retry with `--isolated` for the main proof unless the user is present
   for Google/Drive auth, because it likely loses the authenticated Colab
   session.
5. Have the main thread restart or reload the Playwright MCP server/profile
   after a target-closed initialization failure, then verify a clean
   `browser_tabs {"action":"list"}` before spawning the next fresh agent.

## Bottom Line

Iteration 6 did not test commit `2119773` in Colab. The shared profile no longer
reported the previous explicit ownership lock, but Playwright MCP failed during
initialization with a target/browser-closed error before tabs could be listed.
The next rehearsal should begin only after the main thread reloads or restarts
the Playwright MCP server/profile and confirms tab listing works.
