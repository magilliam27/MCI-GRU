# Colab MCP Training Runbook Review 5

Fresh-agent rehearsal date: 2026-05-22

## Scope

This review tested iteration 5 of the fresh-agent rehearsal for
`docs/handoffs/2026-05-22-colab-mcp-training-run.md`. The intended goal was to
resume an existing Colab tab if possible, verify that Cell 5 in the live GitHub
Colab notebook contains the hardwired PIT input paths from commit `2119773`,
run cells `3`, `5`, `7`, `9`, `11`, then start Cell `13` only far enough to
prove the first training job begins, and immediately disconnect/delete the
runtime.

The rehearsal stopped before reaching Colab because the first required
Playwright MCP preflight call hit the shared Chrome profile lock.

## What Worked

- The runbook and prior reviews were read before browser work.
- The first Playwright MCP call followed the runbook exactly:
  `browser_tabs {"action":"list"}`.
- No notebook cells were run.
- No credentials, passwords, OTPs, passkeys, recovery details, or API keys were
  entered or exposed.
- No broad `chrome.exe` or `node.exe` process kill was attempted.
- The profile-lock diagnostic was run exactly, first normally and then with
  elevation after normal `Get-CimInstance` returned `Access denied`.

## Blocker

The first Playwright MCP call returned:

```text
Error: Browser is already in use for C:\Users\magil\.codex\playwright-mcp-chrome-profile, use --isolated to run multiple instances of the same browser
```

The normal profile-lock diagnostic command:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*playwright-mcp-chrome-profile*' } | Select-Object ProcessId,Name,CommandLine
```

failed with:

```text
Get-CimInstance : Access denied
```

The same profile-specific inspection succeeded with elevation and showed
multiple processes using `C:\Users\magil\.codex\playwright-mcp-chrome-profile`,
including MCP parent processes:

```text
45068 node.exe   ...\playwright-mcp\node_modules\@playwright\mcp\cli.js --browser=chrome ... --user-data-dir=C:\Users\magil\.codex\playwright-mcp-chrome-profile
37668 node.exe   ...\playwright-mcp\node_modules\@playwright\mcp\cli.js --browser=chrome ... --user-data-dir=C:\Users\magil\.codex\playwright-mcp-chrome-profile
29072 node.exe   ...\playwright-mcp\node_modules\@playwright\mcp\cli.js --browser=chrome ... --user-data-dir=C:\Users\magil\.codex\playwright-mcp-chrome-profile
31872 node.exe   ...\playwright-mcp\node_modules\@playwright\mcp\cli.js --browser=chrome ... --user-data-dir=C:\Users\magil\.codex\playwright-mcp-chrome-profile
30140 chrome.exe ... --user-data-dir=C:\Users\magil\.codex\playwright-mcp-chrome-profile --remote-debugging-pipe about:blank
```

Additional Chrome child processes for GPU, network, storage, renderer, video,
and audio services also referenced the same profile.

Per the iteration-5 instruction, this was recorded as a profile-lock blocker
instead of attempting process cleanup.

## Old Session Resume

Old-session resume did not occur. The tab list could not be retrieved because
the shared Playwright MCP Chrome profile was already in use before tab
enumeration completed.

## Cell 5 Status

Cell 5 was not inspected in Colab. This rehearsal cannot independently verify
whether the live GitHub Colab page now reflects commit `2119773` or whether it
contains the fixed paths:

- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv`
- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
- `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json`

No stale Cell 5 behavior was observed in this iteration because Colab was never
reached.

## Runtime Status

No Colab runtime was connected, created, or deleted by this rehearsal. Cell 13
was not reached, and no training command was launched.

Because the browser profile was locked before a tab could be listed, this
iteration could not verify Colab's cloud runtime state through
`Runtime > Disconnect and delete runtime > Yes`.

## Recommended Runbook Edit

Add a hard stop condition before spawning the next fresh-agent rehearsal:

1. If `browser_tabs {"action":"list"}` returns the shared-profile lock, run the
   profile-specific `Get-CimInstance` inspection.
2. If that inspection requires elevation, use elevation only for inspection and
   record the result.
3. If multiple MCP parent `node.exe` processes are already using
   `playwright-mcp-chrome-profile`, stop the rehearsal and clear/restart the
   Playwright MCP profile from the main thread before spawning another fresh
   agent.
4. Do not make each fresh agent independently kill processes. If cleanup is
   approved, it should target only the exact MCP parent processes and Chrome
   children whose command lines include
   `C:\Users\magil\.codex\playwright-mcp-chrome-profile`, followed by a fresh
   MCP/Codex reload and a successful `browser_tabs {"action":"list"}` check.
5. Do not use `--isolated` for the main proof unless the user is present for
   Google/Drive auth, because it likely loses the authenticated Colab session.

## Bottom Line

Iteration 5 did not test commit `2119773` in Colab. The runbook preflight
stopped at a Playwright MCP shared-profile lock, with four MCP parent
`node.exe` processes and Chrome children still attached to
`playwright-mcp-chrome-profile`. The next rehearsal should begin only after the
main thread clears or restarts that profile and verifies tab listing works.
