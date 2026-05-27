# Colab MCP Training Runbook Review

Fresh-agent rehearsal date: 2026-05-22

## Scope

This review tested whether a fresh agent could follow
`docs/handoffs/2026-05-22-colab-mcp-training-run.md` far enough to resume or
reopen the Colab notebook and start the documented training-launch path without
allowing an expensive full training run to proceed.

The rehearsal did not reach Colab execution. The blocker was local Playwright
MCP process/profile ownership before the notebook could be reopened.

## What Worked

- The handoff gave the correct target notebook and URL:
  `notebooks/pit_repeated_seed_replication_colab.ipynb` on
  `codex/pit-universe-validation`.
- The handoff clearly stated the intended runtime (`G4 GPU`), cell order
  (`3`, `5`, `7`, `9`, `11`, `13`), and stop condition after seeing Cell 13
  start the first training job.
- The handoff correctly warned that the worktree is dirty and that the prior
  proof run tag `20260522_014721` is not model evidence.
- The handoff correctly identified Playwright MCP as the preferred route over
  the bundled Chrome plugin.
- The expected Google/Colab prompts and credentials boundaries were clear
  enough: notebook/Drive/runtime prompts can be clicked, but passwords, OTPs,
  passkeys, recovery prompts, and API keys must not be handled by the agent.

## What Did Not Work

- A fresh Playwright MCP call failed immediately with:
  `Browser is already in use for C:\Users\magil\.codex\playwright-mcp-chrome-profile, use --isolated to run multiple instances of the same browser`.
- The old session could not be resumed through the newly exposed MCP tools. The
  shared profile was still locked by existing `@playwright/mcp` node processes
  plus a Chrome instance using
  `C:\Users\magil\.codex\playwright-mcp-chrome-profile`.
- After closing only the stale Playwright MCP profile processes, the current
  tool connection reported:
  `Transport closed`.
- Re-running `tool_search` rediscovered the Playwright tool names, but the
  actual `mcp__playwright__` transport remained closed. Browser navigation and
  tab listing still failed with `Transport closed`.
- Because the tool transport was closed, the fresh agent did not get far enough
  to open Colab, inspect tabs, select `G4 GPU`, rerun notebook cells, or verify
  Cell 13 launch output.

## Old Session Resume

Old session resume did not work.

The handoff says the browser session was left on the notebook page with a
`Reconnect G4 High-RAM` button, but the fresh agent could not inspect that tab
because the shared Playwright profile was already owned by prior MCP processes.
The profile did retain local browser state, but the runbook does not yet explain
how a later agent should take over that profile when another MCP server still
owns it.

## Runtime Kill Status

No new Colab runtime was started by this rehearsal, so there was no new runtime
to kill from Colab.

The local stale Playwright-controlled Chrome processes were closed to free the
profile, but that is not equivalent to `Runtime > Disconnect and delete runtime`
inside Colab. The review did not verify any cloud runtime state after the MCP
transport closed.

## Ambiguous Or Incomplete Instructions

- The runbook should add a "fresh-agent profile lock" section before the browser
  steps. Multiple old `@playwright/mcp` servers can keep the shared Chrome
  profile locked even when the prior Colab runtime was killed.
- The runbook should state that killing the MCP server backing the current tool
  can close the tool transport for the agent. A fresh agent should not assume
  that stopping stale `node.exe` processes will automatically restart MCP tools
  in the same turn.
- The runbook should document the recommended recovery sequence when the shared
  profile is locked:
  1. Try `browser_tabs` first.
  2. If it reports the profile is already in use, inspect for stale MCP node and
     Chrome processes using the profile path.
  3. Close only those profile-specific processes.
  4. Restart the Codex thread/session or otherwise reload MCP tools before
     calling Playwright again.
- The runbook should decide whether fresh-agent tests should use the shared
  authenticated profile or an isolated profile. `--isolated` may avoid the lock
  but likely loses the logged-in Colab/Drive session, so it is not a drop-in
  replacement for this workflow.
- The runbook should separate "local browser closed" from "Colab runtime
  deleted." Closing Chrome is not proof that cloud training stopped.

## Recommended Edits To The Handoff

- Add a preflight check:
  `browser_tabs {"action":"list"}` should be the first MCP call. If it fails
  with the shared-profile lock, do not proceed to notebook instructions yet.
- Add a Windows diagnostic command for profile locks:
  `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*playwright-mcp-chrome-profile*' }`
- Add a caution that stopping the current MCP node process can invalidate the
  current tool transport. The agent should be prepared to restart the Codex
  session after freeing the profile.
- Add a handoff status field after each browser run:
  `MCP profile state: open/closed/unknown`, `Colab runtime state:
  disconnected/deleted/running/unknown`, and `last verified tab`.
- Add an explicit stop rule for failed fresh-agent rehearsals:
  if Colab cannot be reached because MCP transport is closed, write the review
  and do not attempt credential, Drive, or notebook work through an unrelated
  browser path.

## Bottom Line

The runbook is strong once Playwright MCP has control of the browser, but it is
missing the takeover story for fresh agents. The main gap is process ownership:
the shared authenticated Chrome profile is useful, but a previous MCP server can
lock it and prevent the next agent from even listing tabs. The next revision
should make profile-lock detection and MCP restart/reload an explicit preflight.
