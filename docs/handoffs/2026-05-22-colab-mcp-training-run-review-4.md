# Colab MCP Training Runbook Review 4

Fresh-agent rehearsal date: 2026-05-22

## Scope

This review tested iteration 4 of the fresh-agent rehearsal for
`docs/handoffs/2026-05-22-colab-mcp-training-run.md`. The goal was to resume an
existing Colab tab if present, inspect the PIT data input cell before execution,
start Cell 13 only far enough to prove the first training job begins, then
disconnect and delete the runtime.

The rehearsal intentionally stopped before executing any notebook cells because
the live Colab notebook loaded from GitHub still had the stale Cell 5 path
discovery logic.

## What Worked

- The first Playwright MCP call followed the runbook exactly:
  `browser_tabs {"action":"list"}`.
- The shared profile lock was cleared. The first MCP call succeeded instead of
  returning the prior `playwright-mcp-chrome-profile` ownership error.
- The tab list contained only `about:blank`, so there was no useful old Colab
  session to resume.
- The handoff Colab URL opened successfully:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_repeated_seed_replication_colab.ipynb`
- The page showed the expected signed-in Google account and a `Connect` button,
  with no active runtime connected before any cell execution.

## Blocker

The notebook loaded from the GitHub Colab URL appears stale relative to the
handoff. The Cell 5 source under `2. PIT Data Inputs` still contains blank path
variables and recursive Drive discovery:

```python
# Leave these blank to auto-discover by filename under common Drive folders.
MARKET_CSV_PATH = ''
PIT_UNIVERSE_CSV_PATH = ''
MARKET_META_JSON_PATH = ''
DRIVE_SEARCH_ROOTS = [
...
candidates.extend(root.glob(f'**/{filename}') if root.exists() else [])
```

This violates the handoff requirement that Cell 5 contain hardwired paths such
as:

```python
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv
/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json
```

Per the runbook, the rehearsal stopped here and did not run Cell 5.

## Old Session Resume

Old-session resume status: no useful old session existed. The tab list returned
only the current `about:blank` tab, so the rehearsal navigated to the Colab URL
from the handoff.

## Cell 5 Status

Cell 5 did not have hardwired paths. It relied on blank path variables,
`DRIVE_SEARCH_ROOTS`, and recursive `root.glob(f'**/{filename}')` discovery.

No notebook cells were executed.

## Runtime Status

No training job was started. Cell 13 was not reached because Cell 5 failed the
source inspection gate.

No Colab runtime was connected or created by this rehearsal. The page status
before stopping showed `Connect` / `Connect to a new runtime`, so there was no
runtime from this rehearsal to disconnect or delete.

## Recommended Runbook Edit

Add an explicit stale-remote branch recovery step before future rehearsals:

1. If Cell 5 inspection shows blank paths or recursive Drive discovery, stop as
   this rehearsal did.
2. Treat that as evidence that the GitHub branch backing the Colab URL has not
   received the hardwired notebook update, even if the local notebook/generator
   are patched.
3. Before spawning the next fresh agent, update or push
   `notebooks/pit_repeated_seed_replication_colab.ipynb` on
   `codex/pit-universe-validation`, then reopen the same GitHub Colab URL and
   re-check Cell 5.
4. Do not run Cell 5 from a stale notebook, because it may recurse through
   Drive and hang before the intended training proof.

## Bottom Line

Iteration 4 proved the Playwright MCP profile-lock cleanup worked, but the
remote Colab notebook still served the stale Cell 5 implementation. The next
attempt should refresh the notebook on the remote branch or otherwise ensure
the Colab URL loads the hardwired Cell 5 before any runtime work begins.
