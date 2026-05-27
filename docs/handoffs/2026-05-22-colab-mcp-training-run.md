# Colab MCP Training Run Handoff

Last updated: 2026-05-22

## Resume Here

- The Playwright MCP route works for authenticated Colab automation. Use it instead of the bundled Chrome plugin if the native bridge error returns.
- The latest proved notebook path is `notebooks/pit_repeated_seed_replication_colab.ipynb`.
- The GitHub Colab URL used was:
  `https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_repeated_seed_replication_colab.ipynb`
- The last training launch was intentionally killed after proving the method. Do not treat run tag `20260522_014721` as model evidence.
- Immediate next move, if the user wants another run: reconnect the Colab notebook, set runtime to `G4 GPU`, rerun cells `3`, `5`, `7`, `9`, `11`, then `13`, and monitor the first training job.
- The PIT data input cell is now fixed-path only. It should not recurse through Drive; if Cell 5 hangs again, treat that as a regression or stale GitHub notebook rather than normal discovery time.

## Current Objective

The user wants an agent-controlled way to start and babysit full MCI-GRU Colab training notebooks. The immediate proof point was to use Playwright MCP to authenticate, mount Drive, select a G4 GPU runtime, validate PIT data inputs, build the job matrix, and start the training launcher.

## What Changed

- No production code was intentionally changed for this automation proof.
- This handoff file was added to preserve the Colab MCP runbook.
- After iteration 2, the notebook generator and notebook were changed so Cell 5 hardwires the three PIT input paths and avoids recursive Drive discovery. The FRED runtime value was also embedded at the user's request.
- Iteration 3 still failed before Colab because the shared Playwright profile was locked. The main thread then used elevated process inspection and stopped only profile-specific MCP/Chrome processes. A follow-up query showed no remaining `node.exe` or `chrome.exe` using `playwright-mcp-chrome-profile`.
- The local notebook/generator are patched, but the GitHub Colab URL reads from the remote branch. A fresh agent must inspect Cell 5 before running; if it sees blank paths or recursive Drive globbing, the Colab copy is stale and should not be allowed to hang.
- Commit `2119773` (`Hardwire repeated-seed Colab inputs`) was pushed to `origin/codex/pit-universe-validation`, so the GitHub Colab URL should now serve the fixed Cell 5.
- Iteration 5 again stopped at the shared-profile lock. The main thread cleared only profile-specific MCP/Chrome processes and verified no remaining `node.exe` or `chrome.exe` used `playwright-mcp-chrome-profile` before spawning iteration 6.
- Iteration 6 failed with `async initializeServer: Target page, context or browser has been closed`. It left profile-specific Playwright MCP `node.exe` processes but no Chrome children. The main thread stopped those nodes and verified no remaining profile owners before spawning iteration 7.
- Iteration 7 succeeded end-to-end for the proof loop. A fresh subagent opened the GitHub Colab URL, verified Cell 5 had the fixed Drive paths, selected `G4 GPU`, ran cells `3`, `5`, `7`, `9`, `11`, started Cell `13`, saw `Training: pit_seed_314159_replication_2022` and `Attempt: 1/3`, then used `Runtime > Disconnect and delete runtime > Yes` and verified the toolbar returned to `Reconnect G4 High-RAM Click to connect`.
- The Colab runtime was terminated with `Runtime > Disconnect and delete runtime > Yes`.
- The browser session was left on the notebook page with a `Reconnect G4 High-RAM Click to connect` button, indicating no active runtime execution from this session.

## Key Decisions

- Use Playwright MCP for Colab automation. It survived the Colab, Google OAuth, Drive mount, runtime selection, and training-launch flow.
- Avoid relying on the bundled Chrome plugin until its native bridge trust issue is resolved. Earlier attempts hit `privileged native pipe bridge is not available; browser-client is not trusted`.
- Use `G4 GPU` for full runs. Colab reported `G4 (Python 3)` and the setup cell saw CUDA with `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- Click through only prompts directly related to this notebook, Colab runtime, or Drive mount. Pause for passwords, OTP, passkeys, account recovery, or security-setting changes.
- Do not save notebook changes back to GitHub from Colab if any cell contains pasted credentials. Prefer a Colab Secret named `FRED_API_KEY`.

## Fresh-Agent Playwright Preflight

- First MCP call should be:
  `browser_tabs {"action":"list"}`.
- If it returns usable tabs, resume the visible Colab tab when present. If no Colab tab exists, navigate to the GitHub Colab URL in `Resume Here`.
- If it fails with profile ownership such as:
  `Browser is already in use for C:\Users\magil\.codex\playwright-mcp-chrome-profile, use --isolated to run multiple instances of the same browser`,
  do not jump into notebook steps yet.
- Try to identify profile-locking processes with:
  `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*playwright-mcp-chrome-profile*' } | Select-Object ProcessId,Name,CommandLine`
- If that command is permission-denied, record that explicitly. Do not kill broad `chrome.exe` or `node.exe` groups without process-specific evidence.
- If the normal shell gets `Access denied`, retry the same command with elevated permissions. In this setup, elevated `Get-CimInstance` successfully exposed the profile-specific `node.exe` and `chrome.exe` processes.
- If you can identify stale processes that specifically use `playwright-mcp-chrome-profile`, close only those profile-specific processes.
- The cleanup command that worked from the main thread was:
  ```powershell
  $profilePattern='*playwright-mcp-chrome-profile*'
  $targets = Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'node.exe' -or $_.Name -eq 'chrome.exe') -and $_.CommandLine -like $profilePattern }
  $targets | Select-Object ProcessId,Name
  $targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Sleep -Seconds 2
  Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'node.exe' -or $_.Name -eq 'chrome.exe') -and $_.CommandLine -like $profilePattern } | Select-Object ProcessId,Name,CommandLine
  ```
- Some Chrome child PIDs may exit after the browser parent is stopped and print `Cannot find a process with the process identifier ...`; treat that as harmless if the final verification query returns no profile-specific processes.
- After closing MCP/Chrome processes, expect the current `mcp__playwright__` transport may report `Transport closed`. In that case, stop the browser attempt, write the review, and ask for a fresh Codex/MCP reload rather than trying unrelated browser paths.
- If the first `browser_tabs` call fails with `async initializeServer: Target page, context or browser has been closed`, treat it as a separate MCP initialization failure. Do not run notebook steps. The main thread should inspect for profile-specific `node.exe`/`chrome.exe` processes, close them, verify none remain, and spawn a new rehearsal.
- After a target-closed initialization failure, profile-specific `node.exe` processes can remain even when Chrome is gone. They still need cleanup before the next fresh-agent rehearsal.
- OAuth windows can also produce a one-off `Target page, context or browser has been closed` after clicking `Continue`, because the OAuth tab closes itself. That is recoverable when a follow-up `browser_tabs {"action":"list"}` returns to the Colab tab. Do not confuse this with the fatal preflight initialization failure above.
- `--isolated` can avoid the profile lock but probably loses the logged-in Google/Drive session. Treat it as a diagnostic only unless the user is present for auth.
- Closing local Chrome or an MCP server is not proof that Colab cloud training stopped. Only `Runtime > Disconnect and delete runtime > Yes` or a verified disconnected runtime state counts.

## Important Files

- `notebooks/pit_repeated_seed_replication_colab.ipynb`: full PIT repeated-seed Colab notebook used for the proof.
- `scripts/gen_pit_repeated_seed_replication_nb.py`: generator for the notebook; edit this first if changing notebook behavior.
- `tests/test_pit_repeated_seed_replication_notebook.py`: notebook contract tests.
- `docs/DEFAULT_EXPERIMENT_RECIPE.md`: frozen recipe the full notebook is expected to run.
- `docs/handoffs/2026-05-20-pit-repeated-seed-option-a-notebook.md`: prior notebook-specific handoff with Option A context.

## Verification

- Manual browser verification through Playwright MCP:
  - Opened the GitHub Colab notebook.
  - Signed into the existing Google account in the MCP Chrome profile.
  - Accepted the non-Google-authored notebook warning.
  - Mounted Google Drive through the Colab Drive prompt and Google OAuth prompts.
  - Changed runtime type to `G4 GPU`.
  - Verified Colab status showed `G4 (Python 3)`.
  - Ran the setup cell and saw:
    - `CUDA available: True`
    - GPU visible as `NVIDIA RTX PRO 6000 Blackwell Server Edition`
    - PIT `DataConfig` fields available: `pit_breadth_policy`, `pit_min_scoreable_stocks`, `pit_universe_mode`
  - Ran the PIT data input cell and saw:
    - Market CSV rows: `1,849,404`
    - Market stocks: `759`
    - Market dates: `2015-01-02` to `2026-05-13`
    - Market CSV SHA256: `84e1f3f2b79a798246e001e17a372c8daf8bcfc658873ad3d352a99ad993840f`
    - PIT universe rows: `879`
    - PIT columns: `kdcode`, `valid_from`, `valid_to`
  - Ran the job-matrix cell and saw:
    - PIT presets written for 2022, 2023, 2024, 2025
    - Static regime inputs drawn and validated
    - Static regime SHA256: `78cd4cfcbc9699f36a6f8fce657c52c2fbd615151e7b9358a858373c3ae1aba5`
    - Training jobs: `4`
  - Ran the training launcher cell and saw the first job start:
    - `Training: pit_seed_314159_replication_2022`
    - `Attempt: 1/3`
    - Command invoked `/usr/bin/python3 -u /content/MCI-GRU/run_experiment.py`
    - Full settings included `training.num_models=20`, `training.num_epochs=100`, `training.loss_type=ic`, `training.selection_metric=val_ic`, `data.pit_universe_mode=masked_panel`
  - Terminated the runtime after proving the method.
- Post-proof local notebook hardwire verification:
  - Regenerated `notebooks/pit_repeated_seed_replication_colab.ipynb` from `scripts/gen_pit_repeated_seed_replication_nb.py`.
  - Ran `.\.venv\Scripts\python.exe -m pytest tests\test_pit_repeated_seed_replication_notebook.py -v --basetemp=.codex_tmp\pytest_pit_hardwired_inputs_2`.
  - Result: `10 passed`; pytest emitted a cache warning because `.pytest_cache` was permission-denied.
  - Ran `.\.venv\Scripts\ruff.exe check scripts\gen_pit_repeated_seed_replication_nb.py tests\test_pit_repeated_seed_replication_notebook.py`.
  - Result: all checks passed.
- Local evidence pass:
  - Ran `git status --short`.
  - The worktree was already dirty with unrelated modified/untracked files before this handoff. Do not stage or revert them casually.
- Not run:
  - No local pytest or ruff checks were run for this handoff-only change.
  - No full Colab training run was allowed to complete.

## Open Risks

- The partial Colab run tag `20260522_014721` may have small Drive artifacts such as manifest or static regime input files, but it is not a complete run.
- The Playwright MCP Chrome profile is separate from the user's normal Chrome profile. It may need Google login again in a future Codex session.
- Colab can show follow-up prompts after a run starts. Keep watching for Drive/runtime/security prompts instead of assuming the notebook is unattended-safe.
- Full training is expensive: 4 yearly jobs x 20 models x 100 epochs for the current one-seed notebook state.
- The local repo is dirty. Some changes predate this handoff; preserve them unless the user explicitly asks for cleanup.

## Next Actions

1. Use `tool_search` for Playwright tools if `mcp__playwright__` is not already loaded:
   `playwright browser snapshot tab click navigate`.
2. Run the `Fresh-Agent Playwright Preflight` above before opening or resuming Colab.
3. Navigate to the Colab URL above and confirm the notebook is on branch `codex/pit-universe-validation`.
4. Set runtime:
   `Runtime > Change runtime type > G4 GPU > Save`.
5. If prompted, approve only notebook/Drive/runtime prompts:
   - `Run anyway`
   - `Connect to Google Drive`
   - Google OAuth `Continue` for Drive access
6. Run cells in this order:
   - Cell 3 code cell: setup, Drive mount, clone, install, GPU check
   - Cell 5 code cell: PIT data inputs; should use these fixed paths and complete without Drive globbing:
     - `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.csv`
     - `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
     - `/content/drive/MyDrive/MCI_GRU_shared/data/sp500_pit_union_lseg_20150101_20260513.meta.json`
   - Cell 7 code cell: FRED key and run configuration
   - Cell 9 code cell: resumable job matrix and static regime inputs
   - Cell 11 code cell: training/backtest helpers
   - Cell 13 code cell: execute training, breadth checks, and backtests
7. After Cell 13 starts, verify the first output includes:
   `Training: pit_seed_314159_replication_2022` and `Attempt: 1/3`.
8. Immediately stop the proof run unless the user asked for a full run:
   `Runtime > Disconnect and delete runtime > Yes`.
9. Keep monitoring until the runtime status is disconnected/deleted, or record why that could not be verified.

## Iteration Goal

- Spawn a fresh subagent for each rehearsal.
- Each subagent should try the handoff from a fresh-agent perspective and either:
  - start Cell 13, verify the first training command, and delete the runtime, or
  - stop at the first concrete blocker and write a review.
- After every failed rehearsal, update this handoff with the missing instruction before spawning the next subagent.
- Only return to the user for auth/security blockers such as password, OTP, passkey, account recovery, or missing Colab secret/API access.

## Commands Run

- `git status --short`
- `Get-ChildItem -Path docs -Directory | Select-Object -ExpandProperty Name`
- `Get-ChildItem -Path notebooks -Filter '*colab*.ipynb' | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name,LastWriteTime`

## Data/Experiment State

- Runtime was killed on purpose after method proof.
- Last Colab run tag observed: `20260522_014721`.
- Drive output root from notebook: `/content/drive/MyDrive/MCI-GRU-Ablations`.
- Local Colab run root before runtime deletion: `/content/mci_gru_work/pit_repeated_seed_replication/20260522_014721`.
- Staged Colab data paths observed:
  - `/content/MCI-GRU/data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv`
  - `/content/MCI-GRU/data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`
  - `/content/MCI-GRU/data/raw/market/sp500_pit_union_lseg_20150101_20260513.meta.json`
- Cell 5 should no longer run recursive `root.glob(...)` searches. It resolves only the fixed paths above, with local repo fallbacks outside Colab.

## User Preferences

- In Colab/browser babysitting flows, keep watching long-running workflows for follow-up modals or prompts.
- If the user approves Drive access for this notebook, proceed through expected Drive/runtime prompts without stopping each time.
- Give practical runbook details: notebook URL, branch, cell order, runtime type, expected outputs, and artifact paths.

## Do Not Do

- Do not paste or repeat API keys or account credentials.
- Do not enter passwords, OTPs, recovery details, or passkey enrollment choices for the user.
- Do not click security/account-setting prompts that are unrelated to notebook execution.
- Do not treat the killed `20260522_014721` launch as model validation evidence.
- Do not broadly stage or revert the dirty worktree.

## References

- Playwright MCP tools were available through `mcp__playwright__`.
- Playwright MCP config was previously installed outside the repo in `C:\Users\magil\.codex\config.toml` with a Chrome profile under `C:\Users\magil\.codex\playwright-mcp-chrome-profile`.
- If the MCP server is missing, inspect the Codex config before rebuilding it.
