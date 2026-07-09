# Regime CSV Contract Continuation Audit Handoff

Last updated: 2026-07-09

## Resume Here

- Canonical continuation surface: `codex/regime-csv-no-backfill-coverage` at `C:/Users/magil/.codex/worktrees/pr34-regime-no-backfill/MCI-GRU`, or a fresh `origin/main` branch that cherry-picks/mines its single test delta.
- Start by inspecting `git diff origin/main...codex/regime-csv-no-backfill-coverage -- tests/test_regime_features.py`.
- Park `codex/deprecate-regime-csv` unless the user explicitly asks for archival cleanup. Its deprecation behavior has already been mined into `origin/main`; its current branch content is broad and stale.

## Current Objective

- Resolve the cockpit decision queue item for the Regime CSV contract by naming the continuation surface, preserving current-only global regime semantics, and identifying any missing no-backfill/no-lookahead coverage.
- Do not call FRED/LSEG or weaken the explicit `FRED_API_KEY` behavior for production-style regime-enabled runs.

## What Changed

- Added this handoff only.
- No source, config, test, branch, merge, reset, delete, prune, or rebase actions were performed.

## Key Decisions

- Continue from `codex/regime-csv-no-backfill-coverage`, not `codex/deprecate-regime-csv`.
  Reason: the no-backfill branch is narrow, remote-tracked, attached to the named PR34 worktree, and only adds one focused test to `tests/test_regime_features.py`.
- Treat `codex/deprecate-regime-csv` as already mined/parkable.
  Evidence: `d55866b [codex] Deprecate regime CSV workflow (#21)` is an ancestor of current `origin/main`; the local `codex/deprecate-regime-csv` ref is `[gone]`, far behind current main, and direct content diff is huge.
- Keep the current contract: live FRED/LSEG-backed loader is canonical; `features.regime_inputs_csv` is deprecated, seven-variable-only, warns on use, applies optional lag, then forward-fills only.
- Keep production/current-only global regime behavior explicit: normal configs leave `features.regime_inputs_csv: null`; strict regime-enabled production-style runs require `FRED_API_KEY` unless a smoke run explicitly disables global regime; current-only recipes must keep `features.regime_include_subsequent_returns=false`.

## Important Files

- `docs/REGIME_DATA_CONTRACT.md`: canonical regime input contract and deprecated CSV escape hatch rules.
- `docs/CONFIGURATION_GUIDE.md`: current config behavior for live regime loader, `FRED_API_KEY`, and CSV deprecation.
- `mci_gru/data/data_manager.py`: `DataManager.load_regime_inputs()` implements CSV warning, seven-variable validation, optional lag, and `ffill()` only for CSV inputs.
- `mci_gru/regime_contract.py`: shared seven-variable column list.
- `mci_gru/features/regime.py`: regime feature construction and current-only/subsequent-return feature switches.
- `tests/test_regime_features.py`: current main has lagged CSV no-backfill and regime no-lookahead-adjacent tests; PR34 adds the unlagged CSV gap test.
- `docs/agents/cockpit/2026-07-08.md`: absent in this detached `origin/main` checkout, but present in `C:/Users/magil/.codex/worktrees/89a4/MCI-GRU` on `codex/cockpit-refresh-20260708`.
- `docs/agents/workstreams.md`: current checkout has the 2026-07-06 queue; 2026-07-08 version was read from the cockpit-refresh worktree.

## Verification

- Current audit checkout: `C:/Users/magil/.codex/worktrees/ebbd/MCI-GRU`, detached `HEAD` at `16bcea9`, which is `origin/main` / `origin/HEAD`.
- Current checkout status: `git status --short --branch` showed `## HEAD (no branch)` with only repeated global-ignore permission warnings.
- PR34 worktree status: `codex/regime-csv-no-backfill-coverage...origin/codex/regime-csv-no-backfill-coverage`, no local changes reported.
- Branch inventory:
  - `codex/deprecate-regime-csv` local only, `[gone]`, head `1e5d406`.
  - `codex/regime-csv-no-backfill-coverage` local plus remote-tracking, head `3010b16`, worktree `C:/Users/magil/.codex/worktrees/pr34-regime-no-backfill/MCI-GRU`.
  - `remotes/origin/codex/regime-csv-forward-fill-test` also exists locally at `1986f2e`, but cockpit named the two surfaces above.
- Divergence from local `origin/main`:
  - `origin/main...codex/deprecate-regime-csv`: `94 3`.
  - `origin/main...codex/regime-csv-no-backfill-coverage`: `42 2`.
- `git ls-remote --heads origin '*regime*'` failed with `SEC_E_NO_CREDENTIALS`, so remote truth here is limited to existing local remote-tracking refs.
- Focused no-vendor current-main test command passed:
  `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_regime_features.py -k 'regime_csv or subsequent_return_horizon or stock_bond_corr' -v --basetemp .tmp_pytest\pytest -p no:cacheprovider`
  Result: `5 passed, 7 deselected in 6.45s`.
- Focused PR34 worktree test command passed:
  `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_regime_features.py -k 'regime_csv or lag_safety' -v --basetemp C:\Users\magil\.codex\worktrees\ebbd\MCI-GRU\.tmp_pytest\pytest-pr34 -p no:cacheprovider`
  Result: `3 passed, 10 deselected in 1.96s`.
- The test runs used `FRED_API_KEY=dummy-no-live-vendor` only to bypass the repo skip hook. The selected tests use CSV inputs or monkeypatched loaders and did not call live FRED/LSEG.

## Open Risks

- `codex/regime-csv-no-backfill-coverage` is behind current `origin/main` by 42 commits. Rebase/merge or cherry-pick the single test onto a fresh branch before publishing or merging.
- The unlagged CSV no-backfill test is absent from current `origin/main`; it is a distinct coverage gap from `test_regime_csv_lag_safety`.
- `tests/test_regime_features.py` is module-marked/skipped by the repo's FRED hook, so synthetic CSV tests skip when `FRED_API_KEY` is absent. Consider splitting no-vendor CSV tests into a file or marker path that always runs locally without secrets.
- Some legacy docs/generators still mention `REGIME_INPUTS_CSV` or static regime CSV fallback paths. Canonical docs are aligned, but old notebook generators should be classified as legacy or updated to call the CSV path deprecated.
- No live GitHub/remote refresh was performed because credential propagation failed.

## Next Actions

1. Create or switch to the chosen continuation surface, preferably a fresh branch from current `origin/main`, and apply only the PR34 unlagged no-backfill test if the user wants a minimal merge.
2. Run:
   `C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_regime_features.py -k 'regime_csv or lag_safety' -v --basetemp .tmp_pytest\pytest -p no:cacheprovider`
3. Decide whether to split CSV contract tests away from module-level FRED skipping so no-vendor regime CSV coverage runs in normal local/CI contexts.
4. Audit and update legacy `REGIME_INPUTS_CSV` mentions in notebook generators/docs, keeping explicit that CSV is deprecated and production-style runs use live FRED/LSEG with `FRED_API_KEY`.
5. After the focused test is current-main based and green, either merge/PR the minimal coverage branch or explicitly park PR34 and document that its test was mined.

## Commands Run

- `Get-Content` on `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/REGIME_DATA_CONTRACT.md`, `docs/CONFIGURATION_GUIDE.md`, `docs/agents/domain.md`, `docs/agents/workstreams.md`.
- `Get-Content C:/Users/magil/.codex/worktrees/89a4/MCI-GRU/docs/agents/cockpit/2026-07-08.md`.
- `git status --short --branch`
- `git worktree list --porcelain`
- `git branch --all --list '*regime*' --verbose`
- `git log --oneline --decorate --max-count=20 --all --grep='regime'`
- `git rev-list --left-right --count origin/main...codex/deprecate-regime-csv`
- `git rev-list --left-right --count origin/main...codex/regime-csv-no-backfill-coverage`
- `git diff --name-status origin/main...codex/regime-csv-no-backfill-coverage`
- `git diff --unified=80 origin/main...codex/regime-csv-no-backfill-coverage -- tests/test_regime_features.py`
- `git merge-base --is-ancestor d55866b origin/main`
- Focused pytest commands listed in `Verification`.

## Data/Experiment State

- No market data, FRED, LSEG, Colab, model training, paper-trade run, or backtest was launched.
- `.tmp_pytest/` was used for pytest temp output and is gitignored.

## Do Not Do

- Do not merge, reset, delete, prune, or rebase either regime branch without explicit user approval.
- Do not resurrect CSV as the production regime workflow.
- Do not silently disable global regime or current-only semantics to avoid `FRED_API_KEY`.
- Do not treat handoffs or old notebook static-regime flows as canonical over `docs/REGIME_DATA_CONTRACT.md`, current code, and tests.
