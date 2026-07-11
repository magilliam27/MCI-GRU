# Regime CSV Contract Continuation Audit Handoff

Last updated: 2026-07-11

## Resume Here

- Canonical continuation surface: PR #70 / `codex/regime-csv-no-backfill-mined-20260709` at `98bed4b`.
- The focused unlagged no-backfill regression has already been mined from `codex/regime-csv-no-backfill-coverage`; do not repeat that work.
- After PR #70 merges, current `origin/main` is canonical. Keep the older coverage and deprecation branches parked until a separately approved cleanup pass.

## Current Objective

- Merge the reviewed minimal no-backfill regression while preserving current-only global regime semantics.
- Do not call FRED/LSEG or weaken the explicit `FRED_API_KEY` behavior for production-style regime-enabled runs.

## What Changed

- The original 2026-07-09 audit added this handoff and identified the missing unlagged no-backfill coverage.
- The focused test was mined onto `codex/regime-csv-no-backfill-mined-20260709` and published as PR #70.
- The 2026-07-11 merge review corrected this handoff so it no longer routes agents back to the already-mined branch.
- No production source/config behavior, data-vendor call, training, backtest, Colab run, branch deletion, prune, reset, or rebase was performed.

## Key Decisions

- Continue from PR #70, not `codex/regime-csv-no-backfill-coverage` or `codex/deprecate-regime-csv`.
  Reason: PR #70 contains only the focused test plus this audit handoff and is cleanly mergeable into current main.
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
- `docs/agents/cockpit/workstream-decisions.json`: durable reviewed disposition for the Regime CSV workstream and its historical branches.
- `docs/agents/cockpit/2026-07-10.md`: latest merged cockpit snapshot at merge-review time; the 2026-07-08 packet is historical only.
- `docs/agents/workstreams.md`: generated continuation register; do not edit it instead of the decision registry.

## Merge Review Verification

- PR #70 head: `98bed4b`; current `origin/main`: `1d7866b`.
- `git merge-tree --write-tree origin/main origin/codex/regime-csv-no-backfill-mined-20260709` completed without conflicts.
- GitHub CI run 122 passed lint, formatting, docs validation, repository tests, and end-to-end smoke.
- Focused Regime CSV selection reported `3 passed, 10 deselected`; `FRED_API_KEY=dummy` only bypassed the module skip hook and no live vendor was called.
- The PR changes only this handoff and `tests/test_regime_features.py`; production code and configuration are unchanged.

## Original Audit Verification (2026-07-09)

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

- `tests/test_regime_features.py` is module-marked/skipped by the repo's FRED hook, so synthetic CSV tests skip when `FRED_API_KEY` is absent. Consider splitting no-vendor CSV tests into a file or marker path that always runs locally without secrets.
- Some legacy docs/generators still mention `REGIME_INPUTS_CSV` or static regime CSV fallback paths. Canonical docs are aligned, but old notebook generators should be classified as legacy or updated to call the CSV path deprecated.
- The older coverage and deprecation branches remain as cleanup candidates; merging PR #70 does not authorize deleting them.

## Next Actions

1. Merge PR #70 after the corrected handoff passes CI.
2. Record `origin/main` as the landed Regime CSV surface in the cockpit decision registry.
3. Separately decide whether no-vendor CSV contract tests should move out from under the module-level FRED skip hook.
4. Audit legacy `REGIME_INPUTS_CSV` mentions as a separate documentation task; keep CSV explicitly deprecated.

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

- Do not reset, delete, prune, or rebase the older regime branches without explicit cleanup approval.
- Do not resurrect CSV as the production regime workflow.
- Do not silently disable global regime or current-only semantics to avoid `FRED_API_KEY`.
- Do not treat handoffs or old notebook static-regime flows as canonical over `docs/REGIME_DATA_CONTRACT.md`, current code, and tests.
