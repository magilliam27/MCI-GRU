# MCI-GRU Testing Guide

This guide captures the testing patterns agents and humans should use when
changing MCI-GRU. Prefer small saved regression tests with synthetic data before
running broad suites.

## Local Commands

```bash
python -m pytest tests/ -v
python -m pytest tests/ -m "not slow" -v
python -m pytest tests/ -k "test_no_lookahead" -v
ruff check .
```

On Windows PowerShell, prefer the repo venv and repo-local pytest temp:

```powershell
New-Item -ItemType Directory -Force .tmp_pytest | Out-Null
$env:TMP = (Resolve-Path .tmp_pytest).Path
$env:TEMP = $env:TMP
.\.venv\Scripts\python.exe -m pytest tests/ -k "test_no_lookahead" -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\python.exe -m pytest tests/ -m "not slow" -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest tests/ -v --basetemp .tmp_pytest\pytest
```

If pytest passes but emits a Windows permission warning for `.pytest_cache`,
append `-p no:cacheprovider` to disable cache writes for that run.

Use the smallest command that proves the changed behavior first. Run broader
checks before pushing shared pipeline, graph, model, or paper-trade changes.

## Verification Ladder

1. **Targeted proof**: run the specific test file or keyword for the changed
   behavior.
2. **Repo health proof**: run the non-slow suite plus `ruff check .`.
3. **Full confidence proof**: run the full suite for shared contracts, release
   candidates, or important PRs.

Always report exact commands, exit status, skipped tests, and remaining risk.

## Evidence Taxonomy

- **Contract test**: a saved pytest that protects an invariant or regression.
- **Local pytest/ruff**: targeted, non-slow, full-suite, or lint evidence from
  this checkout.
- **Local smoke**: a short `run_experiment.py` or paper-trade command that
  proves wiring outside pytest.
- **Live Colab smoke**: foreground notebook execution that proves the remote
  runtime can start the path.
- **Full Colab/Drive artifact proof**: completed notebook run with Drive-backed
  outputs, metrics, and logs.

FRED, LSEG, and GPU availability are evidence context. Report whether they were
available for the run, but do not treat them as hard requirements for every
local test.

## Closeout Checklist

- Interpreter path used, especially `.\.venv\Scripts\python.exe` on Windows.
- Cwd, branch, and worktree.
- Pytest temp override, including `TMP`/`TEMP` and `--basetemp`.
- Command exit status.
- Pass, skip, warning, and failure counts.
- Residual risk or unverified surfaces.

## Test Categories

- **Active regression tests**: saved tests that protect current behavior.
- **Slow/data-dependent tests**: keep them marked with `slow`, `requires_data`,
  `requires_fred`, or `requires_lseg`.
- **Script-like harnesses**: files that launch experiments or backtests rather
  than asserting behavior. Recommend moving these to `scripts/` or
  `tests/manual/`.
- **Stale or contradictory tests**: tests whose assumptions conflict with the
  current architecture. Explain the conflict before proposing archive/removal.

Do not move, archive, delete, or restructure tests without explicit approval.

## Core Invariants

Tests should protect the repository invariants in `AGENTS.md`:

- normalization stats, graph edges, and labels use strict train-period cutoffs;
- dynamic graph batches resolve edges through `GraphSchedule`;
- `combined_collate_fn` preserves the 9-tuple contract;
- ensemble prediction is the mean of independently trained models;
- paper-trade inference loads frozen `graph_data.pt` and does not import
  `GraphBuilder`.

## Synthetic Data Pattern

Use tiny panels with hand-computable values. Keep helpers local to the test file
unless multiple files need the same fixture.

```python
def _make_feature_panel() -> pd.DataFrame:
    rows = []
    for kdcode, closes in {
        "AAA": [100.0, 110.0, 121.0, 133.1],
        "BBB": [200.0, 180.0, 162.0, 145.8],
    }.items():
        for i, close in enumerate(closes, start=1):
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": f"2020-01-{i:02d}",
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": float(i * 100),
                }
            )
    return pd.DataFrame(rows)
```

Good feature tests usually assert:

- exact values for one or two rows;
- warmup rows are neutral or `NaN` by design;
- each `kdcode` is isolated from other stocks;
- output row count and ordering are unchanged;
- changing a future row does not alter earlier feature values.

## No-Lookahead Canary

For timing-sensitive features, compute output twice: once with normal future
data and once with a future row mutated. Assert rows before the mutation point
are unchanged.

```python
base = _make_feature_panel()
changed = base.copy()
changed.loc[changed["dt"] == "2020-01-04", "close"] *= 10.0

out_a = add_feature(base)
out_b = add_feature(changed)

past_mask = out_a["dt"] < "2020-01-04"
pd.testing.assert_series_equal(
    out_a.loc[past_mask, "feature"].reset_index(drop=True),
    out_b.loc[past_mask, "feature"].reset_index(drop=True),
)
```

This is especially useful for rolling, expanding, EWM, regime, correlation, and
forward-return logic.

## Feature Wiring Checks

When adding a feature family, test all wiring surfaces:

- feature function creates expected columns;
- `FeatureEngineer.transform()` calls the function when the flag is enabled;
- `FeatureEngineer.get_feature_columns()` returns the same columns;
- `build_feature_list()` includes columns when relevant;
- `FeatureConfig` validates knobs and Hydra YAML merges correctly.

Use `tests/test_momentum_blend_modes.py` and `tests/test_regime_features.py` as
models for calculation and no-lookahead tests.

## Graph Checks

Graph tests should use small deterministic panels and verify:

- static graph edge shapes and weights;
- dynamic `GraphSchedule.get_graph_for_date()` chooses the expected snapshot;
- snapshot construction uses only data before the valid-from date;
- edge feature width matches model creation expectations;
- sector-relation outputs preserve the collate 9-tuple contract.

Use `tests/test_dynamic_graph_updates.py` as the main reference.

## Backtest And Paper-Trade Checks

Backtest tests should assert the timing contract directly:

- prediction date;
- execution date;
- return attribution period;
- transaction cost and turnover handling.

Paper-trade tests should guard frozen inference:

```python
source = Path("paper_trade/infer.py").read_text()
assert "GraphBuilder" not in source
```

Prefer behavioral tests where possible, but keep this import guard because the
paper-trade invariant is architectural and easy to regress.

## Regression Test Quality Bar

A saved regression test should:

- fail for the intended bug or missing behavior before the fix when practical;
- use real code, not mocks, unless an external dependency makes that impossible;
- assert observable behavior rather than implementation details;
- be fast enough for targeted local runs;
- include a short comment only when the timing or finance assumption is not
  obvious.

If a test needs real market data, FRED, LSEG, GPU, or long runtime, mark it and
provide a fast synthetic companion test when possible.
