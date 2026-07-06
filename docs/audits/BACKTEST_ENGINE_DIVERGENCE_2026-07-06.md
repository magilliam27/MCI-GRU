# Backtest Engine Divergence Audit (WS-N Step 1)

**Date:** 2026-07-06  
**Branch:** `codex/rearchitecture-phase0-20260704`  
**Scope:** Post–WS-C locations only (read-only analysis; no code changes)

| File | Role | Lines | Top-level functions |
|---|---|---:|---:|
| `mci_gru/evaluation/backtest_engine.py` | Full engine | ~3572 | 44 |
| `scripts/backtest_sp500_daily.py` | Daily fork (self-contained body) | ~2598 | 39 |

**Method:** AST extraction of top-level `def` bodies via throwaway script  
`.tmp_pytest/wsn_divergence_audit.py` (not committed). Raw JSON artifact:  
`.tmp_pytest/wsn_divergence_audit.json`.

**Note on spec count:** WS-N addendum cites ~38 shared functions. This audit finds **39**
shared-by-name functions. The addendum list omitted helpers that exist in both files today
(`evaluate_multiple_models`, `plot_equity_curve`, `setup_backtest_logging`,
`setup_backtest_output_dir`, `calculate_transaction_cost`, and the p-value/t-stat helpers).

---

## Summary counts

| Bucket | Count |
|---|---:|
| **IDENTICAL** | 33 |
| **COSMETIC DRIFT** | 1 |
| **BEHAVIORAL DRIFT** | 5 |
| **Shared total** | 39 |
| **Engine-only** | 5 |
| **Daily-only** | 0 |

---

## Per-function classification (all 39 shared)

| Function | Bucket | Engine lines | Daily lines | Notes |
|---|---|---:|---:|---|
| `_pit_active_kdcodes_on_date` | IDENTICAL | 784–788 | 775–779 | |
| `_pit_active_rows` | IDENTICAL | 773–781 | 764–772 | |
| `adjust_p_value_bhy` | IDENTICAL | 423–459 | 414–450 | |
| `adjust_p_value_bonferroni` | IDENTICAL | 462–488 | 453–479 | |
| `adjust_p_value_holm` | IDENTICAL | 491–523 | 482–514 | |
| `bhy_c_factor` | IDENTICAL | 404–420 | 395–411 | |
| `calculate_arr` | IDENTICAL | 132–161 | 123–152 | |
| `calculate_asr` | IDENTICAL | 216–232 | 207–223 | |
| `calculate_avol` | IDENTICAL | 164–186 | 155–177 | |
| `calculate_cr` | IDENTICAL | 235–251 | 226–242 | |
| `calculate_forward_returns` | IDENTICAL | 682–761 | 673–752 | |
| `calculate_ir` | IDENTICAL | 254–290 | 245–281 | |
| `calculate_mae` | IDENTICAL | 317–338 | 308–329 | |
| `calculate_mdd` | IDENTICAL | 189–213 | 180–204 | |
| `calculate_mse` | IDENTICAL | 293–314 | 284–305 | |
| `calculate_t_statistic` | IDENTICAL | 351–370 | 342–361 | |
| `calculate_transaction_cost` | IDENTICAL | 956–1011 | 947–1002 | |
| `calculate_turnover` | **BEHAVIORAL DRIFT** | 895–953 | 886–944 | See §1 below |
| `calendar_returns_for_evaluation_window` | IDENTICAL | 809–849 | 800–840 | |
| `derive_holdings_summary` | IDENTICAL | 2575–2606 | 1816–1847 | |
| `derive_portfolio_composition` | IDENTICAL | 2533–2572 | 1774–1813 | |
| `equal_weight_benchmark_daily_series` | IDENTICAL | 791–806 | 782–797 | |
| `evaluate` | **BEHAVIORAL DRIFT** | 2143–2311 | 1409–1552 | See §2 below |
| `evaluate_multiple_models` | IDENTICAL | 3113–3173 | 2244–2304 | |
| `haircut_sharpe_ratio` | IDENTICAL | 526–624 | 517–615 | |
| `load_pit_universe_for_backtest` | IDENTICAL | 764–770 | 755–761 | |
| `load_predictions` | IDENTICAL | 852–883 | 843–874 | |
| `load_stock_data` | IDENTICAL | 648–679 | 639–670 | |
| `main` | **BEHAVIORAL DRIFT** | 3181–3568 | 2312–2598 | See §3 below |
| `p_value_to_t_stat` | IDENTICAL | 387–401 | 378–392 | |
| `plot_equity_curve` | **BEHAVIORAL DRIFT** | 2996–3105 | 2150–2236 | See §4 below |
| `print_results` | IDENTICAL | 2314–2466 | 1555–1707 | |
| `resolve_data_file` | IDENTICAL | 632–645 | 623–636 | |
| `save_backtest_results` | **BEHAVIORAL DRIFT** | 2609–2863 | 1850–2104 | See §5 below |
| `save_results` | IDENTICAL | 2469–2509 | 1710–1750 | |
| `setup_backtest_logging` | IDENTICAL | 2866–2906 | 2107–2147 | |
| `setup_backtest_output_dir` | IDENTICAL | 2512–2530 | 1753–1771 | |
| `simulate_trading_strategy` | **COSMETIC DRIFT** | 1019–1411 | 1010–1401 | One inline comment removed in daily; AST identical after docstring strip |
| `t_stat_to_p_value` | IDENTICAL | 373–384 | 364–375 | |

---

## Behavioral drift — detailed decisions

### §1 `calculate_turnover`

**Engine (`backtest_engine.py`):**

```python
k = int(target_k) if target_k is not None else len(curr_set)
k = max(k, 1)
one_way_turnover = shared_calculate_turnover(prev_set, curr_set, target_k=k)
```

**Daily (`backtest_sp500_daily.py`):**

```python
k = int(target_k) if target_k is not None else len(curr_set)
k = max(k, 1)
one_way_turnover = traded_names / (2 * k)
```

**Consequence:** Engine delegates one-way turnover to
`mci_gru.evaluation.portfolio.calculate_turnover` (imported as `shared_calculate_turnover`).
Daily inlines `(num_sold + num_bought) / (2 * k)`. When the wrapper passes explicit `target_k=k`
(as all in-engine call sites do), the formulas are **numerically equivalent**. The portfolio helper
differs only when `target_k=None` (uses `max(len(prev), len(curr))` vs daily's `len(curr)`), but
that branch is not exercised by either backtest wrapper today.

**Git history:** Engine delegation added in `2f77f2c` ("Add Phase 4 evaluation trust layer");
daily fork never picked up the refactor.

**Recommendation:** **Keep engine behavior** — delegate to `portfolio.calculate_turnover` for a
single source of truth shared with paper-trade / portfolio code. Add a unit test asserting wrapper
output matches the inline formula for representative holdings sets (merge step 2 goldens).

---

### §2 `evaluate`

**Engine:**

```python
holding_period = config.get("holding_period", 1)
rebalance_style = config.get("rebalance_style", "staggered")
if holding_period == 1:
    sim_results = simulate_trading_strategy(..., pit_universe_df=pit_universe_df)
elif rebalance_style == "block":
    sim_results = simulate_trading_strategy_block(...)  # no pit_universe_df kwarg
else:
    sim_results = simulate_trading_strategy_staggered(...)  # no pit_universe_df kwarg
```

**Daily:**

```python
sim_results = simulate_trading_strategy(
    ..., pit_universe_df=pit_universe_df,
)
```

**Consequence:** For `holding_period=1` (daily CLI default), outputs are identical. For
`holding_period>1`, engine routes to staggered/block simulators that **do not exist in the daily
file**; daily would silently ignore those config keys if copied verbatim. Remaining diff is three
removed comments in daily (no logic change).

**Git history:** Multi-day dispatch added in `bbd5e37` to `tests/backtest_sp500.py` only; daily
fork stayed on daily-only path.

**Recommendation:** **Keep engine dispatch** in the merged module. Daily CLI wrapper forces
`holding_period=1` so daily goldens stay on `simulate_trading_strategy`. When extending staggered/block
to PIT universes, that is a separate tracked change (currently block/staggered omit
`pit_universe_df`).

---

### §3 `main`

**Engine-only argparse flags (absent from daily):**

| Flag | Type / action | Default |
|---|---|---|
| `--enable_mlflow` | `store_true` | off |
| `--disable_mlflow_autolink` | `store_true` | off |
| `--mlflow_tracking_uri` | `str` | `None` |
| `--mlflow_experiment_name` | `str` | `None` |
| `--holding_period` | `int` | `1` |
| `--rebalance_style` | `str`, choices `staggered\|block` | `staggered` |

**Engine-only runtime (absent from daily):**

```python
tracking_manager, linked_metadata = setup_backtest_tracking(...)
# ... log_params, log_metrics, _log_backtest_artifacts, close()
```

**Daily:** Same shared flags as engine minus the six above; never calls MLflow helpers; always
dispatches simulation via `simulate_trading_strategy` in the `--auto_save` path (mirrors §2).

**Consequence:** Daily CLI cannot run staggered/block backtests or MLflow-tracked backtests. Full
engine CLI (`scripts/backtest_sp500.py` already delegates to `backtest_engine.main`) exposes the
superset.

**Git history:** MLflow flags in `36f7f4d`; holding-period flags in `bbd5e37` (engine lineage).

**Recommendation:** **Keep engine `main` as the canonical argparse surface.** Post-merge:
`scripts/backtest_sp500.py` → full surface (already true); `scripts/backtest_sp500_daily.py` → thin
wrapper that injects `holding_period=1`, omits MLflow flags, and calls shared engine entry point.

---

### §4 `plot_equity_curve`

Same structural drift as §2 `evaluate`: engine dispatches on `holding_period` /
`rebalance_style`; daily always calls `simulate_trading_strategy`.

**Engine snippet:**

```python
if holding_period == 1:
    sim_results = simulate_trading_strategy(..., pit_universe_df=pit_universe_df)
elif rebalance_style == "block":
    sim_results = simulate_trading_strategy_block(...)
else:
    sim_results = simulate_trading_strategy_staggered(...)
```

**Daily snippet:**

```python
sim_results = simulate_trading_strategy(..., pit_universe_df=pit_universe_df)
```

**Consequence:** Equity-curve PNGs differ when `holding_period>1` configs are used; identical for
daily (`holding_period=1`) runs.

**Recommendation:** **Keep engine dispatch**; daily wrapper never passes `holding_period>1`.

---

### §5 `save_backtest_results`

**Engine:**

```python
print("\nAll backtest outputs saved successfully.")
```

**Daily:**

```python
print("\n✓ All backtest outputs saved successfully!")
```

**Consequence:** Console text only; saved files (`summary.txt`, CSVs, JSON) are identical.

**Git history:** Both files got the checkmark variant in `bbd5e37`. Engine lineage later moved to
plain text (present from `2f77f2c` onward in `tests/backtest_sp500.py`); daily fork kept the
`bbd5e37` checkmark string.

**Recommendation:** **Keep engine plain message** (matches current full-engine / `scripts/backtest_sp500.py`
path). Cosmetic only; golden tests should not pin stdout for this line.

---

## Cosmetic drift detail

### `simulate_trading_strategy`

Daily removed one inline comment inside the simulation loop:

```python
# Benchmark: full-universe equal-weight for this calendar day (precomputed; stable vs path)
benchmark_return = float(bm_by_date.get(entry_date, np.nan))
```

AST bodies match after docstring strip. **No merge action** beyond keeping either comment (prefer
engine comment for documentation value).

---

## Functions only in engine (5)

Verified against spec WS-N table — all five match.

| Function | Engine lines | Purpose |
|---|---:|---|
| `simulate_trading_strategy_staggered` | 1419–1784 | Multi-day holding, staggered tranches |
| `simulate_trading_strategy_block` | 1785–2142 | Multi-day holding, block rebalance |
| `_infer_experiment_name` | 2909–2916 | MLflow run naming |
| `setup_backtest_tracking` | 2917–2976 | MLflow backtest tracking setup |
| `_log_backtest_artifacts` | 2977–2995 | MLflow artifact logging |

These stay in the merged engine module. Daily CLI must not duplicate them.

---

## Functions only in daily (0)

No top-level functions exist in `scripts/backtest_sp500_daily.py` that are absent from the engine.
The daily file is a strict subset plus drifted copies of shared names.

---

## `main()` argparse surface diff (flag-by-flag)

Shared flags: **identical** type, default, and help text in both files.

| Flag | Engine | Daily | Match? |
|---|---|---|---|
| `--predictions_dir` | `str`, **required** | same | ✓ |
| `--data_file` | `str`, default `data/raw/market/sp500_data.csv` | same | ✓ |
| `--pit_universe_csv` | `str`, default `None` | same | ✓ |
| `--top_k` | `int`, default `10` | same | ✓ |
| `--test_start` | `str`, default `2025-01-01` | same | ✓ |
| `--test_end` | `str`, default `2025-12-31` | same | ✓ |
| `--label_t` | `int`, default `5` | same | ✓ |
| `--output` | `str`, default `None` | same | ✓ |
| `--plot` | `store_true` | same | ✓ |
| `--multi_model` | `str`, default `None` | same | ✓ |
| `--num_models` | `int`, default `10` | same | ✓ |
| `--num_tests` | `int`, default `1` | same | ✓ |
| `--adjustment_method` | `str`, default `bhy`, choices `bhy\|bonferroni\|holm` | same | ✓ |
| `--transaction_costs` | `store_true` | same | ✓ |
| `--spread` | `float`, default `10.0` | same | ✓ |
| `--slippage` | `float`, default `5.0` | same | ✓ |
| `--auto_save` | `store_true` | same | ✓ |
| `--backtest_suffix` | `str`, default `""` | same | ✓ |
| `--enable_rank_drop_gate` | `store_true` | same | ✓ |
| `--min_rank_drop` | `int`, default `10` | same | ✓ |

**Engine-only flags** (must survive in merged full CLI / `scripts/backtest_sp500.py`):

| Flag | Type / action | Default |
|---|---|---|
| `--enable_mlflow` | `store_true` | off |
| `--disable_mlflow_autolink` | `store_true` | off |
| `--mlflow_tracking_uri` | `str` | `None` |
| `--mlflow_experiment_name` | `str` | `None` |
| `--holding_period` | `int` | `1` |
| `--rebalance_style` | `str` (`staggered`, `block`) | `staggered` |

**Config dict keys wired from args:** Engine adds `holding_period` and `rebalance_style` to the
`config` dict passed to `evaluate` / `save_backtest_results`; daily omits both (implicit daily
simulation only).

---

## Surprises / merge notes

1. **`scripts/backtest_sp500.py` already exists** as a thin wrapper to `backtest_engine.main()`
   (WS-C). WS-N merge work is mainly collapsing the **daily body** into the engine and making
   `scripts/backtest_sp500_daily.py` a wrapper — not inventing the full CLI from scratch.

2. **Shared count is 39, not 38** — spec list was incomplete; no extra daily-only functions.

3. **`calculate_turnover` drift is refactor-only** for current call patterns; numeric outputs should
   match when `target_k` is passed explicitly. Still classify as behavioral in code path terms.

4. **Block/staggered simulators omit `pit_universe_df`** today — pre-existing engine limitation,
   not introduced by this fork. PIT + multi-day holding is undefined behavior until explicitly
   implemented.

5. **~857 lines of duplication** remain byte-identical between forks (33 functions); merge deletes
   the daily copies wholesale with low risk for those symbols.

6. **Daily fork line numbers** are ~9 lines lower throughout due to missing module-level imports
   (`mci_gru.evaluation.portfolio`, `mci_gru.tracking`) and five engine-only function bodies.

---

## Recommended merge order (input for WS-N steps 2–4)

1. Treat `mci_gru/evaluation/backtest_engine.py` as sole implementation body.
2. Apply behavioral decisions §1–§5 (all favor engine side except none change metrics for
   `holding_period=1` daily goldens except §5 stdout).
3. Replace `scripts/backtest_sp500_daily.py` body with thin wrapper (mirror
   `scripts/backtest_sp500.py` pattern).
4. Build golden fixtures from **both** CLIs before merge (step 2) to pin daily + staggered + block
   matrices per WS-N spec.

---

*Generated by WS-N step 1 audit. Script: `.tmp_pytest/wsn_divergence_audit.py`.*
