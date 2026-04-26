---
name: MCI-GRU Iteration Plan
overview: A structured 2-week plan to implement 8 confirmed model improvements (lookback, dropout, weekly momentum, credit spreads, transaction cost analysis, look-ahead bias, monthly rebalancing, macro features, trading reduction tests) with systematic experiment tracking.
todos:
  - id: day1-baseline-lookback
    content: "Day 1: Run baseline + lookback sweep (his_t=20,30,40,50). No code changes needed -- use Hydra multirun."
    status: pending
  - id: day2-dropout
    content: "Day 2: Add dropout to model architecture (config.py, model classes). Sweep dropout=0.3,0.35,0.4,0.45,0.5."
    status: pending
  - id: day3-features
    content: "Day 3: Add weekly_momentum (5d) to momentum.py. Create credit.py for IG/HY spreads. Register in pipeline."
    status: pending
  - id: day4-txcost-bias
    content: "Day 4: Sweep transaction costs (>20 bps) to find breakeven. Document and enforce look-ahead bias mitigations."
    status: pending
  - id: day5-monthly
    content: "Day 5: Implement monthly rebalancing mode in backtest. Test with dropout 0.3, 0.4, 0.5."
    status: pending
  - id: day6-macro
    content: "Day 6: Create macro.py (CPI, jobless claims, yield curve). Implement linear interpolation. Register in pipeline."
    status: pending
  - id: day7-trading-reduction
    content: "Day 7: Implement 3 trading reduction tests (rank drop threshold, momentum filter, churn buffer zone)."
    status: pending
  - id: day8-consolidation
    content: "Day 8: Combine best settings, run final experiment, apply multiple testing haircut, generate report."
    status: pending
isProject: false
---

# MCI-GRU Model Improvement: 2-Week Iteration Plan

## Scope (YES items only)

1. Lookback window >10 days (20, 30, 40, 50) -- YES
2. Add last week's return to momentum -- YES
3. Add credit spread (daily, IG and HY) -- YES
4. Increase transaction cost >0.20 bps + find breakeven -- YES
5. Address look-ahead bias -- YES
6. Macro features: CPI, jobless claims, yield curve inversion -- YES
7. Dropout 0.3-0.5 (step by 0.05 or 0.10) -- YES
8. Monthly predictions instead of daily (test with dropout 0.3, 0.4, 0.5) -- YES
9. Establish 3 tests to reduce trading -- YES

**Removed**: Correlation cap, model out >5 days, flip model / expand portfolio.

---

## Current State

- **Lookback**: 10 days (`model.his_t=10`)
- **Forward period**: 5 days (`model.label_t=5`)
- **Features**: OHLCV + momentum (252d slow, 21d fast) -- no credit spreads, no macro
- **Dropout**: None (0.0)
- **Transaction costs**: 10 bps spread + 5 bps slippage (disabled by default)
- **Rebalancing**: Daily
- **Portfolio**: Top-10 equal-weighted
- **Graph**: Static correlation with threshold 0.8

Key files:

- `[run_experiment.py](run_experiment.py)` -- Hydra-based experiment runner
- `[mci_gru/config.py](mci_gru/config.py)` -- Config dataclasses (no dropout field)
- `[mci_gru/features/registry.py](mci_gru/features/registry.py)` -- Feature pipeline
- `[mci_gru/features/momentum.py](mci_gru/features/momentum.py)` -- Momentum features (missing 5d return)
- `[tests/backtest_sp500.py](tests/backtest_sp500.py)` -- Backtesting with transaction costs
- `[mci_gru_sp500.py](mci_gru_sp500.py)` -- Monolithic model (no dropout in `StockPredictionModel`)

---

## Experiment Tracking Strategy

Create a results tracker (`results/experiment_tracker.csv`) with columns:

- `experiment_id`, `date_run`, `description`, `his_t`, `dropout`, `features_added`, `transaction_cost_bps`, `rebalance_freq`, `top_k`, `ARR`, `AVoL`, `MDD`, `ASR`, `CR`, `IR`, `MSE`, `num_tests`, `haircutted_ASR`

Every experiment run saves to `results/{experiment_name}/{timestamp}/` via Hydra. After each backtest, append a row to the tracker.

---

## Week 1: Infrastructure + Quick Wins (Days 1-5)

### Day 1 (Mon) -- Baseline + Lookback Sweep

**Goal**: Establish baseline metrics and test extended lookback windows.

- **Run 1**: Baseline (his_t=10) -- record all metrics as the control
- **Runs 2-5**: Lookback sweep via Hydra multirun:
  ```
  python run_experiment.py --multirun model.his_t=20,30,40,50 experiment_name=lookback_sweep
  ```
- Backtest each with `tests/backtest_sp500.py`
- Record results in tracker; identify best lookback

**Code changes**: None required -- Hydra sweep already supports this.

---

### Day 2 (Tue) -- Dropout Regularization

**Goal**: Add dropout to the model and sweep values 0.3-0.5.

**Code changes required**:

1. **Add `dropout` to `ModelConfig**` in `[mci_gru/config.py](mci_gru/config.py)`:
  - Add `dropout: float = 0.0` field
2. **Add `nn.Dropout` layers to `StockPredictionModel**` in `[mci_gru_sp500.py](mci_gru_sp500.py)` (lines 479-530):
  - After GRU output (h_gru_1)
  - After GAT layer output (x_gat)
  - After cross-attention outputs (stock_rep_1, stock_rep_2)
  - After concatenation, before final GAT
3. **Wire dropout through model factory** in `[run_experiment.py](run_experiment.py)`
4. **Sweep**: Run with dropout=0.3, 0.35, 0.4, 0.45, 0.5 using best lookback from Day 1
  ```
   python run_experiment.py --multirun model.dropout=0.3,0.35,0.4,0.45,0.5
  ```

---

### Day 3 (Wed) -- New Features: Last Week's Return + Credit Spread

**Goal**: Add 5-day momentum and credit spread features.

**Code changes required**:

1. **Add 5-day return to momentum** in `[mci_gru/features/momentum.py](mci_gru/features/momentum.py)`:
  - Add `weekly_momentum` (5-day trailing return) alongside existing 21d and 252d
  - Add to `MOMENTUM_FEATURES` list
2. **Create credit spread module** at `mci_gru/features/credit.py`:
  - IG spread (ICE BofA US Corporate Index OAS -- FRED series `BAMLC0A0CM`)
  - HY spread (ICE BofA US High Yield OAS -- FRED series `BAMLH0A0HYM2`)
  - Compute: spread level, daily change, z-score
  - These are market-wide features broadcast to all stocks (like VIX)
3. **Data sourcing**: Use FRED API or pre-downloaded CSV for IG/HY spreads (daily frequency)
4. **Register in `FeatureEngineer**` in `[mci_gru/features/registry.py](mci_gru/features/registry.py)`:
  - Add `include_credit_spread: bool` flag
  - Add to `FeatureConfig` in config.py
5. **Run experiments**:
  - (a) Best lookback + best dropout + weekly momentum only
  - (b) Best lookback + best dropout + weekly momentum + credit spreads

---

### Day 4 (Thu) -- Transaction Cost Analysis + Look-Ahead Bias Audit

**Goal**: Answer "What transaction cost wipes out gains?" and document bias prevention.

**Transaction cost sweep** (no code changes needed, use backtest CLI):

```
python tests/backtest_sp500.py --predictions_dir {best_run}/averaged_predictions \
  --transaction_costs --spread X --slippage Y
```

- Test spread values starting above 20 bps: 20, 25, 30, 40, 50, 75, 100 bps
- Find the breakeven point where ARR ~ 0 or ASR ~ 0
- Document the answer to "What level of transaction cost would wipe out the gains?"

**Look-ahead bias audit** -- Document how it's addressed:

- **Current mitigations** (already in code):
  - Normalization stats computed on training period only (`[run_experiment.py](run_experiment.py)` line 183)
  - Trading uses T+1 open prices for entry (backtest lines 990-1002)
  - Forward returns use `close_{T+label_t} / close_{T+1} - 1` (line 680)
  - Correlation graph built from pre-training data only (line 256)
- **Additional mitigations to implement**:
  - Ensure credit spread / macro data is lagged by 1 day (use T-1 values for prediction at T)
  - Ensure weekly/monthly macro data uses last-available value (no future fill)
  - Add explicit lag parameter to feature engineering functions
  - Write up a clear explanation for the team

---

### Day 5 (Fri) -- Monthly Rebalancing Mode

**Goal**: Implement monthly prediction/rebalancing to reduce turnover and costs.

**Code changes required**:

1. **Add `rebalance_frequency` to config** in `[mci_gru/config.py](mci_gru/config.py)`:
  - Options: `daily`, `weekly`, `monthly`
2. **Modify backtest** in `[tests/backtest_sp500.py](tests/backtest_sp500.py)`:
  - Add logic to only rebalance on first trading day of each month
  - Between rebalance dates, hold the same portfolio
  - Accumulate open-to-open returns for the holding period
3. **Test with dropout values**: 0.3, 0.4, 0.5 as requested
  - Run with best lookback + features from earlier in the week

---

## Week 2: Macro Features + Trading Controls + Consolidation (Days 6-8)

### Day 6 (Mon) -- Macroeconomic Features Layer

**Goal**: Add CPI, jobless claims, yield curve inversion as model inputs.

**Code changes required**:

1. **Create macro features module** at `mci_gru/features/macro.py`:
  - **CPI**: Monthly, hold same value for ~30 days until next release (linear interpolation)
  - **Initial Jobless Claims**: Weekly (Thursday), forward-fill to daily until next release
  - **Yield Curve Inversion**: Daily (10Y - 2Y Treasury spread), use `lseg_loader.py`'s existing `get_treasury_yields()` or FRED
  - All macro features must be **lagged 1 day** to prevent look-ahead bias
2. **Linear interpolation helper**:
  - For monthly data (CPI): hold the same value for 30 calendar days until next release
  - For weekly data (claims): hold value until next Thursday release
3. **Register in pipeline**: Add `include_macro: bool` to `FeatureConfig` and `FeatureEngineer`
4. **Run experiments**:
  - Best config so far + macro features
  - Compare with and without macro features

---

### Day 7 (Tue) -- Trading Reduction Tests (3 Tests)

**Goal**: Implement and test 3 mechanisms to reduce unnecessary trading.

**Code changes required in `[tests/backtest_sp500.py](tests/backtest_sp500.py)**`:

1. **Rank drop threshold**: Only sell a stock if it drops more than N positions in ranking
  - Parameter: `min_rank_drop` (e.g., stock drops from rank 3 to rank 12 = 9 positions -- sell; rank 3 to rank 11 = keep)
  - Test values: 5, 10, 15, 20 positions
2. **Momentum continuation filter**: Only sell if the stock's short-term momentum has turned negative
  - Use the weekly_momentum feature from Day 3
  - Don't sell a stock still showing positive weekly momentum even if rank drops
3. **Churn control / buffer zone**: Implement a "buy zone" (top-K) and "sell zone" (below top-K+buffer)
  - Only sell stocks that fall below rank K+buffer (e.g., K=10, buffer=5, so only sell if rank > 15)
  - Test buffer values: 3, 5, 8, 10
4. **Run all 3 tests** independently on best config, then combine best settings

---

### Day 8 (Wed) -- Consolidation + Final Analysis

**Goal**: Combine best settings, run final experiments, prepare results summary.

1. **Combine best settings** from all prior experiments:
  - Best lookback window
  - Best dropout rate
  - Best feature set (weekly momentum + credit spreads + macro)
  - Best trading reduction rules
  - Monthly vs daily rebalancing comparison
2. **Run final combined experiment** with full 10-model ensemble
3. **Apply multiple testing adjustment** (Harvey & Liu) across all experiments run:
  ```
   python tests/backtest_sp500.py --num_tests N --adjustment_method bhy
  ```
4. **Generate final report**:
  - Comparison table of all experiments
  - Transaction cost breakeven analysis
  - Statistical significance after haircut
  - Recommendations for which improvements to keep

---

## Summary of Code Changes by File

- **P0** `mci_gru/config.py` -- Add `dropout`, `rebalance_frequency`, `include_credit_spread`, `include_macro`
- **P0** `mci_gru_sp500.py` (or `mci_gru/models/`) -- Add `nn.Dropout` layers throughout model
- **P0** `mci_gru/features/momentum.py` -- Add 5-day `weekly_momentum` feature
- **P1** `mci_gru/features/credit.py` -- New: IG/HY credit spread features
- **P1** `mci_gru/features/macro.py` -- New: CPI, jobless claims, yield curve features
- **P1** `mci_gru/features/registry.py` -- Register credit + macro features
- **P1** `tests/backtest_sp500.py` -- Monthly rebalancing, trading reduction tests
- **P2** `configs/config.yaml` -- New default fields for all additions
- **P2** `results/experiment_tracker.csv` -- New: centralized experiment tracking

