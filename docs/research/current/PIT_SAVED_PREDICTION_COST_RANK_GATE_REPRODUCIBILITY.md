# PIT Saved-Prediction Cost + Rank-Gate Backtests

This re-evaluation reuses saved averaged predictions; it does not retrain models.
Scenario: transaction costs enabled and rank-drop gate enabled.

## Commands

```text
python scripts/run_pit_saved_prediction_backtests.py --run-root RUN
```

## Yearly Cost-Aware Results

| year | status | returncode | backtest.total_return | backtest.benchmark_return | backtest.excess_return | backtest.ARR | backtest.ASR | backtest.MDD | backtest.avg_daily_turnover | backtest.num_trading_days | backtest.transaction_costs_enabled | backtest.rank_gate_enabled | backtest.min_rank_drop | backtest.days_with_gate_exits | backtest.days_skipped_by_rank_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | OK | 0 | 0.12 | 0.05 | 0.07 | 0.13 | 1.1 | -0.2 | 0.35 | 237 | True | True | 30 | 12 | 0 |

## Side-By-Side Metrics

| year | metric | reviewed_no_cost_no_gate | cost_rank_gate | delta |
| --- | --- | --- | --- | --- |
| 2022 | total_return | 0.18 | 0.12 | -0.06 |
| 2022 | excess_return | 0.13 | 0.07 | -0.06 |
| 2022 | ARR | 0.19 | 0.13 | -0.06 |
| 2022 | ASR | 1.4 | 1.1 | -0.3 |
| 2022 | MDD | -0.18 | -0.2 | -0.02 |
| 2022 | avg_daily_turnover |  | 0.35 |  |
| 2022 | num_trading_days | 237.0 | 237.0 | 0.0 |

## Combined 2022-2025 Summary

| years | cost_rank_gate_compounded_total_return | cost_rank_gate_compounded_benchmark_return | cost_rank_gate_compounded_excess_return | cost_rank_gate_avg_ARR | cost_rank_gate_avg_ASR | cost_rank_gate_worst_MDD | cost_rank_gate_avg_daily_turnover | total_trading_days | transaction_costs_enabled | rank_gate_enabled | reviewed_compounded_total_return | reviewed_compounded_benchmark_return | reviewed_compounded_excess_return | reviewed_avg_ASR | reviewed_worst_MDD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 0.12 | 0.05 | 0.07 | 0.13 | 1.1 | -0.2 | 0.35 | 237 | True | True | 0.18 | 0.05 | 0.13 | 1.4 | -0.18 |
