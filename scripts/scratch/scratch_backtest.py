"""
Run the same backtest on all averaged-prediction folders in seed_results/2025,
then plot all equity curves on one figure.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from tests.backtest_sp500 import (
    load_stock_data,
    calculate_forward_returns,
    load_predictions,
    simulate_trading_strategy,
    DEFAULT_CONFIG,
)

BASE_DIR = os.path.join(REPO_ROOT, "seed_results", "2025")

PREDICTION_PATHS = [
    ("seed 7",   os.path.join("seed7",   "averaged_predictions")),
    ("seed 55",  os.path.join("seed55",  "averaged_predictions")),
    ("seed 88",  os.path.join("seed88",  "averaged_predictions")),
    ("seed 123", os.path.join("seed123", "averaged_predictions")),
    ("seed 256", os.path.join("seed256", "averaged_predictions")),
    ("seed 314", os.path.join("seed314", "averaged_predictions")),
    ("seed 999", os.path.join("seed999", "averaged_predictions")),
]

# Same backtest config: daily, rank gate 30, top_k 20, 5 bps
CONFIG = {
    **DEFAULT_CONFIG,
    'top_k': 20,
    'holding_period': 1,
    'rebalance_style': 'staggered',
    'transaction_costs': {
        'enabled': True,
        'bid_ask_spread': 5 / 10000.0,
        'slippage': 0.0,
    },
    'rank_drop_gate': {
        'enabled': True,
        'min_rank_drop': 30,
    },
    'data_file': 'C:/Users/magil/MCI-GRU/data/raw/market/sp500_2019_universe_data.csv',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
    'label_t': 5,
}


def main():
    tc_config = CONFIG.get('transaction_costs', DEFAULT_CONFIG['transaction_costs'])
    rank_drop_config = CONFIG.get('rank_drop_gate', DEFAULT_CONFIG['rank_drop_gate'])

    print("Loading stock data once...")
    stock_data = load_stock_data(
        CONFIG['data_file'],
        start_date=CONFIG['test_start'],
        end_date=CONFIG['test_end'],
    )
    stock_data = calculate_forward_returns(stock_data, label_t=CONFIG['label_t'])

    curves = []
    benchmark_dates = None
    benchmark_values = None

    for label, rel_path in PREDICTION_PATHS:
        pred_dir = os.path.join(BASE_DIR, rel_path)
        if not os.path.isdir(pred_dir):
            print(f"Skipping {label}: not found -> {pred_dir}")
            continue
        print(f"Backtesting: {label} -> {pred_dir}")
        try:
            predictions_df = load_predictions(pred_dir)
            sim_results = simulate_trading_strategy(
                predictions_df=predictions_df,
                stock_data_df=stock_data,
                top_k=CONFIG['top_k'],
                label_t=CONFIG['label_t'],
                transaction_costs=tc_config,
                rank_drop_gate=rank_drop_config,
            )
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        dates = pd.to_datetime(sim_results['dates'])
        portfolio_values = np.cumprod(1 + sim_results['portfolio_returns'])
        curves.append((label, dates, portfolio_values))
        if benchmark_dates is None:
            benchmark_dates = dates
            benchmark_values = np.cumprod(1 + sim_results['benchmark_returns'])

    if not curves:
        print("No curves to plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, (label, dates, values) in enumerate(curves):
        ax.plot(dates, values, label=label, color=colors[i % len(colors)], linewidth=2)
    if benchmark_dates is not None and benchmark_values is not None:
        ax.plot(benchmark_dates, benchmark_values, 'k--', label='Benchmark', linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_ylabel('Cumulative return')
    ax.set_xlabel('Date')
    ax.set_title('Equity curves: daily rebalance, rank gate 30, top 20, 5 bps TC')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, 'all_seeds_equity_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()