import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

import mci_gru_sp500 as mci


def collect_targets(samples):
    y_true = []
    dates = []
    for sample in samples:
        y_true.append(sample["labels"].numpy())
        dates.append(sample["date"])
    y_true = np.stack(y_true, axis=0)
    return y_true, pd.to_datetime(dates)


def compute_train_stats(normalized_data, tickers, train_end, lookback=None):
    returns = []
    for t in tickers:
        series = normalized_data[t]["Return"].loc[:train_end]
        if lookback is not None:
            series = series.iloc[-lookback:]
        returns.append(series.values)
    return np.stack(returns, axis=0)


def baseline_zero(y_true):
    return np.zeros_like(y_true)


def baseline_mean_per_stock(train_returns, y_true):
    per_stock_mean = train_returns.mean(axis=1)
    return np.tile(per_stock_mean, (y_true.shape[0], 1))


def baseline_global_mean(train_returns, y_true):
    global_mean = train_returns.mean()
    return np.full_like(y_true, global_mean)


def baseline_last_return(normalized_data, tickers, dates):
    preds = []
    for dt in dates:
        vals = []
        for t in tickers:
            series = normalized_data[t]["Return"]
            idx = series.index.get_loc(dt)
            if idx == 0:
                vals.append(0.0)
            else:
                vals.append(float(series.iloc[idx - 1]))
        preds.append(vals)
    return np.array(preds)


def baseline_rolling_mean(normalized_data, tickers, dates, window=5):
    preds = []
    for dt in dates:
        vals = []
        for t in tickers:
            series = normalized_data[t]["Return"]
            idx = series.index.get_loc(dt)
            start = max(idx - window, 0)
            window_vals = series.iloc[start:idx].values
            vals.append(float(window_vals.mean()) if len(window_vals) else 0.0)
        preds.append(vals)
    return np.array(preds)


def baseline_cross_sectional_mean(normalized_data, tickers, dates):
    preds = []
    for dt in dates:
        daily_vals = [float(normalized_data[t]["Return"].loc[dt]) for t in tickers]
        mean_val = float(np.mean(daily_vals))
        preds.append([mean_val] * len(tickers))
    return np.array(preds)


def mse(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))


def run_baselines(ticker_csv_path, use_train_mean_lookback=None):
    print("Loading tickers and data...")
    tickers = mci.load_tickers_from_csv(ticker_csv_path)
    stock_data = mci.download_stock_data(tickers, mci.DATA_START, mci.DATA_END)
    processed = mci.compute_features(stock_data)
    aligned, common_dates = mci.align_stock_data(
        processed, required_start=mci.TRAIN_START, required_end=mci.TEST_END
    )
    tickers = sorted(aligned.keys())
    normalized, _, _ = mci.normalize_features(aligned, mci.TRAIN_END)

    all_dates = [d for d in common_dates if d >= pd.Timestamp(mci.TRAIN_START)]
    samples = mci.create_dataset(
        normalized,
        tickers,
        all_dates,
        hist_days=mci.CONFIG["hist_days"],
        label_days=mci.CONFIG["label_days"],
    )
    train_samples, _, test_samples = mci.split_dataset(
        samples, mci.TRAIN_END, mci.VAL_END
    )

    y_true, test_dates = collect_targets(test_samples)
    train_returns = compute_train_stats(
        normalized, tickers, mci.TRAIN_END, lookback=use_train_mean_lookback
    )

    baselines: Dict[str, np.ndarray] = {
        "zero": baseline_zero(y_true),
        "mean_per_stock": baseline_mean_per_stock(train_returns, y_true),
        "global_mean": baseline_global_mean(train_returns, y_true),
        "last_return": baseline_last_return(normalized, tickers, test_dates),
        "rolling_mean_5": baseline_rolling_mean(normalized, tickers, test_dates, 5),
        "rolling_mean_10": baseline_rolling_mean(normalized, tickers, test_dates, 10),
        "cross_sectional_mean": baseline_cross_sectional_mean(
            normalized, tickers, test_dates
        ),
    }

    results = {name: mse(preds, y_true) for name, preds in baselines.items()}
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline MSE evaluation for MCI-GRU targets."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Path to CSV with tickers.",
    )
    parser.add_argument(
        "--train-mean-lookback",
        type=int,
        default=None,
        help="Optional lookback (days) for mean baselines.",
    )
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    ticker_path = args.tickers
    if not ticker_path:
        default_path = "ticker.csv"
        if not os.path.exists(default_path):
            parser.print_usage()
            raise ValueError(
                "Missing --tickers argument and default 'ticker.csv' "
                "not found in the current directory."
            )
        ticker_path = default_path

    results = run_baselines(ticker_path, args.train_mean_lookback)

    print("\nBaseline MSE results")
    print("=" * 30)
    for name, val in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:24s}: {val:.6f}")


if __name__ == "__main__":
    main()
