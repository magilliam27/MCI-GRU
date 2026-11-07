"""
Backtest fairness of the replay's open-to-open variant (#340 kept it as a labelled variant).

Synthetic data with known overnight gaps and intraday moves checks that:
1. The open_to_open variant buys at open D+1 and sells at open D+2
2. Prediction dates map to the hold that starts the session after them
3. The rank-drop gate keeps a holding until its rank falls far enough
4. Holdings, trades and saved outputs come from the same book
"""

import os
import tempfile

import numpy as np
import pandas as pd

from mci_gru.evaluation import backtest_engine
from mci_gru.evaluation.replay_book import ReplayStrategy, build_replay_book


def create_synthetic_test_data():
    """
    Create synthetic data with known overnight gaps and intraday returns.

    Returns:
        stock_data_df: DataFrame with OHLC data
        predictions_df: DataFrame with predictions
        expected_results: Dict with expected portfolio returns
    """
    dates = pd.date_range("2025-01-01", "2025-01-10", freq="B")  # Business days only
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    data = []

    # Create data with predictable patterns:
    # - Consistent overnight gaps: +1% for all stocks
    # - Variable intraday returns: +0.5% for top-ranked stocks, -0.5% for others

    prev_close = {stock: 100.0 for stock in stocks}

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")

        for stock in stocks:
            # Apply overnight gap (+1%)
            open_price = prev_close[stock] * 1.01

            # Intraday move depends on stock
            if stock in ["AAPL", "MSFT"]:  # Top stocks
                close_price = open_price * 1.005  # +0.5% intraday
            else:  # Other stocks
                close_price = open_price * 0.995  # -0.5% intraday

            high = max(open_price, close_price) * 1.002
            low = min(open_price, close_price) * 0.998

            data.append(
                {
                    "kdcode": stock,
                    "dt": date_str,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "volume": 1000000,
                }
            )

            prev_close[stock] = close_price

    stock_data_df = pd.DataFrame(data)

    # Create predictions that rank AAPL and MSFT highest
    pred_data = []
    for date in dates[:-1]:  # Predictions for all but last day
        date_str = date.strftime("%Y-%m-%d")
        for _i, stock in enumerate(stocks):
            if stock in ["AAPL", "MSFT"]:
                score = 0.02  # High prediction
            else:
                score = -0.01  # Low prediction

            pred_data.append({"kdcode": stock, "dt": date_str, "score": score})

    predictions_df = pd.DataFrame(pred_data)

    # Expected results for current methodology:
    # Portfolio return uses open-to-open on entry day:
    # open_{t+2}/open_{t+1} - 1 = (1 + intraday) * (1 + overnight) - 1
    # For top stocks: intraday=0.5%, overnight=1.0%
    expected_open_to_open = (1.005 * 1.01) - 1.0  # 1.505%

    expected_results = {
        "avg_daily_return": expected_open_to_open,
        "num_days": len(dates) - 2,  # Need both entry day and next open for open-to-open
        "intraday_only_return": 0.005,
    }

    return stock_data_df, predictions_df, expected_results


def _open_book(predictions_df, stock_data_df, top_k=2, min_rank_drop=None):
    dates = sorted(predictions_df["dt"].unique())
    kdcodes = sorted(stock_data_df["kdcode"].unique())
    scores = (
        predictions_df.pivot(index="dt", columns="kdcode", values="score")
        .reindex(index=dates, columns=kdcodes)
        .to_numpy(dtype=float)
    )
    return build_replay_book(
        stock_data_df,
        scores,
        dates,
        kdcodes,
        strategy=ReplayStrategy.open_to_open(),
        top_k=top_k,
        min_rank_drop=min_rank_drop,
    )


def test_simulation_timing():
    """The open_to_open variant earns the open D+1 to open D+2 move."""
    stock_data_df, predictions_df, expected = create_synthetic_test_data()

    book = _open_book(predictions_df, stock_data_df)

    assert len(book.period_returns) == expected["num_days"]
    avg_return = float(np.mean(book.period_returns))
    assert abs(avg_return - expected["avg_daily_return"]) < 1e-9
    assert abs(avg_return - expected["intraday_only_return"]) > 0.001


def test_prediction_date_mapping():
    """A prediction on day t is held from the open of t+1 to the open of t+2."""
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    opens = [100.0, 102.0, 104.0]
    data = []
    for i, date in enumerate(dates):
        for stock in ["AAPL", "MSFT"]:
            close_price = opens[i] * (1 + (i + 1) * 0.01)
            data.append({"kdcode": stock, "dt": date, "open": opens[i], "close": close_price})
    stock_data_df = pd.DataFrame(data)
    predictions_df = pd.DataFrame(
        [
            {"kdcode": k, "dt": d, "score": 0.1}
            for d in ["2025-01-01", "2025-01-02"]
            for k in ["AAPL", "MSFT"]
        ]
    )

    book = _open_book(predictions_df, stock_data_df)

    # Jan 1 is held Jan 2 open to Jan 3 open; Jan 2 has no Jan 4 open, so no hold.
    assert book.signal_dates == ["2025-01-01"]
    assert abs(book.period_returns[0] - (opens[2] / opens[1] - 1.0)) < 1e-12


def _flat_panel(stocks, dates):
    return pd.DataFrame(
        [{"kdcode": s, "dt": d, "open": 100.0, "close": 100.0} for d in dates for s in stocks]
    )


def test_rank_drop_gate_eligible():
    """A holding whose rank fell by >= min_rank_drop is sold and its slot refilled."""
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]
    stocks = [f"S{i:02d}" for i in range(15)]
    preds = []
    for dt in dates[:3]:
        for r, s in enumerate(stocks):
            if dt == "2025-01-02":
                # S00 falls from rank 1 to rank 11: S01..S10 score above it
                score = 0.0 if s == "S00" else (0.5 - r * 0.01 if r < 11 else -0.1 - r * 0.01)
            else:
                score = 1.0 - r * 0.01
            preds.append({"kdcode": s, "dt": dt, "score": score})

    book = _open_book(pd.DataFrame(preds), _flat_panel(stocks, dates), top_k=2, min_rank_drop=10)

    held = pd.DataFrame(book.holdings)
    assert book.rank_gate["enabled"] is True and book.rank_gate["gate_exits"] == 1
    assert "S00" not in set(held.loc[held["signal_date"] == "2025-01-02", "kdcode"])


def test_rank_drop_gate_excluded():
    """With no rank drop >= min_rank_drop, holdings persist."""
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]
    stocks = [f"S{i:02d}" for i in range(12)]
    preds = [
        {"kdcode": s, "dt": dt, "score": 1.0 - r * 0.01}
        for dt in dates[:3]
        for r, s in enumerate(stocks)
    ]

    book = _open_book(pd.DataFrame(preds), _flat_panel(stocks, dates), top_k=2, min_rank_drop=10)

    held = pd.DataFrame(book.holdings).groupby("signal_date")["kdcode"].apply(set)
    assert book.rank_gate["gate_exits"] == 0
    assert all(names == {"S00", "S01"} for names in held)


def test_rank_drop_gate_disabled_regression():
    """With the gate off, the book is the plain top-k book."""
    stock_data_df, predictions_df, _ = create_synthetic_test_data()

    plain = _open_book(predictions_df, stock_data_df)
    gate_off = _open_book(predictions_df, stock_data_df, min_rank_drop=None)

    assert plain.rank_gate["enabled"] is False
    np.testing.assert_array_equal(plain.net_returns, gate_off.net_returns)


def test_portfolio_tracking_outputs_nonregression():
    """Holdings and trades come from the same book, and every output file is written."""
    stock_data_df, predictions_df, _ = create_synthetic_test_data()
    book = _open_book(predictions_df, stock_data_df)

    assert book.holdings and book.trades
    holdings = pd.DataFrame(book.holdings)
    contribution = holdings.groupby("signal_date")["contribution"].sum()
    np.testing.assert_allclose(contribution.to_numpy(), book.period_returns, rtol=1e-12)

    config = {"top_k": 2, "label_t": 5, "transaction_costs": {"enabled": False}}
    results = backtest_engine.replay_metrics(book, config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        backtest_engine.save_backtest_results(results, tmp_dir, book=book, config=config)
        for name in [
            "backtest_results.csv",
            "backtest_metrics.json",
            "daily_returns.csv",
            "cumulative_performance.csv",
            "monthly_performance.csv",
            "summary.txt",
            "holdings.csv",
            "holdings_summary.csv",
            "trade_journal.csv",
            "missing_close_events.csv",
        ]:
            assert os.path.exists(os.path.join(tmp_dir, name)), name
        trades = pd.read_csv(os.path.join(tmp_dir, "trade_journal.csv"))
        first = trades["date"].min()
        assert set(trades.loc[trades["date"] == first, "action"]) == {"BUY"}
