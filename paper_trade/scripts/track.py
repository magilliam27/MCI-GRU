"""
Execution simulation and return tracking for the paper trading pipeline.

Runs each evening after data refresh. Two responsibilities:
  1. Record fill prices for orders placed the prior night (today's open).
  2. Compute open-to-open returns for positions held from yesterday's open
     to today's open, matching the backtest convention in backtest_sp500.py.

The nightly pipeline ordering is:
  refresh_data -> track -> infer -> portfolio

Usage:
    python paper_trade/scripts/track.py
    python paper_trade/scripts/track.py --date 2026-03-06
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV = "data/raw/market/sp500_2019_universe_data_through_2026.csv"
DEFAULT_STATE_DIR = "paper_trade/state"
DEFAULT_RESULTS_DIR = "paper_trade/results"
BENCHMARK_TICKER = "SPY.P"

SLIPPAGE_BPS = 5
BID_ASK_BPS = 5


def load_open_prices(csv_path: str, dates: list, kdcodes: list) -> pd.DataFrame:
    """Load open prices for specific dates and stocks from the master CSV."""
    df = pd.read_csv(csv_path, usecols=["kdcode", "dt", "open"])
    df = df[df["dt"].isin(dates) & df["kdcode"].isin(kdcodes)]
    return df


def load_state(state_dir: Path) -> dict:
    """Load current holdings state."""
    path = state_dir / "current_holdings.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_fill_state(state_dir: Path) -> dict:
    """Load the fill-tracking state (positions with entry prices)."""
    path = state_dir / "active_positions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_fill_state(state_dir: Path, positions: dict):
    """Save active positions with entry prices."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "active_positions.json", "w") as f:
        json.dump(positions, f, indent=2)


def get_trading_dates(csv_path: str, n_recent: int = 5) -> list:
    """Get the most recent N trading dates from the master CSV."""
    df = pd.read_csv(csv_path, usecols=["dt"])
    dates = sorted(df["dt"].unique())
    return dates[-n_recent:]


def get_all_trading_dates(csv_path: str) -> list:
    """All unique trading dates in the master CSV, sorted."""
    df = pd.read_csv(csv_path, usecols=["dt"])
    return sorted(df["dt"].unique().tolist())


def get_previous_trading_date(all_dates: list, date: str) -> str | None:
    """Previous trading day before ``date`` in the calendar, or None if first."""
    if date not in all_dates:
        return None
    i = all_dates.index(date)
    return all_dates[i - 1] if i > 0 else None


def compute_fills(
    orders_path: Path,
    open_prices: pd.DataFrame,
    fill_date: str,
    results_dir: Path,
) -> pd.DataFrame:
    """
    Record fill prices for orders. Fill price = open on fill_date.
    Returns fills DataFrame.
    """
    if not orders_path.exists():
        return pd.DataFrame(columns=["kdcode", "dt", "side", "fill_price", "weight"])

    orders = pd.read_csv(str(orders_path))
    if orders.empty:
        return pd.DataFrame(columns=["kdcode", "dt", "side", "fill_price", "weight"])

    price_map = open_prices[open_prices["dt"] == fill_date].set_index("kdcode")["open"].to_dict()

    records = []
    for _, row in orders.iterrows():
        kd = row["kdcode"]
        records.append({
            "kdcode": kd,
            "dt": fill_date,
            "side": row["side"],
            "fill_price": round(price_map.get(kd, np.nan), 4),
            "reason": row.get("reason", ""),
        })

    fills_df = pd.DataFrame(records)

    fill_dir = results_dir / fill_date
    fill_dir.mkdir(parents=True, exist_ok=True)
    fills_df.to_csv(str(fill_dir / "fills.csv"), index=False)

    return fills_df


def compute_daily_returns(
    active_positions: dict,
    open_prices: pd.DataFrame,
    return_date: str,
    prev_date: str,
) -> dict:
    """
    Compute open-to-open returns for active positions.
    return = open(return_date) / open(prev_date) - 1
    Matches backtest_sp500.py lines 726-731 convention.
    """
    if not active_positions or not active_positions.get("positions"):
        return None

    positions = active_positions["positions"]
    entry_date_for_positions = active_positions.get("fill_date")

    prev_prices = open_prices[open_prices["dt"] == prev_date].set_index("kdcode")["open"].to_dict()
    curr_prices = open_prices[open_prices["dt"] == return_date].set_index("kdcode")["open"].to_dict()

    stock_returns = []
    holdings_records = []

    for pos in positions:
        kd = pos["kdcode"]
        weight = pos["weight"]
        open_prev = prev_prices.get(kd)
        open_curr = curr_prices.get(kd)

        if open_prev is None or open_curr is None or open_prev == 0:
            continue

        ret = (open_curr / open_prev) - 1.0
        contribution = ret * weight

        stock_returns.append(ret)
        holdings_records.append({
            "kdcode": kd,
            "dt": return_date,
            "weight": round(weight, 6),
            "entry_date": pos.get("entry_date", entry_date_for_positions),
            "open_prev": round(open_prev, 4),
            "open_curr": round(open_curr, 4),
            "stock_return": round(ret, 6),
            "contribution": round(contribution, 6),
        })

    if not stock_returns:
        return None

    portfolio_return = np.mean(stock_returns)

    return {
        "portfolio_return": portfolio_return,
        "holdings_records": holdings_records,
        "num_stocks": len(stock_returns),
    }


def compute_benchmark_return(
    open_prices: pd.DataFrame,
    return_date: str,
    prev_date: str,
) -> float:
    """
    Compute SPY open-to-open return as the benchmark.
    """
    prev = open_prices[
        (open_prices["dt"] == prev_date)
        & (open_prices["kdcode"] == BENCHMARK_TICKER)
    ]["open"]
    curr = open_prices[
        (open_prices["dt"] == return_date)
        & (open_prices["kdcode"] == BENCHMARK_TICKER)
    ]["open"]

    if prev.empty or curr.empty or prev.iloc[0] == 0:
        return np.nan
    return float((curr.iloc[0] / prev.iloc[0]) - 1.0)


def update_performance_csv(
    results_dir: Path,
    date: str,
    portfolio_return: float,
    benchmark_return: float,
    turnover: float,
    num_trades: int,
    num_holdings: int,
):
    """Append today's performance to the cumulative performance.csv."""
    perf_path = results_dir / "performance.csv"

    cost_per_side = (BID_ASK_BPS + SLIPPAGE_BPS) / 10_000
    est_cost = turnover * cost_per_side * 2

    net_return = portfolio_return - est_cost
    bm_safe = benchmark_return if not np.isnan(benchmark_return) else 0.0
    excess_return = portfolio_return - bm_safe

    if perf_path.exists():
        perf_df = pd.read_csv(str(perf_path))
        if date in perf_df["dt"].values:
            perf_df = perf_df[perf_df["dt"] != date]

        prev_cum = perf_df["cum_return"].iloc[-1] if len(perf_df) > 0 else 0.0
        prev_cum_bm = perf_df["cum_benchmark"].iloc[-1] if len(perf_df) > 0 else 0.0
        prev_equity = perf_df["equity"].iloc[-1] if len(perf_df) > 0 else 1.0
        prev_peak = perf_df["peak_equity"].iloc[-1] if len(perf_df) > 0 else 1.0
    else:
        perf_df = pd.DataFrame()
        prev_cum = 0.0
        prev_cum_bm = 0.0
        prev_equity = 1.0
        prev_peak = 1.0

    if np.isnan(prev_cum_bm):
        prev_cum_bm = 0.0

    cum_return = prev_cum + net_return
    cum_benchmark = prev_cum_bm + bm_safe
    equity = prev_equity * (1.0 + net_return)
    peak_equity = max(prev_peak, equity)
    drawdown = (equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0

    new_row = pd.DataFrame([{
        "dt": date,
        "daily_return": round(portfolio_return, 6),
        "net_return": round(net_return, 6),
        "benchmark_return": round(benchmark_return, 6),
        "excess_return": round(excess_return, 6),
        "cum_return": round(cum_return, 6),
        "cum_benchmark": round(cum_benchmark, 6),
        "equity": round(equity, 6),
        "peak_equity": round(peak_equity, 6),
        "drawdown": round(drawdown, 6),
        "turnover": round(turnover, 4),
        "est_cost": round(est_cost, 6),
        "num_trades": num_trades,
        "num_holdings": num_holdings,
    }])

    perf_df = pd.concat([perf_df, new_row], ignore_index=True)
    perf_df.to_csv(str(perf_path), index=False)

    return new_row.iloc[0].to_dict()


def update_trade_log(results_dir: Path, fills_df: pd.DataFrame):
    """Append fills to the cumulative trade log."""
    log_path = results_dir / "trade_log.csv"

    if fills_df.empty:
        return

    if log_path.exists():
        existing = pd.read_csv(str(log_path))
        combined = pd.concat([existing, fills_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["kdcode", "dt", "side"], keep="last")
    else:
        combined = fills_df

    combined.to_csv(str(log_path), index=False)


def build_active_positions(
    holdings_state: dict,
    fills_df: pd.DataFrame,
    fill_date: str,
) -> dict:
    """
    Build the active positions dict from current holdings + fill prices.
    This is what we track for computing next day's returns.
    """
    if holdings_state is None:
        return None

    holdings = holdings_state.get("holdings", [])
    if not holdings:
        return None

    fill_prices = {}
    if not fills_df.empty:
        buys = fills_df[fills_df["side"] == "BUY"]
        fill_prices = buys.set_index("kdcode")["fill_price"].to_dict()

    positions = []
    for h in holdings:
        kd = h["kdcode"]
        positions.append({
            "kdcode": kd,
            "weight": h["weight"],
            "entry_date": h.get("entry_date", fill_date),
            "fill_price": fill_prices.get(kd),
        })

    return {
        "fill_date": fill_date,
        "decision_date": holdings_state.get("date"),
        "positions": positions,
    }


def main():
    parser = argparse.ArgumentParser(description="MCI-GRU execution tracking.")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to master OHLCV CSV",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help="Directory for persistent state files",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for results output",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Processing date (default: latest in CSV)",
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    state_dir = PROJECT_ROOT / args.state_dir
    results_dir = PROJECT_ROOT / args.results_dir

    if not csv_path.exists():
        print(f"ERROR: Master CSV not found: {csv_path}")
        sys.exit(1)

    print("=" * 70)
    print("  MCI-GRU Execution Tracker")
    print("=" * 70)

    all_dates = get_all_trading_dates(str(csv_path))
    if not all_dates:
        print("ERROR: No trading dates found in CSV")
        sys.exit(1)

    today = args.date or all_dates[-1]
    if today not in all_dates:
        print(f"ERROR: Date {today} not found in master CSV")
        sys.exit(1)

    yesterday = get_previous_trading_date(all_dates, today)

    print(f"  Today:     {today}")
    print(f"  Yesterday: {yesterday or 'N/A'}")

    holdings_state = load_state(state_dir)
    prev_positions = load_fill_state(state_dir)

    if holdings_state is None:
        print("  No holdings state found. Run portfolio.py first.")
        print("=" * 70)
        return

    decision_date = holdings_state.get("date")
    holdings = holdings_state.get("holdings", [])
    print(f"  Decision date: {decision_date}")
    print(f"  Holdings:      {len(holdings)} stocks")

    all_kdcodes = [h["kdcode"] for h in holdings]
    if prev_positions and prev_positions.get("positions"):
        prev_kdcodes = [p["kdcode"] for p in prev_positions["positions"]]
        all_kdcodes = list(set(all_kdcodes + prev_kdcodes))
    if BENCHMARK_TICKER not in all_kdcodes:
        all_kdcodes.append(BENCHMARK_TICKER)

    dates_needed = [d for d in [yesterday, today] if d is not None]
    open_prices = load_open_prices(str(csv_path), dates_needed, all_kdcodes)
    print(f"  Loaded {len(open_prices)} price records")

    orders_dir = results_dir / decision_date
    orders_path = orders_dir / "orders.csv"
    fills_df = compute_fills(orders_path, open_prices, today, results_dir)
    num_trades = len(fills_df)
    print(f"\n  FILLS ({today}):")
    if num_trades > 0:
        print(fills_df[["kdcode", "side", "fill_price"]].to_string(index=False))
    else:
        print("    No trades")

    active_positions = build_active_positions(holdings_state, fills_df, today)
    save_fill_state(state_dir, active_positions)

    if prev_positions and yesterday:
        print(f"\n  RETURNS (open {yesterday} -> open {today}):")
        returns_data = compute_daily_returns(
            prev_positions, open_prices, today, yesterday,
        )

        if returns_data:
            portfolio_return = returns_data["portfolio_return"]

            benchmark_return = compute_benchmark_return(
                open_prices, today, yesterday,
            )

            prev_held = set(p["kdcode"] for p in prev_positions.get("positions", []))
            curr_held = set(h["kdcode"] for h in holdings)
            sold = prev_held - curr_held
            bought = curr_held - prev_held
            top_k = len(holdings) if holdings else 20
            turnover = (len(sold) + len(bought)) / (2 * top_k)

            day_dir = results_dir / today
            day_dir.mkdir(parents=True, exist_ok=True)

            holdings_df = pd.DataFrame(returns_data["holdings_records"])
            holdings_df.to_csv(str(day_dir / "holdings.csv"), index=False)

            daily_ret_df = pd.DataFrame([{
                "dt": today,
                "portfolio_return": round(portfolio_return, 6),
                "benchmark_return": round(benchmark_return, 6),
                "excess_return": round(portfolio_return - benchmark_return, 6),
                "num_holdings": returns_data["num_stocks"],
                "turnover": round(turnover, 4),
            }])
            daily_ret_df.to_csv(str(day_dir / "daily_return.csv"), index=False)

            perf = update_performance_csv(
                results_dir, today, portfolio_return, benchmark_return,
                turnover, num_trades, returns_data["num_stocks"],
            )

            print(f"    Portfolio:  {portfolio_return:+.4%}")
            print(f"    Benchmark:  {benchmark_return:+.4%}")
            print(f"    Excess:     {portfolio_return - benchmark_return:+.4%}")
            print(f"    Cum return: {perf['cum_return']:+.4%}")
            print(f"    Drawdown:   {perf['drawdown']:.4%}")
            print(f"    Equity:     {perf['equity']:.4f}")
        else:
            print("    No returns to compute (missing price data)")
    else:
        if not prev_positions:
            print("\n  First run - no prior positions to compute returns from.")
        elif not yesterday:
            print("\n  No previous trading day available for return calculation.")

    update_trade_log(results_dir, fills_df)

    print(f"\n  Files saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
