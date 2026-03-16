"""
Daily report generator for the paper trading pipeline.

Reads the per-day artifacts (scores, target portfolio, orders, fills,
holdings, daily returns) and the cumulative performance.csv, then
produces a human-readable markdown report and an equity curve chart.

Usage:
    python paper_trade/scripts/report.py
    python paper_trade/scripts/report.py --date 2026-03-06
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

DEFAULT_RESULTS_DIR = "paper_trade/results"
DEFAULT_STATE_DIR = "paper_trade/state"

TRADING_DAYS_PER_YEAR = 252


def load_csv_safe(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists, else return empty DataFrame."""
    if path.exists():
        return pd.read_csv(str(path))
    return pd.DataFrame()


def compute_rolling_stats(perf_df: pd.DataFrame, window: int = 20) -> dict:
    """Compute rolling volatility, Sharpe proxy, and max drawdown."""
    if len(perf_df) < 2 or "net_return" not in perf_df.columns:
        return {
            "rolling_vol_ann": np.nan,
            "sharpe_proxy": np.nan,
            "max_drawdown": 0.0,
            "num_days": len(perf_df) if "net_return" in perf_df.columns else 0,
            "win_rate": np.nan,
            "avg_daily_return": np.nan,
        }

    returns = perf_df["net_return"].values
    n = len(returns)

    lookback = min(window, n)
    recent = returns[-lookback:]
    vol = np.std(recent, ddof=1) if lookback > 1 else 0.0
    vol_ann = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    mean_ret = np.mean(recent)
    sharpe = (mean_ret / vol * np.sqrt(TRADING_DAYS_PER_YEAR)) if vol > 0 else 0.0

    max_dd = perf_df["drawdown"].min() if "drawdown" in perf_df.columns else 0.0

    win_rate = np.mean(returns > 0) if n > 0 else np.nan
    avg_daily = np.mean(returns) if n > 0 else np.nan

    return {
        "rolling_vol_ann": vol_ann,
        "sharpe_proxy": sharpe,
        "max_drawdown": max_dd,
        "num_days": n,
        "win_rate": win_rate,
        "avg_daily_return": avg_daily,
    }


def generate_equity_chart(perf_df: pd.DataFrame, output_path: Path):
    """Generate an equity curve PNG with portfolio vs benchmark."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping equity chart")
        return

    if len(perf_df) < 2:
        print("  Not enough data for equity chart (need >= 2 days)")
        return

    dates = pd.to_datetime(perf_df["dt"])
    equity = perf_df["equity"].values

    bm_equity = np.cumprod(1.0 + perf_df["benchmark_return"].values)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={"hspace": 0.08})

    ax1 = axes[0]
    ax1.plot(dates, equity, color="#2563eb", linewidth=1.8, label="Portfolio (net)")
    ax1.plot(dates, bm_equity, color="#9ca3af", linewidth=1.2,
             linestyle="--", label="Benchmark")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title("MCI-GRU Paper Trading — Equity Curve", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color="#6b7280", linewidth=0.5, linestyle=":")

    ax2 = axes[1]
    dd = perf_df["drawdown"].values * 100
    ax2.fill_between(dates, dd, 0, color="#ef4444", alpha=0.3)
    ax2.plot(dates, dd, color="#ef4444", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity chart saved: {output_path}")


def build_markdown_report(
    date: str,
    perf_df: pd.DataFrame,
    target_portfolio: pd.DataFrame,
    orders: pd.DataFrame,
    holdings: pd.DataFrame,
    daily_return: pd.DataFrame,
    rolling: dict,
    state_dir: Path,
) -> str:
    """Build the full markdown report string."""
    lines = []

    lines.append(f"# MCI-GRU Paper Trading Report — {date}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # --- Performance Summary ---
    lines.append("## Performance")
    lines.append("")

    has_perf = not perf_df.empty and "dt" in perf_df.columns
    today_perf = perf_df[perf_df["dt"] == date] if has_perf else pd.DataFrame()
    if not today_perf.empty:
        row = today_perf.iloc[0]

        def _fmt_pct(val, default=0):
            """Format a percentage, returning 'N/A' if NaN."""
            v = val if not pd.isna(val) else default
            if pd.isna(v):
                return "N/A"
            return f"{v:+.4%}"

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Daily Return | {_fmt_pct(row.get('daily_return', 0))} |")
        lines.append(f"| Net Return (after costs) | {_fmt_pct(row.get('net_return', 0))} |")
        lines.append(f"| Benchmark Return | {_fmt_pct(row.get('benchmark_return'), np.nan)} |")
        lines.append(f"| Excess Return | {_fmt_pct(row.get('excess_return'), np.nan)} |")
        lines.append(f"| Cumulative Return | {_fmt_pct(row.get('cum_return', 0))} |")
        lines.append(f"| Cumulative Benchmark | {_fmt_pct(row.get('cum_benchmark', 0))} |")
        lines.append(f"| Equity | {row.get('equity', 1):.4f} |")
        lines.append(f"| Drawdown | {row.get('drawdown', 0):.4%} |")
    else:
        lines.append("*No return data available for this date.*")
    lines.append("")

    # --- Rolling Stats ---
    lines.append("## Rolling Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Trading Days | {rolling['num_days']} |")

    if not np.isnan(rolling["rolling_vol_ann"]):
        lines.append(f"| 20D Annualized Vol | {rolling['rolling_vol_ann']:.2%} |")
    else:
        lines.append(f"| 20D Annualized Vol | N/A |")

    if not np.isnan(rolling["sharpe_proxy"]):
        lines.append(f"| 20D Sharpe Proxy | {rolling['sharpe_proxy']:.2f} |")
    else:
        lines.append(f"| 20D Sharpe Proxy | N/A |")

    lines.append(f"| Max Drawdown (to date) | {rolling['max_drawdown']:.4%} |")

    if not np.isnan(rolling["win_rate"]):
        lines.append(f"| Win Rate | {rolling['win_rate']:.1%} |")
    else:
        lines.append(f"| Win Rate | N/A |")

    if not np.isnan(rolling["avg_daily_return"]):
        lines.append(f"| Avg Daily Return | {rolling['avg_daily_return']:+.4%} |")
    else:
        lines.append(f"| Avg Daily Return | N/A |")
    lines.append("")

    # --- Portfolio Snapshot ---
    lines.append("## Portfolio Holdings")
    lines.append("")

    if not target_portfolio.empty:
        tp = target_portfolio.copy()

        if not holdings.empty and "stock_return" in holdings.columns:
            ret_map = holdings.set_index("kdcode")["stock_return"].to_dict()
            contrib_map = holdings.set_index("kdcode")["contribution"].to_dict()
            tp["day_return"] = tp["kdcode"].map(ret_map)
            tp["contribution"] = tp["kdcode"].map(contrib_map)
        else:
            tp["day_return"] = np.nan
            tp["contribution"] = np.nan

        lines.append(f"| Stock | Rank | Score | Weight | Day Return | Contribution | Entry Date |")
        lines.append(f"|-------|------|-------|--------|------------|-------------|------------|")
        for _, row in tp.iterrows():
            dr = f"{row['day_return']:+.4%}" if pd.notna(row.get("day_return")) else "—"
            ct = f"{row['contribution']:+.4%}" if pd.notna(row.get("contribution")) else "—"
            lines.append(
                f"| {row['kdcode']} "
                f"| {int(row['rank']) if pd.notna(row.get('rank')) else '—'} "
                f"| {row['score']:.5f} "
                f"| {row['weight']:.1%} "
                f"| {dr} "
                f"| {ct} "
                f"| {row.get('entry_date', '—')} |"
            )
    else:
        lines.append("*No target portfolio data.*")
    lines.append("")

    # --- Changes ---
    lines.append("## Changes")
    lines.append("")

    if not orders.empty:
        sells = orders[orders["side"] == "SELL"]
        buys = orders[orders["side"] == "BUY"]

        if not sells.empty:
            lines.append(f"### Exits ({len(sells)})")
            lines.append("")
            lines.append(f"| Stock | Reason |")
            lines.append(f"|-------|--------|")
            for _, row in sells.iterrows():
                lines.append(f"| {row['kdcode']} | {row.get('reason', '')} |")
            lines.append("")

        if not buys.empty:
            lines.append(f"### New Entries ({len(buys)})")
            lines.append("")
            lines.append(f"| Stock | Reason |")
            lines.append(f"|-------|--------|")
            for _, row in buys.iterrows():
                lines.append(f"| {row['kdcode']} | {row.get('reason', '')} |")
            lines.append("")

        if sells.empty and buys.empty:
            lines.append("No changes today.")
            lines.append("")
    else:
        lines.append("No changes today.")
        lines.append("")

    # --- Trading Stats ---
    lines.append("## Trading Stats")
    lines.append("")

    if not today_perf.empty:
        row = today_perf.iloc[0]
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Trades | {int(row.get('num_trades', 0))} |")
        lines.append(f"| Turnover | {row.get('turnover', 0):.2%} |")
        lines.append(f"| Est. Cost | {row.get('est_cost', 0):.4%} |")
        lines.append(f"| Holdings | {int(row.get('num_holdings', 0))} |")
    else:
        lines.append("*No trading stats available.*")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by MCI-GRU Paper Trading System*")
    lines.append("")

    return "\n".join(lines)


def build_json_report(
    date: str,
    perf_df: pd.DataFrame,
    target_portfolio: pd.DataFrame,
    orders: pd.DataFrame,
    rolling: dict,
) -> dict:
    """Build the machine-readable JSON report."""
    has_perf = not perf_df.empty and "dt" in perf_df.columns
    today_perf = perf_df[perf_df["dt"] == date] if has_perf else pd.DataFrame()
    perf_dict = today_perf.iloc[0].to_dict() if not today_perf.empty else {}

    report = {
        "date": date,
        "generated_at": datetime.now().isoformat(),
        "performance": {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in perf_dict.items()},
        "rolling_stats": {k: float(v) if isinstance(v, (np.floating, float)) else v
                          for k, v in rolling.items()},
        "holdings": target_portfolio.to_dict(orient="records") if not target_portfolio.empty else [],
        "orders": orders.to_dict(orient="records") if not orders.empty else [],
    }

    return report


def find_report_date(results_dir: Path, requested_date: str = None) -> str:
    """Determine which date to report on."""
    if requested_date:
        return requested_date

    perf_path = results_dir / "performance.csv"
    if perf_path.exists():
        perf = pd.read_csv(str(perf_path))
        if not perf.empty and "dt" in perf.columns:
            return perf["dt"].iloc[-1]

    dated_dirs = sorted(
        [d for d in results_dir.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        key=lambda d: d.name,
    )
    if dated_dirs:
        return dated_dirs[-1].name

    raise FileNotFoundError("No results data found. Run the pipeline first.")


def main():
    parser = argparse.ArgumentParser(description="MCI-GRU daily report generator.")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing results",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help="Directory for state files",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to report on (default: latest)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip equity curve chart generation",
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    state_dir = PROJECT_ROOT / args.state_dir

    print("=" * 70)
    print("  MCI-GRU Daily Report Generator")
    print("=" * 70)

    date = find_report_date(results_dir, args.date)
    print(f"  Report date: {date}")

    day_dir = results_dir / date

    perf_df = load_csv_safe(results_dir / "performance.csv")
    target_portfolio = load_csv_safe(day_dir / "target_portfolio.csv")
    orders = load_csv_safe(day_dir / "orders.csv")
    holdings = load_csv_safe(day_dir / "holdings.csv")
    daily_return = load_csv_safe(day_dir / "daily_return.csv")

    rolling = compute_rolling_stats(perf_df)

    md_report = build_markdown_report(
        date, perf_df, target_portfolio, orders, holdings,
        daily_return, rolling, state_dir,
    )

    day_dir.mkdir(parents=True, exist_ok=True)
    md_path = day_dir / f"daily_report_{date}.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"  Markdown report: {md_path}")

    json_report = build_json_report(date, perf_df, target_portfolio, orders, rolling)
    json_path = day_dir / f"daily_report_{date}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"  JSON report:     {json_path}")

    if not args.no_chart and len(perf_df) >= 2:
        chart_path = results_dir / "equity_curve.png"
        generate_equity_chart(perf_df, chart_path)

    print(f"\n{'-' * 70}")
    print(md_report)
    print("=" * 70)


if __name__ == "__main__":
    main()
