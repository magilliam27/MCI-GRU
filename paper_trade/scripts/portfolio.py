"""
Portfolio decision engine for the paper trading pipeline.

Loads today's scores, applies the rank-drop gate against prior holdings,
determines exits and new entries, and outputs a target portfolio + orders.

Replicates the rank-drop gate logic from tests/backtest_sp500.py lines 1052-1091.

Usage:
    python paper_trade/scripts/portfolio.py
    python paper_trade/scripts/portfolio.py --date 2026-03-06
    python paper_trade/scripts/portfolio.py --scores-dir paper_trade/results
    python paper_trade/scripts/portfolio.py --top-k 20 --min-rank-drop 30
"""

# ruff: noqa: E402, I001

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mci_gru.evaluation.portfolio import (  # noqa: E402
    rank_scores,
    apply_rank_drop_gate as shared_apply_rank_drop_gate,
)

DEFAULT_SCORES_DIR = "paper_trade/results"
DEFAULT_STATE_DIR = "paper_trade/state"
DEFAULT_TOP_K = 20
DEFAULT_MIN_RANK_DROP = 30


def load_scores(scores_path: Path) -> pd.DataFrame:
    """Load and rank a scores CSV. Adds rank column if missing."""
    df = pd.read_csv(str(scores_path))

    required = {"kdcode", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scores CSV missing columns: {missing}")

    return rank_scores(df)


def load_state(state_dir: Path) -> tuple:
    """
    Load previous holdings and ranks from state files.
    Returns (holdings_list, prev_ranks_dict, prev_date) or (None, None, None)
    on first run.
    """
    holdings_path = state_dir / "current_holdings.json"
    ranks_path = state_dir / "prev_ranks.json"

    if not holdings_path.exists() or not ranks_path.exists():
        return None, None, None

    with open(holdings_path) as f:
        holdings_data = json.load(f)

    with open(ranks_path) as f:
        ranks_data = json.load(f)

    holdings_list = holdings_data.get("holdings", [])
    prev_ranks = ranks_data.get("ranks", {})
    prev_date = holdings_data.get("date")

    return holdings_list, prev_ranks, prev_date


def save_state(
    state_dir: Path,
    date: str,
    holdings: list,
    ranks: dict,
):
    """Persist current holdings and rank table for next run."""
    state_dir.mkdir(parents=True, exist_ok=True)

    holdings_data = {
        "date": date,
        "holdings": holdings,
    }
    with open(state_dir / "current_holdings.json", "w") as f:
        json.dump(holdings_data, f, indent=2)

    ranks_data = {
        "date": date,
        "ranks": ranks,
    }
    with open(state_dir / "prev_ranks.json", "w") as f:
        json.dump(ranks_data, f, indent=2)


def apply_rank_drop_gate(
    scores_df: pd.DataFrame,
    prev_holdings: list,
    prev_ranks: dict,
    top_k: int,
    min_rank_drop: int,
) -> dict:
    """
    Core portfolio decision logic. Replicates backtest_sp500.py lines 1052-1091.

    Returns dict with keys:
        target_stocks, survivors, exits, new_entries, exit_details
    """
    return shared_apply_rank_drop_gate(
        scores_df=scores_df,
        prev_holdings=prev_holdings,
        prev_ranks=prev_ranks,
        top_k=top_k,
        min_rank_drop=min_rank_drop,
    )


def build_target_portfolio(
    target_stocks: list,
    scores_df: pd.DataFrame,
    date: str,
    prev_holdings: list,
) -> pd.DataFrame:
    """Build the target portfolio DataFrame with weights and metadata."""
    weight = 1.0 / len(target_stocks) if target_stocks else 0.0

    rank_map = scores_df.set_index("kdcode")["rank"].to_dict()
    score_map = scores_df.set_index("kdcode")["score"].to_dict()

    entry_dates = {}
    if prev_holdings:
        for h in prev_holdings:
            entry_dates[h["kdcode"]] = h.get("entry_date", date)

    records = []
    for kd in target_stocks:
        records.append(
            {
                "kdcode": kd,
                "dt": date,
                "weight": round(weight, 6),
                "rank": rank_map.get(kd),
                "score": round(score_map.get(kd, 0.0), 5),
                "entry_date": entry_dates.get(kd, date),
            }
        )

    df = pd.DataFrame(records)
    return df


def build_orders(
    decision: dict,
    date: str,
) -> pd.DataFrame:
    """Build the orders DataFrame from the portfolio decision."""
    records = []

    for detail in decision["exit_details"]:
        reason = detail.get("reason", "rank_drop_gate")
        if reason == "rank_drop_gate":
            reason_str = (
                f"rank_drop {detail['prev_rank']}->{detail['curr_rank']} ({detail['rank_drop']:+d})"
            )
        else:
            reason_str = reason
        records.append(
            {
                "kdcode": detail["kdcode"],
                "dt": date,
                "side": "SELL",
                "reason": reason_str,
            }
        )

    for kd in decision["new_entries"]:
        records.append(
            {
                "kdcode": kd,
                "dt": date,
                "side": "BUY",
                "reason": "new_entry" if not decision["is_initial"] else "initial_fill",
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["kdcode", "dt", "side", "reason"])
    return df


def find_scores_file(scores_dir: Path, date: str = None) -> tuple:
    """
    Locate the scores CSV for a given date.
    Checks paper_trade/results/YYYY-MM-DD/scores.csv first,
    then falls back to averaged_predictions/YYYY-MM-DD.csv.
    Returns (path, date_str).
    """
    if date:
        candidate = scores_dir / date / "scores.csv"
        if candidate.exists():
            return candidate, date

        avg_dir = (
            PROJECT_ROOT
            / "paper_trade"
            / "Model"
            / "Seed73_trained_to_2062026"
            / "averaged_predictions"
        )
        candidate = avg_dir / f"{date}.csv"
        if candidate.exists():
            return candidate, date

        raise FileNotFoundError(
            f"No scores file found for {date}. "
            f"Checked {scores_dir / date / 'scores.csv'} and {candidate}"
        )

    dated_dirs = sorted(
        [d for d in scores_dir.iterdir() if d.is_dir() and (d / "scores.csv").exists()],
        key=lambda d: d.name,
    )
    if dated_dirs:
        latest = dated_dirs[-1]
        return latest / "scores.csv", latest.name

    avg_dir = (
        PROJECT_ROOT
        / "paper_trade"
        / "Model"
        / "Seed73_trained_to_2062026"
        / "averaged_predictions"
    )
    if avg_dir.exists():
        avg_files = sorted(avg_dir.glob("*.csv"))
        if avg_files:
            latest = avg_files[-1]
            return latest, latest.stem

    raise FileNotFoundError(f"No scores files found in {scores_dir} or averaged_predictions/")


def main():
    parser = argparse.ArgumentParser(description="MCI-GRU portfolio decision engine.")
    parser.add_argument(
        "--scores-dir",
        default=DEFAULT_SCORES_DIR,
        help="Directory containing date-stamped scores subdirectories",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help="Directory for persistent state files",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to process (default: latest available)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of stocks to hold",
    )
    parser.add_argument(
        "--min-rank-drop",
        type=int,
        default=DEFAULT_MIN_RANK_DROP,
        help="Minimum rank deterioration to trigger exit",
    )
    args = parser.parse_args()

    scores_dir = PROJECT_ROOT / args.scores_dir
    state_dir = PROJECT_ROOT / args.state_dir

    print("=" * 70)
    print("  MCI-GRU Portfolio Decision Engine")
    print("=" * 70)

    scores_path, pred_date = find_scores_file(scores_dir, args.date)
    print(f"  Scores file: {scores_path}")
    print(f"  Date:        {pred_date}")
    print(f"  Top-K:       {args.top_k}")
    print(f"  Rank gate:   {args.min_rank_drop}")

    scores_df = load_scores(scores_path)
    print(f"  Scored stocks: {len(scores_df)}")

    prev_holdings, prev_ranks, prev_date = load_state(state_dir)
    if prev_holdings is not None:
        print(f"  Prior holdings: {len(prev_holdings)} stocks (from {prev_date})")
    else:
        print("  Prior holdings: none (initial run)")

    decision = apply_rank_drop_gate(
        scores_df=scores_df,
        prev_holdings=prev_holdings,
        prev_ranks=prev_ranks,
        top_k=args.top_k,
        min_rank_drop=args.min_rank_drop,
    )

    target_portfolio = build_target_portfolio(
        target_stocks=decision["target_stocks"],
        scores_df=scores_df,
        date=pred_date,
        prev_holdings=prev_holdings,
    )

    orders = build_orders(decision, pred_date)

    output_dir = scores_dir / pred_date
    output_dir.mkdir(parents=True, exist_ok=True)

    target_portfolio.to_csv(str(output_dir / "target_portfolio.csv"), index=False)
    orders.to_csv(str(output_dir / "orders.csv"), index=False)

    current_holdings = []
    for _, row in target_portfolio.iterrows():
        current_holdings.append(
            {
                "kdcode": row["kdcode"],
                "entry_date": row["entry_date"],
                "entry_rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
                "weight": row["weight"],
            }
        )

    current_ranks = scores_df.set_index("kdcode")["rank"].to_dict()
    current_ranks = {k: int(v) for k, v in current_ranks.items()}
    save_state(state_dir, pred_date, current_holdings, current_ranks)

    print(f"\n  DECISION SUMMARY ({pred_date})")
    print(f"  {'-' * 50}")
    if decision["is_initial"]:
        print(f"  Initial portfolio: {len(decision['target_stocks'])} stocks")
    else:
        print(f"  Survivors:    {len(decision['survivors'])}")
        print(f"  Exits:        {len(decision['exits'])}")
        print(f"  New entries:  {len(decision['new_entries'])}")

    if decision["exit_details"]:
        print("\n  EXITS:")
        for d in decision["exit_details"]:
            reason = d.get("reason", "rank_drop_gate")
            if reason == "rank_drop_gate":
                print(
                    f"    {d['kdcode']:12s}  rank {d['prev_rank']:>3d} -> {d['curr_rank']:>3d}  (drop {d['rank_drop']:+d})"
                )
            else:
                print(f"    {d['kdcode']:12s}  {reason}")

    if decision["new_entries"]:
        print("\n  NEW ENTRIES:")
        rank_map = scores_df.set_index("kdcode")["rank"].to_dict()
        for kd in decision["new_entries"]:
            print(f"    {kd:12s}  rank {rank_map.get(kd, '?'):>3}")

    print(f"\n  TARGET PORTFOLIO ({len(decision['target_stocks'])} stocks):")
    print(target_portfolio[["kdcode", "rank", "score", "weight"]].to_string(index=False))

    num_orders = len(orders)
    turnover = num_orders / (2 * args.top_k) if args.top_k > 0 else 0.0
    print(f"\n  Orders: {num_orders} ({turnover:.0%} one-way turnover)")
    print(f"  Files saved to: {output_dir}")
    print(f"  State saved to: {state_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
