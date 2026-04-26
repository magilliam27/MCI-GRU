"""Shared portfolio selection and return utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def rank_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Return scores sorted descending with deterministic one-based ranks."""
    required = {"kdcode", "score"}
    missing = required - set(scores_df.columns)
    if missing:
        raise ValueError(f"Scores DataFrame missing columns: {sorted(missing)}")
    ranked = scores_df.copy()
    ranked = ranked.sort_values(["score", "kdcode"], ascending=[False, True]).reset_index(
        drop=True
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def select_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Select top-k indices by descending score with stable index tie-breaks."""
    if top_k <= 0:
        return np.asarray([], dtype=int)
    arr = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(arr.size), -arr))
    return order[: min(top_k, arr.size)]


def apply_rank_drop_gate(
    scores_df: pd.DataFrame,
    prev_holdings: list | None,
    prev_ranks: dict | None,
    top_k: int,
    min_rank_drop: int,
) -> dict:
    """Apply the paper-trade rank-drop exit gate and refill open slots."""
    ranked = rank_scores(scores_df) if "rank" not in scores_df.columns else scores_df.copy()
    current_ranks = ranked.set_index("kdcode")["rank"].to_dict()

    if prev_holdings is None or prev_ranks is None:
        target_stocks = ranked.head(top_k)["kdcode"].tolist()
        return {
            "target_stocks": target_stocks,
            "survivors": [],
            "exits": [],
            "new_entries": target_stocks,
            "exit_details": [],
            "is_initial": True,
        }

    held_kdcodes = [h["kdcode"] for h in prev_holdings]
    current_held = [kd for kd in held_kdcodes if kd in current_ranks]

    survivors: list[str] = []
    exits: list[str] = []
    exit_details: list[dict] = []
    for kdcode in current_held:
        prev_rank = prev_ranks.get(kdcode)
        curr_rank = current_ranks.get(kdcode)
        if prev_rank is None or curr_rank is None:
            survivors.append(kdcode)
            continue
        rank_drop = int(curr_rank) - int(prev_rank)
        if rank_drop >= min_rank_drop:
            exits.append(kdcode)
            exit_details.append(
                {
                    "kdcode": kdcode,
                    "prev_rank": int(prev_rank),
                    "curr_rank": int(curr_rank),
                    "rank_drop": rank_drop,
                }
            )
        else:
            survivors.append(kdcode)

    dropped_from_universe = [kd for kd in held_kdcodes if kd not in current_ranks]
    for kd in dropped_from_universe:
        exits.append(kd)
        exit_details.append(
            {
                "kdcode": kd,
                "prev_rank": prev_ranks.get(kd),
                "curr_rank": None,
                "rank_drop": None,
                "reason": "dropped_from_universe",
            }
        )

    survivor_set = set(survivors)
    refill_candidates = [kd for kd in ranked["kdcode"].tolist() if kd not in survivor_set]
    slots_needed = max(0, top_k - len(survivors))
    new_entries = refill_candidates[:slots_needed]
    target_stocks = survivors + new_entries
    return {
        "target_stocks": target_stocks,
        "survivors": survivors,
        "exits": exits,
        "new_entries": new_entries,
        "exit_details": exit_details,
        "is_initial": False,
    }


def calculate_turnover(prev_holdings, curr_holdings, target_k: int | None = None) -> float:
    """One-way equal-weight turnover between two holding lists."""
    prev = {h["kdcode"] if isinstance(h, dict) else h for h in (prev_holdings or [])}
    curr = {h["kdcode"] if isinstance(h, dict) else h for h in (curr_holdings or [])}
    if target_k is None:
        target_k = max(len(prev), len(curr))
    if target_k <= 0:
        return 0.0
    sold = prev - curr
    bought = curr - prev
    return float((len(sold) + len(bought)) / (2 * target_k))


def top_k_returns(predictions: np.ndarray, true_returns: np.ndarray, top_k: int) -> np.ndarray:
    """Return equal-weight realized returns of the top-k predictions per day."""
    preds = np.asarray(predictions, dtype=np.float64)
    rets = np.asarray(true_returns, dtype=np.float64)
    if preds.shape != rets.shape:
        raise ValueError(f"predictions and true_returns shapes differ: {preds.shape} != {rets.shape}")
    if preds.ndim != 2:
        raise ValueError("predictions and true_returns must be 2-D")
    out: list[float] = []
    for p, r in zip(preds, rets, strict=True):
        idx = select_top_k(p, top_k)
        if idx.size:
            out.append(float(np.nanmean(r[idx])))
    return np.asarray(out, dtype=np.float64)
