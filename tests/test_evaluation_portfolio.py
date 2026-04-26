import numpy as np
import pandas as pd

from mci_gru.evaluation.portfolio import (
    apply_rank_drop_gate,
    calculate_turnover,
    rank_scores,
    top_k_returns,
)


def test_rank_drop_gate_matches_paper_trade_initial_and_survivor_behavior():
    scores = rank_scores(
        pd.DataFrame(
            {
                "kdcode": ["A", "B", "C", "D"],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )
    )

    initial = apply_rank_drop_gate(scores, None, None, top_k=2, min_rank_drop=2)
    assert initial["target_stocks"] == ["A", "B"]
    assert initial["new_entries"] == ["A", "B"]

    prev_holdings = [
        {"kdcode": "A", "entry_date": "2026-01-01"},
        {"kdcode": "C", "entry_date": "2026-01-01"},
    ]
    prev_ranks = {"A": 1, "C": 2}
    updated_scores = rank_scores(
        pd.DataFrame(
            {
                "kdcode": ["B", "A", "D", "C"],
                "score": [0.95, 0.90, 0.5, 0.1],
            }
        )
    )

    decision = apply_rank_drop_gate(updated_scores, prev_holdings, prev_ranks, 2, 2)
    assert decision["survivors"] == ["A"]
    assert decision["exits"] == ["C"]
    assert decision["new_entries"] == ["B"]
    assert decision["target_stocks"] == ["A", "B"]


def test_calculate_turnover_handles_empty_full_and_partial_replacements():
    assert calculate_turnover([], ["A", "B"], target_k=2) == 0.5
    assert calculate_turnover(["A", "B"], ["C", "D"], target_k=2) == 1.0
    assert calculate_turnover(["A", "B"], ["A", "C"], target_k=2) == 0.5


def test_top_k_returns_uses_stable_descending_score_selection():
    predictions = np.array([[0.2, 0.5, 0.5, 0.1], [0.9, 0.1, 0.2, 0.3]])
    returns = np.array([[0.01, 0.02, 0.04, -0.01], [0.05, -0.01, 0.01, 0.03]])

    result = top_k_returns(predictions, returns, top_k=2)

    np.testing.assert_allclose(result, np.array([0.03, 0.04]))
