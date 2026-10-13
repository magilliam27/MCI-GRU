"""Jev's answers graded against the hand labels (#289)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.data.grade_risk_factor_pairs as grade_script
from mci_gru.filings.grading import (
    GradingError,
    accuracy_gap,
    calibration,
    grade,
    graded_table,
    read_labels,
)


def _inputs(labels, probabilities, ranks, headings_match):
    """Four labelled pairs (or as many as given) in the files' own column layout."""
    n = len(labels)
    ids = [f"L{i:03d}" for i in range(1, n + 1)]
    pair_ids = [f"c{i}|p{i}" for i in range(n)]
    sheet = pd.DataFrame({"label_id": ids, "same_risk": labels})
    design = pd.DataFrame(
        {"label_id": ids, "pair_id": pair_ids, "stratum": ["2020|top"] * n, "weight": [1.0] * n}
    )
    pairs = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "fiscal_year": [2020] * n,
            "rank": ranks,
            "similarity": [1.0 / rank for rank in ranks],
            "current_heading": [f"Risk {i}" for i in range(n)],
            "prior_heading": [
                f"  risk {i} " if same else f"Other {i}" for i, same in enumerate(headings_match)
            ],
        }
    )
    scores = pd.DataFrame({"pair_id": pair_ids, "probability": probabilities})
    return sheet, design, pairs, scores


def test_labels_are_read_in_any_letter_case():
    sheet = pd.DataFrame({"label_id": ["L1", "L2", "L3"], "same_risk": ["Yes", " no", "NO"]})
    assert read_labels(sheet).tolist() == [1, 0, 0]


@pytest.mark.parametrize("bad", ["", "maybe", None])
def test_a_blank_or_unknown_label_stops_grading(bad):
    sheet = pd.DataFrame({"label_id": ["L1", "L2"], "same_risk": ["yes", bad]})
    with pytest.raises(GradingError, match="not yes or no"):
        read_labels(sheet)


def test_a_labelled_pair_without_a_jev_score_stops_grading():
    sheet, design, pairs, scores = _inputs(["yes", "no"], [0.9, 0.1], [1, 2], [True, False])
    with pytest.raises(GradingError, match="no Jev score"):
        graded_table(sheet, design, pairs, scores.iloc[:1])


def test_jev_and_both_baselines_answer_each_pair():
    sheet, design, pairs, scores = _inputs(
        ["yes", "no", "yes", "no"], [0.9, 0.2, 0.5, 0.49], [1, 1, 3, 2], [True, False, False, False]
    )
    table = graded_table(sheet, design, pairs, scores)
    assert table["jev"].tolist() == [1, 0, 1, 0]
    assert table["text_top_match"].tolist() == [1, 1, 0, 0]
    assert table["heading_match"].tolist() == [1, 0, 0, 0]


def test_accuracy_and_the_gap_over_each_baseline():
    sheet, design, pairs, scores = _inputs(
        ["yes", "no", "yes", "no"], [0.9, 0.2, 0.8, 0.1], [1, 1, 3, 2], [True, False, False, False]
    )
    summary = grade(graded_table(sheet, design, pairs, scores), seed=1)
    assert summary["accuracy"]["jev"]["accuracy"] == 1.0
    assert summary["accuracy"]["text_top_match"]["accuracy"] == 0.5
    assert summary["accuracy"]["heading_match"]["accuracy"] == 0.75
    assert summary["jev_minus_baseline"]["text_top_match"]["difference"] == 0.5
    assert summary["labelled_yes"] == 2


def test_weighted_accuracy_uses_the_design_weights():
    sheet, design, pairs, scores = _inputs(["yes", "no"], [0.9, 0.9], [1, 2], [False, False])
    design["weight"] = [3.0, 1.0]
    summary = grade(graded_table(sheet, design, pairs, scores), seed=1)
    assert summary["accuracy"]["jev"]["accuracy"] == 0.5
    assert summary["accuracy"]["jev"]["weighted_accuracy"] == 0.75


def test_a_gap_whose_interval_includes_zero_is_not_a_win():
    labels = ["yes", "no"] * 10
    sheet, design, pairs, scores = _inputs(
        labels, [0.9, 0.1] * 10, [1, 2] * 10, [False, False] * 10
    )
    table = graded_table(sheet, design, pairs, scores)
    tie = accuracy_gap(table, "text_top_match", seed=7)
    assert tie.difference == 0.0 and not tie.jev_wins
    win = accuracy_gap(table, "heading_match", seed=7)
    assert win.difference == 0.5 and win.low > 0 and win.jev_wins


def test_calibration_sets_mean_probability_beside_share_labelled_yes():
    sheet, design, pairs, scores = _inputs(
        ["yes", "no", "yes", "yes"], [0.92, 0.95, 0.05, 0.55], [1, 1, 1, 1], [False] * 4
    )
    table = calibration(graded_table(sheet, design, pairs, scores))
    top = table.iloc[-1]
    assert top["pairs"] == 2
    assert top["mean_probability"] == pytest.approx(0.935)
    assert top["share_labelled_yes"] == 0.5
    assert table["pairs"].sum() == 4


def test_grading_script_writes_summary_calibration_and_disagreements_first(tmp_path):
    sheet, design, pairs, scores = _inputs(
        ["Yes", "no", "yes", "no"], [0.9, 0.7, 0.8, 0.1], [1, 1, 3, 2], [True, False, False, False]
    )
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    sheet.to_csv(tmp_path / "label_sheet.csv", index=False)
    design.to_csv(sealed / "label_design.csv", index=False)
    pairs.to_csv(sealed / "pairs.csv", index=False)
    scores.to_csv(sealed / "scores.csv", index=False)

    assert grade_script.main(["--output-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "grading" / "summary.json").read_text(encoding="utf-8"))
    assert summary["accuracy"]["jev"]["accuracy"] == 0.75
    graded = pd.read_csv(tmp_path / "grading" / "graded_pairs.csv")
    assert graded.iloc[0]["label_id"] == "L002"
    assert bool(graded.iloc[0]["jev_disagrees"])
    assert (tmp_path / "grading" / "calibration.csv").exists()


def test_grading_script_refuses_a_half_labelled_sheet(tmp_path, capsys):
    sheet, design, pairs, scores = _inputs(["yes", ""], [0.9, 0.1], [1, 2], [False, False])
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    sheet.to_csv(tmp_path / "label_sheet.csv", index=False)
    design.to_csv(sealed / "label_design.csv", index=False)
    pairs.to_csv(sealed / "pairs.csv", index=False)
    scores.to_csv(sealed / "scores.csv", index=False)

    assert grade_script.main(["--output-dir", str(tmp_path)]) == 1
    assert "not yes or no" in capsys.readouterr().out
    assert not (tmp_path / "grading").exists()
