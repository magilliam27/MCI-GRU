"""Grade Jev's same-risk answers against the hand labels (#289).

Run after ``score_risk_factor_pairs.py --score`` has scored every labelled pair::

    python -m scripts.data.grade_risk_factor_pairs --output-dir /content/risk_pairs

It reads ``label_sheet.csv`` (the labelled sheet) and, from ``sealed/``,
``label_design.csv``, ``pairs.csv`` and ``scores.csv``. It writes under
``grading/``:

- ``summary.json``: accuracy of Jev and the two code baselines (heading match,
  TF-IDF top match), Jev's gap over each with a bootstrap interval, Jev's Brier
  score, and the ROC AUC of Jev's probability and of the TF-IDF similarity.
- ``calibration.csv``: Jev's probabilities in tenths beside the share of those
  pairs labelled yes.
- ``graded_pairs.csv``: every labelled pair with its label, Jev's probability
  and each baseline's answer, disagreements with the label first.

The labels are no longer blind once this runs, which is why it refuses a sheet
with any blank or non yes/no label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mci_gru.filings.grading import BASELINES, GradingError, calibration, grade, graded_table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=289, help="Bootstrap seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    sealed = out / "sealed"
    try:
        table = graded_table(
            pd.read_csv(out / "label_sheet.csv", dtype=str, encoding="utf-8-sig"),
            pd.read_csv(sealed / "label_design.csv"),
            pd.read_csv(sealed / "pairs.csv"),
            pd.read_csv(sealed / "scores.csv"),
        )
    except (GradingError, FileNotFoundError) as error:
        print(f"Cannot grade: {error}")
        return 1
    summary = grade(table, seed=args.seed)

    grading = out / "grading"
    grading.mkdir(parents=True, exist_ok=True)
    (grading / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    calibration(table).to_csv(grading / "calibration.csv", index=False)
    table["jev_disagrees"] = table["jev"] != table["label"]
    columns = [
        "label_id",
        "fiscal_year",
        "label",
        "probability",
        "jev",
        *BASELINES,
        "rank",
        "similarity",
        "jev_disagrees",
        "current_heading",
        "prior_heading",
    ]
    table.sort_values(["jev_disagrees", "label_id"], ascending=[False, True])[columns].to_csv(
        grading / "graded_pairs.csv", index=False
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
