"""Jev's same-risk answers graded against the hand labels (#289).

What this module decides:

- **What counts as truth.** The hand labels on the blind sheet, read as ``yes``
  or ``no`` in any letter case. Any other value, or a blank, stops the grading
  rather than being dropped, so a half-labelled sheet is never graded as whole.
- **What Jev is compared with.** Two code baselines that need no model, both
  computed on the same labelled pairs:

  - *heading match*: the pair is the same risk when the two headings are
    identical after case and spacing are ignored;
  - *text top match*: the pair is the same risk when the prior risk is the
    current risk's most similar one by TF-IDF (shortlist rank 1).

  Jev answers yes when its probability is at least one half.
- **How sure the comparison is.** The accuracy gap between Jev and each
  baseline comes with a paired bootstrap interval over the labelled pairs; a gap
  whose interval includes zero is not a win.
- **Whether Jev's confidence means anything.** Calibration bins its
  probabilities in tenths and sets each bin's mean probability beside the share
  of its pairs labelled yes; the Brier score sums that up in one number, lower
  being better. Ranking quality, threshold-free, is the area under the ROC curve
  for Jev's probability and for the TF-IDF similarity.

Weights: the label sheet is a stratified sample of the TF-IDF top-5 shortlist,
so a plain average over-represents small strata. Weighted accuracy uses each
row's design weight and describes the whole shortlist, not every pair Jev was
asked about.

Not graded here: answer stability. The answer cache never asks the same
question twice, so stability needs a deliberate second pass under its own key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

LABEL_VALUES = {"yes": 1, "no": 0}
JEV_THRESHOLD = 0.5
CALIBRATION_BINS = np.linspace(0.0, 1.0, 11)
BOOTSTRAP_DRAWS = 2000
BASELINES = ("heading_match", "text_top_match")

_SPACE = re.compile(r"\s+")


class GradingError(ValueError):
    """The labels, design, pairs and scores cannot be graded together."""


@dataclass(frozen=True)
class Gap:
    """Jev's accuracy minus a baseline's, with a paired bootstrap interval."""

    baseline: str
    difference: float
    low: float
    high: float

    @property
    def jev_wins(self) -> bool:
        return self.low > 0


def read_labels(sheet: pd.DataFrame) -> pd.Series:
    """The hand labels as 1 (yes) or 0 (no), indexed by label id."""
    values = sheet["same_risk"].fillna("").astype(str).str.strip().str.lower()
    bad = sheet.loc[~values.isin(LABEL_VALUES), "label_id"].tolist()
    if bad:
        raise GradingError(f"{len(bad)} labels are not yes or no: {bad[:5]}")
    return pd.Series(values.map(LABEL_VALUES).to_numpy(), index=sheet["label_id"], name="label")


def _heading_key(text: Any) -> str:
    return _SPACE.sub(" ", str(text)).strip().casefold()


def graded_table(
    sheet: pd.DataFrame, design: pd.DataFrame, pairs: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    """One row per labelled pair: label, weight, Jev probability and both baselines."""
    labels = read_labels(sheet).reset_index()
    table = labels.merge(design[["label_id", "pair_id", "stratum", "weight"]], on="label_id")
    if len(table) != len(labels):
        raise GradingError("some label ids are missing from the sampling design")
    table = table.merge(
        pairs[["pair_id", "fiscal_year", "rank", "similarity", "current_heading", "prior_heading"]],
        on="pair_id",
        how="left",
    )
    if table["rank"].isna().any():
        raise GradingError("some labelled pairs are missing from pairs.csv")
    table = table.merge(scores[["pair_id", "probability"]], on="pair_id", how="left")
    unscored = table.loc[table["probability"].isna(), "label_id"].tolist()
    if unscored:
        raise GradingError(f"{len(unscored)} labelled pairs have no Jev score: {unscored[:5]}")
    table["jev"] = (table["probability"] >= JEV_THRESHOLD).astype(int)
    table["heading_match"] = (
        table["current_heading"].map(_heading_key) == table["prior_heading"].map(_heading_key)
    ).astype(int)
    table["text_top_match"] = (table["rank"] == 1).astype(int)
    return table


def _accuracy(truth: np.ndarray, guess: np.ndarray, weight: np.ndarray | None = None) -> float:
    return float(np.average(truth == guess, weights=weight))


def accuracy_gap(table: pd.DataFrame, baseline: str, *, seed: int) -> Gap:
    """Jev's accuracy minus ``baseline``'s, with a 95% paired bootstrap interval."""
    truth = table["label"].to_numpy()
    jev_right = (table["jev"].to_numpy() == truth).astype(float)
    base_right = (table[baseline].to_numpy() == truth).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(table), size=(BOOTSTRAP_DRAWS, len(table)))
    gaps = jev_right[draws].mean(axis=1) - base_right[draws].mean(axis=1)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return Gap(baseline, float(jev_right.mean() - base_right.mean()), float(low), float(high))


def calibration(table: pd.DataFrame) -> pd.DataFrame:
    """Jev's probabilities in tenths: how many pairs, mean probability, share labelled yes."""
    bins = pd.cut(table["probability"], CALIBRATION_BINS, include_lowest=True)
    grouped = table.groupby(bins, observed=False)
    return pd.DataFrame(
        {
            "pairs": grouped.size(),
            "mean_probability": grouped["probability"].mean(),
            "share_labelled_yes": grouped["label"].mean(),
        }
    ).reset_index(names="probability_bin")


def _auc(truth: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(truth, score)) if len(set(truth)) == 2 else None


def grade(table: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    """The summary: accuracy of Jev and both baselines, the gaps, Brier score and AUCs."""
    truth = table["label"].to_numpy()
    weight = table["weight"].to_numpy()
    accuracy = {
        name: {
            "accuracy": _accuracy(truth, table[name].to_numpy()),
            "weighted_accuracy": _accuracy(truth, table[name].to_numpy(), weight),
        }
        for name in ("jev", *BASELINES)
    }
    gaps = [accuracy_gap(table, baseline, seed=seed) for baseline in BASELINES]
    return {
        "labelled_pairs": len(table),
        "labelled_yes": int(truth.sum()),
        "jev_threshold": JEV_THRESHOLD,
        "accuracy": accuracy,
        "jev_minus_baseline": {
            gap.baseline: {
                "difference": gap.difference,
                "interval_95": [gap.low, gap.high],
                "jev_wins": gap.jev_wins,
            }
            for gap in gaps
        },
        "jev_brier": float(brier_score_loss(truth, table["probability"])),
        "auc": {
            "jev_probability": _auc(truth, table["probability"].to_numpy()),
            "text_similarity": _auc(truth, table["similarity"].to_numpy()),
        },
    }
