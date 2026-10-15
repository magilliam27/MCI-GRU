"""How each risk factor changed from last year, asked as one Choice (#289, milestone 2).

Milestone 1 asked whether two risk factors are the same risk. Jev did not beat
word overlap there, but the disputed pairs showed the more useful question is
how a risk changed, which word overlap cannot answer directly. So each
current-year risk factor is paired with its closest prior-year one by TF-IDF
and Jev picks one of five options: same, broadened, narrowed, reworked, new.

What this module decides:

- **The pairing.** One pair per current risk: its TF-IDF top match. Jev judges
  that pair only, so a risk whose true predecessor is not the closest by text
  can be called ``new`` wrongly; the baseline shares the same pairing.
- **The baseline Jev must beat**, fixed before any milestone-2 label was seen:
  ``new`` when even the closest match is under ``NEW_SIMILARITY``; otherwise by
  the share of distinct words added and removed, each against ``CHANGED_SHARE``
  (both over: ``reworked``; one: ``broadened`` or ``narrowed``; neither:
  ``same``).
- **Option order.** Shuffled per pair from a hash of its id, so order cannot
  steer the answer and a rerun asks the same thing. The order is part of the
  cache key.
- **The label sample.** Stratified by the baseline's answer so rare kinds
  (``new``, ``narrowed``) get labels; design weights undo the oversampling.
- **The grade.** Balanced accuracy (the mean of each true kind's recall,
  design-weighted), so a method cannot score well by calling everything
  ``same``. The stop rule reads Jev's gap over the baseline, with a bootstrap
  interval that resamples within strata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mci_gru.filings.jev import (
    AnswerCache,
    ModelDriftError,
    Question,
    QuestionSetError,
    _allocate,
    cache_key,
    pair_state,
)
from mci_gru.filings.pairing import pair_consecutive_years

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mci_gru.filings.pairing import RiskPair
    from mci_gru.filings.risk_factors import RiskFactor

CHANGE_LABELS = ("same", "broadened", "narrowed", "reworked", "new")
#: Frozen 2026-10-13, before any milestone-2 label existed.
NEW_SIMILARITY = 0.25
CHANGED_SHARE = 0.20
BOOTSTRAP_DRAWS = 2000
LABEL_COLUMNS = (
    "label_id",
    "current_year",
    "current_heading",
    "current_body",
    "prior_year",
    "prior_heading",
    "prior_body",
)

_WORD = re.compile(r"[a-z]+")


class ChangeGradingError(ValueError):
    """Labels, design, pairs and answers cannot be graded together."""


def closest_prior_pairs(risks: Sequence[RiskFactor]) -> list[RiskPair]:
    """Each current-year risk with its closest prior-year risk by TF-IDF."""
    return pair_consecutive_years(risks, k=1)


def word_change(current: str, prior: str) -> tuple[float, float]:
    """Shares of distinct words added (of the current text) and removed (of the prior)."""
    now = set(_WORD.findall(current.lower()))
    before = set(_WORD.findall(prior.lower()))
    added = len(now - before) / len(now) if now else 0.0
    removed = len(before - now) / len(before) if before else 0.0
    return added, removed


def baseline_change(similarity: float, added: float, removed: float) -> str:
    """The word-overlap answer to the change question."""
    if similarity < NEW_SIMILARITY:
        return "new"
    grew, shrank = added >= CHANGED_SHARE, removed >= CHANGED_SHARE
    if grew and shrank:
        return "reworked"
    if grew:
        return "broadened"
    if shrank:
        return "narrowed"
    return "same"


def pair_table(pairs: Sequence[RiskPair]) -> pd.DataFrame:
    """One row per pair: ids, texts, similarity, word change and the baseline's answer."""
    rows = []
    for pair in pairs:
        added, removed = word_change(pair.current.text, pair.prior.text)
        rows.append(
            {
                **pair.to_dict(),
                "prior_fiscal_year": pair.prior.fiscal_year,
                "words_added": added,
                "words_removed": removed,
                "baseline": baseline_change(pair.similarity, added, removed),
            }
        )
    return pd.DataFrame(rows)


def option_order(question: Question, pair_id: str) -> list[str]:
    """The pair's own shuffle of the question's options, the same on every run."""
    names = [name for name, _ in question.options]
    seed = int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:16], 16)
    return [names[i] for i in np.random.default_rng(seed).permutation(len(names))]


def ordered_criteria(question: Question, order: Sequence[str]) -> dict[str, str]:
    """The question's options as Choice criteria (name to description), in ``order``."""
    descriptions = dict(question.options)
    return {name: descriptions[name] for name in order}


@dataclass(frozen=True)
class ScoredChoice:
    pair_id: str
    question_id: str
    choice: str
    confidence: float
    probabilities: dict[str, float]
    option_order: list[str]
    model: str
    request_id: str | None
    input_tokens: int | None
    cached: bool


def score_choices(
    pairs: Sequence[RiskPair],
    question: Question,
    *,
    sdk_question_for: Callable[[dict[str, str]], Any],
    client: Any,
    cache: AnswerCache,
    model: str,
) -> list[ScoredChoice]:
    """Ask the Choice of every pair, options shuffled per pair, from the cache when held."""
    if question.kind != "choice":
        raise QuestionSetError(f"{question.question_id}: only Choice questions are scored here")
    if sorted(name for name, _ in question.options) != sorted(CHANGE_LABELS):
        raise QuestionSetError(
            f"{question.question_id}: options must be exactly {', '.join(CHANGE_LABELS)}"
        )
    scored: list[ScoredChoice] = []
    for pair in pairs:
        order = option_order(question, pair.pair_id)
        spec = {**question.spec(), "option_order": order}
        state = pair_state(pair)
        key = cache_key(state, spec, model)
        cached = key in cache
        if not cached:
            response = client.system_one(
                state=state,
                questions={
                    question.question_id: sdk_question_for(ordered_criteria(question, order))
                },
            )
            if response.model != model:
                raise ModelDriftError(f"asked {model}, answered by {response.model}")
            answer = response.choices[question.question_id]
            cache.put(
                {
                    "key": key,
                    "pair_id": pair.pair_id,
                    "question": spec,
                    "model": response.model,
                    "choice": str(answer.choice),
                    "confidence": float(answer.confidence),
                    "probabilities": {k: float(v) for k, v in answer.probabilities.items()},
                    "request_id": response.request_id,
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "state": state,
                }
            )
        record = cache.get(key)
        scored.append(
            ScoredChoice(
                pair_id=pair.pair_id,
                question_id=question.question_id,
                choice=record["choice"],
                confidence=record["confidence"],
                probabilities=record["probabilities"],
                option_order=order,
                model=record["model"],
                request_id=record["request_id"],
                input_tokens=record["input_tokens"],
                cached=cached,
            )
        )
    return scored


def change_label_sheet(
    table: pd.DataFrame, *, n: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A blind sample stratified by the baseline's answer, and its design.

    The sheet shows the two risk factors with their fiscal years and an empty
    ``change`` column; the stratum, weight and pair id go to the design only.
    """
    sizes = table.groupby("baseline").size().to_dict()
    allocation = _allocate(sizes, n)
    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, int, float]] = []
    for stratum in sorted(sizes):
        rows = np.flatnonzero(table["baseline"].to_numpy() == stratum)
        take = allocation[stratum]
        if take:
            picks = rng.choice(rows, size=take, replace=False)
            chosen.extend((stratum, int(i), len(rows) / take) for i in sorted(picks))
    order = rng.permutation(len(chosen))
    sheet_rows, design_rows = [], []
    for position, index in enumerate(order, start=1):
        stratum, row, weight = chosen[index]
        label_id = f"C{position:03d}"
        record = table.iloc[row]
        sheet_rows.append(
            {
                "label_id": label_id,
                "current_year": record["fiscal_year"],
                "current_heading": record["current_heading"],
                "current_body": record["current_body"],
                "prior_year": record["prior_fiscal_year"],
                "prior_heading": record["prior_heading"],
                "prior_body": record["prior_body"],
            }
        )
        design_rows.append(
            {
                "label_id": label_id,
                "pair_id": record["pair_id"],
                "stratum": stratum,
                "weight": weight,
            }
        )
    sheet = pd.DataFrame(sheet_rows, columns=list(LABEL_COLUMNS)).assign(change="")
    design = pd.DataFrame(design_rows, columns=["label_id", "pair_id", "stratum", "weight"])
    return sheet, design


def read_change_labels(sheet: pd.DataFrame) -> pd.Series:
    """The hand labels, one of ``CHANGE_LABELS`` each, indexed by label id."""
    values = sheet["change"].fillna("").astype(str).str.strip().str.lower()
    bad = sheet.loc[~values.isin(CHANGE_LABELS), "label_id"].tolist()
    if bad:
        raise ChangeGradingError(
            f"{len(bad)} labels are not one of {', '.join(CHANGE_LABELS)}: {bad[:5]}"
        )
    return pd.Series(values.to_numpy(), index=sheet["label_id"], name="label")


def graded_changes(
    sheet: pd.DataFrame, design: pd.DataFrame, pairs: pd.DataFrame, choices: pd.DataFrame
) -> pd.DataFrame:
    """One row per labelled risk: label, weight, stratum, Jev's choice and the baseline's."""
    table = read_change_labels(sheet).reset_index()
    table = table.merge(design[["label_id", "pair_id", "stratum", "weight"]], on="label_id")
    if len(table) != len(sheet):
        raise ChangeGradingError("some label ids are missing from the sampling design")
    table = table.merge(
        pairs[["pair_id", "fiscal_year", "similarity", "baseline", "current_heading"]],
        on="pair_id",
        how="left",
    )
    if table["baseline"].isna().any():
        raise ChangeGradingError("some labelled risks are missing from pairs.csv")
    table = table.merge(choices[["pair_id", "choice"]], on="pair_id", how="left")
    unscored = table.loc[table["choice"].isna(), "label_id"].tolist()
    if unscored:
        raise ChangeGradingError(
            f"{len(unscored)} labelled risks have no Jev answer: {unscored[:5]}"
        )
    return table.rename(columns={"choice": "jev"})


def balanced_accuracy(truth: np.ndarray, guess: np.ndarray, weight: np.ndarray) -> float:
    """The mean, over the true kinds present, of each kind's weighted recall."""
    recalls = [
        np.average(guess[truth == kind] == kind, weights=weight[truth == kind])
        for kind in CHANGE_LABELS
        if (truth == kind).any()
    ]
    return float(np.mean(recalls))


def balanced_gap(table: pd.DataFrame, *, seed: int) -> tuple[float, float, float]:
    """Jev's balanced accuracy minus the baseline's, with a 95% bootstrap interval.

    Each draw resamples within each stratum at its own size, as the sample was drawn.
    """
    truth = table["label"].to_numpy()
    jev = table["jev"].to_numpy()
    base = table["baseline"].to_numpy()
    weight = table["weight"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    strata = list(table.groupby("stratum", sort=True).indices.values())
    gaps = []
    for _ in range(BOOTSTRAP_DRAWS):
        rows = np.concatenate([group[rng.integers(0, len(group), len(group))] for group in strata])
        gaps.append(
            balanced_accuracy(truth[rows], jev[rows], weight[rows])
            - balanced_accuracy(truth[rows], base[rows], weight[rows])
        )
    low, high = np.percentile(gaps, [2.5, 97.5])
    point = balanced_accuracy(truth, jev, weight) - balanced_accuracy(truth, base, weight)
    return float(point), float(low), float(high)


def _per_kind(truth: np.ndarray, guess: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    out = {}
    for kind in CHANGE_LABELS:
        said, really = guess == kind, truth == kind
        out[kind] = {
            "labelled": int(really.sum()),
            "recall": float(np.average(said[really], weights=weight[really]))
            if really.any()
            else None,
            "precision": float(np.average(really[said], weights=weight[said]))
            if said.any()
            else None,
        }
    return out


def grade_changes(table: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    """The summary: balanced and plain accuracy, the gap, and recall/precision per kind."""
    truth = table["label"].to_numpy()
    weight = table["weight"].to_numpy(dtype=float)
    methods = {"jev": table["jev"].to_numpy(), "baseline": table["baseline"].to_numpy()}
    point, low, high = balanced_gap(table, seed=seed)
    return {
        "labelled_risks": len(table),
        "labels": {kind: int((truth == kind).sum()) for kind in CHANGE_LABELS},
        "balanced_accuracy": {
            name: balanced_accuracy(truth, guess, weight) for name, guess in methods.items()
        },
        "weighted_accuracy": {
            name: float(np.average(guess == truth, weights=weight))
            for name, guess in methods.items()
        },
        "jev_minus_baseline": {
            "balanced_accuracy": point,
            "interval_95": [low, high],
            "jev_wins": low > 0,
        },
        "per_kind": {name: _per_kind(truth, guess, weight) for name, guess in methods.items()},
        "confusion": {
            name: pd.crosstab(pd.Series(truth, name="label"), pd.Series(guess, name=name)).to_dict()
            for name, guess in methods.items()
        },
    }
