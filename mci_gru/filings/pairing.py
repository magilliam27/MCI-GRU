"""Each risk factor paired with its most similar prior-year risks (#289).

Asking a decision model to search last year's whole list in one question is the
multi-step judgment its own limits page warns about. So code narrows it first:
TF-IDF cosine similarity over heading and body shortlists the ``k`` most similar
prior-year risks, and each pair becomes one yes-or-no question. A risk is new
when no shortlisted pair is judged to be the same risk.

The vocabulary is fitted on the two years being compared, both already public
when the later one is filed, so the shortlist uses no later text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mci_gru.filings.risk_factors import RiskFactor

DEFAULT_SHORTLIST = 5


class PairingError(ValueError):
    """Risk factors cannot be paired as asked."""


@dataclass(frozen=True)
class RiskPair:
    """One current-year risk and one shortlisted prior-year risk."""

    current: RiskFactor
    prior: RiskFactor
    rank: int
    similarity: float

    @property
    def pair_id(self) -> str:
        return f"{self.current.risk_id}|{self.prior.risk_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "fiscal_year": self.current.fiscal_year,
            "current_risk_id": self.current.risk_id,
            "prior_risk_id": self.prior.risk_id,
            "rank": self.rank,
            "similarity": self.similarity,
            "current_heading": self.current.heading,
            "current_body": self.current.body,
            "prior_heading": self.prior.heading,
            "prior_body": self.prior.body,
        }


def shortlist_prior_risks(
    current: Sequence[RiskFactor],
    prior: Sequence[RiskFactor],
    *,
    k: int | None = DEFAULT_SHORTLIST,
) -> list[RiskPair]:
    """The ``k`` most similar prior risks for every current risk, most similar first.

    ``k=None`` pairs every current risk with every prior risk, still ranked.
    """
    if not current or not prior:
        raise PairingError("both years need at least one risk factor")
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform([risk.text for risk in [*current, *prior]])
    scores = cosine_similarity(matrix[: len(current)], matrix[len(current) :])
    pairs: list[RiskPair] = []
    for row, risk in enumerate(current):
        order = np.argsort(-scores[row], kind="stable")[: len(prior) if k is None else k]
        pairs.extend(
            RiskPair(risk, prior[column], rank, float(scores[row, column]))
            for rank, column in enumerate(order, start=1)
        )
    return pairs


def pair_consecutive_years(
    risks: Sequence[RiskFactor], *, k: int | None = DEFAULT_SHORTLIST
) -> list[RiskPair]:
    """Pairs for every year against the year before; the earliest year is reference only."""
    by_year: dict[int, list[RiskFactor]] = {}
    for risk in risks:
        by_year.setdefault(risk.fiscal_year, []).append(risk)
    years = sorted(by_year)
    pairs: list[RiskPair] = []
    for year in years[1:]:
        if year - 1 not in by_year:
            raise PairingError(f"fiscal {year} has no fiscal {year - 1} risk factors to compare")
        pairs.extend(shortlist_prior_risks(by_year[year], by_year[year - 1], k=k))
    return pairs
