"""E7's per-report change signals for the returns test (#310).

Each member 10-K is compared with its pair partner, the company's previous
report by acceptance time, as the research lead's frozen design says:

- **Signal 1** (:func:`section_change`): 1 minus the cosine similarity of the
  two Item 1A sections' word counts, the Lazy Prices measure. Words are the
  risk baselines' (lower-cased runs of letters), every occurrence counted, no
  stop words removed.
- **Signals 2 and 3** (:func:`risk_pair_table`, :func:`risk_shares`): each risk
  is paired with its closest partner risk by TF-IDF, the same top-one pairing
  :func:`mci_gru.filings.change.closest_prior_pairs` makes within a fiscal year,
  here applied report to report so a changed year end keeps both reports. A risk
  is new when that similarity is under ``NEW_SIMILARITY`` and changed when its
  word score reaches ``WORD_SCORE_CUT`` or it is new. The signals are the shares
  of the report's risks that are new and changed.
- **Signal 4** (:func:`worse_share`): Jev's probability that the change made
  the risk worse, summed over the flagged (changed) risks and divided by all the
  report's risks, so an unflagged risk counts as zero.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from mci_gru.filings.change import NEW_SIMILARITY, WORD_SCORE_CUT, pair_table, word_score
from mci_gru.filings.pairing import shortlist_prior_risks

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from mci_gru.filings.pairing import RiskPair
    from mci_gru.filings.risk_factors import RiskFactor

RISK_PAIR_COLUMNS = (
    "accession_number",
    "prior_accession",
    "pair_id",
    "current_risk_id",
    "prior_risk_id",
    "similarity",
    "words_added",
    "words_removed",
    "word_score",
    "new",
    "changed",
)

_WORD = re.compile(r"[a-z]+")


class SignalError(ValueError):
    """A report cannot be given a signal as asked."""


def word_counts(text: str) -> Counter[str]:
    """Every word of a text, lower-cased, with how often it occurs."""
    return Counter(_WORD.findall(text.lower()))


def section_change(current: str, prior: str) -> float:
    """1 minus the cosine similarity of two texts' word counts (0 same, 1 no shared word)."""
    now, before = word_counts(current), word_counts(prior)
    if not now or not before:
        raise SignalError("both sections need at least one word")
    dot = sum(count * before[word] for word, count in now.items())
    norm = math.sqrt(sum(c * c for c in now.values())) * math.sqrt(
        sum(c * c for c in before.values())
    )
    return 1.0 - dot / norm


def report_pairs(current: Sequence[RiskFactor], prior: Sequence[RiskFactor]) -> list[RiskPair]:
    """Each of the report's risks with its closest risk in the partner report, by TF-IDF."""
    return shortlist_prior_risks(current, prior, k=1)


def risk_pair_table(
    current: Sequence[RiskFactor], prior: Sequence[RiskFactor], *, prior_accession: str
) -> pd.DataFrame:
    """One row per current risk: its closest partner risk, word change and the two flags."""
    table = pair_table(report_pairs(current, prior))
    table["accession_number"] = current[0].accession_number
    table["prior_accession"] = prior_accession
    table["word_score"] = word_score(table)
    table["new"] = table["similarity"] < NEW_SIMILARITY
    table["changed"] = table["new"] | (table["word_score"] >= WORD_SCORE_CUT)
    return table[list(RISK_PAIR_COLUMNS)]


def risk_shares(table: pd.DataFrame) -> dict[str, float | int]:
    """A report's risk count, and the counts and shares of its new and changed risks."""
    n = len(table)
    if n == 0:
        raise SignalError("a report needs at least one risk")
    new, changed = int(table["new"].sum()), int(table["changed"].sum())
    return {
        "n_risks": n,
        "n_new": new,
        "n_changed": changed,
        "new_share": new / n,
        "changed_share": changed / n,
    }


def worse_share(p_worse: Sequence[float], n_risks: int) -> float:
    """Jev's "worse" probability summed over the flagged risks, over all the report's risks."""
    if n_risks <= 0:
        raise SignalError("a report needs at least one risk")
    if len(p_worse) > n_risks:
        raise SignalError("more flagged risks than risks")
    return sum(p_worse) / n_risks
