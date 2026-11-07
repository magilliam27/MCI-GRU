"""Every candidate model ``scripts/score_candidates.py`` can score (#384).

Add a candidate by importing its function and naming it in ``CANDIDATES``. A name
is what the scoring script's ``--arm`` takes and what the pooled report calls the arm,
so keep it stable once a window has been scored under it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mci_gru.evaluation.residual_linear import residual_linear, residual_reversal_1w

if TYPE_CHECKING:
    from mci_gru.evaluation.candidates import Candidate

CANDIDATES: dict[str, Candidate] = {
    "residual_linear": residual_linear,
    "residual_reversal_1w": residual_reversal_1w,
}


def get_candidate(name: str) -> Candidate:
    """The registered candidate called ``name``."""
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}; registered: {sorted(CANDIDATES)}")
    return CANDIDATES[name]
