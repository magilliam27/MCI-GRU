"""Every candidate model ``scripts/score_candidates.py`` can score (#384).

Add a candidate by importing its function and naming it in ``CANDIDATES``. A name
is what the scoring script's ``--arm`` takes and what the pooled report calls the arm,
so keep it stable once a window has been scored under it.

The #360 baselines are registered too, under the names ``scripts/score_baselines.py``
gives them, so a universe with no MCI-GRU run (where that script cannot run) still
gets reversal and momentum beside the candidates. On a run's own axis they score
exactly what ``score_baselines.py`` scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mci_gru.evaluation.baselines import (
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    REVERSAL_LOOKBACKS,
    momentum_scores,
    reversal_scores,
)
from mci_gru.evaluation.candidates import CandidateContext, CandidateResult
from mci_gru.evaluation.company_candidates import (
    earnings_surprise,
    reversal_momentum,
    reversal_momentum_earnings,
)
from mci_gru.evaluation.lightgbm_candidate import lightgbm_5d, lightgbm_20d
from mci_gru.evaluation.residual_linear import residual_linear, residual_reversal_1w
from mci_gru.evaluation.return_history_net import return_net_cnn_lstm, return_net_fnn

if TYPE_CHECKING:
    from mci_gru.evaluation.candidates import Candidate


def _reversal(lookback: int) -> Candidate:
    def reversal(context: CandidateContext) -> CandidateResult:
        scores = reversal_scores(context.panel, context.kdcodes, context.test_dates, lookback)
        return CandidateResult(
            scores, {"definition": f"minus the return from close D-{lookback} to close D"}
        )

    return reversal


def momentum_12_1(context: CandidateContext) -> CandidateResult:
    """The return from close ``D-252`` to close ``D-21``."""
    scores = momentum_scores(context.panel, context.kdcodes, context.test_dates)
    return CandidateResult(
        scores,
        {"definition": f"return from close D-{MOMENTUM_LOOKBACK} to close D-{MOMENTUM_SKIP}"},
    )


CANDIDATES: dict[str, Candidate] = {
    "residual_linear": residual_linear,
    "residual_reversal_1w": residual_reversal_1w,
    **{name: _reversal(lookback) for name, lookback in REVERSAL_LOOKBACKS.items()},
    "momentum_12_1": momentum_12_1,
    "return_net_fnn": return_net_fnn,
    "return_net_cnn_lstm": return_net_cnn_lstm,
    "lightgbm_5d": lightgbm_5d,
    "lightgbm_20d": lightgbm_20d,
    "reversal_momentum": reversal_momentum,
    "earnings_surprise": earnings_surprise,
    "reversal_momentum_earnings": reversal_momentum_earnings,
}


def get_candidate(name: str) -> Candidate:
    """The registered candidate called ``name``."""
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}; registered: {sorted(CANDIDATES)}")
    return CANDIDATES[name]
