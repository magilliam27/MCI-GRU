"""Candidate models scored on a saved run's own axis (#384).

MCI-GRU is not the only model worth scoring on the five protocol windows (#369).
A *candidate* is any other model: a residual-return linear model, gradient-boosted
trees, a small neural net. Each is a function from a :class:`CandidateContext` (what
one window's run knew: its config, its price panel up to the last test date, its stock
axis, its split dates and its own training and validation labels) to a
:class:`CandidateResult` (a score for every test date and stock).

``scripts/score_candidates.py`` builds the context from a saved run, calls the
candidate, and scores the result with the code that wrote the run's own
``evaluation_summary.json``, on the run's labels, dates, names and PIT tradable mask,
so a candidate's IC and MCI-GRU's are the same measurement. ``scripts/pool_candidates.py``
pools the windows.

What a candidate may see, and what it may not:

* ``panel`` holds every row of the run's data file up to and including the last test
  date, and nothing after it: no candidate can read the locked holdout, and none can
  read a test label's closing prices beyond the last test date. A score for date ``D``
  must still use rows up to ``D`` only; ``check_no_lookahead`` tests that.
* Training and validation labels come from ``data`` (the run's own labels, cut at each
  split's end as the pipeline cuts them). The test labels, and every key that depends
  on them (``HIDDEN_DATA_KEYS``), are left out of ``data``.
* ``pit_intervals`` is the run's point-in-time membership file, for a candidate that
  needs to know which names were members on a past date (a market average, say).

Register a candidate in ``mci_gru.evaluation.candidate_registry``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from mci_gru.config import ExperimentConfig

#: ``data`` keys a candidate never gets, because each depends on prices after a test
#: date: the test labels; the test loss mask (tradable and label-available, which needs
#: the closes at D+1 and D+label_t); the PIT eligibility and breadth reports and the
#: admission record that carries them (resolved test labels); and the panel the pipeline
#: built (which runs past the last test date).
HIDDEN_DATA_KEYS = frozenset(
    {"test_labels", "test_loss_mask", "pit_eligibility", "pit_breadth", "admission", "df"}
)


@dataclass(frozen=True)
class CandidateContext:
    """One window's run, as a candidate model may see it."""

    config: ExperimentConfig
    panel: pd.DataFrame  # kdcode, dt (str), open, high, low, close, volume; dt <= test_dates[-1]
    kdcodes: list[str]
    test_dates: list[str]
    rankable_mask: np.ndarray | None  # (test dates, stocks): the run's PIT tradable mask
    pit_intervals: pd.DataFrame | None  # kdcode, valid_from, valid_to (str); None without PIT
    data_loader: Callable[[], dict[str, Any]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.test_dates:
            raise ValueError("a candidate context needs at least one test date")
        last = self.test_dates[-1]
        if (self.panel["dt"].astype(str) > last).any():
            raise ValueError(f"the panel runs past the last test date {last}")

    @functools.cached_property
    def data(self) -> dict[str, Any]:
        """The run's stock-level ``prepare_data`` output, without its test labels.

        Built on first use (a few minutes on CPU) and kept. Keys include
        ``train_dates``, ``val_dates``, ``test_dates``, ``stock_features_train``/``_val``/
        ``_test``, ``train_labels``, ``val_labels``, ``train_loss_mask`` and
        ``val_tradable_mask``.
        """
        built = self.data_loader()
        return {key: value for key, value in built.items() if key not in HIDDEN_DATA_KEYS}

    @property
    def train_dates(self) -> list[str]:
        """The dates ``data['train_labels']`` rows are on."""
        return [str(d) for d in self.data["train_dates"]]

    @property
    def val_dates(self) -> list[str]:
        """The dates ``data['val_labels']`` rows are on."""
        return [str(d) for d in self.data["val_dates"]]

    def with_panel(self, panel: pd.DataFrame) -> CandidateContext:
        """The same context over another panel; used to test for lookahead."""
        clone = CandidateContext(
            config=self.config,
            panel=panel,
            kdcodes=self.kdcodes,
            test_dates=self.test_dates,
            rankable_mask=self.rankable_mask,
            pit_intervals=self.pit_intervals,
            data_loader=self.data_loader,
        )
        if "data" in self.__dict__:
            clone.__dict__["data"] = self.__dict__["data"]
        return clone


@dataclass(frozen=True)
class CandidateResult:
    """A candidate's test scores and what it wants recorded about how it got them."""

    scores: np.ndarray  # (test dates, stocks); NaN where the candidate gives no score
    controls: dict[str, Any] = field(default_factory=dict)


Candidate = Callable[[CandidateContext], CandidateResult]


def panel_up_to(panel: pd.DataFrame, last_date: str) -> pd.DataFrame:
    """Rows dated ``last_date`` or earlier, with ``dt`` and ``kdcode`` as strings."""
    rows = panel.copy()
    rows["dt"] = rows["dt"].astype(str)
    rows["kdcode"] = rows["kdcode"].astype(str)
    return rows.loc[rows["dt"] <= str(last_date)].reset_index(drop=True)


def check_result(result: CandidateResult, context: CandidateContext) -> np.ndarray:
    """The result's scores as float64, shaped (test dates, stocks), or a ValueError."""
    scores = np.asarray(result.scores, dtype=np.float64)
    expected = (len(context.test_dates), len(context.kdcodes))
    if scores.shape != expected:
        raise ValueError(f"candidate scores have shape {scores.shape}, expected {expected}")
    if np.isinf(scores).any():
        raise ValueError("candidate scores hold an infinite value; use NaN for no score")
    return scores


def check_no_lookahead(
    candidate: Candidate,
    context: CandidateContext,
    cut_date: str,
    *,
    seed: int = 0,
    atol: float = 0.0,
) -> None:
    """Scrambling every price and volume after ``cut_date`` leaves earlier scores unchanged.

    Scores on test dates up to ``cut_date`` must be identical (to ``atol``) whether or
    not the rows after ``cut_date`` are replaced by noise. A ValueError names the first
    date that moved.
    """
    base = check_result(candidate(context), context)
    rng = np.random.default_rng(seed)
    scrambled = context.panel.copy()
    later = scrambled["dt"].astype(str) > str(cut_date)
    for column in ("open", "high", "low", "close", "volume"):
        if column in scrambled:
            values = scrambled.loc[later, column].to_numpy(dtype=np.float64)
            scrambled.loc[later, column] = values * rng.uniform(0.5, 1.5, size=values.shape)
    moved = check_result(candidate(context.with_panel(scrambled)), context)
    for row, date in enumerate(context.test_dates):
        if date > str(cut_date):
            break
        if not np.allclose(base[row], moved[row], atol=atol, rtol=0.0, equal_nan=True):
            raise ValueError(f"scores on {date} change with prices after {cut_date}")
