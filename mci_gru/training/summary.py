"""Selected-checkpoint metric reporting and per-window training summaries.

``TrainingResult`` reports each ensemble member's validation loss, IC and rank IC
at its selected checkpoint, with ``None`` for a metric that was unavailable on
that epoch. This module renders those values for logs, builds the
``training_summary.json`` payload, and owns the "mean of the available values,
with coverage" rule that walk-forward aggregation shares.

A value is *available* when it is a finite real number. ``None``, non-numeric
values, booleans, NaN and infinities are unavailable; the non-finite cases cover
sentinels such as ``-inf`` that earlier versions wrote.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mci_gru.training.trainer import TrainingResult

CHECKPOINT_METRICS: tuple[str, ...] = ("best_val_loss", "best_val_ic", "best_val_rank_ic")
"""``TrainingResult`` fields that describe the selected checkpoint's validation."""

_MEMBER_VALUES_KEY = {
    "best_val_loss": "best_val_losses",
    "best_val_ic": "best_val_ics",
    "best_val_rank_ic": "best_val_rank_ics",
}


def available_metric(value: object) -> float | None:
    """Return *value* as a float when it is available, otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def format_checkpoint_metric(value: object) -> str:
    """Render a checkpoint metric for logs: six decimals, or ``unavailable``."""
    number = available_metric(value)
    return "unavailable" if number is None else f"{number:.6f}"


def mean_of_available(values: Sequence[object]) -> tuple[float | None, dict[str, int]]:
    """Mean of the available values, plus ``{"available": n, "total": len(values)}``.

    Unavailable values are left out rather than counted as zero, so the mean is
    ``None`` when nothing is available, including for an empty sequence.
    """
    available = [number for number in map(available_metric, values) if number is not None]
    mean = float(np.mean(available)) if available else None
    return mean, {"available": len(available), "total": len(values)}


def build_training_summary(
    results: Sequence[TrainingResult],
    *,
    experiment_name: str,
    walkforward_window: int,
) -> dict[str, Any]:
    """Build the ``training_summary.json`` payload for one window's ensemble.

    ``best_val_losses``, ``best_val_ics`` and ``best_val_rank_ics`` keep one slot per
    member, in member order, holding ``None`` where the metric was unavailable at
    that member's selected checkpoint. Each ``mean_best_val_*`` averages only the
    available member values, and ``member_coverage`` records how many members
    contributed to it out of the total.
    """
    member_values = {
        metric: [available_metric(getattr(result, metric)) for result in results]
        for metric in CHECKPOINT_METRICS
    }
    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "models_trained": len(results),
    }
    for metric in CHECKPOINT_METRICS:
        summary[_MEMBER_VALUES_KEY[metric]] = member_values[metric]
    member_coverage: dict[str, dict[str, int]] = {}
    for metric in CHECKPOINT_METRICS:
        summary[f"mean_{metric}"], member_coverage[metric] = mean_of_available(
            member_values[metric]
        )
    summary["member_coverage"] = member_coverage
    summary["walkforward_window"] = walkforward_window
    return summary
