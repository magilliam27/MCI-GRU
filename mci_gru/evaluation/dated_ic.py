"""Each test date's IC and rank IC, kept with its date (#326).

``evaluation_summary.json`` reports the mean of the daily IC series, which
``daily_ic_series`` builds by skipping any day it cannot score. That loses which
day each value belongs to, so two runs scored on the same test dates cannot be
compared day by day. This module scores the same arrays one row at a time with
the same function, so the dated series' means equal the summary's ``avg_ic`` and
``avg_rank_ic``, and a day the summary skipped is kept here with an empty value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mci_gru.evaluation.statistics import daily_ic_series

if TYPE_CHECKING:
    from collections.abc import Sequence

DAILY_IC_FILENAME = "daily_ic.csv"
DAILY_IC_COLUMNS = ("dt", "ic", "rank_ic")


def _one_day(predictions: np.ndarray, labels: np.ndarray, method: str) -> float:
    values = daily_ic_series(predictions, labels, method=method)
    return float(values[0]) if values.size else float("nan")


def dated_ic_frame(
    predictions: np.ndarray,
    labels: np.ndarray,
    dates: Sequence[str],
) -> pd.DataFrame:
    """One row per date: ``dt``, Pearson ``ic`` and Spearman ``rank_ic`` (NaN if unscored)."""
    preds = np.asarray(predictions, dtype=np.float64)
    rets = np.asarray(labels, dtype=np.float64)
    if preds.ndim != 2 or preds.shape != rets.shape:
        raise ValueError(
            f"predictions and labels must be the same 2-D shape: {preds.shape} != {rets.shape}"
        )
    if len(dates) != preds.shape[0]:
        raise ValueError(f"{len(dates)} dates for {preds.shape[0]} prediction rows")
    rows = [
        (str(date), _one_day(p, r, "pearson"), _one_day(p, r, "spearman"))
        for date, p, r in zip(dates, preds, rets, strict=True)
    ]
    return pd.DataFrame(rows, columns=list(DAILY_IC_COLUMNS))


def write_dated_ic(
    predictions: np.ndarray,
    labels: np.ndarray,
    dates: Sequence[str],
    directory: str | Path,
) -> Path:
    """Write ``daily_ic.csv`` into *directory* and return its path."""
    path = Path(directory) / DAILY_IC_FILENAME
    dated_ic_frame(predictions, labels, dates).to_csv(path, index=False, float_format="%.17g")
    return path
