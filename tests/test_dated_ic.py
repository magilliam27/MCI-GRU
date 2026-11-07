"""Dated daily IC: each test date keeps its own IC, and the means equal the summary's (#326).

The rerun compares arms day by day, so a day's value must sit on its own date even
when an earlier day could not be scored, and the series must be the one the run's
``evaluation_summary.json`` averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.dated_ic import DAILY_IC_FILENAME, dated_ic_frame, write_dated_ic
from mci_gru.evaluation.metrics import evaluate_predictions

DATES = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]


def _panel(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predictions = rng.normal(size=(len(DATES), 12))
    labels = 0.3 * predictions + rng.normal(size=predictions.shape)
    labels[:, 3] = np.nan  # a masked name on every date
    predictions[1] = 1.0  # constant scores: the summary skips this day
    labels[3, :-1] = np.nan  # one observable label: also skipped
    return predictions, labels


def test_unscored_days_keep_their_date_and_later_days_do_not_shift():
    predictions, labels = _panel()
    frame = dated_ic_frame(predictions, labels, DATES)

    assert frame["dt"].tolist() == DATES
    assert frame.loc[[1, 3], ["ic", "rank_ic"]].isna().all().all()
    finite = frame.drop(index=[1, 3])
    assert finite[["ic", "rank_ic"]].notna().all().all()
    # Day 2's value is day 2's own correlation, not day 4's moved up a slot.
    row = 2
    mask = np.isfinite(labels[row])
    expected = np.corrcoef(predictions[row][mask], labels[row][mask])[0, 1]
    assert frame.loc[row, "ic"] == pytest.approx(expected, abs=1e-12)


def test_means_equal_the_evaluation_summary():
    predictions, labels = _panel(seed=3)
    frame = dated_ic_frame(predictions, labels, DATES)
    metrics = evaluate_predictions(predictions, labels, top_k_values=[5])

    assert frame["ic"].mean() == pytest.approx(metrics["avg_ic"], abs=1e-12)
    assert frame["rank_ic"].mean() == pytest.approx(metrics["avg_rank_ic"], abs=1e-12)


def test_the_csv_round_trips_at_full_precision(tmp_path):
    predictions, labels = _panel(seed=5)
    path = write_dated_ic(predictions, labels, DATES, tmp_path)

    assert path == tmp_path / DAILY_IC_FILENAME
    back = pd.read_csv(path, dtype={"dt": str})
    pd.testing.assert_frame_equal(back, dated_ic_frame(predictions, labels, DATES))


def test_dates_that_do_not_match_the_rows_are_refused():
    predictions, labels = _panel()
    with pytest.raises(ValueError, match="dates"):
        dated_ic_frame(predictions, labels, DATES[:-1])
    with pytest.raises(ValueError, match="shape"):
        dated_ic_frame(predictions, labels[:, :-1], DATES)
