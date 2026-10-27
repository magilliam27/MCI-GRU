"""Issue 367: the pipeline fits normalisation on the training window, not the panel.

Features are computed on the whole panel so the first training date has its
lookback, and those earlier rows stay in the frame. A four-year window must still
be scaled by its own four years, so both normalisation modes fit on
``train_start`` to ``train_end``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.pipeline import PitContext, fit_normalisation

NO_PIT = PitContext(intervals=None, masked_panel=False, csv_path=None)


def _panel() -> pd.DataFrame:
    # Two stocks; the rows before 2020 are far from the training rows, so any leak
    # into the fit moves the statistics by orders of magnitude.
    dates = ["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"]
    rows = []
    for kdcode, offset in (("A", 0.0), ("B", 1.0)):
        for dt, base in zip(dates, [1000.0, 1000.0, 0.0, 2.0, 50.0], strict=True):
            rows.append({"kdcode": kdcode, "dt": dt, "feat": base + offset})
    return pd.DataFrame(rows)


def test_zscore_fit_ignores_rows_before_train_start():
    fit, _ = fit_normalisation(_panel(), ["feat"], "2020-01-02", "2020-01-03", "zscore", NO_PIT)
    assert fit.means["feat"] == pytest.approx(np.mean([0.0, 2.0, 1.0, 3.0]))
    assert fit.stds["feat"] == pytest.approx(np.std([0.0, 2.0, 1.0, 3.0], ddof=1))


def test_zscore_fit_with_an_earlier_start_takes_those_rows_in():
    """Control: the bound is what keeps them out, not the shape of the panel."""
    fit, _ = fit_normalisation(_panel(), ["feat"], "2019-12-30", "2020-01-03", "zscore", NO_PIT)
    assert fit.means["feat"] > 100.0


def test_rank_gauss_fit_ignores_rows_before_train_start():
    fit, _ = fit_normalisation(_panel(), ["feat"], "2020-01-02", "2020-01-03", "rank_gauss", NO_PIT)
    reference = np.asarray(fit.rank_gauss_reference["feat"])
    assert reference.max() < 100.0
    wide, _ = fit_normalisation(
        _panel(), ["feat"], "2019-12-30", "2020-01-03", "rank_gauss", NO_PIT
    )
    assert np.asarray(wide.rank_gauss_reference["feat"]).max() > 100.0
