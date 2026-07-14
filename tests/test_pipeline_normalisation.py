"""Regression tests for train-boundary-only normalisation fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.pipeline import PitContext, fit_normalisation


@pytest.mark.parametrize("mode", ["zscore", "rank_gauss"])
def test_fit_normalisation_excludes_pretrain_warmup_rows(mode: str) -> None:
    df = pd.DataFrame(
        {
            "kdcode": ["A", "A", "A", "A"],
            "dt": ["2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06"],
            "feat": [-1_000.0, 1.0, 3.0, 1_000.0],
        }
    )
    pit = PitContext(intervals=None, masked_panel=False, csv_path=None)

    fit, _ = fit_normalisation(
        df,
        ["feat"],
        train_start="2021-01-04",
        train_end="2021-01-05",
        mode=mode,
        pit=pit,
    )

    if mode == "zscore":
        assert fit.means["feat"] == pytest.approx(2.0)
        assert fit.stds["feat"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    else:
        assert fit.rank_gauss_reference is not None
        assert fit.rank_gauss_reference["feat"].tolist() == [1.0, 3.0]
