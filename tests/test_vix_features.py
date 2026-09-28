"""Behavioral tests for the market-wide VIX feature merge."""

from __future__ import annotations

import pandas as pd
import pytest

from mci_gru.features.volatility import VIX_FEATURES, add_vix_features

PANEL_DATES = [f"2020-01-0{day}" for day in range(1, 8)]
# 2020-01-01 precedes the first observation; 2020-01-04 and 2020-01-06 are gaps.
VIX_OBSERVATIONS = {
    "2020-01-02": 11.0,
    "2020-01-03": 13.0,
    "2020-01-05": 17.0,
    "2020-01-07": 23.0,
}


def _panel() -> pd.DataFrame:
    # One stock: the fill runs in the merged panel's row order, so across stocks
    # its result depends on how the caller sorted the panel. That is not pinned here.
    return pd.DataFrame(
        {
            "kdcode": "AAA",
            "dt": PANEL_DATES,
            "close": [100.0 + i for i in range(len(PANEL_DATES))],
        }
    )


def _vix(observations: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"dt": list(observations), "vix": list(observations.values())})


def _by_date(result: pd.DataFrame, column: str) -> pd.Series:
    return result.set_index("dt")[column].astype(float)


def test_missing_vix_dates_carry_the_last_earlier_level_and_default_to_20() -> None:
    result = add_vix_features(_panel(), _vix(VIX_OBSERVATIONS))

    assert result["dt"].tolist() == PANEL_DATES
    assert _by_date(result, "vix").to_dict() == {
        "2020-01-01": 20.0,  # nothing observed yet
        "2020-01-02": 11.0,
        "2020-01-03": 13.0,
        "2020-01-04": 13.0,  # carried from 2020-01-03, never pulled from 2020-01-05
        "2020-01-05": 17.0,
        "2020-01-06": 17.0,  # carried from 2020-01-05, never pulled from 2020-01-07
        "2020-01-07": 23.0,
    }


@pytest.mark.parametrize("cutoff", PANEL_DATES[1:])
def test_future_vix_observations_do_not_change_earlier_rows(cutoff: str) -> None:
    changed = {
        dt: level * 10.0 if dt >= cutoff else level for dt, level in VIX_OBSERVATIONS.items()
    }

    base = add_vix_features(_panel(), _vix(VIX_OBSERVATIONS))
    mutated = add_vix_features(_panel(), _vix(changed))

    # The perturbation must reach the output, or the comparison below is vacuous.
    base_vix, mutated_vix = _by_date(base, "vix"), _by_date(mutated, "vix")
    assert not base_vix[base_vix.index >= cutoff].equals(mutated_vix[mutated_vix.index >= cutoff])
    for column in VIX_FEATURES:
        before_base = _by_date(base, column)
        before_mutated = _by_date(mutated, column)
        pd.testing.assert_series_equal(
            before_base[before_base.index < cutoff],
            before_mutated[before_mutated.index < cutoff],
        )
