import pandas as pd
import pytest

from scripts.data.export_sp500_pit_gics_top10_mcap import (
    active_constituents_on_date,
    build_asof_dates,
    build_selection_intervals,
    normalise_membership_intervals,
    validate_snapshot_selection,
)


def test_build_asof_dates_includes_first_business_day_month_ends_and_end() -> None:
    dates = build_asof_dates("2024-01-01", "2024-03-15", "monthly")

    assert dates == ["2024-01-01", "2024-01-31", "2024-02-29", "2024-03-15"]


def test_active_constituents_on_date_uses_validity_intervals() -> None:
    intervals = normalise_membership_intervals(
        pd.DataFrame(
            [
                {"constituent_ric": "AAA.N", "valid_from": "2024-01-01", "valid_to": "2024-01-31"},
                {"constituent_ric": "BBB.N", "valid_from": "2024-02-01", "valid_to": "2024-03-31"},
                {"kdcode": "CCC.N", "valid_from": "2024-01-15", "valid_to": "2024-02-15"},
            ]
        )
    )

    assert active_constituents_on_date(intervals, "2024-01-20") == ["AAA.N", "CCC.N"]
    assert active_constituents_on_date(intervals, "2024-03-01") == ["BBB.N"]


def test_build_selection_intervals_coalesces_adjacent_monthly_snapshots() -> None:
    snapshots = pd.DataFrame(
        [
            {"as_of_date": "2024-01-31", "kdcode": "AAA.N"},
            {"as_of_date": "2024-02-29", "kdcode": "AAA.N"},
            {"as_of_date": "2024-03-29", "kdcode": "AAA.N"},
            {"as_of_date": "2024-01-31", "kdcode": "BBB.N"},
            {"as_of_date": "2024-03-29", "kdcode": "BBB.N"},
        ]
    )

    intervals = build_selection_intervals(snapshots, end="2024-04-30")

    assert intervals.to_dict("records") == [
        {"kdcode": "AAA.N", "valid_from": "2024-01-31", "valid_to": "2024-04-30"},
        {"kdcode": "BBB.N", "valid_from": "2024-01-31", "valid_to": "2024-02-28"},
        {"kdcode": "BBB.N", "valid_from": "2024-03-29", "valid_to": "2024-04-30"},
    ]


def test_validate_snapshot_selection_rejects_missing_sector_breadth() -> None:
    selected = pd.DataFrame(
        [
            {"gics_sector": "Energy", "kdcode": "AAA.N"},
            {"gics_sector": "Energy", "kdcode": "BBB.N"},
            {"gics_sector": "Utilities", "kdcode": "CCC.N"},
        ]
    )

    with pytest.raises(ValueError, match="expected 2 with 2 names each"):
        validate_snapshot_selection(
            selected,
            as_of_date="2024-01-31",
            top_n=2,
            expected_sectors=2,
        )
