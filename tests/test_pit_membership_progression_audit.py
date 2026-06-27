from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_pit_membership_progression as audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _snapshot_rows(members_by_date: dict[str, list[str]]) -> list[dict[str, str]]:
    rows = []
    for as_of_date, members in members_by_date.items():
        for member in members:
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "constituent_ric": member,
                    "company_name": member,
                }
            )
    return rows


def test_audit_outputs_change_summary_snapshot_progression_and_market_evidence(
    tmp_path: Path,
) -> None:
    changes_csv = _write_csv(
        tmp_path / "changes.csv",
        [
            {
                "change_date": "2022-01-04",
                "constituent_ric": "CCC.N",
                "company_name": "CCC",
                "change": "Joiner",
            },
            {
                "change_date": "2022-01-04",
                "constituent_ric": "BBB.N",
                "company_name": "BBB",
                "change": "Leaver",
            },
            {
                "change_date": "2023-03-10",
                "constituent_ric": "DDD.N",
                "company_name": "DDD",
                "change": "Joiner",
            },
            {
                "change_date": "2023-03-10",
                "constituent_ric": "AAA.N",
                "company_name": "AAA",
                "change": "Leaver",
            },
        ],
    )
    snapshots_csv = _write_csv(
        tmp_path / "snapshots.csv",
        _snapshot_rows(
            {
                "2021-12-31": ["AAA.N", "BBB.N"],
                "2022-01-04": ["AAA.N", "CCC.N"],
                "2023-03-10": ["CCC.N", "DDD.N"],
            }
        ),
    )
    market_csv = _write_csv(
        tmp_path / "market.csv",
        [
            {"kdcode": "CCC.N", "dt": "2022-01-04", "close": 10.0},
            {"kdcode": "CCC.N", "dt": "2022-01-05", "close": 10.5},
            {"kdcode": "DDD.N", "dt": "2023-03-10", "close": 20.0},
        ],
    )

    result = audit.run_audit(
        changes_csv=changes_csv,
        snapshots_csv=snapshots_csv,
        market_csv=market_csv,
        validation_years=(2022, 2023),
        output_csv=tmp_path / "progression.csv",
        output_markdown=tmp_path / "audit.md",
    )

    assert result.change_summary.to_dict("records") == [
        {"year": 2022, "joiners": 1, "leavers": 1, "total_changes": 2},
        {"year": 2023, "joiners": 1, "leavers": 1, "total_changes": 2},
    ]
    assert result.snapshot_progression.to_dict("records") == [
        {
            "as_of_date": "2021-12-31",
            "year": 2021,
            "member_count": 2,
            "joined_count": 0,
            "left_count": 0,
            "transition_count": 0,
            "net_change": 0,
        },
        {
            "as_of_date": "2022-01-04",
            "year": 2022,
            "member_count": 2,
            "joined_count": 1,
            "left_count": 1,
            "transition_count": 2,
            "net_change": 0,
        },
        {
            "as_of_date": "2023-03-10",
            "year": 2023,
            "member_count": 2,
            "joined_count": 1,
            "left_count": 1,
            "transition_count": 2,
            "net_change": 0,
        },
    ]
    joiner_rows = result.representative_transitions[
        result.representative_transitions["direction"] == "Joiner"
    ]
    assert joiner_rows[["year", "constituent_ric", "market_row_count"]].to_dict("records") == [
        {"year": 2022, "constituent_ric": "CCC.N", "market_row_count": 2},
        {"year": 2023, "constituent_ric": "DDD.N", "market_row_count": 1},
    ]
    assert (tmp_path / "progression.csv").read_text().splitlines()[0] == (
        "as_of_date,year,member_count,joined_count,left_count,transition_count,net_change"
    )
    assert "CCC.N" in (tmp_path / "audit.md").read_text()


def test_validation_period_guard_fails_when_snapshot_membership_never_changes() -> None:
    snapshots = pd.DataFrame(
        _snapshot_rows(
            {
                "2022-01-03": ["AAA.N", "BBB.N"],
                "2023-01-03": ["AAA.N", "BBB.N"],
                "2024-01-03": ["AAA.N", "BBB.N"],
                "2025-01-03": ["AAA.N", "BBB.N"],
            }
        )
    )
    progression = audit.build_snapshot_progression(snapshots)

    with pytest.raises(ValueError, match="Snapshot membership never changes"):
        audit.require_validation_period_changes(progression, validation_years=(2022, 2025))
