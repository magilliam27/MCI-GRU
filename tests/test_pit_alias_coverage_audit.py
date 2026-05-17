import pandas as pd

from scripts import audit_pit_lseg_alias_coverage as audit


def _market_rows(kdcode: str, dates: list[str]) -> list[dict[str, object]]:
    return [
        {
            "kdcode": kdcode,
            "dt": date,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + idx,
            "volume": 1000.0,
        }
        for idx, date in enumerate(dates)
    ]


def test_alias_audit_finds_suffixed_candidates_with_market_rows() -> None:
    pit_universe = pd.DataFrame(
        [
            {"kdcode": "AABA.OQ", "valid_from": "2020-01-02", "valid_to": "2020-01-06"},
            {
                "kdcode": "AABA.OQ^J19",
                "valid_from": "2020-01-02",
                "valid_to": "2020-01-06",
            },
            {"kdcode": "ABMD.OQ", "valid_from": "2020-01-03", "valid_to": "2020-01-06"},
        ]
    )
    market = pd.DataFrame(
        _market_rows(
            "AABA.OQ^J19",
            [
                "2019-12-30",
                "2019-12-31",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
            ],
        )
    )

    result = audit.run_alias_coverage_audit(
        unresolved_originals=["AABA.OQ", "ABMD.OQ"],
        pit_universe=pit_universe,
        market_panel=market,
        validation_start="2020-01-02",
        validation_end="2020-01-06",
        his_t=2,
        label_t=1,
    )

    candidates = result.candidates.set_index(["original", "candidate"])
    aaba = candidates.loc[("AABA.OQ", "AABA.OQ^J19")]
    assert aaba["market_rows"] == 7
    assert aaba["market_date_min"] == "2019-12-30"
    assert aaba["market_date_max"] == "2020-01-07"
    assert aaba["overlaps_validation"] is True
    assert aaba["active_days_in_validation"] == 3
    assert aaba["scoreable_days_in_validation"] == 3

    abmd = candidates.loc[("ABMD.OQ", "")]
    assert abmd["candidate_found"] is False
    assert abmd["has_market_rows"] is False
    assert abmd["active_days_in_validation"] == 0


def test_alias_audit_quantifies_daily_active_and_scoreable_impact() -> None:
    pit_universe = pd.DataFrame(
        [
            {"kdcode": "AABA.OQ", "valid_from": "2020-01-02", "valid_to": "2020-01-06"},
            {
                "kdcode": "AABA.OQ^J19",
                "valid_from": "2020-01-02",
                "valid_to": "2020-01-06",
            },
            {"kdcode": "ABMD.OQ", "valid_from": "2020-01-03", "valid_to": "2020-01-06"},
        ]
    )
    market = pd.DataFrame(
        _market_rows(
            "AABA.OQ^J19",
            [
                "2019-12-30",
                "2019-12-31",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
            ],
        )
    )

    result = audit.run_alias_coverage_audit(
        unresolved_originals=["AABA.OQ", "ABMD.OQ"],
        pit_universe=pit_universe,
        market_panel=market,
        validation_start="2020-01-02",
        validation_end="2020-01-06",
        his_t=2,
        label_t=1,
    )

    daily = result.daily_impact.set_index("date")
    assert daily.loc["2020-01-02", "original_active_count"] == 1
    assert daily.loc["2020-01-02", "covered_by_candidate_count"] == 1
    assert daily.loc["2020-01-02", "scoreable_by_candidate_count"] == 1
    assert daily.loc["2020-01-02", "uncovered_active_count"] == 0

    assert daily.loc["2020-01-03", "original_active_count"] == 2
    assert daily.loc["2020-01-03", "covered_by_candidate_count"] == 1
    assert daily.loc["2020-01-03", "scoreable_by_candidate_count"] == 1
    assert daily.loc["2020-01-03", "uncovered_active_count"] == 1

    assert result.summary["original_active_member_days"] == 5
    assert result.summary["candidate_covered_active_days"] == 3
    assert result.summary["candidate_scoreable_active_days"] == 3
    assert result.summary["uncovered_active_member_days"] == 2


def test_alias_audit_markdown_summarizes_candidate_coverage() -> None:
    result = audit.AliasCoverageAudit(
        candidates=pd.DataFrame(
            [
                {
                    "original": "AABA.OQ",
                    "candidate": "AABA.OQ^J19",
                    "candidate_found": True,
                    "has_market_rows": True,
                    "market_rows": 7,
                    "market_date_min": "2019-12-30",
                    "market_date_max": "2020-01-07",
                    "overlaps_validation": False,
                    "active_days_in_validation": 0,
                    "scoreable_days_in_validation": 0,
                }
            ]
        ),
        daily_impact=pd.DataFrame(
            [
                {
                    "date": "2020-01-02",
                    "original_active_count": 0,
                    "covered_by_candidate_count": 0,
                    "scoreable_by_candidate_count": 0,
                    "uncovered_active_count": 0,
                    "unscoreable_active_count": 0,
                }
            ]
        ),
        summary={
            "unresolved_originals": 1,
            "candidate_rows": 1,
            "candidates_with_market_rows": 1,
            "original_active_member_days": 0,
            "candidate_covered_active_days": 0,
            "candidate_scoreable_active_days": 0,
            "uncovered_active_member_days": 0,
            "unscoreable_active_member_days": 0,
        },
    )

    markdown = audit.render_markdown(result, title="Synthetic alias audit")

    assert "# Synthetic alias audit" in markdown
    assert "AABA.OQ^J19" in markdown
    assert "candidates_with_market_rows" in markdown
