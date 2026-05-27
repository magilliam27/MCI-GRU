import pandas as pd


def test_joiner_leaver_export_builds_pipeline_pit_universe_schema() -> None:
    from scripts.data import export_sp500_joiner_leaver_pit as pit_export

    intervals = pd.DataFrame(
        [
            {
                "constituent_ric": "MSFT.O",
                "company_name": "Microsoft Corp",
                "valid_from": "2016-01-01",
                "valid_to": "2026-05-04",
            },
            {
                "constituent_ric": "AAPL.O",
                "company_name": "Apple Inc",
                "valid_from": "2016-01-01",
                "valid_to": "2020-12-18",
            },
        ]
    )

    pipeline_pit = pit_export.build_pipeline_pit_universe(intervals)

    assert list(pipeline_pit.columns) == ["kdcode", "valid_from", "valid_to"]
    assert pipeline_pit.to_dict("records") == [
        {"kdcode": "AAPL.O", "valid_from": "2016-01-01", "valid_to": "2020-12-18"},
        {"kdcode": "MSFT.O", "valid_from": "2016-01-01", "valid_to": "2026-05-04"},
    ]


def test_joiner_leaver_export_merges_duplicate_pipeline_intervals() -> None:
    from scripts.data import export_sp500_joiner_leaver_pit as pit_export

    intervals = pd.DataFrame(
        [
            {
                "constituent_ric": "BRK.B",
                "company_name": "Berkshire Hathaway",
                "valid_from": "2016-01-01",
                "valid_to": "2020-01-01",
            },
            {
                "constituent_ric": "BRK.B",
                "company_name": "Berkshire Hathaway",
                "valid_from": "2016-01-01",
                "valid_to": "2020-01-01",
            },
        ]
    )

    pipeline_pit = pit_export.build_pipeline_pit_universe(intervals)

    assert len(pipeline_pit) == 1
    assert pipeline_pit.iloc[0].to_dict() == {
        "kdcode": "BRK.B",
        "valid_from": "2016-01-01",
        "valid_to": "2020-01-01",
    }


def test_joiner_leaver_export_adds_unsuffixed_aliases_and_coalesces_overlaps() -> None:
    from scripts.data import export_sp500_joiner_leaver_pit as pit_export

    intervals = pd.DataFrame(
        [
            {
                "constituent_ric": "CTRA.N^E26",
                "company_name": "Coterra Energy",
                "valid_from": "2016-01-01",
                "valid_to": "2026-05-06",
            },
            {
                "constituent_ric": "CB.N",
                "company_name": "Chubb",
                "valid_from": "2016-01-01",
                "valid_to": "2026-05-13",
            },
            {
                "constituent_ric": "CB.N^A16",
                "company_name": "Chubb Old",
                "valid_from": "2016-01-01",
                "valid_to": "2016-01-18",
            },
        ]
    )

    pipeline_pit = pit_export.build_pipeline_pit_universe(intervals)

    assert {
        "kdcode": "CTRA.N",
        "valid_from": "2016-01-01",
        "valid_to": "2026-05-06",
    } in pipeline_pit.to_dict("records")
    assert {
        "kdcode": "CTRA.N^E26",
        "valid_from": "2016-01-01",
        "valid_to": "2026-05-06",
    } in pipeline_pit.to_dict("records")
    assert pipeline_pit[pipeline_pit["kdcode"] == "CB.N"].to_dict("records") == [
        {"kdcode": "CB.N", "valid_from": "2016-01-01", "valid_to": "2026-05-13"}
    ]
