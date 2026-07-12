import numpy as np
import pandas as pd

from scripts.data.export_sp500_gics_top10_mcap import (
    _normalise_metadata,
    select_top_by_sector,
)


def test_select_top_by_sector_keeps_top_n_per_gics_sector() -> None:
    metadata = pd.DataFrame(
        [
            {
                "kdcode": "AAA.N",
                "company_name": "AAA",
                "company_market_cap": 300.0,
                "gics_sector": "Energy",
                "gics_sector_field": "TR.GICSSector",
            },
            {
                "kdcode": "BBB.N",
                "company_name": "BBB",
                "company_market_cap": 100.0,
                "gics_sector": "Energy",
                "gics_sector_field": "TR.GICSSector",
            },
            {
                "kdcode": "CCC.N",
                "company_name": "CCC",
                "company_market_cap": 200.0,
                "gics_sector": "Energy",
                "gics_sector_field": "TR.GICSSector",
            },
            {
                "kdcode": "DDD.N",
                "company_name": "DDD",
                "company_market_cap": 500.0,
                "gics_sector": "Utilities",
                "gics_sector_field": "TR.GICSSector",
            },
            {
                "kdcode": "EEE.N",
                "company_name": "EEE",
                "company_market_cap": 400.0,
                "gics_sector": "Utilities",
                "gics_sector_field": "TR.GICSSector",
            },
        ]
    )

    selected = select_top_by_sector(metadata, top_n=2)

    assert selected[["gics_sector", "kdcode", "sector_market_cap_rank"]].to_dict("records") == [
        {"gics_sector": "Energy", "kdcode": "AAA.N", "sector_market_cap_rank": 1},
        {"gics_sector": "Energy", "kdcode": "CCC.N", "sector_market_cap_rank": 2},
        {"gics_sector": "Utilities", "kdcode": "DDD.N", "sector_market_cap_rank": 1},
        {"gics_sector": "Utilities", "kdcode": "EEE.N", "sector_market_cap_rank": 2},
    ]


def test_normalise_metadata_accepts_current_constituent_columns() -> None:
    raw = pd.DataFrame(
        [
            {
                "Instrument": "AAA.N",
                "Company Common Name": "AAA Corp",
                "Company Market Cap": "1,250.5",
                "GICS Sector": "Industrials",
            },
            {
                "Instrument": "BBB.N",
                "Company Common Name": "BBB Corp",
                "Company Market Cap": "",
                "GICS Sector": "Industrials",
            },
        ]
    )

    metadata = _normalise_metadata(raw, "GICS Sector", "TR.GICSSector")

    assert metadata.to_dict("records") == [
        {
            "kdcode": "AAA.N",
            "company_name": "AAA Corp",
            "company_market_cap": 1250.5,
            "gics_sector": "Industrials",
            "gics_sector_field": "TR.GICSSector",
        }
    ]


def test_normalise_metadata_rejects_missing_constituent_ric_values() -> None:
    ric_values = ["AAA.N", None, np.nan, pd.NaT, pd.NA, "", "   ", "NaN", "NONE", "nAt"]
    raw = pd.DataFrame(
        {
            "Instrument": ric_values,
            "Company Common Name": [f"Company {index}" for index in range(len(ric_values))],
            "Company Market Cap": ["1,250.5"] * len(ric_values),
            "GICS Sector": ["Industrials"] * len(ric_values),
        }
    )

    metadata = _normalise_metadata(raw, "GICS Sector", "TR.GICSSector")

    assert metadata["kdcode"].tolist() == ["AAA.N"]
