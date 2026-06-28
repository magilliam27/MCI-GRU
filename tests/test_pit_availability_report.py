import json
from pathlib import Path

import pandas as pd

from mci_gru.data.pit_audit import build_pit_availability_report


def test_pit_availability_report_keeps_masked_panel_and_reports_tradability() -> None:
    market = pd.DataFrame(
        {
            "dt": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "kdcode": ["AAA", "BBB", "AAA", "CCC"],
            "open": [10.0, 5.0, 10.5, 20.0],
            "close": [10.2, 5.1, 10.6, 21.0],
            "volume": [1_000_000, 0, 800_000, 500_000],
        }
    )
    pit = pd.DataFrame(
        {
            "kdcode": ["AAA", "BBB", "CCC"],
            "valid_from": ["2024-01-01", "2024-01-01", "2024-01-03"],
            "valid_to": ["2024-12-31", "2024-12-31", "2024-12-31"],
        }
    )

    report = build_pit_availability_report(
        market,
        pit,
        min_price=6.0,
        min_dollar_volume=1_000_000.0,
        stale_after_days=1,
    )

    assert report["schema_version"] == 1
    assert report["pit_union_kdcodes"] == 3
    assert report["policy"]["masked_panel_preserved"] is True
    assert report["policy"]["report_only"] is True
    assert report["policy"]["calendar_scope"] == "market_observed"
    assert report["dates"][0] == {
        "dt": "2024-01-02",
        "active_members": 2,
        "observed_members": 2,
        "missing_members": 0,
        "zero_volume_count": 1,
        "stale_count": 0,
        "tradable_count": 1,
    }
    assert report["dates"][1]["active_members"] == 3
    assert report["dates"][1]["observed_members"] == 1
    assert report["dates"][1]["missing_members"] == 2
    assert report["dates"][1]["stale_count"] == 1


def test_pit_availability_report_explicit_calendar_counts_full_day_outage() -> None:
    market = pd.DataFrame(
        {
            "dt": ["2024-01-02"],
            "kdcode": ["AAA"],
            "open": [10.0],
            "close": [10.0],
            "volume": [100_000],
        }
    )
    pit = pd.DataFrame(
        {
            "kdcode": ["AAA"],
            "valid_from": ["2024-01-01"],
            "valid_to": ["2024-12-31"],
        }
    )

    report = build_pit_availability_report(
        market,
        pit,
        min_price=5.0,
        min_dollar_volume=1.0,
        stale_after_days=1,
        calendar=["2024-01-02", "2024-01-03"],
    )

    assert report["policy"]["calendar_scope"] == "explicit"
    assert report["dates"][1]["dt"] == "2024-01-03"
    assert report["dates"][1]["active_members"] == 1
    assert report["dates"][1]["observed_members"] == 0
    assert report["dates"][1]["missing_members"] == 1
    assert report["dates"][1]["stale_count"] == 1


def test_write_pit_availability_report_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    from scripts.write_pit_availability_report import main

    market_path = tmp_path / "market.csv"
    pit_path = tmp_path / "pit.csv"
    calendar_path = tmp_path / "calendar.csv"
    output_path = tmp_path / "report.json"
    pd.DataFrame(
        {
            "dt": ["2024-01-02"],
            "kdcode": ["AAA"],
            "open": [10.0],
            "close": [10.0],
            "volume": [100_000],
        }
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        {
            "kdcode": ["AAA"],
            "valid_from": ["2024-01-01"],
            "valid_to": ["2024-12-31"],
        }
    ).to_csv(pit_path, index=False)
    pd.DataFrame({"dt": ["2024-01-02"]}).to_csv(calendar_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "write_pit_availability_report.py",
            "--market-data",
            str(market_path),
            "--pit-universe",
            str(pit_path),
            "--output",
            str(output_path),
            "--calendar",
            str(calendar_path),
            "--min-price",
            "5",
            "--min-dollar-volume",
            "1000000",
            "--stale-after-days",
            "1",
        ],
    )

    main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["policy"]["calendar_scope"] == "explicit"
    assert payload["dates"][0]["tradable_count"] == 1
