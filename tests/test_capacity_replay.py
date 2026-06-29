import json
from pathlib import Path

import pandas as pd
import pytest

from mci_gru.evaluation.capacity import compute_capacity_replay
from scripts.run_saved_prediction_capacity_replay import main as replay_main


def test_capacity_replay_uses_lagged_dollar_volume_and_t_plus_one_open_timing() -> None:
    predictions = _prediction_frame()
    market = _market_frame()

    report = compute_capacity_replay(
        predictions,
        market,
        aum_values=[1_000_000.0],
        top_k_values=[2],
        adv_lookback_days=1,
        max_adv_participation=0.10,
        spread_bps_values=[10.0],
        slippage_bps_values=[5.0],
    )

    row = report["rows"][0]
    assert row["dt"] == "2024-01-03"
    assert row["entry_dt"] == "2024-01-04"
    assert row["exit_dt"] == "2024-01-05"
    assert row["top_k"] == 2
    assert row["aum"] == 1_000_000.0
    assert row["max_participation"] == pytest.approx(500_000.0 / (20.5 * 100_000))
    assert row["capacity_breach_count"] == 1
    assert row["clipped_count"] == 1
    assert row["turnover"] == pytest.approx(0.5)
    assert row["total_cost"] == pytest.approx(0.001)
    assert row["gross_return"] == pytest.approx(0.025)
    assert row["net_return"] == pytest.approx(0.024)
    assert report["policy"]["uses_lagged_adv"] is True
    assert report["policy"]["timing"] == "score_at_t_close_enter_t_plus_1_open_hold_open_to_open"


def test_capacity_replay_reports_lagged_volatility_gate_breaches() -> None:
    predictions = pd.DataFrame(
        {
            "dt": ["2024-01-05", "2024-01-05"],
            "kdcode": ["AAA", "BBB"],
            "score": [1.0, 0.5],
        }
    )
    market = pd.DataFrame(
        {
            "dt": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
                "2024-01-08",
                "2024-01-08",
                "2024-01-09",
                "2024-01-09",
            ],
            "kdcode": ["AAA", "BBB"] * 7,
            "open": [
                10.0,
                20.0,
                10.1,
                30.0,
                10.2,
                15.0,
                10.3,
                28.0,
                10.4,
                14.0,
                10.5,
                15.0,
                10.6,
                16.0,
            ],
            "close": [
                10.0,
                20.0,
                10.1,
                30.0,
                10.2,
                15.0,
                10.3,
                28.0,
                10.4,
                14.0,
                10.5,
                15.0,
                10.6,
                16.0,
            ],
            "volume": [1_000_000] * 14,
        }
    )

    report = compute_capacity_replay(
        predictions,
        market,
        aum_values=[100_000.0],
        top_k_values=[2],
        adv_lookback_days=3,
        max_adv_participation=0.50,
        max_lagged_volatility_values=[0.10],
    )

    row = report["rows"][0]
    assert report["policy"]["uses_lagged_volatility_gate"] is True
    assert row["max_lagged_volatility_threshold"] == 0.10
    assert row["volatility_breach_count"] == 1
    assert row["gate_breach_count"] >= row["volatility_breach_count"]


def test_capacity_replay_cli_writes_json_and_csv_with_force_guard(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    market_path = tmp_path / "market.csv"
    output_dir = tmp_path / "capacity"
    _prediction_frame().to_csv(predictions_path, index=False)
    _market_frame().to_csv(market_path, index=False)

    args = [
        "--predictions",
        str(predictions_path),
        "--market-data",
        str(market_path),
        "--output-dir",
        str(output_dir),
        "--aum",
        "1000000",
        "--top-k",
        "2",
        "--adv-lookback-days",
        "1",
        "--max-adv-participation",
        "0.10",
        "--spread-bps",
        "10",
        "--slippage-bps",
        "5",
        "--min-rank-drop",
        "30",
        "--max-lagged-volatility",
        "0.5",
    ]
    replay_main(args)

    json_path = output_dir / "capacity_replay.json"
    csv_path = output_dir / "capacity_replay.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["capacity_breach_count"] == 1
    assert payload["rows"][0]["rank_drop_enabled"] is True
    assert payload["rows"][0]["max_lagged_volatility_threshold"] == 0.5
    assert pd.read_csv(csv_path)["capacity_breach_count"].tolist() == [1]
    with pytest.raises(FileExistsError):
        replay_main(args)
    replay_main([*args, "--force"])


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dt": ["2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB"],
            "score": [1.0, 0.5],
        }
    )


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dt": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
            ],
            "kdcode": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "open": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0, 13.2, 17.1],
            "close": [10.5, 20.5, 11.5, 19.5, 12.5, 18.5, 13.0, 17.0],
            "volume": [
                1_000_000,
                100_000,
                9_999_999,
                9_999_999,
                2_000_000,
                150_000,
                1_000_000,
                150_000,
            ],
        }
    )
