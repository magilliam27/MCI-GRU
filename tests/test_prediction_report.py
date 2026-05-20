import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.prediction_report import (
    align_prediction_comparison,
    compute_oos_r2_zero,
    compute_sign_metrics,
    compute_tsfm_prediction_report,
    write_tsfm_prediction_report,
)


def test_oos_r2_uses_zero_return_benchmark() -> None:
    predictions = np.array([0.10, 0.00, -0.05])
    returns = np.array([0.10, 0.10, -0.10])

    r2 = compute_oos_r2_zero(predictions, returns)

    assert r2 == pytest.approx(1.0 - 0.0125 / 0.03)


def test_sign_metrics_include_positive_negative_and_zero_returns() -> None:
    predictions = np.array([-0.20, 0.00, 0.10, -0.10])
    returns = np.array([-0.10, 0.00, 0.20, 0.30])

    metrics = compute_sign_metrics(predictions, returns)

    assert metrics["direction_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1.0 + 2 / 3) / 3)


def test_alignment_intersects_dates_universe_and_optional_baselines() -> None:
    primary = pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB"],
            "score": [0.4, 0.3, 0.2, 0.1],
        }
    )
    returns = pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB", "CCC"],
            "realized_return": [0.04, -0.02, 0.01, -0.03, 0.99],
        }
    )
    baseline = pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "AAA", "CCC"],
            "score": [0.01, 0.02, 0.03],
        }
    )

    aligned = align_prediction_comparison(
        primary_predictions=primary,
        realized_returns=returns,
        baseline_predictions={"baseline": baseline},
    )

    assert aligned[["dt", "kdcode"]].to_dict("records") == [
        {"dt": "2024-01-02", "kdcode": "AAA"},
        {"dt": "2024-01-03", "kdcode": "AAA"},
    ]
    assert aligned["mci_gru_score"].tolist() == [0.4, 0.2]
    assert aligned["baseline_score"].tolist() == [0.01, 0.02]
    assert aligned["realized_return"].tolist() == [0.04, 0.01]


def test_prediction_report_includes_yearly_decay_for_each_model() -> None:
    aligned = pd.DataFrame(
        {
            "dt": ["2022-01-03", "2022-01-04", "2023-01-03", "2023-01-04"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB"],
            "realized_return": [0.05, -0.02, 0.01, -0.04],
            "mci_gru_score": [0.04, -0.01, -0.001, 0.001],
            "zero_score": [0.0, 0.0, 0.0, 0.0],
        }
    )

    report = compute_tsfm_prediction_report(aligned, model_score_columns=["mci_gru_score"])

    assert report["comparison"]["aligned_observations"] == 4
    assert report["models"]["mci_gru"]["oos_r2_zero"] > 0
    assert report["models"]["mci_gru"]["direction_accuracy"] == pytest.approx(0.5)
    assert [row["year"] for row in report["yearly_decay"]["mci_gru"]] == [2022, 2023]
    assert report["yearly_decay"]["mci_gru"][0]["n_observations"] == 2


def test_saved_prediction_report_runs_without_training(tmp_path) -> None:
    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    pd.DataFrame(
        {
            "kdcode": ["AAA", "BBB"],
            "dt": ["2024-01-02", "2024-01-02"],
            "score": [0.10, -0.10],
        }
    ).to_csv(predictions_dir / "2024-01-02.csv", index=False)
    pd.DataFrame(
        {
            "kdcode": ["AAA", "BBB"],
            "dt": ["2024-01-03", "2024-01-03"],
            "score": [0.08, -0.08],
        }
    ).to_csv(predictions_dir / "2024-01-03.csv", index=False)

    market_rows = []
    for kdcode, closes in {
        "AAA": [100.0, 110.0, 121.0, 133.1, 146.41],
        "BBB": [100.0, 90.0, 81.0, 72.9, 65.61],
    }.items():
        for day, close in enumerate(closes, start=1):
            market_rows.append({"kdcode": kdcode, "dt": f"2024-01-{day:02d}", "close": close})
    market_path = tmp_path / "market.csv"
    pd.DataFrame(market_rows).to_csv(market_path, index=False)

    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame(
        {
            "kdcode": ["AAA", "BBB", "AAA", "BBB"],
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "score": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(baseline_path, index=False)

    output_dir = tmp_path / "report"
    result = write_tsfm_prediction_report(
        predictions_dir=predictions_dir,
        market_data_path=market_path,
        output_dir=output_dir,
        label_t=2,
        baseline_prediction_paths={"baseline": baseline_path},
        top_k_values=[1],
    )

    assert result["report"]["comparison"]["aligned_observations"] == 4
    assert set(result["report"]["models"]) == {"mci_gru", "baseline"}
    assert (output_dir / "tsfm_prediction_report.json").is_file()
    assert (output_dir / "tsfm_prediction_report.md").is_file()
    assert (output_dir / "tsfm_aligned_predictions.csv").is_file()
    assert "OOS R2" in (output_dir / "tsfm_prediction_report.md").read_text(encoding="utf-8")
