import numpy as np
import pandas as pd
import pytest

from mci_gru.data.preprocessing import compute_labels
from mci_gru.evaluation.prediction_report import (
    align_prediction_comparison,
    compute_oos_r2_zero,
    compute_sign_metrics,
    compute_tsfm_prediction_report,
    realized_returns_from_market_data,
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


def test_alignment_keeps_the_primary_rows_and_left_joins_baselines() -> None:
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

    # The primary's rows, whatever the baseline covers; CCC was never scored by the
    # primary and stays out (#342: the old inner join dropped BBB on both dates).
    assert aligned[["dt", "kdcode"]].to_dict("records") == [
        {"dt": "2024-01-02", "kdcode": "AAA"},
        {"dt": "2024-01-02", "kdcode": "BBB"},
        {"dt": "2024-01-03", "kdcode": "AAA"},
        {"dt": "2024-01-03", "kdcode": "BBB"},
    ]
    assert aligned["mci_gru_score"].tolist() == [0.4, 0.3, 0.2, 0.1]
    assert aligned["baseline_score"].tolist()[0::2] == [0.01, 0.02]
    assert aligned["baseline_score"].isna().tolist() == [False, True, False, True]
    assert aligned["realized_return"].tolist() == [0.04, -0.02, 0.01, -0.03]


def test_adding_a_baseline_does_not_change_the_primary_numbers() -> None:
    """Logic C: a partial baseline's inner join moved the primary's rank IC 0.077 to 0.118."""
    rng = np.random.default_rng(8)
    dates = [str(day.date()) for day in pd.bdate_range("2024-01-01", periods=30)]
    kdcodes = [f"K{i:02d}" for i in range(12)]
    rows = [(d, k) for d in dates for k in kdcodes]
    signal = rng.standard_normal(len(rows))
    returns = pd.DataFrame(
        {
            "dt": [d for d, _ in rows],
            "kdcode": [k for _, k in rows],
            "realized_return": 0.02 * signal + 0.05 * rng.standard_normal(len(rows)),
        }
    )
    primary = returns[["dt", "kdcode"]].assign(score=signal)
    baseline = primary.sample(frac=0.5, random_state=3).assign(
        score=lambda df: rng.standard_normal(len(df))
    )

    alone = compute_tsfm_prediction_report(
        align_prediction_comparison(primary, returns), top_k_values=[3], label_t=5
    )
    with_baseline = compute_tsfm_prediction_report(
        align_prediction_comparison(primary, returns, baseline_predictions={"b": baseline}),
        top_k_values=[3],
        label_t=5,
    )

    assert with_baseline["models"]["mci_gru"] == alone["models"]["mci_gru"]
    assert with_baseline["models"]["b"]["n_observations"] == len(baseline)


def test_realized_returns_read_the_session_axis_not_the_next_row() -> None:
    """Dates finding 2: a missing bar made the row shift book close D+6 as close D+5."""
    dates = [str(day.date()) for day in pd.bdate_range("2025-03-03", periods=12)]
    rows = []
    for j, kdcode in enumerate(["GAP", "KEEP"]):
        for i, dt in enumerate(dates):
            if kdcode == "GAP" and dt == "2025-03-06":
                continue  # GAP prints no bar on one session; KEEP does
            rows.append({"dt": dt, "kdcode": kdcode, "close": 100.0 * (1.01 + 0.02 * j) ** i})
    market = pd.DataFrame(rows)

    realized = realized_returns_from_market_data(market, label_t=5)
    by_key = realized.set_index(["dt", "kdcode"])["realized_return"]
    labels = compute_labels(market, ["GAP", "KEEP"], dates, label_t=5, fill_missing=False)

    # Wherever training observes the label, the report books exactly it (one row per
    # panel bar: GAP has no row for the session it missed).
    for i, dt in enumerate(dates):
        for j, kdcode in enumerate(["GAP", "KEEP"]):
            if np.isfinite(labels[i, j]) and (dt, kdcode) in by_key.index:
                assert by_key[(dt, kdcode)] == pytest.approx(labels[i, j])
    # 2025-03-05's GAP exit (close D+5) exists; its entry (close D+1 = 03-06) does not,
    # so the position never opens and books cash, not the row-shift's 0.0505.
    assert by_key[("2025-03-05", "GAP")] == 0.0
    # The last label_t - 1 sessions have no exit yet.
    assert by_key[(dates[-1], "KEEP")] != by_key[(dates[-1], "KEEP")]


def test_realized_returns_reject_a_label_with_no_hold() -> None:
    market = pd.DataFrame({"dt": ["2024-01-02"], "kdcode": ["AAA"], "close": [100.0]})
    with pytest.raises(ValueError, match="label_t must be >= 2"):
        realized_returns_from_market_data(market, label_t=1)


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
    # The writer hands the report its market panel, so the book is built (#342).
    assert result["report"]["models"]["mci_gru"]["portfolio_basis"] == "DAILY_TRANCHE_BOOK"
    assert (output_dir / "tsfm_prediction_report.json").is_file()
    assert (output_dir / "tsfm_prediction_report.md").is_file()
    assert (output_dir / "tsfm_aligned_predictions.csv").is_file()
    assert "OOS R2" in (output_dir / "tsfm_prediction_report.md").read_text(encoding="utf-8")


def test_report_portfolio_numbers_come_from_the_daily_tranche_book(tmp_path) -> None:
    """The matrix-only Sharpe annualised overlapping 4-session returns as daily (#342)."""
    rng = np.random.default_rng(21)
    dates = [str(day.date()) for day in pd.bdate_range("2024-01-01", periods=60)]
    kdcodes = [f"K{i:02d}" for i in range(10)]
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (len(dates), len(kdcodes))), 0)
    market = pd.DataFrame(
        {
            "dt": np.repeat(dates, 10),
            "kdcode": np.tile(kdcodes, len(dates)),
            "close": closes.ravel(),
        }
    )
    predictions = pd.DataFrame(
        {
            "dt": np.repeat(dates[:50], 10),
            "kdcode": np.tile(kdcodes, 50),
            "score": rng.standard_normal(500),
        }
    )
    aligned = align_prediction_comparison(
        predictions, realized_returns_from_market_data(market, label_t=5)
    )

    with_panel = compute_tsfm_prediction_report(aligned, top_k_values=[3], panel=market)
    without = compute_tsfm_prediction_report(aligned, top_k_values=[3])

    model = with_panel["models"]["mci_gru"]
    assert model["portfolio_basis"] == "DAILY_TRANCHE_BOOK"
    assert np.isfinite(model["top_3_abs_sharpe"]) and np.isfinite(model["top_3_excess_t_stat"])
    assert without["models"]["mci_gru"]["portfolio_basis"] == "NOT_COMPUTED_WITHOUT_PANEL"
    assert not any("sharpe" in key for key in without["models"]["mci_gru"])
    # IC is the same either way.
    assert model["avg_ic"] == pytest.approx(without["models"]["mci_gru"]["avg_ic"])
