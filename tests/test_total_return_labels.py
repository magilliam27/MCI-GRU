"""Labels and reported returns can measure total return, from the panel's dividends (#361).

The EODHD package carries cash dividends in a ``dividend`` column from revision r4.
``data.return_basis=total`` makes labels, the in-run summary and the replay read a
total-return index instead of the close: ``close`` times the running product of
``1 + dividend / close``. ``price`` (the default) leaves everything as it was, and
the model's inputs never see the dividend column on either basis.
"""

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
    create_config_from_dict,
)
from mci_gru.data.preprocessing import (
    compute_labels,
    return_basis_panel,
    total_return_closes,
)
from mci_gru.evaluation.experiment_summary import (
    compute_evaluation_summary,
    resolved_evaluation_kwargs,
)
from mci_gru.features import FeatureEngineer
from mci_gru.pipeline import prepare_data

N_DAYS = 80
LABEL_T = 3


def _dates(n: int = N_DAYS) -> list[str]:
    return pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d").tolist()


def _frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["kdcode", "dt", "close", "dividend"])


# -- The total-return index ------------------------------------------------------------


def test_one_session_across_an_ex_date_returns_close_plus_dividend_over_the_previous_close():
    df = _frame(
        [
            ("A", "2024-01-02", 100.0, 0.0),
            ("A", "2024-01-03", 98.0, 2.5),  # ex-dividend: holders keep $2.50
            ("A", "2024-01-04", 99.0, 0.0),
            ("B", "2024-01-02", 50.0, 0.0),
            ("B", "2024-01-03", 51.0, 0.0),
            ("B", "2024-01-04", 52.0, 0.0),
        ]
    )
    total = total_return_closes(df)

    assert total.iloc[1] / total.iloc[0] == pytest.approx((98.0 + 2.5) / 100.0)
    assert total.iloc[2] / total.iloc[1] == pytest.approx(99.0 / 98.0)
    # The second dividend-free stock is its closes; nothing carries across stocks.
    assert total.iloc[3:].tolist() == pytest.approx([50.0, 51.0, 52.0])


def test_the_index_follows_each_stocks_dates_whatever_the_row_order():
    df = _frame(
        [
            ("A", "2024-01-04", 99.0, 0.0),
            ("B", "2024-01-03", 51.0, 1.0),
            ("A", "2024-01-02", 100.0, 0.0),
            ("A", "2024-01-03", 98.0, 2.5),
            ("B", "2024-01-02", 50.0, 0.0),
        ]
    )
    total = total_return_closes(df)
    expected = pd.Series([99.0 * 100.5 / 98.0, 52.0, 100.0, 100.5, 50.0], index=df.index)
    pd.testing.assert_series_equal(total, expected, check_names=False)


def test_a_missing_close_keeps_no_value_and_passes_nothing_on():
    df = _frame(
        [
            ("A", "2024-01-02", 100.0, 0.0),
            ("A", "2024-01-03", np.nan, 0.0),
            ("A", "2024-01-04", 99.0, 1.0),
        ]
    )
    total = total_return_closes(df)
    assert np.isnan(total.iloc[1])
    assert total.iloc[2] / total.iloc[0] == pytest.approx(100.0 / 100.0)


def test_a_dividend_that_cannot_be_reinvested_stops():
    unpriced = _frame([("A", "2024-01-02", 100.0, 0.0), ("A", "2024-01-03", np.nan, 1.0)])
    with pytest.raises(ValueError, match="no close"):
        total_return_closes(unpriced)
    negative = _frame([("A", "2024-01-02", 100.0, -1.0)])
    with pytest.raises(ValueError, match="negative"):
        total_return_closes(negative)


def test_the_price_basis_is_the_panel_itself_and_total_needs_dividends():
    df = _frame([("A", "2024-01-02", 100.0, 0.0), ("A", "2024-01-03", 98.0, 2.5)])
    assert return_basis_panel(df, "price") is df

    total = return_basis_panel(df, "total")
    assert total["close"].tolist() == pytest.approx([100.0, 100.5])
    assert df["close"].tolist() == [100.0, 98.0]  # the panel itself is untouched

    with pytest.raises(ValueError, match="dividend column"):
        return_basis_panel(df.drop(columns="dividend"), "total")
    with pytest.raises(ValueError, match="return basis"):
        return_basis_panel(df, "adjusted")


def test_total_return_labels_count_a_dividend_inside_the_hold():
    """Label for D is close D+1 to close D+label_t; a dividend on D+2 is inside it."""
    dates = _dates(6)
    df = _frame(
        [
            ("A", d, c, v)
            for d, c, v in zip(dates, [10, 10, 10, 9.8, 9.9, 10], [0, 0, 0, 0.2, 0, 0], strict=True)
        ]
    )
    price = compute_labels(df, ["A"], dates[:2], LABEL_T, fill_missing=False)
    total = compute_labels(return_basis_panel(df, "total"), ["A"], dates[:2], LABEL_T, False)

    # Signal date 0: entry close 1 (10), exit close 3 (9.8); the dividend on 3 is inside.
    assert price[0, 0] == pytest.approx(9.8 / 10 - 1)
    assert total[0, 0] == pytest.approx((9.8 + 0.2) / 10 - 1)
    # Signal date 1: entry close 2 (10), exit close 4 (9.9).
    assert total[1, 0] == pytest.approx((9.8 + 0.2) / 10 * 9.9 / 9.8 - 1, rel=1e-6)


# -- Configuration ---------------------------------------------------------------------


def test_return_basis_defaults_to_price_and_rejects_anything_else():
    assert DataConfig().return_basis == "price"
    assert DataConfig(return_basis="total").return_basis == "total"
    with pytest.raises(ValueError, match="return_basis"):
        DataConfig(return_basis="dividends")
    with pytest.raises(ValueError, match="index_level"):
        DataConfig(return_basis="total", experiment_mode="index_level")


def test_the_in_run_summary_records_its_return_basis():
    config = create_config_from_dict({"data": {"return_basis": "total"}})
    assert resolved_evaluation_kwargs(config)["return_basis"] == "total"

    dates = _dates(12)
    kdcodes = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        [{"kdcode": k, "dt": d, "close": 100.0 + rng.normal()} for k in kdcodes for d in dates]
    )
    predictions = rng.normal(size=(4, 3))
    labels = rng.normal(size=(4, 3))
    for basis in ("price", "total"):
        config = create_config_from_dict(
            {"data": {"return_basis": basis}, "model": {"label_t": LABEL_T},
             "evaluation": {"top_k_values": [1]}}
        )  # fmt: skip
        summary = compute_evaluation_summary(
            predictions, labels, config, panel=panel, dates=dates[:4], kdcodes=kdcodes,
            rankable_mask=None,
        )  # fmt: skip
        assert summary["return_basis"] == basis


# -- The pipeline ----------------------------------------------------------------------


def _panel(with_dividends: bool) -> pd.DataFrame:
    """Four stocks; S1 pays 2% of its close every 10 sessions."""
    dates = _dates()
    rows = []
    for j in range(4):
        rng = np.random.default_rng(j)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, N_DAYS)))
        for i, dt in enumerate(dates):
            row = {
                "kdcode": f"S{j}",
                "dt": dt,
                "open": close[i] * 0.999,
                "high": close[i] * 1.01,
                "low": close[i] * 0.99,
                "close": close[i],
                "volume": 1.0e6 + 1.0e4 * i,
            }
            if with_dividends:
                row["dividend"] = 0.02 * close[i] if (j == 1 and i % 10 == 5) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _prepare(tmp_path, panel: pd.DataFrame, name: str, return_basis: str) -> dict:
    dates = _dates()
    pit = pd.DataFrame(
        [
            {"kdcode": k, "valid_from": dates[0], "valid_to": dates[-1]}
            for k in panel["kdcode"].unique()
        ]
    )
    pit_csv, panel_csv = tmp_path / f"{name}_pit.csv", tmp_path / f"{name}.csv"
    pit.to_csv(pit_csv, index=False)
    panel.to_csv(panel_csv, index=False)
    config = ExperimentConfig(
        data=DataConfig(
            source="csv",
            filename=str(panel_csv),
            train_start=dates[0],
            train_end=dates[39],
            val_start=dates[45],
            val_end=dates[55],
            test_start=dates[61],
            test_end=dates[-1],
            skip_embargo_check=True,
            use_pit_universe=True,
            pit_universe_csv=str(pit_csv),
            pit_min_scoreable_stocks=0,
            return_basis=return_basis,
        ),
        features=FeatureConfig(include_momentum=False, include_weekly_momentum=False),
        graph=GraphConfig(judge_value=0.9999, use_multi_feature_edges=False),
        model=ModelConfig(his_t=5, label_t=LABEL_T),
        training=TrainingConfig(num_epochs=1, num_models=1, label_type="returns"),
        tracking=TrackingConfig(enabled=False),
    )
    return prepare_data(config, FeatureEngineer(config.features))


def test_a_dividend_column_changes_nothing_on_the_price_basis(tmp_path):
    """r4 on prices reproduces r3: the column reaches neither inputs nor labels."""
    with_column = _prepare(tmp_path, _panel(True), "r4", "price")
    without = _prepare(tmp_path, _panel(False), "r3", "price")

    assert with_column["feature_cols"] == without["feature_cols"]
    assert "dividend" not in with_column["feature_cols"]
    for key in (
        "stock_features_train", "stock_features_test", "x_graph_train",
        "train_labels", "val_labels", "test_labels",
    ):  # fmt: skip
        np.testing.assert_array_equal(with_column[key], without[key], err_msg=key)


def test_the_total_basis_changes_labels_and_returns_but_not_inputs(tmp_path):
    price = _prepare(tmp_path, _panel(True), "price", "price")
    total = _prepare(tmp_path, _panel(True), "total", "total")

    for key in ("stock_features_train", "stock_features_test", "x_graph_train"):
        np.testing.assert_array_equal(price[key], total[key], err_msg=key)

    # Only S1 pays; its labels rise by the dividends inside each hold, the others match.
    column = price["kdcode_list"].index("S1")
    others = [j for j in range(4) if j != column]
    np.testing.assert_array_equal(
        price["train_labels"][:, others], total["train_labels"][:, others]
    )
    gap = total["train_labels"][:, column] - price["train_labels"][:, column]
    assert np.nanmin(gap) >= 0 and np.nanmax(gap) > 0.015

    # The evaluation reads the same closes the labels did.
    panel = _panel(True)
    expected = return_basis_panel(panel, "total")
    got = total["df"].set_index(["kdcode", "dt"])["close"]
    want = expected.set_index(["kdcode", "dt"])["close"]
    np.testing.assert_allclose(got.loc[want.index].to_numpy(), want.to_numpy(), rtol=1e-12)
    assert (price["df"]["close"].to_numpy() == panel["close"].to_numpy()).all()


def test_the_total_basis_stops_on_a_panel_without_dividends(tmp_path):
    with pytest.raises(ValueError, match="dividend column"):
        _prepare(tmp_path, _panel(False), "r3", "total")
