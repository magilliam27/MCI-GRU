import numpy as np
import pandas as pd
import pytest

from mci_gru.config import DataConfig
from mci_gru.data.data_manager import DataManager
from mci_gru.features.regime import (
    REGIME_FEATURES,
    REGIME_OPTIONAL_VARIABLES,
    REGIME_REQUIRED_VARIABLES,
    REGIME_VARIABLES,
    add_regime_features,
    compute_regime_monthly_features,
)


def _make_regime_daily(start: str = "2000-01-01", periods: int = 3800) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="D")
    x = np.linspace(0, 20, len(dates))
    return pd.DataFrame(
        {
            "dt": dates.strftime("%Y-%m-%d"),
            "regime_market": 1000 + 25 * np.sin(x) + np.linspace(0, 100, len(dates)),
            "regime_yield_curve": 1.5 + 0.2 * np.cos(x / 1.7),
            "regime_oil": 60 + 8 * np.sin(x / 1.3),
            "regime_copper": 3.2 + 0.3 * np.cos(x / 2.3),
            "regime_stock_bond_corr": -0.2 + 0.35 * np.sin(x / 2.0),
            "regime_monetary_policy": 2.0 + 0.15 * np.sin(x / 1.8),
            "regime_volatility": 18 + 4 * np.cos(x / 1.5),
        }
    )


def _make_repeating_monthly_regime() -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=10, freq="ME")
    market = np.array([100.0, 101.0, 104.0, 105.0, 108.0, 109.0, 112.0, 113.0, 116.0, 117.0])
    return pd.DataFrame(
        {
            "dt": dates.strftime("%Y-%m-%d"),
            "regime_market": market,
            "regime_yield_curve": np.full(len(dates), 1.5),
            "regime_oil": np.full(len(dates), 60.0),
            "regime_copper": np.full(len(dates), 3.5),
            "regime_stock_bond_corr": np.full(len(dates), -0.1),
            "regime_monetary_policy": np.full(len(dates), 2.0),
            "regime_volatility": np.full(len(dates), 20.0),
        }
    )


def test_compute_regime_monthly_features_outputs_expected_columns():
    regime_df = _make_regime_daily()
    monthly = compute_regime_monthly_features(
        regime_df=regime_df,
        change_months=12,
        norm_window_months=120,
        exclusion_months=1,
        similarity_quantile=0.2,
        min_history_months=24,
    )

    assert set(["dt"] + REGIME_FEATURES).issubset(set(monthly.columns))
    assert len(monthly) > 0
    assert monthly["dt"].is_monotonic_increasing


def test_compute_regime_monthly_features_adds_subsequent_return_signals():
    regime_df = _make_repeating_monthly_regime()
    monthly = compute_regime_monthly_features(
        regime_df=regime_df,
        change_months=1,
        norm_window_months=2,
        exclusion_months=1,
        similarity_quantile=0.4,
        min_history_months=1,
        subsequent_return_horizons=[1, 3],
    )

    october_row = monthly.loc[monthly["dt"] == pd.Timestamp("2000-10-31")].iloc[0]
    expected_similar_return_1m = np.mean([(108.0 / 105.0) - 1.0, (112.0 / 109.0) - 1.0])
    expected_similar_return_3m = np.mean([(112.0 / 105.0) - 1.0, (116.0 / 109.0) - 1.0])
    expected_spread_1m = expected_similar_return_1m - np.mean(
        [(109.0 / 108.0) - 1.0, (113.0 / 112.0) - 1.0]
    )

    assert october_row["regime_similar_subsequent_return_1m"] == pytest.approx(
        expected_similar_return_1m
    )
    assert october_row["regime_similar_subsequent_return_3m"] == pytest.approx(
        expected_similar_return_3m
    )
    assert october_row["regime_subsequent_return_spread_1m"] == pytest.approx(expected_spread_1m)


def test_compute_regime_monthly_features_no_lookahead_exclusion_effect():
    regime_df = _make_regime_daily()

    # Same data, different exclusion window. Values after warmup should differ
    # because exclusion_months changes the allowed historical pool.
    monthly_ex0 = compute_regime_monthly_features(
        regime_df=regime_df,
        exclusion_months=0,
        min_history_months=12,
        include_subsequent_returns=False,
    )
    monthly_ex2 = compute_regime_monthly_features(
        regime_df=regime_df,
        exclusion_months=2,
        min_history_months=12,
        include_subsequent_returns=False,
    )

    joined = monthly_ex0.merge(monthly_ex2, on="dt", suffixes=("_ex0", "_ex2"))
    valid = joined.dropna(subset=["regime_global_score_ex0", "regime_global_score_ex2"])
    assert len(valid) > 10
    # If look-ahead/exclusion wasn't respected, these would be effectively identical.
    assert not np.allclose(
        valid["regime_global_score_ex0"].to_numpy(),
        valid["regime_global_score_ex2"].to_numpy(),
    )


def test_regime_subsequent_return_horizon_excludes_future_window():
    base = _make_regime_daily(start="2000-01-01", periods=2400)
    changed = base.copy()
    changed.loc[changed["dt"] == "2005-12-31", "regime_market"] *= 10.0

    kwargs = {
        "change_months": 1,
        "norm_window_months": 12,
        "exclusion_months": 1,
        "similarity_quantile": 0.2,
        "min_history_months": 12,
        "subsequent_return_horizons": [3],
    }
    out_base = compute_regime_monthly_features(base, **kwargs)
    out_changed = compute_regime_monthly_features(changed, **kwargs)

    past_mask = out_base["dt"] < pd.Timestamp("2005-12-31")
    col = "regime_similar_subsequent_return_3m"
    pd.testing.assert_series_equal(
        out_base.loc[past_mask, col].reset_index(drop=True),
        out_changed.loc[past_mask, col].reset_index(drop=True),
        check_names=False,
    )


def test_add_regime_features_broadcasts_without_row_change():
    stock_dates = pd.date_range("2018-01-01", periods=500, freq="B")
    stock_df = pd.DataFrame(
        {
            "kdcode": ["AAA"] * len(stock_dates) + ["BBB"] * len(stock_dates),
            "dt": list(stock_dates.strftime("%Y-%m-%d")) * 2,
            "close": np.concatenate(
                [
                    100 + np.linspace(0, 5, len(stock_dates)),
                    200 + np.linspace(0, 8, len(stock_dates)),
                ]
            ),
            "open": np.concatenate(
                [
                    99 + np.linspace(0, 5, len(stock_dates)),
                    199 + np.linspace(0, 8, len(stock_dates)),
                ]
            ),
            "high": np.concatenate(
                [
                    101 + np.linspace(0, 5, len(stock_dates)),
                    201 + np.linspace(0, 8, len(stock_dates)),
                ]
            ),
            "low": np.concatenate(
                [
                    98 + np.linspace(0, 5, len(stock_dates)),
                    198 + np.linspace(0, 8, len(stock_dates)),
                ]
            ),
            "volume": 1_000_000,
        }
    )

    regime_df = _make_regime_daily(start="2005-01-01", periods=6000)
    out = add_regime_features(stock_df, regime_df)

    assert len(out) == len(stock_df)
    for col in REGIME_FEATURES:
        assert col in out.columns
    # Broadcast check: same date should have one shared regime value across stocks.
    sample = out[out["dt"] == out["dt"].iloc[-1]]
    assert sample["regime_global_score"].nunique() == 1


def test_regime_input_contract_columns_present():
    regime_df = _make_regime_daily()
    assert set(["dt"] + REGIME_VARIABLES).issubset(set(regime_df.columns))


def test_regime_csv_contract_is_deprecated_and_requires_full_seven_variable_surface():
    """Deprecated CSV override must not silently drop paper-guided variables."""
    import os
    import tempfile

    temp_paths = []

    def write_temp_csv(df: pd.DataFrame) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_paths.append(f.name)
            return f.name

    config = DataConfig(
        train_start="2000-01-01",
        train_end="2000-01-10",
        val_start="2000-01-11",
        val_end="2000-01-12",
        test_start="2000-01-13",
        test_end="2000-01-15",
    )

    try:
        five_variable_csv = write_temp_csv(
            _make_regime_daily(periods=10)[["dt"] + REGIME_REQUIRED_VARIABLES]
        )
        dm = DataManager(config)
        with (
            pytest.warns(DeprecationWarning, match="regime_inputs_csv is deprecated"),
            pytest.raises(ValueError, match="regime_monetary_policy|regime_volatility"),
        ):
            dm.load_regime_inputs(
                regime_inputs_csv=five_variable_csv,
                regime_enforce_lag_days=0,
            )

        full_csv = write_temp_csv(_make_regime_daily(periods=10)[["dt"] + REGIME_VARIABLES])
        dm = DataManager(config)
        with pytest.warns(DeprecationWarning, match="regime_inputs_csv is deprecated"):
            out = dm.load_regime_inputs(regime_inputs_csv=full_csv, regime_enforce_lag_days=0)
            assert set(["dt"] + REGIME_VARIABLES).issubset(set(out.columns))
            assert len(out) >= 1
            for col in REGIME_OPTIONAL_VARIABLES:
                assert not out[col].isna().all()

        missing_required_csv = write_temp_csv(
            _make_regime_daily(periods=10)[["dt"] + REGIME_VARIABLES].drop(
                columns=["regime_copper"]
            )
        )
        dm = DataManager(config)
        with (
            pytest.warns(DeprecationWarning, match="regime_inputs_csv is deprecated"),
            pytest.raises(ValueError, match="regime_copper"),
        ):
            dm.load_regime_inputs(
                regime_inputs_csv=missing_required_csv,
                regime_enforce_lag_days=0,
            )
    finally:
        for path in temp_paths:
            os.unlink(path)


def test_live_regime_stock_bond_corr_uses_three_year_window(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    market_returns = pd.Series(np.linspace(-0.01, 0.012, len(dates)), index=dates)
    market = 100.0 * (1.0 + market_returns).cumprod()
    yield_10y = pd.Series(np.sin(np.linspace(0, 20, len(dates))), index=dates).cumsum()

    series_by_name = {
        "yield_10y": yield_10y,
        "yield_3m": pd.Series(1.0, index=dates),
        "regime_oil": pd.Series(np.linspace(70.0, 80.0, len(dates)), index=dates),
        "regime_volatility": pd.Series(np.linspace(20.0, 25.0, len(dates)), index=dates),
        "regime_market": market,
        "regime_copper": pd.Series(np.linspace(3.0, 4.0, len(dates)), index=dates),
    }

    class FakeFREDLoader:
        def get_series(self, _series_id, _start, _end, value_name, lag_days=1):
            return pd.DataFrame(
                {
                    "dt": dates.strftime("%Y-%m-%d"),
                    value_name: series_by_name[value_name].to_numpy(),
                }
            )

    monkeypatch.setattr("mci_gru.data.fred_loader.FREDLoader", FakeFREDLoader)
    config = DataConfig(
        train_start="2020-01-01",
        train_end="2020-01-10",
        val_start="2020-01-11",
        val_end="2020-01-12",
        test_start="2020-01-13",
        test_end="2022-03-10",
    )

    out = DataManager(config).load_regime_inputs()

    corr = out["regime_stock_bond_corr"]
    assert corr.iloc[:756].isna().all()
    assert not pd.isna(corr.iloc[756])

    expected = market.pct_change().rolling(756, min_periods=756).corr(yield_10y.diff())
    assert corr.iloc[756] == pytest.approx(expected.iloc[756])


def test_live_regime_stock_bond_corr_handles_sparse_merged_panel(monkeypatch):
    dates = pd.date_range("2000-01-01", periods=1600, freq="D")
    even_dates = dates[::2]
    odd_dates = dates[1::2]

    class FakeFREDLoader:
        def get_series(self, _series_id, _start, _end, value_name, lag_days=1):
            if value_name == "yield_10y":
                return pd.DataFrame(
                    {
                        "dt": even_dates.strftime("%Y-%m-%d"),
                        value_name: np.linspace(1.0, 3.0, len(even_dates)),
                    }
                )
            if value_name == "yield_3m":
                return pd.DataFrame(
                    {
                        "dt": even_dates.strftime("%Y-%m-%d"),
                        value_name: np.linspace(0.5, 1.5, len(even_dates)),
                    }
                )
            if value_name == "regime_market":
                returns = np.linspace(-0.005, 0.006, len(odd_dates))
                market = 3000.0 * (1.0 + returns).cumprod()
                return pd.DataFrame(
                    {
                        "dt": odd_dates.strftime("%Y-%m-%d"),
                        value_name: market,
                    }
                )
            return pd.DataFrame(
                {
                    "dt": dates.strftime("%Y-%m-%d"),
                    value_name: np.linspace(10.0, 20.0, len(dates)),
                }
            )

    monkeypatch.setattr("mci_gru.data.fred_loader.FREDLoader", FakeFREDLoader)
    config = DataConfig(
        train_start="2000-01-01",
        train_end="2000-01-10",
        val_start="2000-01-11",
        val_end="2000-01-12",
        test_start="2000-01-13",
        test_end="2004-05-18",
    )

    out = DataManager(config).load_regime_inputs()

    assert not out["regime_stock_bond_corr"].isna().all()


def test_live_regime_fetch_retries_transient_series_failure(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    series_by_name = {
        "yield_10y": pd.Series(np.linspace(1.0, 2.0, len(dates)), index=dates),
        "yield_3m": pd.Series(np.linspace(0.5, 1.0, len(dates)), index=dates),
        "regime_oil": pd.Series(np.linspace(70.0, 80.0, len(dates)), index=dates),
        "regime_volatility": pd.Series(np.linspace(20.0, 25.0, len(dates)), index=dates),
        "regime_market": pd.Series(np.linspace(3000.0, 4000.0, len(dates)), index=dates),
        "regime_copper": pd.Series(np.linspace(3.0, 4.0, len(dates)), index=dates),
    }
    calls_by_name: dict[str, int] = {}

    class FakeFREDLoader:
        def get_series(self, _series_id, _start, _end, value_name, lag_days=1):
            calls_by_name[value_name] = calls_by_name.get(value_name, 0) + 1
            if value_name == "regime_oil" and calls_by_name[value_name] == 1:
                raise RuntimeError("temporary FRED outage")
            return pd.DataFrame(
                {
                    "dt": dates.strftime("%Y-%m-%d"),
                    value_name: series_by_name[value_name].to_numpy(),
                }
            )

    monkeypatch.setattr("mci_gru.data.fred_loader.FREDLoader", FakeFREDLoader)
    monkeypatch.setenv("MCI_GRU_FRED_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("MCI_GRU_FRED_RETRY_SECONDS", "0")
    config = DataConfig(
        train_start="2020-01-01",
        train_end="2020-01-10",
        val_start="2020-01-11",
        val_end="2020-01-12",
        test_start="2020-01-13",
        test_end="2022-03-10",
    )

    out = DataManager(config).load_regime_inputs()

    assert calls_by_name["regime_oil"] == 2
    assert not out["regime_oil"].isna().all()


def test_transform_with_regime_df_produces_nonzero_regime_columns():
    """Regime columns must be non-constant when a real regime_df is passed to transform."""
    from mci_gru.config import FeatureConfig
    from mci_gru.features import FeatureEngineer

    # Build a minimal stock DataFrame: 2 stocks × 3000 daily rows so there is
    # enough history for regime monthly aggregation (regime_min_history_months=24).
    n_days = 3000
    dates = pd.date_range("2000-01-01", periods=n_days, freq="B")
    date_strs = dates.strftime("%Y-%m-%d")
    stocks = ["AAA", "BBB"]
    rows = []
    rng = np.random.default_rng(0)
    for stock in stocks:
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        for _i, (dt, price) in enumerate(zip(date_strs, prices, strict=False)):
            rows.append(
                {
                    "kdcode": stock,
                    "dt": dt,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000_000.0,
                    "turnover": price * 1_000_000.0,
                }
            )
    stock_df = pd.DataFrame(rows)

    regime_df = _make_regime_daily(start="2000-01-01", periods=n_days + 100)

    fe = FeatureEngineer(
        FeatureConfig(
            include_momentum=True,
            include_global_regime=True,
            regime_min_history_months=24,
            regime_include_subsequent_returns=False,
        )
    )
    result = fe.transform(stock_df.copy(), None, None, regime_df)

    regime_col = "regime_global_score"
    assert regime_col in result.columns, f"{regime_col} missing from transform output"
    scores = result[regime_col].dropna()
    assert len(scores) > 0, "All regime_global_score values are NaN"
    non_zero_count = (scores != 0.0).sum()
    assert non_zero_count > 0, (
        "regime_global_score is all zeros — regime_df was not used in transform"
    )
    assert scores.std() > 0, (
        "regime_global_score is constant — regime features appear to be zero-filled"
    )


def test_regime_csv_lag_safety():
    """Lagged CSV regime inputs must not backfill the leading unavailable row."""
    import os
    import tempfile

    values = [10.0, 20.0, 30.0]
    df = pd.DataFrame({"dt": pd.date_range("2000-01-01", periods=3, freq="D")})
    for offset, col in enumerate(REGIME_VARIABLES):
        df[col] = [value + offset for value in values]
    df.loc[1, "regime_oil"] = np.nan

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        config = DataConfig(
            train_start="2000-01-01",
            train_end="2000-01-03",
            val_start="2000-01-04",
            val_end="2000-01-05",
            test_start="2000-01-06",
            test_end="2000-01-07",
        )
        dm = DataManager(config)
        with pytest.warns(DeprecationWarning, match="regime_inputs_csv is deprecated"):
            with_lag = dm.load_regime_inputs(regime_inputs_csv=path, regime_enforce_lag_days=1)

        assert len(with_lag) == len(df)
        assert with_lag.loc[0, REGIME_VARIABLES].isna().all()
        for col in REGIME_VARIABLES:
            assert with_lag.loc[1, col] == df.loc[0, col]
        assert with_lag.loc[2, "regime_market"] == df.loc[1, "regime_market"]
        assert with_lag.loc[2, "regime_oil"] == df.loc[0, "regime_oil"]
    finally:
        os.unlink(path)
