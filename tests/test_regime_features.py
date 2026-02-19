import numpy as np
import pandas as pd

from mci_gru.features.regime import (
    REGIME_FEATURES,
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


def test_compute_regime_monthly_features_no_lookahead_exclusion_effect():
    regime_df = _make_regime_daily()

    # Same data, different exclusion window. Values after warmup should differ
    # because exclusion_months changes the allowed historical pool.
    monthly_ex0 = compute_regime_monthly_features(
        regime_df=regime_df,
        exclusion_months=0,
        min_history_months=12,
    )
    monthly_ex2 = compute_regime_monthly_features(
        regime_df=regime_df,
        exclusion_months=2,
        min_history_months=12,
    )

    joined = monthly_ex0.merge(monthly_ex2, on="dt", suffixes=("_ex0", "_ex2"))
    valid = joined.dropna(subset=["regime_global_score_ex0", "regime_global_score_ex2"])
    assert len(valid) > 10
    # If look-ahead/exclusion wasn't respected, these would be effectively identical.
    assert not np.allclose(
        valid["regime_global_score_ex0"].to_numpy(),
        valid["regime_global_score_ex2"].to_numpy(),
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


def test_regime_csv_contract_required_columns():
    """Canonical regime CSV must have dt + REGIME_VARIABLES; loader rejects missing columns."""
    import tempfile
    from mci_gru.data.data_manager import DataManager
    from mci_gru.config import DataConfig

    good = _make_regime_daily(periods=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        good.to_csv(f.name, index=False)
        path = f.name
    try:
        config = DataConfig(
            train_start="2000-01-01",
            train_end="2000-01-10",
            val_start="2000-01-11",
            val_end="2000-01-12",
            test_start="2000-01-13",
            test_end="2000-01-15",
        )
        dm = DataManager(config)
        out = dm.load_regime_inputs(regime_inputs_csv=path, regime_enforce_lag_days=0)
        assert set(["dt"] + REGIME_VARIABLES).issubset(set(out.columns))
        assert len(out) >= 1
    finally:
        import os
        os.unlink(path)

    bad = good.drop(columns=["regime_copper"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        bad.to_csv(f.name, index=False)
        path_bad = f.name
    try:
        dm2 = DataManager(config)
        try:
            dm2.load_regime_inputs(regime_inputs_csv=path_bad, regime_enforce_lag_days=0)
            assert False, "Expected ValueError for missing column"
        except ValueError as e:
            assert "regime_copper" in str(e) or "missing" in str(e).lower()
    finally:
        import os
        os.unlink(path_bad)


def test_regime_csv_lag_safety():
    """With regime_enforce_lag_days=1, value at date T should reflect prior-day data (no look-ahead)."""
    import tempfile
    from mci_gru.data.data_manager import DataManager
    from mci_gru.config import DataConfig

    df = _make_regime_daily(start="2000-01-01", periods=5)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        config = DataConfig(
            train_start="2000-01-01",
            train_end="2000-01-05",
            val_start="2000-01-06",
            val_end="2000-01-07",
            test_start="2000-01-08",
            test_end="2000-01-10",
        )
        dm = DataManager(config)
        no_lag = dm.load_regime_inputs(regime_inputs_csv=path, regime_enforce_lag_days=0)
        with_lag = dm.load_regime_inputs(regime_inputs_csv=path, regime_enforce_lag_days=1)
        # With lag, row at index i should have regime values from no_lag row i-1
        assert len(with_lag) == len(no_lag)
        # First row with lag is NaN/ffill so skip; check second row
        if len(no_lag) >= 2:
            for col in REGIME_VARIABLES:
                np.testing.assert_array_almost_equal(
                    with_lag.iloc[1][col],
                    no_lag.iloc[0][col],
                    err_msg=f"Lag not applied for {col}",
                )
    finally:
        import os
        os.unlink(path)
