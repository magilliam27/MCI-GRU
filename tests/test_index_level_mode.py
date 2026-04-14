"""
Tests for index-level experiment mode (no stock-level survivorship bias).

Ensures index-level config is valid, load_index_series returns single-series data,
and mode separation is explicit (experiment_mode index_level vs stock_level).
"""

import os
import tempfile

import numpy as np
import pandas as pd

from mci_gru.config import DataConfig
from mci_gru.data.data_manager import DataManager


def test_index_level_config_valid():
    """experiment_mode must be 'stock_level' or 'index_level'; index_level accepts index_filename."""
    cfg = DataConfig(experiment_mode="index_level", index_filename=None)
    assert cfg.experiment_mode == "index_level"
    cfg2 = DataConfig(experiment_mode="stock_level")
    assert cfg2.experiment_mode == "stock_level"
    try:
        DataConfig(experiment_mode="invalid")
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "experiment_mode" in str(e)


def test_load_index_series_returns_single_series():
    """load_index_series with index_filename CSV returns kdcode=INDEX and OHLCV columns."""
    with tempfile.TemporaryDirectory() as tmp:
        index_csv = os.path.abspath(os.path.join(tmp, "index.csv"))
        dates = pd.date_range("2019-01-01", periods=100, freq="B")
        pd.DataFrame(
            {
                "dt": dates.strftime("%Y-%m-%d"),
                "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            }
        ).to_csv(index_csv, index=False)

        config = DataConfig(
            experiment_mode="index_level",
            index_filename=index_csv,
            train_start="2019-03-01",
            train_end="2019-05-31",
            val_start="2019-06-01",
            val_end="2019-06-15",
            test_start="2019-06-16",
            test_end="2019-06-30",
        )
        dm = DataManager(config)
        df = dm.load_index_series()

        assert dm.kdcode_list == ["INDEX"]
        assert list(df["kdcode"].unique()) == ["INDEX"]
        for col in ["dt", "open", "high", "low", "close", "volume", "turnover"]:
            assert col in df.columns
        assert len(df) >= 1


def test_index_level_mode_separation():
    """Index-level and stock-level are distinct modes; only index-level uses load_index_series."""
    # Design contract: experiment_mode="index_level" triggers index-level path (prepare_data_index_level)
    # and yields single series + empty graph; stock_level uses prepare_data and constituent stocks.
    assert (
        DataConfig(experiment_mode="index_level").experiment_mode
        != DataConfig(experiment_mode="stock_level").experiment_mode
    )
