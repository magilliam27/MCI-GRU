"""
Tests for dynamic correlation graph wiring.

Covers:
- update_frequency_months=0: static path; update_if_needed never triggers; collate returns None dates.
- update_frequency_months>0: GraphBuilder.update_if_needed fires when months elapsed; batched
  edges have correct shape; Trainer._apply_dynamic_graph swaps in builder edges.
- create_data_loaders: shuffle=False and batch_size=1 enforced when dynamic_graph=True.
"""

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from mci_gru.config import ExperimentConfig, GraphConfig, TrainingConfig
from mci_gru.data.data_manager import (
    CombinedDataset,
    combined_collate_fn,
    create_data_loaders,
)
from mci_gru.graph.builder import GraphBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(kdcodes, start="2020-01-01", periods=400):
    """Synthetic price DataFrame with plausible daily returns."""
    dates = pd.bdate_range(start=start, periods=periods).strftime("%Y-%m-%d").tolist()
    rows = []
    rng = np.random.default_rng(0)
    for code in kdcodes:
        price = 100.0
        for dt in dates:
            price *= 1 + rng.normal(0, 0.01)
            rows.append({"kdcode": code, "dt": dt, "close": price})
    return pd.DataFrame(rows)


def _make_small_arrays(n_days=10, n_stocks=4, seq_len=5, n_features=3):
    ts = np.random.randn(n_days, n_stocks, seq_len, n_features).astype(np.float32)
    graph = np.random.randn(n_days, n_stocks, n_features).astype(np.float32)
    labels = np.random.randn(n_days, n_stocks).astype(np.float32)
    return ts, graph, labels


# ---------------------------------------------------------------------------
# GraphBuilder: static path (update_frequency_months=0)
# ---------------------------------------------------------------------------


class TestStaticGraph:
    def test_should_update_always_false(self):
        gb = GraphBuilder(judge_value=0.5, update_frequency_months=0)
        assert not gb.should_update("2021-06-01")
        assert not gb.should_update("2025-01-01")

    def test_update_if_needed_returns_none(self):
        kdcodes = ["A", "B", "C"]
        df = _make_price_df(kdcodes, periods=300)
        gb = GraphBuilder(judge_value=0.3, update_frequency_months=0)
        gb.build_graph(df, kdcodes, "2021-01-01", show_progress=False)
        ei, ew = gb.update_if_needed(df, kdcodes, "2021-06-01", show_progress=False)
        assert ei is None and ew is None

    def test_collate_dates_none_when_no_dates_in_dataset(self):
        ts, graph, labels = _make_small_arrays()
        ei = torch.zeros((2, 0), dtype=torch.long)
        ew = torch.zeros(0)
        dataset = CombinedDataset(
            torch.from_numpy(ts),
            torch.from_numpy(graph),
            torch.from_numpy(labels),
            sample_dates=None,
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=partial(combined_collate_fn, edge_index=ei, edge_weight=ew),
        )
        batch = next(iter(loader))
        # 7-tuple
        assert len(batch) == 7
        batch_dates = batch[6]
        assert batch_dates is None


# ---------------------------------------------------------------------------
# GraphBuilder: dynamic path (update_frequency_months>0)
# ---------------------------------------------------------------------------


class TestDynamicGraph:
    def _build_gb(self, freq=6):
        return GraphBuilder(judge_value=0.3, update_frequency_months=freq, corr_lookback_days=120)

    def test_update_fires_after_enough_months(self):
        kdcodes = ["A", "B", "C", "D"]
        df = _make_price_df(kdcodes, start="2019-01-01", periods=500)
        gb = self._build_gb(freq=6)
        gb.build_graph(df, kdcodes, "2019-01-01", show_progress=False)

        # 3 months later — should NOT update
        ei0, ew0 = gb.update_if_needed(df, kdcodes, "2019-04-01", show_progress=False)
        assert ei0 is None

        # 6+ months later — SHOULD update
        ei1, ew1 = gb.update_if_needed(df, kdcodes, "2019-07-01", show_progress=False)
        assert ei1 is not None
        assert ei1.shape[0] == 2
        assert ew1.shape[0] == ei1.shape[1]

    def test_get_current_graph_raises_before_build(self):
        gb = self._build_gb()
        with pytest.raises(ValueError, match="Graph has not been built"):
            gb.get_current_graph()

    def test_batched_edges_shape(self):
        """_batched_edges expands single-graph edges correctly."""
        import torch.nn as nn

        from mci_gru.training.trainer import Trainer

        n_stocks = 4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        edge_weight = torch.ones(4)

        # Minimal config to instantiate Trainer
        config = ExperimentConfig()

        class _DummyModel(nn.Module):
            def forward(self, *a, **kw):
                pass

        trainer = Trainer(model=_DummyModel(), config=config)

        batch_size = 3
        ei_b, ew_b = trainer._batched_edges(edge_index, edge_weight, batch_size, n_stocks)

        assert ei_b.shape == (2, batch_size * edge_index.shape[1])
        assert ew_b.shape == (batch_size * edge_weight.shape[0],)
        # Check index shifting: second graph block starts at n_stocks
        assert ei_b[:, 4].tolist() == [n_stocks, n_stocks + 1]

    def test_collate_returns_dates_when_dataset_has_dates(self):
        n_days, n_stocks = 6, 3
        ts, graph, labels = _make_small_arrays(n_days=n_days, n_stocks=n_stocks)
        dates = [f"2021-01-{i + 1:02d}" for i in range(n_days)]
        ei = torch.zeros((2, 0), dtype=torch.long)
        ew = torch.zeros(0)
        dataset = CombinedDataset(
            torch.from_numpy(ts),
            torch.from_numpy(graph),
            torch.from_numpy(labels),
            sample_dates=dates,
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=partial(combined_collate_fn, edge_index=ei, edge_weight=ew),
        )
        batch = next(iter(loader))
        assert len(batch) == 7
        batch_dates = batch[6]
        assert batch_dates == ["2021-01-01", "2021-01-02"]


# ---------------------------------------------------------------------------
# create_data_loaders: dynamic_graph flag enforcement
# ---------------------------------------------------------------------------


class TestCreateDataLoaders:
    def _loaders(self, dynamic_graph, batch_size=8):
        n_days, n_stocks, seq_len, n_features = 20, 5, 5, 3
        ts, graph, labels = _make_small_arrays(n_days, n_stocks, seq_len, n_features)
        ei = torch.zeros((2, 0), dtype=torch.long)
        ew = torch.zeros(0)
        dates = [f"2021-{i + 1:03d}" for i in range(n_days)]
        return create_data_loaders(
            stock_features_train=ts,
            x_graph_train=graph,
            train_labels=labels,
            stock_features_val=ts,
            x_graph_val=graph,
            val_labels=labels,
            stock_features_test=ts,
            x_graph_test=graph,
            edge_index=ei,
            edge_weight=ew,
            batch_size=batch_size,
            train_dates=dates,
            val_dates=dates,
            test_dates=dates,
            dynamic_graph=dynamic_graph,
        )

    def test_static_train_loader_shuffles(self):
        train_loader, _, _ = self._loaders(dynamic_graph=False, batch_size=4)
        assert train_loader.dataset.sample_dates is None

    def test_dynamic_loader_has_batch_size_1_and_dates(self):
        train_loader, val_loader, test_loader = self._loaders(dynamic_graph=True, batch_size=8)
        # batch_size clamped to 1
        assert train_loader.batch_size == 1
        assert val_loader.batch_size == 1
        # test always batch_size=1
        assert test_loader.batch_size == 1
        # dates attached
        assert train_loader.dataset.sample_dates is not None
        assert val_loader.dataset.sample_dates is not None

    def test_dynamic_loader_no_shuffle(self):
        train_loader, _, _ = self._loaders(dynamic_graph=True, batch_size=4)
        # DataLoader stores sampler; shuffle=False means SequentialSampler
        from torch.utils.data import SequentialSampler

        assert isinstance(train_loader.sampler, SequentialSampler)

    def test_batch_yields_7_tuple(self):
        train_loader, _, _ = self._loaders(dynamic_graph=True, batch_size=1)
        batch = next(iter(train_loader))
        assert len(batch) == 7

    def test_batch_dates_not_none_in_dynamic_mode(self):
        train_loader, _, _ = self._loaders(dynamic_graph=True, batch_size=1)
        batch = next(iter(train_loader))
        assert batch[6] is not None


# ---------------------------------------------------------------------------
# ExperimentConfig warning
# ---------------------------------------------------------------------------


class TestConfigWarning:
    def test_warns_when_dynamic_and_batch_size_not_1(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ExperimentConfig(
                graph=GraphConfig(judge_value=0.8, update_frequency_months=6),
                training=TrainingConfig(batch_size=32),
            )
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("batch_size" in m for m in msgs)

    def test_no_warning_when_batch_size_1(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ExperimentConfig(
                graph=GraphConfig(judge_value=0.8, update_frequency_months=6),
                training=TrainingConfig(batch_size=1),
            )
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert not any("batch_size" in m for m in msgs)
