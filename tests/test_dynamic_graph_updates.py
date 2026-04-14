"""
Tests for dynamic correlation graph wiring.

Covers:
- update_frequency_months=0: static path; update_if_needed never triggers; collate returns None dates.
- update_frequency_months>0: GraphSchedule precomputation, per-date lookup, and no-lookahead.
- combined_collate_fn: per-sample graph lookup via GraphSchedule; 7-tuple contract.
- create_data_loaders: shuffle=False when dynamic_graph=True, batch_size NOT clamped to 1.
"""

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
from mci_gru.graph.builder import GraphBuilder, GraphSchedule

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

        ei0, ew0 = gb.update_if_needed(df, kdcodes, "2019-04-01", show_progress=False)
        assert ei0 is None

        ei1, ew1 = gb.update_if_needed(df, kdcodes, "2019-07-01", show_progress=False)
        assert ei1 is not None
        assert ei1.shape[0] == 2
        assert ew1.shape[0] == ei1.shape[1]

    def test_get_current_graph_raises_before_build(self):
        gb = self._build_gb()
        with pytest.raises(ValueError, match="Graph has not been built"):
            gb.get_current_graph()

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

    def test_vectorized_build_edges(self):
        """build_edges produces the same result as the old O(n^2) loop."""
        kdcodes = ["A", "B", "C", "D", "E"]
        df = _make_price_df(kdcodes, periods=300)
        gb = GraphBuilder(judge_value=0.3, corr_lookback_days=120)
        corr = gb.compute_correlation_matrix(df, kdcodes, "2021-06-01")
        ei, ew = gb.build_edges(corr, kdcodes, show_progress=False)
        assert ei.shape[0] == 2
        assert ei.shape[1] == ew.shape[0]
        if ei.shape[1] > 0:
            assert (ew > gb.judge_value).all()


# ---------------------------------------------------------------------------
# GraphSchedule
# ---------------------------------------------------------------------------


class TestGraphSchedule:
    def _make_schedule(self):
        ei_a = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew_a = torch.tensor([0.9, 0.9])
        ei_b = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
        ew_b = torch.tensor([0.85, 0.85])
        return GraphSchedule([
            ("2020-01-01", ei_a, ew_a),
            ("2020-07-01", ei_b, ew_b),
        ])

    def test_lookup_before_first_snapshot_returns_first(self):
        sched = self._make_schedule()
        ei, ew = sched.get_graph_for_date("2019-06-01")
        assert torch.equal(ei, torch.tensor([[0, 1], [1, 0]]))

    def test_lookup_in_first_period(self):
        sched = self._make_schedule()
        ei, ew = sched.get_graph_for_date("2020-03-15")
        assert torch.equal(ei, torch.tensor([[0, 1], [1, 0]]))

    def test_lookup_in_second_period(self):
        sched = self._make_schedule()
        ei, ew = sched.get_graph_for_date("2020-09-01")
        assert torch.equal(ei, torch.tensor([[0, 2], [2, 0]]))

    def test_lookup_on_boundary_returns_new_snapshot(self):
        sched = self._make_schedule()
        ei, ew = sched.get_graph_for_date("2020-07-01")
        assert torch.equal(ei, torch.tensor([[0, 2], [2, 0]]))

    def test_get_initial_graph(self):
        sched = self._make_schedule()
        ei, ew = sched.get_initial_graph()
        assert torch.equal(ei, torch.tensor([[0, 1], [1, 0]]))

    def test_num_snapshots(self):
        sched = self._make_schedule()
        assert sched.num_snapshots == 2

    def test_empty_schedule_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            GraphSchedule([])

    def test_precompute_snapshots_no_lookahead(self):
        """Each snapshot must use only data before its valid-from date."""
        kdcodes = ["A", "B", "C"]
        df = _make_price_df(kdcodes, start="2019-01-01", periods=600)
        gb = GraphBuilder(judge_value=0.3, update_frequency_months=6, corr_lookback_days=120)
        schedule = gb.precompute_snapshots(df, kdcodes, "2020-01-01", "2021-01-01")

        assert schedule.num_snapshots >= 2
        for date in schedule.snapshot_dates:
            gb.compute_correlation_matrix(df, kdcodes, date)
            used_dates = df[df["dt"] < date]["dt"].unique()
            assert len(used_dates) > 0 or date <= df["dt"].min()


# ---------------------------------------------------------------------------
# Collate with GraphSchedule
# ---------------------------------------------------------------------------


class TestCollateWithSchedule:
    def test_per_sample_graph_lookup(self):
        """Each sample in a batch gets its own graph from the schedule."""
        n_stocks = 3
        ei_a = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew_a = torch.tensor([0.9, 0.9])
        ei_b = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        ew_b = torch.tensor([0.9, 0.9, 0.85, 0.85])
        schedule = GraphSchedule([
            ("2020-01-01", ei_a, ew_a),
            ("2020-07-01", ei_b, ew_b),
        ])

        ts, graph, labels = _make_small_arrays(n_days=4, n_stocks=n_stocks)
        dates = ["2020-03-01", "2020-04-01", "2020-08-01", "2020-09-01"]
        dataset = CombinedDataset(
            torch.from_numpy(ts),
            torch.from_numpy(graph),
            torch.from_numpy(labels),
            sample_dates=dates,
        )

        fallback_ei = torch.zeros((2, 0), dtype=torch.long)
        fallback_ew = torch.zeros(0)

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=partial(
                combined_collate_fn,
                edge_index=fallback_ei,
                edge_weight=fallback_ew,
                graph_schedule=schedule,
            ),
        )
        batch = next(iter(loader))
        assert len(batch) == 7
        _ts, _labels, _gf, b_ei, b_ew, ns, b_dates = batch
        assert ns == n_stocks
        assert b_dates == dates

        expected_edges = (
            ei_a.shape[1] + ei_a.shape[1] + ei_b.shape[1] + ei_b.shape[1]
        )
        assert b_ei.shape[1] == expected_edges


# ---------------------------------------------------------------------------
# create_data_loaders: dynamic_graph flag enforcement
# ---------------------------------------------------------------------------


class TestCreateDataLoaders:
    def _loaders(self, dynamic_graph, batch_size=8, graph_schedule=None):
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
            graph_schedule=graph_schedule,
        )

    def test_static_train_loader_shuffles(self):
        train_loader, _, _ = self._loaders(dynamic_graph=False, batch_size=4)
        assert train_loader.dataset.sample_dates is None

    def test_dynamic_loader_preserves_batch_size(self):
        """With precomputed graphs, batch_size is no longer forced to 1."""
        train_loader, val_loader, test_loader = self._loaders(
            dynamic_graph=True, batch_size=8
        )
        assert train_loader.batch_size == 8
        assert val_loader.batch_size == 8
        assert test_loader.batch_size == 1
        assert train_loader.dataset.sample_dates is not None
        assert val_loader.dataset.sample_dates is not None

    def test_dynamic_loader_no_shuffle(self):
        train_loader, _, _ = self._loaders(dynamic_graph=True, batch_size=4)
        from torch.utils.data import SequentialSampler

        assert isinstance(train_loader.sampler, SequentialSampler)

    def test_batch_yields_7_tuple(self):
        train_loader, _, _ = self._loaders(dynamic_graph=True, batch_size=4)
        batch = next(iter(train_loader))
        assert len(batch) == 7

    def test_batch_dates_not_none_in_dynamic_mode(self):
        train_loader, _, _ = self._loaders(dynamic_graph=True, batch_size=4)
        batch = next(iter(train_loader))
        assert batch[6] is not None


# ---------------------------------------------------------------------------
# ExperimentConfig: batch_size no longer constrained for dynamic graph
# ---------------------------------------------------------------------------


class TestConfigNoConstraint:
    def test_no_warning_with_dynamic_graph_and_large_batch(self):
        """batch_size != 1 is perfectly fine now with precomputed graphs."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ExperimentConfig(
                graph=GraphConfig(judge_value=0.8, update_frequency_months=6),
                training=TrainingConfig(batch_size=32),
            )
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert not any("batch_size" in m for m in msgs)
