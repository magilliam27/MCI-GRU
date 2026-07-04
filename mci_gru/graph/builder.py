"""Correlation-based stock graphs: static or dynamic (periodic) updates.

GraphBuilder computes Pearson-correlation graphs.  GraphSchedule holds a
time-indexed sequence of precomputed snapshots so that dynamic-graph mode
no longer requires batch_size=1 during training.
"""

from datetime import datetime

import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

from mci_gru.graph.correlation import (
    _daily_returns_pivot as _daily_returns_pivot_impl,
)
from mci_gru.graph.correlation import (
    build_edges as build_edges_impl,
)
from mci_gru.graph.correlation import (
    compute_correlation_matrix as compute_correlation_matrix_impl,
)
from mci_gru.graph.schedule import GraphSchedule

__all__ = ["GraphBuilder", "GraphSchedule"]


class GraphBuilder:
    _VALID_TOP_K_METRICS = ("corr", "abs_corr")

    def __init__(
        self,
        judge_value: float = 0.8,
        update_frequency_months: int = 0,
        corr_lookback_days: int = 252,
        top_k: int = 0,
        top_k_metric: str = "corr",
        use_multi_feature_edges: bool = False,
        use_lead_lag_features: bool = False,
        lead_lag_days: list[int] | None = None,
    ):
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if top_k_metric not in self._VALID_TOP_K_METRICS:
            raise ValueError(
                f"top_k_metric must be one of {self._VALID_TOP_K_METRICS}, got {top_k_metric!r}"
            )

        self.judge_value = judge_value
        self.update_frequency_months = update_frequency_months
        self.corr_lookback_days = corr_lookback_days
        self.top_k = top_k
        self.top_k_metric = top_k_metric
        self.use_multi_feature_edges = use_multi_feature_edges
        self.use_lead_lag_features = use_lead_lag_features
        self.lead_lag_days = list(lead_lag_days) if lead_lag_days is not None else [1, 2, 3, 5]
        self.last_update_date: str | None = None
        self.current_edge_index: torch.Tensor | None = None
        self.current_edge_weight: torch.Tensor | None = None
        self.correlation_matrix: pd.DataFrame | None = None

    def _daily_returns_pivot(
        self, df: pd.DataFrame, kdcode_list: list[str], end_date: str
    ) -> pd.DataFrame:
        return _daily_returns_pivot_impl(df, kdcode_list, end_date, self.corr_lookback_days)

    def compute_correlation_matrix(
        self, df: pd.DataFrame, kdcode_list: list[str], end_date: str
    ) -> pd.DataFrame:
        return compute_correlation_matrix_impl(df, kdcode_list, end_date, self.corr_lookback_days)

    def build_edges(
        self,
        corr_matrix: pd.DataFrame,
        kdcode_list: list[str],
        show_progress: bool = True,
        returns_pivot: pd.DataFrame | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return build_edges_impl(
            corr_matrix,
            kdcode_list,
            show_progress,
            returns_pivot,
            self.judge_value,
            self.top_k,
            self.top_k_metric,
            self.use_multi_feature_edges,
            self.use_lead_lag_features,
            self.lead_lag_days,
        )

    def build_graph(
        self, df: pd.DataFrame, kdcode_list: list[str], end_date: str, show_progress: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k > 0:
            mode = f"top_k={self.top_k} ({self.top_k_metric})"
        else:
            mode = f"judge_value={self.judge_value}"
        n_feat = 4
        if self.use_multi_feature_edges and self.use_lead_lag_features:
            n_feat += 2
        feat_mode = f"multi-feature({n_feat})" if self.use_multi_feature_edges else "scalar"
        print(
            f"Building graph ({mode}, lookback={self.corr_lookback_days} days, "
            f"edges={feat_mode})..."
        )
        pivot = self._daily_returns_pivot(df, kdcode_list, end_date)
        self.correlation_matrix = pivot.corr()
        rp = pivot if self.use_lead_lag_features else None
        edge_index, edge_weight = self.build_edges(
            self.correlation_matrix, kdcode_list, show_progress, returns_pivot=rp
        )
        self.last_update_date = end_date
        self.current_edge_index = edge_index
        self.current_edge_weight = edge_weight

        print(f"  Graph built: {edge_index.shape[1]} edges for {len(kdcode_list)} nodes")

        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Pre-computation API (replaces lazy per-batch rebuilding)
    # ------------------------------------------------------------------

    def precompute_snapshots(
        self,
        df: pd.DataFrame,
        kdcode_list: list[str],
        start_date: str,
        end_date: str,
    ) -> GraphSchedule:
        """Build all graph snapshots up-front and return a ``GraphSchedule``.

        The schedule covers *start_date* through *end_date*, with one snapshot
        per update interval.  Each snapshot uses only data **before** its
        valid-from date (no lookahead).
        """
        update_dates = self.get_update_dates(start_date, end_date)
        snapshots: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        print(
            f"Precomputing {len(update_dates)} graph snapshot(s) "
            f"({start_date} to {end_date}, every {self.update_frequency_months} months)..."
        )

        for date in update_dates:
            ei, ew = self.build_graph(df, kdcode_list, date, show_progress=False)
            snapshots.append((date, ei, ew))

        schedule = GraphSchedule(snapshots)
        print(f"  GraphSchedule ready: {schedule.num_snapshots} snapshots")
        return schedule

    # ------------------------------------------------------------------
    # Legacy lazy-update helpers (kept for backward compat / tests)
    # ------------------------------------------------------------------

    def should_update(self, current_date: str) -> bool:
        if self.update_frequency_months == 0:
            return False

        if self.last_update_date is None:
            return True

        try:
            last_update = datetime.strptime(self.last_update_date, "%Y-%m-%d")
            current = datetime.strptime(current_date, "%Y-%m-%d")
        except ValueError:
            last_update = pd.to_datetime(self.last_update_date)
            current = pd.to_datetime(current_date)

        months_elapsed = (current.year - last_update.year) * 12 + (
            current.month - last_update.month
        )

        return months_elapsed >= self.update_frequency_months

    def update_if_needed(
        self,
        df: pd.DataFrame,
        kdcode_list: list[str],
        current_date: str,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.should_update(current_date):
            return None, None

        print(f"Updating graph (last update: {self.last_update_date}, current: {current_date})")
        return self.build_graph(df, kdcode_list, current_date, show_progress)

    def get_current_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_edge_index is None or self.current_edge_weight is None:
            raise ValueError("Graph has not been built yet. Call build_graph() first.")
        return self.current_edge_index, self.current_edge_weight

    def get_update_dates(self, start_date: str, end_date: str) -> list[str]:
        if self.update_frequency_months == 0:
            return [start_date]

        update_dates = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            update_dates.append(current.strftime("%Y-%m-%d"))
            current = current + relativedelta(months=self.update_frequency_months)

        return update_dates

    def get_stats(self) -> dict:
        if self.current_edge_index is None:
            return {"built": False}

        n_edges = self.current_edge_index.shape[1]
        n_unique_edges = n_edges // 2

        stats = {
            "built": True,
            "last_update_date": self.last_update_date,
            "n_edges": n_edges,
            "n_unique_edges": n_unique_edges,
            "judge_value": self.judge_value,
            "update_frequency_months": self.update_frequency_months,
        }

        if self.current_edge_weight is not None and len(self.current_edge_weight) > 0:
            stats["avg_edge_weight"] = float(self.current_edge_weight.mean())
            stats["min_edge_weight"] = float(self.current_edge_weight.min())
            stats["max_edge_weight"] = float(self.current_edge_weight.max())

        return stats
