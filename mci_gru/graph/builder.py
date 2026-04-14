"""Correlation-based stock graphs: static or dynamic (periodic) updates.

GraphBuilder computes Pearson-correlation graphs.  GraphSchedule holds a
time-indexed sequence of precomputed snapshots so that dynamic-graph mode
no longer requires batch_size=1 during training.
"""

import bisect
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta


class GraphSchedule:
    """Pre-computed graph snapshots indexed by their valid-from date.

    Each snapshot covers the period from its ``valid_from`` date until the
    next snapshot's ``valid_from`` (or the end of time for the last entry).
    Lookups use bisect for O(log n) per query.
    """

    def __init__(
        self,
        snapshots: list[tuple[str, torch.Tensor, torch.Tensor]],
    ):
        if not snapshots:
            raise ValueError("GraphSchedule requires at least one snapshot")
        self._dates: list[str] = [s[0] for s in snapshots]
        self._edge_indices: list[torch.Tensor] = [s[1] for s in snapshots]
        self._edge_weights: list[torch.Tensor] = [s[2] for s in snapshots]

    def get_graph_for_date(self, date: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (edge_index, edge_weight) valid for *date*."""
        idx = bisect.bisect_right(self._dates, date) - 1
        idx = max(idx, 0)
        return self._edge_indices[idx], self._edge_weights[idx]

    def get_initial_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first snapshot (used for graph_data.pt / static fallback)."""
        return self._edge_indices[0], self._edge_weights[0]

    @property
    def num_snapshots(self) -> int:
        return len(self._dates)

    @property
    def snapshot_dates(self) -> list[str]:
        return list(self._dates)


class GraphBuilder:
    def __init__(
        self,
        judge_value: float = 0.8,
        update_frequency_months: int = 0,
        corr_lookback_days: int = 252,
    ):
        self.judge_value = judge_value
        self.update_frequency_months = update_frequency_months
        self.corr_lookback_days = corr_lookback_days
        self.last_update_date: str | None = None
        self.current_edge_index: torch.Tensor | None = None
        self.current_edge_weight: torch.Tensor | None = None
        self.correlation_matrix: pd.DataFrame | None = None

    def compute_correlation_matrix(
        self, df: pd.DataFrame, kdcode_list: list[str], end_date: str
    ) -> pd.DataFrame:
        """Per paper Section 3.3.2: use past year of returns for correlation."""
        df = df.copy()

        if "prev_close" in df.columns:
            df["daily_return"] = df["close"] / df["prev_close"] - 1
        else:
            df = df.sort_values(["kdcode", "dt"])
            df["daily_return"] = df.groupby("kdcode")["close"].pct_change()

        df = df[df["dt"] < end_date]
        all_dates = sorted(df["dt"].unique())
        if len(all_dates) > self.corr_lookback_days:
            start_date = all_dates[-self.corr_lookback_days]
            df = df[df["dt"] >= start_date]

        df = df[df["kdcode"].isin(kdcode_list)]
        pivot = df.pivot_table(index="dt", columns="kdcode", values="daily_return")
        pivot = pivot.reindex(columns=kdcode_list)
        pivot = pivot.fillna(0)
        corr_matrix = pivot.corr()

        return corr_matrix

    def build_edges(
        self, corr_matrix: pd.DataFrame, kdcode_list: list[str], show_progress: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build edge tensors from a correlation matrix using vectorised numpy ops."""
        corr = corr_matrix.values
        mask = (~np.isnan(corr)) & (corr > self.judge_value)
        np.fill_diagonal(mask, False)
        rows, cols = np.where(mask)

        if len(rows) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float),
            )

        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        edge_weight = torch.tensor(corr[rows, cols], dtype=torch.float)
        return edge_index, edge_weight

    def build_graph(
        self, df: pd.DataFrame, kdcode_list: list[str], end_date: str, show_progress: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print(
            f"Building graph (judge_value={self.judge_value}, lookback={self.corr_lookback_days} days)..."
        )
        self.correlation_matrix = self.compute_correlation_matrix(df, kdcode_list, end_date)
        edge_index, edge_weight = self.build_edges(
            self.correlation_matrix, kdcode_list, show_progress
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
