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

    The third element of each snapshot tuple ("edge_weight") may be either a
    1-D tensor of shape ``(E,)`` (legacy scalar edge weight) or a 2-D tensor
    of shape ``(E, F)`` (multi-feature edges). The class is shape-agnostic;
    consumers must handle both shapes.
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
        """Return (edge_index, edge_attr) valid for *date*.

        ``edge_attr`` is shape ``(E,)`` in legacy mode and ``(E, F)`` in
        multi-feature mode (see class docstring).
        """
        idx = bisect.bisect_right(self._dates, date) - 1
        idx = max(idx, 0)
        return self._edge_indices[idx], self._edge_weights[idx]

    def snapshot_valid_from_for_date(self, date: str) -> str:
        """Return the snapshot ``valid_from`` date string active on *date*."""
        idx = bisect.bisect_right(self._dates, date) - 1
        idx = max(idx, 0)
        return self._dates[idx]

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
                f"top_k_metric must be one of {self._VALID_TOP_K_METRICS}, "
                f"got {top_k_metric!r}"
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
        """Aligned daily return matrix (dates × stocks) up to *end_date* (exclusive)."""
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
        return pivot

    def compute_correlation_matrix(
        self, df: pd.DataFrame, kdcode_list: list[str], end_date: str
    ) -> pd.DataFrame:
        """Per paper Section 3.3.2: use past year of returns for correlation."""
        pivot = self._daily_returns_pivot(df, kdcode_list, end_date)
        return pivot.corr()

    def build_edges(
        self,
        corr_matrix: pd.DataFrame,
        kdcode_list: list[str],
        show_progress: bool = True,
        returns_pivot: pd.DataFrame | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build edge tensors from a correlation matrix using vectorised numpy ops.

        Two selection paths and two output shapes are supported:

        Selection (controlled by ``self.top_k``):
        - ``top_k == 0`` (legacy): keep edges with ``corr > self.judge_value``.
        - ``top_k > 0``: per-row K-nearest selection by ``self.top_k_metric``
          ("corr" => K most-positive; "abs_corr" => K largest by |corr|, which
          recovers strong negative correlations - the "1b-lite" path).

        Output (controlled by ``self.use_multi_feature_edges``):
        - ``False`` (legacy): returns ``(edge_index (2,E), edge_weight (E,))`` with
          the signed correlation as the scalar weight.
        - ``True``: returns ``(edge_index (2,E), edge_attr (E,4+))`` with columns
          ``[corr, |corr|, corr^2, rank_pct]`` plus optional lead–lag columns when
          ``use_lead_lag_features`` and *returns_pivot* are set.
        """
        corr = corr_matrix.values

        if self.top_k > 0:
            rows, cols, kept_corr, rank_pct = self._select_edges_topk(corr)
        else:
            rows, cols, kept_corr = self._select_edges_threshold(corr)
            rank_pct = np.zeros_like(kept_corr)

        n_extra = 2 if (self.use_lead_lag_features and self.use_multi_feature_edges) else 0
        base_f = 4

        if len(rows) == 0:
            empty_attr = (
                torch.zeros((0, base_f + n_extra), dtype=torch.float)
                if self.use_multi_feature_edges
                else torch.zeros(0, dtype=torch.float)
            )
            return torch.zeros((2, 0), dtype=torch.long), empty_attr

        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

        if self.use_multi_feature_edges:
            edge_attr = np.stack(
                [kept_corr, np.abs(kept_corr), kept_corr * kept_corr, rank_pct],
                axis=1,
            ).astype(np.float32)
            if self.use_lead_lag_features and returns_pivot is not None:
                lag_n, lag_c = self._lead_lag_columns(returns_pivot, rows, cols)
                edge_attr = np.concatenate([edge_attr, lag_n[:, None], lag_c[:, None]], axis=1)
            return edge_index, torch.from_numpy(edge_attr)

        return edge_index, torch.tensor(kept_corr, dtype=torch.float)

    def _lead_lag_columns(
        self,
        returns_pivot: pd.DataFrame,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Best lag (among 0 and ``lead_lag_days``) by strongest |corr|; signed corr stored."""
        vals = returns_pivot.to_numpy(dtype=np.float64)
        if vals.size == 0:
            z = np.zeros(len(rows), dtype=np.float32)
            return z, z
        max_lag = max(self.lead_lag_days) if self.lead_lag_days else 1
        lag_norms = np.zeros(len(rows), dtype=np.float32)
        lag_corrs = np.zeros(len(rows), dtype=np.float32)
        for e, (ri, ci) in enumerate(zip(rows, cols)):
            a = vals[:, int(ri)]
            b = vals[:, int(ci)]
            L = min(len(a), len(b))
            if L < 5:
                continue
            a = a[:L]
            b = b[:L]
            best_abs = -1.0
            best_lag = 0
            best_c = 0.0
            for lag in [0, *self.lead_lag_days]:
                if lag == 0:
                    aa, bb = a, b
                else:
                    if lag >= L:
                        continue
                    aa = a[: L - lag]
                    bb = b[lag:L]
                m = len(aa)
                if m < 5 or np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
                    continue
                cc = float(np.corrcoef(aa, bb)[0, 1])
                if abs(cc) > best_abs:
                    best_abs = abs(cc)
                    best_lag = lag
                    best_c = cc
            lag_norms[e] = float(best_lag) / float(max_lag) if max_lag > 0 else 0.0
            lag_corrs[e] = float(best_c)
        return lag_norms, lag_corrs

    def _select_edges_threshold(
        self, corr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy threshold-based selection: keep ``corr > judge_value`` off-diagonal."""
        mask = (~np.isnan(corr)) & (corr > self.judge_value)
        np.fill_diagonal(mask, False)
        rows, cols = np.where(mask)
        kept_corr = corr[rows, cols].astype(np.float64)
        return rows, cols, kept_corr

    def _select_edges_topk(
        self, corr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-row top-K neighbour selection by ``self.top_k_metric``.

        Diagonal entries and NaNs are masked to ``-inf`` so they cannot win a
        slot.  When a row has fewer than ``top_k`` valid candidates (rare; e.g.
        a tiny universe), only the valid ones are kept and ``rank_pct`` is
        scaled against the actual kept count.

        Returns
        -------
        rows, cols : np.ndarray
            Index pairs of kept directed edges.
        kept_corr : np.ndarray
            Signed correlation ``corr[row, col]`` for each kept edge.  This is
            the *signed* value even when ``top_k_metric == "abs_corr"`` so that
            column 0 of the multi-feature edge tensor preserves sign.
        rank_pct : np.ndarray
            Within-row percentile rank by the selection metric, in ``(0, 1]``,
            with 1.0 = strongest neighbour in that row.
        """
        n = corr.shape[0]
        score = np.where(np.isnan(corr), -np.inf, corr)
        if self.top_k_metric == "abs_corr":
            score = np.where(np.isfinite(score), np.abs(score), -np.inf)
        np.fill_diagonal(score, -np.inf)

        valid_mask = np.isfinite(score)
        valid_per_row = valid_mask.sum(axis=1)
        k_per_row = np.minimum(valid_per_row, self.top_k)

        rows_out: list[np.ndarray] = []
        cols_out: list[np.ndarray] = []
        kept_corr_out: list[np.ndarray] = []
        rank_pct_out: list[np.ndarray] = []

        for i in range(n):
            k_i = int(k_per_row[i])
            if k_i == 0:
                continue
            row_score = score[i]
            sorted_idx = np.argpartition(-row_score, kth=k_i - 1)[:k_i]
            sorted_idx = sorted_idx[np.argsort(-row_score[sorted_idx])]

            positions = np.arange(1, k_i + 1, dtype=np.float64)
            rank_pct_row = (k_i - positions + 1) / k_i

            rows_out.append(np.full(k_i, i, dtype=np.int64))
            cols_out.append(sorted_idx.astype(np.int64))
            kept_corr_out.append(corr[i, sorted_idx].astype(np.float64))
            rank_pct_out.append(rank_pct_row)

        if not rows_out:
            empty_i = np.zeros(0, dtype=np.int64)
            empty_f = np.zeros(0, dtype=np.float64)
            return empty_i, empty_i, empty_f, empty_f

        rows = np.concatenate(rows_out)
        cols = np.concatenate(cols_out)
        kept_corr = np.concatenate(kept_corr_out)
        rank_pct = np.concatenate(rank_pct_out)
        return rows, cols, kept_corr, rank_pct

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
        feat_mode = (
            f"multi-feature({n_feat})" if self.use_multi_feature_edges else "scalar"
        )
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
