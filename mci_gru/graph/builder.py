"""Correlation-based stock graphs: static or dynamic (periodic) updates.

GraphBuilder computes Pearson-correlation graphs.  GraphSchedule holds a
time-indexed sequence of precomputed snapshots so that dynamic-graph mode
no longer requires batch_size=1 during training.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

from mci_gru.data.pit import active_membership_mask
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


def admissible_mask_for_date(
    kdcode_list: list[str],
    date: str,
    pit_intervals: pd.DataFrame | None,
) -> np.ndarray | None:
    """Boolean mask over *kdcode_list* of names admissible on *date*.

    Returns ``None`` when *pit_intervals* is ``None``, so callers can pass the
    result straight through and keep the no-PIT path byte-identical to its
    behaviour before issue #123.
    """
    if pit_intervals is None:
        return None
    return active_membership_mask(kdcode_list, [date], pit_intervals)[0]


logger = logging.getLogger(__name__)

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
        exclude_edge_pairs: list[tuple[str, str]] | None = None,
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
        self.exclude_edge_pairs = (
            [tuple(pair) for pair in exclude_edge_pairs] if exclude_edge_pairs else []
        )
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
        admissible_mask: np.ndarray | None = None,
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
            admissible_mask,
            exclude_pairs=self.exclude_edge_pairs or None,
        )

    def build_graph(
        self,
        df: pd.DataFrame,
        kdcode_list: list[str],
        end_date: str,
        show_progress: bool = True,
        admissible_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k > 0:
            mode = f"top_k={self.top_k} ({self.top_k_metric})"
        else:
            mode = f"judge_value={self.judge_value}"
        n_feat = 4
        if self.use_multi_feature_edges and self.use_lead_lag_features:
            n_feat += 2
        feat_mode = f"multi-feature({n_feat})" if self.use_multi_feature_edges else "scalar"
        logger.info(
            f"Building graph ({mode}, lookback={self.corr_lookback_days} days, "
            f"edges={feat_mode})..."
        )
        if self.use_multi_feature_edges and self.top_k == 0:
            # Issue #114, corrected by issue #170. rank_pct is populated only by
            # top-K selection, so on the threshold path column 3 is a constant
            # zero. GATConv.lin_edge has no bias, so that column is bitwise inert:
            # it receives zero gradient and cannot change the output. That half of
            # the warning holds at every threshold, which is why the condition is
            # not narrowed -- doing so would leave the negative-threshold arm with
            # a silently dead channel.
            #
            # The |corr| half is conditional. `corr > judge_value` can admit only
            # positive correlations while judge_value >= 0, and GraphConfig has
            # accepted the whole of [-1, 1) since issue #162. Zero sits on the
            # degenerate side: the comparison is strict, so judge_value == 0 still
            # keeps positives only.
            if self.judge_value >= 0:
                logger.warning(
                    "graph.use_multi_feature_edges=true with graph.top_k=0: of the 4 edge "
                    "channels [corr, |corr|, corr^2, rank_pct], rank_pct is identically "
                    "zero and |corr| duplicates corr, so the tensor has numerical rank 2. "
                    "The model is still sized for %d channels. Set graph.top_k>0 to "
                    "populate rank_pct, or graph.use_multi_feature_edges=false for a "
                    "scalar edge weight.",
                    n_feat,
                )
            else:
                logger.warning(
                    "graph.use_multi_feature_edges=true with graph.top_k=0: of the 4 edge "
                    "channels [corr, |corr|, corr^2, rank_pct], rank_pct is identically "
                    "zero, so that channel is inert and the tensor has numerical rank at "
                    "most 3. judge_value=%s is negative, so selection is not restricted "
                    "to positive correlations: |corr| stays a copy of corr only if every "
                    "kept pair happens to be non-negative. The model is still sized for "
                    "%d channels. Set graph.top_k>0 to populate rank_pct, or "
                    "graph.use_multi_feature_edges=false for a scalar edge weight.",
                    self.judge_value,
                    n_feat,
                )
        if self.exclude_edge_pairs:
            axis = set(kdcode_list)
            for pair in self.exclude_edge_pairs:
                if pair[0] not in axis or pair[1] not in axis:
                    logger.warning(
                        "graph.exclude_edge_pairs: %s is not fully on this node axis; "
                        "exclusion is a no-op for it",
                        pair,
                    )
        pivot = self._daily_returns_pivot(df, kdcode_list, end_date)
        self.correlation_matrix = pivot.corr()
        rp = pivot if self.use_lead_lag_features else None
        edge_index, edge_weight = self.build_edges(
            self.correlation_matrix,
            kdcode_list,
            show_progress,
            returns_pivot=rp,
            admissible_mask=admissible_mask,
        )
        self.last_update_date = end_date
        self.current_edge_index = edge_index
        self.current_edge_weight = edge_weight

        logger.info(f"  Graph built: {edge_index.shape[1]} edges for {len(kdcode_list)} nodes")

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
        first_sample_date: str | None = None,
        pit_intervals: pd.DataFrame | None = None,
    ) -> GraphSchedule:
        """Build all graph snapshots up-front and return a ``GraphSchedule``.

        The schedule covers *start_date* through *end_date*, with one snapshot
        per update interval.  Each snapshot uses only data **before** its
        valid-from date (no lookahead).

        *first_sample_date* is the earliest date the schedule will be asked
        about; supplying it lets ``GraphSchedule`` assert its readiness contract
        (mandatory warm-up plus first-sample coverage) at construction.

        *pit_intervals* is an optional point-in-time membership table
        (``kdcode``, ``valid_from``, ``valid_to``).  When supplied, each
        snapshot restricts edge **selection** to the names admissible at that
        snapshot's own date, which is the only place per-snapshot admissibility
        can be applied -- the caller supplies one ``kdcode_list`` for the whole
        schedule and cannot vary it per date.  See issue #123.

        The node axis is never narrowed: inadmissible names keep their index so
        the fixed-axis contract every downstream tensor depends on is preserved.
        Only their edges are withheld.
        """
        update_dates = self.get_update_dates(start_date, end_date)
        snapshots: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        logger.info(
            f"Precomputing {len(update_dates)} graph snapshot(s) "
            f"({start_date} to {end_date}, every {self.update_frequency_months} months)"
            f"{', PIT-restricted selection' if pit_intervals is not None else ''}..."
        )

        for date in update_dates:
            mask = admissible_mask_for_date(kdcode_list, date, pit_intervals)
            ei, ew = self.build_graph(
                df, kdcode_list, date, show_progress=False, admissible_mask=mask
            )
            snapshots.append((date, ei, ew))

        warmup_sessions = int(df.loc[df["dt"] < update_dates[0], "dt"].nunique())
        schedule = GraphSchedule(
            snapshots,
            corr_lookback_days=self.corr_lookback_days,
            sessions_before_first_snapshot=warmup_sessions,
            first_sample_date=first_sample_date,
        )
        logger.info(
            f"  GraphSchedule ready: {schedule.num_snapshots} snapshots, "
            f"{warmup_sessions} warm-up session(s) before {update_dates[0]} "
            f"(readiness verified: {schedule.is_ready})"
        )
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

        logger.info(
            f"Updating graph (last update: {self.last_update_date}, current: {current_date})"
        )
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
