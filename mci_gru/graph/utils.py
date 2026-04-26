"""Shared graph configuration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mci_gru.config import GraphConfig


def edge_feature_dim(graph_cfg: GraphConfig | dict) -> int:
    """Final edge feature width consumed by GAT blocks."""
    if isinstance(graph_cfg, dict):
        use_multi = bool(graph_cfg.get("use_multi_feature_edges", False))
        use_lead_lag = bool(graph_cfg.get("use_lead_lag_features", False))
        append_age = bool(graph_cfg.get("append_snapshot_age_days", False))
    else:
        use_multi = graph_cfg.use_multi_feature_edges
        use_lead_lag = graph_cfg.use_lead_lag_features
        append_age = graph_cfg.append_snapshot_age_days

    if not use_multi:
        return 1
    width = 4
    if use_lead_lag:
        width += 2
    if append_age:
        width += 1
    return width
