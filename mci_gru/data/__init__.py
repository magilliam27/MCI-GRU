"""
Data loading, management, and preprocessing for MCI-GRU experiments.

Modules:
- data_manager: Unified data interface and data loaders
- preprocessing: Tensor construction and label computation
- lseg_loader: LSEG/Refinitiv data fetching
- universes: Stock universe definitions
"""

from mci_gru.data.universes import UNIVERSES, get_universe_info
from mci_gru.data.data_manager import DataManager
from mci_gru.data.preprocessing import (
    generate_time_series_features,
    generate_graph_features,
    compute_labels,
    apply_rank_labels,
)

__all__ = [
    "UNIVERSES",
    "get_universe_info",
    "DataManager",
    "generate_time_series_features",
    "generate_graph_features",
    "compute_labels",
    "apply_rank_labels",
]
