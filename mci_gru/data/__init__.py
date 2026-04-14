"""
Data loading, management, and preprocessing for MCI-GRU experiments.

Modules:
- data_manager: Unified data interface, data loaders, and DataLoader creation
- preprocessing: Tensor construction (time series, graph features) and label computation
- reshape: Shared LSEG MultiIndex-to-flat-OHLCV transformation
- lseg_loader: LSEG/Refinitiv data fetching
- fred_loader: FRED API loader for credit spreads and macro series
- path_resolver: Project-aware data file path resolution
- universes: Stock universe definitions (S&P 500, Russell 1000, etc.)
"""

from mci_gru.data.data_manager import DataManager
from mci_gru.data.preprocessing import (
    apply_rank_labels,
    compute_labels,
    generate_graph_features,
    generate_time_series_features,
)
from mci_gru.data.universes import UNIVERSES, get_universe_info

__all__ = [
    "UNIVERSES",
    "get_universe_info",
    "DataManager",
    "generate_time_series_features",
    "generate_graph_features",
    "compute_labels",
    "apply_rank_labels",
]
