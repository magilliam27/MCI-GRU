"""
Data loading and management for MCI-GRU experiments.

Modules:
- lseg_loader: LSEG/Refinitiv data fetching
- data_manager: Unified data interface
- universes: Stock universe definitions
"""

from mci_gru.data.universes import UNIVERSES, get_universe_info
from mci_gru.data.data_manager import DataManager

__all__ = [
    "UNIVERSES",
    "get_universe_info",
    "DataManager",
]
