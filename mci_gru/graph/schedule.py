"""Pre-computed graph snapshot schedule for dynamic-graph mode."""

import bisect

import torch


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
