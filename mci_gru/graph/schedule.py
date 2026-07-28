"""Pre-computed graph snapshot schedule for dynamic-graph mode."""

import bisect
from datetime import date, datetime

import torch


def _looks_canonical(text: str) -> bool:
    """Cheap ``YYYY-MM-DD`` shape check (length and separator positions only)."""
    return (
        len(text) == 10
        and text.isascii()
        and text[4] == "-"
        and text[7] == "-"
        and text[:4].isdigit()
        and text[5:7].isdigit()
        and text[8:10].isdigit()
    )


def require_canonical_date(value: object, field: str) -> str:
    """Return *value* unchanged after asserting it is a canonical ``YYYY-MM-DD`` string.

    Lookups compare dates lexicographically through :mod:`bisect`, which is only
    order-preserving for zero-padded ``YYYY-MM-DD``. ``"2020-1-5"`` sorts *after*
    ``"2020-07-01"`` and would silently resolve to a future snapshot, so anything
    that is not already canonical is rejected here rather than mis-resolved.
    """
    if not isinstance(value, str) or not _looks_canonical(value):
        raise ValueError(
            f"{field} must be a canonical YYYY-MM-DD date string, got {value!r}. "
            "Non-canonical dates resolve lexicographically to the wrong snapshot."
        )
    return value


def canonical_date(value: object, field: str = "date") -> str:
    """Normalise *value* to a canonical ``YYYY-MM-DD`` string.

    Accepts :class:`datetime.date` / :class:`datetime.datetime` (including
    ``pandas.Timestamp``), ``numpy.datetime64``, and ISO strings that carry a
    time suffix such as ``"2020-01-05 00:00:00"``. Rejects anything whose day
    boundary is ambiguous (``"2020-1-5"``, ``"01/05/2020"``, ``"20200105"``).

    This is the normalisation used once at the dataset boundary so that every
    per-batch schedule lookup only has to run the cheap canonical-shape check.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = value if isinstance(value, str) else str(value)
    head = text[:10]
    if _looks_canonical(head) and (len(text) == 10 or text[10] in (" ", "T")):
        try:
            date.fromisoformat(head)
        except ValueError as exc:
            raise ValueError(f"{field} is not a real calendar date: {value!r}") from exc
        return head
    raise ValueError(
        f"{field} must be an unambiguous YYYY-MM-DD date (optionally with a time "
        f"suffix), got {value!r}"
    )


class GraphSchedule:
    """Pre-computed graph snapshots indexed by their valid-from date.

    Each snapshot covers the period from its ``valid_from`` date until the
    next snapshot's ``valid_from`` (or the end of time for the last entry).
    Lookups use bisect for O(log n) per query.

    The third element of each snapshot tuple ("edge_weight") may be either a
    1-D tensor of shape ``(E,)`` (legacy scalar edge weight) or a 2-D tensor
    of shape ``(E, F)`` (multi-feature edges). The class is shape-agnostic;
    consumers must handle both shapes.

    **Readiness (mandatory warm-up).** A schedule is only usable for a sample
    axis when both hold:

    1. its first snapshot is in place on or before the first sample, so no
       sample is ever served a snapshot built after it; and
    2. that first snapshot was built from a full ``corr_lookback_days`` window
       of sessions, so the schedule does not open with an under-determined
       correlation graph.

    Both are asserted at construction from the evidence the caller supplies
    (``first_sample_date``, ``corr_lookback_days``,
    ``sessions_before_first_snapshot``); :attr:`is_ready` reports whether the
    full contract was verified. Dates before the first snapshot are refused at
    lookup rather than clamped to it.
    """

    def __init__(
        self,
        snapshots: list[tuple[str, torch.Tensor, torch.Tensor]],
        *,
        corr_lookback_days: int | None = None,
        sessions_before_first_snapshot: int | None = None,
        first_sample_date: object | None = None,
    ):
        if not snapshots:
            raise ValueError("GraphSchedule requires at least one snapshot")
        self._dates: list[str] = [
            require_canonical_date(s[0], "GraphSchedule snapshot valid_from") for s in snapshots
        ]
        if len(set(self._dates)) != len(self._dates):
            raise ValueError("GraphSchedule snapshot dates must be unique")
        if self._dates != sorted(self._dates):
            raise ValueError("GraphSchedule snapshot dates must be sorted")
        self._edge_indices: list[torch.Tensor] = [s[1] for s in snapshots]
        self._edge_weights: list[torch.Tensor] = [s[2] for s in snapshots]

        self._corr_lookback_days = corr_lookback_days
        self._sessions_before_first_snapshot = sessions_before_first_snapshot
        self._first_sample_date = (
            canonical_date(first_sample_date, "GraphSchedule first_sample_date")
            if first_sample_date is not None
            else None
        )
        self._ready = self._assert_ready()

    def _assert_ready(self) -> bool:
        """Raise when readiness is disproved; return whether it was fully verified."""
        start = self._dates[0]
        covers_samples = False
        if self._first_sample_date is not None:
            if self._first_sample_date < start:
                raise ValueError(
                    f"GraphSchedule is not ready: the first snapshot is valid from {start}, "
                    f"which is after the first sample {self._first_sample_date}. Samples in the "
                    "schedule's prehistory have no graph built from their own past; start the "
                    "schedule on or before the first sample."
                )
            covers_samples = True

        warmed_up = False
        if (
            self._corr_lookback_days is not None
            and self._sessions_before_first_snapshot is not None
        ):
            if self._sessions_before_first_snapshot < self._corr_lookback_days:
                raise ValueError(
                    f"GraphSchedule is not ready: the first snapshot ({start}) has only "
                    f"{self._sessions_before_first_snapshot} session(s) of history behind it, but "
                    f"corr_lookback_days={self._corr_lookback_days} sessions are required. Load at "
                    f"least {self._corr_lookback_days} sessions before {start}, or start the "
                    "schedule later (mandatory warm-up)."
                )
            warmed_up = True

        return covers_samples and warmed_up

    @property
    def is_ready(self) -> bool:
        """True when both warm-up and first-sample coverage were verified at construction."""
        return self._ready

    def _index_for_date(self, date: str) -> int:
        require_canonical_date(date, "GraphSchedule lookup date")
        idx = bisect.bisect_right(self._dates, date) - 1
        if idx < 0:
            raise ValueError(
                f"No graph snapshot is valid for {date}: the schedule starts {self._dates[0]}. "
                "A snapshot built after the sample must never serve it."
            )
        return idx

    def get_graph_for_date(self, date: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (edge_index, edge_attr) valid for *date*.

        ``edge_attr`` is shape ``(E,)`` in legacy mode and ``(E, F)`` in
        multi-feature mode (see class docstring).
        """
        idx = self._index_for_date(date)
        return self._edge_indices[idx], self._edge_weights[idx]

    def snapshot_valid_from_for_date(self, date: str) -> str:
        """Return the snapshot ``valid_from`` date string active on *date*."""
        return self._dates[self._index_for_date(date)]

    def get_initial_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first snapshot (used for graph_data.pt / static fallback)."""
        return self._edge_indices[0], self._edge_weights[0]

    @property
    def num_snapshots(self) -> int:
        return len(self._dates)

    @property
    def snapshot_dates(self) -> list[str]:
        return list(self._dates)
