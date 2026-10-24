"""Validation labels never read test-period closes (#342, dates finding 5).

The calendar-day check leaves walk-forward windows six calendar days between
validation and test, which can hold three sessions, so the last validation labels
(exit at session ``D+label_t``) landed on test closes and steered early stopping.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from mci_gru.config import ExperimentConfig
from mci_gru.data.preprocessing import resolve_label_endpoints
from mci_gru.pipeline import _embargo_validation_sessions
from mci_gru.walkforward import generate_walkforward_configs

LABEL_T = 5


def _panel(start: str, end: str) -> pd.DataFrame:
    sessions = pd.bdate_range(start, end).strftime("%Y-%m-%d")
    return pd.DataFrame(
        [{"dt": dt, "kdcode": kdcode, "close": 100.0} for dt in sessions for kdcode in "AB"]
    )


def _sessions(panel: pd.DataFrame, start: str, end: str) -> list[str]:
    return sorted(panel.loc[(panel["dt"] >= start) & (panel["dt"] <= end), "dt"].unique())


def test_val_tail_whose_labels_exit_in_test_is_purged() -> None:
    panel = _panel("2024-12-02", "2025-02-28")
    val_dates = _sessions(panel, "2024-12-02", "2025-01-10")
    kept = _embargo_validation_sessions(val_dates, panel, ["A", "B"], "2025-01-16", LABEL_T)

    sessions = sorted(panel["dt"].unique())
    _, exits = resolve_label_endpoints(sessions, kept, LABEL_T)
    assert all(exit_date < "2025-01-16" for exit_date in exits)
    assert kept == val_dates[: len(kept)] and len(kept) < len(val_dates)
    # 2025-01-08 exits 2025-01-15, the last session before test; 01-09 exits on 01-16.
    assert kept[-1] == "2025-01-08"


def test_a_wide_gap_purges_nothing() -> None:
    panel = _panel("2024-12-02", "2025-02-28")
    val_dates = _sessions(panel, "2024-12-02", "2024-12-31")
    assert _embargo_validation_sessions(val_dates, panel, ["A", "B"], "2025-01-22", LABEL_T) == (
        val_dates
    )


def test_no_walk_forward_window_leaks_test_closes_into_validation() -> None:
    """r6: 9 of 11 default windows leaked before the purge."""
    base = ExperimentConfig()
    base = replace(
        base,
        data=replace(
            base.data, train_start="2016-01-04", test_end="2025-12-31", use_pit_universe=False
        ),
        training=replace(
            base.training, walkforward=replace(base.training.walkforward, enabled=True)
        ),
    )
    panel = _panel("2015-01-01", "2026-03-31")
    sessions = sorted(panel["dt"].unique())
    leaks_before = leaks_after = 0
    windows = generate_walkforward_configs(base)
    for window in windows:
        data = window.data
        val_dates = _sessions(panel, data.val_start, data.val_end)
        kept = _embargo_validation_sessions(
            val_dates, panel, ["A", "B"], data.test_start, window.model.label_t
        )
        for dates, counter in ((val_dates, "before"), (kept, "after")):
            _, exits = resolve_label_endpoints(sessions, dates, window.model.label_t)
            leaked = any(e is not None and e >= data.test_start for e in exits)
            if counter == "before":
                leaks_before += leaked
            else:
                leaks_after += leaked
    assert leaks_before >= len(windows) // 2  # the defect is present on this calendar
    assert leaks_after == 0
