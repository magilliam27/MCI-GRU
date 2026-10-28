"""The experiment protocol's fixed dates: the locked holdout and the window schedule.

PRD #369, approved by the owner on 2026-10-28. Each window trains on four calendar
years, validates on the next and tests on the year after; each is its own full run
with only these dates changed. W5 is the recipe as written (#367).

The 2026 signal dates from ``HOLDOUT_START`` to ``HOLDOUT_END`` are the holdout. No
run may train on, validate on or score them, and no label may reach into them,
except the one promotion run per candidate that #374 will unlock. Nothing after
``HOLDOUT_START`` is open either: the data after 2026-07-31 is the next holdout
(decision 3).

The dates are code, not a Hydra key, so moving them takes a reviewed change; a test
pins every one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

HOLDOUT_START = "2026-01-22"
HOLDOUT_END = "2026-07-24"

#: The guard runs before any panel is loaded, so it counts weekdays, and each exchange
#: closure pushes a label's real exit one weekday later. It allows three closures plus
#: one for every five sessions of label. Against the NYSE calendar, every signal date
#: from 2016 to the holdout and every label of 1 to 89 sessions crosses no more; the
#: tightest case is four closures inside a 15-session label (Christmas 2024 to Martin
#: Luther King Jr. Day 2025, with the 2025-01-09 day of mourning). Slack can only make
#: the guard refuse early, never late.
HOLIDAY_SLACK_WEEKDAYS = 3
HOLIDAY_SLACK_SESSIONS_PER_WEEKDAY = 5


class HoldoutError(Exception):
    """A run or a score would read the locked holdout.

    Not a ``ValueError``: code that falls back on an invalid config (the walk-forward
    window builder does) must not swallow this.
    """


@dataclass(frozen=True)
class Window:
    """One run's train, validation and test spans."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def spans(self) -> dict[str, str]:
        """The six dates under their ``data.*`` config keys."""
        return {
            "data.train_start": self.train_start,
            "data.train_end": self.train_end,
            "data.val_start": self.val_start,
            "data.val_end": self.val_end,
            "data.test_start": self.test_start,
            "data.test_end": self.test_end,
        }


def _window(name: str, first_train_session: str, test_year: int) -> Window:
    return Window(
        name=name,
        train_start=first_train_session,
        train_end=f"{test_year - 2}-12-31",
        val_start=f"{test_year - 1}-01-22",
        val_end=f"{test_year - 1}-12-31",
        test_start=f"{test_year}-01-22",
        test_end=f"{test_year}-12-31",
    )


#: Training starts on the first session of its first year; the four years end on
#: December 31 of the fourth. Validation and test start on January 22.
WINDOWS: dict[str, Window] = {
    "W1": _window("W1", "2016-01-04", 2021),
    "W2": _window("W2", "2017-01-03", 2022),
    "W3": _window("W3", "2018-01-02", 2023),
    "W4": _window("W4", "2019-01-02", 2024),
    "W5": _window("W5", "2020-01-02", 2025),
}

#: The window a promotion scores: the next one, its test span cut to the holdout.
HOLDOUT_WINDOW = Window(
    name="holdout",
    train_start="2021-01-04",
    train_end="2024-12-31",
    val_start="2025-01-22",
    val_end="2025-12-31",
    test_start=HOLDOUT_START,
    test_end=HOLDOUT_END,
)


def latest_label_end_bound(signal_date: str, label_t: int) -> str:
    """A date no earlier than the exit of a ``label_t``-session label on ``signal_date``."""
    if label_t < 1:
        raise ValueError(f"label_t must be at least 1, got {label_t}")
    slack = HOLIDAY_SLACK_WEEKDAYS + label_t // HOLIDAY_SLACK_SESSIONS_PER_WEEKDAY
    bound = np.busday_offset(np.datetime64(signal_date, "D"), label_t + slack, roll="forward")
    return str(bound)


def _iso_date(name: str, value: object) -> str:
    """``value`` as YYYY-MM-DD, so a date written another way cannot sort past the guard."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}={value!r} is not a date") from error
    if pd.isna(stamp):
        raise ValueError(f"{name}={value!r} is not a date")
    return stamp.strftime("%Y-%m-%d")


def require_outside_holdout(spans: Mapping[str, object], *, label_t: int, what: str) -> None:
    """Refuse ``spans`` if any date, or the label of the latest one, reaches the holdout.

    ``spans`` maps a name (a config key, say) to a date in any form ``pd.Timestamp``
    reads. ``label_t`` is the furthest session after a signal date that is read, the
    label's exit or a trade's. ``what`` names the caller in the error, so a refusal
    says which run or score was stopped.
    """
    if not spans:
        raise ValueError("require_outside_holdout needs at least one date")
    dates = {name: _iso_date(name, value) for name, value in spans.items()}
    inside = {name: value for name, value in dates.items() if value >= HOLDOUT_START}
    if inside:
        listed = ", ".join(f"{name}={value}" for name, value in sorted(inside.items()))
        raise HoldoutError(
            f"{what} reaches the locked holdout ({HOLDOUT_START} onward, PRD #369): {listed}. "
            "Only a promotion run may read it (#374)."
        )
    last_name, last = max(dates.items(), key=lambda item: item[1])
    bound = latest_label_end_bound(last, label_t)
    if bound >= HOLDOUT_START:
        raise HoldoutError(
            f"{what}: the {label_t}-session label of {last_name}={last} may exit as late as "
            f"{bound}, inside the locked holdout ({HOLDOUT_START} onward, PRD #369). End the "
            f"span earlier, so that its labels close before {HOLDOUT_START}."
        )


def require_predictions_outside_holdout(
    signal_dates: Iterable[object], *, label_t: int, what: str
) -> None:
    """Refuse saved predictions whose signal dates, or their labels, reach the holdout."""
    stamps = pd.to_datetime(pd.Series(list(signal_dates)), errors="raise").dropna()
    if stamps.empty:
        return
    require_outside_holdout(
        {"first signal date": stamps.min(), "last signal date": stamps.max()},
        label_t=label_t,
        what=what,
    )
