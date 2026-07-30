"""Guard the 110-name GICS top-10 universe config against silent misconfiguration.

Two properties here fail invisibly rather than loudly:

* ``pit_min_scoreable_stocks`` inherited from the S&P-scale default (450) would
  reject every session on a 110-name universe.
* A calendar-day split gap that looks generous can still leave fewer clear
  trading sessions than ``label_t``, so the last training label consumes the
  close of ``val_start`` itself. That is the defect recorded in #115.

Both are asserted against the config file rather than against a running
experiment, so they hold without data access.
"""

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "data" / "gics_top10_110.yaml"

# configs/config.yaml model.label_t
LABEL_T = 5

# Measured minimum admissible names per session across the train..test span.
MEASURED_MIN_ADMISSIBLE = 109


def _config() -> dict:
    raw = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    assert isinstance(raw, dict)
    return raw


def test_breadth_threshold_is_scaled_to_a_110_name_universe():
    cfg = _config()
    threshold = cfg["pit_min_scoreable_stocks"]

    assert threshold < MEASURED_MIN_ADMISSIBLE, (
        f"pit_min_scoreable_stocks={threshold} is at or above the measured "
        f"minimum admissible count ({MEASURED_MIN_ADMISSIBLE}), so "
        f"pit_breadth_policy would reject sessions that are genuinely fine."
    )
    assert threshold > MEASURED_MIN_ADMISSIBLE // 2, (
        f"pit_min_scoreable_stocks={threshold} is so low it would no longer "
        f"detect a real breadth collapse on a ~110 name universe."
    )


def test_pit_filtering_is_enabled_because_it_defines_the_universe():
    cfg = _config()

    assert cfg["use_pit_universe"] is True, (
        "Without PIT filtering this panel resolves to all 191 names that ever "
        "appear, which is survivorship-biased by construction."
    )
    assert cfg["pit_universe_csv"], "pit_universe_csv must be set for this universe"
    assert cfg["pit_breadth_policy"] == "error", (
        "Breadth must fail closed; 'warn' or 'off' would let a degraded universe through unnoticed."
    )


def test_split_gaps_clear_label_t_in_sessions_not_calendar_days():
    cfg = _config()
    boundaries = [
        (cfg["train_end"], cfg["val_start"], "train_end -> val_start"),
        (cfg["val_end"], cfg["test_start"], "val_end -> test_start"),
    ]

    for earlier, later, label in boundaries:
        gap_days = (_date(later) - _date(earlier)).days
        # A trading week is 5 sessions per 7 calendar days. Requiring
        # LABEL_T sessions of clearance therefore needs strictly more than
        # LABEL_T * 7 / 5 calendar days, plus a day for the boundary itself.
        min_days = int(LABEL_T * 7 / 5) + 1
        assert gap_days > min_days, (
            f"{label}: {gap_days} calendar days between {earlier} and {later} "
            f"cannot guarantee {LABEL_T} clear sessions. The last training "
            f"label would consume a close inside the next split. See #115."
        )


def _date(value: str):
    from datetime import date

    year, month, day = (int(part) for part in str(value).split("-"))
    return date(year, month, day)
