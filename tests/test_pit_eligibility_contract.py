"""Dated PIT eligibility and fixed-session label contract (#225).

Exercises Mac's 2026-10-03 rulings on #225 through public ``prepare_data`` with tiny
file fixtures:

1. The forecast for date D is made at 20:00 America/New_York.
2. A cessation leaves daily eligibility only once it is both effective and known by that
   clock. Date-only knowledge counts from the clock on the next session. A cessation with
   no dated evidence stops the run, naming the stock. Acquisition time is never used.
3. Training and evaluation use observable labels only; every prediction is kept and an
   unobservable one is reported with its reason.
4. Labels run from the close of session D+1 to the close of session D+5 on the panel's own
   trading dates, with no fill, and the embargo uses the same resolver.

The two-name fixture is the Windows packet's (#225 comment 5964728627), adjusted to those
rulings; see ``docs/agents/pit-eligibility-label-contract.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)
from mci_gru.data.pit import (
    CessationEvidence,
    PITEligibilityError,
    PredictionClock,
    build_pit_masks,
    cessation_exclusion_mask,
    label_available_mask,
    load_cessation_events,
    parse_cessation_events,
    pit_split_report,
    resolve_cessation_events,
)
from mci_gru.data.preprocessing import (
    assert_training_labels_respect_embargo,
    compute_labels,
    resolve_labels,
)
from mci_gru.pipeline import prepare_data

# 29 declared synthetic sessions; no exchange-calendar package is consulted.
SESSIONS = [
    *(
        f"2024-01-{d:02d}"
        for d in (2, 3, 4, 5, 8, 9, 10, 11, 12, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31)
    ),
    *(f"2024-02-{d:02d}" for d in (1, 2, 5, 6, 7, 8, 9, 12)),
]
D = "2024-02-05"  # the focal prediction date, session index 23
LAST_STOP_SESSION = D
# 20:00 America/New_York on 2024-02-05 (EST) is 01:00 UTC on 2024-02-06.
T_UTC = pd.Timestamp("2024-02-06T01:00:00Z")
BASE_FEATURES = ["close", "open", "high", "low", "volume", "turnover"]


def _write_panel(path) -> pd.DataFrame:
    rows = []
    for i, session in enumerate(SESSIONS):
        for kdcode, base in (("KEEP", 100.0), ("STOP", 200.0)):
            if kdcode == "STOP" and session > LAST_STOP_SESSION:
                continue
            close = base + i
            volume = 1000.0 + i
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": session,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * close,
                }
            )
    panel = pd.DataFrame(rows)
    panel.to_csv(path, index=False)
    return panel


def _write_events(path, **overrides: str) -> None:
    event = {
        "event_id": "STOP-cessation-1",
        "kdcode": "STOP",
        "effective_at": "2024-02-05T22:00:00Z",
        "known_from": "2024-02-05T21:30:00Z",
        "acquired_at": "2026-09-20T00:00:00Z",
        "evidence": "synthetic fixture",
    }
    event.update(overrides)
    pd.DataFrame([event]).to_csv(path, index=False)


class _PassThroughFeatureEngineer:
    def transform(self, df, *_args):
        return df

    def get_feature_columns(self) -> list[str]:
        return BASE_FEATURES


def _config(tmp_path, events_path=None) -> ExperimentConfig:
    pit_path = tmp_path / "pit.csv"
    pd.DataFrame(
        [
            {"kdcode": "KEEP", "valid_from": "2024-01-01", "valid_to": "2024-02-29"},
            {"kdcode": "STOP", "valid_from": "2024-01-01", "valid_to": "2024-02-29"},
        ]
    ).to_csv(pit_path, index=False)
    return ExperimentConfig(
        data=DataConfig(
            source="csv",
            filename=str(tmp_path / "panel.csv"),
            train_start="2024-01-02",
            train_end="2024-01-12",
            val_start="2024-01-22",
            val_end="2024-01-22",
            test_start=D,
            test_end=D,
            use_pit_universe=True,
            pit_universe_csv=str(pit_path),
            pit_universe_mode="masked_panel",
            pit_min_scoreable_stocks=0,
            pit_cessation_events_csv=None if events_path is None else str(events_path),
        ),
        features=FeatureConfig(
            base_features=BASE_FEATURES,
            include_momentum=False,
            include_weekly_momentum=False,
        ),
        graph=GraphConfig(judge_value=0.9999, use_multi_feature_edges=False),
        model=ModelConfig(his_t=2, label_t=5),
        training=TrainingConfig(num_epochs=1, num_models=1, label_type="returns"),
        tracking=TrackingConfig(enabled=False),
    )


def _prepare(tmp_path, **event_overrides: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_panel(tmp_path / "panel.csv")
    events_path = None
    if event_overrides.pop("_declared", "yes") == "yes":
        events_path = tmp_path / "events.csv"
        _write_events(events_path, **event_overrides)
    return prepare_data(_config(tmp_path, events_path), _PassThroughFeatureEngineer())


def _focal_counts(data: dict) -> dict:
    (row,) = data["pit_eligibility"]["splits"]["test"]["daily"]
    assert row["date"] == D
    return row


# Case -> (event overrides, STOP excluded at D?)
CASES = {
    # Effective 17:00 and known 16:30 New York, both before the 20:00 forecast.
    "K_known_and_effective": ({}, True),
    # Known 21:00 New York, an hour after the forecast. Acquired long before: ignored.
    "L_known_after_clock": (
        {"known_from": "2024-02-06T02:00:00Z", "acquired_at": "2024-01-02T00:00:00Z"},
        False,
    ),
    # Date-only Friday 2024-02-02 counts from 20:00 New York on Monday 2024-02-05: at T.
    "D1_date_only_known_by_D": ({"known_from": "2024-02-02"}, True),
    # Date-only 2024-02-05 counts from 20:00 New York on 2024-02-06: after T.
    "D2_date_only_known_after_D": ({"known_from": "2024-02-05"}, False),
    # Known in time, but not effective until the next day.
    "F_not_yet_effective": ({"effective_at": "2024-02-06T22:00:00Z"}, False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_prepare_data_cessation_requires_effective_and_known_time(tmp_path, case: str) -> None:
    overrides, excluded = CASES[case]
    data = _prepare(tmp_path, **overrides)

    # Selection, the union axis and the original rows never change.
    assert data["kdcode_list"] == ["KEEP", "STOP"]
    assert len(data["df"]) == 53
    assert data["stock_features_test"].shape[1] == 2
    assert data["test_active_member_mask"].tolist() == [[True, True]]
    assert data["test_feature_ready_mask"].tolist() == [[True, True]]

    # Only the focal date can change; earlier samples keep both names.
    assert data["train_dates"] == ["2024-01-04", "2024-01-05"]
    assert data["train_tradable_mask"].all() and data["val_tradable_mask"].all()
    assert data["test_tradable_mask"].tolist() == [[True, not excluded]]

    # KEEP's fixed-session label at D: close of 2024-02-12 (128) over 2024-02-06 (124).
    assert data["test_labels"][0, 0] == pytest.approx(128 / 124 - 1)
    # STOP has no close after D: unobservable either way, never filled.
    assert np.isnan(data["test_labels"][0, 1])
    assert data["test_loss_mask"].tolist() == [[True, False]]

    counts = _focal_counts(data)
    assert counts["selected"] == 2
    assert counts["cessation_excluded"] == int(excluded)
    assert counts["eligible"] == counts["predictions"] == 2 - int(excluded)
    assert counts["label_observable"] == 1
    assert counts["label_omitted"] == int(not excluded)

    omitted = data["pit_eligibility"]["splits"]["test"]["omitted_labels"]
    expected_omitted = (
        []
        if excluded
        else [
            {
                "date": D,
                "kdcode": "STOP",
                "entry_date": "2024-02-06",
                "exit_date": "2024-02-12",
                "reason": "entry_and_exit_close_missing",
            }
        ]
    )
    assert omitted == expected_omitted


def test_later_knowledge_changes_only_the_cessation_decision(tmp_path) -> None:
    """No-lookahead canary: moving known_from past T changes nothing but eligibility."""
    known = _prepare(tmp_path / "k", **CASES["K_known_and_effective"][0])
    late = _prepare(tmp_path / "l", **CASES["L_known_after_clock"][0])

    for key in ("stock_features_test", "x_graph_test", "test_labels", "train_labels"):
        np.testing.assert_array_equal(known[key], late[key])
    np.testing.assert_array_equal(known["train_tradable_mask"], late["train_tradable_mask"])
    assert known["test_tradable_mask"].tolist() != late["test_tradable_mask"].tolist()


def test_cessation_without_dated_evidence_rejects_the_run_naming_the_stock(tmp_path) -> None:
    with pytest.raises(PITEligibilityError) as caught:
        _prepare(tmp_path, known_from="")

    error = caught.value
    assert error.code == "undated_cessation"
    assert error.kdcodes == ("STOP",)
    assert "STOP" in str(error)
    assert error.role == "data.pit_cessation_events_csv"
    assert error.stage == "pit_eligibility"
    assert error.source == str(tmp_path / "events.csv")


def test_acquisition_time_is_never_availability(tmp_path) -> None:
    """A blank known_from is undated even when the row says when it was acquired."""
    with pytest.raises(PITEligibilityError, match="without dated evidence"):
        _prepare(tmp_path, known_from="", acquired_at="2024-01-02T00:00:00Z")


@pytest.mark.parametrize("value", ["2024-02-05 21:30", "Feb 5 2024", "2024-02-31"])
def test_malformed_or_naive_timestamp_stops_rather_than_counting_as_unknown(
    tmp_path, value: str
) -> None:
    with pytest.raises(PITEligibilityError) as caught:
        _prepare(tmp_path, known_from=value)
    assert caught.value.code == "malformed_timestamp"
    assert caught.value.kdcodes == ("STOP",)


def test_missing_declared_event_file_stops(tmp_path) -> None:
    _write_panel(tmp_path / "panel.csv")
    with pytest.raises(FileNotFoundError):
        prepare_data(
            _config(tmp_path, tmp_path / "absent.csv"),
            _PassThroughFeatureEngineer(),
        )


def test_undeclared_event_file_excludes_nothing_and_says_so(tmp_path) -> None:
    data = _prepare(tmp_path, _declared="no")

    assert data["test_tradable_mask"].tolist() == [[True, True]]
    fragment = data["pit_eligibility"]
    assert fragment["schema"] == "mci_gru.pit_eligibility.v1"
    assert fragment["cessation_events"] == {
        "declared": False,
        "configured_path": None,
        "events": [],
    }
    assert fragment["prediction_clock"] == {"time": "20:00", "timezone": "America/New_York"}
    assert fragment["label_endpoints"]["entry_session_offset"] == 1
    assert fragment["label_endpoints"]["exit_session_offset"] == 5
    assert fragment["label_endpoints"]["close_to_close_intervals"] == 4
    coverage = fragment["splits"]["test"]["label_coverage"]
    assert (coverage["numerator"], coverage["denominator"], coverage["value"]) == (
        "label_observable",
        "predictions",
        0.5,
    )


def test_declared_event_file_is_read_through_the_input_carrier(tmp_path) -> None:
    data = _prepare(tmp_path)

    inputs = data["input_observations"].data_inputs()
    assert inputs["data.pit_cessation_events_csv"]["configured_path"] == str(
        tmp_path / "events.csv"
    )
    (event,) = data["pit_eligibility"]["cessation_events"]["events"]
    assert event["known_utc"] == "2024-02-05T21:30:00Z"
    assert event["acquired_at"] == "2026-09-20T00:00:00Z"


def test_date_only_known_from_resolves_to_the_clock_on_the_next_session() -> None:
    """Never midnight: 2024-02-02 resolves to 20:00 New York on Monday 2024-02-05."""
    frame = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "kdcode": "STOP",
                "effective_at": "2024-02-01",
                "known_from": "2024-02-02",
            }
        ]
    )
    evidence = parse_cessation_events(frame, source=None)
    clock = PredictionClock()
    (resolved,) = resolve_cessation_events(evidence, SESSIONS, clock)
    assert resolved.known_utc == T_UTC
    assert resolved.effective_utc == pd.Timestamp("2024-02-01T05:00:00Z")

    mask = cessation_exclusion_mask([resolved], ["KEEP", "STOP"], ["2024-02-02", D], clock)
    assert mask.tolist() == [[False, False], [False, True]]


def test_no_events_file_means_no_evidence() -> None:
    assert load_cessation_events(None) == CessationEvidence(configured_path=None, events=())


# ── fixed-session endpoint discriminator (ruling 4) ──────────────────────

GAP_SESSIONS = ["2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09"] + [
    "2024-02-12",
    "2024-02-13",
]


def _gap_panel(gap_closes: dict[str, float]) -> pd.DataFrame:
    rows = [{"kdcode": "CTRL", "dt": s, "close": 50.0 + i} for i, s in enumerate(GAP_SESSIONS)]
    rows += [{"kdcode": "GAP", "dt": s, "close": c} for s, c in gap_closes.items()]
    return pd.DataFrame(rows)


GAP_CLOSES = {
    "2024-02-05": 99.0,
    "2024-02-06": 100.0,
    # 2024-02-07 missing
    "2024-02-08": 102.0,
    "2024-02-09": 103.0,
    "2024-02-12": 104.0,
    "2024-02-13": 105.0,
}


def test_gap_label_uses_fixed_sessions_and_all_three_consumers_agree() -> None:
    """Row shifts read 2024-02-13 (5%); fixed sessions read 2024-02-12 (4%)."""
    panel = _gap_panel(GAP_CLOSES)
    stocks = ["CTRL", "GAP"]

    labels = compute_labels(panel, stocks, [D], 5, fill_missing=False)
    assert labels[0, 1] == pytest.approx(104 / 100 - 1)
    assert label_available_mask(panel, stocks, [D], 5).tolist() == [[True, True]]

    # The embargo checks the same exit: 2024-02-12 is before a 2024-02-13 val_start,
    # where a row-shift exit (2024-02-13) would be a violation.
    summary = assert_training_labels_respect_embargo(panel, stocks, [D], "2024-02-13", 5)
    assert summary["last_outcome_date"] == "2024-02-12"


def test_missing_entry_close_makes_the_label_unobservable_without_substitution() -> None:
    closes = {k: v for k, v in GAP_CLOSES.items() if k != "2024-02-06"}
    panel = _gap_panel(closes)
    stocks = ["CTRL", "GAP"]

    labels = compute_labels(panel, stocks, [D], 5, fill_missing=False)
    assert np.isfinite(labels[0, 0]) and np.isnan(labels[0, 1])
    assert label_available_mask(panel, stocks, [D], 5).tolist() == [[True, False]]


@pytest.mark.parametrize(
    ("dropped", "reason"),
    [
        ("2024-02-06", "entry_close_missing"),
        ("2024-02-12", "exit_close_missing"),
    ],
)
def test_omitted_label_reports_which_endpoint_is_missing(dropped: str, reason: str) -> None:
    closes = {"2024-02-02": 98.0, **{k: v for k, v in GAP_CLOSES.items() if k != dropped}}
    panel = _gap_panel(closes)
    panel = pd.concat(
        [panel, pd.DataFrame([{"kdcode": "CTRL", "dt": "2024-02-02", "close": 49.0}])]
    )
    stocks = ["CTRL", "GAP"]
    intervals = pd.DataFrame(
        [{"kdcode": k, "valid_from": "2024-02-01", "valid_to": "2024-02-29"} for k in stocks]
    )
    masks = build_pit_masks(panel, panel, stocks, [D], 1, 5, intervals)
    report = pit_split_report([D], stocks, masks, resolve_labels(panel, stocks, [D], 5))

    assert report["omitted_labels"] == [
        {
            "date": D,
            "kdcode": "GAP",
            "entry_date": "2024-02-06",
            "exit_date": "2024-02-12",
            "reason": reason,
        }
    ]
    assert report["totals"]["predictions"] == 2
    assert report["totals"]["label_omitted"] == 1


def test_a_missing_price_masks_only_that_session_and_is_counted_per_stock() -> None:
    """#223 ruling 13: a genuine gap is masked per session, never imputed into a prediction."""
    panel = _gap_panel(GAP_CLOSES)
    stocks = ["CTRL", "GAP"]
    intervals = pd.DataFrame(
        [{"kdcode": k, "valid_from": "2024-02-01", "valid_to": "2024-02-29"} for k in stocks]
    )
    dates = ["2024-02-06", "2024-02-07"]
    masks = build_pit_masks(panel, panel, stocks, dates, 1, 5, intervals)

    # GAP is feature-ready on 2024-02-07 (it has the 2024-02-06 bar) but has no close.
    assert masks.feature_ready.tolist() == [[True, True], [True, True]]
    assert masks.tradable.tolist() == [[True, True], [True, False]]
    report = pit_split_report(dates, stocks, masks, resolve_labels(panel, stocks, dates, 5))
    assert [row["price_gap"] for row in report["daily"]] == [0, 1]
    assert report["price_gaps_by_stock"] == {"GAP": 1}
