import pandas as pd

from mci_gru.data.pit import (
    PITKnowledgeClass,
    classify_pit_knowledge_as_of,
    normalise_pit_intervals,
)


def test_optional_known_from_survives_normalization_and_is_checked_at_signal_close() -> None:
    intervals = pd.DataFrame(
        [
            {
                "kdcode": "AAA",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-31",
                "known_from": "2020-01-02 15:30:00-05:00",
            }
        ]
    )

    normalized = normalise_pit_intervals(intervals)

    assert normalized.columns.tolist() == [
        "kdcode",
        "valid_from",
        "valid_to",
        "known_from",
    ]
    assert normalized.loc[0, "known_from"] == "2020-01-02T20:30:00Z"
    assert (
        classify_pit_knowledge_as_of(
            normalized,
            signal_close="2020-01-02 16:00:00-05:00",
        )
        is PITKnowledgeClass.KNOWN_AS_OF
    )


def test_legacy_intervals_without_known_from_remain_effective_only() -> None:
    intervals = pd.DataFrame(
        [
            {
                "kdcode": "AAA",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-31",
            }
        ]
    )

    normalized = normalise_pit_intervals(intervals)

    assert normalized.columns.tolist() == ["kdcode", "valid_from", "valid_to"]
    assert (
        classify_pit_knowledge_as_of(
            normalized,
            signal_close="2020-01-02 16:00:00-05:00",
        )
        is PITKnowledgeClass.EFFECTIVE_ONLY
    )


def test_future_known_from_is_not_treated_as_known_at_signal_close() -> None:
    intervals = pd.DataFrame(
        [
            {
                "kdcode": "AAA",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-31",
                "known_from": "2020-01-02 16:01:00-05:00",
            }
        ]
    )

    assert (
        classify_pit_knowledge_as_of(
            intervals,
            signal_close="2020-01-02 16:00:00-05:00",
        )
        is PITKnowledgeClass.UNKNOWN
    )


def test_naive_known_from_uses_the_declared_timezone() -> None:
    intervals = pd.DataFrame(
        [
            {
                "kdcode": "AAA",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-31",
                "known_from": "2020-01-02 16:30:00",
            }
        ]
    )

    normalized = normalise_pit_intervals(
        intervals,
        known_from_timezone="America/New_York",
    )

    assert normalized.loc[0, "known_from"] == "2020-01-02T21:30:00Z"
    assert (
        classify_pit_knowledge_as_of(
            intervals,
            signal_close="2020-01-02 16:00:00-05:00",
            known_from_timezone="America/New_York",
        )
        is PITKnowledgeClass.UNKNOWN
    )


def test_known_from_is_never_inferred_from_valid_from() -> None:
    intervals = pd.DataFrame(
        [
            {
                "kdcode": "AAA",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-31",
            }
        ]
    )

    normalized = normalise_pit_intervals(intervals)

    assert "known_from" not in normalized.columns


def test_missing_supplied_known_from_is_unknown_not_effective_only() -> None:
    intervals = pd.DataFrame(
        [
            {
                "kdcode": "AAA",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-31",
                "known_from": None,
            }
        ]
    )

    assert (
        classify_pit_knowledge_as_of(
            intervals,
            signal_close="2020-01-02 16:00:00-05:00",
        )
        is PITKnowledgeClass.UNKNOWN
    )
