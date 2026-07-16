"""Deterministic matched nulls for saved-prediction selection research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


WITHIN_DATE_SCORE_PERMUTATION_V1 = "WITHIN_DATE_SCORE_PERMUTATION_V1"
ASSIGNMENT_DIGEST_STREAM_V1 = "SHA256_ASSIGNMENT_INDEX_STREAM_V1"
MINIMUM_NULL_DRAWS = 1000


@dataclass(frozen=True)
class DatedScoreOutcome:
    """One valid date's fixed score/outcome cross-section."""

    signal_dt: str
    instrument_keys: tuple[str, ...]
    scores: tuple[float, ...]
    outcomes: tuple[float, ...]

    def __post_init__(self) -> None:
        signal_dt = str(self.signal_dt)
        keys = tuple(str(key) for key in self.instrument_keys)
        scores = tuple(float(score) for score in self.scores)
        outcomes = tuple(float(outcome) for outcome in self.outcomes)
        object.__setattr__(self, "signal_dt", signal_dt)
        object.__setattr__(self, "instrument_keys", keys)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "outcomes", outcomes)

        if not signal_dt:
            raise ValueError("signal_dt must not be empty")
        if len(keys) < 2:
            raise ValueError("each date must contain at least two instruments")
        if len(keys) != len(scores) or len(keys) != len(outcomes):
            raise ValueError("instrument_keys, scores, and outcomes must have equal length")
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("instrument_keys must be nonempty and unique within a date")
        score_array = np.asarray(scores, dtype=np.float64)
        outcome_array = np.asarray(outcomes, dtype=np.float64)
        if not np.all(np.isfinite(score_array)) or not np.all(np.isfinite(outcome_array)):
            raise ValueError("scores and outcomes must contain only finite values")
        if np.ptp(score_array) == 0 or np.ptp(outcome_array) == 0:
            raise ValueError("scores and outcomes must each be nonconstant")


@dataclass(frozen=True)
class WithinDateScorePermutation:
    """One deterministic score reassignment, aligned to the caller's key order."""

    instrument_keys: tuple[str, ...]
    permuted_scores: tuple[float, ...]
    source_instrument_keys: tuple[str, ...]
    assignment_digest: str
    method: str = WITHIN_DATE_SCORE_PERMUTATION_V1


@dataclass(frozen=True)
class WithinDatePermutationNullResult:
    """Matched-null statistics and a digest of every score assignment."""

    method: str
    draw_count: int
    valid_draw_count: int
    null_seed: int
    top_k: int
    rank_ic_means: tuple[float, ...]
    top_k_spread_means: tuple[float, ...]
    assignment_digest: str
    assignment_digest_method: str = ASSIGNMENT_DIGEST_STREAM_V1


@dataclass(frozen=True)
class _PreparedCrossSection:
    signal_dt: str
    instrument_keys: tuple[str, ...]
    source_scores: np.ndarray
    source_score_ranks_centered: np.ndarray
    outcome_ranks_centered: np.ndarray
    rank_denominator: float
    outcomes: np.ndarray
    expected_mean: float


def permute_scores_within_date(
    *,
    instrument_keys: Sequence[str],
    scores: Sequence[float],
    signal_dt: str,
    null_seed: int,
    draw_id: int,
) -> WithinDateScorePermutation:
    """Reassign a date's score multiset using only seed, draw, date, and keys."""
    keys = tuple(str(key) for key in instrument_keys)
    score_values = tuple(float(score) for score in scores)
    signal_date = str(signal_dt)
    if len(keys) != len(score_values):
        raise ValueError("instrument_keys and scores must have equal length")
    if not keys:
        raise ValueError("instrument_keys must not be empty")
    if any(not key for key in keys):
        raise ValueError("instrument_keys must be nonempty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("instrument_keys must be unique within a date")
    if not np.all(np.isfinite(np.asarray(score_values, dtype=np.float64))):
        raise ValueError("scores must contain only finite values")
    if not isinstance(null_seed, int):
        raise TypeError("null_seed must be an int")
    if not isinstance(draw_id, int) or draw_id < 0:
        raise ValueError("draw_id must be a nonnegative int")

    source_rows = sorted(zip(keys, score_values, strict=True), key=lambda item: item[0])
    destination_keys = tuple(
        keys[index]
        for index in _destination_order(
            keys,
            signal_dt=signal_date,
            null_seed=null_seed,
            draw_id=draw_id,
        )
    )
    source_by_destination = {
        destination_key: source_key
        for destination_key, (source_key, _) in zip(
            destination_keys,
            source_rows,
            strict=True,
        )
    }
    score_by_source = dict(source_rows)
    permuted_scores = tuple(score_by_source[source_by_destination[key]] for key in keys)
    source_keys = tuple(source_by_destination[key] for key in keys)
    assignment_digest = _assignment_digest(
        signal_dt=signal_date,
        null_seed=null_seed,
        draw_id=draw_id,
        source_by_destination=source_by_destination,
    )
    return WithinDateScorePermutation(
        instrument_keys=keys,
        permuted_scores=permuted_scores,
        source_instrument_keys=source_keys,
        assignment_digest=assignment_digest,
    )


def run_within_date_permutation_null(
    cross_sections: Sequence[DatedScoreOutcome],
    *,
    n_draws: int,
    null_seed: int,
    top_k: int,
) -> WithinDatePermutationNullResult:
    """Run the matched within-date score null over complete valid dates."""
    dated_sections = tuple(sorted(cross_sections, key=lambda item: item.signal_dt))
    if not dated_sections:
        raise ValueError("cross_sections must not be empty")
    if len({item.signal_dt for item in dated_sections}) != len(dated_sections):
        raise ValueError("signal_dt must be unique across cross_sections")
    if not isinstance(n_draws, int) or n_draws < MINIMUM_NULL_DRAWS:
        raise ValueError(f"n_draws must be at least {MINIMUM_NULL_DRAWS}")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive int")
    if any(top_k > len(item.instrument_keys) for item in dated_sections):
        raise ValueError("top_k cannot exceed any date's cross-sectional breadth")

    dates = tuple(_prepare_cross_section(item) for item in dated_sections)

    rank_ic_means: list[float] = []
    top_k_spread_means: list[float] = []
    digest = hashlib.sha256()
    digest.update(ASSIGNMENT_DIGEST_STREAM_V1.encode("ascii"))
    for item in dates:
        digest.update(item.signal_dt.encode("utf-8"))
        digest.update(b"\0")
        for key in item.instrument_keys:
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
    for draw_id in range(n_draws):
        daily_rank_ic: list[float] = []
        daily_top_k_spread: list[float] = []
        for item in dates:
            destination_order = _destination_order(
                item.instrument_keys,
                signal_dt=item.signal_dt,
                null_seed=null_seed,
                draw_id=draw_id,
            )
            digest.update(destination_order.astype("<u4", copy=False).tobytes())
            permuted_scores = np.empty_like(item.source_scores)
            permuted_scores[destination_order] = item.source_scores
            permuted_ranks = np.empty_like(item.source_score_ranks_centered)
            permuted_ranks[destination_order] = item.source_score_ranks_centered
            daily_rank_ic.append(
                float(np.dot(permuted_ranks, item.outcome_ranks_centered) / item.rank_denominator)
            )
            daily_top_k_spread.append(
                _top_k_spread_prepared(
                    permuted_scores,
                    item.outcomes,
                    item.expected_mean,
                    top_k=top_k,
                )
            )
        rank_ic_means.append(float(np.mean(daily_rank_ic)))
        top_k_spread_means.append(float(np.mean(daily_top_k_spread)))

    return WithinDatePermutationNullResult(
        method=WITHIN_DATE_SCORE_PERMUTATION_V1,
        draw_count=n_draws,
        valid_draw_count=n_draws,
        null_seed=null_seed,
        top_k=top_k,
        rank_ic_means=tuple(rank_ic_means),
        top_k_spread_means=tuple(top_k_spread_means),
        assignment_digest=digest.hexdigest(),
    )


def _prepare_cross_section(item: DatedScoreOutcome) -> _PreparedCrossSection:
    order = sorted(
        range(len(item.instrument_keys)),
        key=lambda index: item.instrument_keys[index],
    )
    keys = tuple(item.instrument_keys[index] for index in order)
    source_scores = np.asarray([item.scores[index] for index in order], dtype=np.float64)
    outcomes = np.asarray([item.outcomes[index] for index in order], dtype=np.float64)
    source_ranks = _average_ranks(source_scores)
    outcome_ranks = _average_ranks(outcomes)
    source_centered = source_ranks - np.mean(source_ranks)
    outcome_centered = outcome_ranks - np.mean(outcome_ranks)
    denominator = float(
        np.sqrt(
            np.dot(source_centered, source_centered) * np.dot(outcome_centered, outcome_centered)
        )
    )
    return _PreparedCrossSection(
        signal_dt=item.signal_dt,
        instrument_keys=keys,
        source_scores=source_scores,
        source_score_ranks_centered=source_centered,
        outcome_ranks_centered=outcome_centered,
        rank_denominator=denominator,
        outcomes=outcomes,
        expected_mean=float(np.mean(outcomes)),
    )


def _destination_order(
    instrument_keys: tuple[str, ...],
    *,
    signal_dt: str,
    null_seed: int,
    draw_id: int,
) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(instrument_keys)),
            key=lambda index: (
                _destination_hash(
                    null_seed,
                    draw_id,
                    signal_dt,
                    instrument_keys[index],
                ),
                instrument_keys[index],
            ),
        ),
        dtype=np.int64,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _top_k_spread_prepared(
    scores: np.ndarray,
    outcomes: np.ndarray,
    expected_mean: float,
    *,
    top_k: int,
) -> float:
    selected = np.lexsort((np.arange(scores.size), -scores))[:top_k]
    return float(np.mean(outcomes[selected]) - expected_mean)


def _destination_hash(
    null_seed: int,
    draw_id: int,
    signal_dt: str,
    instrument_key: str,
) -> bytes:
    payload = f"{null_seed}|{draw_id}|{signal_dt}|{instrument_key}".encode()
    return hashlib.sha256(payload).digest()


def _assignment_digest(
    *,
    signal_dt: str,
    null_seed: int,
    draw_id: int,
    source_by_destination: dict[str, str],
) -> str:
    payload = {
        "assignments": [
            [destination, source_by_destination[destination]]
            for destination in sorted(source_by_destination)
        ],
        "draw_id": draw_id,
        "method": WITHIN_DATE_SCORE_PERMUTATION_V1,
        "null_seed": null_seed,
        "signal_dt": signal_dt,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
