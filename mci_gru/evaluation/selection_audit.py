"""Saved-prediction model-selection audit helpers."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mci_gru.data.pit import (
    PITKnowledgeClass,
    classify_pit_knowledge_as_of,
    normalise_pit_intervals,
)
from mci_gru.evaluation.artifacts import (
    build_research_study_id,
    canonical_json_bytes,
    write_json_artifact,
    write_selection_research_bundle,
)
from mci_gru.evaluation.portfolio import top_k_returns
from mci_gru.evaluation.prediction_report import (
    align_prediction_comparison,
    load_prediction_files,
    realized_returns_from_market_data,
)
from mci_gru.evaluation.selection_nulls import (
    DatedScoreOutcome,
    run_within_date_permutation_null,
)
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    dated_daily_ic,
    empirical_one_sided_p_value,
    moving_block_bootstrap_ci,
    moving_block_mean_ci,
    newey_west_mean_inference,
    newey_west_std,
)
from mci_gru.evaluation.trial_ledger import validate_trial_family

MIN_RESEARCH_NULL_DRAWS = 1_000
MIN_PRELIMINARY_DATES = 60
RESEARCH_NULL_FAMILY = "WITHIN_DATE_SCORE_PERMUTATION_V1"
MCI_GRU_FORWARD_CLOSE_V1 = "MCI_GRU_FORWARD_CLOSE_V1"
DATE_EVIDENCE_COLUMNS = (
    "signal_dt",
    "label_start_dt",
    "label_end_dt",
    "PIT_active_count",
    "expected_scorable_count",
    "prediction_count",
    "finite_score_count",
    "complete_outcome_count",
    "daily_rank_ic",
    "top_k_label_return",
    "expected_set_label_return",
    "top_k_spread",
    "date_status",
    "reason_codes",
)


@dataclass(frozen=True)
class SelectionResearchProtocol:
    """Frozen runtime inputs and statistical choices for one prediction set."""

    research_semantics_version: str
    study_name: str
    trial_family_id: str
    predictions_dir: str | Path
    market_data_path: str | Path
    pit_universe_path: str | Path | None
    expected_scorable_path: str | Path | None
    calendar_path: str | Path
    label_horizon: int
    test_start: str
    test_end: str
    data_as_of: str
    top_k: int
    price_basis: str
    price_adjustment_provenance: str
    null_draws: int
    null_seed: int
    hac_lag: int
    bootstrap_block_length: int
    bootstrap_resamples: int
    bootstrap_seed: int
    ci_level: float
    alpha: float
    null_family: str = RESEARCH_NULL_FAMILY
    trial_ledger_path: str | Path | None = None
    trial_ledger_complete: bool = False
    expected_trial_ids: tuple[str, ...] = ()
    oos_previously_accessed: bool = True
    exchange_timezone: str = "America/New_York"
    signal_close_local_time: str = "16:00:00"
    calendar_source: str = "DECLARED_SESSION_CSV"
    pit_known_from_timezone: str = "UTC"
    prediction_source_run_id: str | None = None
    prediction_ensemble_rule: str = "SAVED_PREDICTION_SET"
    prediction_ensemble_member_count: int | None = None
    prediction_seed_id: str | None = None
    prediction_source_code_commit: str | None = None
    prediction_label_contract: str = "UNKNOWN"
    prediction_label_horizon: int | None = None

    def __post_init__(self) -> None:
        if not self.research_semantics_version.strip():
            raise ValueError("research_semantics_version must not be empty")
        if not self.study_name.strip() or not self.trial_family_id.strip():
            raise ValueError("study_name and trial_family_id must not be empty")
        if self.label_horizon <= 0:
            raise ValueError("label_horizon must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.price_adjustment_provenance.strip():
            raise ValueError("price_adjustment_provenance must not be empty")
        if not self.prediction_ensemble_rule.strip():
            raise ValueError("prediction_ensemble_rule must not be empty")
        if (
            self.prediction_ensemble_member_count is not None
            and self.prediction_ensemble_member_count <= 0
        ):
            raise ValueError("prediction_ensemble_member_count must be positive")
        if self.prediction_label_horizon is not None and self.prediction_label_horizon <= 0:
            raise ValueError("prediction_label_horizon must be positive")
        if self.null_family != RESEARCH_NULL_FAMILY:
            raise ValueError(f"null_family must be {RESEARCH_NULL_FAMILY}")
        if self.null_draws < MIN_RESEARCH_NULL_DRAWS:
            raise ValueError(f"null_draws must be at least {MIN_RESEARCH_NULL_DRAWS}")
        if self.hac_lag < self.label_horizon - 1:
            raise ValueError("hac_lag must be at least label_horizon - 1")
        if self.bootstrap_block_length < self.label_horizon:
            raise ValueError("bootstrap_block_length must be at least label_horizon")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must be in (0, 1)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.trial_ledger_complete and self.trial_ledger_path is None:
            raise ValueError("trial_ledger_complete requires a hashed trial_ledger_path")
        if self.trial_ledger_complete and not self.expected_trial_ids:
            raise ValueError("trial_ledger_complete requires expected_trial_ids")
        try:
            pd.Timestamp(f"2000-01-03T{self.signal_close_local_time}").tz_localize(
                self.exchange_timezone
            )
            pd.Timestamp("2000-01-03T12:00:00").tz_localize(self.pit_known_from_timezone)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "exchange, close-time, and PIT known-from timezones must be valid"
            ) from exc
        start = pd.Timestamp(self.test_start)
        end = pd.Timestamp(self.test_end)
        data_as_of = pd.Timestamp(self.data_as_of)
        if start > end:
            raise ValueError("test_start must not be after test_end")
        if data_as_of < start:
            raise ValueError("data_as_of must not be before test_start")


@dataclass(frozen=True)
class SelectionResearchEvidence:
    """In-memory form of the canonical five-file research bundle."""

    protocol: dict[str, Any]
    input_hashes: dict[str, Any]
    code_identity: dict[str, Any]
    date_evidence: pd.DataFrame
    result: dict[str, Any]
    report: str


def build_selection_research_evidence(
    protocol: SelectionResearchProtocol,
) -> SelectionResearchEvidence:
    """Evaluate one frozen prediction set without training or trading simulation."""
    predictions = _load_research_predictions(protocol.predictions_dir)
    market = _load_research_market(protocol.market_data_path)
    calendar, signal_closes, signal_close_rule = _load_session_calendar(
        protocol.calendar_path,
        exchange_timezone=protocol.exchange_timezone,
        signal_close_local_time=protocol.signal_close_local_time,
    )
    requested_dates = [
        value
        for value in calendar
        if _date_text(protocol.test_start) <= value <= _date_text(protocol.test_end)
    ]
    if not requested_dates:
        raise ValueError("No canonical sessions fall inside the requested test window")

    expected, score_denominator = _load_expected_scorable(
        protocol.expected_scorable_path,
        predictions,
    )
    pit_intervals = _load_optional_pit(
        protocol.pit_universe_path,
        known_from_timezone=protocol.pit_known_from_timezone,
    )
    input_hashes = _research_input_hashes(protocol)
    code_identity = _research_code_identity()
    canonical_protocol = _canonical_research_protocol(
        protocol,
        input_hashes=input_hashes,
        score_denominator=score_denominator,
    )
    canonical_protocol["code_identity"] = code_identity
    canonical_protocol["requested_signal_dates"] = requested_dates
    canonical_protocol["calendar"]["signal_close_rule"] = signal_close_rule
    matured_signal_dates = [
        signal_dt
        for signal_dt in requested_dates
        if calendar.index(signal_dt) + protocol.label_horizon < len(calendar)
        and calendar[calendar.index(signal_dt) + protocol.label_horizon]
        <= _date_text(protocol.data_as_of)
    ]
    canonical_protocol["realized_outcome_cutoff"] = (
        matured_signal_dates[-1] if matured_signal_dates else None
    )

    calendar_index = {value: index for index, value in enumerate(calendar)}
    market_key_counts = market.groupby(["dt", "kdcode"], sort=False).size()
    market_close_by_key = market.groupby(["dt", "kdcode"], sort=False)["close"].first()
    prediction_key_counts = predictions.groupby(["dt", "kdcode"], sort=False).size()
    prediction_score_by_key = predictions.groupby(["dt", "kdcode"], sort=False)["score"].first()
    predictions_by_date = predictions.set_index("dt", drop=False).sort_index()
    expected_by_date = expected.set_index("dt", drop=False).sort_index()
    empty_predictions = predictions.iloc[0:0]
    empty_expected = expected.iloc[0:0]

    rows: list[dict[str, Any]] = []
    valid_cross_sections: list[DatedScoreOutcome] = []
    pit_classes: list[PITKnowledgeClass] = []
    data_as_of = _date_text(protocol.data_as_of)

    for signal_dt in requested_dates:
        reasons: list[str] = []
        signal_index = calendar_index[signal_dt]
        label_start_dt = calendar[signal_index + 1] if signal_index + 1 < len(calendar) else None
        label_end_dt = (
            calendar[signal_index + protocol.label_horizon]
            if signal_index + protocol.label_horizon < len(calendar)
            else None
        )

        prediction_rows = _date_rows(
            predictions_by_date,
            signal_dt,
            empty=empty_predictions,
        )
        expected_rows = _date_rows(
            expected_by_date,
            signal_dt,
            empty=empty_expected,
        )
        expected_true = expected_rows[expected_rows["expected_scorable"]]
        expected_keys = tuple(sorted(set(expected_true["kdcode"].astype(str))))
        if expected_true["kdcode"].duplicated(keep=False).any():
            _add_reason(reasons, "DUPLICATE_EXPECTED_SCORABLE")
        if not expected_keys:
            _add_reason(reasons, "EMPTY_EXPECTED_SCORABLE_SET")

        active_keys: set[str] = set()
        pit_class = PITKnowledgeClass.UNKNOWN
        if pit_intervals is not None:
            active = pit_intervals[
                (pit_intervals["valid_from"] <= signal_dt)
                & (pit_intervals["valid_to"] >= signal_dt)
            ]
            active_keys = set(active["kdcode"].astype(str))
            pit_class = classify_pit_knowledge_as_of(
                pit_intervals,
                signal_closes[signal_dt],
                known_from_timezone=protocol.pit_known_from_timezone,
            )
            if not set(expected_keys).issubset(active_keys):
                _add_reason(reasons, "EXPECTED_SET_OUTSIDE_PIT_UNIVERSE")
            if pit_class is PITKnowledgeClass.UNKNOWN:
                _add_reason(reasons, "PIT_KNOWLEDGE_UNKNOWN")
            if score_denominator == "EXPECTED_SCORABLE":
                active_expected_rows = expected_rows[expected_rows["kdcode"].isin(active_keys)]
                represented_active_keys = set(active_expected_rows["kdcode"].astype(str))
                if represented_active_keys != active_keys:
                    _add_reason(reasons, "INCOMPLETE_EXPECTED_DENOMINATOR")
                if active_expected_rows["kdcode"].duplicated(keep=False).any():
                    _add_reason(reasons, "DUPLICATE_EXPECTED_DENOMINATOR_ROW")
                excluded_active = active_expected_rows[~active_expected_rows["expected_scorable"]]
                missing_exclusion_reason = (
                    excluded_active["exclusion_reason"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .any()
                )
                if missing_exclusion_reason:
                    _add_reason(reasons, "MISSING_EXCLUSION_REASON")
        else:
            _add_reason(reasons, "PIT_UNIVERSE_MISSING")
        pit_classes.append(pit_class)

        prediction_keys = prediction_rows["kdcode"].astype(str)
        if prediction_keys.duplicated(keep=False).any():
            _add_reason(reasons, "DUPLICATE_SCORE")
        observed_keys = set(prediction_keys)
        missing_scores = set(expected_keys) - observed_keys
        extra_scores = observed_keys - set(expected_keys)
        if missing_scores:
            _add_reason(reasons, "MISSING_EXPECTED_SCORE")
        if extra_scores:
            _add_reason(reasons, "UNEXPECTED_SCORE")

        score_by_key: dict[str, float] = {}
        for key in expected_keys:
            prediction_key = (signal_dt, key)
            if int(prediction_key_counts.get(prediction_key, 0)) != 1:
                continue
            value = float(prediction_score_by_key.get(prediction_key, float("nan")))
            if np.isfinite(value):
                score_by_key[key] = value
            else:
                _add_reason(reasons, "NONFINITE_EXPECTED_SCORE")

        matured = label_end_dt is not None and label_end_dt <= data_as_of
        complete_outcomes: dict[str, float] = {}
        if not matured or label_start_dt is None:
            has_pre_outcome_failure = bool(reasons)
            _add_reason(reasons, "OUTCOME_NOT_MATURED")
            date_status = "INVALID_PRIMARY" if has_pre_outcome_failure else "UNMATURED_OUTCOME"
        else:
            for key in expected_keys:
                start_key = (label_start_dt, key)
                end_key = (label_end_dt, key)
                start_count = int(market_key_counts.get(start_key, 0))
                end_count = int(market_key_counts.get(end_key, 0))
                if start_count == 0 or end_count == 0:
                    continue
                if start_count != 1 or end_count != 1:
                    _add_reason(reasons, "DUPLICATE_MARKET_PRICE")
                    continue
                start_close = float(market_close_by_key.get(start_key, float("nan")))
                end_close = float(market_close_by_key.get(end_key, float("nan")))
                if (
                    not np.isfinite(start_close)
                    or not np.isfinite(end_close)
                    or start_close <= 0.0
                    or end_close <= 0.0
                ):
                    _add_reason(reasons, "INVALID_MARKET_PRICE")
                    continue
                complete_outcomes[key] = end_close / start_close - 1.0
            if set(complete_outcomes) != set(expected_keys):
                _add_reason(reasons, "MISSING_MATURED_OUTCOME")
            if protocol.top_k > len(expected_keys):
                _add_reason(reasons, "TOP_K_EXCEEDS_BREADTH")
            date_status = "INVALID_PRIMARY" if reasons else "VALID_PRIMARY"

        daily_rank_ic: float | None = None
        top_k_return: float | None = None
        expected_return: float | None = None
        top_k_spread: float | None = None
        if date_status == "VALID_PRIMARY":
            scores = np.asarray([score_by_key[key] for key in expected_keys], dtype=np.float64)
            outcomes = np.asarray(
                [complete_outcomes[key] for key in expected_keys], dtype=np.float64
            )
            ic_observation = dated_daily_ic(
                (signal_dt,),
                (scores,),
                (outcomes,),
                method="spearman",
            )[0]
            if ic_observation.status != "VALID_PRIMARY":
                date_status = "INVALID_PRIMARY"
                _add_reason(reasons, ic_observation.status)
            else:
                daily_rank_ic = ic_observation.daily_ic
                ranked_indices = sorted(
                    range(len(expected_keys)),
                    key=lambda index: (-scores[index], expected_keys[index]),
                )
                top_indices = ranked_indices[: protocol.top_k]
                top_k_return = float(np.mean(outcomes[top_indices]))
                expected_return = float(np.mean(outcomes))
                top_k_spread = top_k_return - expected_return
                valid_cross_sections.append(
                    DatedScoreOutcome(
                        signal_dt=signal_dt,
                        instrument_keys=expected_keys,
                        scores=tuple(float(value) for value in scores),
                        outcomes=tuple(float(value) for value in outcomes),
                    )
                )

        rows.append(
            {
                "signal_dt": signal_dt,
                "label_start_dt": label_start_dt,
                "label_end_dt": label_end_dt,
                "PIT_active_count": len(active_keys),
                "expected_scorable_count": len(expected_keys),
                "prediction_count": int(len(prediction_rows)),
                "finite_score_count": len(score_by_key),
                "complete_outcome_count": len(complete_outcomes),
                "daily_rank_ic": daily_rank_ic,
                "top_k_label_return": top_k_return,
                "expected_set_label_return": expected_return,
                "top_k_spread": top_k_spread,
                "date_status": date_status,
                "reason_codes": "|".join(reasons),
            }
        )

    date_evidence = pd.DataFrame(rows, columns=DATE_EVIDENCE_COLUMNS)
    universe_knowledge = _aggregate_pit_class(pit_classes)
    canonical_protocol["evidence_contract"]["universe_knowledge"] = universe_knowledge
    result = _build_research_result(
        protocol,
        canonical_protocol=canonical_protocol,
        input_hashes=input_hashes,
        code_identity=code_identity,
        date_evidence=date_evidence,
        valid_cross_sections=valid_cross_sections,
        universe_knowledge=universe_knowledge,
        score_denominator=score_denominator,
    )
    report = _build_research_report(canonical_protocol, result, date_evidence)
    return SelectionResearchEvidence(
        protocol=canonical_protocol,
        input_hashes=input_hashes,
        code_identity=code_identity,
        date_evidence=date_evidence,
        result=result,
        report=report,
    )


def write_selection_research_evidence(
    evidence: SelectionResearchEvidence,
    output_root: str | Path,
) -> dict[str, str | Path]:
    """Write one immutable canonical five-file research evidence bundle."""
    return write_selection_research_bundle(
        output_root,
        research_semantics_version=str(evidence.protocol["research_semantics_version"]),
        protocol=evidence.protocol,
        input_hashes=evidence.input_hashes,
        code_identity=evidence.code_identity,
        date_evidence=evidence.date_evidence.to_dict(orient="records"),
        date_evidence_columns=DATE_EVIDENCE_COLUMNS,
        result=evidence.result,
        report=evidence.report,
    )


def _load_research_predictions(predictions_dir: str | Path) -> pd.DataFrame:
    frame = load_prediction_files(predictions_dir).copy()
    frame["dt"] = pd.to_datetime(frame["dt"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["kdcode"] = frame["kdcode"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if (frame["kdcode"] == "").any():
        raise ValueError("Prediction input contains an empty kdcode")
    return frame.sort_values(["dt", "kdcode"], kind="mergesort").reset_index(drop=True)


def _load_research_market(market_data_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(market_data_path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = {"dt", "kdcode", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"Market input missing columns: {sorted(missing)}")
    frame = frame[["dt", "kdcode", "close"]].copy()
    frame["dt"] = pd.to_datetime(frame["dt"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["kdcode"] = frame["kdcode"].astype(str).str.strip()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if (frame["kdcode"] == "").any():
        raise ValueError("Market input contains an empty kdcode")
    return frame.sort_values(["dt", "kdcode"], kind="mergesort").reset_index(drop=True)


def _date_rows(
    indexed_frame: pd.DataFrame,
    signal_dt: str,
    *,
    empty: pd.DataFrame,
) -> pd.DataFrame:
    if signal_dt not in indexed_frame.index:
        return empty
    return indexed_frame.loc[[signal_dt]]


def _load_session_calendar(
    calendar_path: str | Path,
    *,
    exchange_timezone: str,
    signal_close_local_time: str,
) -> tuple[list[str], dict[str, pd.Timestamp], str]:
    frame = pd.read_csv(calendar_path)
    if frame.empty:
        raise ValueError("Session calendar must contain at least one row")
    column = "dt" if "dt" in frame.columns else str(frame.columns[0])
    dates = pd.to_datetime(frame[column], errors="raise").dt.strftime("%Y-%m-%d")
    if dates.duplicated().any():
        raise ValueError("Session calendar contains duplicate dates")
    if "signal_close" in frame:
        close_values = [pd.Timestamp(value) for value in frame["signal_close"]]
        if any(value.tzinfo is None for value in close_values):
            raise ValueError(
                "Explicit session-calendar signal_close values must include a timezone"
            )
        parsed_close = [value.tz_convert("UTC") for value in close_values]
        signal_closes = dict(zip(dates, parsed_close, strict=True))
        signal_close_rule = "EXPLICIT_PER_SESSION_UTC"
    else:
        signal_closes = {
            dt: pd.Timestamp(f"{dt}T{signal_close_local_time}").tz_localize(exchange_timezone)
            for dt in dates
        }
        signal_close_rule = "DECLARED_LOCAL_CLOSE_TIME"
    ordered_dates = sorted(dates.tolist())
    for dt in ordered_dates:
        local_close_date = signal_closes[dt].tz_convert(exchange_timezone).strftime("%Y-%m-%d")
        if local_close_date != dt:
            raise ValueError(f"Session {dt} has signal_close on local date {local_close_date}")
    ordered_closes = [signal_closes[dt].tz_convert("UTC") for dt in ordered_dates]
    if any(
        later <= earlier for earlier, later in zip(ordered_closes, ordered_closes[1:], strict=False)
    ):
        raise ValueError("Session signal_close timestamps must be strictly increasing")
    return ordered_dates, signal_closes, signal_close_rule


def _load_expected_scorable(
    path: str | Path | None,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if path is None:
        frame = predictions[["dt", "kdcode"]].drop_duplicates().copy()
        frame["expected_scorable"] = True
        frame["exclusion_reason"] = ""
        return frame, "SCORED_SET_ONLY"

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = {"dt", "kdcode"} - set(frame.columns)
    if missing:
        raise ValueError(f"Expected-scorable input missing columns: {sorted(missing)}")
    frame["dt"] = pd.to_datetime(frame["dt"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["kdcode"] = frame["kdcode"].astype(str).str.strip()
    if (frame["kdcode"] == "").any():
        raise ValueError("Expected-scorable input contains an empty kdcode")
    if "expected_scorable" not in frame:
        frame["expected_scorable"] = True
    else:
        frame["expected_scorable"] = frame["expected_scorable"].map(_parse_expected_bool)
    if "exclusion_reason" not in frame:
        frame["exclusion_reason"] = ""
    return (
        frame[["dt", "kdcode", "expected_scorable", "exclusion_reason"]]
        .sort_values(["dt", "kdcode"], kind="mergesort")
        .reset_index(drop=True),
        "EXPECTED_SCORABLE",
    )


def _parse_expected_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid expected_scorable value: {value!r}")


def _load_optional_pit(
    path: str | Path | None,
    *,
    known_from_timezone: str,
) -> pd.DataFrame | None:
    if path is None:
        return None
    return normalise_pit_intervals(
        pd.read_csv(path),
        known_from_timezone=known_from_timezone,
    )


def _canonical_research_protocol(
    protocol: SelectionResearchProtocol,
    *,
    input_hashes: dict[str, Any],
    score_denominator: str,
) -> dict[str, Any]:
    return {
        "schema": "mci_gru.selection_research_protocol.v1",
        "research_semantics_version": protocol.research_semantics_version,
        "study_name": protocol.study_name,
        "trial_family_id": protocol.trial_family_id,
        "primary_endpoint": "mean_daily_spearman_rank_ic",
        "secondary_endpoint": "mean_top_k_label_return_minus_expected_set_mean",
        "label_definition": "adjusted_close[T+h] / adjusted_close[T+1] - 1",
        "label_horizon": int(protocol.label_horizon),
        "test_start": _date_text(protocol.test_start),
        "test_end": _date_text(protocol.test_end),
        "data_as_of": _date_text(protocol.data_as_of),
        "top_k": int(protocol.top_k),
        "prediction_provenance": {
            "source_run_id": protocol.prediction_source_run_id,
            "ensemble_rule": protocol.prediction_ensemble_rule,
            "ensemble_member_count": protocol.prediction_ensemble_member_count,
            "seed_id": protocol.prediction_seed_id,
            "source_code_commit": protocol.prediction_source_code_commit,
            "label_contract": protocol.prediction_label_contract,
            "label_horizon": protocol.prediction_label_horizon,
            "prediction_set_sha256": input_hashes["predictions"]["aggregate_sha256"],
        },
        "calendar": {
            "source": protocol.calendar_source,
            "exchange_timezone": protocol.exchange_timezone,
            "default_signal_close_local_time": protocol.signal_close_local_time,
        },
        "null": {
            "family": protocol.null_family,
            "draws": int(protocol.null_draws),
            "seed": int(protocol.null_seed),
            "alternative": "observed_mean_rank_ic_greater_than_null",
            "empirical_p_value": "plus_one_upper_tail",
        },
        "inference": {
            "alpha": float(protocol.alpha),
            "ci_level": float(protocol.ci_level),
            "hac_lag": int(protocol.hac_lag),
            "bootstrap_block_length": int(protocol.bootstrap_block_length),
            "bootstrap_resamples": int(protocol.bootstrap_resamples),
            "bootstrap_seed": int(protocol.bootstrap_seed),
            "unit": "signal_date",
        },
        "evidence_contract": {
            "price_basis": protocol.price_basis,
            "price_adjustment_provenance": protocol.price_adjustment_provenance,
            "pit_known_from_timezone": protocol.pit_known_from_timezone,
            "universe_knowledge": "PENDING_DATE_VALIDATION",
            "score_denominator": score_denominator,
            "fill_missing_prices": False,
        },
        "multiplicity": {
            "trial_ledger_complete": bool(protocol.trial_ledger_complete),
            "expected_trial_ids": sorted(protocol.expected_trial_ids),
            "oos_previously_accessed": bool(protocol.oos_previously_accessed),
        },
        "inputs": input_hashes,
    }


def _research_input_hashes(protocol: SelectionResearchProtocol) -> dict[str, Any]:
    prediction_path = Path(protocol.predictions_dir)
    prediction_files = sorted(prediction_path.glob("*.csv"))
    prediction_records = [
        {
            "source_id": path.name,
            "sha256": _sha256_file(path),
            "row_count": int(len(pd.read_csv(path))),
        }
        for path in prediction_files
    ]
    if not prediction_records:
        raise FileNotFoundError(f"No prediction CSV files found in {prediction_path}")
    prediction_hash = hashlib.sha256(canonical_json_bytes(prediction_records)).hexdigest()
    return {
        "predictions": {
            "source_id": prediction_path.name,
            "aggregate_sha256": prediction_hash,
            "file_count": len(prediction_records),
            "files": prediction_records,
        },
        "market": _hashed_csv_input(Path(protocol.market_data_path)),
        "pit_universe": _optional_hashed_csv_input(protocol.pit_universe_path),
        "expected_scorable": _optional_hashed_csv_input(protocol.expected_scorable_path),
        "session_calendar": _hashed_csv_input(Path(protocol.calendar_path)),
        "trial_ledger": _optional_hashed_input(protocol.trial_ledger_path),
    }


def _hashed_csv_input(path: Path) -> dict[str, Any]:
    return {
        "source_id": path.name,
        "sha256": _sha256_file(path),
        "row_count": int(len(pd.read_csv(path))),
    }


def _optional_hashed_csv_input(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_SUPPLIED"}
    return _hashed_csv_input(Path(path))


def _optional_hashed_input(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_SUPPLIED"}
    input_path = Path(path)
    return {"source_id": input_path.name, "sha256": _sha256_file(input_path)}


def _research_code_identity() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "mci_gru/data/pit.py",
        "mci_gru/data/preprocessing.py",
        "mci_gru/evaluation/artifacts.py",
        "mci_gru/evaluation/prediction_report.py",
        "mci_gru/evaluation/selection_audit.py",
        "mci_gru/evaluation/selection_nulls.py",
        "mci_gru/evaluation/statistics.py",
        "scripts/run_saved_prediction_selection_audit.py",
    )
    source_hashes = {
        path: _sha256_file(repo_root / path)
        for path in relative_paths
        if (repo_root / path).is_file()
    }
    source_tree_hash = hashlib.sha256(canonical_json_bytes(source_hashes)).hexdigest()
    git_commit = _git_text(repo_root, "rev-parse", "HEAD")
    diff = _git_bytes(
        repo_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        *relative_paths,
    )
    return {
        "schema": "mci_gru.selection_research_code_identity.v1",
        "git_commit": git_commit,
        "git_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "working_tree_source_sha256": source_tree_hash,
        "source_hashes": source_hashes,
        "runtime_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "python": _package_version("python"),
            "scipy": _package_version("scipy"),
        },
    }


def _git_text(repo_root: Path, *args: str) -> str | None:
    output = _git_bytes(repo_root, *args)
    text = output.decode("utf-8", errors="replace").strip()
    return text or None


def _package_version(package: str) -> str | None:
    if package == "python":
        return ".".join(str(value) for value in sys.version_info[:3])
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return b""
    return completed.stdout if completed.returncode == 0 else b""


def _build_research_result(
    protocol: SelectionResearchProtocol,
    *,
    canonical_protocol: dict[str, Any],
    input_hashes: dict[str, Any],
    code_identity: dict[str, Any],
    date_evidence: pd.DataFrame,
    valid_cross_sections: list[DatedScoreOutcome],
    universe_knowledge: str,
    score_denominator: str,
) -> dict[str, Any]:
    valid_rows = date_evidence[date_evidence["date_status"] == "VALID_PRIMARY"]
    rank_ic = valid_rows["daily_rank_ic"].to_numpy(dtype=np.float64)
    top_k_spread = valid_rows["top_k_spread"].to_numpy(dtype=np.float64)
    observed_mean_rank_ic = float(np.mean(rank_ic)) if rank_ic.size else None
    observed_mean_top_k_spread = float(np.mean(top_k_spread)) if top_k_spread.size else None

    hac = newey_west_mean_inference(
        rank_ic,
        lags=protocol.hac_lag,
        label_horizon=protocol.label_horizon,
    )
    bootstrap = moving_block_mean_ci(
        rank_ic,
        block_size=protocol.bootstrap_block_length,
        label_horizon=protocol.label_horizon,
        n_resamples=protocol.bootstrap_resamples,
        seed=protocol.bootstrap_seed,
        ci_level=protocol.ci_level,
    )

    empirical_p_value: float | None = None
    null_payload: dict[str, Any] = {
        "family": protocol.null_family,
        "draw_count": int(protocol.null_draws),
        "valid_draw_count": 0,
        "seed": int(protocol.null_seed),
        "top_k": int(protocol.top_k),
        "assignment_digest": None,
        "assignment_digest_method": None,
        "rank_ic_mean_quantiles": None,
        "top_k_spread_mean_quantiles": None,
    }
    if valid_cross_sections:
        null_result = run_within_date_permutation_null(
            valid_cross_sections,
            n_draws=protocol.null_draws,
            null_seed=protocol.null_seed,
            top_k=protocol.top_k,
        )
        null_rank = np.asarray(null_result.rank_ic_means, dtype=np.float64)
        null_spread = np.asarray(null_result.top_k_spread_means, dtype=np.float64)
        if observed_mean_rank_ic is not None:
            empirical_p_value = empirical_one_sided_p_value(
                observed_mean_rank_ic,
                null_rank,
            )
        null_payload.update(
            {
                "method": null_result.method,
                "valid_draw_count": int(null_result.valid_draw_count),
                "assignment_digest": null_result.assignment_digest,
                "assignment_digest_method": null_result.assignment_digest_method,
                "rank_ic_mean_quantiles": _null_quantiles(null_rank),
                "top_k_spread_mean_quantiles": _null_quantiles(null_spread),
            }
        )

    invalid_primary_count = int((date_evidence["date_status"] == "INVALID_PRIMARY").sum())
    unmatured_count = int((date_evidence["date_status"] == "UNMATURED_OUTCOME").sum())
    failed_guards: list[str] = []
    hard_invalid_guards: list[str] = []
    if protocol.price_basis != "ADJUSTED_RESEARCH":
        hard_invalid_guards.append("PRICE_BASIS_NOT_ADJUSTED_RESEARCH")
    if protocol.price_adjustment_provenance.strip().upper() == "UNKNOWN":
        hard_invalid_guards.append("PRICE_ADJUSTMENT_PROVENANCE_UNKNOWN")
    if protocol.prediction_source_code_commit is None:
        hard_invalid_guards.append("PREDICTION_SOURCE_CODE_COMMIT_MISSING")
    if protocol.prediction_label_contract != MCI_GRU_FORWARD_CLOSE_V1:
        hard_invalid_guards.append("PREDICTION_LABEL_CONTRACT_MISMATCH")
    if protocol.prediction_label_horizon != protocol.label_horizon:
        hard_invalid_guards.append("PREDICTION_LABEL_HORIZON_MISMATCH")
    if universe_knowledge == "UNKNOWN":
        hard_invalid_guards.append("PIT_KNOWLEDGE_UNKNOWN")
    if score_denominator == "UNKNOWN":
        hard_invalid_guards.append("SCORE_DENOMINATOR_UNKNOWN")
    if protocol.trial_ledger_complete and not _trial_ledger_matches_family(
        protocol.trial_ledger_path,
        protocol.trial_family_id,
        protocol.expected_trial_ids,
    ):
        hard_invalid_guards.append("TRIAL_LEDGER_FAMILY_MISMATCH")
    failed_guards.extend(hard_invalid_guards)
    if invalid_primary_count:
        failed_guards.append("INVALID_PRIMARY_DATES_PRESENT")

    limitations: list[str] = []
    if protocol.oos_previously_accessed:
        limitations.append("OOS_PREVIOUSLY_ACCESSED")
    if not protocol.trial_ledger_complete:
        limitations.append("INCOMPLETE_TRIAL_FAMILY_LEDGER")
    if universe_knowledge == "EFFECTIVE_ONLY":
        limitations.append("PIT_EFFECTIVE_DATES_ONLY")
    if score_denominator == "SCORED_SET_ONLY":
        limitations.append("SCORED_SET_ONLY_DENOMINATOR")
    if protocol.prediction_source_run_id is None:
        limitations.append("PREDICTION_SOURCE_RUN_NOT_DECLARED")
    if protocol.prediction_ensemble_member_count is None:
        limitations.append("ENSEMBLE_MEMBER_COUNT_NOT_DECLARED")
    if protocol.prediction_seed_id is None:
        limitations.append("PREDICTION_SEED_NOT_DECLARED")
    if canonical_protocol["calendar"]["signal_close_rule"] != "EXPLICIT_PER_SESSION_UTC":
        limitations.append("DECLARED_FIXED_SESSION_CLOSE_RULE")
    if len(valid_rows) < MIN_PRELIMINARY_DATES:
        limitations.append("FEWER_THAN_60_VALID_DATES")
    if unmatured_count:
        limitations.append("UNMATURED_TAIL_DATES_PRESENT")

    bootstrap_lower = bootstrap.lower
    positive_evidence = (
        observed_mean_rank_ic is not None
        and observed_mean_rank_ic > 0.0
        and bootstrap_lower is not None
        and bootstrap_lower > 0.0
        and empirical_p_value is not None
        and empirical_p_value <= protocol.alpha
    )
    if hard_invalid_guards or invalid_primary_count:
        claim_status = "INVALID_EVIDENCE"
    elif (
        len(valid_rows) < MIN_PRELIMINARY_DATES
        or null_payload["valid_draw_count"] < MIN_RESEARCH_NULL_DRAWS
    ):
        claim_status = "INSUFFICIENT_EVIDENCE"
    elif positive_evidence:
        claim_status = "PRELIMINARY_SIGNAL_EVIDENCE"
    else:
        claim_status = "NO_DETECTABLE_SIGNAL"

    headline_rank_ic = observed_mean_rank_ic
    headline_top_k_spread = observed_mean_top_k_spread
    headline_p_value = empirical_p_value
    hac_payload = asdict(hac)
    bootstrap_payload = asdict(bootstrap)
    if claim_status == "INVALID_EVIDENCE":
        headline_rank_ic = None
        headline_top_k_spread = None
        headline_p_value = None
        for field in ("mean", "standard_error", "t_stat", "p_value"):
            hac_payload[field] = None
        bootstrap_payload["lower"] = None
        bootstrap_payload["upper"] = None

    study_id = build_research_study_id(
        research_semantics_version=protocol.research_semantics_version,
        protocol=canonical_protocol,
        input_hashes=input_hashes,
        code_identity=code_identity,
    )
    return {
        "schema": "mci_gru.selection_research_result.v1",
        "study_id": study_id,
        "claim_status": claim_status,
        "evidence_class": {
            "price_basis": protocol.price_basis,
            "price_adjustment_provenance": protocol.price_adjustment_provenance,
            "universe_knowledge": universe_knowledge,
            "score_denominator": score_denominator,
            "maximum_claim": "PRELIMINARY_SIGNAL_EVIDENCE",
        },
        "requested_date_count": int(len(date_evidence)),
        "valid_date_count": int(len(valid_rows)),
        "invalid_date_count": invalid_primary_count,
        "unmatured_date_count": unmatured_count,
        "observed_mean_rank_ic": headline_rank_ic,
        "observed_mean_top_k_spread": headline_top_k_spread,
        "hac_mean_inference": hac_payload,
        "moving_block_bootstrap": bootstrap_payload,
        "null_test": null_payload,
        "empirical_p_value": headline_p_value,
        "multiplicity_status": (
            "DECLARED_COMPLETE_TRIAL_LEDGER"
            if protocol.trial_ledger_complete
            else "UNADJUSTED_EXPLORATORY"
        ),
        "adjusted_p_value": None,
        "failed_guards": failed_guards,
        "limitation_codes": limitations,
    }


def _build_research_report(
    protocol: dict[str, Any],
    result: dict[str, Any],
    date_evidence: pd.DataFrame,
) -> str:
    status = str(result["claim_status"])
    decisions = {
        "INVALID_EVIDENCE": (
            "Repair the declared input or coverage guards before interpreting signal."
        ),
        "INSUFFICIENT_EVIDENCE": (
            "Preserve this result and collect more predeclared valid OOS dates; do not tune "
            "on this sample."
        ),
        "NO_DETECTABLE_SIGNAL": (
            "Stop the active engineering path; this frozen test did not distinguish the "
            "scores from the matched noise null."
        ),
        "PRELIMINARY_SIGNAL_EVIDENCE": (
            "Freeze a separate economic-replay and future-confirmation plan before adding "
            "trading realism."
        ),
    }
    null_test = result["null_test"]
    bootstrap = result["moving_block_bootstrap"]
    hac = result["hac_mean_inference"]
    input_lines = [
        f"- {name}: {_input_hash_text(description)}"
        for name, description in protocol["inputs"].items()
    ]
    reason_counts = _reason_count_text(date_evidence)
    return "\n".join(
        [
            f"# Selection research result: {protocol['study_name']}",
            "",
            f"**Conclusion:** `{status}`.",
            "",
            "This study tests whether frozen MCI-GRU score ranks have positive dated "
            "cross-sectional association with the exact trained-label return and beat "
            "within-date random score assignments. It does not estimate executable returns.",
            "",
            "## Frozen protocol",
            "",
            f"- Test window: {protocol['test_start']} through {protocol['test_end']}",
            f"- Label horizon: {protocol['label_horizon']} canonical sessions",
            f"- Null: {null_test['family']} with {null_test['draw_count']} draws",
            f"- Trial family: {protocol['trial_family_id']}",
            f"- Prediction source run: {protocol['prediction_provenance']['source_run_id']}",
            f"- Ensemble rule: {protocol['prediction_provenance']['ensemble_rule']}",
            "- Ensemble member count: "
            f"{protocol['prediction_provenance']['ensemble_member_count']}",
            f"- Prediction seed: {protocol['prediction_provenance']['seed_id']}",
            "- Prediction source code commit: "
            f"{protocol['prediction_provenance']['source_code_commit']}",
            f"- Prediction label contract: {protocol['prediction_provenance']['label_contract']}",
            f"- Prediction label horizon: {protocol['prediction_provenance']['label_horizon']}",
            "",
            "## Input identities",
            "",
            *input_lines,
            "",
            "## Study and code identity",
            "",
            f"- Study ID: {result['study_id']}",
            f"- Git commit: {protocol['code_identity']['git_commit']}",
            f"- Git dirty diff SHA-256: {protocol['code_identity']['git_dirty_diff_sha256']}",
            "- Working source tree SHA-256: "
            f"{protocol['code_identity']['working_tree_source_sha256']}",
            "",
            "## Evidence and coverage",
            "",
            f"- Valid dates: {result['valid_date_count']} of {len(date_evidence)}",
            f"- Invalid dates: {result['invalid_date_count']}",
            f"- Unmatured dates: {result['unmatured_date_count']}",
            f"- Price basis: {result['evidence_class']['price_basis']}",
            "- Price adjustment provenance: "
            f"{result['evidence_class']['price_adjustment_provenance']}",
            f"- PIT knowledge: {result['evidence_class']['universe_knowledge']}",
            f"- Score denominator: {result['evidence_class']['score_denominator']}",
            f"- Invalid reason counts: {reason_counts}",
            "",
            "## Primary result",
            "",
            f"- Mean daily Rank IC: {_report_number(result['observed_mean_rank_ic'])}",
            "- Moving-block interval: "
            f"[{_report_number(bootstrap['lower'])}, {_report_number(bootstrap['upper'])}]",
            f"- HAC lag: {hac['lags']}",
            f"- HAC standard error: {_report_number(hac['standard_error'])}",
            f"- HAC t-statistic: {_report_number(hac['t_stat'])}",
            f"- HAC two-sided p-value: {_report_number(hac['p_value'])}",
            f"- One-sided permutation p-value: {_report_number(result['empirical_p_value'])}",
            "- Mean top-k spread over the expected set: "
            f"{_report_number(result['observed_mean_top_k_spread'])}",
            "",
            "## Multiplicity and limitations",
            "",
            f"- Multiplicity: {result['multiplicity_status']}",
            "- Failed guards: " + (", ".join(result["failed_guards"]) or "none"),
            "- Limitations: " + (", ".join(result["limitation_codes"]) or "none"),
            "",
            "## Recommendation",
            "",
            decisions[status],
            "",
        ]
    )


def _aggregate_pit_class(values: list[PITKnowledgeClass]) -> str:
    if not values or PITKnowledgeClass.UNKNOWN in values:
        return PITKnowledgeClass.UNKNOWN.value
    if PITKnowledgeClass.EFFECTIVE_ONLY in values:
        return PITKnowledgeClass.EFFECTIVE_ONLY.value
    return PITKnowledgeClass.KNOWN_AS_OF.value


def _null_quantiles(values: np.ndarray) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    quantiles = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def _input_hash_text(description: dict[str, Any]) -> str:
    digest = description.get("aggregate_sha256") or description.get("sha256")
    return str(digest) if digest is not None else str(description.get("status", "UNKNOWN"))


def _reason_count_text(date_evidence: pd.DataFrame) -> str:
    counts: dict[str, int] = {}
    for value in date_evidence["reason_codes"].fillna("").astype(str):
        for reason in value.split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{reason}={counts[reason]}" for reason in sorted(counts))


def _date_text(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _trial_ledger_matches_family(
    path: str | Path | None,
    family_id: str,
    expected_trial_ids: tuple[str, ...],
) -> bool:
    if path is None:
        return False
    ledger_path = Path(path)
    try:
        if ledger_path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            frame = pd.DataFrame(records)
        else:
            frame = pd.read_csv(ledger_path)
        validate_trial_family(
            frame,
            family_id=family_id,
            expected_trial_ids=expected_trial_ids,
        )
    except (OSError, ValueError):
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _report_number(value: object) -> str:
    if value is None:
        return "not available"
    numeric = float(value)
    return f"{numeric:.6g}" if np.isfinite(numeric) else "not available"


def bhy_adjust_p_value(p_value: float, trial_count: int) -> float:
    """Apply the Benjamini-Hochberg-Yekutieli single p-value inflation."""
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    harmonic = sum(1.0 / i for i in range(1, trial_count + 1))
    return float(min(1.0, p_value * trial_count * harmonic))


def build_selection_audit(
    *,
    predictions_dir: str | Path,
    market_data_path: str | Path,
    label_t: int,
    top_k_values: list[int],
    trial_count: int,
    bootstrap_resamples: int = 500,
    bootstrap_seed: int = 123,
) -> dict[str, Any]:
    """Compute no-retraining selection evidence from saved prediction CSVs."""
    predictions = load_prediction_files(predictions_dir)
    market_data = pd.read_csv(market_data_path)
    realized = realized_returns_from_market_data(market_data, label_t=label_t)
    aligned = align_prediction_comparison(predictions, realized)
    if aligned.empty:
        score_matrix = np.empty((0, 0), dtype=np.float64)
        return_matrix = np.empty((0, 0), dtype=np.float64)
    else:
        score_matrix = _pivot(aligned, "mci_gru_score")
        return_matrix = _pivot(aligned, "realized_return")

    pearson = daily_ic_series(score_matrix, return_matrix, method="pearson")
    spearman = daily_ic_series(score_matrix, return_matrix, method="spearman")
    rank_ic_mean = _nanmean(spearman)
    valid_spearman_count = int(np.isfinite(spearman).sum())
    if valid_spearman_count:
        nw_std = newey_west_std(spearman, lags=max(0, label_t - 1))
        t_stat = _newey_west_t_stat(rank_ic_mean, nw_std, len(spearman))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(len(spearman) - 1, 1)))
        bootstrap_ci = moving_block_bootstrap_ci(
            spearman,
            statistic=lambda values: float(np.nanmean(values)),
            block_size=max(1, label_t),
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            ci_level=0.95,
        )
    else:
        t_stat = None
        p_value = None
        bootstrap_ci = None
    top_k_return_map = _top_k_return_map(score_matrix, return_matrix, top_k_values)
    reasons = _insufficient_evidence_reasons(
        aligned,
        valid_spearman_count=valid_spearman_count,
        top_k_return_map=top_k_return_map,
    )

    return {
        "schema_version": 1,
        "status": "INSUFFICIENT_EVIDENCE" if reasons else "OK",
        "insufficient_evidence_reasons": reasons,
        "predictions_dir": str(Path(predictions_dir).resolve()),
        "market_data_path": str(Path(market_data_path).resolve()),
        "label_t": int(label_t),
        "trial_count": int(trial_count),
        "sample": {
            "aligned_observations": int(len(aligned)),
            "n_dates": int(aligned["dt"].nunique()) if "dt" in aligned else 0,
            "n_kdcodes": int(aligned["kdcode"].nunique()) if "kdcode" in aligned else 0,
            "valid_ic_days": valid_spearman_count,
        },
        "ic": {
            "pearson_mean": _nanmean(pearson),
            "spearman_mean": rank_ic_mean,
            "spearman_newey_west_t": t_stat,
            "spearman_p_value": p_value,
            "spearman_bootstrap_ci": bootstrap_ci,
        },
        "top_k": _top_k_summary(top_k_return_map),
        "deflated_sharpe": {
            str(top_k): deflated_sharpe_ratio(returns, trial_count=trial_count)
            for top_k, returns in top_k_return_map.items()
        },
        "multiple_testing": {
            "method": "bhy_single_family_v0",
            "bhy_adjusted_p_value": bhy_adjust_p_value(p_value, trial_count)
            if p_value is not None
            else None,
        },
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    trial_count: int,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Estimate deflated Sharpe evidence for a return series."""
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    clean = np.asarray(returns, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return {
            "method": "bailey_lopez_de_prado_v0",
            "n_obs": int(clean.size),
            "trial_count": int(trial_count),
            "period_sharpe": None,
            "annualized_sharpe": None,
            "expected_max_sharpe": None,
            "z_stat": None,
            "p_value": None,
            "skew": None,
            "kurtosis": None,
        }

    mean_return = float(np.mean(clean))
    std_return = float(np.std(clean, ddof=1))
    if std_return <= 0.0 or not np.isfinite(std_return):
        period_sharpe = None
        annualized_sharpe = None
        z_stat = None
        p_value = None
        expected_max_sharpe = None
    else:
        period_sharpe = mean_return / std_return
        skew = _finite_or_default(stats.skew(clean, bias=False), 0.0)
        kurtosis = _finite_or_default(stats.kurtosis(clean, fisher=False, bias=False), 3.0)
        variance_term = 1.0 - skew * period_sharpe + ((kurtosis - 1.0) / 4.0) * (period_sharpe**2)
        standard_error = math.sqrt(max(variance_term, 1e-12) / max(clean.size - 1, 1))
        expected_max_sharpe = _expected_max_sharpe(trial_count, standard_error)
        z_stat = (period_sharpe - expected_max_sharpe) / standard_error
        p_value = float(stats.norm.sf(z_stat))
        annualized_sharpe = period_sharpe * math.sqrt(periods_per_year)
        return {
            "method": "bailey_lopez_de_prado_v0",
            "n_obs": int(clean.size),
            "trial_count": int(trial_count),
            "period_sharpe": float(period_sharpe),
            "annualized_sharpe": float(annualized_sharpe),
            "expected_max_sharpe": float(expected_max_sharpe),
            "z_stat": float(z_stat),
            "p_value": p_value,
            "skew": float(skew),
            "kurtosis": float(kurtosis),
        }

    return {
        "method": "bailey_lopez_de_prado_v0",
        "n_obs": int(clean.size),
        "trial_count": int(trial_count),
        "period_sharpe": period_sharpe,
        "annualized_sharpe": annualized_sharpe,
        "expected_max_sharpe": expected_max_sharpe,
        "z_stat": z_stat,
        "p_value": p_value,
        "skew": _finite_or_none(stats.skew(clean, bias=False)),
        "kurtosis": _finite_or_none(stats.kurtosis(clean, fisher=False, bias=False)),
    }


def write_selection_audit(
    audit: dict[str, Any], output_dir: str | Path, *, force: bool = False
) -> Path:
    """Write an additive selection audit JSON artifact."""
    out_dir = Path(output_dir)
    path = out_dir / "selection_audit_summary.json"
    return write_json_artifact(path, audit, force=force)


def _pivot(frame: pd.DataFrame, value_col: str) -> np.ndarray:
    wide = frame.pivot(index="dt", columns="kdcode", values=value_col).sort_index()
    return wide.to_numpy(dtype=np.float64)


def _nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else float("nan")


def _newey_west_t_stat(mean_value: float, nw_std: float, n_obs: int) -> float:
    if n_obs <= 0 or nw_std <= 0 or not np.isfinite(mean_value):
        return 0.0
    return float(mean_value / (nw_std / np.sqrt(n_obs)))


def _top_k_return_map(
    score_matrix: np.ndarray,
    return_matrix: np.ndarray,
    top_k_values: list[int],
) -> dict[int, np.ndarray]:
    return {
        int(top_k): top_k_returns(score_matrix, return_matrix, top_k=top_k)
        for top_k in top_k_values
    }


def _top_k_summary(return_map: dict[int, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for top_k, returns in return_map.items():
        summary[str(top_k)] = {
            "mean_return": _nanmean(returns),
            "n_days": int(returns.size),
        }
    return summary


def _insufficient_evidence_reasons(
    aligned: pd.DataFrame,
    *,
    valid_spearman_count: int,
    top_k_return_map: dict[int, np.ndarray],
) -> list[str]:
    reasons: list[str] = []
    if aligned.empty:
        reasons.append("no_aligned_observations")
    if valid_spearman_count == 0:
        reasons.append("no_valid_ic_days")
    if not any(returns.size for returns in top_k_return_map.values()):
        reasons.append("no_top_k_return_observations")
    return reasons


def _expected_max_sharpe(trial_count: int, standard_error: float) -> float:
    if trial_count <= 1:
        return 0.0
    gamma = 0.5772156649015329
    q_1 = stats.norm.ppf(1.0 - 1.0 / trial_count)
    q_2 = stats.norm.ppf(1.0 - 1.0 / (trial_count * math.e))
    estimate = standard_error * ((1.0 - gamma) * q_1 + gamma * q_2)
    return float(estimate) if np.isfinite(estimate) else 0.0


def _finite_or_default(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else default


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
