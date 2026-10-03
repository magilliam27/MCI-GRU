"""Point-in-time universe masks for fixed-axis stock panels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from mci_gru.data.preprocessing import LabelResolution, resolve_labels

if TYPE_CHECKING:
    from mci_gru.data.input_observations import InputObservationContext


@dataclass(frozen=True)
class PITMaskSet:
    """Daily ``(dates, stocks)`` populations on the fixed union axis.

    ``active_member`` is unchanged monthly selection. ``cessation_excluded`` marks a
    cessation that is both effective and known at the prediction time (#225), and
    ``eligible`` is ``active_member & ~cessation_excluded``. ``tradable`` (the
    prediction population) is ``eligible & feature_ready``; ``loss`` adds an
    observable label. Eligibility never reads ``label_available``.
    """

    active_member: np.ndarray
    feature_ready: np.ndarray
    loss: np.ndarray
    tradable: np.ndarray
    cessation_excluded: np.ndarray | None = None
    eligible: np.ndarray | None = None
    label_available: np.ndarray | None = None


class PITKnowledgeClass(str, Enum):
    """Strength of the membership-provenance evidence at a signal close."""

    KNOWN_AS_OF = "KNOWN_AS_OF"
    EFFECTIVE_ONLY = "EFFECTIVE_ONLY"
    UNKNOWN = "UNKNOWN"


def _as_utc_timestamp(
    value: object,
    *,
    naive_timezone: str = "UTC",
) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        try:
            return timestamp.tz_localize(naive_timezone).tz_convert("UTC")
        except (TypeError, ValueError):
            return None
    return timestamp.tz_convert("UTC")


def _normalise_known_from(value: object, *, naive_timezone: str) -> object:
    timestamp = _as_utc_timestamp(value, naive_timezone=naive_timezone)
    if timestamp is None:
        return pd.NA
    return timestamp.isoformat().replace("+00:00", "Z")


def normalise_pit_intervals(
    pit_intervals: pd.DataFrame,
    *,
    known_from_timezone: str = "UTC",
) -> pd.DataFrame:
    """Return PIT intervals with normalised dates and optional knowledge time.

    Legacy effective-date-only inputs remain valid. ``known_from`` is preserved
    only when supplied; it is never inferred from ``valid_from``. Naive
    ``known_from`` values use the explicitly supplied ``known_from_timezone``
    (UTC by default), and timezone-aware values are converted to canonical UTC.
    """
    frame = pit_intervals.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "constituent_ric" in frame.columns and "kdcode" not in frame.columns:
        frame = frame.rename(columns={"constituent_ric": "kdcode"})
    required = {"kdcode", "valid_from", "valid_to"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PIT intervals missing columns: {sorted(missing)}")
    columns = ["kdcode", "valid_from", "valid_to"]
    if "known_from" in frame.columns:
        columns.append("known_from")
    frame = frame[columns].copy()
    frame = frame.dropna(subset=["kdcode", "valid_from", "valid_to"])
    frame["kdcode"] = frame["kdcode"].astype(str).str.strip()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"]).dt.strftime("%Y-%m-%d")
    frame["valid_to"] = pd.to_datetime(frame["valid_to"]).dt.strftime("%Y-%m-%d")
    if "known_from" in frame.columns:
        frame["known_from"] = frame["known_from"].map(
            lambda value: _normalise_known_from(
                value,
                naive_timezone=known_from_timezone,
            )
        )
    return frame[frame["kdcode"] != ""].reset_index(drop=True)


def classify_pit_knowledge_as_of(
    pit_intervals: pd.DataFrame,
    signal_close: str | pd.Timestamp,
    *,
    known_from_timezone: str = "UTC",
) -> PITKnowledgeClass:
    """Classify active PIT membership evidence as known at ``signal_close``.

    A legacy input without a ``known_from`` column is ``EFFECTIVE_ONLY``. When
    the column is supplied, every interval active on the signal date must have
    a valid timestamp no later than the signal close to be ``KNOWN_AS_OF``.
    Missing, malformed, future-known, or non-active evidence is ``UNKNOWN``.
    """
    try:
        local_signal_close = pd.Timestamp(signal_close)
    except (TypeError, ValueError):
        return PITKnowledgeClass.UNKNOWN
    signal_close_utc = _as_utc_timestamp(local_signal_close)
    if signal_close_utc is None:
        return PITKnowledgeClass.UNKNOWN

    intervals = normalise_pit_intervals(
        pit_intervals,
        known_from_timezone=known_from_timezone,
    )
    signal_date = local_signal_close.strftime("%Y-%m-%d")
    active = intervals[
        (intervals["valid_from"] <= signal_date) & (intervals["valid_to"] >= signal_date)
    ]
    if active.empty:
        return PITKnowledgeClass.UNKNOWN
    if "known_from" not in active.columns:
        return PITKnowledgeClass.EFFECTIVE_ONLY

    known_timestamps = [_as_utc_timestamp(value) for value in active["known_from"]]
    if any(timestamp is None for timestamp in known_timestamps):
        return PITKnowledgeClass.UNKNOWN
    if any(timestamp > signal_close_utc for timestamp in known_timestamps if timestamp is not None):
        return PITKnowledgeClass.UNKNOWN
    return PITKnowledgeClass.KNOWN_AS_OF


def load_pit_intervals(
    csv_path: str, *, input_observations: InputObservationContext | None = None
) -> pd.DataFrame:
    frame = (
        input_observations.read_csv(
            csv_path, role="data.pit_universe_csv", configured_path=csv_path
        )
        if input_observations is not None
        else pd.read_csv(csv_path)
    )
    return normalise_pit_intervals(frame)


def active_kdcodes_in_period(
    pit_intervals: pd.DataFrame,
    start: str,
    end: str,
    available_kdcodes: set[str] | None = None,
) -> list[str]:
    """Return tickers whose PIT interval overlaps ``[start, end]``."""
    intervals = normalise_pit_intervals(pit_intervals)
    mask = (intervals["valid_from"] <= end) & (intervals["valid_to"] >= start)
    values = set(intervals.loc[mask, "kdcode"].astype(str))
    if available_kdcodes is not None:
        values &= {str(k) for k in available_kdcodes}
    return sorted(values)


def active_membership_mask(
    kdcode_list: list[str],
    dates: list[str],
    pit_intervals: pd.DataFrame,
) -> np.ndarray:
    """Boolean ``(dates, stocks)`` membership mask from PIT intervals."""
    intervals = normalise_pit_intervals(pit_intervals)
    by_kdcode: dict[str, list[tuple[str, str]]] = {}
    for row in intervals.itertuples(index=False):
        by_kdcode.setdefault(str(row.kdcode), []).append((row.valid_from, row.valid_to))

    out = np.zeros((len(dates), len(kdcode_list)), dtype=bool)
    for j, kdcode in enumerate(kdcode_list):
        ranges = by_kdcode.get(str(kdcode), [])
        if not ranges:
            continue
        for i, date in enumerate(dates):
            out[i, j] = any(start <= date <= end for start, end in ranges)
    return out


def feature_ready_mask(
    df_for_features: pd.DataFrame,
    kdcode_list: list[str],
    sample_dates: list[str],
    his_t: int,
) -> np.ndarray:
    """True when a stock has a complete pre-sample lookback window."""
    all_dates = sorted(df_for_features["dt"].astype(str).unique())
    date_to_idx = {date: i for i, date in enumerate(all_dates)}
    date_index = {date: idx for idx, date in enumerate(all_dates)}
    stock_index = {kdcode: idx for idx, kdcode in enumerate(kdcode_list)}
    presence = np.zeros((len(all_dates), len(kdcode_list)), dtype=bool)

    subset = df_for_features[["kdcode", "dt"]].drop_duplicates()
    for row in subset.itertuples(index=False):
        kdcode = str(row.kdcode)
        date = str(row.dt)
        if kdcode in stock_index and date in date_index:
            presence[date_index[date], stock_index[kdcode]] = True

    out = np.zeros((len(sample_dates), len(kdcode_list)), dtype=bool)
    for i, date in enumerate(sample_dates):
        date = str(date)
        end_idx = date_to_idx.get(date)
        if end_idx is None or end_idx < his_t:
            continue
        window = presence[end_idx - his_t : end_idx, :]
        out[i, :] = window.all(axis=0)
    return out


def label_available_mask(
    df_for_labels: pd.DataFrame,
    kdcode_list: list[str],
    sample_dates: list[str],
    label_t: int,
) -> np.ndarray:
    """True when the fixed-session label is observable (same resolver as the labels)."""
    return resolve_labels(df_for_labels, kdcode_list, sample_dates, label_t).observable


def build_pit_masks(
    df_for_features: pd.DataFrame,
    df_for_labels: pd.DataFrame,
    kdcode_list: list[str],
    sample_dates: list[str],
    his_t: int,
    label_t: int,
    pit_intervals: pd.DataFrame,
    cessation_excluded: np.ndarray | None = None,
) -> PITMaskSet:
    active = active_membership_mask(kdcode_list, sample_dates, pit_intervals)
    ready = feature_ready_mask(df_for_features, kdcode_list, sample_dates, his_t)
    labels = label_available_mask(df_for_labels, kdcode_list, sample_dates, label_t)
    excluded = (
        np.zeros_like(active)
        if cessation_excluded is None
        else np.asarray(cessation_excluded, dtype=bool)
    )
    if excluded.shape != active.shape:
        raise ValueError(
            f"cessation mask shape {excluded.shape} does not match the PIT axis {active.shape}"
        )
    eligible = active & ~excluded
    tradable = eligible & ready
    loss = tradable & labels
    return PITMaskSet(
        active_member=active,
        feature_ready=ready,
        loss=loss,
        tradable=tradable,
        cessation_excluded=excluded,
        eligible=eligible,
        label_available=labels,
    )


# ── dated cessation eligibility (#225) ───────────────────────────────────

CESSATION_EVENTS_ROLE = "data.pit_cessation_events_csv"
PIT_ELIGIBILITY_SCHEMA = "mci_gru.pit_eligibility.v1"
_CESSATION_REQUIRED_COLUMNS = ("event_id", "kdcode", "effective_at", "known_from")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CLOCK_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class PITEligibilityError(ValueError):
    """The declared cessation evidence cannot support a run; it must stop visibly.

    Raised before any tensor is built. It carries what #223's single run-failure
    report needs: ``role``, ``source``, ``stage``, ``reason``, the named ``kdcodes``
    and the JSON-ready PIT ``fragment``.
    """

    def __init__(
        self,
        reason: str,
        *,
        code: str,
        source: str | None,
        kdcodes: list[str] | tuple[str, ...] = (),
        fragment: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.role = CESSATION_EVENTS_ROLE
        self.stage = "pit_eligibility"
        self.code = code
        self.source = source
        self.reason = reason
        self.kdcodes = tuple(kdcodes)
        self.fragment = fragment


@dataclass(frozen=True)
class PredictionClock:
    """The time of day, in one timezone, at which the forecast for date D is made.

    Production is 20:00 America/New_York (#225 ruling 1); a fixture may declare its own.
    """

    time: str = "20:00"
    timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        if not _CLOCK_TIME.match(str(self.time)):
            raise ValueError(f"prediction clock time must be HH:MM, got {self.time!r}")
        try:
            pd.Timestamp("2000-01-03").tz_localize(self.timezone)
        except Exception as exc:
            raise ValueError(f"unknown prediction clock timezone {self.timezone!r}") from exc

    def at(self, date: str) -> pd.Timestamp:
        """UTC instant of the forecast for ``date``."""
        local = pd.Timestamp(f"{date}T{self.time}:00").tz_localize(
            self.timezone, ambiguous="raise", nonexistent="raise"
        )
        return local.tz_convert("UTC")

    def to_dict(self) -> dict[str, str]:
        return {"time": self.time, "timezone": self.timezone}


@dataclass(frozen=True)
class CessationEvent:
    """One declared cessation, with its evidence kept as supplied.

    ``effective_at`` and ``known_from`` hold the original text. Their precision is
    ``timestamp`` (timezone-aware) or ``date``. ``acquired_at`` and ``evidence`` are
    retained for the report and are never read as availability.
    """

    event_id: str
    kdcode: str
    effective_at: str
    effective_precision: str
    known_from: str
    known_precision: str
    acquired_at: str
    evidence: str


@dataclass(frozen=True)
class CessationEvidence:
    """The declared event file, parsed and validated; ``None`` path means undeclared."""

    configured_path: str | None
    events: tuple[CessationEvent, ...]

    @property
    def declared(self) -> bool:
        return self.configured_path is not None


def _precision(value: str) -> str | None:
    """``date`` or ``timestamp`` for a well-formed value, ``None`` when malformed."""
    if _DATE_ONLY.match(value):
        try:
            pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        return "date"
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp) or stamp.tzinfo is None:
        return None
    return "timestamp"


def parse_cessation_events(frame: pd.DataFrame, *, source: str | None) -> CessationEvidence:
    """Validate a cessation event table read as text.

    Stops with ``PITEligibilityError`` on missing columns, a blank or duplicate
    ``event_id``, a blank ``kdcode``, a malformed or timezone-naive timestamp, or a
    cessation without dated evidence (blank ``effective_at`` or ``known_from``),
    naming every affected stock. A date-only value is kept as a date: it is never
    read as midnight.
    """
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in _CESSATION_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise PITEligibilityError(
            f"cessation event file is missing column(s) {missing}",
            code="missing_columns",
            source=source,
        )
    frame = frame.fillna("").astype(str)
    for column in frame.columns:
        frame[column] = frame[column].str.strip()

    blank_kdcode = frame.index[frame["kdcode"] == ""].tolist()
    if blank_kdcode:
        raise PITEligibilityError(
            f"cessation event row(s) {blank_kdcode} have a blank kdcode",
            code="blank_kdcode",
            source=source,
        )
    blank_ids = frame.loc[frame["event_id"] == "", "kdcode"].tolist()
    if blank_ids:
        raise PITEligibilityError(
            f"cessation events for {sorted(set(blank_ids))} have a blank event_id",
            code="blank_event_id",
            source=source,
            kdcodes=sorted(set(blank_ids)),
        )
    duplicated = frame.loc[frame["event_id"].duplicated(keep=False)]
    if not duplicated.empty:
        raise PITEligibilityError(
            "cessation event_id values must be unique; duplicated: "
            f"{sorted(set(duplicated['event_id']))}",
            code="duplicate_event_id",
            source=source,
            kdcodes=sorted(set(duplicated["kdcode"])),
        )

    undated = frame.loc[(frame["effective_at"] == "") | (frame["known_from"] == ""), "kdcode"]
    if not undated.empty:
        names = sorted(set(undated))
        raise PITEligibilityError(
            f"cessation without dated evidence for {names}: every declared cessation needs "
            "an effective_at and a known_from date or timestamp; the run cannot decide "
            "when these stocks stopped being eligible",
            code="undated_cessation",
            source=source,
            kdcodes=names,
        )

    events: list[CessationEvent] = []
    malformed: list[str] = []
    for row in frame.itertuples(index=False):
        effective_precision = _precision(row.effective_at)
        known_precision = _precision(row.known_from)
        if effective_precision is None or known_precision is None:
            malformed.append(str(row.kdcode))
            continue
        events.append(
            CessationEvent(
                event_id=str(row.event_id),
                kdcode=str(row.kdcode),
                effective_at=str(row.effective_at),
                effective_precision=effective_precision,
                known_from=str(row.known_from),
                known_precision=known_precision,
                acquired_at=str(getattr(row, "acquired_at", "")),
                evidence=str(getattr(row, "evidence", "")),
            )
        )
    if malformed:
        names = sorted(set(malformed))
        raise PITEligibilityError(
            f"cessation events for {names} have a malformed effective_at or known_from: "
            "use YYYY-MM-DD or a timezone-aware ISO timestamp",
            code="malformed_timestamp",
            source=source,
            kdcodes=names,
        )
    return CessationEvidence(configured_path=source, events=tuple(events))


def load_cessation_events(
    csv_path: str | None,
    *,
    input_observations: InputObservationContext | None = None,
) -> CessationEvidence:
    """Read the declared event file through the input carrier, as text.

    No path means no cessation evidence was declared, which excludes nothing.
    A missing or unreadable file stops the run (the settled single-source rule).
    """
    if not csv_path:
        return CessationEvidence(configured_path=None, events=())

    def _parse(content: bytes) -> pd.DataFrame:
        return pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)

    if input_observations is not None:
        frame = input_observations.read_file(
            csv_path,
            role=CESSATION_EVENTS_ROLE,
            configured_path=csv_path,
            parse=_parse,
            parser={
                "name": "pandas.read_csv",
                "options": {"dtype": "str", "keep_default_na": False},
            },
        )
    else:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return parse_cessation_events(frame, source=csv_path)


def _resolve_instant(
    value: str,
    precision: str,
    *,
    kind: str,
    sessions: list[str],
    clock: PredictionClock,
) -> pd.Timestamp | None:
    """UTC instant an event fact holds from; ``None`` when after the panel ends.

    * ``timestamp``: as supplied.
    * ``date`` for ``known``: the prediction clock on the first panel session after
      that date (#225 ruling 2), so evidence dated D is first usable for D+1.
    * ``date`` for ``effective``: the start of that date in the clock's timezone, so a
      cessation dated E is in effect for the forecast made on E.
    """
    if precision == "timestamp":
        return pd.Timestamp(value).tz_convert("UTC")
    if kind == "known":
        later = [session for session in sessions if session > value]
        return clock.at(later[0]) if later else None
    return pd.Timestamp(value).tz_localize(clock.timezone).tz_convert("UTC")


@dataclass(frozen=True)
class ResolvedCessation:
    event: CessationEvent
    effective_utc: pd.Timestamp | None
    known_utc: pd.Timestamp | None


def resolve_cessation_events(
    evidence: CessationEvidence,
    sessions: list[str],
    clock: PredictionClock,
) -> list[ResolvedCessation]:
    return [
        ResolvedCessation(
            event=event,
            effective_utc=_resolve_instant(
                event.effective_at,
                event.effective_precision,
                kind="effective",
                sessions=sessions,
                clock=clock,
            ),
            known_utc=_resolve_instant(
                event.known_from,
                event.known_precision,
                kind="known",
                sessions=sessions,
                clock=clock,
            ),
        )
        for event in evidence.events
    ]


def cessation_exclusion_mask(
    resolved: list[ResolvedCessation],
    kdcode_list: list[str],
    sample_dates: list[str],
    clock: PredictionClock,
) -> np.ndarray:
    """True where a cessation is both effective and known by the forecast for D.

    The comparison is ``<=``: a fact holding exactly at the prediction instant counts.
    Acquisition time is never consulted, and future labels are never read.
    """
    out = np.zeros((len(sample_dates), len(kdcode_list)), dtype=bool)
    stock_index = {kdcode: j for j, kdcode in enumerate(kdcode_list)}
    predictions = [clock.at(str(date)) for date in sample_dates]
    for item in resolved:
        j = stock_index.get(item.event.kdcode)
        if j is None or item.effective_utc is None or item.known_utc is None:
            continue
        holds_from = max(item.effective_utc, item.known_utc)
        for i, prediction in enumerate(predictions):
            if holds_from <= prediction:
                out[i, j] = True
    return out


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def _omitted_reason(entry_date, exit_date, entry_close: float, exit_close: float) -> str:
    if entry_date is None or exit_date is None:
        return "endpoint_after_panel_end"
    entry_missing = not np.isfinite(entry_close)
    exit_missing = not np.isfinite(exit_close)
    if entry_missing and exit_missing:
        return "entry_and_exit_close_missing"
    if entry_missing:
        return "entry_close_missing"
    if exit_missing:
        return "exit_close_missing"
    return "non_finite_return"


def pit_split_report(
    dates: list[str],
    kdcode_list: list[str],
    masks: PITMaskSet,
    labels: LabelResolution,
) -> dict[str, Any]:
    """Daily populations, coverage totals and every omitted label for one split.

    Every prediction (``tradable``) is kept. One whose label is unobservable is
    listed with its endpoints and reason and counted as omitted (#225 ruling 3).
    """
    assert masks.cessation_excluded is not None and masks.eligible is not None
    omitted = masks.tradable & ~masks.loss
    daily = [
        {
            "date": str(date),
            "selected": int(masks.active_member[i].sum()),
            "cessation_excluded": int((masks.active_member[i] & masks.cessation_excluded[i]).sum()),
            "eligible": int(masks.eligible[i].sum()),
            "feature_ready": int(masks.feature_ready[i].sum()),
            "predictions": int(masks.tradable[i].sum()),
            "label_observable": int(masks.loss[i].sum()),
            "label_omitted": int(omitted[i].sum()),
        }
        for i, date in enumerate(dates)
    ]
    counts = (
        "selected",
        "cessation_excluded",
        "eligible",
        "feature_ready",
        "predictions",
        "label_observable",
        "label_omitted",
    )
    totals = {key: sum(int(row[key]) for row in daily) for key in counts}
    omitted_rows = []
    for i, j in zip(*np.nonzero(omitted), strict=True):
        omitted_rows.append(
            {
                "date": str(dates[i]),
                "kdcode": str(kdcode_list[j]),
                "entry_date": labels.entry_dates[i],
                "exit_date": labels.exit_dates[i],
                "reason": _omitted_reason(
                    labels.entry_dates[i],
                    labels.exit_dates[i],
                    float(labels.entry_close[i, j]),
                    float(labels.exit_close[i, j]),
                ),
            }
        )
    return {
        "daily": daily,
        "totals": totals,
        "label_coverage": {
            "numerator": "label_observable",
            "denominator": "predictions",
            "value": (
                totals["label_observable"] / totals["predictions"]
                if totals["predictions"]
                else None
            ),
        },
        "omitted_labels": omitted_rows,
    }


def pit_eligibility_fragment(
    *,
    clock: PredictionClock,
    evidence: CessationEvidence,
    resolved: list[ResolvedCessation] | None,
    label_t: int,
    splits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The JSON-ready PIT fragment #223's report embeds (schema ``pit_eligibility.v1``)."""
    by_id = {item.event.event_id: item for item in resolved or []}
    events = []
    for event in evidence.events:
        item = by_id.get(event.event_id)
        events.append(
            {
                "event_id": event.event_id,
                "kdcode": event.kdcode,
                "effective_at": event.effective_at,
                "effective_precision": event.effective_precision,
                "effective_utc": _iso(item.effective_utc) if item else None,
                "known_from": event.known_from,
                "known_precision": event.known_precision,
                "known_utc": _iso(item.known_utc) if item else None,
                "acquired_at": event.acquired_at or None,
                "evidence": event.evidence or None,
            }
        )
    return {
        "schema": PIT_ELIGIBILITY_SCHEMA,
        "prediction_clock": clock.to_dict(),
        "label_endpoints": {
            "rule": "fixed_sessions",
            "session_axis": "panel trading dates of the selected union",
            "entry_session_offset": 1,
            "exit_session_offset": label_t,
            "close_to_close_intervals": label_t - 1,
            "missing_endpoint": "unobservable; no fill or next-available substitute",
        },
        "cessation_events": {
            "declared": evidence.declared,
            "configured_path": evidence.configured_path,
            "events": events,
        },
        "claim_scope": (
            "Training and evaluation use observable labels only. No complete "
            "economic-return claim is made until #129 values terminal outcomes."
        ),
        "splits": splits or {},
    }


def apply_label_mask(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.float32).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    return out


def candidate_breadth(dates: list[str], tradable_mask: np.ndarray) -> list[dict[str, int | str]]:
    mask = np.asarray(tradable_mask, dtype=bool)
    return [
        {"date": str(date), "scoreable_count": int(mask[i].sum())} for i, date in enumerate(dates)
    ]


def filter_edges_by_stock_mask(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    stock_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove edges whose source or destination node is inactive."""
    if edge_index.numel() == 0:
        return edge_index, edge_weight
    mask = stock_mask.to(dtype=torch.bool, device=edge_index.device)
    edge_keep = mask[edge_index[0]] & mask[edge_index[1]]
    return edge_index[:, edge_keep], edge_weight[edge_keep]
