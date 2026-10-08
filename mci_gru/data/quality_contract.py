"""Admission of required inputs before any training work (#223).

Validation inspects the rows a native reader returned, before any transform,
and records verdicts in an :class:`AdmissionLedger`. Preparation enforces the
ledger at fixed checkpoints: a required item whose verdict is not ``valid``
stops preparation with an :class:`AdmissionError`, and the runner writes that
error to ``run_failure.json``.

Nothing here repairs a source. Rows are never filled, deduplicated, dropped or
substituted; findings only describe them. See
``docs/agents/data-quality-contract.md`` for the rules and their rulings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mci_gru.data.input_observations import InputObservationError, InputObservations

if TYPE_CHECKING:
    from collections.abc import Collection

    from mci_gru.data.input_observations import InputObservationContext

logger = logging.getLogger(__name__)

RUN_FAILURE_SCHEMA = "mci_gru.run_failure.v1"
RUN_FAILURE_FILENAME = "run_failure.json"
ADMISSION_SCHEMA = "mci_gru.admission.v1"

MARKET_COLUMNS = ("kdcode", "dt", "open", "high", "low", "close", "volume")
OHLC_FIELDS = ("open", "high", "low", "close")
DAILY_DATE_FORMAT = "%Y-%m-%d"
# Offending keys listed per finding; the counts are always complete.
EVIDENCE_LIMIT = 20


class Verdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class AdmissionItem:
    """One rule's verdict on one required input role."""

    role: str
    rule: str
    verdict: Verdict
    reason_code: str
    reason: str
    stage: str = "validate"
    source: str = "file"
    configured_path: str | None = None
    required: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.required and self.verdict is not Verdict.VALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source": self.source,
            "configured_path": self.configured_path,
            "stage": self.stage,
            "rule": self.rule,
            "verdict": self.verdict.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "required": self.required,
            "evidence": self.evidence,
        }


class AdmissionError(ValueError):
    """Preparation stopped on required items that are not admitted."""

    def __init__(
        self,
        failures: list[AdmissionItem],
        *,
        input_observations: InputObservations | None = None,
        admission: dict[str, Any] | None = None,
    ) -> None:
        self.failures = list(failures)
        self.input_observations = input_observations
        self.admission = admission
        summary = "; ".join(
            f"{item.role} {item.stage}/{item.reason_code}" for item in self.failures[:5]
        )
        more = f" (+{len(self.failures) - 5} more)" if len(self.failures) > 5 else ""
        super().__init__(f"Input admission failed: {summary}{more}")


class AdmissionLedger:
    """Verdicts and coverage counts for one preparation window."""

    def __init__(self) -> None:
        self._items: list[AdmissionItem] = []
        self._coverage: dict[str, Any] = {}

    def record(self, item: AdmissionItem) -> None:
        self._items.append(item)

    def record_all(self, items: list[AdmissionItem]) -> None:
        self._items.extend(items)

    def record_coverage(self, role: str, coverage: dict[str, Any]) -> None:
        self._coverage[role] = coverage

    @property
    def items(self) -> tuple[AdmissionItem, ...]:
        return tuple(self._items)

    def blocking(self) -> list[AdmissionItem]:
        return [item for item in self._items if item.blocks]

    def to_dict(self) -> dict[str, Any]:
        blocking = self.blocking()
        return {
            "schema": ADMISSION_SCHEMA,
            "admitted": not blocking,
            "items": [item.to_dict() for item in self._items],
            "coverage": self._coverage,
        }

    def require_admitted(self, input_observations: InputObservationContext | None = None) -> None:
        """Raise when any required item is not admitted; the window's observations seal."""
        blocking = self.blocking()
        if not blocking:
            return
        raise AdmissionError(
            blocking,
            input_observations=input_observations.freeze()
            if input_observations is not None
            else None,
            admission=self.to_dict(),
        )


def input_failure(
    *,
    role: str,
    configured_path: str | None,
    stage: str,
    reason_code: str,
    reason: str,
    source: str = "file",
) -> AdmissionItem:
    """A selected required input that could not be resolved, read or parsed."""
    return AdmissionItem(
        role=role,
        rule="selected_input_readable",
        verdict=Verdict.INVALID,
        reason_code=reason_code,
        reason=reason,
        stage=stage,
        source=source,
        configured_path=configured_path,
    )


# ── market panel ─────────────────────────────────────────────────────────


def _blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype(str).str.strip().eq("")


def _keys(frame: pd.DataFrame, mask: pd.Series) -> list[dict[str, Any]]:
    rows = frame.loc[mask, ["kdcode", "dt"]].head(EVIDENCE_LIMIT)
    return [
        {"row": int(index), "kdcode": str(row.kdcode), "dt": str(row.dt)}
        for index, row in zip(rows.index, rows.itertuples(index=False), strict=True)
    ]


def assess_market_panel(
    frame: pd.DataFrame,
    *,
    role: str,
    configured_path: str | None,
) -> tuple[list[AdmissionItem], dict[str, Any]]:
    """Assess native-read market rows against the settled #193 rules.

    Blank price cells are genuine missingness: they are counted, never failed.
    Returns the findings and the per-stock missing-price coverage. The frame
    is not modified.
    """

    def finding(rule: str, verdict: Verdict, code: str, reason: str, **evidence: Any):
        return AdmissionItem(
            role=role,
            rule=rule,
            verdict=verdict,
            reason_code=code,
            reason=reason,
            configured_path=configured_path,
            evidence=evidence,
        )

    missing = [column for column in MARKET_COLUMNS if column not in frame.columns]
    if missing:
        return [
            finding(
                "market_structure",
                Verdict.INVALID,
                "missing_columns",
                f"Market panel is missing required columns {missing}",
                missing_columns=missing,
            )
        ], {}

    findings: list[AdmissionItem] = []
    blank_id = _blank(frame["kdcode"])
    dates = pd.to_datetime(frame["dt"].astype(str), format=DAILY_DATE_FORMAT, errors="coerce")
    bad_date = dates.isna()
    for mask, code, reason in (
        (blank_id, "blank_kdcode", "Rows with a blank stock identifier"),
        (bad_date, "invalid_date", f"Rows whose dt is not a {DAILY_DATE_FORMAT} date"),
    ):
        if mask.any():
            findings.append(
                finding(
                    "market_structure",
                    Verdict.INVALID,
                    code,
                    reason,
                    count=int(mask.sum()),
                    rows=_keys(frame, mask),
                )
            )

    keyed = ~blank_id & ~bad_date
    kdcode = frame["kdcode"].astype(str).str.strip()
    duplicate = keyed & pd.DataFrame({"k": kdcode, "d": dates}).duplicated(keep=False)
    if duplicate.any():
        findings.append(
            finding(
                "market_structure",
                Verdict.INVALID,
                "duplicate_stock_session",
                "Rows repeating one stock/session key; duplicates are rejected, not deduplicated",
                count=int(duplicate.sum()),
                rows=_keys(frame, duplicate),
            )
        )

    numeric: dict[str, pd.Series] = {}
    for column in (*OHLC_FIELDS, "volume"):
        original = frame[column]
        values = pd.to_numeric(original, errors="coerce")
        numeric[column] = values
        unparsed = ~_blank(original) & values.isna()
        if column == "volume":
            out_of_range = values.notna() & ~(np.isfinite(values) & (values >= 0))
            bound = "finite and nonnegative"
        else:
            out_of_range = values.notna() & ~(np.isfinite(values) & (values > 0))
            bound = "finite and positive"
        for mask, code, reason in (
            (unparsed, "non_numeric", f"Non-numeric {column} values"),
            (out_of_range, "out_of_range", f"Observed {column} values must be {bound}"),
        ):
            if mask.any():
                findings.append(
                    finding(
                        "market_structure",
                        Verdict.INVALID,
                        f"{code}_{column}",
                        reason,
                        field=column,
                        count=int(mask.sum()),
                        rows=_keys(frame, mask),
                    )
                )

    findings.extend(_frozen_history_findings(kdcode[keyed], dates[keyed], numeric, keyed, finding))
    coverage = _missing_price_coverage(kdcode[keyed], dates[keyed], numeric, keyed)
    return findings, coverage


def _frozen_history_findings(kdcode, dates, numeric, keyed, finding) -> list[AdmissionItem]:
    """Per-stock whole-history rule: all four OHLC fields constant with n>=2."""
    stats: dict[str, pd.DataFrame] = {}
    stocks = pd.Index(sorted(kdcode.unique()))
    for column in OHLC_FIELDS:
        values = numeric[column][keyed]
        finite = values.notna() & np.isfinite(values)
        observed = pd.DataFrame(
            {"kdcode": kdcode[finite], "dt": dates[finite], "v": values[finite]}
        )
        grouped = observed.groupby("kdcode").agg(n=("dt", "nunique"), u=("v", "nunique"))
        stats[column] = grouped.reindex(stocks, fill_value=0)

    n = pd.DataFrame({column: stats[column]["n"] for column in OHLC_FIELDS})
    u = pd.DataFrame({column: stats[column]["u"] for column in OHLC_FIELDS})
    assessable = (n >= 2).all(axis=1)
    frozen = assessable & (u == 1).all(axis=1)
    short = ~assessable

    def per_stock(mask: pd.Series) -> list[dict[str, Any]]:
        return [
            {
                "kdcode": str(stock),
                **{f"{column}_n": int(n.at[stock, column]) for column in OHLC_FIELDS},
                **{f"{column}_u": int(u.at[stock, column]) for column in OHLC_FIELDS},
            }
            for stock in mask[mask].index[:EVIDENCE_LIMIT]
        ]

    findings = []
    if frozen.any():
        findings.append(
            finding(
                "frozen_history",
                Verdict.INVALID,
                "frozen_ohlc_history",
                "Stocks whose four OHLC histories are each constant across their whole history",
                count=int(frozen.sum()),
                stocks=per_stock(frozen),
            )
        )
    if short.any():
        findings.append(
            finding(
                "frozen_history",
                Verdict.INSUFFICIENT_EVIDENCE,
                "fewer_than_two_observations",
                "Stocks with fewer than 2 dated finite observations in an OHLC field",
                count=int(short.sum()),
                stocks=per_stock(short),
            )
        )
    # A constant field alone does not invalidate a history; it is reported.
    mixed = assessable & ~frozen & (u == 1).any(axis=1)
    if mixed.any():
        findings.append(
            finding(
                "frozen_history",
                Verdict.VALID,
                "constant_field_reported",
                "Stocks with at least one, but not all four, constant OHLC fields",
                count=int(mixed.sum()),
                stocks=per_stock(mixed),
            )
        )
    return findings


def _missing_price_coverage(kdcode, dates, numeric, keyed) -> dict[str, Any]:
    """Count genuine price gaps per stock; counts decide nothing on their own."""
    blank_price = pd.Series(False, index=kdcode.index)
    for column in OHLC_FIELDS:
        blank_price |= numeric[column][keyed].isna()
    calendar = np.sort(dates.unique())
    frame = pd.DataFrame({"kdcode": kdcode, "dt": dates, "blank": blank_price})
    grouped = frame.groupby("kdcode").agg(
        rows=("dt", "size"), first=("dt", "min"), last=("dt", "max"), blank=("blank", "sum")
    )
    span = np.searchsorted(calendar, grouped["last"].to_numpy(), side="right") - np.searchsorted(
        calendar, grouped["first"].to_numpy(), side="left"
    )
    grouped["absent"] = span - grouped["rows"]
    gaps = grouped[(grouped["blank"] > 0) | (grouped["absent"] > 0)]
    return {
        "stocks": int(len(grouped)),
        "sessions": int(len(calendar)),
        "rows_with_missing_price": int(grouped["blank"].sum()),
        "absent_sessions_within_history": int(grouped["absent"].sum()),
        "per_stock": {
            str(stock): {
                "rows_with_missing_price": int(row.blank),
                "absent_sessions_within_history": int(row.absent),
            }
            for stock, row in gaps.iterrows()
        },
    }


# ── PIT universe ─────────────────────────────────────────────────────────


def assess_pit_intervals(
    frame: pd.DataFrame,
    *,
    configured_path: str | None,
    open_valid_to: str | None,
) -> list[AdmissionItem]:
    """Assess native-read PIT intervals: malformed rows, open ends, inversions, overlaps.

    A blank ``valid_to`` is an open interval through ``open_valid_to`` (the
    declared export cutoff); with no cutoff declared it is not admissible.
    """
    role = "data.pit_universe_csv"

    def finding(code: str, reason: str, **evidence: Any) -> AdmissionItem:
        return AdmissionItem(
            role=role,
            rule="pit_interval_structure",
            verdict=Verdict.INVALID,
            reason_code=code,
            reason=reason,
            configured_path=configured_path,
            evidence=evidence,
        )

    columns = {str(c).strip().lower(): c for c in frame.columns}
    if "constituent_ric" in columns and "kdcode" not in columns:
        columns["kdcode"] = columns.pop("constituent_ric")
    missing = sorted({"kdcode", "valid_from", "valid_to"} - set(columns))
    if missing:
        return [finding("missing_columns", f"PIT intervals missing columns {missing}")]

    kdcode = frame[columns["kdcode"]]
    start = frame[columns["valid_from"]]
    end = frame[columns["valid_to"]]

    def rows(mask: pd.Series) -> list[dict[str, Any]]:
        return [
            {
                "row": int(i),
                "kdcode": str(kdcode.loc[i]),
                "valid_from": str(start.loc[i]),
                "valid_to": str(end.loc[i]),
            }
            for i in mask[mask].index[:EVIDENCE_LIMIT]
        ]

    findings = []
    blank_end = _blank(end)
    start_dates = pd.to_datetime(start.astype(str), format=DAILY_DATE_FORMAT, errors="coerce")
    end_dates = pd.to_datetime(end.astype(str), format=DAILY_DATE_FORMAT, errors="coerce")
    if open_valid_to is not None:
        end_dates = end_dates.mask(blank_end, pd.Timestamp(open_valid_to))
    checks = [
        (_blank(kdcode), "blank_kdcode", "Intervals with a blank stock identifier"),
        (start_dates.isna(), "invalid_valid_from", "Intervals whose valid_from is not a date"),
        (
            end_dates.isna() & ~blank_end,
            "invalid_valid_to",
            "Intervals whose valid_to is not a date",
        ),
    ]
    if open_valid_to is None:
        checks.append(
            (
                blank_end,
                "open_interval_without_cutoff",
                "Intervals with a blank valid_to, and no data.pit_export_cutoff declared",
            )
        )
    for mask, code, reason in checks:
        if mask.any():
            findings.append(finding(code, reason, count=int(mask.sum()), rows=rows(mask)))
    if findings:
        return findings

    inverted = start_dates > end_dates
    if inverted.any():
        findings.append(
            finding(
                "inverted_interval",
                "Intervals whose valid_from is after valid_to",
                count=int(inverted.sum()),
                rows=rows(inverted),
            )
        )

    ordered = pd.DataFrame(
        {"kdcode": kdcode.astype(str).str.strip(), "start": start_dates, "end": end_dates}
    ).sort_values(["kdcode", "start", "end"])
    reach = ordered.groupby("kdcode")["end"].cummax().groupby(ordered["kdcode"]).shift()
    overlapping = ordered["start"] <= reach
    if overlapping.any():
        names = sorted(ordered.loc[overlapping, "kdcode"].unique())
        findings.append(
            finding(
                "overlapping_intervals",
                "Stocks with membership intervals that share at least one date",
                count=len(names),
                kdcodes=names[:EVIDENCE_LIMIT],
                rows=rows(
                    pd.Series(frame.index.isin(ordered.index[overlapping]), index=frame.index)
                ),
            )
        )
    return findings


def fill_open_valid_to(frame: pd.DataFrame, open_valid_to: str | None) -> pd.DataFrame:
    """Return a copy whose blank ``valid_to`` cells carry the declared export cutoff."""
    if open_valid_to is None:
        return frame
    out = frame.copy()
    column = next(c for c in out.columns if str(c).strip().lower() == "valid_to")
    out[column] = out[column].mask(_blank(out[column]), open_valid_to)
    return out


def assess_pit_panel_coverage(
    intervals: pd.DataFrame,
    panel_kdcodes: set[str],
    start: str,
    end: str,
    *,
    configured_path: str | None,
    declared_absent: Collection[str] = (),
) -> list[AdmissionItem]:
    """PIT members in the experiment period must have panel rows; none are dropped.

    ``declared_absent`` names members the panel declares it does not carry
    (``data.pit_absent_kdcodes``). Their missing rows are admitted and recorded;
    a declaration that is not true of this panel, because the name has rows or
    no membership in the period, stops the run.
    """
    active = intervals[(intervals["valid_from"] <= end) & (intervals["valid_to"] >= start)]
    active_names = set(active["kdcode"].astype(str))
    panel = {str(k) for k in panel_kdcodes}
    declared = {str(k) for k in declared_absent}
    absent = active_names - panel
    undeclared = sorted(absent - declared)
    stale = sorted(declared - absent)
    items = []
    if undeclared:
        items.append(
            AdmissionItem(
                role="data.pit_universe_csv",
                rule="pit_panel_coverage",
                verdict=Verdict.INVALID,
                reason_code="pit_names_without_panel_rows",
                reason="PIT members in the experiment period with no rows in the market panel",
                configured_path=configured_path,
                evidence={"count": len(undeclared), "kdcodes": undeclared},
            )
        )
    if stale:
        items.append(
            AdmissionItem(
                role="data.pit_universe_csv",
                rule="pit_panel_coverage",
                verdict=Verdict.INVALID,
                reason_code="pit_declared_absent_not_absent",
                reason=(
                    "Names declared absent from the panel that have panel rows, or no "
                    "PIT membership in the experiment period"
                ),
                configured_path=configured_path,
                evidence={
                    "count": len(stale),
                    "kdcodes": stale,
                    "with_panel_rows": sorted(set(stale) & panel),
                },
            )
        )
    if declared & absent:
        items.append(
            AdmissionItem(
                role="data.pit_universe_csv",
                rule="pit_panel_coverage",
                verdict=Verdict.VALID,
                reason_code="pit_names_declared_absent",
                reason="PIT members the panel declares it does not carry; left off the stock axis",
                configured_path=configured_path,
                evidence={"count": len(declared & absent), "kdcodes": sorted(declared & absent)},
            )
        )
    return items


def breadth_floor_failure(
    split_name: str, low: list[dict[str, int | str]], min_scoreable: int, reason: str
) -> AdmissionItem:
    """The session floor (``pit_min_scoreable_stocks``, ``pit_breadth_policy: error``) failed."""
    return AdmissionItem(
        role="data.pit_universe_csv",
        rule="session_breadth_floor",
        verdict=Verdict.INVALID,
        reason_code="breadth_below_floor",
        reason=reason,
        stage="admit",
        evidence={
            "split": split_name,
            "min_scoreable_stocks": min_scoreable,
            "count": len(low),
            "dates": low[:EVIDENCE_LIMIT],
        },
    )


# ── run failure report ───────────────────────────────────────────────────


def _failure_items(error: Exception) -> list[dict[str, Any]]:
    if isinstance(error, AdmissionError):
        return [item.to_dict() for item in error.failures]
    facts = dict(getattr(error, "facts", {}) or {})
    return [
        {
            "role": facts.pop("role", None),
            "source": facts.pop("source", None),
            "configured_path": facts.pop("configured_path", None),
            "stage": facts.pop("stage", None),
            "rule": "selected_input_readable",
            "verdict": Verdict.INVALID.value,
            "reason_code": facts.pop("reason_code", type(error).__name__),
            "reason": facts.pop("reason", str(error)),
            "required": facts.pop("required", True),
            "evidence": facts,
        }
    ]


def build_run_failure(
    error: AdmissionError | InputObservationError,
    *,
    walkforward_window: int,
    resolved_config_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Serialize only the evidence the error carries; nothing is re-read."""
    observations = getattr(error, "input_observations", None)
    return {
        "schema": RUN_FAILURE_SCHEMA,
        "outcome": "failed",
        "stage": "preparation",
        "walkforward_window": walkforward_window,
        "resolved_config": resolved_config_identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "failures": _failure_items(error),
        "admission": getattr(error, "admission", None),
        "input_observations": observations.to_dict()
        if isinstance(observations, InputObservations)
        else None,
    }


def write_run_failure(
    error: AdmissionError | InputObservationError,
    output_dir: str | Path,
    *,
    walkforward_window: int,
    resolved_config_identity: dict[str, Any] | None,
) -> Path | None:
    """Write ``run_failure.json`` and log role/source/stage/reason for the owner.

    A failure to write the report is logged and returns ``None``; the caller
    still re-raises the preparation failure, so the run stays unsuccessful.
    """
    report = build_run_failure(
        error,
        walkforward_window=walkforward_window,
        resolved_config_identity=resolved_config_identity,
    )
    for item in report["failures"]:
        logger.error(
            "Preparation stopped: role=%s source=%s stage=%s reason=%s (%s)",
            item["role"],
            item["source"],
            item["stage"],
            item["reason_code"],
            item["reason"],
        )
    path = Path(output_dir) / RUN_FAILURE_FILENAME
    try:
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logger.error("Run failure report could not be saved to %s: %s", path, exc)
        return None
    logger.error("Run failure report saved to: %s", path)
    return path
