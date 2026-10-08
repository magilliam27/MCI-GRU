"""EODHD daily stock prices for a point-in-time universe (#281).

Pure functions only: no network access and no credentials. Acquisition lives in
``scripts/data/export_eodhd_pit_prices.py``, which feeds vendor responses
through these functions and writes the package.

What this module decides:

- **Which EODHD symbol carries each panel identifier.** Panel identifiers stay
  the LSEG RICs of the point-in-time universe file, so the same membership file
  masks an EODHD panel unchanged. Most RICs map by rule (``AAPL.OQ`` to
  ``AAPL.US``, ``BRKb.N`` to ``BRK-B.US``, ``ATVI.OQ^J23`` to ``ATVI.US``). A
  reviewed symbol map overrides the rest. These are renamed, reused or
  predecessor tickers, given as dated segments with candidate symbols and an
  optional company-name hint.
- **The adjustment basis.** EODHD's daily open, high, low, close and volume are
  raw. They are split-adjusted here from EODHD's own split records, as of the
  panel end: prices before a split are divided by its ratio and volumes are
  multiplied by it. EODHD's ``adjusted_close`` (splits and dividends) is kept
  only to check the split handling. A wrong split would show as a jump in the
  ratio between the two.
- **Corporate actions the split records miss.** A spin-off changes the price
  basis without a split. The map can declare one per identifier: prices before
  its date are multiplied by a stated factor, or by the step EODHD's
  ``adjusted_close`` takes on that date.
- **How a mapping is proven.** When a reference panel is supplied (the preserved
  LSEG panel), each name's EODHD close-to-close returns must agree with the
  reference's over the span the universe needs. A ticker mix-up cannot pass
  that check.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

PANEL_COLUMNS = ("kdcode", "dt", "open", "high", "low", "close", "volume")
PRICE_FIELDS = ("open", "high", "low", "close")
DATE_FORMAT = "%Y-%m-%d"
SYMBOL_MAP_SCHEMA = 1

#: Calendar days of history a name needs before its first membership window.
#: One year covers the 252-session correlation lookback the configs use.
WARMUP_DAYS = 365
#: Calendar days after the last window, so the final 5-session labels resolve.
LABEL_TAIL_DAYS = 10

#: A day-over-day change in log(adjusted_close / split-adjusted close) beyond
#: this cannot be a dividend. It is a split applied twice or missed (a 3-for-2
#: split is log 1.5 = 0.41).
SPLIT_JUMP_LOG_LIMIT = 0.18
#: A step in that ratio beyond this is larger than an ordinary dividend. It is
#: reported (not blocking) as a possible spin-off or special dividend.
LARGE_ADJUSTMENT_LOG = 0.03

#: Identity and coverage thresholds for agreement with the reference panel.
MIN_RETURN_CORRELATION = 0.95
MAX_MEDIAN_ABS_RETURN_DIFF = 0.002
MAX_MISSING_SHARE = 0.02
#: Days of slack at either end of the needed span (listing-day conventions).
EDGE_SLACK_SESSIONS = 5
#: One day's return may differ from the reference by at most this much, unless
#: the map accepts that date. A spin-off, a missed split or a level gap at a
#: splice shows up here even when the whole-window statistics look fine.
MAX_DAILY_ABS_DIFF = 0.03
#: Below this many return pairs a correlation means little; short windows are
#: judged on coverage and the daily limit alone.
MIN_PAIRS_FOR_CORRELATION = 20
#: Dates listed as evidence, per finding.
EVIDENCE_DATES = 20


class SymbolMapError(ValueError):
    """The symbol map cannot be accepted."""


@dataclass(frozen=True)
class SymbolSegment:
    """One dated stretch of a panel identifier's history and where to read it.

    ``start`` and ``end`` are inclusive ``YYYY-MM-DD`` dates; ``None`` is open.
    ``candidates`` are tried in order; ``name_hint`` adds the vendor's listed
    symbols whose company name contains it (case-insensitive).
    """

    candidates: tuple[str, ...]
    start: str | None = None
    end: str | None = None
    name_hint: str | None = None


@dataclass(frozen=True)
class PriceAdjustment:
    """A corporate action the split records do not carry, such as a spin-off.

    Prices before ``date`` are multiplied by a factor: ``factor`` when stated;
    ``1 - cash / close`` when ``cash`` per share is stated, using the last close
    before the date; otherwise the step EODHD's ``adjusted_close`` takes on that
    date. ``share_conversion`` marks a change in share count, such as a merger
    exchange ratio, so volume before the date is divided by the factor too.
    """

    date: str
    reason: str
    factor: float | None = None
    cash: float | None = None
    share_conversion: bool = False


@dataclass(frozen=True)
class SymbolPlan:
    """Every segment of one panel identifier, in date order, and its declared events."""

    kdcode: str
    segments: tuple[SymbolSegment, ...]
    note: str | None = None
    overridden: bool = False
    adjustments: tuple[PriceAdjustment, ...] = ()
    #: Dates whose one-day difference from the reference is explained and accepted.
    accepted_differences: tuple[str, ...] = ()
    #: Why the vendor has no history for this identifier; such a plan has no segments.
    unavailable: str | None = None


@dataclass
class Finding:
    """One check result. ``blocking`` findings stop the package."""

    kdcode: str | None
    code: str
    blocking: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kdcode": self.kdcode,
            "code": self.code,
            "blocking": self.blocking,
            "detail": self.detail,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

_SHARE_CLASS = re.compile(r"^([A-Z0-9]+?)([a-z])$")


def default_symbol(ric: str) -> str:
    """The EODHD US symbol a RIC maps to when no override names one.

    Drops the delisting suffix (``^J23``) and the exchange code (``.OQ``),
    and writes LSEG's lowercase share-class letter as EODHD's ``-B``.
    """
    base = ric.strip().split("^", 1)[0]
    if not base:
        raise SymbolMapError(f"Cannot derive a symbol from identifier {ric!r}")
    root = base.rsplit(".", 1)[0] if "." in base else base
    match = _SHARE_CLASS.match(root)
    if match:
        root = f"{match.group(1)}-{match.group(2).upper()}"
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]*", root):
        raise SymbolMapError(f"Cannot derive a symbol from identifier {ric!r}")
    return f"{root}.US"


def _date_or_none(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SymbolMapError(f"{where}: dates must be YYYY-MM-DD strings")
    try:
        parsed = pd.to_datetime(value, format=DATE_FORMAT)
    except ValueError as error:
        raise SymbolMapError(f"{where}: {value!r} is not a YYYY-MM-DD date") from error
    if parsed.strftime(DATE_FORMAT) != value:
        raise SymbolMapError(f"{where}: {value!r} is not a YYYY-MM-DD date")
    return value


def parse_symbol_map(payload: Mapping[str, Any]) -> dict[str, SymbolPlan]:
    """Validate a symbol map document and return its overrides by identifier."""
    if payload.get("schema") != SYMBOL_MAP_SCHEMA:
        raise SymbolMapError(f"Symbol map schema must be {SYMBOL_MAP_SCHEMA}")
    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        raise SymbolMapError("Symbol map needs an 'overrides' object")
    plans: dict[str, SymbolPlan] = {}
    for kdcode, entry in overrides.items():
        if not isinstance(entry, dict):
            raise SymbolMapError(f"{kdcode}: must be an object")
        unknown = set(entry) - {
            "segments",
            "note",
            "adjustments",
            "accepted_differences",
            "unavailable",
        }
        if unknown:
            raise SymbolMapError(f"{kdcode}: unknown fields {sorted(unknown)}")
        if "unavailable" in entry:
            plans[kdcode] = _unavailable_plan(kdcode, entry)
            continue
        raw_segments = entry.get("segments", [{"candidates": [default_symbol(kdcode)]}])
        if not isinstance(raw_segments, list):
            raise SymbolMapError(f"{kdcode}: 'segments' must be a list")
        segments = []
        for index, raw in enumerate(raw_segments):
            where = f"{kdcode} segment {index}"
            if not isinstance(raw, dict):
                raise SymbolMapError(f"{where}: must be an object")
            unknown = set(raw) - {"candidates", "start", "end", "name_hint"}
            if unknown:
                raise SymbolMapError(f"{where}: unknown fields {sorted(unknown)}")
            candidates = raw.get("candidates", [])
            if not isinstance(candidates, list) or any(
                not isinstance(c, str) or not c.endswith(".US") for c in candidates
            ):
                raise SymbolMapError(f"{where}: candidates must be '.US' symbols")
            hint = raw.get("name_hint")
            if hint is not None and (not isinstance(hint, str) or not hint.strip()):
                raise SymbolMapError(f"{where}: name_hint must be a nonempty string")
            if not candidates and hint is None:
                raise SymbolMapError(f"{where}: needs candidates or a name_hint")
            segments.append(
                SymbolSegment(
                    candidates=tuple(candidates),
                    start=_date_or_none(raw.get("start"), where),
                    end=_date_or_none(raw.get("end"), where),
                    name_hint=hint,
                )
            )
        _validate_segment_order(kdcode, segments)
        plans[kdcode] = SymbolPlan(
            kdcode,
            tuple(segments),
            note=entry.get("note"),
            overridden=True,
            adjustments=_parse_adjustments(kdcode, entry.get("adjustments", [])),
            accepted_differences=_parse_accepted(kdcode, entry.get("accepted_differences", [])),
        )
    return plans


def _unavailable_plan(kdcode: str, entry: Mapping[str, Any]) -> SymbolPlan:
    """A declared gap: the vendor holds no history for this identifier."""
    reason = entry["unavailable"]
    if not isinstance(reason, str) or not reason.strip():
        raise SymbolMapError(f"{kdcode}: 'unavailable' must be a nonempty reason")
    if set(entry) - {"unavailable", "note"}:
        raise SymbolMapError(f"{kdcode}: an unavailable identifier takes only a note")
    return SymbolPlan(kdcode, (), note=entry.get("note"), overridden=True, unavailable=reason)


def _reason(raw: Mapping[str, Any], where: str) -> str:
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise SymbolMapError(f"{where}: needs a nonempty 'reason'")
    return reason


def _parse_adjustments(kdcode: str, raw_list: Any) -> tuple[PriceAdjustment, ...]:
    if not isinstance(raw_list, list):
        raise SymbolMapError(f"{kdcode}: 'adjustments' must be a list")
    adjustments = []
    for index, raw in enumerate(raw_list):
        where = f"{kdcode} adjustment {index}"
        allowed = {"date", "reason", "factor", "cash", "share_conversion"}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise SymbolMapError(
                f"{where}: allowed fields are date, reason, factor, cash and share_conversion"
            )
        factor = _positive_or_none(raw.get("factor"), f"{where}: factor")
        cash = _positive_or_none(raw.get("cash"), f"{where}: cash")
        if factor is not None and cash is not None:
            raise SymbolMapError(f"{where}: state a factor or a cash amount, not both")
        share_conversion = raw.get("share_conversion", False)
        if not isinstance(share_conversion, bool):
            raise SymbolMapError(f"{where}: share_conversion must be true or false")
        if share_conversion and factor is None:
            raise SymbolMapError(f"{where}: a share conversion needs a stated factor")
        date = _date_or_none(raw.get("date"), where)
        if date is None:
            raise SymbolMapError(f"{where}: needs a date")
        adjustments.append(
            PriceAdjustment(date, _reason(raw, where), factor, cash, share_conversion)
        )
    dates = [a.date for a in adjustments]
    if len(set(dates)) != len(dates):
        raise SymbolMapError(f"{kdcode}: adjustments repeat a date")
    return tuple(sorted(adjustments, key=lambda a: a.date))


def _positive_or_none(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SymbolMapError(f"{where} must be a positive number")
    return float(value)


def _parse_accepted(kdcode: str, raw_list: Any) -> tuple[str, ...]:
    if not isinstance(raw_list, list):
        raise SymbolMapError(f"{kdcode}: 'accepted_differences' must be a list")
    dates = []
    for index, raw in enumerate(raw_list):
        where = f"{kdcode} accepted difference {index}"
        if not isinstance(raw, dict) or set(raw) - {"date", "reason"}:
            raise SymbolMapError(f"{where}: allowed fields are date and reason")
        _reason(raw, where)
        date = _date_or_none(raw.get("date"), where)
        if date is None:
            raise SymbolMapError(f"{where}: needs a date")
        dates.append(date)
    return tuple(sorted(set(dates)))


def _validate_segment_order(kdcode: str, segments: list[SymbolSegment]) -> None:
    if not segments:
        raise SymbolMapError(f"{kdcode}: needs at least one segment")
    for index, segment in enumerate(segments):
        if segment.start and segment.end and segment.start > segment.end:
            raise SymbolMapError(f"{kdcode} segment {index}: start is after end")
        if index > 0 and segment.start is None:
            raise SymbolMapError(f"{kdcode} segment {index}: only the first may be open-start")
        if index < len(segments) - 1 and segment.end is None:
            raise SymbolMapError(f"{kdcode} segment {index}: only the last may be open-ended")
        if index > 0 and segments[index - 1].end >= segment.start:
            raise SymbolMapError(f"{kdcode}: segments {index - 1} and {index} overlap")


def load_symbol_map(path: Path) -> dict[str, SymbolPlan]:
    """Read and validate a symbol map JSON file."""
    return parse_symbol_map(json.loads(path.read_text(encoding="utf-8")))


def build_symbol_plans(
    kdcodes: Iterable[str], overrides: Mapping[str, SymbolPlan]
) -> dict[str, SymbolPlan]:
    """One plan per identifier: its override, else the rule-derived symbol."""
    kdcodes = sorted(set(kdcodes))
    stray = sorted(set(overrides) - set(kdcodes))
    if stray:
        raise SymbolMapError(f"Overrides name identifiers outside the universe: {stray}")
    plans = {}
    for kdcode in kdcodes:
        if kdcode in overrides:
            plans[kdcode] = overrides[kdcode]
        else:
            plans[kdcode] = SymbolPlan(kdcode, (SymbolSegment((default_symbol(kdcode),)),))
    return plans


def name_hint_candidates(
    listing: Iterable[Mapping[str, Any]], hint: str, *, limit: int = 8
) -> list[str]:
    """Symbols from an EODHD exchange listing whose company name contains ``hint``."""
    needle = hint.casefold()
    found = []
    for row in listing:
        name = str(row.get("Name") or "")
        code = str(row.get("Code") or "").strip()
        kind = str(row.get("Type") or "Common Stock")
        if code and needle in name.casefold() and kind == "Common Stock":
            symbol = f"{code}.US"
            if symbol not in found:
                found.append(symbol)
    return found[:limit]


# ---------------------------------------------------------------------------
# Vendor rows and split adjustment
# ---------------------------------------------------------------------------


def parse_split_ratio(text: str) -> float:
    """EODHD's ``"new/old"`` split text as shares after per share before."""
    parts = str(text).split("/")
    if len(parts) != 2:
        raise ValueError(f"Split ratio {text!r} is not 'new/old'")
    new, old = float(parts[0]), float(parts[1])
    if not (math.isfinite(new) and math.isfinite(old)) or new <= 0 or old <= 0:
        raise ValueError(f"Split ratio {text!r} must be two positive numbers")
    return new / old


def eod_rows_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """EODHD ``/eod`` JSON rows as a frame keyed by ``dt`` (raw, unadjusted)."""
    frame = pd.DataFrame(list(rows))
    columns = ["date", *PRICE_FIELDS, "adjusted_close", "volume"]
    if frame.empty:
        empty = pd.DataFrame({"dt": pd.Series(dtype=str)})
        for column in (*PRICE_FIELDS, "adjusted_close", "volume"):
            empty[column] = pd.Series(dtype=float)
        return empty
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"EOD rows are missing fields {missing}")
    frame = frame[columns].rename(columns={"date": "dt"})
    frame["dt"] = pd.to_datetime(frame["dt"], format=DATE_FORMAT).dt.strftime(DATE_FORMAT)
    for column in (*PRICE_FIELDS, "adjusted_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    if frame["dt"].duplicated().any():
        raise ValueError("EOD rows repeat a date")
    return frame.sort_values("dt").reset_index(drop=True)


def splits_frame(rows: Iterable[Mapping[str, Any]], *, as_of: str) -> pd.DataFrame:
    """EODHD ``/splits`` rows on or before ``as_of`` as ``dt, ratio``."""
    records = [
        {"dt": str(row["date"]), "ratio": parse_split_ratio(row["split"])}
        for row in rows
        if str(row["date"]) <= as_of
    ]
    frame = pd.DataFrame(records, columns=["dt", "ratio"])
    if frame["dt"].duplicated().any():
        raise ValueError("Split rows repeat a date")
    return frame.sort_values("dt").reset_index(drop=True)


def split_adjust(eod: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Split-adjust raw OHLCV as of the last split given.

    A split dated D (its first session on the new basis) divides every price
    before D by its ratio and multiplies every volume before D by it.
    """
    out = eod.copy()
    factor = np.ones(len(out))
    dates = out["dt"].to_numpy()
    for split_dt, ratio in zip(splits["dt"], splits["ratio"], strict=True):
        factor[dates < split_dt] *= ratio
    for column in PRICE_FIELDS:
        out[column] = out[column] / factor
    out["volume"] = out["volume"] * factor
    return out


def adjustment_steps(adjusted: pd.DataFrame) -> pd.DataFrame:
    """Per session, the step in adjusted_close / split-adjusted close from the session before.

    ``step`` is the factor that session's vendor adjustment applies to earlier
    prices: below 1 on an ex-dividend date, about 1 on an ordinary day.
    """
    usable = adjusted[(adjusted["close"] > 0) & (adjusted["adjusted_close"] > 0)]
    ratio = (usable["adjusted_close"] / usable["close"]).to_numpy()
    if len(ratio) < 2:
        return pd.DataFrame({"dt": pd.Series(dtype=str), "step": pd.Series(dtype=float)})
    return pd.DataFrame({"dt": usable["dt"].to_numpy()[1:], "step": ratio[:-1] / ratio[1:]})


def split_basis_findings(
    kdcode: str, symbol: str, adjusted: pd.DataFrame, exempt: Iterable[str] = ()
) -> list[Finding]:
    """Steps in adjusted_close / split-adjusted close that no dividend explains.

    A split-sized step blocks: a split is missing or applied twice. A smaller
    step that is still larger than a dividend is reported as a possible
    corporate action. Dates the map already declares are exempt.
    """
    steps = adjustment_steps(adjusted)
    steps = steps[~steps["dt"].isin(set(exempt))]
    size = np.abs(np.log(steps["step"].to_numpy()))
    findings = []
    split_sized = steps[size > SPLIT_JUMP_LOG_LIMIT]
    if len(split_sized):
        findings.append(
            Finding(
                kdcode,
                "split_basis_jump",
                True,
                f"{symbol}: adjusted_close and the split-adjusted close disagree by a "
                "split-sized step; a split is missing from the split records or the raw "
                "prices already carry it",
                {"symbol": symbol, "dates": split_sized["dt"].head(EVIDENCE_DATES).tolist()},
            )
        )
    large = steps[(size > LARGE_ADJUSTMENT_LOG) & (size <= SPLIT_JUMP_LOG_LIMIT)]
    if len(large):
        findings.append(
            Finding(
                kdcode,
                "large_adjustment_step",
                False,
                f"{symbol}: EODHD's adjusted_close steps by more than a dividend on these "
                "dates; a spin-off or special dividend the split-adjusted prices do not carry",
                {
                    "symbol": symbol,
                    "steps": {
                        str(d): round(float(v), 6)
                        for d, v in zip(
                            large["dt"].head(EVIDENCE_DATES),
                            large["step"].head(EVIDENCE_DATES),
                            strict=True,
                        )
                    },
                },
            )
        )
    return findings


def apply_adjustments(
    kdcode: str, frame: pd.DataFrame, adjustments: Iterable[PriceAdjustment]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[Finding]]:
    """Apply the map's declared corporate actions to one identifier's rows.

    Prices before each date are multiplied by its factor. Volume is unchanged
    unless the adjustment is a share conversion. Rows that hold nothing before the
    date, such as a segment that starts on it, need no adjustment. A factor read
    from ``adjusted_close`` needs vendor rows on and before the date; a cash
    factor needs a close before it.
    """
    out = frame.copy()
    applied, findings = [], []
    steps = adjustment_steps(frame).set_index("dt")["step"]
    for adjustment in adjustments:
        before = out["dt"] < adjustment.date
        if not before.any():
            continue
        factor, basis, problem = adjustment.factor, "declared_factor", None
        if factor is None and adjustment.cash is not None:
            basis = "declared_cash"
            closes = out.loc[before & out["close"].notna(), "close"]
            prior = float(closes.iloc[-1]) if len(closes) else np.nan
            factor = 1.0 - adjustment.cash / prior
            if not (math.isfinite(factor) and factor > 0):
                problem = f"No close before {adjustment.date} above the cash amount"
        elif factor is None:
            basis = "vendor_adjusted_close"
            factor = float(steps.get(adjustment.date, np.nan))
            if not math.isfinite(factor):
                problem = f"No adjusted_close step on {adjustment.date} to read the factor from"
        if problem is not None:
            findings.append(
                Finding(
                    kdcode,
                    "adjustment_unresolved",
                    True,
                    problem,
                    {"date": adjustment.date, "reason": adjustment.reason},
                )
            )
            continue
        for column in PRICE_FIELDS:
            out.loc[before, column] = out.loc[before, column] * factor
        if adjustment.share_conversion:
            out.loc[before, "volume"] = out.loc[before, "volume"] / factor
        applied.append(
            {
                "date": adjustment.date,
                "factor": factor,
                "basis": basis,
                "share_conversion": adjustment.share_conversion,
                "reason": adjustment.reason,
            }
        )
    return out, applied, findings


def clean_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Blank nonpositive prices and negative volumes; return the count blanked."""
    out = frame.copy()
    blanked = 0
    for column in PRICE_FIELDS:
        bad = out[column].notna() & ~(np.isfinite(out[column]) & (out[column] > 0))
        blanked += int(bad.sum())
        out.loc[bad, column] = np.nan
    bad = out["volume"].notna() & ~(np.isfinite(out["volume"]) & (out["volume"] >= 0))
    blanked += int(bad.sum())
    out.loc[bad, "volume"] = np.nan
    return out, blanked


def clip_segment(frame: pd.DataFrame, segment: SymbolSegment) -> pd.DataFrame:
    """Rows inside the segment's inclusive date bounds."""
    mask = pd.Series(True, index=frame.index)
    if segment.start:
        mask &= frame["dt"] >= segment.start
    if segment.end:
        mask &= frame["dt"] <= segment.end
    return frame[mask]


def assemble_panel(pieces: Mapping[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Stack each identifier's clipped segments into the long panel format.

    Keeps ``adjusted_close`` for the checks; :func:`panel_for_write` drops it.
    """
    frames = []
    for kdcode, parts in sorted(pieces.items()):
        nonempty = [part for part in parts if not part.empty]
        if not nonempty:
            continue
        joined = pd.concat(nonempty, ignore_index=True)
        if joined["dt"].duplicated().any():
            raise ValueError(f"{kdcode}: segments overlap in the vendor rows")
        joined.insert(0, "kdcode", kdcode)
        frames.append(joined)
    if not frames:
        return pd.DataFrame(columns=[*PANEL_COLUMNS, "adjusted_close"])
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["kdcode", "dt"]).reset_index(drop=True)


def panel_for_write(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel in the CSV loader's column order, sorted by stock then date."""
    return panel[list(PANEL_COLUMNS)].sort_values(["kdcode", "dt"]).reset_index(drop=True)


def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows and first/last date per identifier, as the LSEG export records it."""
    if panel.empty:
        return pd.DataFrame(columns=["kdcode", "row_count", "first_dt", "last_dt"])
    return (
        panel.groupby("kdcode")["dt"]
        .agg(row_count="size", first_dt="min", last_dt="max")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Need and agreement with a reference panel
# ---------------------------------------------------------------------------


def needed_spans(pit: pd.DataFrame) -> pd.DataFrame:
    """Per identifier: one year before its first window to just after its last.

    Blank ``valid_to`` cells must be filled with the export cutoff first.
    """
    frame = pit.copy()
    for column in ("valid_from", "valid_to"):
        blank = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(f"PIT {column} has blank cells; fill them with the export cutoff")
    frame["valid_from"] = pd.to_datetime(frame["valid_from"], format=DATE_FORMAT)
    frame["valid_to"] = pd.to_datetime(frame["valid_to"], format=DATE_FORMAT)
    spans = frame.groupby("kdcode").agg(first=("valid_from", "min"), last=("valid_to", "max"))
    spans["need_start"] = (spans["first"] - pd.Timedelta(days=WARMUP_DAYS)).dt.strftime(DATE_FORMAT)
    spans["need_end"] = (spans["last"] + pd.Timedelta(days=LABEL_TAIL_DAYS)).dt.strftime(
        DATE_FORMAT
    )
    return spans[["need_start", "need_end"]].reset_index()


def _close_returns(frame: pd.DataFrame, column: str = "close") -> pd.Series:
    series = frame.set_index("dt")[column].astype(float)
    series = series[series > 0]
    return series


def return_agreement(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    start: str,
    end: str,
    *,
    candidate_column: str = "close",
    accepted: Iterable[str] = (),
) -> dict[str, Any]:
    """Coverage and close-to-close return agreement over ``[start, end]``.

    Reference sessions with a close set the calendar. Returns are taken between
    consecutive reference sessions where both sources have a close. Days whose
    returns differ by more than ``MAX_DAILY_ABS_DIFF`` are counted, and those
    not in ``accepted`` are listed.
    """
    ref = _close_returns(reference)
    ref = ref[(ref.index >= start) & (ref.index <= end)]
    cand = _close_returns(candidate, candidate_column)
    cand = cand[(cand.index >= start) & (cand.index <= end)]
    result: dict[str, Any] = {
        "reference_sessions": len(ref),
        "missing_sessions": int((~ref.index.isin(cand.index)).sum()),
        "extra_sessions": int((~cand.index.isin(ref.index)).sum()),
        "return_pairs": 0,
        "correlation": None,
        "median_abs_diff": None,
        "max_abs_diff": None,
        "share_abs_diff_over_1pct": None,
        "accepted_large_diffs": 0,
        "unexplained_large_diffs": 0,
        "unexplained_large_diff_dates": [],
    }
    aligned = pd.DataFrame({"ref": ref, "cand": cand.reindex(ref.index)})
    rets = aligned.pct_change(fill_method=None).dropna()
    result["return_pairs"] = len(rets)
    if rets.empty:
        return result
    diff = (rets["cand"] - rets["ref"]).abs()
    large = diff[diff > MAX_DAILY_ABS_DIFF]
    accepted = set(accepted)
    unexplained = large[~large.index.isin(accepted)]
    result.update(
        median_abs_diff=float(diff.median()),
        max_abs_diff=float(diff.max()),
        share_abs_diff_over_1pct=float((diff > 0.01).mean()),
        accepted_large_diffs=len(large) - len(unexplained),
        unexplained_large_diffs=len(unexplained),
        unexplained_large_diff_dates=[
            {
                "dt": str(d),
                "eodhd": float(rets.at[d, "cand"]),
                "reference": float(rets.at[d, "ref"]),
            }
            for d in unexplained.index[:EVIDENCE_DATES]
        ],
    )
    if len(rets) >= MIN_PAIRS_FOR_CORRELATION:
        corr = rets["cand"].corr(rets["ref"])
        result["correlation"] = None if pd.isna(corr) else float(corr)
    return result


def agreement_passes(stats: Mapping[str, Any]) -> bool:
    """Whether agreement statistics clear the identity and coverage thresholds."""
    sessions = stats["reference_sessions"]
    if sessions == 0:
        return True
    if stats["missing_sessions"] > max(EDGE_SLACK_SESSIONS, MAX_MISSING_SHARE * sessions):
        return False
    if stats["unexplained_large_diffs"]:
        return False
    if stats["return_pairs"] < MIN_PAIRS_FOR_CORRELATION:
        return True
    if stats["correlation"] is None:
        return False
    return (
        stats["correlation"] >= MIN_RETURN_CORRELATION
        and stats["median_abs_diff"] <= MAX_MEDIAN_ABS_RETURN_DIFF
    )


def check_window(
    need: Mapping[str, str], reference: pd.DataFrame, segment: SymbolSegment | None = None
) -> tuple[str, str] | None:
    """The span to compare: the need, inside the reference's span and the segment."""
    if reference.empty:
        return None
    start = max(need["need_start"], str(reference["dt"].min()))
    end = min(need["need_end"], str(reference["dt"].max()))
    if segment is not None:
        if segment.start:
            start = max(start, segment.start)
        if segment.end:
            end = min(end, segment.end)
    return (start, end) if start <= end else None


def reference_check(
    panel: pd.DataFrame,
    reference: pd.DataFrame,
    pit: pd.DataFrame,
    accepted: Mapping[str, Iterable[str]] | None = None,
    unavailable: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[Finding]]:
    """Compare every identifier against the reference over its needed span.

    Identifiers declared ``unavailable`` are reported as such and not compared.
    """
    accepted = accepted or {}
    unavailable = set(unavailable)
    spans = needed_spans(pit).set_index("kdcode")
    by_code = dict(tuple(panel.groupby("kdcode"))) if not panel.empty else {}
    ref_by_code = dict(tuple(reference.groupby("kdcode"))) if not reference.empty else {}
    rows, findings = [], []
    empty = pd.DataFrame(columns=list(panel.columns) or ["dt", "close", "adjusted_close"])
    for kdcode in spans.index:
        need = spans.loc[kdcode]
        ref = ref_by_code.get(kdcode, pd.DataFrame(columns=["dt", "close"]))
        mine = by_code.get(kdcode, empty)
        window = check_window(need, ref)
        row: dict[str, Any] = {
            "kdcode": kdcode,
            "need_start": need["need_start"],
            "need_end": need["need_end"],
            "check_start": window[0] if window else None,
            "check_end": window[1] if window else None,
        }
        if kdcode in unavailable:
            row.update(verdict="declared_unavailable")
            rows.append(row)
            continue
        if window is None:
            row.update(verdict="no_reference")
            findings.append(
                Finding(kdcode, "no_reference_overlap", False, "Nothing to compare against")
            )
            rows.append(row)
            continue
        stats = return_agreement(mine, ref, *window, accepted=accepted.get(kdcode, ()))
        total = return_agreement(mine, ref, *window, candidate_column="adjusted_close")
        row.update({k: v for k, v in stats.items() if k != "unexplained_large_diff_dates"})
        row["total_return_median_abs_diff"] = total["median_abs_diff"]
        passed = agreement_passes(stats)
        row["verdict"] = "pass" if passed else "fail"
        if not passed:
            findings.append(
                Finding(
                    kdcode,
                    "reference_disagreement",
                    True,
                    "EODHD rows do not cover or do not match the reference over the span "
                    "this name is needed",
                    {"window": list(window), **stats},
                )
            )
        rows.append(row)
    return pd.DataFrame(rows), findings


def adjustment_basis_summary(check: pd.DataFrame) -> dict[str, Any]:
    """Which EODHD basis the reference's returns sit closer to, across names."""
    usable = check.dropna(subset=["median_abs_diff", "total_return_median_abs_diff"])
    if usable.empty:
        return {"names": 0}
    split_only = float(usable["median_abs_diff"].mean())
    total = float(usable["total_return_median_abs_diff"].mean())
    return {
        "names": len(usable),
        "mean_median_abs_diff_split_adjusted": split_only,
        "mean_median_abs_diff_total_return": total,
        "closer_basis": "split_adjusted" if split_only <= total else "total_return",
    }
