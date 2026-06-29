"""Audit LSEG alias coverage for unresolved PIT-universe identifiers.

The PIT market pull can fail on an original/dead RIC while the historical
LSEG tombstone RIC, such as ``ABMD.OQ^L22``, is present in both the PIT
universe and the market panel. This utility makes that coverage explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.pit import build_pit_masks, normalise_pit_intervals

DEFAULT_META_JSON = Path("data/raw/market/sp500_pit_union_lseg_20150101_20260513.meta.json")
DEFAULT_MARKET_CSV = Path("data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv")
DEFAULT_PIT_UNIVERSE_CSV = Path(
    "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv"
)
DEFAULT_OUTPUT = Path("docs/PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16.md")
DEFAULT_UNRESOLVED_ORIGINALS = (
    "AABA.OQ",
    "ABMD.OQ",
    "AET.N",
    "AGN.N",
    "AIRC.N",
    "ALXN.OQ",
)
DEFAULT_VALIDATION_START = "2022-01-01"
DEFAULT_VALIDATION_END = "2025-12-31"


@dataclass(frozen=True)
class AliasCoverageAudit:
    """Audit tables and summary metrics."""

    candidates: pd.DataFrame
    daily_impact: pd.DataFrame
    summary: dict[str, Any]


def _date_str(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalise_market_panel(market_panel: pd.DataFrame) -> pd.DataFrame:
    frame = market_panel.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"kdcode", "dt", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"market_panel is missing required columns: {sorted(missing)}")
    frame = frame[["kdcode", "dt", "close"]].copy()
    frame = frame.dropna(subset=["kdcode", "dt"])
    frame["kdcode"] = frame["kdcode"].astype(str).str.strip()
    frame["dt"] = pd.to_datetime(frame["dt"]).dt.strftime("%Y-%m-%d")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame[frame["kdcode"] != ""].reset_index(drop=True)


def load_unresolved_originals(meta_json: Path) -> list[str]:
    """Load unresolved original identifiers from the LSEG pull metadata."""
    if not meta_json.exists():
        return list(DEFAULT_UNRESOLVED_ORIGINALS)

    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)

    values = meta.get("currently_unresolved_original_failures", [])
    unresolved = [str(item.get("kdcode", "")).strip() for item in values if item.get("kdcode")]
    return unresolved or list(DEFAULT_UNRESOLVED_ORIGINALS)


def find_suffixed_candidates(original: str, pit_universe: pd.DataFrame) -> list[str]:
    """Return PIT-universe RICs that look like historical aliases for ``original``."""
    intervals = normalise_pit_intervals(pit_universe)
    prefix = f"{original}^"
    values = {
        str(kdcode) for kdcode in intervals["kdcode"].astype(str) if str(kdcode).startswith(prefix)
    }
    return sorted(values)


def _format_intervals(intervals: pd.DataFrame) -> str:
    if intervals.empty:
        return ""
    return "; ".join(
        f"{row.valid_from}..{row.valid_to}" for row in intervals.itertuples(index=False)
    )


def _active_dates_for_intervals(
    intervals: pd.DataFrame,
    validation_dates: list[str],
) -> set[str]:
    dates: set[str] = set()
    for row in intervals.itertuples(index=False):
        valid_from = str(row.valid_from)
        valid_to = str(row.valid_to)
        dates.update(date for date in validation_dates if valid_from <= date <= valid_to)
    return dates


def _candidate_market_summary(market_panel: pd.DataFrame) -> dict[str, dict[str, object]]:
    if market_panel.empty:
        return {}
    grouped = market_panel.groupby("kdcode")["dt"]
    summary = grouped.agg(["count", "min", "max"]).reset_index()
    return {
        str(row["kdcode"]): {
            "market_rows": int(row["count"]),
            "market_date_min": str(row["min"]),
            "market_date_max": str(row["max"]),
        }
        for row in summary.to_dict("records")
    }


def _validation_dates(market_panel: pd.DataFrame, start: str, end: str) -> list[str]:
    values = sorted(
        date for date in market_panel["dt"].astype(str).unique().tolist() if start <= date <= end
    )
    if values:
        return values
    return [_date_str(date) for date in pd.date_range(start=start, end=end, freq="B")]


def _scoreable_dates_by_candidate(
    market_panel: pd.DataFrame,
    intervals: pd.DataFrame,
    candidate_kdcodes: list[str],
    validation_dates: list[str],
    his_t: int,
    label_t: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    if not candidate_kdcodes or not validation_dates:
        return {}, {}

    masks = build_pit_masks(
        df_for_features=market_panel,
        df_for_labels=market_panel,
        kdcode_list=candidate_kdcodes,
        sample_dates=validation_dates,
        his_t=his_t,
        label_t=label_t,
        pit_intervals=intervals[intervals["kdcode"].isin(candidate_kdcodes)],
    )

    tradable: dict[str, set[str]] = {kdcode: set() for kdcode in candidate_kdcodes}
    loss: dict[str, set[str]] = {kdcode: set() for kdcode in candidate_kdcodes}
    for col, kdcode in enumerate(candidate_kdcodes):
        for row, date in enumerate(validation_dates):
            if bool(masks.tradable[row, col]):
                tradable[kdcode].add(date)
            if bool(masks.loss[row, col]):
                loss[kdcode].add(date)
    return tradable, loss


def _coerce_bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [
        "candidate_found",
        "has_market_rows",
        "overlaps_validation",
        "original_overlaps_validation",
    ]
    for col in bool_cols:
        if col in frame.columns:
            frame[col] = frame[col].map(bool).astype(object)
    return frame


def run_alias_coverage_audit(
    unresolved_originals: list[str],
    pit_universe: pd.DataFrame,
    market_panel: pd.DataFrame,
    validation_start: str = DEFAULT_VALIDATION_START,
    validation_end: str = DEFAULT_VALIDATION_END,
    his_t: int = 10,
    label_t: int = 5,
) -> AliasCoverageAudit:
    """Audit PIT alias candidates and validation-window breadth impact."""
    intervals = normalise_pit_intervals(pit_universe)
    market = _normalise_market_panel(market_panel)
    validation_dates = _validation_dates(market, validation_start, validation_end)
    market_summary = _candidate_market_summary(market)

    candidates_by_original = {
        original: find_suffixed_candidates(original, intervals) for original in unresolved_originals
    }
    candidate_kdcodes = sorted(
        {kdcode for values in candidates_by_original.values() for kdcode in values}
    )
    tradable_dates, loss_dates = _scoreable_dates_by_candidate(
        market,
        intervals,
        candidate_kdcodes,
        validation_dates,
        his_t,
        label_t,
    )

    original_active_dates: dict[str, set[str]] = {}
    candidate_active_dates_by_original: dict[str, set[str]] = {}
    candidate_scoreable_dates_by_original: dict[str, set[str]] = {}
    candidate_rows: list[dict[str, object]] = []

    for original in unresolved_originals:
        original_intervals = intervals[intervals["kdcode"] == original]
        original_dates = _active_dates_for_intervals(original_intervals, validation_dates)
        original_active_dates[original] = original_dates
        original_overlaps_validation = bool(original_dates)
        original_market = market_summary.get(
            original,
            {"market_rows": 0, "market_date_min": "", "market_date_max": ""},
        )

        candidates = candidates_by_original[original]
        candidate_active_dates_by_original[original] = set()
        candidate_scoreable_dates_by_original[original] = set()

        if not candidates:
            candidate_rows.append(
                {
                    "original": original,
                    "original_interval_count": int(len(original_intervals)),
                    "original_intervals": _format_intervals(original_intervals),
                    "original_overlaps_validation": original_overlaps_validation,
                    "original_active_days_in_validation": int(len(original_dates)),
                    "original_market_rows": int(original_market["market_rows"]),
                    "candidate": "",
                    "candidate_found": False,
                    "candidate_valid_from": "",
                    "candidate_valid_to": "",
                    "overlaps_validation": False,
                    "has_market_rows": False,
                    "market_rows": 0,
                    "market_date_min": "",
                    "market_date_max": "",
                    "active_days_in_validation": 0,
                    "scoreable_days_in_validation": 0,
                    "loss_days_in_validation": 0,
                }
            )
            continue

        for candidate in candidates:
            candidate_intervals = intervals[intervals["kdcode"] == candidate]
            candidate_market = market_summary.get(
                candidate,
                {"market_rows": 0, "market_date_min": "", "market_date_max": ""},
            )
            has_market_rows = int(candidate_market["market_rows"]) > 0
            candidate_dates = _active_dates_for_intervals(candidate_intervals, validation_dates)
            if has_market_rows:
                candidate_active_dates_by_original[original].update(candidate_dates)
            candidate_scoreable_dates_by_original[original].update(
                candidate_dates & tradable_dates.get(candidate, set())
            )

            for interval in candidate_intervals.itertuples(index=False):
                interval_frame = pd.DataFrame(
                    [
                        {
                            "kdcode": candidate,
                            "valid_from": interval.valid_from,
                            "valid_to": interval.valid_to,
                        }
                    ]
                )
                interval_dates = _active_dates_for_intervals(interval_frame, validation_dates)
                scoreable_dates = interval_dates & tradable_dates.get(candidate, set())
                candidate_rows.append(
                    {
                        "original": original,
                        "original_interval_count": int(len(original_intervals)),
                        "original_intervals": _format_intervals(original_intervals),
                        "original_overlaps_validation": original_overlaps_validation,
                        "original_active_days_in_validation": int(len(original_dates)),
                        "original_market_rows": int(original_market["market_rows"]),
                        "candidate": candidate,
                        "candidate_found": True,
                        "candidate_valid_from": str(interval.valid_from),
                        "candidate_valid_to": str(interval.valid_to),
                        "overlaps_validation": bool(interval_dates),
                        "has_market_rows": has_market_rows,
                        "market_rows": int(candidate_market["market_rows"]),
                        "market_date_min": str(candidate_market["market_date_min"]),
                        "market_date_max": str(candidate_market["market_date_max"]),
                        "active_days_in_validation": int(len(interval_dates)),
                        "scoreable_days_in_validation": int(len(scoreable_dates)),
                        "loss_days_in_validation": int(
                            len(interval_dates & loss_dates.get(candidate, set()))
                        ),
                    }
                )

    candidates = _coerce_bool_columns(pd.DataFrame(candidate_rows))

    daily_rows: list[dict[str, object]] = []
    for date in validation_dates:
        active_originals = {
            original for original, dates in original_active_dates.items() if date in dates
        }
        covered_originals = {
            original
            for original in active_originals
            if date in candidate_active_dates_by_original.get(original, set())
        }
        scoreable_originals = {
            original
            for original in active_originals
            if date in candidate_scoreable_dates_by_original.get(original, set())
        }
        daily_rows.append(
            {
                "date": date,
                "original_active_count": int(len(active_originals)),
                "covered_by_candidate_count": int(len(covered_originals)),
                "scoreable_by_candidate_count": int(len(scoreable_originals)),
                "uncovered_active_count": int(len(active_originals - covered_originals)),
                "unscoreable_active_count": int(len(active_originals - scoreable_originals)),
            }
        )
    daily_impact = pd.DataFrame(daily_rows)

    found_rows = (
        candidates[candidates["candidate_found"].map(bool)] if not candidates.empty else candidates
    )
    found_market_rows = (
        found_rows[found_rows["has_market_rows"].map(bool)] if not found_rows.empty else found_rows
    )
    summary = {
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_dates": int(len(validation_dates)),
        "his_t": int(his_t),
        "label_t": int(label_t),
        "unresolved_originals": int(len(unresolved_originals)),
        "originals_with_any_candidate": int(
            sum(bool(candidates_by_original[original]) for original in unresolved_originals)
        ),
        "originals_with_candidate_market_rows": int(
            sum(
                any(candidate in market_summary for candidate in candidates_by_original[original])
                for original in unresolved_originals
            )
        ),
        "candidate_rows": int(len(found_rows)),
        "distinct_candidates": int(found_rows["candidate"].nunique())
        if not found_rows.empty
        else 0,
        "candidates_with_market_rows": int(len(found_market_rows)),
        "original_active_member_days": int(daily_impact["original_active_count"].sum()),
        "candidate_covered_active_days": int(daily_impact["covered_by_candidate_count"].sum()),
        "candidate_scoreable_active_days": int(daily_impact["scoreable_by_candidate_count"].sum()),
        "uncovered_active_member_days": int(daily_impact["uncovered_active_count"].sum()),
        "unscoreable_active_member_days": int(daily_impact["unscoreable_active_count"].sum()),
        "max_daily_original_active_count": int(daily_impact["original_active_count"].max())
        if not daily_impact.empty
        else 0,
        "max_daily_uncovered_active_count": int(daily_impact["uncovered_active_count"].max())
        if not daily_impact.empty
        else 0,
        "max_daily_unscoreable_active_count": int(daily_impact["unscoreable_active_count"].max())
        if not daily_impact.empty
        else 0,
    }
    return AliasCoverageAudit(candidates=candidates, daily_impact=daily_impact, summary=summary)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
    ]
    for row in view.to_dict("records"):
        values = [str(row[col]).replace("|", "\\|") for col in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _yearly_impact(daily_impact: pd.DataFrame) -> pd.DataFrame:
    if daily_impact.empty:
        return daily_impact
    frame = daily_impact.copy()
    frame["year"] = frame["date"].astype(str).str.slice(0, 4)
    return (
        frame.groupby("year", as_index=False)[
            [
                "original_active_count",
                "covered_by_candidate_count",
                "scoreable_by_candidate_count",
                "uncovered_active_count",
                "unscoreable_active_count",
            ]
        ]
        .sum()
        .rename(
            columns={
                "original_active_count": "original_active_member_days",
                "covered_by_candidate_count": "candidate_covered_active_days",
                "scoreable_by_candidate_count": "candidate_scoreable_active_days",
                "uncovered_active_count": "uncovered_active_member_days",
                "unscoreable_active_count": "unscoreable_active_member_days",
            }
        )
    )


def render_markdown(
    result: AliasCoverageAudit, title: str = "PIT LSEG Alias Coverage Audit"
) -> str:
    """Render a compact Markdown audit artifact."""
    summary_rows = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in result.summary.items()]
    )
    candidate_columns = [
        "original",
        "candidate",
        "candidate_valid_from",
        "candidate_valid_to",
        "has_market_rows",
        "market_rows",
        "market_date_min",
        "market_date_max",
        "overlaps_validation",
        "active_days_in_validation",
        "scoreable_days_in_validation",
        "loss_days_in_validation",
    ]
    yearly = _yearly_impact(result.daily_impact)
    peak_daily = result.daily_impact[
        (result.daily_impact["original_active_count"] > 0)
        | (result.daily_impact["uncovered_active_count"] > 0)
        | (result.daily_impact["unscoreable_active_count"] > 0)
    ].copy()
    if len(peak_daily) > 40:
        peak_daily = pd.concat([peak_daily.head(20), peak_daily.tail(20)], ignore_index=True)

    return (
        "\n\n".join(
            [
                f"# {title}",
                "This audit checks whether unresolved original LSEG identifiers have suffixed "
                "historical RIC candidates in the PIT universe and market rows in the PIT-union "
                "market panel.",
                "## Summary",
                _markdown_table(summary_rows, ["metric", "value"]),
                "## Candidate Coverage",
                _markdown_table(result.candidates, candidate_columns),
                "## Validation Breadth Impact By Year",
                _markdown_table(
                    yearly,
                    [
                        "year",
                        "original_active_member_days",
                        "candidate_covered_active_days",
                        "candidate_scoreable_active_days",
                        "uncovered_active_member_days",
                        "unscoreable_active_member_days",
                    ],
                ),
                "## Daily Impact Rows",
                "Rows are shown for dates with non-zero unresolved-original activity. "
                "For long outputs this table shows the first and last 20 such dates; use the "
                "companion daily CSV for the full daily series.",
                _markdown_table(
                    peak_daily,
                    [
                        "date",
                        "original_active_count",
                        "covered_by_candidate_count",
                        "scoreable_by_candidate_count",
                        "uncovered_active_count",
                        "unscoreable_active_count",
                    ],
                ),
            ]
        )
        + "\n"
    )


def write_audit_outputs(
    result: AliasCoverageAudit,
    output_path: Path,
    output_format: str,
) -> list[Path]:
    """Write the requested artifact plus companion CSVs for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if output_format == "csv":
        result.candidates.to_csv(output_path, index=False)
        written.append(output_path)
    else:
        output_path.write_text(render_markdown(result), encoding="utf-8")
        written.append(output_path)

    daily_path = output_path.with_name(f"{output_path.stem}_daily_impact.csv")
    candidates_path = output_path.with_name(f"{output_path.stem}_candidates.csv")
    result.daily_impact.to_csv(daily_path, index=False)
    result.candidates.to_csv(candidates_path, index=False)
    written.extend([daily_path, candidates_path])
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit unresolved LSEG alias coverage for the PIT universe."
    )
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument("--pit-universe-csv", type=Path, default=DEFAULT_PIT_UNIVERSE_CSV)
    parser.add_argument("--market-csv", type=Path, default=DEFAULT_MARKET_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-format", choices=["markdown", "csv"], default=None)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--his-t", type=int, default=10)
    parser.add_argument("--label-t", type=int, default=5)
    parser.add_argument(
        "--unresolved",
        nargs="*",
        default=None,
        help="Override unresolved originals. Defaults to metadata currently_unresolved_original_failures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    unresolved = (
        args.unresolved
        if args.unresolved is not None
        else load_unresolved_originals(args.meta_json)
    )
    output_format = args.output_format
    if output_format is None:
        output_format = "csv" if args.output.suffix.lower() == ".csv" else "markdown"

    pit_universe = pd.read_csv(args.pit_universe_csv)
    market_panel = pd.read_csv(args.market_csv, usecols=["kdcode", "dt", "close"])
    result = run_alias_coverage_audit(
        unresolved_originals=unresolved,
        pit_universe=pit_universe,
        market_panel=market_panel,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        his_t=args.his_t,
        label_t=args.label_t,
    )
    written = write_audit_outputs(result, args.output, output_format)

    print(f"Unresolved originals audited: {result.summary['unresolved_originals']}")
    print(
        "Candidate rows with market rows: "
        f"{result.summary['candidates_with_market_rows']}/{result.summary['candidate_rows']}"
    )
    print(
        "Validation active member-days covered by candidates: "
        f"{result.summary['candidate_covered_active_days']}/"
        f"{result.summary['original_active_member_days']}"
    )
    print(f"Uncovered active member-days: {result.summary['uncovered_active_member_days']}")
    print(f"Unscoreable active member-days: {result.summary['unscoreable_active_member_days']}")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
