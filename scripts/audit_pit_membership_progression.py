"""Audit S&P 500 PIT membership progression from Joiner/Leaver artifacts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CHANGES_CSV = Path(
    "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_changes.csv"
)
DEFAULT_SNAPSHOTS_CSV = Path(
    "data/raw/constituents/sp500_pit_joiner_leaver_20160101_20260513_snapshots.csv"
)
DEFAULT_MARKET_CSV = Path("data/raw/market/sp500_pit_union_lseg_20150101_20260513.csv")
DEFAULT_OUTPUT_CSV = Path("docs/audits/pit_membership_progression_snapshot_counts.csv")
DEFAULT_OUTPUT_MARKDOWN = Path("docs/audits/pit_membership_progression_audit.md")
DEFAULT_VALIDATION_YEARS = (2022, 2023, 2024, 2025)


@dataclass(frozen=True)
class AuditResult:
    """In-memory outputs from the PIT membership progression audit."""

    change_summary: pd.DataFrame
    snapshot_progression: pd.DataFrame
    transition_events: pd.DataFrame
    representative_transitions: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _format_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _path_display(path: Path) -> str:
    return path.as_posix()


def _normalise_validation_years(validation_years: Sequence[int]) -> tuple[int, ...]:
    years = tuple(sorted({int(year) for year in validation_years}))
    if not years:
        raise ValueError("validation_years must contain at least one year")
    return years


def load_changes(path: Path | str) -> pd.DataFrame:
    changes = pd.read_csv(path)
    _require_columns(
        changes,
        {"change_date", "constituent_ric", "company_name", "change"},
        str(path),
    )
    changes = changes.copy()
    changes["change_date"] = pd.to_datetime(changes["change_date"])
    changes["constituent_ric"] = changes["constituent_ric"].astype(str).str.strip()
    changes["change"] = changes["change"].astype(str).str.strip()
    return changes.sort_values(["change_date", "change", "constituent_ric"]).reset_index(drop=True)


def load_snapshots(path: Path | str) -> pd.DataFrame:
    snapshots = pd.read_csv(path)
    _require_columns(snapshots, {"as_of_date", "constituent_ric", "company_name"}, str(path))
    snapshots = snapshots.copy()
    snapshots["as_of_date"] = pd.to_datetime(snapshots["as_of_date"])
    snapshots["constituent_ric"] = snapshots["constituent_ric"].astype(str).str.strip()
    return snapshots.sort_values(["as_of_date", "constituent_ric"]).reset_index(drop=True)


def load_market_availability(path: Path | str) -> pd.DataFrame:
    market = pd.read_csv(path, usecols=["kdcode", "dt"])
    _require_columns(market, {"kdcode", "dt"}, str(path))
    market = market.dropna(subset=["kdcode", "dt"]).copy()
    market["kdcode"] = market["kdcode"].astype(str).str.strip()
    market["dt"] = pd.to_datetime(market["dt"])
    availability = (
        market.groupby("kdcode", sort=True)["dt"]
        .agg(market_row_count="size", market_first_dt="min", market_last_dt="max")
        .reset_index()
    )
    availability["market_first_dt"] = availability["market_first_dt"].map(_format_date)
    availability["market_last_dt"] = availability["market_last_dt"].map(_format_date)
    return availability


def summarize_changes_by_year(changes: pd.DataFrame) -> pd.DataFrame:
    frame = changes.copy()
    _require_columns(frame, {"change_date", "change"}, "changes")
    frame["change_date"] = pd.to_datetime(frame["change_date"])
    frame["year"] = frame["change_date"].dt.year.astype(int)
    frame["change_norm"] = frame["change"].astype(str).str.strip().str.lower()

    counts = (
        frame.groupby(["year", "change_norm"], sort=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={"joiner": "joiners", "leaver": "leavers"})
    )
    for column in ("joiners", "leavers"):
        if column not in counts.columns:
            counts[column] = 0
    counts = counts[["joiners", "leavers"]].astype(int)
    counts["total_changes"] = counts["joiners"] + counts["leavers"]
    return counts.reset_index()


def _snapshot_diffs(
    snapshots: pd.DataFrame,
) -> list[tuple[pd.Timestamp, set[str], set[str], set[str]]]:
    frame = snapshots.copy()
    _require_columns(frame, {"as_of_date", "constituent_ric"}, "snapshots")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    frame["constituent_ric"] = frame["constituent_ric"].astype(str).str.strip()

    rows: list[tuple[pd.Timestamp, set[str], set[str], set[str]]] = []
    previous_members: set[str] | None = None
    for as_of_date, group in frame.sort_values(["as_of_date", "constituent_ric"]).groupby(
        "as_of_date",
        sort=True,
    ):
        members = set(group["constituent_ric"].dropna())
        joined = set() if previous_members is None else members - previous_members
        left = set() if previous_members is None else previous_members - members
        rows.append((pd.Timestamp(as_of_date), members, joined, left))
        previous_members = members
    return rows


def build_snapshot_progression(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for as_of_date, members, joined, left in _snapshot_diffs(snapshots):
        rows.append(
            {
                "as_of_date": _format_date(as_of_date),
                "year": int(as_of_date.year),
                "member_count": len(members),
                "joined_count": len(joined),
                "left_count": len(left),
                "transition_count": len(joined) + len(left),
                "net_change": len(joined) - len(left),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "as_of_date",
            "year",
            "member_count",
            "joined_count",
            "left_count",
            "transition_count",
            "net_change",
        ],
    )


def build_snapshot_transition_events(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for as_of_date, _members, joined, left in _snapshot_diffs(snapshots):
        for constituent_ric in sorted(joined):
            rows.append(
                {
                    "as_of_date": _format_date(as_of_date),
                    "year": int(as_of_date.year),
                    "direction": "Joiner",
                    "constituent_ric": constituent_ric,
                }
            )
        for constituent_ric in sorted(left):
            rows.append(
                {
                    "as_of_date": _format_date(as_of_date),
                    "year": int(as_of_date.year),
                    "direction": "Leaver",
                    "constituent_ric": constituent_ric,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["as_of_date", "year", "direction", "constituent_ric"],
    )


def require_validation_period_changes(
    snapshot_progression: pd.DataFrame,
    validation_years: Sequence[int],
) -> None:
    years = _normalise_validation_years(validation_years)
    validation_rows = snapshot_progression[snapshot_progression["year"].isin(years)]
    total_transitions = int(validation_rows["transition_count"].sum())
    if validation_rows.empty or total_transitions == 0:
        raise ValueError(
            "Snapshot membership never changes across the validation period "
            f"{years[0]}-{years[-1]}."
        )


def _market_lookup(availability: pd.DataFrame) -> dict[str, dict[str, object]]:
    exact: dict[str, dict[str, object]] = {}
    suffixed_by_base: dict[str, dict[str, object]] = {}
    for row in availability.to_dict("records"):
        kdcode = str(row["kdcode"])
        exact[kdcode] = row
        if "^" in kdcode:
            base = kdcode.split("^", maxsplit=1)[0]
            suffixed_by_base.setdefault(base, row)
    return {"exact": exact, "suffixed_by_base": suffixed_by_base}


def attach_market_availability(
    transition_events: pd.DataFrame,
    market_availability: pd.DataFrame,
) -> pd.DataFrame:
    lookup = _market_lookup(market_availability)
    exact = lookup["exact"]
    suffixed_by_base = lookup["suffixed_by_base"]

    rows = []
    for event in transition_events.to_dict("records"):
        ric = str(event["constituent_ric"])
        market_row = exact.get(ric)
        if market_row is None and "^" in ric:
            market_row = exact.get(ric.split("^", maxsplit=1)[0])
        if market_row is None:
            market_row = suffixed_by_base.get(ric)

        out = dict(event)
        if market_row is None:
            out.update(
                {
                    "market_kdcode": "",
                    "market_row_count": 0,
                    "market_first_dt": "",
                    "market_last_dt": "",
                }
            )
        else:
            out.update(
                {
                    "market_kdcode": market_row["kdcode"],
                    "market_row_count": int(market_row["market_row_count"]),
                    "market_first_dt": market_row["market_first_dt"],
                    "market_last_dt": market_row["market_last_dt"],
                }
            )
        rows.append(out)

    return pd.DataFrame(
        rows,
        columns=[
            "as_of_date",
            "year",
            "direction",
            "constituent_ric",
            "market_kdcode",
            "market_row_count",
            "market_first_dt",
            "market_last_dt",
        ],
    )


def select_representative_transitions(
    transition_events: pd.DataFrame,
    validation_years: Sequence[int],
) -> pd.DataFrame:
    years = _normalise_validation_years(validation_years)
    representatives = []
    for year in years:
        year_events = transition_events[transition_events["year"] == year].sort_values(
            ["as_of_date", "direction", "constituent_ric"]
        )
        if year_events.empty:
            raise ValueError(f"No snapshot transition found for validation year {year}.")

        joiners = year_events[year_events["direction"] == "Joiner"]
        if joiners.empty:
            raise ValueError(f"No snapshot joiner transition found for validation year {year}.")
        joiners_with_market = joiners[joiners["market_row_count"] > 0]
        if joiners_with_market.empty:
            raise ValueError(
                "No representative snapshot joiner with PIT-union market rows found "
                f"for validation year {year}."
            )
        joiners_with_market = joiners_with_market.copy()
        joiners_with_market["_as_of_dt"] = pd.to_datetime(joiners_with_market["as_of_date"])
        joiners_with_market["_market_first_dt"] = pd.to_datetime(
            joiners_with_market["market_first_dt"]
        )
        joiners_with_market["_market_last_dt"] = pd.to_datetime(
            joiners_with_market["market_last_dt"]
        )
        market_covers_transition = joiners_with_market[
            (joiners_with_market["_market_first_dt"] <= joiners_with_market["_as_of_dt"])
            & (joiners_with_market["_market_last_dt"] >= joiners_with_market["_as_of_dt"])
        ]
        joiner_choice = (
            market_covers_transition.iloc[0]
            if not market_covers_transition.empty
            else joiners_with_market.iloc[0]
        )
        representatives.append(
            {
                key: value
                for key, value in joiner_choice.to_dict().items()
                if not key.startswith("_")
            }
        )

        leavers = year_events[year_events["direction"] == "Leaver"]
        if not leavers.empty:
            representatives.append(leavers.iloc[0].to_dict())

    return pd.DataFrame(representatives).reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(_markdown_escape(column) for column in columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            "| " + " | ".join(_markdown_escape(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(lines)


def render_markdown_report(
    result: AuditResult,
    changes_csv: Path,
    snapshots_csv: Path,
    market_csv: Path,
    output_csv: Path | None,
    validation_years: Sequence[int],
) -> str:
    progression = result.snapshot_progression
    transition_by_year = (
        progression[progression["year"].isin(_normalise_validation_years(validation_years))]
        .groupby("year", sort=True)[["joined_count", "left_count", "transition_count"]]
        .sum()
        .astype(int)
        .reset_index()
    )
    member_min = int(progression["member_count"].min()) if not progression.empty else 0
    member_max = int(progression["member_count"].max()) if not progression.empty else 0
    total_joiners = int(result.change_summary["joiners"].sum())
    total_leavers = int(result.change_summary["leavers"].sum())

    lines = [
        "# PIT Membership Progression Audit",
        "",
        "This audit makes the Joiner/Leaver point-in-time membership progression "
        "visible for the masked-panel validation workflow.",
        "",
        "## Sources",
        "",
        f"- Changes: `{_path_display(changes_csv)}`",
        f"- Snapshots: `{_path_display(snapshots_csv)}`",
        f"- PIT-union market panel: `{_path_display(market_csv)}`",
    ]
    if output_csv is not None:
        lines.append(f"- Snapshot progression CSV: `{_path_display(output_csv)}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Changes file totals: {total_joiners} joiners and {total_leavers} leavers.",
            f"- Snapshot active membership range: {member_min}-{member_max} names.",
            "- Validation-year guard: snapshot membership changes within "
            f"{min(validation_years)}-{max(validation_years)}.",
            "",
            "## Joiners And Leavers By Year",
            "",
            _markdown_table(result.change_summary),
            "",
            "## Validation-Year Snapshot Transition Counts",
            "",
            _markdown_table(transition_by_year),
            "",
            "## Representative Snapshot Transitions",
            "",
            _markdown_table(result.representative_transitions),
        ]
    )
    return "\n".join(lines) + "\n"


def _write_markdown(
    result: AuditResult,
    path: Path,
    changes_csv: Path,
    snapshots_csv: Path,
    market_csv: Path,
    output_csv: Path | None,
    validation_years: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(
            result=result,
            changes_csv=changes_csv,
            snapshots_csv=snapshots_csv,
            market_csv=market_csv,
            output_csv=output_csv,
            validation_years=validation_years,
        ),
        encoding="utf-8",
    )


def run_audit(
    changes_csv: Path | str = DEFAULT_CHANGES_CSV,
    snapshots_csv: Path | str = DEFAULT_SNAPSHOTS_CSV,
    market_csv: Path | str = DEFAULT_MARKET_CSV,
    validation_years: Sequence[int] = DEFAULT_VALIDATION_YEARS,
    output_csv: Path | str | None = DEFAULT_OUTPUT_CSV,
    output_markdown: Path | str | None = DEFAULT_OUTPUT_MARKDOWN,
) -> AuditResult:
    changes_path = Path(changes_csv)
    snapshots_path = Path(snapshots_csv)
    market_path = Path(market_csv)
    output_csv_path = Path(output_csv) if output_csv is not None else None
    output_markdown_path = Path(output_markdown) if output_markdown is not None else None

    changes = load_changes(changes_path)
    snapshots = load_snapshots(snapshots_path)
    market_availability = load_market_availability(market_path)

    change_summary = summarize_changes_by_year(changes)
    snapshot_progression = build_snapshot_progression(snapshots)
    require_validation_period_changes(snapshot_progression, validation_years)

    transition_events = build_snapshot_transition_events(snapshots)
    transition_events = attach_market_availability(transition_events, market_availability)
    representative_transitions = select_representative_transitions(
        transition_events,
        validation_years,
    )

    result = AuditResult(
        change_summary=change_summary,
        snapshot_progression=snapshot_progression,
        transition_events=transition_events,
        representative_transitions=representative_transitions,
    )

    if output_csv_path is not None:
        _write_csv(snapshot_progression, output_csv_path)
    if output_markdown_path is not None:
        _write_markdown(
            result=result,
            path=output_markdown_path,
            changes_csv=changes_path,
            snapshots_csv=snapshots_path,
            market_csv=market_path,
            output_csv=output_csv_path,
            validation_years=validation_years,
        )

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PIT S&P 500 membership progression from Joiner/Leaver files."
    )
    parser.add_argument("--changes-csv", type=Path, default=DEFAULT_CHANGES_CSV)
    parser.add_argument("--snapshots-csv", type=Path, default=DEFAULT_SNAPSHOTS_CSV)
    parser.add_argument("--market-csv", type=Path, default=DEFAULT_MARKET_CSV)
    parser.add_argument(
        "--validation-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_YEARS),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(
        changes_csv=args.changes_csv,
        snapshots_csv=args.snapshots_csv,
        market_csv=args.market_csv,
        validation_years=tuple(args.validation_years),
        output_csv=args.output_csv,
        output_markdown=args.output_markdown,
    )
    progression = result.snapshot_progression
    print("PIT membership progression audit complete")
    print(
        "Changes: "
        f"{int(result.change_summary['joiners'].sum())} joiners, "
        f"{int(result.change_summary['leavers'].sum())} leavers"
    )
    print(
        "Snapshots: "
        f"{len(progression)} dates, "
        f"{int(progression['member_count'].min())}-{int(progression['member_count'].max())} "
        "members"
    )
    print(f"Representative transitions: {len(result.representative_transitions)} rows")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
