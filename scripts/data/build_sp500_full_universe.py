"""The full point-in-time S&P 500 universe priced from EODHD: membership file and checks (#386).

The 110-name universe is a market-cap selection inside the S&P 500. This script
builds the whole index instead, from the same preserved LSEG membership the
110-name package was cut from, and checks the EODHD prices that
``scripts/data/export_eodhd_pit_prices.py`` pulls for it. Two steps::

    python -m scripts.data.build_sp500_full_universe membership \\
        --intervals <lseg>/constituents/<lseg_prefix>_sp500_membership_intervals.csv \\
        --reference-manifest data/manifests/<lseg_prefix>.r1.json \\
        --constituents <saved EODHD fundamentals/GSPC.INDX?historical=1 response> \\
        --output-dir /content/sp500_full --prefix sp500_pit_full_20160104_20260731

    python -m scripts.data.build_sp500_full_universe check \\
        --panel <package>/market/<prefix>_eodhd_<start>_<end>.csv \\
        --pit-universe <package>/constituents/<prefix>_pit_universe.csv \\
        --metadata <lseg>/constituents/<lseg_prefix>_all_metadata_snapshots.csv \\
        --review data/mappings/eodhd_review_sp500_full.json --output <report>.json

**membership** byte-checks the LSEG intervals file against the manifest that
records it, then writes ``constituents/<prefix>_pit_universe.csv`` with the same
rows under the ``kdcode, valid_from, valid_to`` header the price export reads.
Identifiers stay LSEG RICs, so a delisted company keeps its ``^`` code and the
symbol map and reference panel of the 110-name package apply unchanged. When
EODHD's own S&P 500 history is given (``--constituents``), it is compared
member-day by member-day and the agreement is written to
``<prefix>_membership_check.json``. That comparison is evidence only: EODHD
files past members under today's tickers, so most disagreements are labels.

**check** reads the pulled panel against what LSEG recorded independently of
it, since the LSEG price panel covers only 206 of the names:

- *Market cap*: LSEG's monthly company market cap moves with the share price
  except where the share count changes. For each name and each pair of
  adjacent month-ends inside its membership, the change in log market cap less
  the change in log EODHD close is the change in implied share count. LSEG
  updates share counts rarely, so for a correct series that difference is
  almost always exactly zero, and a name whose median exceeds
  ``MCAP_MEDIAN_LIMIT`` is flagged. A month where the close moved by more than
  ``PRICE_MOVE_LIMIT`` and the market cap did not follow (a share step over
  ``MCAP_STEP_LIMIT``) is a price step: a missed split, a bad splice or a wrong
  series. A share step with a steady close is a merger or LSEG's share count
  updating, and is listed but not flagged. LSEG's own record can be wrong: it
  gives Johnson Controls Inc. (``JCI.N^I16``) Tyco's market cap, so agreement
  with it is evidence, not proof.
- *Identity*: market cap over at least ``MIN_IDENTITY_PAIRS`` month pairs, or,
  for a delisted name LSEG has no market cap for, a last panel row within a
  month of the month its ``^`` code names (``^J23`` is October 2023). A name
  with neither needs a reviewed reason.
- *Delisting month*: a delisted name whose rows run on more than a month past,
  or stop more than a month before, its ``^`` code month.
- *Coverage*: sessions inside a name's membership with no panel row, except
  those after a delisted name's last trade.
- *Extreme moves*: one-day close-to-close moves inside membership above
  ``RISE_LIMIT`` or below ``FALL_LIMIT``, unless LSEG's market cap made the
  same move that month.

Every remaining flag needs a review entry with a reason in ``--review``; one
without a reason is refused. Unreviewed flags make the exit status 1.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.data.eodhd_prices import default_symbol
from mci_gru.data.input_manifest import read_input_manifest

#: A month-end-to-month-end change in log implied share count larger than this is a step.
MCAP_STEP_LIMIT = 0.10
#: A step is a price error only when the EODHD close also moved more than this (log).
PRICE_MOVE_LIMIT = 0.25
#: A name whose typical monthly share step exceeds this does not track LSEG's market cap.
MCAP_MEDIAN_LIMIT = 0.02
#: Month pairs needed before market cap counts as evidence of the series' identity.
MIN_IDENTITY_PAIRS = 3
#: A month-end close is matched to a panel session at most this many calendar days earlier.
MCAP_MATCH_DAYS = 6
#: One-day close-to-close moves inside membership outside these bounds are flagged.
RISE_LIMIT = 0.40
FALL_LIMIT = -0.35
#: A panel session counts toward the calendar when at least this many names trade.
CALENDAR_MIN_NAMES = 100
#: Review sections a flag can be decided in; ``note`` carries context only.
REVIEW_SECTIONS = (
    "note",
    "price_steps",
    "extreme_moves",
    "missing_sessions",
    "identity",
    "delisting_month",
)
DELISTED = "^"
#: LSEG's delisting code month letters, January to December.
MONTH_LETTERS = "ABCDEFGHIJKL"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


def pit_from_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """The LSEG intervals as the membership file: same rows, ``kdcode`` header."""
    frame = intervals.rename(columns={"constituent_ric": "kdcode"})
    missing = {"kdcode", "valid_from", "valid_to"} - set(frame.columns)
    if missing:
        raise SystemExit(f"Intervals file lacks columns {sorted(missing)}")
    frame = frame[["kdcode", "valid_from", "valid_to"]]
    if frame.isna().any().any() or (frame["valid_from"] > frame["valid_to"]).any():
        raise SystemExit("Intervals file has blank dates or an interval that ends before it starts")
    return frame


def symbol_root(kdcode: str) -> str:
    """The ticker a RIC maps to by rule, as EODHD writes tickers (``BRKb.N`` is ``BRK-B``)."""
    return default_symbol(kdcode).removesuffix(".US")


def eodhd_snapshots(payload: dict[str, Any]) -> tuple[list[str], list[frozenset[str]]]:
    """EODHD's dated S&P 500 member lists, oldest first, with ``_old`` suffixes removed."""
    history = payload.get("HistoricalComponents") or {}
    dates = sorted(history)
    members = []
    for day in dates:
        codes = {str(row["Code"]).split("_old", 1)[0] for row in history[day].values()}
        members.append(frozenset(codes))
    return dates, members


def membership_agreement(
    pit: pd.DataFrame, dates: list[str], members: list[frozenset[str]], sessions: list[str]
) -> dict[str, Any]:
    """Member-days on which LSEG and EODHD agree, by ticker, and the largest disagreements."""
    pit = pit.assign(code=pit["kdcode"].map(symbol_root))
    lseg_only: dict[str, int] = {}
    eodhd_only: dict[str, int] = {}
    total = agree = 0
    for day in sessions:
        index = bisect.bisect_right(dates, day) - 1
        theirs = members[index] if index >= 0 else frozenset()
        ours = set(pit.loc[(pit["valid_from"] <= day) & (pit["valid_to"] >= day), "code"])
        total += len(ours)
        agree += len(ours & theirs)
        for code in ours - theirs:
            lseg_only[code] = lseg_only.get(code, 0) + 1
        for code in theirs - ours:
            eodhd_only[code] = eodhd_only.get(code, 0) + 1

    def top(counts: dict[str, int]) -> dict[str, int]:
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:40])

    return {
        "sessions": len(sessions),
        "lseg_member_days": total,
        "member_days_in_both": agree,
        "share_in_both": agree / total if total else math.nan,
        "lseg_only_member_days": sum(lseg_only.values()),
        "eodhd_only_member_days": sum(eodhd_only.values()),
        "lseg_only_by_ticker": top(lseg_only),
        "eodhd_only_by_ticker": top(eodhd_only),
    }


def run_membership(args: argparse.Namespace) -> int:
    snapshot = read_input_manifest(args.reference_manifest, expected_sha256=args.reference_sha256)
    records = {Path(record.path).name: record for record in snapshot.manifest.files}
    record = records.get(args.intervals.name)
    if record is None:
        raise SystemExit(f"{args.intervals.name} is not recorded in {args.reference_manifest}")
    if args.intervals.stat().st_size != record.size_bytes or _sha256(args.intervals) != (
        record.sha256
    ):
        raise SystemExit(f"{args.intervals} differs from its record in {args.reference_manifest}")
    pit = pit_from_intervals(pd.read_csv(args.intervals, dtype=str))
    out = args.output_dir / "constituents" / f"{args.prefix}_pit_universe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pit.to_csv(out, index=False, lineterminator="\n")
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intervals": args.intervals.name,
        "intervals_sha256": record.sha256,
        "reference_manifest": args.reference_manifest.name,
        "reference_manifest_sha256": snapshot.sha256,
        "pit_universe": out.name,
        "pit_universe_sha256": _sha256(out),
        "rows": len(pit),
        "identifiers": int(pit["kdcode"].nunique()),
        "delisted_identifiers": int(
            pit["kdcode"].drop_duplicates().str.contains("^", regex=False).sum()
        ),
        "first_day": str(pit["valid_from"].min()),
        "last_day": str(pit["valid_to"].max()),
    }
    sessions = pd.bdate_range(summary["first_day"], summary["last_day"]).strftime("%Y-%m-%d")
    counts = [
        int(((pit["valid_from"] <= day) & (pit["valid_to"] >= day)).sum()) for day in sessions
    ]
    summary["members_per_weekday"] = {
        "min": min(counts),
        "median": float(np.median(counts)),
        "max": max(counts),
    }
    if args.constituents is not None:
        payload = json.loads(args.constituents.read_text(encoding="utf-8"))
        dates, members = eodhd_snapshots(payload)
        summary["eodhd_constituents"] = args.constituents.name
        summary["eodhd_constituents_sha256"] = _sha256(args.constituents)
        summary["eodhd_agreement"] = membership_agreement(pit, dates, members, list(sessions))
    report = args.output_dir / f"{args.prefix}_membership_check.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "eodhd_agreement"}, indent=2))
    if "eodhd_agreement" in summary:
        print(f"EODHD agreement: {summary['eodhd_agreement']['share_in_both']:.4f} of member-days")
    return 0


# ---------------------------------------------------------------------------
# Checks on a pulled panel
# ---------------------------------------------------------------------------


def member_mask(dates: pd.Series, spans: pd.DataFrame) -> np.ndarray:
    """Which dates fall inside any of one name's membership intervals."""
    inside = np.zeros(len(dates), dtype=bool)
    values = dates.to_numpy()
    for start, end in zip(spans["valid_from"], spans["valid_to"], strict=True):
        inside |= (values >= start) & (values <= end)
    return inside


def mcap_pairs(
    closes: pd.DataFrame, caps: pd.DataFrame, spans: pd.DataFrame, month_ends: list[str]
) -> list[dict[str, Any]]:
    """Month-to-month log changes in LSEG market cap and EODHD close, and their difference.

    ``closes`` has ``dt, close`` for one name; ``caps`` has ``as_of_date,
    company_market_cap``. Only month-ends inside membership count, each matched
    to the last panel close at most ``MCAP_MATCH_DAYS`` before it, and a pair is
    only formed between month-ends adjacent in ``month_ends`` (every as-of date
    of the metadata file), so a pair never spans a gap in membership. The
    difference, ``share_step``, is the change in implied share count.
    """
    caps = caps[member_mask(caps["as_of_date"], spans)].sort_values("as_of_date")
    if caps.empty or closes.empty:
        return []
    days = closes["dt"].to_numpy()
    values = closes["close"].to_numpy(dtype=float)
    matched = []
    for as_of, cap in zip(caps["as_of_date"], caps["company_market_cap"], strict=True):
        index = int(np.searchsorted(days, as_of, side="right")) - 1
        if index < 0 or not cap > 0 or not values[index] > 0:
            matched.append(None)
            continue
        gap = (pd.Timestamp(as_of) - pd.Timestamp(days[index])).days
        matched.append(values[index] if gap <= MCAP_MATCH_DAYS else None)
    as_of = caps["as_of_date"].tolist()
    cap_values = caps["company_market_cap"].astype(float).tolist()
    position = {day: i for i, day in enumerate(month_ends)}
    pairs = []
    for i in range(1, len(as_of)):
        if matched[i] is None or matched[i - 1] is None:
            continue
        if position[as_of[i]] != position[as_of[i - 1]] + 1:
            continue
        cap_change = math.log(cap_values[i] / cap_values[i - 1])
        close_change = math.log(matched[i] / matched[i - 1])
        pairs.append(
            {
                "from": as_of[i - 1],
                "to": as_of[i],
                "share_step": round(cap_change - close_change, 4),
                "mcap_log_change": round(cap_change, 4),
                "close_log_change": round(close_change, 4),
            }
        )
    return pairs


def price_steps(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pairs where the close moved and the market cap did not follow.

    A stock merger or a buyback moves the share count with the price steady, so
    a share step alone is LSEG's share count updating. A wrong series, a missed
    split or a bad splice moves the EODHD close by itself: both a large share
    step and a large close change.
    """
    return [
        p
        for p in pairs
        if abs(p["share_step"]) > MCAP_STEP_LIMIT and abs(p["close_log_change"]) > PRICE_MOVE_LIMIT
    ]


def extreme_moves(closes: pd.DataFrame, spans: pd.DataFrame) -> list[dict[str, Any]]:
    """One-day moves inside membership beyond the limits."""
    if len(closes) < 2:
        return []
    change = closes["close"].astype(float).pct_change()
    inside = member_mask(closes["dt"], spans)
    flagged = (change > RISE_LIMIT) | (change < FALL_LIMIT)
    rows = []
    for position in np.flatnonzero(flagged.to_numpy() & inside):
        rows.append(
            {
                "dt": str(closes["dt"].iloc[position]),
                "prev_dt": str(closes["dt"].iloc[position - 1]),
                "move": round(float(change.iloc[position]), 4),
            }
        )
    return rows


def corroborating_pair(move: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The month whose market cap moved with this one-day move, if LSEG recorded it.

    The move is corroborated when it falls inside a month pair whose share step
    is within ``MCAP_STEP_LIMIT``: LSEG's market cap made the same move.
    """
    for pair in pairs:
        if pair["from"] < move["dt"] <= pair["to"] and abs(pair["share_step"]) <= MCAP_STEP_LIMIT:
            return pair
    return None


def delisting_month_gap(kdcode: str, last_dt: str) -> int | None:
    """Months from the panel's last row to the month a ``^`` code names (``^J23`` is 2023-10)."""
    if DELISTED not in kdcode:
        return None
    code = kdcode.split(DELISTED, 1)[1]
    if len(code) != 3 or code[0] not in MONTH_LETTERS or not code[1:].isdigit():
        raise SystemExit(f"{kdcode}: cannot read the delisting code")
    month = MONTH_LETTERS.index(code[0]) + 1
    year = 2000 + int(code[1:])
    last = pd.Timestamp(last_dt)
    return (year * 12 + month) - (last.year * 12 + last.month)


def missing_sessions(
    dates: pd.Series, calendar: np.ndarray, spans: pd.DataFrame, *, delisted: bool
) -> tuple[list[str], list[str]]:
    """Calendar sessions inside membership with no row: flagged, and after a delisted name's last trade.

    An index removal is dated a few sessions after a delisted company's last
    trade, so sessions after its last row are expected and not flagged.
    """
    have = set(dates)
    needed = calendar[member_mask(pd.Series(calendar), spans)]
    missing = [str(day) for day in needed if day not in have]
    if delisted and len(dates):
        last = str(dates.max())
        return [d for d in missing if d <= last], [d for d in missing if d > last]
    return missing, []


def load_review(path: Path | None, kdcodes: set[str]) -> dict[str, dict[str, Any]]:
    """Per-identifier decisions; each needs a reason, and only known sections are read."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {})
    unknown = sorted(set(entries) - kdcodes)
    if unknown:
        raise SystemExit(f"Review names identifiers outside the universe: {unknown}")
    for kdcode, entry in entries.items():
        stray = set(entry) - set(REVIEW_SECTIONS)
        if stray:
            raise SystemExit(f"{kdcode}: unknown review sections {sorted(stray)}")
        for section in ("price_steps", "extreme_moves"):
            for key, reason in entry.get(section, {}).items():
                if not str(reason).strip():
                    raise SystemExit(f"{kdcode}: {section} {key} needs a reason")
        for section in ("missing_sessions", "identity", "delisting_month"):
            if section in entry and not str(entry[section]).strip():
                raise SystemExit(f"{kdcode}: {section} needs a reason")
    return entries


def check_name(
    kdcode: str,
    closes: pd.DataFrame,
    caps: pd.DataFrame,
    spans: pd.DataFrame,
    month_ends: list[str],
    calendar: np.ndarray,
    decision: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Every check on one name, and the flags its review entry does not decide."""
    pairs = mcap_pairs(closes, caps, spans, month_ends)
    median_step = float(np.median([abs(p["share_step"]) for p in pairs])) if pairs else None
    steps = price_steps(pairs)
    moves = extreme_moves(closes, spans)
    for move in moves:
        move["corroborated_by"] = corroborating_pair(move, pairs)
    delisted = DELISTED in kdcode
    missing, after_last = missing_sessions(closes["dt"], calendar, spans, delisted=delisted)
    gap = delisting_month_gap(kdcode, str(closes["dt"].max())) if len(closes) else None
    identity = "mcap" if len(pairs) >= MIN_IDENTITY_PAIRS else None
    if identity is None and gap is not None and abs(gap) <= 1:
        identity = "delisting_month"
    entry = {
        "rows": len(closes),
        "mcap_pairs": len(pairs),
        "mcap_median_abs_share_step": median_step,
        "share_steps_over_limit": [p for p in pairs if abs(p["share_step"]) > MCAP_STEP_LIMIT],
        "price_steps": steps,
        "extreme_moves": moves,
        "delisting_month_gap": gap,
        "identity_evidence": identity,
        "missing_sessions": len(missing),
        "missing_examples": missing[:10],
        "missing_after_last_trade": after_last,
    }
    flags: list[dict[str, Any]] = []
    if not len(closes):
        return entry, flags  # the export reports a name with no rows, or declared it unavailable
    if len(pairs) >= MIN_IDENTITY_PAIRS and median_step > MCAP_MEDIAN_LIMIT:
        flags.append({"check": "mcap_median", "median": median_step})
    for step in steps:
        if step["to"] not in decision.get("price_steps", {}):
            flags.append({"check": "price_step", **step})
    for move in moves:
        if move["corroborated_by"] is None and move["dt"] not in decision.get("extreme_moves", {}):
            flags.append({"check": "extreme_move", **{k: move[k] for k in ("dt", "move")}})
    if missing and "missing_sessions" not in decision:
        flags.append({"check": "missing_sessions", "count": len(missing), "first": missing[0]})
    if gap is not None and abs(gap) > 1 and "delisting_month" not in decision:
        flags.append({"check": "delisting_month", "gap_months": gap})
    if identity is None and "identity" not in decision:
        flags.append({"check": "identity", "mcap_pairs": len(pairs)})
    return entry, flags


def run_check(args: argparse.Namespace) -> int:
    panel = pd.read_csv(args.panel, usecols=["kdcode", "dt", "close"], dtype={"kdcode": str})
    panel["dt"] = panel["dt"].astype(str)
    panel = panel.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    pit = pd.read_csv(args.pit_universe, dtype=str)
    meta = pd.read_csv(
        args.metadata, usecols=["as_of_date", "kdcode", "company_market_cap"], dtype={"kdcode": str}
    )
    meta["as_of_date"] = meta["as_of_date"].astype(str)
    kdcodes = set(pit["kdcode"])
    review = load_review(args.review, kdcodes)
    counts = panel.groupby("dt").size()
    calendar = np.array(sorted(counts.index[counts >= CALENDAR_MIN_NAMES]))
    month_ends = sorted(meta["as_of_date"].unique())
    by_code = dict(tuple(panel.groupby("kdcode")))
    caps_by_code = dict(tuple(meta.groupby("kdcode")))
    spans_by_code = dict(tuple(pit.groupby("kdcode")))
    empty_closes = pd.DataFrame({"dt": pd.Series(dtype=str), "close": pd.Series(dtype=float)})
    empty_caps = pd.DataFrame(
        {"as_of_date": pd.Series(dtype=str), "company_market_cap": pd.Series(dtype=float)}
    )
    names: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    for kdcode in sorted(kdcodes):
        entry, flags = check_name(
            kdcode,
            by_code.get(kdcode, empty_closes),
            caps_by_code.get(kdcode, empty_caps),
            spans_by_code[kdcode],
            month_ends,
            calendar,
            review.get(kdcode, {}),
        )
        names[kdcode] = entry
        unresolved += [{"kdcode": kdcode, **flag} for flag in flags]
    priced = [v for v in names.values() if v["rows"]]
    evidence: dict[str, int] = {}
    for v in priced:
        key = v["identity_evidence"] or "review"
        evidence[key] = evidence.get(key, 0) + 1
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": args.panel.name,
        "panel_sha256": _sha256(args.panel),
        "pit_universe": args.pit_universe.name,
        "pit_universe_sha256": _sha256(args.pit_universe),
        "metadata": args.metadata.name,
        "metadata_sha256": _sha256(args.metadata),
        "review": args.review.name if args.review else None,
        "review_sha256": _sha256(args.review) if args.review else None,
        "limits": {
            "share_step": MCAP_STEP_LIMIT,
            "price_move": PRICE_MOVE_LIMIT,
            "median_share_step": MCAP_MEDIAN_LIMIT,
            "min_identity_pairs": MIN_IDENTITY_PAIRS,
            "rise": RISE_LIMIT,
            "fall": FALL_LIMIT,
            "calendar_min_names": CALENDAR_MIN_NAMES,
        },
        "identifiers": len(kdcodes),
        "identifiers_with_rows": len(priced),
        "identity_evidence": dict(sorted(evidence.items())),
        "mcap_pairs": int(sum(v["mcap_pairs"] for v in priced)),
        "flags": {
            "price_steps": int(sum(len(v["price_steps"]) for v in priced)),
            "extreme_moves": int(sum(len(v["extreme_moves"]) for v in priced)),
            "extreme_moves_corroborated": int(
                sum(1 for v in priced for m in v["extreme_moves"] if m["corroborated_by"])
            ),
            "names_missing_sessions": int(sum(1 for v in priced if v["missing_sessions"])),
        },
        "unresolved": unresolved,
        "names": names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("names", "unresolved")}, indent=2)
    )
    for item in unresolved:
        print("UNRESOLVED", json.dumps(item))
    if unresolved:
        print(f"{len(unresolved)} flags need a review decision")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    member = commands.add_parser("membership", help="Write the membership file")
    member.add_argument("--intervals", type=Path, required=True)
    member.add_argument("--reference-manifest", type=Path, required=True)
    member.add_argument("--reference-sha256")
    member.add_argument("--constituents", type=Path, help="EODHD GSPC.INDX fundamentals JSON")
    member.add_argument("--output-dir", type=Path, required=True)
    member.add_argument("--prefix", required=True)
    check = commands.add_parser("check", help="Check a pulled panel")
    check.add_argument("--panel", type=Path, required=True)
    check.add_argument("--pit-universe", type=Path, required=True)
    check.add_argument("--metadata", type=Path, required=True)
    check.add_argument("--review", type=Path)
    check.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_membership(args) if args.command == "membership" else run_check(args)


if __name__ == "__main__":
    sys.exit(main())
