"""Draw the small-caps universe: a random point-in-time sample of Russell 2000 names (#379).

The small-caps test runs the recipe unchanged on Russell 2000 companies instead of
the S&P 500's largest. Membership comes from the iShares Russell 2000 ETF's SEC
holdings filings, as ``scripts/data/build_r2k_members.py`` (#317) builds it; this
script reads that builder's output and writes what the price export and the
pipeline read::

    python -m scripts.data.build_r2k_sample_universe \\
        --holdings <r2k>/holdings_matched.csv.gz --membership <r2k>/r2k_membership.csv \\
        --price-cache <r2k_cache>/eod --output-dir /content/r2k_sample \\
        --prefix r2k_sample_400_20200102_20251231

Steps:

1. **Frame.** Every company (SEC CIK) with a membership spell that overlaps
   ``--window-start`` to ``--window-end`` and at least one holdings row whose
   price agreed with an EODHD close (``how == "price"``). A company with no such
   row cannot be tied to a price series with confidence, so it is left out and
   counted.
2. **Draw.** ``random.Random(seed).sample`` of the frame's sorted CIKs. The draw
   reads nothing about prices or returns. Its size is the frame's size scaled by
   ``--target-members`` over the median number of frame companies held per report
   date in the window, so that about that many are members on a typical day.
3. **Membership file.** Each drawn company's spells, clipped to the window end,
   as ``kdcode, valid_from, valid_to`` with ``kdcode = CIK<10 digits>``. The
   window end is also the file's export cutoff, so no row is open-ended.
4. **Symbol map.** The symbols the company's price-confirmed holdings rows were
   matched to, in report-date order, as ``export_eodhd_pit_prices`` segments. When
   the ticker changes, the earlier symbol's segment ends on its last close before
   the later symbol's first report date (read from ``--price-cache`` when given),
   and the later one starts the next day. When one company has two share classes
   held on the same report date, the larger holding by value is its symbol there.

What it writes under ``--output-dir``: ``constituents/<prefix>_pit_universe.csv``,
``eodhd_symbols_<prefix>.json`` and ``<prefix>_sample.json`` (the frame, the draw
and every count above).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

#: The symbol map schema ``mci_gru.data.eodhd_prices.parse_symbol_map`` reads.
SYMBOL_MAP_SCHEMA = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--holdings", type=Path, required=True, help="holdings_matched.csv.gz")
    parser.add_argument("--membership", type=Path, required=True, help="r2k_membership.csv")
    parser.add_argument(
        "--price-cache", type=Path, help="build_r2k_members' eod/ folder of <symbol>.csv.gz"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--window-start", default="2020-01-02")
    parser.add_argument("--window-end", default="2025-12-31")
    parser.add_argument("--target-members", type=int, default=400)
    parser.add_argument("--seed", type=int, default=379)
    return parser.parse_args(argv)


def parse_spells(text: str) -> list[tuple[str, str | None]]:
    """``"2019-09-30:2020-06-29;2021-06-30:"`` as ``[(start, end or None), ...]``."""
    spells = []
    for part in str(text).split(";"):
        if not part:
            continue
        start, _, end = part.partition(":")
        spells.append((start, end or None))
    return spells


def overlaps(spells: list[tuple[str, str | None]], start: str, end: str) -> bool:
    return any(s <= end and (e is None or e >= start) for s, e in spells)


def frame_companies(
    membership: pd.DataFrame, holdings: pd.DataFrame, start: str, end: str
) -> tuple[list[str], dict[str, int]]:
    """CIKs with a spell in the window and a price-confirmed holding, and what was left out."""
    in_window = {
        str(row.cik).zfill(10)
        for row in membership.itertuples()
        if overlaps(parse_spells(row.spells), start, end)
    }
    confirmed = set(holdings.loc[holdings["how"] == "price", "cik"])
    frame = sorted(in_window & confirmed)
    return frame, {
        "with_spell_in_window": len(in_window),
        "left_out_no_price_confirmed_holding": len(in_window - confirmed),
        "frame": len(frame),
    }


def median_members(holdings: pd.DataFrame, frame: list[str], start: str, end: str) -> float:
    """Median count of frame companies held per report date inside the window."""
    held = holdings[holdings["cik"].isin(set(frame))]
    held = held[(held["snap"] >= start) & (held["snap"] <= end)]
    counts = held.groupby("snap")["cik"].nunique()
    if counts.empty:
        raise SystemExit(f"No report date between {start} and {end} holds a frame company")
    return float(counts.median())


def draw_size(frame_size: int, typical_members: float, target: int) -> int:
    """How many to draw so that about *target* of them are members on a typical day."""
    return min(frame_size, round(frame_size * target / typical_members))


def draw(ciks: list[str], seed: int, size: int) -> list[str]:
    """``size`` CIKs drawn with ``seed``, in draw order."""
    return random.Random(seed).sample(sorted(ciks), size)


def kdcode_for(cik: str) -> str:
    return f"CIK{str(cik).zfill(10)}"


def pit_rows(membership: pd.DataFrame, drawn: list[str], end: str) -> pd.DataFrame:
    """The drawn companies' spells, each clipped to *end*; spells after *end* are dropped."""
    spells_by_cik = {str(row.cik).zfill(10): row.spells for row in membership.itertuples()}
    rows = []
    for cik in sorted(drawn):
        for start, stop in parse_spells(spells_by_cik[cik]):
            if start > end:
                continue
            stop = end if stop is None or stop > end else stop
            rows.append({"kdcode": kdcode_for(cik), "valid_from": start, "valid_to": stop})
    return pd.DataFrame(rows, columns=["kdcode", "valid_from", "valid_to"])


def symbol_runs(rows: pd.DataFrame) -> list[dict[str, str]]:
    """Consecutive report dates on one symbol, from one company's price-confirmed rows.

    Returns ``[{"symbol", "first", "last"}, ...]`` in date order. Where two rows share
    a report date (two share classes), the larger holding by value counts.
    """
    confirmed = rows[(rows["how"] == "price") & (rows["symbol"] != "")]
    primary = confirmed.sort_values(["snap", "value"], ascending=[True, False]).drop_duplicates(
        "snap"
    )
    runs: list[dict[str, str]] = []
    for snap, symbol in zip(primary["snap"], primary["symbol"], strict=True):
        if runs and runs[-1]["symbol"] == symbol:
            runs[-1]["last"] = snap
        else:
            runs.append({"symbol": symbol, "first": snap, "last": snap})
    return runs


def _next_day(day: str) -> str:
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()


def segments_for(runs: list[dict[str, str]], last_close) -> list[dict[str, Any]]:
    """``export_eodhd_pit_prices`` segments for one company's symbol runs.

    ``last_close(symbol, before)`` is the symbol's last trading date strictly before
    *before*, or ``None`` when unknown; the earlier symbol then ends the day before
    the later symbol's first report date.
    """
    segments: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        segment: dict[str, Any] = {"candidates": [run["symbol"]]}
        if index > 0:
            segment["start"] = _next_day(segments[-1]["end"])
        if index < len(runs) - 1:
            later_first = runs[index + 1]["first"]
            end = last_close(run["symbol"], later_first)
            if end is None or end < run["last"]:
                end = (pd.Timestamp(later_first) - pd.Timedelta(days=1)).date().isoformat()
            segment["end"] = end
        segments.append(segment)
    return segments


def price_cache_reader(folder: Path | None):
    """``last_close(symbol, before)`` over build_r2k_members' cached EOD files."""
    cache: dict[str, pd.Series | None] = {}

    def dates(symbol: str) -> pd.Series | None:
        if symbol not in cache:
            path = folder / f"{symbol}.csv.gz" if folder else None
            series = None
            if path is not None and path.exists():
                body = gzip.decompress(path.read_bytes())
                if body.strip():
                    table = pd.read_csv(io.BytesIO(body))
                    if "Date" in table:
                        series = table["Date"].astype(str).sort_values()
            cache[symbol] = series
        return cache[symbol]

    def last_close(symbol: str, before: str) -> str | None:
        series = dates(symbol)
        if series is None:
            return None
        earlier = series[series < before]
        return str(earlier.iloc[-1]) if len(earlier) else None

    return last_close


def symbol_map(holdings: pd.DataFrame, drawn: list[str], last_close) -> dict[str, Any]:
    overrides = {}
    by_cik = dict(tuple(holdings[holdings["cik"].isin(set(drawn))].groupby("cik")))
    for cik in sorted(drawn):
        runs = symbol_runs(by_cik[cik])
        name = by_cik[cik].sort_values("snap")["name"].iloc[-1]
        overrides[kdcode_for(cik)] = {
            "note": f"{name}; symbols from IWM holdings rows price-matched to EODHD (#317)",
            "segments": segments_for(runs, last_close),
        }
    return {"schema": SYMBOL_MAP_SCHEMA, "overrides": overrides}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    holdings = pd.read_csv(args.holdings, dtype=str, keep_default_na=False)
    holdings["cik"] = holdings["cik"].map(lambda c: c.zfill(10) if c else "")
    holdings["value"] = pd.to_numeric(holdings["value"], errors="coerce")
    membership = pd.read_csv(args.membership, dtype=str, keep_default_na=False)
    start, end = args.window_start, args.window_end

    frame, counts = frame_companies(membership, holdings, start, end)
    typical = median_members(holdings, frame, start, end)
    size = draw_size(len(frame), typical, args.target_members)
    drawn = draw(frame, args.seed, size)

    out = args.output_dir
    pit_path = out / "constituents" / f"{args.prefix}_pit_universe.csv"
    pit_path.parent.mkdir(parents=True, exist_ok=True)
    pit = pit_rows(membership, drawn, end)
    pit.to_csv(pit_path, index=False)
    map_path = out / f"eodhd_symbols_{args.prefix}.json"
    mapping = symbol_map(holdings, drawn, price_cache_reader(args.price_cache))
    map_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sessions = pd.bdate_range(start, end)
    active = [
        int(((pit["valid_from"] <= d) & (pit["valid_to"] >= d)).sum())
        for d in sessions.strftime("%Y-%m-%d")
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/data/build_r2k_members.py output (#317): IWM SEC holdings, "
        "price-matched to EODHD, CIK-checked against SEC",
        "holdings": args.holdings.name,
        "holdings_sha256": _sha256(args.holdings),
        "membership": args.membership.name,
        "membership_sha256": _sha256(args.membership),
        "window": [start, end],
        "seed": args.seed,
        "target_members": args.target_members,
        "counts": counts,
        "median_frame_members_per_report_date": typical,
        "drawn": size,
        "draw_order": [kdcode_for(c) for c in drawn],
        "members_per_weekday": {
            "min": min(active),
            "median": float(pd.Series(active).median()),
            "max": max(active),
        },
        "companies_with_ticker_changes": sum(
            1 for entry in mapping["overrides"].values() if len(entry["segments"]) > 1
        ),
        "pit_universe_sha256": _sha256(pit_path),
        "symbol_map_sha256": _sha256(map_path),
    }
    report_path = out / f"{args.prefix}_sample.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "draw_order"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
