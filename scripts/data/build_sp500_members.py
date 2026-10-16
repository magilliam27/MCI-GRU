"""Build the S&P 500 company table for the members' 10-K pull, every mapping checked by name (#309).

Experiment E7 reads the 10-Ks of companies while they were in the S&P 500. The
membership comes from dated member lists of tickers; this script turns each
ticker's runs in the index into companies with SEC CIKs::

    python -m scripts.data.build_sp500_members \\
        --user-agent "Your Name you@example.com" --cache-dir /content/sec_cache \\
        --member-lists sp500_historical_components.csv \\
        --wiki-current list_of_sp500_companies.html \\
        --wiki-changes historical_components_of_the_sp500.html \\
        --sec-tickers company_tickers.json --cik-lookup cik-lookup-data.txt \\
        --manual manual_names.csv --output-dir /content/sp500_members

Inputs, each saved so a run can be repeated:

- ``--member-lists``: one row per change date, ``date`` and comma-separated
  ``tickers`` (github.com/fja05680/sp500).
- ``--wiki-changes``: Wikipedia's "Historical components of the S&P 500" page
  as HTML. Its changes after the last member list extend the lists, and its
  added and removed names name each ticker.
- ``--wiki-current``: Wikipedia's "List of S&P 500 companies" page as HTML,
  whose table gives each current member's name and CIK.
- ``--sec-tickers`` (SEC ``company_tickers.json``) and ``--cik-lookup`` (SEC
  ``cik-lookup-data.txt``, every name EDGAR has held, with its CIK) propose CIKs.
- ``--manual``: ``ticker``, ``start``, ``name``, ``cik`` (``;`` between several
  in turn) and ``note``, for runs the pages do not name or name differently from
  SEC.

For each run that overlaps ``--window-start`` onwards, the candidate CIKs are,
in order: the manual row's, Wikipedia's, SEC's for the ticker, then each CIK
EDGAR lists under one of the run's names, then names starting with them. A CIK
is confirmed when it filed a 10-K during the run while going by a name that
matches one of the run's names (``mci_gru.filings.membership.names_match``).
The first confirmed CIK is the company; another confirmed CIK joins it only as
its successor (``successors``), and the manual row's CIKs always do.

What it writes under ``--output-dir``:

- ``sp500_ticker_runs.csv``: one row per ticker run: dates, the names used,
  the CIKs chosen, SEC's names for them, how they were found (``method``),
  ``status`` (``confirmed``, ``no 10-K while a member``, ``unconfirmed``) and
  every candidate tried.
- ``sp500_companies.csv``: one row per company (one chain of CIKs): ``key``,
  ``name``, ``ciks``, ``tickers``, ``spells`` and ``release``: ``r1`` when its
  last CIK is in SEC's current ticker file, ``r2`` otherwise.
- ``sp500_members.meta.json``: inputs, counts and the runs not confirmed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.edgar import SUBMISSIONS_FILE_URL, SUBMISSIONS_URL
from mci_gru.filings.membership import (
    Spell,
    apply_changes,
    name_key,
    names_match,
    names_on,
    spells,
    successors,
)
from scripts.data.export_sec_risk_factors import (
    EdgarClient,
    Transport,
    _git_commit,
    _urllib_transport,
)

ANNUAL_FORMS = ("10-K", "10-KT")
#: Reports a company must have filed under a ticker to have held part of its run.
MIN_HANDOVER_REPORTS = 2
#: Name-prefix candidates tried per name, at most.
MAX_PREFIX_CANDIDATES = 40
#: A name shorter than this is not searched by prefix.
MIN_PREFIX_CHARS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--member-lists", type=Path, required=True)
    parser.add_argument("--wiki-current", type=Path, required=True)
    parser.add_argument("--wiki-changes", type=Path, required=True)
    parser.add_argument("--sec-tickers", type=Path, required=True)
    parser.add_argument("--cik-lookup", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--window-start", default="2013-01-01")
    parser.add_argument("--as-of", default=None, help="Last day counted (default today, UTC)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


def read_member_lists(path: Path) -> list[tuple[pd.Timestamp, frozenset[str]]]:
    table = pd.read_csv(path, dtype=str)
    return [
        (pd.Timestamp(date), frozenset(ticker for ticker in tickers.split(",") if ticker))
        for date, tickers in zip(table["date"], table["tickers"], strict=True)
    ]


class _Row(list):
    """A table row's cell texts; ``header`` when every cell is a ``th``."""

    header = False


class _FirstTable(HTMLParser):
    """The first ``table`` in a page as rows of cell text, spans filled in."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_Row] = []
        self._depth = 0
        self._done = False
        self._cells: list[tuple[str, int, int, bool]] | None = None
        self._text: list[str] | None = None
        self._cell: tuple[int, int, bool] = (1, 1, False)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        if tag == "table":
            self._depth += 1
        elif self._depth == 1 and tag == "tr":
            self._cells = []
        elif self._depth == 1 and tag in ("td", "th") and self._cells is not None:
            spans = dict(attrs)
            self._cell = (
                _span_of(spans.get("rowspan")),
                _span_of(spans.get("colspan")),
                tag == "th",
            )
            self._text = []
        elif self._text is not None and tag == "br":
            self._text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._done:
            return
        if tag == "table":
            self._depth -= 1
            self._done = self._depth == 0
        elif self._depth == 1 and tag in ("td", "th") and self._text is not None:
            assert self._cells is not None
            text = _SPACES.sub(" ", "".join(self._text)).strip()
            self._cells.append((text, *self._cell))
            self._text = None
        elif self._depth == 1 and tag == "tr" and self._cells is not None:
            self._place(self._cells)
            self._cells = None

    def handle_data(self, data: str) -> None:
        if self._text is not None and self._depth == 1:
            self._text.append(data)

    def _place(self, cells: list[tuple[str, int, int, bool]]) -> None:
        row_index = len(self.rows)
        while len(self._grid) <= row_index:
            self._grid.append({})
        filled = self._grid[row_index]
        column = 0
        for text, rows, columns, is_header in cells:
            while column in filled:
                column += 1
            for down in range(rows):
                while len(self._grid) <= row_index + down:
                    self._grid.append({})
                for across in range(columns):
                    self._grid[row_index + down][column + across] = (text, is_header)
            column += columns
        self.rows.append(_Row())
        self._finish(row_index)

    def _finish(self, row_index: int) -> None:
        cells = self._grid[row_index]
        row = _Row(cells[column][0] for column in sorted(cells))
        row.header = bool(cells) and all(is_header for _, is_header in cells.values())
        self.rows[row_index] = row

    _grid: list[dict[int, tuple[str, bool]]]

    def reset(self) -> None:
        super().reset()
        self._grid = []


_SPACES = re.compile(r"\s+")


def _span_of(value: str | None) -> int:
    digits = re.match(r"\d+", value or "")
    return max(int(digits.group()), 1) if digits else 1


def first_table(html: str) -> list[_Row]:
    """The first table in ``html`` as rows of cell text, with row and column spans filled in."""
    reader = _FirstTable()
    reader.feed(html)
    reader.close()
    return reader.rows


def read_wiki_changes(path: Path) -> pd.DataFrame:
    """Wikipedia's changes table: ``date``, ``added``, ``added_name``, ``removed``, ``removed_name``."""
    rows = [row[:5] for row in first_table(path.read_text(encoding="utf-8")) if not row.header]
    table = pd.DataFrame(
        [row + [""] * (5 - len(row)) for row in rows],
        columns=["date", "added", "added_name", "removed", "removed_name"],
    ).replace("", None)
    table["date"] = pd.to_datetime(table["date"])
    return table.astype(object).where(table.notna(), None)


def read_wiki_current(path: Path) -> dict[str, tuple[str, str]]:
    """Ticker to (name, CIK) from Wikipedia's current member table."""
    rows = first_table(path.read_text(encoding="utf-8"))
    header = rows[0]
    symbol, security, cik = (header.index(name) for name in ("Symbol", "Security", "CIK"))
    return {
        row[symbol]: (row[security], f"{int(row[cik]):010d}") for row in rows[1:] if not row.header
    }


def read_cik_lookup(path: Path) -> dict[str, set[str]]:
    """Name key to the CIKs EDGAR lists under a name with that key."""
    by_key: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").rsplit(":", 2)
            if len(parts) == 3 and parts[1].isdigit():
                by_key[name_key(parts[0])].add(parts[1].zfill(10))
    return by_key


def _run_names(
    run: Spell,
    current: dict[str, tuple[str, str]],
    changes: pd.DataFrame,
    manual: dict[tuple[str, str], dict[str, str]],
) -> list[tuple[str, str]]:
    """``(source, name)`` for a ticker run, nearest sources first."""
    names: list[tuple[str, str]] = []
    row = manual.get((run.ticker, run.start.date().isoformat()))
    if row and row["name"]:
        names += [("manual", name.strip()) for name in row["name"].split(";") if name.strip()]
    if run.end is None and run.ticker in current:
        names.append(("wiki_current", current[run.ticker][0]))
    near = pd.Timedelta(days=10)
    for change in changes.itertuples(index=False):
        if (
            change.added == run.ticker
            and change.added_name
            and abs(change.date - run.start) <= near
        ):
            names.append(("wiki_added", change.added_name))
        if (
            run.end is not None
            and change.removed == run.ticker
            and change.removed_name
            and abs(change.date - run.end) <= near
        ):
            names.append(("wiki_removed", change.removed_name))
    if not names:
        for change in changes.itertuples(index=False):
            if change.added == run.ticker and change.added_name:
                names.append(("wiki_added_other_date", change.added_name))
            if change.removed == run.ticker and change.removed_name:
                names.append(("wiki_removed_other_date", change.removed_name))
    return list(dict.fromkeys(names))


class Filers:
    """SEC submissions per CIK, read once: names and 10-K acceptance dates."""

    def __init__(self, client: EdgarClient) -> None:
        self.client = client
        self._seen: dict[str, dict[str, Any] | None] = {}

    def get(self, cik: str) -> dict[str, Any] | None:
        if cik not in self._seen:
            try:
                main = self.client.get_json(SUBMISSIONS_URL.format(cik=cik))
            except Exception:  # noqa: BLE001 - a CIK SEC does not serve is no candidate
                self._seen[cik] = None
                return None
            reports = _annual_reports(main["filings"]["recent"])
            for entry in main["filings"].get("files", []):
                older = self.client.get_json(SUBMISSIONS_FILE_URL.format(name=entry["name"]))
                reports += _annual_reports(older)
            reports.sort()
            self._seen[cik] = {
                **main,
                "annual": reports,
                "annual_dates": [when for when, _ in reports],
            }
        return self._seen[cik]


def _annual_reports(block: dict[str, Any]) -> list[tuple[pd.Timestamp, str]]:
    """``(acceptance date, accession)`` of each 10-K or 10-KT in a submissions block."""
    return [
        (pd.Timestamp(str(accepted)[:10]), str(accession))
        for accepted, form, accession in zip(
            block["acceptanceDateTime"], block["form"], block["accessionNumber"], strict=True
        )
        if form in ANNUAL_FORMS
    ]


def confirm(
    filer: dict[str, Any], names: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, str]]:
    """The CIK's 10-Ks in ``[start, end)`` filed under a name matching one of ``names``."""
    return [
        (when, accession)
        for when, accession in filer["annual"]
        if start <= when < end
        and any(names_match(name, held) for name in names for held in names_on(filer, when))
    ]


def _filed_between(filer: dict[str, Any] | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return filer is not None and any(start <= when < end for when in filer["annual_dates"])


def _span(
    cik: str, filer: dict[str, Any]
) -> tuple[str, pd.Timestamp, pd.Timestamp, list[str], list[str]]:
    """A CIK's first and last 10-K ever, with its names on those days."""
    first, last = filer["annual_dates"][0], filer["annual_dates"][-1]
    return cik, first, last, names_on(filer, first), names_on(filer, last)


def _try(
    filers: Filers,
    tried: dict[str, tuple[str, list[pd.Timestamp], str, int]],
    how: str,
    cik: str,
    names: list[str],
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> None:
    """Record a candidate CIK once: how it was found and the 10-Ks it counts."""
    if cik in tried:
        return
    filer = filers.get(cik)
    if filer is None:
        tried[cik] = (how, [], "not served by SEC", 0)
        return
    counted = confirm(filer, names, *window)
    tried[cik] = (how, counted, str(filer.get("name", "")), len(filer["annual_dates"]))


def _handovers(
    run: Spell,
    chosen: list[str],
    confirmed: list[str],
    tried: dict[str, tuple[str, list[tuple[pd.Timestamp, str]], str, int]],
) -> list[tuple[Spell, str]]:
    """Parts of a run that belong to another company the ticker passed from or to.

    A confirmed CIK that was not chosen, with at least ``MIN_HANDOVER_REPORTS``
    reports of its own (not the chosen company's filed jointly) all before or
    all after the chosen company's, held the ticker for that part of the run:
    Chubb Corp until ACE took its name, Twenty-First Century Fox until Fox Corp.
    """
    held = {accession for cik in chosen for _, accession in tried[cik][1]}
    dates = [when for cik in chosen for when, _ in tried[cik][1]]
    if not dates:
        return []
    first, last = min(dates), max(dates)
    parts = []
    for cik in confirmed:
        if cik in chosen:
            continue
        own = [when for when, accession in tried[cik][1] if accession not in held]
        if len(own) < MIN_HANDOVER_REPORTS:
            continue
        if max(own) < first:
            parts.append((Spell(run.ticker, run.start, first), cik))
        elif min(own) > last:
            parts.append((Spell(run.ticker, last + pd.Timedelta(days=1), run.end), cik))
    return parts


def build(args: argparse.Namespace, client: EdgarClient) -> dict[str, Any]:
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC").tz_convert(None)
    window_start = pd.Timestamp(args.window_start)
    changes = read_wiki_changes(args.wiki_changes)
    lists = read_member_lists(args.member_lists)
    last_list = max(date for date, _ in lists)
    lists = apply_changes(
        lists,
        [
            (row.date, row.added, row.removed)
            for row in changes.itertuples(index=False)
            if row.date <= as_of
        ],
    )
    runs = [run for run in spells(lists) if run.end is None or run.end > window_start]
    current = read_wiki_current(args.wiki_current)
    manual_table = pd.read_csv(args.manual, dtype=str, keep_default_na=False)
    manual = {(row["ticker"], row["start"]): dict(row) for _, row in manual_table.iterrows()}
    sec_tickers = {
        str(entry["ticker"]): f"{int(entry['cik_str']):010d}"
        for entry in json.loads(args.sec_tickers.read_text(encoding="utf-8")).values()
    }
    listed_ciks = set(sec_tickers.values())
    by_key = read_cik_lookup(args.cik_lookup)
    by_first_word: dict[str, list[str]] = defaultdict(list)
    for key in by_key:
        if key:
            by_first_word[key.split()[0]].append(key)
    filers = Filers(client)

    run_rows: list[dict[str, Any]] = []
    chains: list[tuple[Spell, list[str]]] = []
    for run in runs:
        names = _run_names(run, current, changes, manual)
        plain = [name for _, name in names]
        lo = max(run.start, window_start)
        hi = run.end if run.end is not None else as_of + pd.Timedelta(days=1)
        candidates: list[tuple[str, str]] = []
        row = manual.get((run.ticker, run.start.date().isoformat()))
        manual_ciks = [cik.zfill(10) for cik in (row or {}).get("cik", "").split(";") if cik]
        candidates += [("manual", cik) for cik in manual_ciks]
        if run.end is None and run.ticker in current:
            candidates.append(("wiki_cik", current[run.ticker][1]))
        ticker = run.ticker.replace(".", "-")
        if ticker in sec_tickers:
            candidates.append(("sec_ticker", sec_tickers[ticker]))
        for name in plain:
            candidates += [("name_lookup", cik) for cik in sorted(by_key.get(name_key(name), ()))]

        tried: dict[str, tuple[str, list[pd.Timestamp], str, int]] = {}

        for how, cik in candidates:
            _try(filers, tried, how, cik, plain, (lo, hi))
        if not any(entry[1] for entry in tried.values()):
            for name in plain:
                key = name_key(name)
                if len(key) < MIN_PREFIX_CHARS:
                    continue
                found = sorted(
                    {
                        cik
                        for other in by_first_word.get(key.split()[0], ())
                        if other == key or other.startswith(key + " ")
                        for cik in by_key[other]
                    }
                )
                for cik in found[:MAX_PREFIX_CANDIDATES]:
                    _try(filers, tried, "name_prefix", cik, plain, (lo, hi))

        confirmed = [cik for cik, entry in tried.items() if entry[1]]
        # Any candidate may carry the company's reports before or after the
        # window (Eaton Corp's last 10-K is the pair partner of Eaton plc's first).
        spans = [
            _span(cik, filer)
            for cik, filer in ((cik, filers.get(cik)) for cik in tried)
            if filer is not None and filer["annual_dates"]
        ]
        if manual_ciks:
            # Every manual CIK must either be confirmed or have filed no 10-K in
            # the run (ExxonMobil Holdings had filed none by the time of the pull).
            agree = all(
                tried[cik][1] or not _filed_between(filers.get(cik), lo, hi) for cik in manual_ciks
            )
            counted = any(tried[cik][1] for cik in manual_ciks)
            chosen = manual_ciks if agree and counted else []
        elif confirmed:
            chosen = successors(confirmed[0], spans)
        else:
            chosen = []
        handovers = _handovers(run, chosen, confirmed, tried) if not manual_ciks else []
        for part, cik in handovers:
            chains.append((part, successors(cik, spans)))
        if (
            manual_ciks
            and not chosen
            and any(_filed_between(filers.get(cik), lo, hi) for cik in manual_ciks)
        ):
            status = "manual row not confirmed"
        elif chosen:
            status = "confirmed"
        elif any(
            any(names_match(name, held) for name in plain for held in names_on(filer, hi))
            for filer in (filers.get(cik) for cik in tried)
            if filer is not None
        ):
            status = "no 10-K while a member"
        else:
            status = "unconfirmed"
        run_rows.append(
            {
                "ticker": run.ticker,
                "start": run.start.date().isoformat(),
                "end": run.end.date().isoformat() if run.end is not None else "",
                "names": " | ".join(f"{source}: {name}" for source, name in names),
                "ciks": ";".join(chosen),
                "sec_names": " | ".join(tried[cik][2] for cik in chosen),
                "method": ";".join(dict.fromkeys(tried[cik][0] for cik in chosen)),
                "reports_counted": sum(len(tried[cik][1]) for cik in chosen),
                "handed_over": " ; ".join(
                    f"{cik} {part.start.date().isoformat()} to "
                    f"{part.end.date().isoformat() if part.end is not None else 'now'}"
                    for part, cik in handovers
                ),
                "status": status,
                "note": (row or {}).get("note", ""),
                "candidates": " ; ".join(
                    f"{how}:{cik}:{sec_name}:{len(counted)}"
                    for cik, (how, counted, sec_name, _) in tried.items()
                ),
            }
        )
        if chosen:
            chains.append((run, chosen))

    companies = _companies(chains, filers, listed_ciks)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(out / "sp500_ticker_runs.csv", index=False)
    pd.DataFrame(companies).to_csv(out / "sp500_companies.csv", index=False)
    status_counts = pd.Series([row["status"] for row in run_rows]).value_counts()
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.date().isoformat(),
        "window_start": args.window_start,
        "member_lists": str(args.member_lists),
        "member_lists_last_date": last_list.date().isoformat(),
        "inputs": {
            name: str(getattr(args, name))
            for name in ("wiki_current", "wiki_changes", "sec_tickers", "cik_lookup", "manual")
        },
        "repository_commit": _git_commit(),
        "network_requests": client.requests,
        "ticker_runs": len(run_rows),
        "runs_by_status": {str(key): int(value) for key, value in status_counts.items()},
        "companies": len(companies),
        "companies_by_release": dict(pd.Series([c["release"] for c in companies]).value_counts()),
        "runs_not_confirmed": [
            f"{row['ticker']} {row['start']} to {row['end'] or 'now'}: {row['status']}"
            for row in run_rows
            if row["status"] != "confirmed"
        ],
    }
    meta["companies_by_release"] = {
        key: int(value) for key, value in meta["companies_by_release"].items()
    }
    (out / "sp500_members.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _companies(
    chains: list[tuple[Spell, list[str]]], filers: Filers, listed_ciks: set[str]
) -> list[dict[str, Any]]:
    """One company per chain of CIKs; a CIK on several chains joins them."""
    parent: dict[str, str] = {}

    def find(cik: str) -> str:
        while parent.setdefault(cik, cik) != cik:
            cik = parent[cik]
        return cik

    for _, chain in chains:
        for cik in chain[1:]:
            parent[find(cik)] = find(chain[0])
    groups: dict[str, list[tuple[Spell, list[str]]]] = defaultdict(list)
    for run, chain in chains:
        groups[find(chain[0])].append((run, chain))
    companies = []
    keys: set[str] = set()
    for runs in groups.values():
        ciks: list[str] = []
        for _, chain in sorted(runs, key=lambda entry: entry[0].start):
            ciks += [cik for cik in chain if cik not in ciks]
        ordered = sorted(
            runs, key=lambda entry: (entry[0].end is None, entry[0].end or entry[0].start)
        )
        latest = ordered[-1][0]
        key = latest.ticker.lower().replace(".", "_")
        if key in keys:
            key = f"{key}_{ciks[-1].lstrip('0')}"
        keys.add(key)
        filer = filers.get(ciks[-1]) or {}
        companies.append(
            {
                "key": key,
                "name": filer.get("name", ""),
                "ciks": ";".join(ciks),
                "tickers": ";".join(dict.fromkeys(run.ticker for run, _ in ordered)),
                "spells": ";".join(
                    f"{run.ticker}:{run.start.date().isoformat()}:"
                    f"{run.end.date().isoformat() if run.end is not None else ''}"
                    for run, _ in sorted(runs, key=lambda entry: entry[0].start)
                ),
                "release": "r1" if ciks[-1] in listed_ciks else "r2",
            }
        )
    return sorted(companies, key=lambda company: company["key"])


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(args.user_agent, args.cache_dir, transport or _urllib_transport)
    meta = build(args, client)
    print(
        json.dumps(
            {key: value for key, value in meta.items() if key != "runs_not_confirmed"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
