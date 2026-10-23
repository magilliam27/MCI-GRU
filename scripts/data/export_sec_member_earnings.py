"""Pull index members' earnings 8-Ks, each with the company's previous release (E17).

Experiment E17 asks Jev to read an earnings release (8-K item 2.02) against the
company's previous one (``mci_gru.filings.earnings``). This pull finds every
candidate from SEC's submissions listings, draws from them, and reads only the
drawn releases and their previous ones::

    python -m scripts.data.export_sec_member_earnings \\
        --user-agent "Your Name you@example.com" \\
        --companies r2k_companies.csv \\
        --package-root /content/earnings --cache-dir /content/sec_cache \\
        --accepted-from 2014-01-01 --accepted-to 2026-09-09 \\
        --draw reading:100:1700 --draw test:7000:17 --exclude scouted.txt

A candidate is an original 8-K listing item 2.02 accepted from
``--accepted-from`` to ``--accepted-to`` on a day the company was a member,
whose previous release was accepted 45 to 120 days before with no other 2.02
8-K in the 14 days before. Each ``--draw name:size:seed`` is taken in turn from
what earlier draws left (a size of ``-1`` takes all); ``--exclude`` names
accessions never drawn. Drawn events are then checked again on their index
pages' acceptance times (SEC's listings run hours late for some filers), and
an event is left out when the index time breaks a rule, a document cannot be
read, either text is shorter than 500 characters, or E10's item checks flag
either filing.

What it writes under ``--package-root``:

- ``earnings_candidates.csv``: every candidate, with its previous release and
  its draw (empty when not drawn), by listed times.
- ``earnings_events.jsonl``: one drawn event per line that passed, keyed by
  ``event_id`` (the accession): ``accepted_utc`` (index page), both texts, and
  the previous release under ``previous``.
- ``earnings_left_out.csv``: each drawn event left out, and why.
- ``earnings_events.meta.json``: provenance and counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.earnings import (
    CLEAR_DAYS,
    EARNINGS_ITEM,
    MIN_RELEASE_CHARS,
    PREVIOUS_MAX_DAYS,
    PREVIOUS_MIN_DAYS,
    draw,
    previous_release,
)
from mci_gru.filings.eight_k import MAX_TEXT_CHARS
from mci_gru.filings.filers import Filer
from scripts.data.export_sec_8k_events import original_8ks, read_listings
from scripts.data.export_sec_member_8k import (
    _download,
    _read_event,
    eastern_day,
    is_member,
)
from scripts.data.export_sec_member_10k import read_spells
from scripts.data.export_sec_risk_factors import (
    EdgarClient,
    Transport,
    _git_commit,
    _urllib_transport,
)

PROGRESS_EVERY = 500
CANDIDATE_COLUMNS = [
    "event_id",
    "company",
    "cik",
    "listed_accepted_utc",
    "document",
    "items",
    "previous_id",
    "previous_cik",
    "previous_listed_accepted_utc",
    "previous_document",
    "previous_items",
    "draw",
]
LEFT_OUT_COLUMNS = ["event_id", "company", "draw", "year", "reason", "detail"]


def parse_draw(value: str) -> tuple[str, int, int]:
    name, size, seed = value.split(":")
    return name, int(size), int(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--accepted-from", default="2014-01-01")
    parser.add_argument("--accepted-to", default=None, help="Last acceptance day (Eastern)")
    parser.add_argument(
        "--draw", type=parse_draw, action="append", required=True, help="name:size:seed"
    )
    parser.add_argument("--exclude", type=Path, default=None, help="Accessions, one a line")
    parser.add_argument(
        "--fresh-submissions",
        action="store_true",
        help="Read each CIK's main submissions file from SEC, not the cache",
    )
    parser.add_argument("--workers", type=int, default=1, help="Processes reading documents")
    parser.add_argument("--download-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


def company_candidates(
    company: Any, listings: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """One company's candidates by listed times.

    The listed time can be hours late, so a member day before it counts too;
    each drawn event is checked again on its index page's time.
    """
    spells = read_spells(str(company.spells))
    eight_ks = original_8ks(listings)
    by_id = {row["accession"]: row for row in listings}
    rows = []
    for row in listings:
        if row["form"] != "8-K" or EARNINGS_ITEM not in row["items"]:
            continue
        if not any(
            is_member(spells, eastern_day(row["accepted"]) - pd.Timedelta(days=lag), args)
            for lag in (0, 1)
        ):
            continue
        previous, reason = previous_release(eight_ks, row["accepted"], row["accession"])
        if previous is None:
            continue
        before = by_id[previous[1]]
        rows.append(
            {
                "event_id": row["accession"],
                "company": company.key,
                "cik": row["cik"],
                "listed_accepted_utc": row["accepted"].isoformat(),
                "document": row["document"],
                "items": ";".join(row["items"]),
                "previous_id": before["accession"],
                "previous_cik": before["cik"],
                "previous_listed_accepted_utc": before["accepted"].isoformat(),
                "previous_document": before["document"],
                "previous_items": ";".join(before["items"]),
            }
        )
    return rows


def _filing(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    """The download row of the event (prefix "") or its previous release ("previous_")."""
    accession = row["event_id"] if not prefix else row["previous_id"]
    return {
        "cik": row[f"{prefix}cik"],
        "accession": accession,
        "document": row[f"{prefix}document"],
        "items": tuple(filter(None, str(row[f"{prefix}items"]).split(";"))),
    }


def check_drawn(
    row: dict[str, Any],
    current: dict[str, Any],
    previous: dict[str, Any],
    spells: list,
    args: argparse.Namespace,
) -> tuple[str, str]:
    """Why a drawn event is left out on its index-page times and texts, or ``("", "")``."""
    accepted = current["index_accepted"] or pd.Timestamp(row["listed_accepted_utc"])
    before = previous["index_accepted"] or pd.Timestamp(row["previous_listed_accepted_utc"])
    if not is_member(spells, eastern_day(accepted), args):
        return "outside_membership_or_window_by_index_page", accepted.isoformat()
    days = (accepted - before) / pd.Timedelta(days=1)
    if not PREVIOUS_MIN_DAYS <= days <= PREVIOUS_MAX_DAYS:
        return "previous_gap_by_index_page", f"{days:.1f} days"
    for name, read in (("current", current), ("previous", previous)):
        if len(read["text"]) < MIN_RELEASE_CHARS:
            return f"{name}_text_short", f"{len(read['text'])} characters"
    return "", ""


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(args.user_agent, args.cache_dir, transport or _urllib_transport)
    acquired_at = datetime.now(timezone.utc).isoformat()
    companies = pd.read_csv(args.companies, dtype=str, keep_default_na=False)
    out = args.package_root
    out.mkdir(parents=True, exist_ok=True)
    excluded = set()
    if args.exclude is not None:
        excluded = {line.strip() for line in args.exclude.read_text().splitlines() if line.strip()}

    def listed(company: Any) -> list[dict[str, Any]]:
        filer = Filer(company.key, company.name, tuple(str(company.ciks).split(";")))
        return company_candidates(
            company, read_listings(client, filer, fresh=args.fresh_submissions), args
        )

    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for done, rows in enumerate(pool.map(listed, companies.itertuples(index=False)), 1):
            candidates += rows
            if done % PROGRESS_EVERY == 0:
                print(f"listed {done} of {len(companies)} companies", flush=True)
    seen = Counter(row["event_id"] for row in candidates)
    repeated = sorted(event_id for event_id, n in seen.items() if n > 1)
    # A filing listed under two companies is nobody's alone: left out of every draw.
    pool_ids = [
        row["event_id"]
        for row in candidates
        if row["event_id"] not in excluded and seen[row["event_id"]] == 1
    ]
    drawn = draw(pool_ids, args.draw)
    for row in candidates:
        row["draw"] = drawn.get(row["event_id"], "")
    table = pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)
    table.to_csv(out / "earnings_candidates.csv", index=False)
    chosen = [row for row in candidates if row["draw"]]
    print(f"{len(candidates)} candidates; {len(chosen)} drawn", flush=True)

    filings = [_filing(row, prefix) for row in chosen for prefix in ("", "previous_")]
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        errors = []
        for done, error in enumerate(pool.map(lambda f: _download(client, f), filings), 1):
            errors.append(error)
            if done % PROGRESS_EVERY == 0:
                print(f"downloaded {done} of {len(filings)} filings", flush=True)
    tasks = [
        (str(args.cache_dir), f["cik"], f["accession"], f["document"], f["items"])
        for f, error in zip(filings, errors, strict=True)
        if not error
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = iter(list(pool.map(_read_event, tasks, chunksize=8)))
    else:
        results = iter([_read_event(task) for task in tasks])
    reads: list[tuple[dict[str, Any] | None, list[tuple[str, str]]]] = []
    for error in errors:
        reads.append((None, [("document_unreadable", error)]) if error else next(results))

    spells = {c.key: read_spells(str(c.spells)) for c in companies.itertuples(index=False)}
    names = dict(zip(companies["key"], companies["name"], strict=True))
    releases = dict(zip(companies["key"], companies.get("release", ""), strict=True))
    kept, left_out = [], []
    for k, row in enumerate(chosen):
        (current, problems), (previous, previous_problems) = reads[2 * k], reads[2 * k + 1]
        reason, detail = "", ""
        flagged = [(f"current_{c}", d) for c, d in problems]
        flagged += [(f"previous_{c}", d) for c, d in previous_problems]
        if flagged:
            reason, detail = ";".join(dict.fromkeys(c for c, _ in flagged)), flagged[0][1]
        else:
            reason, detail = check_drawn(row, current, previous, spells[row["company"]], args)
        if reason:
            left_out.append(
                {
                    "event_id": row["event_id"],
                    "company": row["company"],
                    "draw": row["draw"],
                    "year": row["listed_accepted_utc"][:4],
                    "reason": reason,
                    "detail": detail,
                }
            )
            continue
        accepted = current["index_accepted"] or pd.Timestamp(row["listed_accepted_utc"])
        before = previous["index_accepted"] or pd.Timestamp(row["previous_listed_accepted_utc"])
        kept.append(
            {
                "event_id": row["event_id"],
                "company": row["company"],
                "company_name": names[row["company"]],
                "release": releases[row["company"]],
                "draw": row["draw"],
                "cik": row["cik"],
                "items": row["items"].split(";"),
                "accepted_utc": accepted.isoformat(),
                "listed_accepted_utc": row["listed_accepted_utc"],
                "accepted_from_index_page": current["index_accepted"] is not None,
                "text": current["text"],
                "truncated": current["truncated"],
                "press_release": current["press_release"],
                "document_url": current["document_url"],
                "document_sha256": current["document_sha256"],
                "previous": {
                    "event_id": row["previous_id"],
                    "cik": row["previous_cik"],
                    "items": row["previous_items"].split(";"),
                    "accepted_utc": before.isoformat(),
                    "listed_accepted_utc": row["previous_listed_accepted_utc"],
                    "text": previous["text"],
                    "truncated": previous["truncated"],
                    "press_release": previous["press_release"],
                    "document_url": previous["document_url"],
                    "document_sha256": previous["document_sha256"],
                },
                "previous_gap_days": round((accepted - before) / pd.Timedelta(days=1), 2),
            }
        )

    events_path = out / "earnings_events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in kept:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    pd.DataFrame(left_out, columns=LEFT_OUT_COLUMNS).to_csv(
        out / "earnings_left_out.csv", index=False
    )
    meta = {
        "source": "SEC EDGAR submissions and filing documents",
        "companies_table": str(args.companies),
        "accepted_from": args.accepted_from,
        "accepted_to": args.accepted_to,
        "rule": (
            f"original 8-K listing item {EARNINGS_ITEM}, accepted on a member day; previous"
            f" release {PREVIOUS_MIN_DAYS} to {PREVIOUS_MAX_DAYS} days before; no other"
            f" {EARNINGS_ITEM} 8-K in the {CLEAR_DAYS} days before"
        ),
        "draws": [list(entry) for entry in args.draw],
        "excluded_given": len(excluded),
        "max_text_chars": MAX_TEXT_CHARS,
        "min_release_chars": MIN_RELEASE_CHARS,
        "acquired_at": acquired_at,
        "network_requests": client.requests,
        "repository_commit": _git_commit(),
        "companies": len(companies),
        "candidates": len(candidates),
        "candidates_by_year": dict(
            sorted(Counter(r["listed_accepted_utc"][:4] for r in candidates).items())
        ),
        "candidates_listed_under_two_companies": len(repeated),
        "drawn": dict(Counter(row["draw"] for row in chosen)),
        "events": dict(Counter(event["draw"] for event in kept)),
        "events_by_year": dict(sorted(Counter(e["accepted_utc"][:4] for e in kept).items())),
        "left_out": dict(Counter(row["draw"] for row in left_out)),
        "left_out_by_reason": dict(Counter(row["reason"] for row in left_out)),
        "truncated": sum(event["truncated"] for event in kept),
        "previous_truncated": sum(event["previous"]["truncated"] for event in kept),
        "without_press_release": sum(event["press_release"] is None for event in kept),
    }
    (out / "earnings_events.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
