"""Find each drawn 8-K's latest earnings release and read it from SEC (E18).

Experiment E18 asks E16's rating question again with the company's latest
earnings release beside the 8-K (``mci_gru.filings.context``). This pull reads
each frame company's SEC submissions listing, finds every frame event's latest
release by listed times, draws from the events that have one, and reads only
the drawn events' releases::

    python -m scripts.data.export_sec_context_releases \\
        --user-agent "Your Name you@example.com" \\
        --frame frame.csv --companies r2k_companies.csv \\
        --events part_a.jsonl --events part_b.jsonl \\
        --package-root /content/e18 --cache-dir /content/sec_cache \\
        --draw probe:neutral:40:1800 --draw test:neutral:4000:1801

The frame is a CSV of 8-K events (``event_id``, ``company``, ``cik``,
``accepted_utc`` by index page, ``stratum``); ``--events`` are the E10-format
event files they come from. Each ``--draw name:stratum:size:seed`` is taken in
turn from the stratum's events with a release that earlier draws left. A drawn
event is then left out when its release's index-page time is not before the
8-K or is more than 120 days before it, a document cannot be read, the
release's text is shorter than 500 characters, or E10's item checks flag it.

What it writes under ``--package-root``:

- ``context_frame.csv``: every frame event, with its release by listed times
  (or why it has none) and its draw (empty when not drawn).
- ``sample_events.jsonl``: the kept drawn events' records, copied unchanged
  from ``--events``.
- ``context_releases.jsonl``: one line per kept drawn event: ``event_id``,
  ``draw``, ``stratum``, the release under ``release`` (accession, items,
  index-page time, text) and ``release_age_days``.
- ``context_left_out.csv`` and ``context_releases.meta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.context import RELEASE_MAX_DAYS, draw_strata, latest_release
from mci_gru.filings.earnings import EARNINGS_ITEM, MIN_RELEASE_CHARS
from mci_gru.filings.eight_k import MAX_TEXT_CHARS
from mci_gru.filings.filers import Filer
from scripts.data.export_sec_8k_events import original_8ks, read_listings
from scripts.data.export_sec_member_8k import _download, _read_event
from scripts.data.export_sec_risk_factors import EdgarClient, Transport, _git_commit

PROGRESS_EVERY = 500
FRAME_COLUMNS = [
    "event_id",
    "company",
    "stratum",
    "accepted_utc",
    "release_id",
    "release_cik",
    "release_listed_accepted_utc",
    "release_document",
    "release_items",
    "no_release_reason",
    "draw",
]
LEFT_OUT_COLUMNS = ["event_id", "company", "draw", "stratum", "year", "reason", "detail"]


def parse_draw(value: str) -> tuple[str, str, int, int]:
    name, stratum, size, seed = value.split(":")
    return name, stratum, int(size), int(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--events", type=Path, action="append", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--draw", type=parse_draw, action="append", required=True, help="name:stratum:size:seed"
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds per SEC request")
    parser.add_argument("--workers", type=int, default=1, help="Processes reading documents")
    parser.add_argument("--download-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


def timed_transport(timeout: float) -> Transport:
    """A urllib GET that gives up after ``timeout`` seconds, so a stalled read is retried soon."""

    def get(url: str, headers: dict[str, str]) -> bytes:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    return get


def company_releases(events: pd.DataFrame, listings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each of one company's frame events with its latest release by listed times.

    SEC's listed times run late, never early, against the index page, so a
    release listed before the 8-K was accepted before it too.
    """
    eight_ks = original_8ks(listings)
    by_id = {row["accession"]: row for row in listings}
    rows = []
    for event in events.itertuples(index=False):
        accepted = pd.Timestamp(event.accepted_utc)
        found, reason = latest_release(eight_ks, accepted, event.event_id)
        row: dict[str, Any] = {
            "event_id": event.event_id,
            "company": event.company,
            "stratum": event.stratum,
            "accepted_utc": accepted.isoformat(),
            "no_release_reason": reason,
        }
        if found is not None:
            release = by_id[found[1]]
            row |= {
                "release_id": release["accession"],
                "release_cik": release["cik"],
                "release_listed_accepted_utc": release["accepted"].isoformat(),
                "release_document": release["document"],
                "release_items": ";".join(release["items"]),
            }
        rows.append(row)
    return rows


def check_release(row: dict[str, Any], read: dict[str, Any]) -> tuple[str, str]:
    """Why a drawn event's release is left out on its index-page time and text, or ``("", "")``."""
    accepted = pd.Timestamp(row["accepted_utc"])
    released = read["index_accepted"] or pd.Timestamp(row["release_listed_accepted_utc"])
    days = (accepted - released) / pd.Timedelta(days=1)
    if days <= 0:
        return "release_not_before_8k_by_index_page", f"{days:.3f} days"
    if days > RELEASE_MAX_DAYS:
        return "release_too_old_by_index_page", f"{days:.1f} days"
    if len(read["text"]) < MIN_RELEASE_CHARS:
        return "release_text_short", f"{len(read['text'])} characters"
    return "", ""


def copy_events(paths: list[Path], keep: set[str], out: Path) -> int:
    """Copy the kept events' lines unchanged, in the files' order; every one must be found."""
    found = 0
    with out.open("w", encoding="utf-8") as handle:
        for path in paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if json.loads(line)["event_id"] in keep:
                        handle.write(line if line.endswith("\n") else line + "\n")
                        found += 1
    if found != len(keep):
        raise ValueError(f"{len(keep) - found} kept events are not in the event files")
    return found


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(
        args.user_agent, args.cache_dir, transport or timed_transport(args.timeout)
    )
    acquired_at = datetime.now(timezone.utc).isoformat()
    frame = pd.read_csv(args.frame, dtype=str, keep_default_na=False)
    companies = pd.read_csv(args.companies, dtype=str, keep_default_na=False)
    companies = companies[companies["key"].isin(set(frame["company"]))]
    by_company = dict(tuple(frame.groupby("company")))
    out = args.package_root
    out.mkdir(parents=True, exist_ok=True)

    def listed(company: Any) -> list[dict[str, Any]]:
        filer = Filer(company.key, company.name, tuple(str(company.ciks).split(";")))
        # Cached listings, so a rerun from the same cache draws the same sample.
        listings = read_listings(client, filer, fresh=False)
        return company_releases(by_company[company.key], listings)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for done, found in enumerate(pool.map(listed, companies.itertuples(index=False)), 1):
            rows += found
            if done % PROGRESS_EVERY == 0:
                print(f"listed {done} of {len(companies)} companies", flush=True)
    if len(rows) != len(frame):
        raise ValueError(f"{len(frame) - len(rows)} frame events have no company listing")
    eligible = [(r["event_id"], r["stratum"]) for r in rows if not r["no_release_reason"]]
    drawn = draw_strata(eligible, args.draw)
    for row in rows:
        row["draw"] = drawn.get(row["event_id"], "")
    pd.DataFrame(rows, columns=FRAME_COLUMNS).to_csv(out / "context_frame.csv", index=False)
    chosen = [row for row in rows if row["draw"]]
    print(f"{len(eligible)} of {len(rows)} frame events have a release; {len(chosen)} drawn")

    filings = [
        {
            "cik": row["release_cik"],
            "accession": row["release_id"],
            "document": row["release_document"],
            "items": tuple(filter(None, row["release_items"].split(";"))),
        }
        for row in chosen
    ]
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        errors = []
        for done, error in enumerate(pool.map(lambda f: _download(client, f), filings), 1):
            errors.append(error)
            if done % PROGRESS_EVERY == 0:
                print(f"downloaded {done} of {len(filings)} releases", flush=True)
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

    kept, left_out = [], []
    for row, error in zip(chosen, errors, strict=True):
        read, problems = (None, [("document_unreadable", error)]) if error else next(results)
        if problems:
            reason = ";".join(dict.fromkeys(f"release_{code}" for code, _ in problems))
            detail = problems[0][1]
        else:
            reason, detail = check_release(row, read)
        if reason:
            left_out.append(
                {
                    "event_id": row["event_id"],
                    "company": row["company"],
                    "draw": row["draw"],
                    "stratum": row["stratum"],
                    "year": row["accepted_utc"][:4],
                    "reason": reason,
                    "detail": detail,
                }
            )
            continue
        released = read["index_accepted"] or pd.Timestamp(row["release_listed_accepted_utc"])
        accepted = pd.Timestamp(row["accepted_utc"])
        kept.append(
            {
                "event_id": row["event_id"],
                "company": row["company"],
                "draw": row["draw"],
                "stratum": row["stratum"],
                "accepted_utc": row["accepted_utc"],
                "release": {
                    "event_id": row["release_id"],
                    "cik": row["release_cik"],
                    "items": row["release_items"].split(";"),
                    "accepted_utc": released.isoformat(),
                    "listed_accepted_utc": row["release_listed_accepted_utc"],
                    "accepted_from_index_page": read["index_accepted"] is not None,
                    "text": read["text"],
                    "truncated": read["truncated"],
                    "press_release": read["press_release"],
                    "document_url": read["document_url"],
                    "document_sha256": read["document_sha256"],
                },
                "release_age_days": round((accepted - released) / pd.Timedelta(days=1), 3),
            }
        )

    with (out / "context_releases.jsonl").open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    copied = copy_events(args.events, {r["event_id"] for r in kept}, out / "sample_events.jsonl")
    pd.DataFrame(left_out, columns=LEFT_OUT_COLUMNS).to_csv(
        out / "context_left_out.csv", index=False
    )
    meta = {
        "source": "SEC EDGAR submissions and filing documents",
        "frame": str(args.frame),
        "companies_table": str(args.companies),
        "event_files": [str(path) for path in args.events],
        "rule": (
            f"latest original 8-K listing item {EARNINGS_ITEM} accepted before the 8-K, at"
            f" most {RELEASE_MAX_DAYS} days before"
        ),
        "draws": [list(entry) for entry in args.draw],
        "max_text_chars": MAX_TEXT_CHARS,
        "min_release_chars": MIN_RELEASE_CHARS,
        "acquired_at": acquired_at,
        "network_requests": client.requests,
        "repository_commit": _git_commit(),
        "companies": len(companies),
        "frame_events": len(rows),
        "frame_by_stratum": dict(Counter(r["stratum"] for r in rows)),
        "no_release": dict(Counter(r["no_release_reason"] for r in rows if r["no_release_reason"])),
        "eligible_by_stratum": dict(Counter(stratum for _, stratum in eligible)),
        "drawn": dict(Counter(row["draw"] for row in chosen)),
        "kept": dict(Counter(record["draw"] for record in kept)),
        "kept_events_copied": copied,
        "left_out": dict(Counter(row["draw"] for row in left_out)),
        "left_out_by_reason": dict(Counter(row["reason"] for row in left_out)),
        "release_truncated": sum(record["release"]["truncated"] for record in kept),
        "release_without_press_release": sum(
            record["release"]["press_release"] is None for record in kept
        ),
    }
    (out / "context_releases.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
