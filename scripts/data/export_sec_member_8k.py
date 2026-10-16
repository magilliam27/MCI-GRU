"""Pull S&P 500 members' 8-K events from SEC EDGAR, with filing-time context (#311).

An 8-K returns test needs every event a company reported while it was in the
S&P 500. This pull applies E10's 8-K contract (``export_sec_8k_events``) to the
member table ``build_sp500_members`` writes::

    python -m scripts.data.export_sec_member_8k \\
        --user-agent "Your Name you@example.com" \\
        --companies sp500_companies.csv \\
        --package-root /content/member_8k --cache-dir /content/sec_cache \\
        --manifest-output /content/member_8k.r1.json

For each company (of ``--release``, or all) it reads the submissions files of
every CIK the company filed under. An event is an original 8-K carrying an event
item (``mci_gru.filings.eight_k``) accepted from ``--accepted-from`` to
``--accepted-to`` on a day the company was a member: the Eastern day of its
acceptance, by the time its index page shows (SEC's submissions files list some
filers' acceptance times hours late; ``listed_accepted_utc`` keeps theirs). Each event gets E10's text and context: revenue
as tagged in the company's latest earlier 10-K, and the item codes of its other
8-Ks in the 90 days before. Documents download into the gzipped ``--cache-dir``
on ``--download-workers`` threads, which share SEC's pace, and are read with
``--workers`` processes.

What it writes under ``--package-root``:

- ``member_8k_events.jsonl``: one event per line, keyed by ``event_id`` (the
  accession), in E10's fields plus ``listed_accepted_utc``,
  ``accepted_from_index_page``, ``nearby_202_listed_utc`` (the listed times of
  the company's other earnings 8-Ks, item 2.02, within five calendar days either
  side), ``release`` and ``sic``. ``accepted_utc`` is the index page's time.
- ``member_8k_checks.csv``: one row per check finding.
- ``member_8k_left_out.csv``: with ``--leave-out-flagged``, each event left out
  because a check failed or a document could not be read, with its year and
  reason.
- ``member_8k_events.meta.json``: provenance and counts, including events by
  item and by year, and the share left out per year.

A finding stops ``--manifest-output`` unless ``--leave-out-flagged`` leaves the
flagged events out.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from mci_gru.filings.edgar import SUBMISSIONS_URL
from mci_gru.filings.eight_k import (
    EDGAR_TIMEZONE,
    EVENT_ITEMS,
    LEFT_OUT_ITEMS,
    MAX_TEXT_CHARS,
    PRIOR_DAYS,
    REVENUE_CONCEPTS,
    index_exhibits,
    is_event,
    press_release,
    tagged_revenue,
)
from mci_gru.filings.filers import Filer
from scripts.data.export_sec_8k_events import (
    COMPANY_FACTS_URL,
    _archive_url,
    event_record,
    original_8ks,
    read_event_text,
    read_listings,
    ten_k_index,
)
from scripts.data.export_sec_member_10k import read_spells
from scripts.data.export_sec_risk_factors import (
    EdgarClient,
    Transport,
    _git_commit,
    _urllib_transport,
    decode_document,
)

if TYPE_CHECKING:
    from mci_gru.filings.membership import Spell

PROGRESS_EVERY = 1000
#: How many calendar days either side of an event an earnings 8-K is "nearby".
NEARBY_DAYS = 5
EARNINGS_ITEM = "2.02"
CHECK_COLUMNS = ["event_id", "company", "check", "detail"]
LEFT_OUT_COLUMNS = ["event_id", "company", "cik", "accepted_utc", "year", "items", "reason"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--release", default=None, help="Only this release (default all)")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--accepted-from", default="2013-01-01")
    parser.add_argument("--accepted-to", default=None, help="Last acceptance date (UTC)")
    parser.add_argument(
        "--fresh-submissions",
        action="store_true",
        help="Read each CIK's main submissions file from SEC, not the cache",
    )
    parser.add_argument("--workers", type=int, default=1, help="Processes reading events")
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--leave-out-flagged", action="store_true")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--package-id", default="member_8k_events")
    parser.add_argument("--package-revision", default="r1")
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


def eastern_day(accepted: pd.Timestamp) -> pd.Timestamp:
    """The calendar day in EDGAR's Eastern time on which a filing was accepted."""
    return accepted.tz_convert(EDGAR_TIMEZONE).tz_localize(None).normalize()


def is_member(spells: list[Spell], day: pd.Timestamp, args: argparse.Namespace) -> bool:
    """Whether a day is in the window and inside one of the company's spells."""
    if day < pd.Timestamp(args.accepted_from):
        return False
    if args.accepted_to and day > pd.Timestamp(args.accepted_to):
        return False
    return any(spell.holds(day) for spell in spells)


def member_events(
    listings: list[dict[str, Any]], spells: list[Spell], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """The listed 8-K events that may have been accepted on a member day in the window.

    The listed acceptance time can be hours late (``index_accepted``), so a day
    before it counts too; each event is checked again on its index page's time.
    """
    return [
        row
        for row in listings
        if is_event(row["form"], row["items"])
        and any(
            is_member(spells, eastern_day(row["accepted"]) - pd.Timedelta(days=lag), args)
            for lag in (0, 1)
        )
    ]


def nearby_earnings(
    eight_ks: list[tuple[pd.Timestamp, str, tuple[str, ...]]],
    accepted: pd.Timestamp,
    accession: str,
) -> list[str]:
    """Listed acceptance times of the company's other 8-Ks carrying item 2.02 nearby.

    Nearby is within ``NEARBY_DAYS`` calendar days either side, by Eastern day
    (each listed time as SEC lists it, which can run hours late).
    """
    day = eastern_day(accepted)
    return [
        when.isoformat()
        for when, other, items in sorted(eight_ks)
        if other != accession
        and EARNINGS_ITEM in items
        and abs((eastern_day(when) - day).days) <= NEARBY_DAYS
    ]


def _download(client: EdgarClient, row: dict[str, Any]) -> str:
    """Fetch an event's documents into the cache; the error, if one could not be read."""
    cik, accession = row["cik"], row["accession"]
    try:
        client.get(_archive_url(cik, accession, row["document"]))
        index_page = client.get(_archive_url(cik, accession, f"{accession}-index.htm"))
        release = press_release(index_exhibits(decode_document(index_page)[0]))
        if release is not None:
            client.get(_archive_url(cik, accession, release[1]))
    except Exception as error:  # noqa: BLE001 - recorded as the event's reason
        return f"{type(error).__name__}: {error}"
    return ""


def _cache_only(url: str, headers: dict[str, str]) -> bytes:
    raise RuntimeError(f"{url} is not in the cache")


def _read_event(task: tuple[str, str, str, str, tuple[str, ...]]) -> tuple[dict, list]:
    """Read one cached event's text and its check problems (runs in a worker)."""
    cache_dir, cik, accession, document, items = task
    client = EdgarClient("cache only", Path(cache_dir), _cache_only)
    return read_event_text(client.get, cik, accession, document, items)


class CompanyFacts:
    """SEC company facts per CIK, read once; a CIK SEC holds none for gives none."""

    def __init__(self, client: EdgarClient) -> None:
        self.client = client
        self._seen: dict[str, dict[str, Any]] = {}

    def revenue(self, cik: str, accession: str) -> dict[str, Any] | None:
        if cik not in self._seen:
            try:
                self._seen[cik] = self.client.get_json(COMPANY_FACTS_URL.format(cik=cik))
            except Exception:  # noqa: BLE001 - no facts means no tagged revenue
                self._seen[cik] = {}
        return tagged_revenue(self._seen[cik], accession)


def _by_year(events: list[dict[str, Any]], left: list[dict[str, Any]]) -> dict[str, dict]:
    kept = Counter(event["accepted_utc"][:4] for event in events)
    gone = Counter(row["year"] for row in left)
    return {
        year: {
            "events": kept[year] + gone[year],
            "left_out": gone[year],
            "share_left_out": round(gone[year] / (kept[year] + gone[year]), 4),
        }
        for year in sorted(set(kept) | set(gone))
    }


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(args.user_agent, args.cache_dir, transport or _urllib_transport)
    acquired_at = datetime.now(timezone.utc).isoformat()
    companies = pd.read_csv(args.companies, dtype=str, keep_default_na=False)
    if args.release:
        companies = companies[companies["release"] == args.release]
    out = args.package_root
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    context: dict[str, tuple] = {}
    for company in companies.itertuples(index=False):
        ciks = tuple(str(company.ciks).split(";"))
        filer = Filer(company.key, company.name, ciks)
        listings = read_listings(client, filer, fresh=args.fresh_submissions)
        # Submissions were just read, so these come from the cache.
        sic = {
            cik: str(client.get_json(SUBMISSIONS_URL.format(cik=cik)).get("sic") or "")
            for cik in ciks
        }
        context[company.key] = (
            company,
            ten_k_index(listings, filer.preferred_cik),
            original_8ks(listings),
            sic,
        )
        spells = read_spells(str(company.spells))
        context[company.key] = (*context[company.key], spells)
        rows += [{**row, "company": company.key} for row in member_events(listings, spells, args)]
    print(f"{len(rows)} events from {len(companies)} companies", flush=True)

    # Documents download on several threads under the client's one pace.
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        errors = []
        for done, error in enumerate(pool.map(lambda row: _download(client, row), rows), 1):
            errors.append(error)
            if done % PROGRESS_EVERY == 0:
                print(f"downloaded {done} of {len(rows)} events", flush=True)

    tasks = [
        (str(args.cache_dir), row["cik"], row["accession"], row["document"], row["items"])
        for row, error in zip(rows, errors, strict=True)
        if not error
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = iter(list(pool.map(_read_event, tasks, chunksize=8)))
    else:
        results = iter([_read_event(task) for task in tasks])

    facts = CompanyFacts(client)
    # Per event: its listing, its line (None when unread) and its problems.
    read_rows: list[tuple[dict[str, Any], dict[str, Any] | None, list[tuple[str, str]]]] = []
    corrected, outside = 0, []
    for row, error in zip(rows, errors, strict=True):
        company, ten_ks, eight_ks, sic, spells = context[row["company"]]
        if error:
            read_rows.append((row, None, [("document_unreadable", error)]))
            continue
        read, problems = next(results)
        listed = row["accepted"]
        row = {**row, "accepted": read["index_accepted"] or listed}
        corrected += read["index_accepted"] is not None and read["index_accepted"] != listed
        if not is_member(spells, eastern_day(row["accepted"]), args):
            outside.append(row["accession"])
            continue
        event = event_record(
            row,
            company.key,
            company.name,
            read,
            ten_ks,
            lambda prior: facts.revenue(prior.cik, prior.accession_number),
            eight_ks,
        )
        event.update(
            listed_accepted_utc=listed.isoformat(),
            accepted_from_index_page=read["index_accepted"] is not None,
            nearby_202_listed_utc=nearby_earnings(eight_ks, row["accepted"], row["accession"]),
            release=company.release,
            sic=sic.get(row["cik"], ""),
        )
        read_rows.append((row, event, list(problems)))
    seen = Counter(row["accession"] for row, _, _ in read_rows)
    for row, _, problems in read_rows:
        if seen[row["accession"]] > 1:
            problems.append(
                ("event_id_repeated", f"listed under {seen[row['accession']]} companies")
            )

    findings = [
        {"event_id": row["accession"], "company": row["company"], "check": check, "detail": detail}
        for row, _, problems in read_rows
        for check, detail in problems
    ]
    kept = [event for _, event, problems in read_rows if event is not None and not problems]
    left_out = [
        {
            "event_id": row["accession"],
            "company": row["company"],
            "cik": row["cik"],
            "accepted_utc": row["accepted"].isoformat(),
            "year": row["accepted"].isoformat()[:4],
            "items": ";".join(row["items"]),
            "reason": ";".join(dict.fromkeys(check for check, _ in problems)),
        }
        for row, _, problems in read_rows
        if problems
    ]
    if not args.leave_out_flagged:
        # Without leaving them out, flagged events stay in and stop the manifest.
        kept = [event for _, event, _ in read_rows if event is not None]
        left_out = []

    events_path = out / "member_8k_events.jsonl"
    checks_path = out / "member_8k_checks.csv"
    left_out_path = out / "member_8k_left_out.csv"
    meta_path = out / "member_8k_events.meta.json"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in kept:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    pd.DataFrame(findings, columns=CHECK_COLUMNS).to_csv(checks_path, index=False)
    pd.DataFrame(left_out, columns=LEFT_OUT_COLUMNS).to_csv(left_out_path, index=False)

    by_item = Counter(code for event in kept for code in event["event_items"])
    by_year_item: dict[str, Counter] = {}
    for event in kept:
        for code in event["event_items"]:
            by_year_item.setdefault(event["accepted_utc"][:4], Counter())[code] += 1
    with_events = {event["company"] for event in kept} | {row["company"] for row in left_out}
    meta = {
        "source": "SEC EDGAR submissions, filing documents and XBRL company facts",
        "companies_table": str(args.companies),
        "release": args.release or "all",
        "forms": ["8-K"],
        "event_items": list(EVENT_ITEMS),
        "left_out_items": list(LEFT_OUT_ITEMS),
        "accepted_from": args.accepted_from,
        "accepted_to": args.accepted_to,
        "membership_rule": (
            "accepted, by the filing's index page, on an Eastern day inside one of the"
            " company's spells"
        ),
        "max_text_chars": MAX_TEXT_CHARS,
        "prior_days": PRIOR_DAYS,
        "revenue_concepts": list(REVENUE_CONCEPTS),
        "acquired_at": acquired_at,
        "network_requests": client.requests,
        "repository_commit": _git_commit(),
        "companies": len(companies),
        "companies_with_events": len(with_events),
        "companies_without_events": sorted(set(companies["key"]) - with_events),
        "events": len(kept),
        "events_by_item": dict(sorted(by_item.items())),
        "events_by_year": dict(sorted(Counter(e["accepted_utc"][:4] for e in kept).items())),
        "events_by_year_and_item": {
            year: dict(sorted(counts.items())) for year, counts in sorted(by_year_item.items())
        },
        "events_by_release": dict(Counter(event["release"] for event in kept)),
        "with_nearby_earnings_8k": sum(bool(event["nearby_202_listed_utc"]) for event in kept),
        "truncated": sum(event["truncated"] for event in kept),
        "without_revenue": sum(event["prior_revenue_usd"] is None for event in kept),
        "with_press_release": sum(event["press_release"] is not None for event in kept),
        "acceptance_corrected_from_index_page": int(corrected),
        "acceptance_from_listing_only": sum(not e["accepted_from_index_page"] for e in kept),
        "outside_membership_by_index_page": len(outside),
        "check_findings": len(findings),
        "findings_by_check": dict(Counter(item["check"] for item in findings)),
        "events_left_out": len(left_out),
        "events_left_out_by_year": _by_year(kept, left_out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in meta.items() if key != "events_by_year_and_item"},
            indent=2,
        )
    )
    blocking = [] if args.leave_out_flagged else findings
    if blocking:
        print(f"{len(blocking)} check findings; no manifest published.")
        return 1

    if args.manifest_output is not None:
        snapshot = write_input_manifest(
            args.manifest_output,
            out,
            package_id=args.package_id,
            package_revision=args.package_revision,
            files=[
                InputFileSpec(events_path.name, "8-K events, one per line"),
                InputFileSpec(checks_path.name, "Event check findings"),
                InputFileSpec(left_out_path.name, "Events left out, and why"),
                InputFileSpec(meta_path.name, "Pull provenance and counts"),
            ],
            provenance={
                "source": meta["source"],
                "acquisition_mode": client.acquisition_mode,
                "acquired_at": acquired_at,
                "producing_command": "python -m scripts.data.export_sec_member_8k",
                "producing_arguments": sys.argv[1:] if argv is None else list(argv),
                "unknowns": [],
            },
            metadata={
                "description": f"S&P 500 members' 8-K events (#311), release {meta['release']}.",
                "repository_commit": _git_commit(),
            },
        )
        print(json.dumps({"manifest": str(args.manifest_output), "sha256": snapshot.sha256}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
