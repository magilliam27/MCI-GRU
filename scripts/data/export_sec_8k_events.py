"""Pull companies' 8-K events from SEC EDGAR, with filing-time context (#308).

Run from the repository root wherever EDGAR is reachable::

    python -m scripts.data.export_sec_8k_events \\
        --user-agent "Your Name you@example.com" \\
        --package-root /content/sec_8k_events \\
        --cache-dir /content/sec_cache \\
        --filer jpm --filer dal \\
        --manifest-output /content/8k_events.r1.json

For each ``--filer`` (a company in ``mci_gru.filings.filers``) it reads the
submissions files of every CIK the company filed under. Every original 8-K filed
from ``--first-date`` to ``--last-date`` that carries an event item (see
``mci_gru.filings.eight_k``) is an event. For each event it downloads the primary
document, the filing's index page and its press-release exhibit, and adds two
pieces of context that were public when the 8-K was accepted: revenue as tagged
in the company's latest earlier 10-K (from SEC's XBRL company facts), and the
item codes of the company's other 8-Ks in the 90 days before.

What it writes under ``--package-root``:

- ``8k_events.jsonl``: one event per line, in the field names the Questions
  lane's loader reads (``event_id``, ``company``, ``company_name``, ``items``,
  ``text``, ``prior_revenue_usd``, ``prior_revenue_fiscal_year``,
  ``prior_90d_items``), with provenance fields beside them.
- ``8k_events_checks.csv``: one row per check finding; empty when clean.
- ``8k_events.meta.json``: provenance and counts.

An event whose listed items have no section, whose item section is empty, or
whose press release reads empty is reported and stops the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from mci_gru.filings.edgar import (
    SUBMISSIONS_FILE_URL,
    SUBMISSIONS_URL,
    FilingRecord,
    build_filing_index,
    parse_acceptance,
    submissions_blocks,
)
from mci_gru.filings.eight_k import (
    EVENT_ITEMS,
    LEFT_OUT_ITEMS,
    MAX_TEXT_CHARS,
    PRIOR_DAYS,
    REVENUE_CONCEPTS,
    document_text,
    event_text,
    index_exhibits,
    is_event,
    item_sections,
    listed_items,
    press_release,
    prior_items,
    tagged_revenue,
)
from mci_gru.filings.filers import FILERS, Filer
from scripts.data.export_sec_risk_factors import (
    EdgarClient,
    Transport,
    _git_commit,
    _urllib_transport,
    decode_document,
)

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{name}"
CHECK_COLUMNS = ["event_id", "company", "check", "detail"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--filer",
        action="append",
        required=True,
        choices=sorted(FILERS),
        help="Company key; repeat for several",
    )
    parser.add_argument("--first-date", default="2013-01-01", help="Earliest filing date")
    parser.add_argument("--last-date", default="2025-12-31", help="Latest filing date")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--package-id", default="8k_events")
    parser.add_argument("--package-revision", default="r1")
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    args.filer = list(dict.fromkeys(args.filer))
    return args


def _archive_url(cik: str, accession: str, name: str) -> str:
    return ARCHIVE_URL.format(cik=int(cik), accession=accession.replace("-", ""), name=name)


def read_listings(client: EdgarClient, filer: Filer) -> list[dict[str, Any]]:
    """Every filing listed under the company's CIKs, once per accession.

    A filing listed under two CIKs keeps the listing of the CIK named last.
    """
    listings: dict[str, dict[str, Any]] = {}
    for cik in filer.ciks:
        recent, older = submissions_blocks(
            client.get_json(SUBMISSIONS_URL.format(cik=cik), fresh=True)
        )
        blocks = [recent] + [
            client.get_json(SUBMISSIONS_FILE_URL.format(name=name)) for name in older
        ]
        for block in blocks:
            items = block.get("items") or [""] * len(block["accessionNumber"])
            for i in range(len(block["accessionNumber"])):
                accession = str(block["accessionNumber"][i])
                listings[accession] = {
                    "cik": cik,
                    "accession": accession,
                    "form": str(block["form"][i]),
                    "filing_date": str(block["filingDate"][i]),
                    "report_date": str(block["reportDate"][i]),
                    "accepted": parse_acceptance(str(block["acceptanceDateTime"][i])),
                    "document": str(block["primaryDocument"][i]),
                    "items": listed_items(str(items[i] or "")),
                }
    return sorted(listings.values(), key=lambda row: (row["accepted"], row["accession"]))


def _ten_ks(listings: list[dict[str, Any]], filer: Filer) -> list[FilingRecord]:
    records = [
        FilingRecord(
            cik=row["cik"],
            accession_number=row["accession"],
            form=row["form"],
            filing_date=row["filing_date"],
            report_date=row["report_date"],
            accepted_utc=row["accepted"],
            primary_document=row["document"],
        )
        for row in listings
        if row["form"] == "10-K"
    ]
    return build_filing_index(
        records, forms=("10-K",), accepted_from="1990-01-01", preferred_cik=filer.preferred_cik
    ).filings


def export_filer(
    client: EdgarClient, filer: Filer, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """One company's events, and the check findings they raised."""
    listings = read_listings(client, filer)
    eight_ks = [
        (row["accepted"], row["accession"], row["items"])
        for row in listings
        if row["form"] == "8-K"
    ]
    ten_ks = _ten_ks(listings, filer)
    facts = {
        cik: client.get_json(COMPANY_FACTS_URL.format(cik=cik), fresh=True) for cik in filer.ciks
    }
    events, findings = [], []
    for row in listings:
        if not is_event(row["form"], row["items"]):
            continue
        if not args.first_date <= row["filing_date"] <= args.last_date:
            continue
        accession, cik = row["accession"], row["cik"]
        document = client.get(_archive_url(cik, accession, row["document"]))
        sections = item_sections(decode_document(document)[0])
        index_page = client.get(_archive_url(cik, accession, f"{accession}-index.htm"))
        release = press_release(index_exhibits(decode_document(index_page)[0]))
        exhibit_text = None
        if release is not None:
            exhibit_text = document_text(
                decode_document(client.get(_archive_url(cik, accession, release[1])))[0]
            )
        text, truncated = event_text(sections, exhibit_text, release[0] if release else None)

        found = {section.code for section in sections}
        problems = [
            ("item_section_missing", f"item {code} is listed but has no section")
            for code in row["items"]
            if code not in LEFT_OUT_ITEMS and code not in found
        ]
        problems += [
            ("item_section_empty", f"item {section.code} has no text under its heading")
            for section in sections
            if section.code not in LEFT_OUT_ITEMS
            and not any(character.isalnum() for character in section.body)
        ]
        if release is not None and not exhibit_text:
            problems.append(("press_release_empty", f"{release[1]} reads empty"))
        findings += [
            {"event_id": accession, "company": filer.key, "check": check, "detail": detail}
            for check, detail in problems
        ]

        earlier = [record for record in ten_ks if record.accepted_utc < row["accepted"]]
        prior = earlier[-1] if earlier else None
        revenue = tagged_revenue(facts[prior.cik], prior.accession_number) if prior else None
        events.append(
            {
                "event_id": accession,
                "company": filer.key,
                "company_name": filer.name,
                "items": list(row["items"]),
                "event_items": [code for code in row["items"] if code in EVENT_ITEMS],
                "text": text,
                "truncated": truncated,
                "prior_revenue_usd": revenue["value"] if revenue else None,
                "prior_revenue_fiscal_year": revenue["fiscal_year"] if revenue else None,
                "prior_revenue_concept": revenue["concept"] if revenue else None,
                "prior_revenue_period_end": revenue["period_end"] if revenue else None,
                "prior_10k_accession": prior.accession_number if prior else None,
                "prior_90d_items": prior_items(eight_ks, row["accepted"], accession),
                "accepted_utc": row["accepted"].isoformat(),
                "filing_date": row["filing_date"],
                "cik": cik,
                "sections": [section.code for section in sections],
                "press_release": release[1] if release else None,
                "document_url": _archive_url(cik, accession, row["document"]),
                "document_sha256": hashlib.sha256(document).hexdigest(),
            }
        )
    return events, findings


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(args.user_agent, args.cache_dir, transport or _urllib_transport)
    acquired_at = datetime.now(timezone.utc).isoformat()

    events: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    for key in args.filer:
        filer_events, filer_findings = export_filer(client, FILERS[key], args)
        events += filer_events
        findings += filer_findings
    repeated = [
        key for key, seen in Counter(event["event_id"] for event in events).items() if seen > 1
    ]
    findings += [
        {"event_id": key, "company": "", "check": "event_id_repeated", "detail": key}
        for key in repeated
    ]

    args.package_root.mkdir(parents=True, exist_ok=True)
    events_path = args.package_root / "8k_events.jsonl"
    checks_path = args.package_root / "8k_events_checks.csv"
    meta_path = args.package_root / "8k_events.meta.json"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    pd.DataFrame(findings, columns=CHECK_COLUMNS).to_csv(checks_path, index=False)
    by_item = Counter(code for event in events for code in event["event_items"])
    meta = {
        "source": "SEC EDGAR submissions, filing documents and XBRL company facts",
        "filers": {key: list(FILERS[key].ciks) for key in args.filer},
        "forms": ["8-K"],
        "event_items": list(EVENT_ITEMS),
        "left_out_items": list(LEFT_OUT_ITEMS),
        "first_date": args.first_date,
        "last_date": args.last_date,
        "max_text_chars": MAX_TEXT_CHARS,
        "prior_days": PRIOR_DAYS,
        "revenue_concepts": list(REVENUE_CONCEPTS),
        "acquired_at": acquired_at,
        "network_requests": client.requests,
        "repository_commit": _git_commit(),
        "events": len(events),
        "events_by_company": dict(Counter(event["company"] for event in events)),
        "events_by_item": dict(sorted(by_item.items())),
        "truncated": sum(event["truncated"] for event in events),
        "without_revenue": sum(event["prior_revenue_usd"] is None for event in events),
        "with_press_release": sum(event["press_release"] is not None for event in events),
        "check_findings": len(findings),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    for item in findings:
        print(f"BLOCKING {item['event_id']}: {item['check']}: {item['detail']}")
    if findings:
        print("No manifest published: fix the extraction for the events above and re-run.")
        return 1

    if args.manifest_output is not None:
        names = ", ".join(FILERS[key].name for key in args.filer)
        snapshot = write_input_manifest(
            args.manifest_output,
            args.package_root,
            package_id=args.package_id,
            package_revision=args.package_revision,
            files=[
                InputFileSpec(events_path.name, "8-K events, one per line"),
                InputFileSpec(checks_path.name, "Event check findings"),
                InputFileSpec(meta_path.name, "Pull provenance and counts"),
            ],
            provenance={
                "source": meta["source"],
                "acquisition_mode": client.acquisition_mode,
                "acquired_at": acquired_at,
                "producing_command": "python -m scripts.data.export_sec_8k_events",
                "producing_arguments": sys.argv[1:] if argv is None else list(argv),
                "unknowns": [],
            },
            metadata={
                "description": f"8-K events with filing-time context (#308): {names}.",
                "repository_commit": _git_commit(),
            },
        )
        print(json.dumps({"manifest": str(args.manifest_output), "sha256": snapshot.sha256}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
