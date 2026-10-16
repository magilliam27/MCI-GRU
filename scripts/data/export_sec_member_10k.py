"""Pull S&P 500 members' 10-K risk factors from SEC EDGAR (#309).

Experiment E7 asks whether what Jev reads in 10-Ks predicts returns. It needs
every 10-K filed while its company was in the S&P 500, each with the report
before it::

    python -m scripts.data.export_sec_member_10k \\
        --user-agent "Your Name you@example.com" \\
        --companies sp500_companies.csv --release r1 \\
        --package-root /content/member_10k --cache-dir /content/sec_cache \\
        --manifest-output /content/member_10k.r1.json

``--companies`` is the company table ``build_sp500_members`` writes: one row per
company with its confirmed CIKs, its membership spells and its release. For each
company of ``--release`` it reads the submissions files of every CIK, keeps one
copy of every 10-K and 10-KT, and picks what the membership calls for (see
``mci_gru.filings.membership.member_filings``): every report accepted from
``--accepted-from`` to ``--accepted-to`` while the company was a member, and the
report before each one filed while it was not. Each report's pair partner is the
company's previous report by acceptance time, so a company that changed its
fiscal year end keeps both reports of that year. A company the registry names is
read with its own layout; any other company's report is read with the first of
``LAYOUTS`` whose reading passes every check, and with the first when none does.
Documents are downloaded once
into the gzipped ``--cache-dir`` on ``--download-workers`` threads, which
share SEC's pace, and read with ``--workers`` processes.

What it writes under ``--package-root``:

- ``member_10k_index.csv``: the reports read, one per row: company, CIK,
  accession, form, acceptance time in UTC, period of report, ``role``
  (``member`` or ``prior``), ``prior_accession``, the filer's SIC code, the
  layout read, whether its risks and its section are published, and the
  document's SHA-256.
- ``member_risk_factors.jsonl``: one risk factor per line, with ``company``.
- ``member_item_1a.jsonl``: one report's whole Item 1A per line, its paragraphs
  joined by blank lines, with its word count and ``risks_passed``: whether its
  split into risks passed every check. Every report whose Item 1A holds at least
  ``MIN_SECTION_WORDS`` words is here, so a measure of the whole section reaches
  reports whose risks could not be split. A shorter Item 1A points to another
  document (an exhibit or the annual report), which is not read.
- ``member_extraction_report.csv`` and ``member_extraction_checks.csv``: counts
  and the checks' findings per report.
- ``member_left_out.csv``: each company of the release with no report read, and
  why; with ``--leave-out-flagged``, also each report whose risks are left out,
  with its year, SIC code and reason.
- ``member_10k.meta.json``: provenance and counts, including member reports, and
  member reports left out by year and by SIC division.

A report that fails a check is kept and listed; ``--manifest-output`` is
refused while any remain unless ``--leave-out-flagged`` leaves their risks out
(their sections stay in ``member_item_1a.jsonl``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from mci_gru.filings.checks import check_extraction
from mci_gru.filings.edgar import SUBMISSIONS_URL, build_filing_index
from mci_gru.filings.filers import FILERS
from mci_gru.filings.membership import Spell, member_filings
from mci_gru.filings.risk_factors import LAYOUTS, RiskFactorExtractionError, read_item_1a
from scripts.data.export_sec_risk_factors import (
    EdgarClient,
    Transport,
    _git_commit,
    _urllib_transport,
    decode_document,
    read_filer,
)

FORMS = ("10-K", "10-KT")
PROGRESS_EVERY = 250
#: The earliest acceptance a prior report may have; before it EDGAR's 10-Ks are
#: not read here.
PRIOR_FROM = "2009-01-01"
#: Layouts chosen for the hand-checked companies, by CIK.
KNOWN_LAYOUTS = {cik: filer.layout for filer in FILERS.values() for cik in filer.ciks}
#: An Item 1A shorter than this points to another document rather than holding
#: the risks (Bank of New York Mellon's to its annual report).
MIN_SECTION_WORDS = 400
#: SEC's SIC divisions, by the first two digits' ranges.
SIC_DIVISIONS = (
    (1, 9, "Agriculture, Forestry and Fishing"),
    (10, 14, "Mining"),
    (15, 17, "Construction"),
    (20, 39, "Manufacturing"),
    (40, 49, "Transportation, Communications and Utilities"),
    (50, 51, "Wholesale Trade"),
    (52, 59, "Retail Trade"),
    (60, 67, "Finance, Insurance and Real Estate"),
    (70, 89, "Services"),
    (91, 99, "Public Administration"),
)


def sic_division(sic: str) -> str:
    """The SIC division of a four-digit SIC code, or ``unknown``."""
    if not sic.isdigit():
        return "unknown"
    group = int(sic) // 100
    return next((name for low, high, name in SIC_DIVISIONS if low <= group <= high), "unknown")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--accepted-from", default="2013-01-01")
    parser.add_argument("--accepted-to", default=None, help="Last acceptance date (UTC)")
    parser.add_argument("--workers", type=int, default=1, help="Processes reading reports")
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--leave-out-flagged", action="store_true")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--package-id", default="member_10k_risk_factors")
    parser.add_argument("--package-revision", default="r1")
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


def read_spells(value: str) -> list[Spell]:
    """``TICKER:start:end`` entries separated by ``;``; an empty end is still open."""
    spells = []
    for entry in filter(None, value.split(";")):
        ticker, start, end = entry.split(":")
        spells.append(Spell(ticker, pd.Timestamp(start), pd.Timestamp(end) if end else None))
    return spells


def _read_report(task: tuple[str, str, tuple[str, ...], str, int]) -> dict[str, Any]:
    """Read one cached report: its section, risks, findings and counts (runs in a worker).

    Each of ``layouts`` is tried in turn and the first whose reading passes every
    check is kept; when none does, the first layout's reading is reported.
    """
    cache_dir, url, layouts, accession, fiscal_year = task
    document = EdgarClient("cache only", Path(cache_dir), _cache_only).get(url)
    text, encoding = decode_document(document)
    base = {"sha256": hashlib.sha256(document).hexdigest(), "encoding": encoding}
    readings = []
    for layout in layouts:
        result: dict[str, Any] = {
            **base,
            "layout": layout,
            "paragraphs": [],
            "risks": [],
            "findings": [],
            "error": "",
        }
        try:
            reading = read_item_1a(text, layout=layout)
        except RiskFactorExtractionError as error:
            # No Item 1A: the same under every layout.
            result["error"] = str(error)
            return result
        result["paragraphs"] = list(reading.paragraphs)
        try:
            risks = reading.risk_factors(accession_number=accession, fiscal_year=fiscal_year)
        except RiskFactorExtractionError as error:
            result["error"] = str(error)
        else:
            result["risks"] = [risk.to_dict() for risk in risks]
            result["categories"] = len({risk.category for risk in risks if risk.category})
            result["findings"] = [finding.to_dict() for finding in check_extraction(reading, risks)]
            if not result["findings"]:
                return result
        readings.append(result)
    return readings[0]


def _cache_only(url: str, headers: dict[str, str]) -> bytes:
    raise RuntimeError(f"{url} is not in the cache")


def select_reports(
    client: EdgarClient, company: pd.Series, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """The reports a company's membership calls for, oldest first, with their roles."""
    ciks = str(company["ciks"]).split(";")
    records = [record for cik in ciks for record in read_filer(client, cik)]
    index = build_filing_index(
        records, forms=FORMS, accepted_from=PRIOR_FROM, preferred_cik=ciks[-1]
    )
    reports = list(index.filings)
    start = pd.Timestamp(args.accepted_from, tz="UTC")
    end = (
        pd.Timestamp(args.accepted_to, tz="UTC") + pd.Timedelta(days=1)
        if args.accepted_to
        else None
    )
    spells = read_spells(str(company["spells"]))
    # Membership counts only inside the window; outside it a report is at most a prior.
    in_window = [
        Spell(spell.ticker, max(spell.start, start.tz_localize(None)), _earlier(spell.end, end))
        for spell in spells
    ]
    chosen = member_filings(
        [
            (record.accepted_utc.tz_localize(None).normalize(), record.accession_number)
            for record in reports
        ],
        [spell for spell in in_window if spell.end is None or spell.start < spell.end],
    )
    by_accession = {record.accession_number: record for record in reports}
    # Submissions were just fetched, so these come from the cache.
    sic = {
        cik: str(client.get_json(SUBMISSIONS_URL.format(cik=cik)).get("sic") or "") for cik in ciks
    }
    previous = {
        later.accession_number: earlier.accession_number
        for earlier, later in zip(reports, reports[1:], strict=False)
    }
    return [
        {
            "company": company["key"],
            **by_accession[accession].to_dict(),
            "role": role,
            "prior_accession": previous.get(accession, "") if role == "member" else "",
            "sic": sic.get(by_accession[accession].cik, ""),
        }
        for accession, role in chosen
    ]


LEFT_OUT_COLUMNS = [
    "company",
    "name",
    "spells",
    "accession_number",
    "role",
    "year",
    "sic",
    "section_published",
    "reason",
]


def _share_by(
    rows: list[dict[str, Any]], left: list[dict[str, Any]], key: Any
) -> dict[str, dict[str, float]]:
    """Per group: member reports, those whose risks are left out, and the share."""
    totals, gone = Counter(key(row) for row in rows), Counter(key(row) for row in left)
    return {
        group: {
            "reports": totals[group],
            "left_out": gone[group],
            "share": round(gone[group] / totals[group], 3),
        }
        for group in sorted(totals)
    }


def _earlier(left: pd.Timestamp | None, right: pd.Timestamp | None) -> pd.Timestamp | None:
    if left is None:
        return None if right is None else right.tz_localize(None)
    if right is None:
        return left
    return min(left, right.tz_localize(None))


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(args.user_agent, args.cache_dir, transport or _urllib_transport)
    acquired_at = datetime.now(timezone.utc).isoformat()
    companies = pd.read_csv(args.companies, dtype=str, keep_default_na=False)
    companies = companies[companies["release"] == args.release]
    out = args.package_root
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    for company in companies.itertuples(index=False):
        series = pd.Series(company._asdict())
        reports = select_reports(client, series, args)
        if not reports:
            left_out.append(
                {
                    "company": series["key"],
                    "name": series["name"],
                    "spells": series["spells"],
                    "reason": "no 10-K accepted while a member in the window",
                }
            )
            continue
        rows.extend(reports)

    # Documents download on several threads under the client's one pace.
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for done, _ in enumerate(pool.map(client.get, [row["document_url"] for row in rows]), 1):
            if done % PROGRESS_EVERY == 0:
                print(f"downloaded {done} of {len(rows)} reports", flush=True)

    tasks = [
        (
            str(args.cache_dir),
            row["document_url"],
            (KNOWN_LAYOUTS[row["cik"]],) if row["cik"] in KNOWN_LAYOUTS else LAYOUTS,
            row["accession_number"],
            row["fiscal_year"],
        )
        for row in rows
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_read_report, tasks, chunksize=4))
    else:
        results = [_read_report(task) for task in tasks]

    names = dict(zip(companies["key"], companies["name"], strict=True))
    index_rows, report_rows, check_rows = [], [], []
    risks_file = (out / "member_risk_factors.jsonl").open("w", encoding="utf-8")
    sections_file = (out / "member_item_1a.jsonl").open("w", encoding="utf-8")
    with risks_file, sections_file:
        for row, result in zip(rows, results, strict=True):
            passed = not (result["error"] or result["findings"])
            words = sum(len(paragraph.split()) for paragraph in result["paragraphs"])
            section = words >= MIN_SECTION_WORDS
            risks_published = passed or not args.leave_out_flagged
            index_rows.append(
                {
                    **row,
                    "layout": result["layout"],
                    "risks_published": risks_published,
                    "section_published": section,
                    "document_sha256": result["sha256"],
                }
            )
            report_rows.append(
                {
                    "company": row["company"],
                    "accession_number": row["accession_number"],
                    "layout": result["layout"],
                    "encoding": result["encoding"],
                    "section_words": words,
                    "risks": len(result["risks"]),
                    "categories": result.get("categories", 0),
                    "check_findings": len(result["findings"]),
                    "error": result["error"],
                }
            )
            for finding in result["findings"]:
                check_rows.append(
                    {
                        "company": row["company"],
                        "accession_number": row["accession_number"],
                        **finding,
                    }
                )
            if section:
                sections_file.write(
                    json.dumps(
                        {
                            "company": row["company"],
                            "accession_number": row["accession_number"],
                            "accepted_utc": row["accepted_utc"],
                            "role": row["role"],
                            "prior_accession": row["prior_accession"],
                            "section_words": words,
                            "risks_passed": passed,
                            "text": "\n\n".join(result["paragraphs"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if not risks_published:
                left_out.append(
                    {
                        "company": row["company"],
                        "name": names[row["company"]],
                        "accession_number": row["accession_number"],
                        "role": row["role"],
                        "year": row["accepted_utc"][:4],
                        "sic": row["sic"],
                        "section_published": section,
                        "reason": result["error"]
                        or "failed the checks: "
                        + ", ".join(sorted({finding["check"] for finding in result["findings"]})),
                    }
                )
                continue
            for risk in result["risks"]:
                risks_file.write(
                    json.dumps({"company": row["company"], **risk}, ensure_ascii=False) + "\n"
                )

    pd.DataFrame(index_rows).to_csv(out / "member_10k_index.csv", index=False)
    pd.DataFrame(report_rows).to_csv(out / "member_extraction_report.csv", index=False)
    pd.DataFrame(
        check_rows, columns=["company", "accession_number", "check", "risk_id", "detail"]
    ).to_csv(out / "member_extraction_checks.csv", index=False)
    pd.DataFrame(left_out, columns=LEFT_OUT_COLUMNS).to_csv(
        out / "member_left_out.csv", index=False
    )

    members = [row for row in index_rows if row["role"] == "member"]
    failing = [row for row in report_rows if row["error"] or row["check_findings"]]
    blocking = [] if args.leave_out_flagged else failing
    left_members = [row for row in members if not row["risks_published"]]
    meta = {
        "source": "SEC EDGAR submissions and primary documents",
        "companies_table": str(args.companies),
        "release": args.release,
        "forms": list(FORMS),
        "accepted_from": args.accepted_from,
        "accepted_to": args.accepted_to,
        "acquired_at": acquired_at,
        "network_requests": client.requests,
        "repository_commit": _git_commit(),
        "companies": int(len(companies)),
        "companies_read": len({row["company"] for row in index_rows}),
        "companies_left_out": sum(1 for row in left_out if "accession_number" not in row),
        "reports": len(index_rows),
        "member_reports": len(members),
        "member_reports_by_year": dict(
            sorted(Counter(row["accepted_utc"][:4] for row in members).items())
        ),
        "risks": int(sum(row["risks"] for row in report_rows)),
        "reports_failing_checks": len(failing),
        "companies_failing_checks": len({row["company"] for row in failing}),
        "findings_by_check": dict(Counter(row["check"] for row in check_rows)),
        "read_errors": sum(1 for row in report_rows if row["error"]),
        "reports_by_layout": dict(Counter(row["layout"] for row in report_rows)),
        "sections_published": sum(1 for row in index_rows if row["section_published"]),
        "member_sections_published": sum(1 for row in members if row["section_published"]),
        "member_reports_risks_left_out": len(left_members),
        "member_reports_risks_left_out_by_year": _share_by(
            members, left_members, lambda row: row["accepted_utc"][:4]
        ),
        "member_reports_risks_left_out_by_sic_division": _share_by(
            members, left_members, lambda row: sic_division(row["sic"])
        ),
        "sic_note": "SIC is each filer's code in its SEC submissions file today, not at filing.",
    }
    (out / "member_10k.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if blocking:
        print(f"{len(blocking)} reports failed the checks; no manifest published.")
        return 1

    if args.manifest_output is not None:
        files = [
            InputFileSpec("member_10k_index.csv", "10-K reports read, one per row"),
            InputFileSpec("member_risk_factors.jsonl", "Item 1A risk factors, one per line"),
            InputFileSpec("member_item_1a.jsonl", "Whole Item 1A sections, one per line"),
            InputFileSpec("member_extraction_report.csv", "Extraction counts per report"),
            InputFileSpec("member_extraction_checks.csv", "Extraction check findings"),
            InputFileSpec("member_left_out.csv", "Companies and reports left out, and why"),
            InputFileSpec("member_10k.meta.json", "Pull provenance and counts"),
        ]
        snapshot = write_input_manifest(
            args.manifest_output,
            out,
            package_id=args.package_id,
            package_revision=args.package_revision,
            files=files,
            provenance={
                "source": "SEC EDGAR submissions and primary documents",
                "acquisition_mode": client.acquisition_mode,
                "acquired_at": acquired_at,
                "producing_command": "python -m scripts.data.export_sec_member_10k",
                "producing_arguments": sys.argv[1:] if argv is None else list(argv),
                "unknowns": [],
            },
            metadata={
                "description": f"S&P 500 members' 10-K risk factors (#309), release {args.release}.",
                "repository_commit": _git_commit(),
            },
        )
        print(json.dumps({"manifest": str(args.manifest_output), "sha256": snapshot.sha256}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
