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
fiscal year end keeps both reports of that year. Documents are downloaded once
into the gzipped ``--cache-dir`` on ``--download-workers`` threads, which
share SEC's pace, and read with ``--workers`` processes.

What it writes under ``--package-root``:

- ``member_10k_index.csv``: the reports read, one per row: company, CIK,
  accession, form, acceptance time in UTC, period of report, ``role``
  (``member`` or ``prior``), ``prior_accession`` and the document's SHA-256.
- ``member_risk_factors.jsonl``: one risk factor per line, with ``company``.
- ``member_extraction_report.csv`` and ``member_extraction_checks.csv``: counts
  and the six checks' findings per report.
- ``member_left_out.csv``: each company of the release with no report read, and
  why.
- ``member_10k.meta.json``: provenance and counts, including member reports and
  left-out companies per year.

A report that fails a check is kept and listed; ``--manifest-output`` is
refused while any remain unless ``--leave-out-flagged`` moves their companies'
reports to ``member_left_out.csv``.
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
from mci_gru.filings.edgar import build_filing_index
from mci_gru.filings.filers import FILERS
from mci_gru.filings.membership import Spell, member_filings
from mci_gru.filings.risk_factors import RiskFactorExtractionError, read_item_1a
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


def _read_report(task: tuple[str, str, str, str, int]) -> dict[str, Any]:
    """Read one cached report: its risks, findings and counts (runs in a worker)."""
    cache_dir, url, layout, accession, fiscal_year = task
    document = EdgarClient("cache only", Path(cache_dir), _cache_only).get(url)
    text, encoding = decode_document(document)
    result: dict[str, Any] = {
        "sha256": hashlib.sha256(document).hexdigest(),
        "encoding": encoding,
        "risks": [],
        "findings": [],
        "error": "",
    }
    try:
        reading = read_item_1a(text, layout=layout)
        risks = reading.risk_factors(accession_number=accession, fiscal_year=fiscal_year)
    except RiskFactorExtractionError as error:
        result["error"] = str(error)
        return result
    result["risks"] = [risk.to_dict() for risk in risks]
    result["categories"] = len({risk.category for risk in risks if risk.category})
    result["findings"] = [finding.to_dict() for finding in check_extraction(reading, risks)]
    return result


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
        }
        for accession, role in chosen
    ]


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
        for report in reports:
            report["layout"] = KNOWN_LAYOUTS.get(report["cik"], "blocks")
            rows.append(report)

    # Documents download on several threads under the client's one pace.
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for done, _ in enumerate(pool.map(client.get, [row["document_url"] for row in rows]), 1):
            if done % PROGRESS_EVERY == 0:
                print(f"downloaded {done} of {len(rows)} reports", flush=True)

    tasks = [
        (
            str(args.cache_dir),
            row["document_url"],
            row["layout"],
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

    flagged = {
        row["company"]
        for row, result in zip(rows, results, strict=True)
        if result["error"] or result["findings"]
    }
    if args.leave_out_flagged:
        for key in sorted(flagged):
            company = companies[companies["key"] == key].iloc[0]
            left_out.append(
                {
                    "company": key,
                    "name": company["name"],
                    "spells": company["spells"],
                    "reason": "a report failed the extraction checks",
                }
            )
        kept = [
            (row, result)
            for row, result in zip(rows, results, strict=True)
            if row["company"] not in flagged
        ]
    else:
        kept = list(zip(rows, results, strict=True))

    index_rows, report_rows, check_rows = [], [], []
    with (out / "member_risk_factors.jsonl").open("w", encoding="utf-8") as handle:
        for row, result in kept:
            index_rows.append({**row, "document_sha256": result["sha256"]})
            report_rows.append(
                {
                    "company": row["company"],
                    "accession_number": row["accession_number"],
                    "encoding": result["encoding"],
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
            for risk in result["risks"]:
                handle.write(
                    json.dumps({"company": row["company"], **risk}, ensure_ascii=False) + "\n"
                )

    pd.DataFrame(index_rows).to_csv(out / "member_10k_index.csv", index=False)
    pd.DataFrame(report_rows).to_csv(out / "member_extraction_report.csv", index=False)
    pd.DataFrame(
        check_rows, columns=["company", "accession_number", "check", "risk_id", "detail"]
    ).to_csv(out / "member_extraction_checks.csv", index=False)
    pd.DataFrame(left_out, columns=["company", "name", "spells", "reason"]).to_csv(
        out / "member_left_out.csv", index=False
    )

    members = [row for row in index_rows if row["role"] == "member"]
    failing = [row for row in report_rows if row["error"] or row["check_findings"]]
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
        "companies_left_out": len(left_out),
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
    }
    (out / "member_10k.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if failing:
        print(f"{len(failing)} reports failed the checks; no manifest published.")
        return 1

    if args.manifest_output is not None:
        files = [
            InputFileSpec("member_10k_index.csv", "10-K reports read, one per row"),
            InputFileSpec("member_risk_factors.jsonl", "Item 1A risk factors, one per line"),
            InputFileSpec("member_extraction_report.csv", "Extraction counts per report"),
            InputFileSpec("member_extraction_checks.csv", "Extraction check findings"),
            InputFileSpec("member_left_out.csv", "Companies with no report read, and why"),
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
