"""Pull Alphabet's 10-K risk factors from SEC EDGAR for the Jev pilot (#289).

Run from the repository root wherever EDGAR is reachable (since 2026-10-13 that
includes the project's cloud environment). SEC asks every automated client to
name itself with a contact email and to stay under 10 requests a second::

    python -m scripts.data.export_sec_risk_factors \\
        --user-agent "Your Name you@example.com" \\
        --package-root /content/sec_risk_factors \\
        --cache-dir /content/sec_cache \\
        --manifest-output /content/alphabet_risk_factors.r1.json \\
        --package-revision r1

It reads the submissions files of Google Inc. and Alphabet Inc. (with their
older history files), keeps one copy of every 10-K accepted from
``--accepted-from`` (default 2014-01-01, so fiscal 2013 is the reference year
for fiscal 2014), downloads each primary document once into ``--cache-dir``
(outside any checkout), and cuts out Item 1A.

What it writes under ``--package-root/filings``:

- ``alphabet_10k_index.csv``: the kept filings, with acceptance time in UTC and
  the SHA-256 of the document read.
- ``alphabet_10k_duplicates.csv``: copies of the same report under the other CIK.
- ``alphabet_risk_factors.jsonl``: one risk factor per line.
- ``alphabet_extraction_report.csv``: per filing, the encoding read, how many
  risks and categories were found, and how many check findings it raised.
- ``alphabet_extraction_checks.csv``: one row per finding of the checks in
  ``mci_gru.filings.checks`` (#292), run on every filing; empty when clean.
- ``alphabet_risk_factors.meta.json``: provenance.

A filing whose Item 1A cannot be read, or that fails a check, is reported and
stops the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from mci_gru.filings.checks import CheckFinding, check_extraction
from mci_gru.filings.edgar import (
    ALPHABET_CIK,
    GOOGLE_INC_CIK,
    SUBMISSIONS_FILE_URL,
    SUBMISSIONS_URL,
    build_filing_index,
    filings_from_block,
    submissions_blocks,
)
from mci_gru.filings.risk_factors import RiskFactorExtractionError, read_item_1a

#: Seconds between requests: SEC's stated ceiling is 10 a second.
MIN_INTERVAL_SECONDS = 0.11
RETRY_STATUSES = (429, 500, 502, 503, 504)
ATTEMPTS = 4
#: Back-off after a server error or a dropped connection, doubled per attempt.
RETRY_WAIT_SECONDS = 10.0
#: SEC limits an over-eager client for about ten minutes; wait that out on a 429.
RATE_LIMITED_WAIT_SECONDS = 600.0
FILERS = (GOOGLE_INC_CIK, ALPHABET_CIK)
PREFIX = "alphabet"

Transport = Callable[[str, dict[str, str]], bytes]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--accepted-from", default="2014-01-01")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--package-revision", default="r1")
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


def _urllib_transport(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


class EdgarClient:
    """Throttled, cached GETs against EDGAR.

    Documents and older submissions files never change, so each is read from the
    network once. A filer's main submissions file does change (new filings), so
    it is read fresh on every run.
    """

    def __init__(self, user_agent: str, cache_dir: Path, transport: Transport) -> None:
        self.headers = {"User-Agent": user_agent}
        self.cache_dir = cache_dir
        self.transport = transport
        self.requests = 0
        self.cache_hits = 0
        self._last = 0.0

    def get(self, url: str, *, fresh: bool = False) -> bytes:
        name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + Path(url).suffix
        path = self.cache_dir / name
        if path.exists() and not fresh:
            self.cache_hits += 1
            return path.read_bytes()
        for attempt in range(ATTEMPTS):
            wait = MIN_INTERVAL_SECONDS - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                body = self.transport(url, self.headers)
                break
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == ATTEMPTS - 1:
                    raise
                if error.code == 429:
                    time.sleep(RATE_LIMITED_WAIT_SECONDS)
                else:
                    time.sleep(RETRY_WAIT_SECONDS * 2**attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == ATTEMPTS - 1:
                    raise
                time.sleep(RETRY_WAIT_SECONDS * 2**attempt)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    @property
    def acquisition_mode(self) -> str:
        if not self.requests:
            return "cache"
        return "live and cache" if self.cache_hits else "live"

    def get_json(self, url: str, *, fresh: bool = False) -> dict[str, Any]:
        return json.loads(self.get(url, fresh=fresh).decode("utf-8"))


def decode_document(document: bytes) -> tuple[str, str]:
    """The document's text and the encoding read; older filings are not always UTF-8."""
    try:
        return document.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return document.decode("cp1252", errors="replace"), "cp1252"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_filer(client: EdgarClient, cik: str) -> list:
    """Every filing a filer lists, from the recent block and each older file."""
    recent, older = submissions_blocks(client.get_json(SUBMISSIONS_URL.format(cik=cik), fresh=True))
    records = filings_from_block(cik, recent)
    for name in older:
        records.extend(
            filings_from_block(cik, client.get_json(SUBMISSIONS_FILE_URL.format(name=name)))
        )
    return records


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    args = parse_args(argv)
    client = EdgarClient(args.user_agent, args.cache_dir, transport or _urllib_transport)
    acquired_at = datetime.now(timezone.utc).isoformat()

    records = [record for cik in FILERS for record in read_filer(client, cik)]
    index = build_filing_index(records, forms=("10-K",), accepted_from=args.accepted_from)

    out = args.package_root / "filings"
    out.mkdir(parents=True, exist_ok=True)
    index_rows, report_rows, failures, check_rows = [], [], [], []
    risk_path = out / f"{PREFIX}_risk_factors.jsonl"
    with risk_path.open("w", encoding="utf-8") as handle:
        for record in index.filings:
            document = client.get(record.document_url)
            index_rows.append(
                {**record.to_dict(), "document_sha256": hashlib.sha256(document).hexdigest()}
            )
            text, encoding = decode_document(document)
            try:
                reading = read_item_1a(text)
                risks = reading.risk_factors(
                    accession_number=record.accession_number,
                    fiscal_year=record.fiscal_year,
                )
            except RiskFactorExtractionError as error:
                failures.append(f"{record.accession_number}: {error}")
                report_rows.append(
                    {
                        "accession_number": record.accession_number,
                        "fiscal_year": record.fiscal_year,
                        "encoding": encoding,
                        "risks": 0,
                        "categories": 0,
                        "check_findings": 0,
                        "error": str(error),
                    }
                )
                continue
            findings: list[CheckFinding] = check_extraction(reading, risks)
            for finding in findings:
                failures.append(f"{record.accession_number}: {finding.check}: {finding.detail}")
                check_rows.append(
                    {
                        "accession_number": record.accession_number,
                        "fiscal_year": record.fiscal_year,
                        **finding.to_dict(),
                    }
                )
            for risk in risks:
                handle.write(json.dumps(risk.to_dict(), ensure_ascii=False) + "\n")
            report_rows.append(
                {
                    "accession_number": record.accession_number,
                    "fiscal_year": record.fiscal_year,
                    "encoding": encoding,
                    "risks": len(risks),
                    "categories": len({risk.category for risk in risks if risk.category}),
                    "check_findings": len(findings),
                    "error": "",
                }
            )

    index_path = out / f"{PREFIX}_10k_index.csv"
    duplicates_path = out / f"{PREFIX}_10k_duplicates.csv"
    report_path = out / f"{PREFIX}_extraction_report.csv"
    checks_path = out / f"{PREFIX}_extraction_checks.csv"
    meta_path = out / f"{PREFIX}_risk_factors.meta.json"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    pd.DataFrame([record.to_dict() for record in index.duplicates]).to_csv(
        duplicates_path, index=False
    )
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    pd.DataFrame(
        check_rows, columns=["accession_number", "fiscal_year", "check", "risk_id", "detail"]
    ).to_csv(checks_path, index=False)
    meta = {
        "source": "SEC EDGAR submissions and primary documents",
        "filers": list(FILERS),
        "forms": ["10-K"],
        "accepted_from": args.accepted_from,
        "acquired_at": acquired_at,
        "network_requests": client.requests,
        "repository_commit": _git_commit(),
        "filings": len(index.filings),
        "duplicates_dropped": len(index.duplicates),
        "check_findings": len(check_rows),
        "failures": failures,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in meta.items() if key != "failures"}, indent=2))
    for failure in failures:
        print(f"BLOCKING {failure}")
    if failures:
        print("No manifest published: fix the extraction for the filings above and re-run.")
        return 1

    if args.manifest_output is not None:
        files = [
            InputFileSpec(f"filings/{index_path.name}", "10-K filings read, one per report"),
            InputFileSpec(f"filings/{duplicates_path.name}", "Copies dropped as duplicates"),
            InputFileSpec(f"filings/{risk_path.name}", "Item 1A risk factors, one per line"),
            InputFileSpec(f"filings/{report_path.name}", "Extraction counts per filing"),
            InputFileSpec(f"filings/{checks_path.name}", "Extraction check findings"),
            InputFileSpec(f"filings/{meta_path.name}", "Pull provenance"),
        ]
        snapshot = write_input_manifest(
            args.manifest_output,
            args.package_root,
            package_id=f"{PREFIX}_10k_risk_factors",
            package_revision=args.package_revision,
            files=files,
            provenance={
                "source": meta["source"],
                "acquisition_mode": client.acquisition_mode,
                "acquired_at": acquired_at,
                "producing_command": "python -m scripts.data.export_sec_risk_factors",
                "producing_arguments": sys.argv[1:] if argv is None else list(argv),
                "unknowns": [],
            },
            metadata={
                "description": "Alphabet and Google Inc. 10-K Item 1A risk factors (#289).",
                "repository_commit": meta["repository_commit"],
            },
        )
        print(json.dumps({"manifest": str(args.manifest_output), "sha256": snapshot.sha256}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
