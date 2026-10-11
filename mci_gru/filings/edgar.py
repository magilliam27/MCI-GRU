"""The EDGAR filing index and when each filing was knowable (#289).

What this module decides:

- **Which filings are in the index.** EDGAR's submissions file for a filer holds
  parallel arrays (``accessionNumber``, ``form``, ``acceptanceDateTime``, ...):
  the ``recent`` block and, for long histories, older blocks in separate files.
  Every block is read the same way.
- **One copy per report.** Alphabet became Google's holding company in October
  2015, and some reports were filed under both CIKs with different accession
  numbers (the Q3 2015 10-Q). A report is identified by its form and period of
  report; of several copies the earliest accepted is kept, because that is when
  the text was first public, and ties go to the preferred filer.
- **When a filing was knowable.** ``acceptanceDateTime`` is true UTC. That was
  checked against filing headers, which state Eastern time: the Q2 2026 results
  8-K reads 16:01:36 in its header and ``2026-07-22T20:01:36.000Z`` in the
  submissions file, and the Q4 2025 one 16:01:37 against
  ``2026-02-04T21:01:37.000Z``. A filing first counts for the forecast of the
  first session whose prediction-clock instant is at or after its acceptance
  (#225: equality counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mci_gru.data.pit import PredictionClock

GOOGLE_INC_CIK = "0001288776"
ALPHABET_CIK = "0001652044"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

_REQUIRED_ARRAYS = (
    "accessionNumber",
    "form",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "primaryDocument",
)


class FilingIndexError(ValueError):
    """A submissions block cannot be read as a filing index."""


@dataclass(frozen=True)
class FilingRecord:
    """One filing, as EDGAR lists it."""

    cik: str
    accession_number: str
    form: str
    filing_date: str
    report_date: str
    accepted_utc: pd.Timestamp
    primary_document: str

    @property
    def document_url(self) -> str:
        return DOCUMENT_URL.format(
            cik=int(self.cik),
            accession=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )

    @property
    def fiscal_year(self) -> int:
        """Year of the period of report; a calendar fiscal year for Alphabet."""
        return int(self.report_date[:4])

    def to_dict(self) -> dict[str, Any]:
        return {
            "cik": self.cik,
            "accession_number": self.accession_number,
            "form": self.form,
            "filing_date": self.filing_date,
            "report_date": self.report_date,
            "accepted_utc": self.accepted_utc.isoformat(),
            "primary_document": self.primary_document,
            "fiscal_year": self.fiscal_year,
            "document_url": self.document_url,
        }


def parse_acceptance(value: str) -> pd.Timestamp:
    """EDGAR's ``acceptanceDateTime`` as a UTC instant; a naive value is refused."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise FilingIndexError(f"acceptanceDateTime {value!r} carries no timezone")
    return stamp.tz_convert("UTC")


def submissions_blocks(main: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The ``recent`` block of a submissions file and the names of its older files."""
    filings = main.get("filings")
    if not isinstance(filings, dict) or not isinstance(filings.get("recent"), dict):
        raise FilingIndexError("submissions file has no filings.recent block")
    names = [str(entry["name"]) for entry in filings.get("files", [])]
    return filings["recent"], names


def filings_from_block(cik: str, block: Mapping[str, Any]) -> list[FilingRecord]:
    """Every filing in one block of parallel arrays."""
    missing = [key for key in _REQUIRED_ARRAYS if key not in block]
    if missing:
        raise FilingIndexError(f"submissions block for CIK {cik} lacks {missing}")
    lengths = {len(block[key]) for key in _REQUIRED_ARRAYS}
    if len(lengths) != 1:
        raise FilingIndexError(f"submissions block for CIK {cik} has ragged arrays")
    return [
        FilingRecord(
            cik=cik,
            accession_number=str(block["accessionNumber"][i]),
            form=str(block["form"][i]),
            filing_date=str(block["filingDate"][i]),
            report_date=str(block["reportDate"][i]),
            accepted_utc=parse_acceptance(str(block["acceptanceDateTime"][i])),
            primary_document=str(block["primaryDocument"][i]),
        )
        for i in range(lengths.pop())
    ]


@dataclass(frozen=True)
class FilingIndex:
    """The kept filings, oldest first, and the copies dropped as duplicates."""

    filings: tuple[FilingRecord, ...]
    duplicates: tuple[FilingRecord, ...]


def build_filing_index(
    records: Iterable[FilingRecord],
    *,
    forms: Sequence[str],
    accepted_from: str,
    preferred_cik: str = ALPHABET_CIK,
) -> FilingIndex:
    """Filter to ``forms`` accepted on or after ``accepted_from`` (UTC date), one copy per report."""
    start = pd.Timestamp(accepted_from, tz="UTC")
    wanted = set(forms)
    seen: dict[str, FilingRecord] = {}
    candidates = [
        record
        for record in records
        if record.form in wanted and record.accepted_utc >= start and record.report_date
    ]
    for record in candidates:
        if record.accession_number in seen:
            continue
        seen[record.accession_number] = record

    def _priority(record: FilingRecord) -> tuple[pd.Timestamp, int]:
        return record.accepted_utc, 0 if record.cik == preferred_cik else 1

    kept: dict[tuple[str, str], FilingRecord] = {}
    duplicates: list[FilingRecord] = []
    for record in sorted(seen.values(), key=_priority):
        report = (record.form, record.report_date)
        if report in kept:
            duplicates.append(record)
        else:
            kept[report] = record
    filings = sorted(kept.values(), key=lambda record: record.accepted_utc)
    return FilingIndex(tuple(filings), tuple(duplicates))


def first_forecast_session(
    accepted_utc: pd.Timestamp,
    sessions: Sequence[str],
    clock: PredictionClock,
) -> str | None:
    """First session whose forecast may use a filing; ``None`` after the last session."""
    for session in sorted(sessions):
        if clock.at(session) >= accepted_utc:
            return session
    return None
