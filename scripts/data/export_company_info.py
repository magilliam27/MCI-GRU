"""Pull the full S&P 500 panel's earnings releases from EODHD into a preserved CSV (#393).

Each panel code was priced from EODHD symbols over dated segments (the price pull's
``*_symbols.json``). This script reads each of those symbols' earnings history
(``fundamentals/<symbol>?filter=General,Earnings::History``, about 10 API calls each)
and keeps the releases whose report date falls inside the segment that symbol priced,
so a reused ticker never lends its releases to the wrong company::

    python -m scripts.data.export_company_info \\
        --symbols path/to/sp500_pit_full_..._symbols.json \\
        --cache-dir /content/eodhd_cache --out-dir /content/company_info \\
        --first-date 2015-01-01 --last-date 2025-12-31 --key-from-proxy

``--last-date`` defaults to 2025-12-31, the last W5 test date, so the locked 2026
holdout is never read; a later date is refused.

**SEC fundamentals** (``--sec-user-agent``). Each panel code's CIK is the one EODHD
gives for the symbol that priced it (``--cik-overrides`` names any EODHD lacks), and
the company's SEC companyfacts file gives its first-filed values
(``mci_gru.data.company_info.first_filed_facts``). Share counts are put on the price
panel's split basis with the split records in the price pull's ``--price-raw`` log.
A value filed after ``--last-date`` is left out. SEC asks every client to name a
contact in its User-Agent; the script sends nothing to SEC without one::

    python -m scripts.data.export_company_info ... \
        --sec-user-agent "Name contact@example.com" --sec-cache-dir /content/sec_cache \
        --price-raw path/to/..._raw.jsonl --cik-overrides cik_overrides.json

What it writes under ``--out-dir``:

- ``earnings_events.csv``: one row per release (``mci_gru.data.company_info``).
- ``company_identity.csv``: per EODHD symbol, the name and CIK EODHD gives, whether it
  is delisted, and how many releases it gave the panel.
- ``earnings_events.meta.json``: inputs with their sha256, counts by timing and year,
  and the symbols with no history.
- ``earnings_raw.jsonl``: every response read, cached or fresh.
- With ``--sec-user-agent``: ``fundamentals.csv`` (``FUNDAMENTAL_ROW_COLUMNS``),
  ``fundamentals_identity.csv`` (per panel code and CIK: EODHD's and SEC's names,
  whether they match, rows kept, any error) and ``fundamentals.meta.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.company_info import (
    earnings_from_eodhd,
    first_filed_facts,
    fundamental_rows,
    symbol_segments,
)
from mci_gru.data.eodhd_prices import splits_frame
from mci_gru.filings.membership import names_match
from scripts.data.export_eodhd_pit_prices import KEY_ENV, EodhdClient
from scripts.data.export_sec_risk_factors import EdgarClient, _urllib_transport

HOLDOUT_START = "2026-01-22"
DEFAULT_LAST_DATE = "2025-12-31"
FUNDAMENTALS_FILTER = "General,Earnings::History"
THREADS = 8
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
#: The price panel's split basis: split records up to its last date adjust it.
PANEL_SPLIT_AS_OF = "2026-07-31"
#: Filings before a later segment's start that still describe that segment's company.
SEGMENT_HISTORY_DAYS = 400


class FundamentalsClient(EodhdClient):
    """The price pull's cached client, with the fundamentals endpoint."""

    def earnings(self, symbol: str) -> dict[str, Any]:
        body = self._get(
            "fundamentals", symbol, f"fundamentals/{symbol}", {"filter": FUNDAMENTALS_FILTER}
        )
        return body if isinstance(body, dict) else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", type=Path, required=True, help="price pull *_symbols.json")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--first-date", default="2015-01-01")
    parser.add_argument("--last-date", default=DEFAULT_LAST_DATE)
    parser.add_argument(
        "--key-from-proxy",
        action="store_true",
        help=f"Send no key when {KEY_ENV} is unset: a network proxy adds it to each request",
    )
    parser.add_argument("--sec-user-agent", help="SEC contact, 'Name email'; enables the SEC pull")
    parser.add_argument("--sec-cache-dir", type=Path)
    parser.add_argument("--price-raw", type=Path, help="the price pull's *_raw.jsonl log")
    parser.add_argument(
        "--cik-overrides", type=Path, help="JSON {kdcode: cik} for CIKs EODHD lacks"
    )
    args = parser.parse_args(argv)
    if args.sec_user_agent is not None:
        if "@" not in args.sec_user_agent:
            parser.error("--sec-user-agent must carry a contact email, as SEC asks")
        if args.sec_cache_dir is None or args.price_raw is None:
            parser.error("--sec-user-agent needs --sec-cache-dir and --price-raw")
    if args.last_date >= HOLDOUT_START:
        parser.error(f"--last-date must be before the locked holdout ({HOLDOUT_START})")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_rows(bodies: dict[str, dict[str, Any]], events: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: EODHD's name, CIK and delisting, and the releases the panel kept."""
    kept = Counter(events["source_symbol"])
    rows = []
    for symbol, body in sorted(bodies.items()):
        general = body.get("General") or {}
        rows.append(
            {
                "symbol": symbol,
                "name": general.get("Name"),
                "cik": general.get("CIK"),
                "is_delisted": general.get("IsDelisted"),
                "delisted_date": general.get("DelistedDate"),
                "history_rows": len(body.get("Earnings::History") or {}),
                "releases_kept": kept.get(symbol, 0),
            }
        )
    return pd.DataFrame(rows)


def panel_splits(price_raw: Path, segments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each panel code's split records (``dt, ratio``), as its price panel applied them.

    A symbol's split counts for a panel code when its date falls inside the segment that
    symbol priced. Only records up to ``PANEL_SPLIT_AS_OF`` adjust the panel.
    """
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    with price_raw.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            endpoint = str(record.get("endpoint", ""))
            if not endpoint.startswith("splits/") or not isinstance(record.get("body"), list):
                continue
            symbol = endpoint.split("/", 1)[1]
            by_symbol.setdefault(symbol, []).extend(record["body"])
    out: dict[str, pd.DataFrame] = {}
    for row in segments.itertuples(index=False):
        records = {str(r["date"]): r for r in by_symbol.get(row.symbol, [])}
        inside = [r for d, r in records.items() if row.start <= d <= row.end]
        frame = splits_frame(inside, as_of=PANEL_SPLIT_AS_OF)
        if row.kdcode in out:
            frame = pd.concat([out[row.kdcode], frame], ignore_index=True)
        out[row.kdcode] = frame.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    return out


def segment_ciks(
    segments: pd.DataFrame, identity: pd.DataFrame, overrides: dict[str, str]
) -> pd.DataFrame:
    """``kdcode, symbol, start, end, cik, eodhd_name`` per segment; CIK empty if unknown."""
    names = identity.set_index("symbol")
    rows = []
    for row in segments.itertuples(index=False):
        cik = overrides.get(row.kdcode)
        name = names["name"].get(row.symbol)
        if cik is None:
            raw = names["cik"].get(row.symbol)
            cik = str(int(float(raw))).zfill(10) if pd.notna(raw) and str(raw) else ""
        rows.append({**row._asdict(), "cik": str(cik).zfill(10) if cik else "", "eodhd_name": name})
    return pd.DataFrame(rows)


def export_fundamentals(
    args: argparse.Namespace, segments: pd.DataFrame, identity: pd.DataFrame
) -> dict[str, Any]:
    """Pull companyfacts for every panel CIK and write the fundamentals files."""
    overrides = {}
    if args.cik_overrides is not None:
        overrides = json.loads(args.cik_overrides.read_text(encoding="utf-8"))
    mapped = segment_ciks(segments, identity, overrides)
    splits = panel_splits(args.price_raw, segments)
    client = EdgarClient(args.sec_user_agent, args.sec_cache_dir, _urllib_transport)
    args.sec_cache_dir.mkdir(parents=True, exist_ok=True)
    ciks = sorted(set(mapped["cik"]) - {""})
    facts: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    def fetch(cik: str) -> None:
        try:
            facts[cik] = client.get_json(COMPANY_FACTS_URL.format(cik=cik))
        except Exception as error:  # a missing filer is recorded, not fatal
            errors[cik] = type(error).__name__ + ": " + str(error)[:200]

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, ciks))

    parts, checks = [], []
    first_start = mapped.groupby("kdcode")["start"].min()
    for row in mapped.itertuples(index=False):
        check = {
            "kdcode": row.kdcode,
            "symbol": row.symbol,
            "cik": row.cik,
            "eodhd_name": row.eodhd_name,
            "sec_name": None,
            "names_match": None,
            "rows": 0,
            "error": errors.get(row.cik, "" if row.cik else "no CIK"),
        }
        body = facts.get(row.cik)
        if body is not None:
            sec_name = str(body.get("entityName") or "")
            check["sec_name"] = sec_name
            check["names_match"] = bool(row.eodhd_name) and names_match(
                str(row.eodhd_name), sec_name
            )
            table = fundamental_rows(
                first_filed_facts(body), row.kdcode, row.cik, splits.get(row.kdcode)
            )
            keep = (table["filed"] <= args.last_date) & (table["filed"] <= row.end)
            if row.start > first_start[row.kdcode]:
                floor = pd.Timestamp(row.start) - pd.Timedelta(days=SEGMENT_HISTORY_DAYS)
                keep &= table["filed"] >= floor.strftime("%Y-%m-%d")
            table = table.loc[keep]
            check["rows"] = len(table)
            parts.append(table)
        checks.append(check)
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows = rows.drop_duplicates(["kdcode", "field", "end"], keep="last")
    path = args.out_dir / "fundamentals.csv"
    rows.to_csv(path, index=False, lineterminator="\n")
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(args.out_dir / "fundamentals_identity.csv", index=False, lineterminator="\n")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "issue": 393,
        "source": "SEC companyfacts, first-filed values (mci_gru.data.company_info)",
        "last_filed": args.last_date,
        "ciks": len(ciks),
        "cik_errors": errors,
        "panel_codes_with_rows": int(rows["kdcode"].nunique()) if len(rows) else 0,
        "segments_without_cik": sorted(mapped.loc[mapped["cik"] == "", "kdcode"]),
        "segments_name_mismatch": check_frame.loc[
            check_frame["names_match"].eq(False), ["kdcode", "eodhd_name", "sec_name"]
        ].to_dict("records"),
        "rows_by_field": dict(Counter(rows["field"])) if len(rows) else {},
        "sec_requests": client.requests,
        "sec_cache_hits": client.cache_hits,
        "price_raw_sha256": _sha256(args.price_raw),
        "cik_overrides": overrides,
        "fundamentals_sha256": _sha256(path),
    }
    (args.out_dir / "fundamentals.meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = json.loads(args.symbols.read_text(encoding="utf-8"))
    segments = symbol_segments(record)
    symbols = sorted(set(segments["symbol"]))
    raw_log: list[dict[str, Any]] = []
    client = FundamentalsClient(
        os.environ.get(KEY_ENV, ""), args.cache_dir, raw_log, key_from_proxy=args.key_from_proxy
    )
    with ThreadPoolExecutor(THREADS) as pool:
        bodies = dict(zip(symbols, pool.map(client.earnings, symbols), strict=True))
    history = {s: (b.get("Earnings::History") or {}) for s, b in bodies.items()}
    events = earnings_from_eodhd(history, segments, args.first_date, args.last_date)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "earnings_events.csv"
    events.to_csv(events_path, index=False, lineterminator="\n")
    identity = identity_rows(bodies, events)
    identity.to_csv(args.out_dir / "company_identity.csv", index=False, lineterminator="\n")
    with (args.out_dir / "earnings_raw.jsonl").open("w", encoding="utf-8") as handle:
        for item in raw_log:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    years = Counter(d[:4] for d in events["report_date"])
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "issue": 393,
        "source": f"EODHD fundamentals/<symbol>?filter={FUNDAMENTALS_FILTER}",
        "symbols_json": args.symbols.name,
        "symbols_json_sha256": _sha256(args.symbols),
        "first_date": args.first_date,
        "last_date": args.last_date,
        "symbols": len(symbols),
        "symbols_without_history": sorted(s for s, h in history.items() if not h),
        "symbols_without_cik": sorted(identity.loc[identity["cik"].isna(), "symbol"]),
        "panel_codes_with_releases": int(events["kdcode"].nunique()),
        "panel_codes": int(segments["kdcode"].nunique()),
        "releases": len(events),
        "releases_with_estimate": int(events["eps_estimate"].notna().sum()),
        "releases_by_timing": dict(Counter(events["timing"])),
        "releases_by_year": dict(sorted(years.items())),
        "vendor_calls": client.calls,
        "earnings_events_sha256": _sha256(events_path),
    }
    (args.out_dir / "earnings_events.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: meta[k] for k in ("releases", "panel_codes_with_releases")}))
    if args.sec_user_agent is not None:
        sec = export_fundamentals(args, segments, identity)
        print(json.dumps({k: sec[k] for k in ("ciks", "panel_codes_with_rows", "rows_by_field")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
