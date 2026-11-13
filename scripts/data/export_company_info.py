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

What it writes under ``--out-dir``:

- ``earnings_events.csv``: one row per release (``mci_gru.data.company_info``).
- ``company_identity.csv``: per EODHD symbol, the name and CIK EODHD gives, whether it
  is delisted, and how many releases it gave the panel.
- ``earnings_events.meta.json``: inputs with their sha256, counts by timing and year,
  and the symbols with no history.
- ``earnings_raw.jsonl``: every response read, cached or fresh.
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

from mci_gru.data.company_info import earnings_from_eodhd, symbol_segments
from scripts.data.export_eodhd_pit_prices import KEY_ENV, EodhdClient

HOLDOUT_START = "2026-01-22"
DEFAULT_LAST_DATE = "2025-12-31"
FUNDAMENTALS_FILTER = "General,Earnings::History"
THREADS = 8


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
    args = parser.parse_args(argv)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
