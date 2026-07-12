"""Export current S&P 500 top-10-by-market-cap names per GICS sector from LSEG.

The output is a reproducible current-snapshot universe artifact plus a reduced
OHLCV panel for the selected RICs. This is intentionally a current snapshot;
use PIT/rebalanced selection logic before treating results as headline
point-in-time evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_FIELDS = ["TR.RIC", "TR.CommonName", "TR.CompanyMarketCap"]
GICS_SECTOR_CANDIDATES = [
    "TR.GICSSector",
    "TR.GICSSectorName",
    "TR.GICSSectorCode",
]

KNOWN_NON_SECTOR_COLUMNS = {
    "instrument",
    "ric",
    "common name",
    "company common name",
    "company market cap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export S&P 500 top 10 by market cap within each GICS sector."
    )
    parser.add_argument("--chain-ric", default="0#.SPX")
    parser.add_argument("--as-of", required=True, help="Snapshot label, YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--history-start", default="2015-01-01")
    parser.add_argument("--history-end", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--batch-delay", type=float, default=1.0)
    parser.add_argument(
        "--constituents-dir",
        type=Path,
        default=Path("data/raw/constituents"),
    )
    parser.add_argument(
        "--market-dir",
        type=Path,
        default=Path("data/raw/market"),
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only export metadata and selected universe; do not pull OHLCV.",
    )
    return parser.parse_args()


def _clean_name(name: object) -> str:
    return str(name).strip().lower()


def _column_by_names(frame: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {_clean_name(c): c for c in frame.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def _normalise_market_cap(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _normalise_metadata(raw: pd.DataFrame, sector_column: str, sector_field: str) -> pd.DataFrame:
    ric_col = _column_by_names(raw, ["constituent ric", "ric", "instrument"])
    name_col = _column_by_names(raw, ["company common name", "common name", "company name"])
    mcap_col = _column_by_names(raw, ["company market cap", "market cap"])

    missing = [
        label
        for label, value in [
            ("constituent RIC", ric_col),
            ("company name", name_col),
            ("company market cap", mcap_col),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(
            f"LSEG metadata is missing required column(s): {missing}. "
            f"Returned columns: {raw.columns.tolist()}"
        )

    out = pd.DataFrame(
        {
            "kdcode": raw[ric_col].astype(str).str.strip(),
            "company_name": raw[name_col].astype(str).str.strip(),
            "company_market_cap": _normalise_market_cap(raw[mcap_col]),
            "gics_sector": raw[sector_column].astype(str).str.strip(),
        }
    )
    out["gics_sector_field"] = sector_field
    out = out.dropna(subset=["kdcode", "company_market_cap"])
    out = out[(out["kdcode"] != "") & (out["gics_sector"] != "")]
    out = out[~out["kdcode"].str.lower().isin({"nan", "none", "nat"})]
    out = out[out["gics_sector"].str.lower() != "nan"]
    out = out.drop_duplicates(subset=["kdcode"], keep="first")
    return out.sort_values(["gics_sector", "company_market_cap"], ascending=[True, False])


def _find_sector_column(raw: pd.DataFrame, field: str) -> str | None:
    sectorish = [
        c
        for c in raw.columns
        if "sector" in _clean_name(c)
        and _clean_name(c) not in KNOWN_NON_SECTOR_COLUMNS
        and raw[c].notna().any()
    ]
    if sectorish:
        return sectorish[0]

    recognised = {
        _column_by_names(raw, ["constituent ric", "ric", "instrument"]),
        _column_by_names(raw, ["company common name", "common name", "company name"]),
        _column_by_names(raw, ["company market cap", "market cap"]),
    }
    candidates = [
        c
        for c in raw.columns
        if c not in recognised
        and raw[c].notna().any()
        and raw[c].astype(str).str.strip().ne("").any()
    ]
    if len(candidates) == 1:
        return candidates[0]
    print(f"  Could not infer sector column for {field}; columns={raw.columns.tolist()}")
    return None


def fetch_current_gics_metadata(loader: Any, chain_ric: str) -> tuple[pd.DataFrame, str]:
    """Fetch current constituent metadata with the first working GICS sector field."""
    assert loader.rd is not None
    last_error: Exception | None = None
    for sector_field in GICS_SECTOR_CANDIDATES:
        fields = BASE_FIELDS + [sector_field]
        print(f"Trying LSEG sector field {sector_field}...")
        try:
            raw = loader.rd.get_data(universe=[chain_ric], fields=fields)
        except Exception as exc:
            print(f"  {sector_field} failed: {type(exc).__name__}: {exc}")
            last_error = exc
            continue

        if raw is None or raw.empty:
            print(f"  {sector_field} returned no rows")
            continue

        sector_column = _find_sector_column(raw, sector_field)
        if sector_column is None:
            continue

        metadata = _normalise_metadata(raw, sector_column, sector_field)
        if metadata["gics_sector"].nunique() >= 10:
            print(
                f"  Using {sector_field}: {len(metadata)} rows, "
                f"{metadata['gics_sector'].nunique()} sectors"
            )
            return metadata, sector_field
        print(
            f"  {sector_field} produced only {metadata['gics_sector'].nunique()} sectors; "
            "trying next candidate"
        )

    raise RuntimeError(f"Could not fetch GICS sector metadata from LSEG. Last error: {last_error}")


def select_top_by_sector(metadata: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = metadata.copy()
    selected = selected.sort_values(
        ["gics_sector", "company_market_cap", "kdcode"],
        ascending=[True, False, True],
    )
    selected["sector_market_cap_rank"] = selected.groupby("gics_sector").cumcount() + 1
    selected = selected[selected["sector_market_cap_rank"] <= top_n].copy()
    selected = selected.sort_values(["gics_sector", "sector_market_cap_rank", "kdcode"])
    return selected.reset_index(drop=True)


def write_outputs(
    metadata: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    sector_field: str,
) -> dict[str, Any]:
    args.constituents_dir.mkdir(parents=True, exist_ok=True)
    safe_as_of = args.as_of.replace("-", "")
    prefix = f"sp500_gics_top{args.top_n}_mcap_{safe_as_of}"
    metadata_path = args.constituents_dir / f"{prefix}_all_current_metadata.csv"
    selected_path = args.constituents_dir / f"{prefix}.csv"
    sector_map_path = args.constituents_dir / f"{prefix}_sector_map.csv"
    meta_path = args.constituents_dir / f"{prefix}_meta.json"

    metadata.to_csv(metadata_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected[["kdcode", "gics_sector"]].rename(columns={"gics_sector": "sector"}).to_csv(
        sector_map_path, index=False
    )

    sector_counts = selected.groupby("gics_sector")["kdcode"].nunique().to_dict()
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "refinitiv.data.get_data",
        "chain_ric": args.chain_ric,
        "as_of": args.as_of,
        "top_n_per_sector": args.top_n,
        "sector_field": sector_field,
        "metadata_rows": int(len(metadata)),
        "selected_rows": int(len(selected)),
        "selected_unique_kdcodes": int(selected["kdcode"].nunique()),
        "sector_counts": {str(k): int(v) for k, v in sector_counts.items()},
        "outputs": {
            "all_current_metadata": str(metadata_path),
            "selected_universe": str(selected_path),
            "sector_map": str(sector_map_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    meta["outputs"]["metadata"] = str(meta_path)
    return meta


def fetch_history(
    loader: Any,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> dict[str, Any]:
    args.market_dir.mkdir(parents=True, exist_ok=True)
    safe_as_of = args.as_of.replace("-", "")
    start_safe = args.history_start.replace("-", "")
    end_safe = args.history_end.replace("-", "")
    price_path = (
        args.market_dir
        / f"sp500_gics_top{args.top_n}_mcap_{safe_as_of}_lseg_{start_safe}_{end_safe}.csv"
    )
    price_meta_path = price_path.with_suffix(".meta.json")

    rics = selected["kdcode"].dropna().astype(str).drop_duplicates().tolist()
    print(
        f"Fetching OHLCV for {len(rics)} selected RICs: {args.history_start} to {args.history_end}"
    )
    prices = loader.get_historical_prices(
        rics,
        start=args.history_start,
        end=args.history_end,
        batch_size=args.batch_size,
        delay_between_batches=args.batch_delay,
    )
    prices.to_csv(price_path, index=False)

    coverage = (
        prices.groupby("kdcode")["dt"]
        .agg(row_count="size", first_dt="min", last_dt="max")
        .reset_index()
    )
    missing = sorted(set(rics) - set(coverage["kdcode"].astype(str)))
    coverage_path = price_path.with_name(price_path.stem + "_coverage.csv")
    coverage.to_csv(coverage_path, index=False)

    price_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "refinitiv.data.get_history",
        "selected_universe": meta["outputs"]["selected_universe"],
        "start": args.history_start,
        "end": args.history_end,
        "requested_identifiers": len(rics),
        "resolved_identifiers_with_rows": int(coverage["kdcode"].nunique()),
        "missing_identifiers": missing,
        "rows": int(len(prices)),
        "date_min": str(prices["dt"].min()) if len(prices) else None,
        "date_max": str(prices["dt"].max()) if len(prices) else None,
        "batch_size": args.batch_size,
        "outputs": {
            "prices": str(price_path),
            "coverage": str(coverage_path),
        },
    }
    with price_meta_path.open("w", encoding="utf-8") as f:
        json.dump(price_meta, f, indent=2)
    price_meta["outputs"]["metadata"] = str(price_meta_path)
    return price_meta


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    from mci_gru.data.lseg_loader import LSEGLoader

    loader = LSEGLoader()
    loader.connect()
    try:
        metadata, sector_field = fetch_current_gics_metadata(loader, args.chain_ric)
        selected = select_top_by_sector(metadata, args.top_n)
        meta = write_outputs(metadata, selected, args, sector_field)
        print(json.dumps(meta, indent=2))

        if not args.skip_history:
            price_meta = fetch_history(loader, selected, args, meta)
            print(json.dumps(price_meta, indent=2))
    finally:
        loader.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
