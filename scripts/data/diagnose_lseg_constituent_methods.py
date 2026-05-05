"""
Compare LSEG S&P 500 constituent retrieval methods.

This is a diagnostic script for point-in-time constituent provenance:

1. Legacy repo method: ``0#.SPX`` with ``SDate``.
2. LSEG documented dated chain: ``0#.SPX(YYYYMMDD)``.
3. LSEG Joiner/Leaver fields for reconstructing historical membership.

Requires Refinitiv Workspace to be running.
"""

from __future__ import annotations

import argparse

import refinitiv.data as rd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose LSEG constituent methods.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-05-04")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31", "2026-05-01"],
    )
    return parser.parse_args()


def _ric_set(df) -> set[str]:
    if df is None or "Instrument" not in df.columns:
        return set()
    return set(df["Instrument"].dropna().astype(str))


def fetch_legacy(date: str) -> set[str]:
    df = rd.get_data(
        universe=["0#.SPX"],
        fields=["TR.IndexConstituentRIC", "TR.CommonName", "TR.CompanyMarketCap"],
        parameters={"SDate": date},
    )
    rics = _ric_set(df)
    print(f"legacy 0#.SPX SDate={date}: shape={df.shape} unique={len(rics)}")
    print(f"  head={sorted(rics)[:8]}")
    return rics


def fetch_dated_chain(date: str) -> set[str]:
    chain = f"0#.SPX({date.replace('-', '')})"
    df = rd.get_data(
        universe=[chain],
        fields=["TR.PriceClose"],
        parameters={"SDATE": date, "EDATE": date},
    )
    rics = _ric_set(df)
    print(f"dated chain {chain}: shape={df.shape} unique={len(rics)}")
    print(f"  head={sorted(rics)[:8]}")
    return rics


def fetch_joiner_leaver(start: str, end: str):
    fields = [
        "TR.IndexJLConstituentChangeDate",
        "TR.IndexJLConstituentRIC",
        "TR.IndexJLConstituentName",
        "TR.IndexJLConstituentChange",
    ]
    df = rd.get_data(
        universe=[".SPX"],
        fields=fields,
        parameters={"SDATE": start, "EDATE": end, "IC": "B"},
    )
    print(f"joiner/leaver .SPX {start}..{end}: shape={df.shape}")
    print(df.head(20).to_string(index=False))
    print(f"  columns={df.columns.tolist()}")
    return df


def main() -> int:
    args = parse_args()
    rd.open_session()
    try:
        legacy = {}
        dated = {}
        for date in args.dates:
            legacy[date] = fetch_legacy(date)
            dated[date] = fetch_dated_chain(date)

        print("\nDiffs vs first date")
        base_date = args.dates[0]
        for label, sets in [("legacy", legacy), ("dated_chain", dated)]:
            base = sets[base_date]
            print(label)
            for date in args.dates:
                print(f"  {date}: diff={len(sets[date] ^ base)}")

        print("\nJoiner/Leaver")
        fetch_joiner_leaver(args.start, args.end)
    finally:
        rd.close_session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
