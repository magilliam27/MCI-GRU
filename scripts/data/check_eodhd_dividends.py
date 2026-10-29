"""Check that a dividend revision of an EODHD package changes nothing but the dividends (#361).

Revision r4 adds a ``dividend`` column to r3's panel and is meant to change
nothing else, so a price-basis run on r4 reproduces r3. This compares the two
panels and reads the pull's own findings:

1. The same rows (``kdcode``, ``dt``) in both revisions.
2. ``open``, ``high``, ``low``, ``close`` and ``volume`` identical, row for row.
3. Every dividend is a finite, nonnegative number, and the previous revision
   has no ``dividend`` column, or one that is all zero.
4. The pull recorded no ``dividend_disagrees_with_adjusted_close`` finding: each
   dividend matches the step EODHD's ``adjusted_close`` takes on its ex-date.

It exits 1 when any of these fails, and writes the evidence either way: a
summary, every differing row, and per-name dividend counts and trailing yields
for reading. Reads only files::

    python -m scripts.data.check_eodhd_dividends \\
        --package-root <r4 package> --previous-root <r3 package> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UNCHANGED = ["open", "high", "low", "close", "volume"]
DISAGREES = "dividend_disagrees_with_adjusted_close"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _one(root: Path, pattern: str) -> Path:
    (path,) = sorted((root / "market").glob(pattern))
    return path


def changed_rows(panel: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Rows only one revision has, or whose unchanged columns differ at all."""
    merged = panel.merge(
        previous, on=["kdcode", "dt"], how="outer", suffixes=("", "_prev"), indicator=True
    )
    differs = merged["_merge"] != "both"
    for column in UNCHANGED:
        a, b = merged[column], merged[f"{column}_prev"]
        differs |= ~((a == b) | (a.isna() & b.isna()))
    keep = ["kdcode", "dt", "_merge", *[c for col in UNCHANGED for c in (col, f"{col}_prev")]]
    return merged.loc[differs, keep]


def dividend_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Per name: dividend rows, total paid, and the mean yearly sum over the mean close."""
    paid = panel[panel["dividend"] > 0]
    if paid.empty:
        return pd.DataFrame(columns=["kdcode", "dividend_rows", "first_dt", "last_dt"])
    by_name = paid.groupby("kdcode").agg(
        dividend_rows=("dividend", "size"),
        first_dt=("dt", "min"),
        last_dt=("dt", "max"),
        largest=("dividend", "max"),
    )
    years = panel.assign(year=panel["dt"].str[:4])
    yearly = years.groupby(["kdcode", "year"]).agg(
        paid=("dividend", "sum"), close=("close", "mean")
    )
    by_name["mean_yearly_yield"] = (yearly["paid"] / yearly["close"]).groupby("kdcode").mean()
    return by_name.reset_index().sort_values("mean_yearly_yield", ascending=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = _one(args.package_root, "*_eodhd_*[0-9].csv")
    panel = pd.read_csv(panel_path, dtype={"kdcode": str, "dt": str})
    previous = pd.read_csv(
        _one(args.previous_root, panel_path.name), dtype={"kdcode": str, "dt": str}
    )
    meta = json.loads(_one(args.package_root, "*.meta.json").read_text(encoding="utf-8"))

    changed = changed_rows(panel, previous)
    changed.to_csv(args.out_dir / "changed_rows.csv", index=False)
    has_column = "dividend" in panel.columns
    dividend = panel["dividend"] if has_column else pd.Series(dtype=float)
    invalid = panel.loc[~(np.isfinite(dividend) & (dividend >= 0))] if has_column else panel
    invalid.to_csv(args.out_dir / "invalid_dividends.csv", index=False)
    previous_paid = int((previous["dividend"] > 0).sum()) if "dividend" in previous else 0
    disagreements = [f for f in meta.get("findings", []) if f["code"] == DISAGREES]
    summary_by_name = dividend_summary(panel) if has_column else pd.DataFrame()
    summary_by_name.to_csv(args.out_dir / "dividends_by_name.csv", index=False)

    summary = {
        "panel": panel_path.name,
        "rows": len(panel),
        "previous_rows": len(previous),
        "has_dividend_column": has_column,
        "changed_rows": len(changed),
        "invalid_dividends": len(invalid) if has_column else None,
        "previous_dividend_rows": previous_paid,
        "dividend_rows": int((dividend > 0).sum()),
        "names_with_dividends": int(panel.loc[dividend > 0, "kdcode"].nunique())
        if has_column
        else 0,
        "adjusted_close_disagreements": len(disagreements),
        "names_with_adjusted_close_disagreements": sorted(
            {str(f["kdcode"]) for f in disagreements}
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = (
        has_column
        and summary["changed_rows"] == 0
        and summary["invalid_dividends"] == 0
        and previous_paid == 0
        and not disagreements
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
