"""Check an EODHD package's volume against the vendor responses it was built from (#347).

The #322 check found the r2 panel's volume split-adjusted twice. This re-runs it
on a rebuilt package, from the pull's own cache, and compares the panel with the
revision it replaces:

1. Every row's volume equals the vendor's, less the splits after the panel end
   that the pull recorded for the stretch it was read from, and any declared
   share conversion.
2. No PIT training row's volume exceeds the vendor's.
3. Volume does not step at a split: median volume over the 20 sessions from
   each split on, over the 20 before, for the panel and for the vendor.
4. The pooled z-score band over training rows (the #322 anomaly: sd about 40x
   the bottom-90% spread when volume was doubled up, about 2x when it is not).
5. Prices match the previous revision row for row; every difference is listed.

It exits 1 when a row's volume is unexplained (1), when a row has no cached
vendor row to explain it, when a training row exceeds the vendor's volume (2),
or when a price differs from the previous revision without a matching
``--allow-price-change``. Checks 3 and 4 are reported for reading.

Reads only files; makes no vendor calls::

    python -m scripts.data.check_eodhd_volume_basis \\
        --package-root <r3 package> --previous-root <r2 package> \\
        --cache-dir <the pull's --cache-dir> --start 2015-01-01 --end 2026-07-31 \\
        --out-dir <dir> [--allow-price-change KDCODE:BEFORE ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = ("2016-01-04", "2023-12-31")
PRICES = ["open", "high", "low", "close"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-price-change",
        action="append",
        default=[],
        metavar="KDCODE:BEFORE",
        help="a deliberate price change from the previous revision, on this name's rows "
        "before this date; repeatable",
    )
    return parser.parse_args(argv)


def _one(root: Path, pattern: str) -> Path:
    (path,) = sorted((root / "market").glob(pattern))
    return path


def _cached(cache_dir: Path, kind: str, symbol: str, path: str, params: dict) -> list:
    """The cached body for exactly this request, keyed as the pull's client keys it."""
    digest = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    file = cache_dir / kind / f"{symbol}.{digest[:16]}.json"
    body = json.loads(file.read_text(encoding="utf-8"))["body"] if file.exists() else []
    return body if isinstance(body, list) else []


def _ratio(text: str) -> float:
    new, old = (float(x) for x in str(text).split("/"))
    return new / old


def vendor_rows(symbols: dict, cache_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Vendor close and volume per (kdcode, dt), through the symbol each stretch used."""
    parts = []
    for kdcode, info in symbols.items():
        for segment in info.get("segments", []):
            symbol = segment.get("symbol")
            if not symbol:
                continue
            params = {"period": "d", "from": start, "to": end}
            eod = pd.DataFrame(_cached(cache_dir, "eod", symbol, f"eod/{symbol}", params))
            if eod.empty:
                continue
            lo, hi = segment.get("start") or start, segment.get("end") or end
            eod = eod[(eod["date"] >= max(lo, start)) & (eod["date"] <= min(hi, end))]
            parts.append(
                pd.DataFrame(
                    {
                        "kdcode": kdcode,
                        "dt": eod["date"].astype(str),
                        "symbol": symbol,
                        "vendor_close": eod["close"].astype(float),
                        "vendor_volume": eod["volume"].astype(float),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True).drop_duplicates(["kdcode", "dt"], keep="last")


def expected_divisor(panel: pd.DataFrame, symbols: dict) -> np.ndarray:
    """Per row, what the vendor's volume is divided by: later splits and share conversions.

    A later split belongs to the symbol one stretch was read from, so it applies
    only inside that stretch.
    """
    divisor = np.ones(len(panel))
    codes, dates = panel["kdcode"].to_numpy(), panel["dt"].to_numpy()
    for kdcode, info in symbols.items():
        for segment in info.get("segments", []):
            span = (dates >= (segment.get("start") or "")) & (
                dates <= (segment.get("end") or "9999")
            )
            for split_dt, ratio in segment.get("volume_later_splits", {}).items():
                divisor[(codes == kdcode) & span & (dates < split_dt)] *= ratio
    for kdcode, info in symbols.items():
        for applied in info.get("adjustments_applied", []):
            if applied.get("share_conversion") and applied.get("factor"):
                divisor[(codes == kdcode) & (dates < applied["date"])] *= applied["factor"]
    return divisor


def training_rows(panel: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Panel rows in the train window on days the name is a PIT member."""
    window = panel[(panel["dt"] >= TRAIN[0]) & (panel["dt"] <= TRAIN[1])]
    codes, dates = window["kdcode"].to_numpy(), window["dt"].to_numpy()
    member = np.zeros(len(window), dtype=bool)
    for row in pit.itertuples(index=False):
        valid_to = row.valid_to or "9999-12-31"
        member |= (codes == row.kdcode) & (dates >= row.valid_from) & (dates <= valid_to)
    return window[member]


def band(volume: pd.Series) -> dict[str, float]:
    """The #322 measures: width of the central 90% of pooled z, and sd over the bottom-90% span."""
    z = (volume - volume.mean()) / volume.std()
    lo, hi = z.quantile([0.05, 0.95])
    q0, q90 = volume.quantile([0.0, 0.9])
    return {
        "z_central_90_width": float(hi - lo),
        "sd_over_bottom_90_span": float(volume.std() / (q90 - q0)),
    }


def split_steps(
    panel: pd.DataFrame, vendor: pd.DataFrame, symbols: dict, cache_dir: Path, start: str, end: str
) -> pd.DataFrame:
    """Median volume 20 sessions from each split on over the 20 before, panel and vendor."""
    steps = []
    for kdcode, info in symbols.items():
        for segment in info.get("segments", []):
            symbol = segment.get("symbol")
            if not symbol:
                continue
            params = {"from": start, "to": end}
            for split in _cached(cache_dir, "splits", symbol, f"splits/{symbol}", params):
                dt = str(split["date"])
                g = panel[panel["kdcode"] == kdcode].sort_values("dt")
                v = vendor[vendor["kdcode"] == kdcode].sort_values("dt")
                before, after = (
                    g[g["dt"] < dt]["volume"].tail(20),
                    g[g["dt"] >= dt]["volume"].head(20),
                )
                if len(before) < 5 or len(after) < 5:
                    continue
                vb = v[v["dt"] < dt]["vendor_volume"].tail(20)
                va = v[v["dt"] >= dt]["vendor_volume"].head(20)
                steps.append(
                    {
                        "kdcode": kdcode,
                        "split_dt": dt,
                        "ratio": _ratio(split["split"]),
                        "panel_after_over_before": after.median() / before.median(),
                        "vendor_after_over_before": va.median() / vb.median(),
                    }
                )
    return pd.DataFrame(steps).drop_duplicates(["kdcode", "split_dt"])


def price_differences(
    panel: pd.DataFrame, previous: pd.DataFrame, allowed: list[str]
) -> pd.DataFrame:
    """Rows whose prices differ from the previous revision, or that only one revision has.

    ``allowed`` marks deliberate changes, ``KDCODE:BEFORE``: that name's rows before that date.
    """
    merged = panel.merge(
        previous, on=["kdcode", "dt"], how="outer", suffixes=("", "_prev"), indicator=True
    )
    differs = merged["_merge"] != "both"
    for column in PRICES:
        a, b = merged[column], merged[f"{column}_prev"]
        differs |= ~(np.isclose(a, b, rtol=1e-12, atol=0) | (a.isna() & b.isna()))
    out = merged.loc[differs, ["kdcode", "dt", "_merge", "close", "close_prev"]]
    expected = np.zeros(len(out), dtype=bool)
    for item in allowed:
        kdcode, before = item.rsplit(":", 1)
        expected |= (out["kdcode"] == kdcode).to_numpy() & (out["dt"] < before).to_numpy()
    expected &= (out["_merge"] == "both").to_numpy()
    return out.assign(close_ratio=out["close"] / out["close_prev"], allowed=expected)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root, prev_root = args.package_root, args.previous_root
    panel_path = _one(root, "*_eodhd_*[0-9].csv")
    panel = pd.read_csv(panel_path, dtype={"kdcode": str, "dt": str})
    previous = pd.read_csv(_one(prev_root, panel_path.name), dtype={"kdcode": str, "dt": str})
    symbols = json.loads(_one(root, "*_symbols.json").read_text(encoding="utf-8"))
    pit_path = next((root / "constituents").glob("*_pit_universe.csv"))
    pit = pd.read_csv(pit_path, dtype=str, keep_default_na=False)

    vendor = vendor_rows(symbols, args.cache_dir, args.start, args.end)
    merged = panel.merge(vendor, on=["kdcode", "dt"], how="left")
    merged["divisor"] = expected_divisor(merged, symbols)
    merged["expected_volume"] = merged["vendor_volume"] / merged["divisor"]
    known = merged["volume"].notna() & merged["expected_volume"].notna()
    mismatch = known & ~np.isclose(merged["volume"], merged["expected_volume"], rtol=1e-9, atol=0.5)
    merged.loc[mismatch].to_csv(args.out_dir / "volume_mismatches.csv", index=False)

    train = training_rows(merged, pit)
    # A reverse split after the panel end, or a conversion to fewer shares, raises volume
    # legitimately; check 1 covers those rows exactly.
    above_vendor = train[
        (train["volume"] > train["vendor_volume"] * (1 + 1e-9)) & (train["divisor"] >= 1)
    ]
    above_vendor.to_csv(args.out_dir / "training_rows_above_vendor.csv", index=False)
    top = train.nlargest(20, "volume")[["kdcode", "dt", "close", "volume", "vendor_volume"]]
    top.to_csv(args.out_dir / "top20_training_volume.csv", index=False)
    prev_train = training_rows(previous, pit)

    steps = split_steps(panel, vendor, symbols, args.cache_dir, args.start, args.end)
    steps.to_csv(args.out_dir / "split_volume_steps.csv", index=False)
    prices = price_differences(panel, previous, args.allow_price_change)
    prices.to_csv(args.out_dir / "price_differences.csv", index=False)
    volume_ratio = panel.merge(previous, on=["kdcode", "dt"], suffixes=("", "_prev"))
    volume_ratio = volume_ratio.assign(ratio=volume_ratio["volume_prev"] / volume_ratio["volume"])
    by_name = volume_ratio.groupby("kdcode")["ratio"].agg(["min", "max"])
    by_name[(by_name["min"] < 0.999) | (by_name["max"] > 1.001)].to_csv(
        args.out_dir / "volume_ratio_previous_over_new_by_name.csv"
    )

    summary = {
        "panel": panel_path.name,
        "rows": len(panel),
        "rows_without_vendor_volume": int((~known & merged["volume"].notna()).sum()),
        "volume_mismatches": int(mismatch.sum()),
        "training_rows": len(train),
        "training_rows_above_vendor_volume": len(above_vendor),
        "training_band": band(train["volume"]),
        "previous_training_band": band(prev_train["volume"]),
        "max_training_volume": float(train["volume"].max()),
        "previous_max_training_volume": float(prev_train["volume"].max()),
        "split_steps": len(steps),
        "split_steps_panel_outside_0p5_2": int(
            (~steps["panel_after_over_before"].between(0.5, 2.0)).sum()
        )
        if len(steps)
        else 0,
        "price_difference_rows": len(prices),
        "unexpected_price_difference_rows": int((~prices["allowed"]).sum()),
        "price_differences_by_name": prices.groupby("kdcode")["dt"]
        .agg(["min", "max", "size"])
        .reset_index()
        .to_dict("records"),
        "names_whose_volume_changed": len(
            by_name[(by_name["min"] < 0.999) | (by_name["max"] > 1.001)]
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = all(
        summary[key] == 0
        for key in (
            "rows_without_vendor_volume",
            "volume_mismatches",
            "training_rows_above_vendor_volume",
            "unexpected_price_difference_rows",
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
