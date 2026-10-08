"""Check that delisted names stop at their last real close (first-run precondition).

The first run declares no cessation event file (``data.pit_cessation_events_csv:
null``). That is safe only while every delisted name has no finite close after
its last real trading session: then ``price_observed`` already removes the
post-delisting sessions, and a declared cessation would change no prediction,
label or metric. A vendor tail of carried or frozen closes would break that, and
the #223 frozen-price rule does not catch it, because it flags a stock only when
its whole history is constant. See docs/DEFAULT_EXPERIMENT_RECIPE.md.

Read-only. By default it reads the market CSV of the data config the recipe
selects (its ``data=<group>`` line), inspects the names carrying an LSEG delisted
suffix (``^``), prints one line per name, and exits 1 when a name ends in a run
of identical closes or zero volume at least ``--min-stale`` sessions long, or in
one row that repeats the previous close with zero volume: a carried row, such as
a vendor's entry for the delisting day itself. The EODHD panel keeps the LSEG
identifiers, so the suffix still marks its delisted names.

    python scripts/check_delisted_tails.py
    python scripts/check_delisted_tails.py --market-csv path/to/panel.csv --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "docs" / "DEFAULT_EXPERIMENT_RECIPE.md"
# The recipe's override block, and `data=<group>` on its own line inside it.
OVERRIDE_BLOCK = re.compile(r"^## Hydra Overrides\n+```text\n(.*?)^```", re.M | re.S)
DATA_SELECTOR = re.compile(r"^data=([A-Za-z0-9_]+)$", re.M)
DELISTED_MARKER = "^"


def _trailing_run(values: np.ndarray, target: float) -> int:
    """Length of the run of ``target`` at the end of ``values``."""
    run = 0
    for value in values[::-1]:
        if value != target:
            break
        run += 1
    return run


def tail_report(
    frame: pd.DataFrame, kdcodes: list[str], min_stale: int = 2
) -> list[dict[str, Any]]:
    """One record per name describing how its price history ends.

    ``trailing_repeat_sessions`` counts sessions after the first of a final run
    of identical finite closes, so a clean history scores 0. ``stale`` is true
    when that count or the final run of zero-volume rows reaches ``min_stale``,
    or when the last row both repeats the previous close and has zero volume.
    """
    records = []
    for kdcode in kdcodes:
        rows = frame.loc[frame["kdcode"] == kdcode].sort_values("dt")
        close = pd.to_numeric(rows["close"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(close)
        if not finite.any():
            records.append(
                {
                    "kdcode": kdcode,
                    "rows": int(len(rows)),
                    "last_finite_close_dt": None,
                    "trailing_repeat_sessions": 0,
                    "trailing_zero_volume_sessions": 0,
                    "stale": False,
                }
            )
            continue
        closes = close[finite]
        dates = rows["dt"].astype(str).to_numpy()[finite]
        last = closes[-1]
        repeat = _trailing_run(closes, last) - 1
        if "volume" in rows.columns:
            volume = pd.to_numeric(rows["volume"], errors="coerce").to_numpy(dtype=float)[finite]
            zero_volume = _trailing_run(volume, 0.0)
        else:
            zero_volume = 0
        records.append(
            {
                "kdcode": kdcode,
                "rows": int(len(rows)),
                "last_finite_close_dt": str(dates[-1]),
                "trailing_repeat_sessions": int(repeat),
                "trailing_zero_volume_sessions": int(zero_volume),
                "stale": bool(
                    repeat >= min_stale
                    or zero_volume >= min_stale
                    or (repeat >= 1 and zero_volume >= 1)
                ),
            }
        )
    return records


def recipe_data_config(recipe: Path = RECIPE) -> Path:
    """The data config the recipe selects, so the check reads the panel the run reads."""
    block = OVERRIDE_BLOCK.search(recipe.read_text(encoding="utf-8"))
    match = DATA_SELECTOR.search(block.group(1)) if block else None
    if match is None:
        raise SystemExit(f"{recipe}'s override block selects no data config; pass --market-csv")
    return PROJECT_ROOT / "configs" / "data" / f"{match.group(1)}.yaml"


def _default_market_csv() -> Path:
    return PROJECT_ROOT / str(OmegaConf.load(recipe_data_config()).filename)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market-csv", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="inspect every name, not only ^ names")
    parser.add_argument("--min-stale", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="print the records as JSON")
    args = parser.parse_args(argv)

    path = args.market_csv or _default_market_csv()
    frame = pd.read_csv(path, usecols=lambda c: c in {"kdcode", "dt", "close", "volume"})
    names = sorted(frame["kdcode"].astype(str).unique())
    if not args.all:
        names = [name for name in names if DELISTED_MARKER in name]
    records = tail_report(frame, names, args.min_stale)

    if args.json:
        print(json.dumps({"market_csv": str(path), "records": records}, indent=2))
    else:
        print(f"market csv: {path}")
        for record in records:
            print(
                f"{record['kdcode']:<16} last close {record['last_finite_close_dt']}  "
                f"repeated closes after it: {record['trailing_repeat_sessions']}  "
                f"zero-volume tail: {record['trailing_zero_volume_sessions']}  "
                f"{'STALE' if record['stale'] else 'clean'}"
            )
    return 1 if any(record["stale"] for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
