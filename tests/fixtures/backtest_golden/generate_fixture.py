"""Generate deterministic backtest golden input fixtures (WS-N step 2).

Run from repo root:
  python tests/fixtures/backtest_golden/generate_fixture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_ROOT = Path(__file__).resolve().parent
RUN_DIR = FIXTURE_ROOT / "run"
PREDICTIONS_DIR = RUN_DIR / "averaged_predictions"

# Shared with golden CLI invocations in tests/test_backtest_engine_golden.py
STOCK_DATA_START = "2025-01-02"
STOCK_DATA_END = "2025-05-15"
TEST_START = "2025-02-03"
TEST_END = "2025-04-30"
TOP_K = 5
LABEL_T = 5
RNG_SEED = 1729

STOCKS = [f"STK{i:02d}" for i in range(1, 9)]


def _business_days(start: str, end: str) -> list[str]:
    return pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()


def write_stock_data(path: Path) -> None:
    rng = np.random.default_rng(RNG_SEED)
    dates = _business_days(STOCK_DATA_START, STOCK_DATA_END)
    rows: list[dict[str, object]] = []
    prev_close = {stock: 100.0 + 2.0 * idx for idx, stock in enumerate(STOCKS)}

    for date in dates:
        for stock in STOCKS:
            drift = float(rng.normal(0.0005, 0.004))
            open_price = prev_close[stock] * (1.0 + float(rng.normal(0.001, 0.002)))
            close_price = open_price * (1.0 + drift)
            high = max(open_price, close_price) * (1.0 + abs(float(rng.normal(0.0, 0.001))))
            low = min(open_price, close_price) * (1.0 - abs(float(rng.normal(0.0, 0.001))))
            volume = int(rng.integers(500_000, 2_000_000))
            rows.append(
                {
                    "kdcode": stock,
                    "dt": date,
                    "open": round(open_price, 6),
                    "high": round(high, 6),
                    "low": round(low, 6),
                    "close": round(close_price, 6),
                    "volume": volume,
                }
            )
            prev_close[stock] = close_price

    pd.DataFrame(rows).to_csv(path, index=False)


def write_predictions(path: Path) -> list[str]:
    rng = np.random.default_rng(RNG_SEED + 1)
    pred_dates = _business_days(TEST_START, TEST_END)
    path.mkdir(parents=True, exist_ok=True)

    for day_index, pred_date in enumerate(pred_dates):
        day_rows: list[dict[str, object]] = []
        for stock_index, stock in enumerate(STOCKS):
            # Rotating cross-section so top-k membership shifts every few days.
            phase = 2.0 * np.pi * (day_index / 10.0 + stock_index / len(STOCKS))
            score = float(np.sin(phase) + rng.normal(0.0, 0.05))
            day_rows.append(
                {
                    "kdcode": stock,
                    "dt": pred_date,
                    "score": round(score, 8),
                }
            )
        pd.DataFrame(day_rows).to_csv(path / f"{pred_date}.csv", index=False)

    return pred_dates


def write_pit_universe(path: Path) -> None:
    """STK07 leaves mid-window; STK08 joins mid-window."""
    rows = [
        {"kdcode": stock, "valid_from": TEST_START, "valid_to": TEST_END}
        for stock in STOCKS
        if stock not in {"STK07", "STK08"}
    ]
    rows.append({"kdcode": "STK07", "valid_from": TEST_START, "valid_to": "2025-03-14"})
    rows.append({"kdcode": "STK08", "valid_from": "2025-03-17", "valid_to": TEST_END})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=FIXTURE_ROOT,
        help="Root directory for generated fixture files",
    )
    args = parser.parse_args()
    root = args.fixture_root.resolve()
    run_dir = root / "run"
    predictions_dir = run_dir / "averaged_predictions"

    run_dir.mkdir(parents=True, exist_ok=True)
    write_stock_data(root / "stock_data.csv")
    pred_dates = write_predictions(predictions_dir)
    write_pit_universe(root / "pit_universe.csv")

    print(f"Wrote stock_data.csv ({STOCK_DATA_START}..{STOCK_DATA_END}, {len(STOCKS)} stocks)")
    print(f"Wrote {len(pred_dates)} prediction CSVs in {predictions_dir}")
    print("Wrote pit_universe.csv (STK07 leaves 2025-03-14, STK08 joins 2025-03-17)")


if __name__ == "__main__":
    main()
