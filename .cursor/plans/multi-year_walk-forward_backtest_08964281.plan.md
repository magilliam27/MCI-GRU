---
name: Multi-Year Walk-Forward Backtest
overview: Build a script that merges per-year prediction CSVs into one continuous folder and runs a single unified backtest across 2022+2023+2025 using the 2019 universe superset, producing one seamless equity curve.
todos:
  - id: create-script
    content: Create multiyear_backtest.py with CLI argument parsing (--prediction_dirs, --data_file, --test_start, --test_end, plus all standard backtest flags)
    status: pending
  - id: merge-predictions
    content: "Implement prediction merge: copy/symlink all date CSVs from multiple source dirs into one temp folder, validate no date collisions"
    status: pending
  - id: run-unified-backtest
    content: Import and call existing backtest functions from tests/backtest_sp500.py on the merged predictions folder with the 2019 universe superset
    status: pending
  - id: handle-gap
    content: "Detect and handle the 2024 date gap: log a warning, annotate on equity curve, ensure portfolio value carries through unchanged"
    status: pending
  - id: per-year-metrics
    content: Slice the unified daily returns by calendar year and print a per-year breakdown table (ARR, ASR, MDD, IR per segment)
    status: pending
  - id: output-artifacts
    content: Generate equity curve plot (with year-boundary markers and gap annotation), daily returns CSV, and summary text file
    status: pending
isProject: false
---

# Multi-Year Walk-Forward Backtest (Option A)

## How It Works

```mermaid
flowchart LR
  subgraph predictions [Per-Year Prediction Sources]
    Y2022["2022 predictions\n(Model trained 2016-2020)"]
    Y2023["2023 predictions\n(Model trained 2017-2021)"]
    Y2025["2025 predictions\n(Model trained 2019-2023)"]
  end
  subgraph merge [Merge Step]
    Merged["merged_predictions/\n~751 CSV files\n(2022-01-03 to 2025-12-31)"]
  end
  subgraph backtest [Single Unified Backtest]
    BT["backtest_sp500.py\nusing sp500_2019_universe_data.csv\ntest_start=2022-01-01\ntest_end=2025-12-31"]
  end
  Y2022 --> Merged
  Y2023 --> Merged
  Y2025 --> Merged
  Merged --> BT
```



The portfolio is **continuous** -- no liquidation between years. When the underlying model changes (e.g., Jan 2 2023), it's just another rebalance day. The rank-drop gate smooths the transition naturally.

## Prediction Sources

- **2022**: `seed_results/2022/seed7/averaged_predictions/` (251 files)
- **2023**: `seed_results/2023/2023_averaged_predictions/` (250 files)
- **2025**: user-specified seed folder, e.g. `seed_results/2025/seed7/averaged_predictions/` (250 files)
- **2024**: skipped for now (gap in equity curve is acceptable; can be added later)

Since dates don't overlap, merging is just copying/symlinking all CSVs into one flat folder.

## Universe File

Use `data/raw/market/sp500_2019_universe_data.csv` for the entire backtest. This is the superset -- it contains all stocks from earlier universes plus additions through 2019. Stocks that aren't scored by a given year's model simply won't appear in that year's prediction CSVs, so the backtest engine naturally restricts the tradeable set per year with no extra logic.

## What to Build

A single Python script (`multiyear_backtest.py`) at the repo root that:

1. **Accepts CLI arguments** for:
  - A list of `--prediction_dirs` (one per year, in chronological order)
  - `--data_file` (universe CSV, default: 2019 superset)
  - `--test_start` / `--test_end` (full date range)
  - All existing backtest flags (`--top_k`, `--enable_rank_drop_gate`, `--min_rank_drop`, `--transaction_costs`, `--spread`, `--auto_save`, `--plot`)
2. **Merges predictions** by creating a temporary folder and symlinking (or copying) all date CSVs from each source folder into it. Validates no date collisions.
3. **Calls the existing backtest** by importing `load_stock_data`, `calculate_forward_returns`, `load_predictions`, `simulate_trading_strategy` from `tests/backtest_sp500.py` -- no engine duplication.
4. **Handles the 2024 gap** gracefully: the backtest will simply have no predictions for 2024 dates, so no trades happen and the portfolio sits in cash (or we can log a warning and skip those dates). A cleaner option: detect the gap and annotate it on the equity curve plot.
5. **Outputs** the same artifacts as the regular backtest (equity curve, daily returns CSV, summary metrics) but across the full multi-year window, plus a per-year breakdown table showing ARR/ASR/MDD for each segment.

## Key Design Decisions

- **No code changes to `backtest_sp500.py`** -- we only import from it
- The merge folder is created under the output directory (or a temp dir), not polluting the source prediction folders
- The script validates that prediction date ranges don't overlap before merging
- Per-year metrics are computed by slicing the unified daily returns by calendar year

