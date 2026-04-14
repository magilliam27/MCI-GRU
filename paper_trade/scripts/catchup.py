"""
Catch-up script for the MCI-GRU paper trading pipeline.

When you miss one or more trading days, this script:
  1. Refreshes LSEG data once (pulls all missing bars in one shot)
  2. Identifies which trading days need processing since the last run
  3. Runs track -> infer -> portfolio -> report for each missed day, in order

State files are updated sequentially so each day's positions flow
correctly into the next day's return computation.

Usage:
    python paper_trade/scripts/catchup.py
    python paper_trade/scripts/catchup.py --skip-refresh
    python paper_trade/scripts/catchup.py --dry-run
    python paper_trade/scripts/catchup.py --model-dir paper_trade/Model/seed7_w_regime
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "market" / "sp500_2019_universe_data_through_2026.csv"
STATE_DIR = PROJECT_ROOT / "paper_trade" / "state"
RESULTS_DIR = PROJECT_ROOT / "paper_trade" / "results"
DEFAULT_MODEL_DIR = "paper_trade/Model/seed7_w_regime"

DAY_STEPS = [
    {"name": "Execution Tracker", "script": "track.py"},
    {"name": "Model Inference", "script": "infer.py"},
    {"name": "Portfolio Decision", "script": "portfolio.py"},
    {"name": "Daily Report", "script": "report.py"},
]


def get_last_processed_date() -> str | None:
    """Read the last decision date from current_holdings.json."""
    path = STATE_DIR / "current_holdings.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("date")


def get_trading_dates_after(csv_path: Path, after_date: str) -> list[str]:
    """Return sorted list of trading dates in the CSV strictly after after_date."""
    df = pd.read_csv(str(csv_path), usecols=["dt"])
    dates = sorted(df["dt"].unique())
    return [d for d in dates if d > after_date]


def run_script(
    script_name: str, python_exe: str, extra_args: list = None, dry_run: bool = False
) -> dict:
    """Run a pipeline script as a subprocess. Returns result dict."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [python_exe, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    if dry_run:
        print(f"    [DRY RUN] {' '.join(cmd)}")
        return {"name": script_name, "status": "SKIPPED", "duration": 0}

    print(f"    Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        duration = time.time() - start

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"      {line}")

        if result.returncode != 0:
            print(f"    FAILED ({duration:.1f}s)")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"      ERROR: {line}")
            return {
                "name": script_name,
                "status": "FAILED",
                "duration": duration,
                "error": result.stderr.strip() if result.stderr else "",
            }

        return {"name": script_name, "status": "OK", "duration": duration}

    except subprocess.TimeoutExpired:
        return {"name": script_name, "status": "TIMEOUT", "duration": time.time() - start}
    except Exception as e:
        return {
            "name": script_name,
            "status": "ERROR",
            "duration": time.time() - start,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Catch up on missed paper-trading days.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip LSEG data refresh (if CSV is already up to date)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing anything",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Model directory passed to infer.py (default: seed7_w_regime)",
    )
    args = parser.parse_args()

    start_time = datetime.now()

    print()
    print("=" * 70)
    print("  MCI-GRU Paper Trading - Catch-Up")
    print(f"  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    last_date = get_last_processed_date()
    if last_date is None:
        print("\n  ERROR: No current_holdings.json found - nothing to catch up from.")
        print("  Run the normal pipeline first: python paper_trade/scripts/run_nightly.py")
        sys.exit(1)

    print(f"\n  Last processed date: {last_date}")

    # --- Step 1: Data refresh ---
    if not args.skip_refresh:
        print(f"\n{'-' * 70}")
        print("  STEP 1: Data Refresh (one-time)")
        print(f"{'-' * 70}")
        result = run_script("refresh_data.py", args.python, dry_run=args.dry_run)
        if result["status"] == "FAILED":
            print("\n  Data refresh failed. Fix the issue or use --skip-refresh")
            print("  if the CSV is already up to date.")
            sys.exit(1)
    else:
        print("\n  Skipping data refresh (--skip-refresh)")

    # --- Identify missed days ---
    if not DEFAULT_CSV.exists():
        print(f"\n  ERROR: Master CSV not found: {DEFAULT_CSV}")
        sys.exit(1)

    missed_days = get_trading_dates_after(DEFAULT_CSV, last_date)

    if not missed_days:
        print("\n  No missed trading days found. You're already up to date!")
        print("=" * 70)
        return

    print(f"\n  Found {len(missed_days)} missed trading day(s): {', '.join(missed_days)}")

    # --- Step 2: Process each day ---
    all_results = []
    for i, trade_date in enumerate(missed_days, 1):
        print(f"\n{'-' * 70}")
        print(f"  DAY {i}/{len(missed_days)}: {trade_date}")
        print(f"{'-' * 70}")

        day_ok = True
        for step in DAY_STEPS:
            extra = ["--date", trade_date]
            if step["script"] == "infer.py":
                extra.extend(["--model-dir", args.model_dir])
            result = run_script(
                step["script"],
                args.python,
                extra_args=extra,
                dry_run=args.dry_run,
            )
            result["date"] = trade_date
            result["step"] = step["name"]
            all_results.append(result)

            if result["status"] == "FAILED":
                print(f"\n  ABORTING day {trade_date}: {step['name']} failed.")
                day_ok = False
                break

        if not day_ok:
            print("  Stopping catch-up - fix the error and re-run.")
            break

    # --- Summary ---
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'=' * 70}")
    print("  CATCH-UP SUMMARY")
    print(f"{'=' * 70}")
    print()

    current_date = None
    for r in all_results:
        if r.get("date") != current_date:
            current_date = r.get("date")
            print(f"  {current_date}:")
        status = r["status"]
        dur = r["duration"]
        print(f"    {r.get('step', r['name']):25s}  {status:10s}  {dur:6.1f}s")

    all_ok = all(r["status"] in ("OK", "SKIPPED") for r in all_results)
    days_processed = len(set(r["date"] for r in all_results if r["status"] in ("OK", "SKIPPED")))

    print()
    print(f"  Days processed: {days_processed}/{len(missed_days)}")
    print(f"  Total elapsed:  {elapsed:.1f}s")
    print(f"  All OK:         {all_ok}")
    print(f"{'=' * 70}")

    # Update the manifest
    manifest = {
        "run_start": start_time.isoformat(),
        "run_end": datetime.now().isoformat(),
        "python": args.python,
        "mode": "catchup",
        "days_processed": days_processed,
        "days_total": len(missed_days),
        "steps": all_results,
        "all_ok": all_ok,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
