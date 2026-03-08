"""
Nightly orchestrator for the MCI-GRU paper trading pipeline.

Single entry point that runs the full sequence:
  1. Refresh LSEG data (append today's bars to master CSV)
  2. Track fills + compute returns for prior day's positions
  3. Run inference (score the full universe)
  4. Run portfolio decision (rank-drop gate, generate orders)
  5. Generate daily report + equity curve

The ordering matters:
  - refresh_data must run first so the CSV has today's bar
  - track runs second because it needs today's open price to record fills
    and yesterday's open to compute prior-day returns
  - infer runs third, scoring the universe with data through today
  - portfolio runs fourth, comparing today's ranks to prior ranks
  - report runs last, consuming all the outputs above

Usage:
    python paper_trade/scripts/run_nightly.py
    python paper_trade/scripts/run_nightly.py --skip-refresh
    python paper_trade/scripts/run_nightly.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "name": "Data Refresh",
        "script": "refresh_data.py",
        "phase": 2,
        "skippable": True,
    },
    {
        "name": "Execution Tracker",
        "script": "track.py",
        "phase": 5,
        "skippable": False,
    },
    {
        "name": "Model Inference",
        "script": "infer.py",
        "phase": 3,
        "skippable": False,
    },
    {
        "name": "Portfolio Decision",
        "script": "portfolio.py",
        "phase": 4,
        "skippable": False,
    },
    {
        "name": "Daily Report",
        "script": "report.py",
        "phase": 6,
        "skippable": False,
    },
]


def check_lseg_available() -> bool:
    """Check if the refinitiv.data library can be imported."""
    try:
        import refinitiv.data
        return True
    except ImportError:
        return False


def run_step(
    step: dict,
    python_exe: str,
    extra_args: list = None,
    dry_run: bool = False,
) -> dict:
    """Run a single pipeline step as a subprocess."""
    script_path = SCRIPTS_DIR / step["script"]
    if not script_path.exists():
        return {
            "name": step["name"],
            "status": "MISSING",
            "duration": 0,
            "error": f"Script not found: {script_path}",
        }

    cmd = [python_exe, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return {
            "name": step["name"],
            "status": "SKIPPED (dry run)",
            "duration": 0,
        }

    print(f"  Running: {' '.join(cmd)}")
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
                print(f"    {line}")

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print(f"  FAILED ({duration:.1f}s)")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"    ERROR: {line}")
            return {
                "name": step["name"],
                "status": "FAILED",
                "duration": duration,
                "returncode": result.returncode,
                "error": error_msg,
            }

        return {
            "name": step["name"],
            "status": "OK",
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        print(f"  TIMEOUT after {duration:.1f}s")
        return {
            "name": step["name"],
            "status": "TIMEOUT",
            "duration": duration,
            "error": "Exceeded 10 minute timeout",
        }
    except Exception as e:
        duration = time.time() - start
        print(f"  ERROR: {e}")
        return {
            "name": step["name"],
            "status": "ERROR",
            "duration": duration,
            "error": str(e),
        }


def save_manifest(results_dir: Path, state_dir: Path, results: list, start_time: str):
    """Save a run manifest recording what happened."""
    state_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_start": start_time,
        "run_end": datetime.now().isoformat(),
        "python": sys.executable,
        "steps": results,
        "all_ok": all(r["status"] == "OK" for r in results
                      if r["status"] not in ("SKIPPED", "SKIPPED (dry run)")),
    }

    manifest_path = state_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="MCI-GRU paper trading nightly pipeline.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip LSEG data refresh (use when data is already current)",
    )
    parser.add_argument(
        "--skip-track",
        action="store_true",
        help="Skip execution tracking (first run or no prior positions)",
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
    args = parser.parse_args()

    start_time = datetime.now()

    print()
    print("=" * 70)
    print("  MCI-GRU Paper Trading — Nightly Pipeline")
    print(f"  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    if not args.skip_refresh and not args.dry_run:
        if not check_lseg_available():
            print("  WARNING: refinitiv.data not found in current environment.")
            print("  Make sure you activated lseg_env before running.")
            print("  Use --skip-refresh if data is already up to date.")
            print()

    results_dir = PROJECT_ROOT / "paper_trade" / "results"
    state_dir = PROJECT_ROOT / "paper_trade" / "state"

    step_results = []

    for step in STEPS:
        step_name = step["name"]
        print(f"{'-' * 70}")
        print(f"  STEP {step['phase']}: {step_name}")
        print(f"{'-' * 70}")

        if step["script"] == "refresh_data.py" and args.skip_refresh:
            print(f"  SKIPPED (--skip-refresh)")
            step_results.append({
                "name": step_name,
                "status": "SKIPPED",
                "duration": 0,
            })
            print()
            continue

        if step["script"] == "track.py" and args.skip_track:
            print(f"  SKIPPED (--skip-track)")
            step_results.append({
                "name": step_name,
                "status": "SKIPPED",
                "duration": 0,
            })
            print()
            continue

        extra_args = []
        if step["script"] == "refresh_data.py" and args.dry_run:
            extra_args.append("--dry-run")

        result = run_step(step, args.python, extra_args, dry_run=args.dry_run)
        step_results.append(result)
        print()

        if result["status"] == "FAILED" and not step.get("skippable"):
            print(f"  ABORTING: {step_name} failed and is required.")
            break

    manifest = save_manifest(results_dir, state_dir, step_results, start_time.isoformat())

    elapsed = (datetime.now() - start_time).total_seconds()

    print("=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print()
    for r in step_results:
        status = r["status"]
        dur = r["duration"]
        icon = "OK" if status == "OK" else status
        print(f"  {r['name']:25s}  {icon:15s}  {dur:6.1f}s")
    print()
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  All steps OK:  {manifest['all_ok']}")
    print(f"  Manifest:      {state_dir / 'run_manifest.json'}")
    print("=" * 70)

    if not manifest["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
