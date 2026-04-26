"""
Feature drift monitor for the paper trading pipeline.

Compares the latest normalized inference features against the train-window
feature reference saved with the frozen model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mci_gru.evaluation.drift import compute_feature_drift, summarize_drift  # noqa: E402

DEFAULT_MODEL_DIR = "paper_trade/Model/Seed73_trained_to_2062026"
DEFAULT_RESULTS_DIR = "paper_trade/results"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_monitor_date(results_dir: Path, requested_date: str | None = None) -> str:
    if requested_date:
        return requested_date
    dated_dirs = sorted(
        [
            d
            for d in results_dir.iterdir()
            if d.is_dir() and (d / "normalized_features.csv").exists()
        ],
        key=lambda d: d.name,
    )
    if not dated_dirs:
        raise FileNotFoundError("No normalized_features.csv files found. Run infer.py first.")
    return dated_dirs[-1].name


def _feature_reference_path(model_dir: Path, metadata: dict) -> Path:
    rel = metadata.get("feature_reference_path", "feature_reference.json")
    path = Path(rel)
    if not path.is_absolute():
        path = model_dir / path
    return path


def run_monitor(model_dir: Path, results_dir: Path, date: str | None = None) -> dict:
    """Run drift monitoring and write CSV/JSON outputs for one paper-trade date."""
    monitor_date = find_monitor_date(results_dir, date)
    day_dir = results_dir / monitor_date
    features_path = day_dir / "normalized_features.csv"
    metadata = load_json(model_dir / "run_metadata.json")
    reference_path = _feature_reference_path(model_dir, metadata)

    if not reference_path.exists():
        rows = []
        summary = {
            "date": monitor_date,
            "overall_status": "NOT_AVAILABLE",
            "warn_features": 0,
            "alert_features": 0,
            "features_evaluated": 0,
            "reason": f"Feature reference not found: {reference_path}",
            "top_features": [],
        }
    else:
        reference = load_json(reference_path)
        feature_cols = metadata["feature_cols"]
        df = pd.read_csv(features_path)
        observed = df[feature_cols].to_numpy(dtype=float)
        rows = compute_feature_drift(observed, feature_cols, reference)
        summary = summarize_drift(rows)
        summary["date"] = monitor_date
        summary["top_features"] = sorted(
            [
                row
                for row in rows
                if row.get("status") in {"OK", "WARN", "ALERT"}
            ],
            key=lambda row: row.get("psi", 0.0),
            reverse=True,
        )[:10]

    day_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(day_dir / "feature_drift.csv", index=False)
    with (day_dir / "feature_drift.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def main():
    parser = argparse.ArgumentParser(description="MCI-GRU feature drift monitor.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir
    results_dir = PROJECT_ROOT / args.results_dir
    summary = run_monitor(model_dir, results_dir, args.date)
    print("=" * 70)
    print("  MCI-GRU Feature Drift Monitor")
    print("=" * 70)
    print(f"  Date:   {summary.get('date')}")
    print(f"  Status: {summary.get('overall_status')}")
    print(f"  Warn:   {summary.get('warn_features', 0)}")
    print(f"  Alert:  {summary.get('alert_features', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
