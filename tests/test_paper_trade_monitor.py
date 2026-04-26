import json
from pathlib import Path

import pandas as pd

from paper_trade.scripts.monitor import run_monitor
from paper_trade.scripts.report import build_markdown_report


def test_monitor_writes_feature_drift_outputs(tmp_path: Path):
    model_dir = tmp_path / "model"
    results_dir = tmp_path / "results"
    day_dir = results_dir / "2026-01-05"
    model_dir.mkdir()
    day_dir.mkdir(parents=True)

    reference = {
        "features": {
            "x": {"bins": [-2, -1, 0, 1, 2], "counts": [10, 10, 10, 10]},
            "y": {"bins": [-2, -1, 0, 1, 2], "counts": [10, 10, 10, 10]},
        }
    }
    (model_dir / "feature_reference.json").write_text(json.dumps(reference), encoding="utf-8")
    (model_dir / "run_metadata.json").write_text(
        json.dumps({"feature_cols": ["x", "y"], "feature_reference_path": "feature_reference.json"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "kdcode": ["A", "B", "C", "D"],
            "dt": ["2026-01-05"] * 4,
            "x": [1.5, 1.6, 1.7, 1.8],
            "y": [-1.5, -0.5, 0.5, 1.5],
        }
    ).to_csv(day_dir / "normalized_features.csv", index=False)

    summary = run_monitor(model_dir, results_dir, "2026-01-05")

    assert summary["overall_status"] == "ALERT"
    assert (day_dir / "feature_drift.csv").exists()
    assert (day_dir / "feature_drift.json").exists()


def test_report_includes_feature_drift_section():
    drift = {
        "overall_status": "WARN",
        "warn_features": 2,
        "alert_features": 0,
        "top_features": [{"feature": "x", "psi": 0.12, "ks": 0.03, "status": "WARN"}],
    }
    report = build_markdown_report(
        date="2026-01-05",
        perf_df=pd.DataFrame(),
        target_portfolio=pd.DataFrame(),
        orders=pd.DataFrame(),
        holdings=pd.DataFrame(),
        daily_return=pd.DataFrame(),
        rolling={"num_days": 0, "rolling_vol_ann": float("nan"), "sharpe_proxy": float("nan"), "max_drawdown": 0.0, "win_rate": float("nan"), "avg_daily_return": float("nan")},
        state_dir=Path("."),
        drift_summary=drift,
    )

    assert "## Feature Drift" in report
    assert "WARN" in report
    assert "x" in report
