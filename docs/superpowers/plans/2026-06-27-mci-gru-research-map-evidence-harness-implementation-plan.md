# MCI-GRU Evidence Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first June 21 research-map wave: run manifests, trial ledgers, saved-prediction selection audits, PIT availability/tradability/staleness reports, and execution/capacity replay without retraining.

**Architecture:** Keep this as an additive evidence harness around existing artifacts. Add small reusable modules under `mci_gru/evaluation/` and `mci_gru/data/`, thin CLI scripts under `scripts/`, and focused pytest coverage with tiny synthetic data. Do not modify the frozen default recipe, PIT masked-panel semantics, existing backtest timing, or paper-trade frozen inference behavior.

**Tech Stack:** Python 3.10, pandas, numpy, scipy, pytest, existing `mci_gru.evaluation.statistics`, existing `mci_gru.evaluation.portfolio`, existing `mci_gru.evaluation.prediction_report`, existing `tests/backtest_sp500_daily.py`.

---

## Source Plan

Primary research map:

- `docs/research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md`

Relevant June 21 wave ordering:

1. Wave 0: docs-only and saved-artifact governance.
2. Wave 1: no-retraining replay and diagnostics.
3. Wave 2+: additive data/features and model changes only after the evidence layer exists.

This plan intentionally implements Wave 0 plus the first Wave 1 replay surfaces. It does not train models, launch Colab, change losses, change graphs, change feature defaults, or mutate paper-trade inference.

## File Structure

- Create `mci_gru/evaluation/run_bundle.py`: run-folder manifest and validation helpers.
- Create `mci_gru/evaluation/trial_ledger.py`: flatten run metadata and write trial ledger CSV/JSONL.
- Create `mci_gru/evaluation/selection_audit.py`: saved-prediction IC, rank IC, top-k, p-value, bootstrap, and multiple-testing audit.
- Create `mci_gru/evaluation/capacity.py`: lagged ADV/volatility, participation, and capacity-breach calculations.
- Create `mci_gru/data/pit_audit.py`: PIT availability, tradability, and staleness report helpers.
- Create `scripts/build_run_bundle_manifest.py`: CLI for manifest/validation output beside an existing run.
- Create `scripts/build_trial_ledger.py`: CLI for collecting existing run folders into a ledger.
- Create `scripts/run_saved_prediction_selection_audit.py`: CLI for saved-prediction model-selection audit.
- Create `scripts/write_pit_availability_report.py`: CLI for PIT availability/tradability/staleness reports.
- Create `scripts/run_saved_prediction_capacity_replay.py`: CLI for no-retraining capacity replay.
- Create `docs/evaluation/EVIDENCE_HARNESS.md`: operator-facing artifact contract and command cookbook.
- Create tests:
  - `tests/test_run_bundle_manifest.py`
  - `tests/test_trial_ledger.py`
  - `tests/test_saved_prediction_selection_audit.py`
  - `tests/test_pit_availability_report.py`
  - `tests/test_capacity_replay.py`

## Guardrails

- Preserve `data.pit_universe_mode=masked_panel`; do not replace it with complete-stock or stayer filtering.
- Prediction date `T` may only use information available by `T` close.
- Backtest/capacity replay must enter from the next valid open and label timing explicitly.
- Use lagged rolling dollar ADV/volatility for capacity gates; realized `T+1` volume may appear only as ex-post diagnostics.
- Include failed, skipped, and ugly variants in trial ledgers.
- Filesystem artifacts remain source of truth; MLflow is additive.
- Keep all outputs additive. Do not overwrite `training_summary.json`, `run_metadata.json`, `evaluation_summary.json`, saved predictions, or backtest outputs.

---

### Task 1: Run-Bundle Manifest And Validation

**Files:**
- Create: `mci_gru/evaluation/run_bundle.py`
- Create: `scripts/build_run_bundle_manifest.py`
- Create: `tests/test_run_bundle_manifest.py`
- Later doc update: `docs/evaluation/EVIDENCE_HARNESS.md`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_bundle_manifest.py`:

```python
import json
from pathlib import Path

from mci_gru.evaluation.run_bundle import build_run_manifest, validate_run_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_run_manifest_hashes_core_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    predictions_dir = run_dir / "averaged_predictions"
    predictions_dir.mkdir(parents=True)
    _write_json(run_dir / "run_metadata.json", {"config": {"seed": 314159}})
    _write_json(run_dir / "training_summary.json", {"mean_best_val_ic": 0.01})
    _write_json(run_dir / "evaluation_summary.json", {"avg_rank_ic": 0.02})
    (predictions_dir / "2024-01-02.csv").write_text(
        "dt,kdcode,score\n2024-01-02,AAA,0.1\n",
        encoding="utf-8",
    )

    manifest = build_run_manifest(
        run_dir,
        selection_rule="max validation rank IC",
        sibling_trial_ids=["trial-a", "trial-b"],
    )

    assert manifest["schema_version"] == 1
    assert manifest["run_dir"] == str(run_dir.resolve())
    assert manifest["selection_rule"] == "max validation rank IC"
    assert manifest["sibling_trial_ids"] == ["trial-a", "trial-b"]
    assert manifest["artifacts"]["run_metadata.json"]["exists"] is True
    assert manifest["artifacts"]["training_summary.json"]["sha256"]
    assert manifest["prediction_artifact"]["file_count"] == 1
    assert manifest["prediction_artifact"]["sha256"]


def test_validate_run_bundle_reports_missing_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    validation = validate_run_bundle(run_dir)

    assert validation["schema_version"] == 1
    assert validation["status"] == "FAILED"
    assert "run_metadata.json" in validation["missing_artifacts"]
    assert "averaged_predictions" in validation["missing_artifacts"]
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_bundle_manifest.py -v --basetemp .tmp_pytest\pytest
```

Expected: import failure for `mci_gru.evaluation.run_bundle`.

- [ ] **Step 3: Implement `mci_gru/evaluation/run_bundle.py`**

Create `mci_gru/evaluation/run_bundle.py` with these public functions:

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE_ARTIFACTS = [
    "run_metadata.json",
    "training_summary.json",
    "evaluation_summary.json",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path, pattern: str = "*.csv") -> dict[str, Any]:
    root = Path(path)
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    digest = hashlib.sha256()
    entries = []
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        file_hash = sha256_file(file_path)
        entries.append({"path": rel, "sha256": file_hash, "size_bytes": file_path.stat().st_size})
        digest.update(rel.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return {"path": str(root.resolve()), "file_count": len(files), "sha256": digest.hexdigest(), "files": entries}


def describe_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def build_run_manifest(
    run_dir: str | Path,
    *,
    selection_rule: str | None = None,
    sibling_trial_ids: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    predictions_dir = root / "averaged_predictions"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root.resolve()),
        "selection_rule": selection_rule,
        "sibling_trial_ids": sibling_trial_ids or [],
        "artifacts": {name: describe_artifact(root / name) for name in CORE_ARTIFACTS},
        "prediction_artifact": (
            sha256_directory(predictions_dir) if predictions_dir.exists() else {"exists": False, "path": str(predictions_dir)}
        ),
    }


def validate_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    missing = [name for name in CORE_ARTIFACTS if not (root / name).is_file()]
    if not (root / "averaged_predictions").is_dir():
        missing.append("averaged_predictions")
    return {
        "schema_version": 1,
        "run_dir": str(root.resolve()),
        "status": "OK" if not missing else "FAILED",
        "missing_artifacts": missing,
    }


def write_run_manifest(run_dir: str | Path, *, selection_rule: str | None = None) -> dict[str, Path]:
    root = Path(run_dir)
    manifest = build_run_manifest(root, selection_rule=selection_rule)
    validation = validate_run_bundle(root)
    manifest_path = root / "run_manifest.json"
    validation_path = root / "artifact_validation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "validation": validation_path}
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/build_run_bundle_manifest.py`:

```python
from __future__ import annotations

import argparse

from mci_gru.evaluation.run_bundle import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write additive run manifest artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection-rule", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_run_manifest(args.run_dir, selection_rule=args.selection_rule)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_bundle_manifest.py -v --basetemp .tmp_pytest\pytest
```

Expected: all tests pass.

---

### Task 2: Trial Ledger

**Files:**
- Create: `mci_gru/evaluation/trial_ledger.py`
- Create: `scripts/build_trial_ledger.py`
- Create: `tests/test_trial_ledger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trial_ledger.py`:

```python
import json
from pathlib import Path

import pandas as pd

from mci_gru.evaluation.trial_ledger import build_trial_record, write_trial_ledger


def test_build_trial_record_flattens_existing_summary_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"config": {"seed": 314159, "training": {"loss_type": "ic"}}}),
        encoding="utf-8",
    )
    (run_dir / "training_summary.json").write_text(
        json.dumps({"mean_best_val_ic": 0.011, "mean_best_val_rank_ic": 0.012}),
        encoding="utf-8",
    )

    row = build_trial_record(
        run_dir,
        trial_id="trial-001",
        family_id="pure-ic-baseline",
        status="OK",
    )

    assert row["trial_id"] == "trial-001"
    assert row["family_id"] == "pure-ic-baseline"
    assert row["status"] == "OK"
    assert row["run_metadata.config.seed"] == 314159
    assert row["run_metadata.config.training.loss_type"] == "ic"
    assert row["training_summary.mean_best_val_ic"] == 0.011


def test_write_trial_ledger_writes_csv_and_jsonl(tmp_path: Path) -> None:
    output_dir = tmp_path / "ledger"
    paths = write_trial_ledger(
        [{"trial_id": "a", "status": "OK"}, {"trial_id": "b", "status": "FAILED"}],
        output_dir,
    )

    assert pd.read_csv(paths["csv"])["trial_id"].tolist() == ["a", "b"]
    jsonl_rows = [json.loads(line) for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in jsonl_rows] == ["OK", "FAILED"]
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_trial_ledger.py -v --basetemp .tmp_pytest\pytest
```

Expected: import failure for `mci_gru.evaluation.trial_ledger`.

- [ ] **Step 3: Implement ledger helpers**

Create `mci_gru/evaluation/trial_ledger.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def flatten_mapping(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_mapping(value, name))
        else:
            out[name] = value
    return out


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_trial_record(
    run_dir: str | Path,
    *,
    trial_id: str,
    family_id: str,
    status: str,
) -> dict[str, Any]:
    root = Path(run_dir)
    row: dict[str, Any] = {
        "trial_id": trial_id,
        "family_id": family_id,
        "status": status,
        "run_dir": str(root.resolve()),
    }
    for file_name, prefix in [
        ("run_metadata.json", "run_metadata"),
        ("training_summary.json", "training_summary"),
        ("evaluation_summary.json", "evaluation_summary"),
        ("run_manifest.json", "run_manifest"),
        ("artifact_validation.json", "artifact_validation"),
    ]:
        payload = read_json_if_present(root / file_name)
        row.update(flatten_mapping(payload, prefix))
    return row


def write_trial_ledger(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trial_ledger.csv"
    jsonl_path = out_dir / "trial_ledger.jsonl"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {"csv": csv_path, "jsonl": jsonl_path}
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/build_trial_ledger.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from mci_gru.evaluation.trial_ledger import build_trial_record, write_trial_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trial ledger from existing run folders.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status", default="UNKNOWN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        build_trial_record(
            Path(run_dir),
            trial_id=Path(run_dir).name,
            family_id=args.family_id,
            status=args.status,
        )
        for run_dir in args.run_dir
    ]
    paths = write_trial_ledger(records, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_trial_ledger.py -v --basetemp .tmp_pytest\pytest
```

Expected: all tests pass.

---

### Task 3: Saved-Prediction Selection Audit V0

**Files:**
- Create: `mci_gru/evaluation/selection_audit.py`
- Create: `scripts/run_saved_prediction_selection_audit.py`
- Create: `tests/test_saved_prediction_selection_audit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_saved_prediction_selection_audit.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from mci_gru.evaluation.selection_audit import build_selection_audit


def test_selection_audit_computes_ic_topk_and_multiple_testing(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB"],
            "score": [0.5, -0.5, 0.4, -0.4],
        }
    ).to_csv(predictions_dir / "predictions.csv", index=False)
    market = pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [100.0, 100.0, 110.0, 90.0, 121.0, 81.0],
        }
    )
    market_path = tmp_path / "market.csv"
    market.to_csv(market_path, index=False)

    audit = build_selection_audit(
        predictions_dir=predictions_dir,
        market_data_path=market_path,
        label_t=2,
        top_k_values=[1],
        trial_count=4,
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )

    assert audit["schema_version"] == 1
    assert audit["trial_count"] == 4
    assert audit["sample"]["aligned_observations"] == 4
    assert audit["ic"]["pearson_mean"] == pytest.approx(1.0)
    assert audit["ic"]["spearman_mean"] == pytest.approx(1.0)
    assert audit["top_k"]["1"]["mean_return"] > 0
    assert 0.0 <= audit["multiple_testing"]["bhy_adjusted_p_value"] <= 1.0
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_saved_prediction_selection_audit.py -v --basetemp .tmp_pytest\pytest
```

Expected: import failure for `mci_gru.evaluation.selection_audit`.

- [ ] **Step 3: Implement selection audit helpers**

Create `mci_gru/evaluation/selection_audit.py` with public functions:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mci_gru.evaluation.portfolio import top_k_returns
from mci_gru.evaluation.prediction_report import (
    align_prediction_comparison,
    load_prediction_files,
    realized_returns_from_market_data,
)
from mci_gru.evaluation.statistics import (
    daily_ic_series,
    moving_block_bootstrap_ci,
    newey_west_std,
)


def bhy_adjust_p_value(p_value: float, trial_count: int) -> float:
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    harmonic = sum(1.0 / i for i in range(1, trial_count + 1))
    return float(min(1.0, p_value * trial_count * harmonic))


def _pivot(frame: pd.DataFrame, value_col: str) -> np.ndarray:
    wide = frame.pivot(index="dt", columns="kdcode", values=value_col).sort_index()
    return wide.to_numpy(dtype=np.float64)


def build_selection_audit(
    *,
    predictions_dir: str | Path,
    market_data_path: str | Path,
    label_t: int,
    top_k_values: list[int],
    trial_count: int,
    bootstrap_resamples: int = 500,
    bootstrap_seed: int = 123,
) -> dict[str, Any]:
    predictions = load_prediction_files(predictions_dir)
    market = pd.read_csv(market_data_path)
    realized = realized_returns_from_market_data(market, label_t=label_t)
    aligned = align_prediction_comparison(predictions, realized)
    score_matrix = _pivot(aligned, "mci_gru_score")
    return_matrix = _pivot(aligned, "realized_return")

    pearson = daily_ic_series(score_matrix, return_matrix, method="pearson")
    spearman = daily_ic_series(score_matrix, return_matrix, method="spearman")
    nw_std = newey_west_std(spearman, lags=max(0, label_t - 1))
    rank_ic_mean = float(np.nanmean(spearman)) if spearman.size else float("nan")
    t_stat = float(rank_ic_mean / (nw_std / np.sqrt(max(len(spearman), 1)))) if nw_std > 0 else 0.0
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(len(spearman) - 1, 1)))

    topk: dict[str, Any] = {}
    for top_k in top_k_values:
        returns = top_k_returns(score_matrix, return_matrix, top_k=top_k)
        topk[str(top_k)] = {
            "mean_return": float(np.nanmean(returns)) if returns.size else float("nan"),
            "n_days": int(returns.size),
        }

    return {
        "schema_version": 1,
        "predictions_dir": str(Path(predictions_dir).resolve()),
        "market_data_path": str(Path(market_data_path).resolve()),
        "label_t": label_t,
        "trial_count": trial_count,
        "sample": {
            "aligned_observations": int(len(aligned)),
            "n_dates": int(aligned["dt"].nunique()),
            "n_kdcodes": int(aligned["kdcode"].nunique()),
        },
        "ic": {
            "pearson_mean": float(np.nanmean(pearson)) if pearson.size else float("nan"),
            "spearman_mean": rank_ic_mean,
            "spearman_newey_west_t": t_stat,
            "spearman_p_value": p_value,
            "spearman_bootstrap_ci": moving_block_bootstrap_ci(
                spearman,
                statistic=lambda values: float(np.nanmean(values)),
                block_size=max(1, label_t),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                ci_level=0.95,
            ),
        },
        "top_k": topk,
        "multiple_testing": {
            "method": "bhy_single_family_v0",
            "bhy_adjusted_p_value": bhy_adjust_p_value(p_value, trial_count),
        },
    }


def write_selection_audit(audit: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "selection_audit_summary.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return path
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/run_saved_prediction_selection_audit.py`:

```python
from __future__ import annotations

import argparse

from mci_gru.evaluation.selection_audit import build_selection_audit, write_selection_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit saved predictions without retraining.")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--market-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-t", type=int, default=5)
    parser.add_argument("--top-k", type=int, action="append", default=[10, 20, 50])
    parser.add_argument("--trial-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_selection_audit(
        predictions_dir=args.predictions_dir,
        market_data_path=args.market_data_path,
        label_t=args.label_t,
        top_k_values=args.top_k,
        trial_count=args.trial_count,
    )
    path = write_selection_audit(audit, args.output_dir)
    print(f"selection_audit_summary: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_saved_prediction_selection_audit.py tests\test_prediction_report.py -v --basetemp .tmp_pytest\pytest
```

Expected: all tests pass.

---

### Task 4: PIT Availability, Tradability, And Staleness Report

**Files:**
- Create: `mci_gru/data/pit_audit.py`
- Create: `scripts/write_pit_availability_report.py`
- Create: `tests/test_pit_availability_report.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pit_availability_report.py`:

```python
import pandas as pd

from mci_gru.data.pit_audit import build_pit_availability_report


def test_pit_availability_report_keeps_masked_panel_and_reports_tradability() -> None:
    market = pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "kdcode": ["AAA", "BBB", "AAA"],
            "open": [10.0, 5.0, 10.5],
            "close": [10.2, 5.1, 10.6],
            "volume": [1_000_000, 0, 800_000],
        }
    )
    pit = pd.DataFrame(
        {
            "kdcode": ["AAA", "BBB"],
            "valid_from": ["2024-01-01", "2024-01-01"],
            "valid_to": ["2024-12-31", "2024-12-31"],
        }
    )

    report = build_pit_availability_report(
        market,
        pit,
        min_price=6.0,
        min_dollar_volume=1_000_000.0,
        stale_after_days=1,
    )

    assert report["schema_version"] == 1
    assert report["pit_union_kdcodes"] == 2
    assert report["dates"][0]["active_members"] == 2
    assert report["dates"][0]["tradable_count"] == 1
    assert report["dates"][0]["zero_volume_count"] == 1
    assert report["policy"]["masked_panel_preserved"] is True
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pit_availability_report.py -v --basetemp .tmp_pytest\pytest
```

Expected: import failure for `mci_gru.data.pit_audit`.

- [ ] **Step 3: Implement additive PIT audit**

Create `mci_gru/data/pit_audit.py` with a report-only implementation. The helper must not filter model tensors or modify PIT masks.

```python
from __future__ import annotations

from typing import Any

import pandas as pd


def build_pit_availability_report(
    market_data: pd.DataFrame,
    pit_universe: pd.DataFrame,
    *,
    min_price: float,
    min_dollar_volume: float,
    stale_after_days: int,
) -> dict[str, Any]:
    market = market_data.copy()
    pit = pit_universe.copy()
    market["dt"] = pd.to_datetime(market["dt"])
    pit["valid_from"] = pd.to_datetime(pit["valid_from"])
    pit["valid_to"] = pd.to_datetime(pit["valid_to"])
    market["dollar_volume"] = market["close"] * market["volume"]
    dates = []

    for date, day in market.groupby("dt", sort=True):
        active = pit[(pit["valid_from"] <= date) & (pit["valid_to"] >= date)]
        active_codes = set(active["kdcode"])
        day_active = day[day["kdcode"].isin(active_codes)].copy()
        zero_volume = int((day_active["volume"] <= 0).sum())
        tradable = day_active[
            (day_active["open"] >= min_price)
            & (day_active["close"] >= min_price)
            & (day_active["volume"] > 0)
            & (day_active["dollar_volume"] >= min_dollar_volume)
        ]
        dates.append(
            {
                "dt": date.strftime("%Y-%m-%d"),
                "active_members": int(len(active_codes)),
                "observed_members": int(day_active["kdcode"].nunique()),
                "missing_members": int(len(active_codes - set(day_active["kdcode"]))),
                "zero_volume_count": zero_volume,
                "tradable_count": int(tradable["kdcode"].nunique()),
            }
        )

    return {
        "schema_version": 1,
        "pit_union_kdcodes": int(pit["kdcode"].nunique()),
        "policy": {
            "masked_panel_preserved": True,
            "min_price": min_price,
            "min_dollar_volume": min_dollar_volume,
            "stale_after_days": stale_after_days,
        },
        "dates": dates,
    }
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/write_pit_availability_report.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mci_gru.data.pit_audit import build_pit_availability_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a report-only PIT availability audit.")
    parser.add_argument("--market-data", required=True)
    parser.add_argument("--pit-universe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--stale-after-days", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_pit_availability_report(
        pd.read_csv(args.market_data),
        pd.read_csv(args.pit_universe),
        min_price=args.min_price,
        min_dollar_volume=args.min_dollar_volume,
        stale_after_days=args.stale_after_days,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pit_availability_report: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pit_availability_report.py -v --basetemp .tmp_pytest\pytest
```

Expected: all tests pass.

---

### Task 5: Execution-Cost And Capacity Replay V0

**Files:**
- Create: `mci_gru/evaluation/capacity.py`
- Create: `scripts/run_saved_prediction_capacity_replay.py`
- Create: `tests/test_capacity_replay.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_capacity_replay.py`:

```python
import pandas as pd

from mci_gru.evaluation.capacity import compute_capacity_replay


def test_capacity_replay_uses_lagged_dollar_volume() -> None:
    predictions = pd.DataFrame(
        {
            "dt": ["2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB"],
            "score": [1.0, 0.5],
        }
    )
    market = pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB"],
            "open": [10.0, 20.0, 11.0, 19.0],
            "close": [10.5, 20.5, 11.5, 19.5],
            "volume": [1_000_000, 100_000, 9_999_999, 9_999_999],
        }
    )

    report = compute_capacity_replay(
        predictions,
        market,
        aum_values=[1_000_000.0],
        top_k_values=[2],
        adv_lookback_days=1,
        max_adv_participation=0.10,
    )

    row = report["rows"][0]
    assert row["dt"] == "2024-01-03"
    assert row["top_k"] == 2
    assert row["aum"] == 1_000_000.0
    assert row["max_participation"] > 0
    assert row["capacity_breach_count"] == 0
    assert report["policy"]["uses_lagged_adv"] is True
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_capacity_replay.py -v --basetemp .tmp_pytest\pytest
```

Expected: import failure for `mci_gru.evaluation.capacity`.

- [ ] **Step 3: Implement capacity helpers**

Create `mci_gru/evaluation/capacity.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_lagged_adv(
    market_data: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    market = market_data.copy()
    market["dt"] = pd.to_datetime(market["dt"])
    market["dollar_volume"] = market["close"] * market["volume"]
    market = market.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    market["lagged_adv"] = (
        market.groupby("kdcode")["dollar_volume"]
        .transform(lambda series: series.shift(1).rolling(lookback_days, min_periods=1).mean())
    )
    return market


def compute_capacity_replay(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    aum_values: list[float],
    top_k_values: list[int],
    adv_lookback_days: int,
    max_adv_participation: float,
) -> dict[str, Any]:
    preds = predictions.copy()
    preds["dt"] = pd.to_datetime(preds["dt"])
    market = add_lagged_adv(market_data, lookback_days=adv_lookback_days)
    merged = preds.merge(market[["dt", "kdcode", "lagged_adv"]], on=["dt", "kdcode"], how="left")
    rows = []
    for date, day in merged.groupby("dt", sort=True):
        ranked = day.sort_values(["score", "kdcode"], ascending=[False, True])
        for top_k in top_k_values:
            selected = ranked.head(top_k).copy()
            if selected.empty:
                continue
            for aum in aum_values:
                target_notional = aum / top_k
                selected["participation"] = target_notional / selected["lagged_adv"]
                finite_participation = selected["participation"].replace([np.inf, -np.inf], np.nan)
                rows.append(
                    {
                        "dt": date.strftime("%Y-%m-%d"),
                        "top_k": int(top_k),
                        "aum": float(aum),
                        "target_notional_per_name": float(target_notional),
                        "max_participation": float(finite_participation.max(skipna=True)),
                        "median_participation": float(finite_participation.median(skipna=True)),
                        "capacity_breach_count": int((finite_participation > max_adv_participation).sum()),
                        "missing_adv_count": int(finite_participation.isna().sum()),
                    }
                )
    return {
        "schema_version": 1,
        "policy": {
            "uses_lagged_adv": True,
            "adv_lookback_days": adv_lookback_days,
            "max_adv_participation": max_adv_participation,
        },
        "rows": rows,
    }
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/run_saved_prediction_capacity_replay.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mci_gru.evaluation.capacity import compute_capacity_replay
from mci_gru.evaluation.prediction_report import load_prediction_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved predictions through capacity diagnostics.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--market-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aum", type=float, action="append", required=True)
    parser.add_argument("--top-k", type=int, action="append", default=[10])
    parser.add_argument("--adv-lookback-days", type=int, default=20)
    parser.add_argument("--max-adv-participation", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compute_capacity_replay(
        load_prediction_input(args.predictions),
        pd.read_csv(args.market_data),
        aum_values=args.aum,
        top_k_values=args.top_k,
        adv_lookback_days=args.adv_lookback_days,
        max_adv_participation=args.max_adv_participation,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "capacity_replay.json"
    csv_path = out_dir / "capacity_replay.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["rows"]).to_csv(csv_path, index=False)
    print(f"capacity_replay_json: {json_path}")
    print(f"capacity_replay_csv: {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_capacity_replay.py -v --basetemp .tmp_pytest\pytest
```

Expected: all tests pass.

---

### Task 6: Evidence Harness Documentation And Focused Verification

**Files:**
- Create: `docs/evaluation/EVIDENCE_HARNESS.md`
- Modify: `docs/index.md`
- Run: focused pytest and ruff checks

- [ ] **Step 1: Write `docs/evaluation/EVIDENCE_HARNESS.md`**

Create a short operator document with these sections:

```markdown
# Evidence Harness

This harness implements the first wave of `docs/research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md`.

## Outputs

| Artifact | Writer | Purpose |
| --- | --- | --- |
| `run_manifest.json` | `scripts/build_run_bundle_manifest.py` | Hashes and describes an existing run folder. |
| `artifact_validation.json` | `scripts/build_run_bundle_manifest.py` | Reports missing core artifacts without modifying the run. |
| `trial_ledger.csv` / `trial_ledger.jsonl` | `scripts/build_trial_ledger.py` | Lists all candidate, failed, skipped, and promoted trials. |
| `selection_audit_summary.json` | `scripts/run_saved_prediction_selection_audit.py` | Computes saved-prediction IC/top-k/multiple-testing evidence. |
| `pit_availability_report.json` | `scripts/write_pit_availability_report.py` | Reports PIT breadth, missingness, staleness, and tradability. |
| `capacity_replay.json` / `capacity_replay.csv` | `scripts/run_saved_prediction_capacity_replay.py` | Replays saved scores through lagged ADV capacity diagnostics. |

## Invariants

- The harness does not retrain.
- The harness does not change model defaults.
- The harness does not rewrite saved predictions or backtest outputs.
- PIT masked-panel breadth stays intact; tradability is reported separately.
- Capacity calculations use lagged ADV known by prediction date.
```

- [ ] **Step 2: Link the doc from `docs/index.md`**

Add one row under Operations And Evaluation:

```markdown
| [evaluation/EVIDENCE_HARNESS.md](evaluation/EVIDENCE_HARNESS.md) | Additive run manifests, trial ledgers, saved-prediction audits, PIT availability reports, and capacity replay. |
```

- [ ] **Step 3: Run focused test set**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_run_bundle_manifest.py tests\test_trial_ledger.py tests\test_saved_prediction_selection_audit.py tests\test_pit_availability_report.py tests\test_capacity_replay.py tests\test_prediction_report.py tests\test_pit_saved_prediction_backtests.py -v --basetemp .tmp_pytest\pytest
```

Expected: all tests pass.

- [ ] **Step 4: Run lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check mci_gru\evaluation mci_gru\data scripts tests\test_run_bundle_manifest.py tests\test_trial_ledger.py tests\test_saved_prediction_selection_audit.py tests\test_pit_availability_report.py tests\test_capacity_replay.py
```

Expected: no ruff errors.

- [ ] **Step 5: Run no-retraining smoke commands on tiny synthetic fixtures**

Use the pytest-created fixture patterns rather than real data. If a developer wants a manual smoke, create a local temp run under `.tmp_evidence_harness/` and run:

```powershell
.\.venv\Scripts\python.exe scripts\build_run_bundle_manifest.py --run-dir .tmp_evidence_harness\run --selection-rule "validation rank IC"
.\.venv\Scripts\python.exe scripts\build_trial_ledger.py --run-dir .tmp_evidence_harness\run --family-id smoke --output-dir .tmp_evidence_harness\ledger --status OK
```

Expected: commands write only additive artifacts under `.tmp_evidence_harness/`.

---

## Self-Review Checklist

- June 21 source map is promoted in `docs/research/current/`.
- First implementation wave is no-retraining and additive.
- Every planned helper has a focused pytest target.
- Capacity gates use lagged ADV, not future volume.
- PIT report preserves masked-panel breadth.
- Trial ledger includes failed/skipped/ugly variants by design.
- Verification uses repo venv and repo-local pytest temp.
