import json
from pathlib import Path

import pytest

from mci_gru.evaluation.run_bundle import (
    build_run_manifest,
    validate_run_bundle,
    write_run_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_run_manifest_hashes_core_artifacts_and_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    predictions_dir = run_dir / "averaged_predictions"
    predictions_dir.mkdir(parents=True)
    _write_json(run_dir / "run_metadata.json", {"config": {"seed": 314159}})
    _write_json(run_dir / "training_summary.json", {"mean_best_val_ic": 0.01})
    _write_json(run_dir / "evaluation_summary.json", {"avg_rank_ic": 0.02})
    _write_json(run_dir / "config.json", {"training": {"num_models": 2}})
    (run_dir / "graph_data.pt").write_bytes(b"graph")
    (run_dir / "model_seed_1.pth").write_bytes(b"checkpoint")
    (predictions_dir / "2024-01-02.csv").write_text(
        "dt,kdcode,score\n2024-01-02,AAA,0.1\n",
        encoding="utf-8",
    )

    manifest = build_run_manifest(
        run_dir,
        selection_rule="max validation rank IC",
        sibling_trial_ids=["trial-a", "trial-b"],
        command="python run_experiment.py training.num_models=2",
        feature_lag_policy="strict current-only",
        graph_policy="frozen graph_data.pt",
        paper_trade_eligible=True,
    )

    assert manifest["schema_version"] == 1
    assert manifest["run_dir"] == str(run_dir.resolve())
    assert manifest["selection_rule"] == "max validation rank IC"
    assert manifest["sibling_trial_ids"] == ["trial-a", "trial-b"]
    assert manifest["provenance"]["command"].startswith("python run_experiment.py")
    assert manifest["feature_lag_policy"] == "strict current-only"
    assert manifest["config"]["artifact"]["sha256"]
    assert manifest["config"]["metadata_config_sha256"]
    assert manifest["graph"]["artifact"]["exists"] is True
    assert manifest["checkpoints"]["file_count"] == 1
    assert manifest["paper_trade_eligible"] is True
    assert manifest["paper_trade_eligibility_inputs"]["has_frozen_graph"] is True
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


def test_validate_run_bundle_rejects_empty_prediction_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "averaged_predictions").mkdir(parents=True)
    _write_json(run_dir / "run_metadata.json", {"config": {}})
    _write_json(run_dir / "training_summary.json", {})
    _write_json(run_dir / "evaluation_summary.json", {})

    validation = validate_run_bundle(run_dir)

    assert validation["status"] == "FAILED"
    assert "averaged_predictions/*.csv" in validation["missing_artifacts"]


def test_write_run_manifest_refuses_overwrite_without_force(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    predictions_dir = run_dir / "averaged_predictions"
    predictions_dir.mkdir(parents=True)
    _write_json(run_dir / "run_metadata.json", {"config": {}})
    _write_json(run_dir / "training_summary.json", {})
    _write_json(run_dir / "evaluation_summary.json", {})
    (predictions_dir / "predictions.csv").write_text("dt,kdcode,score\n", encoding="utf-8")

    write_run_manifest(run_dir)

    with pytest.raises(FileExistsError):
        write_run_manifest(run_dir)

    paths = write_run_manifest(run_dir, force=True)
    assert paths["manifest"].is_file()
