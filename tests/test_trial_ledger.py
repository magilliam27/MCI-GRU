import json
from pathlib import Path

import pandas as pd
import pytest

from mci_gru.evaluation.trial_ledger import (
    build_trial_record,
    validate_trial_family,
    write_trial_ledger,
)


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
    jsonl_rows = [
        json.loads(line) for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in jsonl_rows] == ["OK", "FAILED"]


def test_write_trial_ledger_strict_jsonl_and_force_guard(tmp_path: Path) -> None:
    output_dir = tmp_path / "ledger"
    records = [{"trial_id": "a", "score": float("nan"), "risk": float("inf")}]

    paths = write_trial_ledger(records, output_dir)
    jsonl_text = paths["jsonl"].read_text(encoding="utf-8")

    assert "NaN" not in jsonl_text
    assert "Infinity" not in jsonl_text
    assert json.loads(jsonl_text)["score"] is None
    assert json.loads(jsonl_text)["risk"] is None
    with pytest.raises(FileExistsError):
        write_trial_ledger(records, output_dir)
    forced_paths = write_trial_ledger(records, output_dir, force=True)
    assert forced_paths["csv"].is_file()


def test_validate_trial_family_rejects_missing_expected_member() -> None:
    records = pd.DataFrame([{"trial_id": "seed-1", "family_id": "study-a", "status": "OK"}])

    with pytest.raises(ValueError, match=r"missing=\['seed-2'\]"):
        validate_trial_family(
            records,
            family_id="study-a",
            expected_trial_ids=["seed-1", "seed-2"],
        )
