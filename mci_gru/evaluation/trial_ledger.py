from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from mci_gru.evaluation.artifacts import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Collection


def validate_trial_family(
    records: pd.DataFrame,
    *,
    family_id: str,
    expected_trial_ids: Collection[str],
) -> None:
    """Require exact, unique, successful membership for one declared trial family."""
    required = {"trial_id", "family_id", "status"}
    missing_columns = required - set(records.columns)
    if missing_columns:
        raise ValueError(f"Trial ledger missing columns: {sorted(missing_columns)}")

    expected = {str(trial_id) for trial_id in expected_trial_ids}
    if not expected:
        raise ValueError("expected_trial_ids must not be empty")
    family = records[records["family_id"].astype(str) == str(family_id)].copy()
    duplicates = sorted(
        family.loc[family["trial_id"].astype(str).duplicated(), "trial_id"].astype(str).unique()
    )
    actual = set(family["trial_id"].astype(str))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    incomplete = sorted(
        family.loc[
            ~family["status"].astype(str).str.upper().isin({"OK", "COMPLETE"}),
            "trial_id",
        ]
        .astype(str)
        .unique()
    )
    if duplicates or missing or extra or incomplete:
        raise ValueError(
            f"Trial family {family_id!r} is incomplete: duplicates={duplicates}, "
            f"missing={missing}, extra={extra}, incomplete={incomplete}"
        )


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


def write_trial_ledger(
    records: list[dict[str, Any]], output_dir: str | Path, *, force: bool = False
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    csv_path = out_dir / "trial_ledger.csv"
    jsonl_path = out_dir / "trial_ledger.jsonl"
    if not force:
        for path in (csv_path, jsonl_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonable_records = [to_jsonable(record) for record in records]
    pd.DataFrame(jsonable_records).to_csv(csv_path, index=False)
    jsonl_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in jsonable_records
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "jsonl": jsonl_path}
