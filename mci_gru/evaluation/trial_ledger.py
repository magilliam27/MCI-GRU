from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.evaluation.artifacts import to_jsonable


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
