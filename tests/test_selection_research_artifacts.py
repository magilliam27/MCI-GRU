import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mci_gru.evaluation.artifacts import (
    build_research_study_id,
    write_selection_research_bundle,
)

DATE_EVIDENCE_COLUMNS = (
    "signal_dt",
    "prediction_set_id",
    "daily_rank_ic",
    "valid_date_count",
    "status",
)


def _bundle_inputs(root: Path) -> dict:
    return {
        "research_semantics_version": "selection-research-v1",
        "protocol": {
            "primary_endpoint": "mean_daily_rank_ic",
            "source_path": root / "inputs" / "predictions.csv",
            "top_k": 10,
        },
        "input_hashes": {
            "predictions": "a" * 64,
            "market": "b" * 64,
            "operational_root": str(root.resolve()),
        },
        "code_identity": {
            "git_commit": "d6b0f60ba414d2152dfb0cc6ef715e0860d6a1fe",
            "repo_root": root.resolve(),
        },
        "date_evidence": [
            {
                "signal_dt": "2026-01-05",
                "prediction_set_id": "seed-2",
                "daily_rank_ic": np.float64(-0.0),
                "valid_date_count": np.int64(1),
                "status": "VALID",
            },
            {
                "signal_dt": "2026-01-04",
                "prediction_set_id": "seed-1",
                "daily_rank_ic": np.float64(0.125),
                "valid_date_count": 1,
                "status": "VALID",
            },
        ],
        "date_evidence_columns": DATE_EVIDENCE_COLUMNS,
        "result": {
            "claim_status": "INVALID_EVIDENCE",
            "headline": {
                "effect": float("nan"),
                "p_value": float("inf"),
            },
        },
        "report": "# Research result\r\n\r\nOperational root: " + str(root.resolve()),
    }


def test_write_selection_research_bundle_is_canonical_and_manifested(tmp_path: Path) -> None:
    paths = write_selection_research_bundle(tmp_path / "evidence", **_bundle_inputs(tmp_path))
    bundle_dir = paths["bundle_dir"]

    assert isinstance(bundle_dir, Path)
    assert sorted(path.name for path in bundle_dir.iterdir()) == [
        "date_evidence.csv",
        "manifest.json",
        "protocol.json",
        "report.md",
        "result.json",
    ]
    assert "force" not in inspect.signature(write_selection_research_bundle).parameters

    for path in bundle_dir.iterdir():
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw

    protocol_raw = (bundle_dir / "protocol.json").read_bytes()
    assert protocol_raw == (b'{"primary_endpoint":"mean_daily_rank_ic","top_k":10}\n')

    result = json.loads((bundle_dir / "result.json").read_text(encoding="utf-8"))
    assert result["claim_status"] == "INVALID_EVIDENCE"
    assert result["headline"] == {"effect": None, "p_value": None}

    with (bundle_dir / "date_evidence.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(DATE_EVIDENCE_COLUMNS)
    assert [row["signal_dt"] for row in rows] == ["2026-01-04", "2026-01-05"]
    assert rows[0]["daily_rank_ic"] == "0.125"
    assert rows[1]["daily_rank_ic"] == "0"

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["study_id"] == paths["study_id"]
    assert sorted(manifest["artifacts"]) == [
        "date_evidence.csv",
        "protocol.json",
        "report.md",
        "result.json",
    ]
    for name, description in manifest["artifacts"].items():
        raw = (bundle_dir / name).read_bytes()
        assert description == {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }


def test_cross_root_semantic_inputs_produce_identical_ids_and_bytes(tmp_path: Path) -> None:
    left_root = tmp_path / "left-worktree"
    right_root = tmp_path / "right-worktree"

    left = write_selection_research_bundle(
        tmp_path / "left-output",
        **_bundle_inputs(left_root),
    )
    right = write_selection_research_bundle(
        tmp_path / "right-output",
        **_bundle_inputs(right_root),
    )

    assert left["study_id"] == right["study_id"]
    for name in (
        "protocol.json",
        "date_evidence.csv",
        "result.json",
        "report.md",
        "manifest.json",
    ):
        assert (left["bundle_dir"] / name).read_bytes() == (right["bundle_dir"] / name).read_bytes()
    assert str(left_root).encode() not in (left["bundle_dir"] / "report.md").read_bytes()
    assert str(right_root).encode() not in (right["bundle_dir"] / "report.md").read_bytes()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("research_semantics_version", "selection-research-v2"),
        ("protocol", {"primary_endpoint": "mean_top_k_spread", "top_k": 10}),
        ("input_hashes", {"predictions": "c" * 64, "market": "b" * 64}),
        ("code_identity", {"git_commit": "another-commit"}),
    ],
)
def test_study_id_changes_for_each_semantic_identity_component(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    inputs = _bundle_inputs(tmp_path)
    base = build_research_study_id(
        research_semantics_version=inputs["research_semantics_version"],
        protocol=inputs["protocol"],
        input_hashes=inputs["input_hashes"],
        code_identity=inputs["code_identity"],
    )
    inputs[field] = replacement

    changed = build_research_study_id(
        research_semantics_version=inputs["research_semantics_version"],
        protocol=inputs["protocol"],
        input_hashes=inputs["input_hashes"],
        code_identity=inputs["code_identity"],
    )

    assert changed != base


def test_bundle_verifies_identical_rerun_and_rejects_conflicting_bytes(tmp_path: Path) -> None:
    output_root = tmp_path / "evidence"
    inputs = _bundle_inputs(tmp_path)
    first = write_selection_research_bundle(output_root, **inputs)

    second = write_selection_research_bundle(output_root, **inputs)

    assert second == first
    (first["bundle_dir"] / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="byte mismatch"):
        write_selection_research_bundle(output_root, **inputs)

    assert first["bundle_dir"].is_dir()
