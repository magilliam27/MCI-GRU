import json
from pathlib import Path

import pandas as pd
import pytest

from mci_gru.evaluation.selection_audit import build_selection_audit, write_selection_audit
from scripts.run_saved_prediction_selection_audit import main as selection_audit_main


def test_selection_audit_computes_ic_topk_multiple_testing_and_deflated_sharpe(
    tmp_path: Path,
) -> None:
    predictions_dir, market_path = _write_selection_fixture(tmp_path)

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
    assert audit["status"] == "OK"
    assert audit["trial_count"] == 4
    assert audit["sample"]["aligned_observations"] == 4
    assert audit["ic"]["pearson_mean"] == pytest.approx(1.0)
    assert audit["ic"]["spearman_mean"] == pytest.approx(1.0)
    assert audit["top_k"]["1"]["mean_return"] > 0
    assert audit["deflated_sharpe"]["1"]["method"] == "bailey_lopez_de_prado_v0"
    assert 0.0 <= audit["deflated_sharpe"]["1"]["p_value"] <= 1.0
    assert 0.0 <= audit["multiple_testing"]["bhy_adjusted_p_value"] <= 1.0


def test_selection_audit_flags_insufficient_evidence_when_alignment_is_empty(
    tmp_path: Path,
) -> None:
    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    pd.DataFrame({"dt": ["2024-01-02"], "kdcode": ["AAA"], "score": [0.5]}).to_csv(
        predictions_dir / "predictions.csv", index=False
    )
    market_path = tmp_path / "market.csv"
    pd.DataFrame({"dt": ["2024-01-02"], "kdcode": ["AAA"], "close": [100.0]}).to_csv(
        market_path, index=False
    )

    audit = build_selection_audit(
        predictions_dir=predictions_dir,
        market_data_path=market_path,
        label_t=1,
        top_k_values=[1],
        trial_count=2,
        bootstrap_resamples=5,
    )

    assert audit["status"] == "INSUFFICIENT_EVIDENCE"
    assert "no_aligned_observations" in audit["insufficient_evidence_reasons"]
    assert audit["ic"]["spearman_p_value"] is None


def test_selection_audit_cli_uses_requested_top_k_without_default_duplicates(
    tmp_path: Path,
) -> None:
    predictions_dir, market_path = _write_selection_fixture(tmp_path)
    output_dir = tmp_path / "audit"

    selection_audit_main(
        [
            "--predictions-dir",
            str(predictions_dir),
            "--market-data-path",
            str(market_path),
            "--output-dir",
            str(output_dir),
            "--label-t",
            "2",
            "--top-k",
            "1",
            "--trial-count",
            "4",
        ]
    )

    payload = json.loads((output_dir / "selection_audit_summary.json").read_text())
    assert sorted(payload["top_k"]) == ["1"]


def test_write_selection_audit_strict_json_and_overwrite_guard(tmp_path: Path) -> None:
    audit = {"schema_version": 1, "bad_float": float("nan"), "nested": [float("inf")]}

    path = write_selection_audit(audit, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["bad_float"] is None
    assert payload["nested"] == [None]
    with pytest.raises(FileExistsError):
        write_selection_audit(audit, tmp_path)
    write_selection_audit(audit, tmp_path, force=True)


def _write_selection_fixture(tmp_path: Path) -> tuple[Path, Path]:
    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    pd.DataFrame(
        {
            "dt": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "kdcode": ["AAA", "BBB", "AAA", "BBB"],
            "score": [0.5, -0.5, 0.4, -0.4],
        }
    ).to_csv(predictions_dir / "predictions.csv", index=False)
    market_path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "dt": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
            ],
            "kdcode": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [100.0, 100.0, 110.0, 90.0, 121.0, 81.0, 130.0, 78.0],
        }
    ).to_csv(market_path, index=False)
    return predictions_dir, market_path
