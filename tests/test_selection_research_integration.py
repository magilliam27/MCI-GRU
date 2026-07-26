from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mci_gru.evaluation.selection_audit import (
    SelectionResearchProtocol,
    build_selection_research_evidence,
    write_selection_research_evidence,
)
from scripts.run_saved_prediction_selection_audit import main as selection_audit_main


def test_saved_prediction_selection_research_never_invokes_training(
    tmp_path: Path,
) -> None:
    paths = _write_research_fixture(tmp_path)
    output_dir = tmp_path / "no-training-evidence"
    cli_args = [
        "--predictions-dir",
        str(paths["predictions"]),
        "--market-data-path",
        str(paths["market"]),
        "--output-dir",
        str(output_dir),
        "--research-evidence",
        "--pit-universe-csv",
        str(paths["pit"]),
        "--expected-scorable-csv",
        str(paths["expected"]),
        "--calendar-csv",
        str(paths["calendar"]),
        "--study-name",
        "no-training-tracer",
        "--trial-family-id",
        "fixture-family",
        "--label-t",
        "2",
        "--top-k",
        "2",
        "--test-start",
        "2024-01-02",
        "--test-end",
        "2024-01-02",
        "--data-as-of",
        "2024-01-08",
        "--null-draws",
        "1000",
        "--bootstrap-resamples",
        "20",
    ]
    bootstrap = "\n".join(
        [
            "import json",
            "import sys",
            "from scripts.run_saved_prediction_selection_audit import main",
            "main(json.loads(sys.argv[1]))",
            "forbidden = sorted(name for name in sys.modules if "
            "name == 'run_experiment' or name.startswith('mci_gru.training'))",
            "if forbidden:",
            "    raise AssertionError(f'Training modules loaded: {forbidden}')",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, json.dumps(cli_args)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "selection_research_study_id:" in completed.stdout


def test_research_evidence_uses_canonical_sessions_and_keeps_invalid_dates(
    tmp_path: Path,
) -> None:
    paths = _write_research_fixture(tmp_path)
    evidence = build_selection_research_evidence(_protocol(paths, null_draws=1000))

    by_date = evidence.date_evidence.set_index("signal_dt")
    assert list(by_date.index) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert by_date.loc["2024-01-02", "label_start_dt"] == "2024-01-03"
    assert by_date.loc["2024-01-02", "label_end_dt"] == "2024-01-04"
    assert by_date.loc["2024-01-02", "date_status"] == "VALID_PRIMARY"
    assert by_date.loc["2024-01-02", "daily_rank_ic"] == pytest.approx(1.0)
    assert by_date.loc["2024-01-03", "date_status"] == "INVALID_PRIMARY"
    assert "MISSING_EXPECTED_SCORE" in by_date.loc["2024-01-03", "reason_codes"]
    assert by_date.loc["2024-01-03", "expected_scorable_count"] == 4
    assert by_date.loc["2024-01-03", "prediction_count"] == 3
    assert by_date.loc["2024-01-04", "date_status"] == "INVALID_PRIMARY"
    assert "MISSING_MATURED_OUTCOME" in by_date.loc["2024-01-04", "reason_codes"]
    assert pd.isna(by_date.loc["2024-01-04", "daily_rank_ic"])


def test_future_market_rows_do_not_change_prior_dated_evidence(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    protocol = _protocol(paths, test_end="2024-01-02", null_draws=1000)
    first = build_selection_research_evidence(protocol)

    market = pd.read_csv(paths["market"])
    market.loc[market["dt"] == "2024-01-08", "close"] *= 100.0
    market.to_csv(paths["market"], index=False)
    second = build_selection_research_evidence(protocol)

    pd.testing.assert_frame_equal(first.date_evidence, second.date_evidence)
    assert first.result["observed_mean_rank_ic"] == second.result["observed_mean_rank_ic"]


def test_missing_middle_session_is_not_replaced_by_later_stock_row(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    market = pd.read_csv(paths["market"])
    market = market[~((market["dt"] == "2024-01-03") & (market["kdcode"] == "DDD"))]
    market.to_csv(paths["market"], index=False)

    evidence = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-02", null_draws=1000)
    )

    row = evidence.date_evidence.iloc[0]
    assert row["date_status"] == "INVALID_PRIMARY"
    assert "MISSING_MATURED_OUTCOME" in row["reason_codes"]
    assert pd.isna(row["daily_rank_ic"])


def test_unmatured_tail_outcome_is_reported_and_never_zeroed(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    protocol = replace(
        _protocol(paths, null_draws=1000),
        data_as_of="2024-01-05",
    )

    evidence = build_selection_research_evidence(protocol)

    row = evidence.date_evidence.set_index("signal_dt").loc["2024-01-04"]
    assert row["date_status"] == "UNMATURED_OUTCOME"
    assert row["reason_codes"] == "OUTCOME_NOT_MATURED"
    assert pd.isna(row["daily_rank_ic"])
    assert pd.isna(row["top_k_spread"])


def test_top_k_spread_uses_the_same_expected_set_denominator(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)

    evidence = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-02", null_draws=1000)
    )

    row = evidence.date_evidence.iloc[0]
    assert row["top_k_label_return"] == pytest.approx(0.35)
    assert row["expected_set_label_return"] == pytest.approx(0.25)
    assert row["top_k_spread"] == pytest.approx(0.10)


def test_pit_knowledge_after_signal_close_invalidates_the_date(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    pit = pd.read_csv(paths["pit"])
    pit.loc[pit["kdcode"] == "EEE", "known_from"] = "2024-01-02T21:30:00Z"
    pit.to_csv(paths["pit"], index=False)

    evidence = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-02", null_draws=1000)
    )

    row = evidence.date_evidence.iloc[0]
    assert row["date_status"] == "INVALID_PRIMARY"
    assert "PIT_KNOWLEDGE_UNKNOWN" in row["reason_codes"]
    assert evidence.result["claim_status"] == "INVALID_EVIDENCE"


def test_explicit_signal_close_must_belong_to_its_local_session_date(
    tmp_path: Path,
) -> None:
    paths = _write_research_fixture(tmp_path)
    calendar = pd.read_csv(paths["calendar"])
    calendar["signal_close"] = [
        "2024-01-03T21:00:00Z",
        "2024-01-03T21:00:00Z",
        "2024-01-04T21:00:00Z",
        "2024-01-05T21:00:00Z",
        "2024-01-08T21:00:00Z",
    ]
    calendar.to_csv(paths["calendar"], index=False)

    with pytest.raises(ValueError, match="has signal_close on local date"):
        build_selection_research_evidence(_protocol(paths, test_end="2024-01-02", null_draws=1000))


def test_expected_denominator_covers_every_active_member_with_exclusion_reason(
    tmp_path: Path,
) -> None:
    paths = _write_research_fixture(tmp_path)
    expected = pd.read_csv(paths["expected"])
    expected = expected[~((expected["dt"] == "2024-01-02") & (expected["kdcode"] == "EEE"))]
    expected.to_csv(paths["expected"], index=False)

    missing_member = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-02", null_draws=1000)
    )
    assert "INCOMPLETE_EXPECTED_DENOMINATOR" in missing_member.date_evidence.iloc[0]["reason_codes"]

    expected = pd.read_csv(paths["expected"])
    expected.loc[len(expected)] = {
        "dt": "2024-01-02",
        "kdcode": "EEE",
        "expected_scorable": False,
        "exclusion_reason": "",
    }
    expected.to_csv(paths["expected"], index=False)
    missing_reason = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-02", null_draws=1000)
    )
    assert "MISSING_EXCLUSION_REASON" in missing_reason.date_evidence.iloc[0]["reason_codes"]


def test_pit_entry_and_exit_boundaries_change_only_the_declared_daily_denominator(
    tmp_path: Path,
) -> None:
    paths = _write_research_fixture(tmp_path)
    pit = pd.read_csv(paths["pit"])
    pit.loc[pit["kdcode"] == "EEE", "valid_to"] = "2024-01-02"
    pit = pd.concat(
        [
            pit,
            pd.DataFrame(
                [
                    {
                        "kdcode": kdcode,
                        "valid_from": "2024-01-03",
                        "valid_to": "2024-12-31",
                        "known_from": "2023-12-29",
                    }
                    for kdcode in ("FFF", "GGG")
                ]
            ),
        ],
        ignore_index=True,
    )
    pit.to_csv(paths["pit"], index=False)
    expected = pd.read_csv(paths["expected"])
    expected = pd.concat(
        [
            expected,
            pd.DataFrame(
                [
                    {
                        "dt": "2024-01-03",
                        "kdcode": kdcode,
                        "expected_scorable": False,
                        "exclusion_reason": "feature_warmup",
                    }
                    for kdcode in ("FFF", "GGG")
                ]
            ),
        ],
        ignore_index=True,
    )
    expected.to_csv(paths["expected"], index=False)

    evidence = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-03", null_draws=1000)
    )

    rows = evidence.date_evidence.set_index("signal_dt")
    assert rows.loc["2024-01-02", "PIT_active_count"] == 5
    assert rows.loc["2024-01-03", "PIT_active_count"] == 6
    assert "INCOMPLETE_EXPECTED_DENOMINATOR" not in rows.loc["2024-01-02", "reason_codes"]
    assert "INCOMPLETE_EXPECTED_DENOMINATOR" not in rows.loc["2024-01-03", "reason_codes"]


def test_invalid_price_provenance_nulls_headline_and_report_claim(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    protocol = replace(
        _protocol(paths, test_end="2024-01-02", null_draws=1000),
        price_basis="UNKNOWN",
    )

    evidence = build_selection_research_evidence(protocol)
    written = write_selection_research_evidence(evidence, tmp_path / "invalid-evidence")
    result = json.loads(written["result"].read_text(encoding="utf-8"))

    assert result["claim_status"] == "INVALID_EVIDENCE"
    assert result["observed_mean_rank_ic"] is None
    assert result["moving_block_bootstrap"]["lower"] is None
    assert result["empirical_p_value"] is None
    assert "CONFIRMATORY_SIGNAL_EVIDENCE" not in written["report"].read_text(encoding="utf-8")


def test_complete_trial_ledger_declaration_requires_a_hashed_ledger(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)

    with pytest.raises(ValueError, match="requires a hashed trial_ledger_path"):
        replace(
            _protocol(paths, test_end="2024-01-02", null_draws=1000),
            trial_ledger_complete=True,
        )


def test_complete_trial_ledger_declaration_requires_expected_members(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    ledger = tmp_path / "trial_ledger.csv"
    pd.DataFrame([{"trial_id": "seed-1", "family_id": "fixture-family", "status": "OK"}]).to_csv(
        ledger, index=False
    )

    with pytest.raises(ValueError, match="requires expected_trial_ids"):
        replace(
            _protocol(paths, test_end="2024-01-02", null_draws=1000),
            trial_ledger_path=ledger,
            trial_ledger_complete=True,
        )


def test_complete_trial_ledger_rejects_missing_declared_member(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    ledger = tmp_path / "trial_ledger.csv"
    pd.DataFrame([{"trial_id": "seed-1", "family_id": "fixture-family", "status": "OK"}]).to_csv(
        ledger, index=False
    )
    protocol = replace(
        _protocol(paths, test_end="2024-01-02", null_draws=1000),
        trial_ledger_path=ledger,
        trial_ledger_complete=True,
        expected_trial_ids=("seed-1", "seed-2"),
    )

    evidence = build_selection_research_evidence(protocol)

    assert evidence.protocol["multiplicity"]["expected_trial_ids"] == ["seed-1", "seed-2"]
    assert "TRIAL_LEDGER_FAMILY_MISMATCH" in evidence.result["failed_guards"]
    assert evidence.result["claim_status"] == "INVALID_EVIDENCE"


def test_end_to_end_bundle_is_byte_identical_across_output_roots(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    evidence = build_selection_research_evidence(
        _protocol(paths, test_end="2024-01-02", null_draws=1000)
    )

    left = write_selection_research_evidence(evidence, tmp_path / "left")
    right = write_selection_research_evidence(evidence, tmp_path / "right")

    assert left["study_id"] == right["study_id"] == evidence.result["study_id"]
    for name in (
        "protocol.json",
        "date_evidence.csv",
        "result.json",
        "report.md",
        "manifest.json",
    ):
        assert (left["bundle_dir"] / name).read_bytes() == (right["bundle_dir"] / name).read_bytes()


def test_cli_research_mode_writes_only_the_five_file_bundle(tmp_path: Path) -> None:
    paths = _write_research_fixture(tmp_path)
    output_dir = tmp_path / "evidence"

    selection_audit_main(
        [
            "--predictions-dir",
            str(paths["predictions"]),
            "--market-data-path",
            str(paths["market"]),
            "--output-dir",
            str(output_dir),
            "--research-evidence",
            "--pit-universe-csv",
            str(paths["pit"]),
            "--expected-scorable-csv",
            str(paths["expected"]),
            "--calendar-csv",
            str(paths["calendar"]),
            "--study-name",
            "perfect-signal-tracer",
            "--trial-family-id",
            "fixture-family",
            "--price-basis",
            "ADJUSTED_RESEARCH",
            "--price-adjustment-provenance",
            "fixture-adjusted-close-v1",
            "--prediction-source-run-id",
            "fixture-run",
            "--prediction-ensemble-rule",
            "averaged_predictions",
            "--prediction-ensemble-member-count",
            "20",
            "--prediction-seed-id",
            "314159",
            "--prediction-source-code-commit",
            "d6b0f60",
            "--prediction-label-contract",
            "MCI_GRU_FORWARD_CLOSE_V1",
            "--prediction-label-horizon",
            "2",
            "--label-t",
            "2",
            "--top-k",
            "2",
            "--test-start",
            "2024-01-02",
            "--test-end",
            "2024-01-04",
            "--data-as-of",
            "2024-01-08",
            "--null-draws",
            "1000",
            "--bootstrap-resamples",
            "40",
            "--null-seed",
            "7",
            "--bootstrap-seed",
            "11",
        ]
    )

    studies = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(studies) == 1
    assert sorted(path.name for path in studies[0].iterdir()) == [
        "date_evidence.csv",
        "manifest.json",
        "protocol.json",
        "report.md",
        "result.json",
    ]
    result = json.loads((studies[0] / "result.json").read_text(encoding="utf-8"))
    assert result["claim_status"] == "INVALID_EVIDENCE"
    assert result["valid_date_count"] == 1
    assert result["null_test"]["draw_count"] == 1000


def _protocol(
    paths: dict[str, Path],
    *,
    test_end: str = "2024-01-04",
    null_draws: int,
) -> SelectionResearchProtocol:
    return SelectionResearchProtocol(
        research_semantics_version="selection-research-v1",
        study_name="perfect-signal-tracer",
        trial_family_id="fixture-family",
        predictions_dir=paths["predictions"],
        market_data_path=paths["market"],
        pit_universe_path=paths["pit"],
        expected_scorable_path=paths["expected"],
        calendar_path=paths["calendar"],
        label_horizon=2,
        test_start="2024-01-02",
        test_end=test_end,
        data_as_of="2024-01-08",
        top_k=2,
        price_basis="ADJUSTED_RESEARCH",
        price_adjustment_provenance="fixture-adjusted-close-v1",
        null_draws=null_draws,
        null_seed=7,
        hac_lag=1,
        bootstrap_block_length=2,
        bootstrap_resamples=40,
        bootstrap_seed=11,
        ci_level=0.95,
        alpha=0.05,
        prediction_source_run_id="fixture-run",
        prediction_ensemble_rule="averaged_predictions",
        prediction_ensemble_member_count=20,
        prediction_seed_id="314159",
        prediction_source_code_commit="d6b0f60",
        prediction_label_contract="MCI_GRU_FORWARD_CLOSE_V1",
        prediction_label_horizon=2,
    )


def _write_research_fixture(tmp_path: Path) -> dict[str, Path]:
    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    predictions = []
    scores = {
        "2024-01-02": {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
        "2024-01-03": {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0},
        "2024-01-04": {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
    }
    for dt, by_stock in scores.items():
        for kdcode, score in by_stock.items():
            predictions.append({"dt": dt, "kdcode": kdcode, "score": score})
    pd.DataFrame(predictions).to_csv(predictions_dir / "predictions.csv", index=False)

    calendar_path = tmp_path / "calendar.csv"
    calendar_dates = pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d")
    pd.DataFrame({"dt": calendar_dates}).to_csv(calendar_path, index=False)

    growth = {"AAA": 1.40, "BBB": 1.30, "CCC": 1.20, "DDD": 1.10}
    market_rows = []
    for date_idx, dt in enumerate(calendar_dates):
        for kdcode, end_multiplier in growth.items():
            if dt == "2024-01-08" and kdcode == "DDD":
                continue
            close = 100.0
            if date_idx == 2:
                close *= end_multiplier
            elif date_idx > 2:
                close *= end_multiplier * (1.0 + 0.01 * (date_idx - 2))
            market_rows.append({"dt": dt, "kdcode": kdcode, "close": close})
    market_path = tmp_path / "market.csv"
    pd.DataFrame(market_rows).to_csv(market_path, index=False)

    pit_path = tmp_path / "pit.csv"
    pit_kdcodes = [*growth, "EEE"]
    pd.DataFrame(
        {
            "kdcode": pit_kdcodes,
            "valid_from": ["2024-01-01"] * len(pit_kdcodes),
            "valid_to": ["2024-12-31"] * len(pit_kdcodes),
            "known_from": ["2023-12-29"] * len(pit_kdcodes),
        }
    ).to_csv(pit_path, index=False)

    expected_rows = [
        {"dt": dt, "kdcode": kdcode, "expected_scorable": True, "exclusion_reason": ""}
        for dt in ["2024-01-02", "2024-01-03", "2024-01-04"]
        for kdcode in growth
    ]
    expected_rows.append(
        {
            "dt": "2024-01-02",
            "kdcode": "EEE",
            "expected_scorable": False,
            "exclusion_reason": "feature_warmup",
        }
    )
    for dt in ["2024-01-03", "2024-01-04"]:
        expected_rows.append(
            {
                "dt": dt,
                "kdcode": "EEE",
                "expected_scorable": False,
                "exclusion_reason": "feature_warmup",
            }
        )
    expected_path = tmp_path / "expected_scorable.csv"
    pd.DataFrame(expected_rows).to_csv(expected_path, index=False)

    return {
        "predictions": predictions_dir,
        "market": market_path,
        "pit": pit_path,
        "expected": expected_path,
        "calendar": calendar_path,
    }
