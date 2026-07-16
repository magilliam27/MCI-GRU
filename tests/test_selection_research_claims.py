from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mci_gru.evaluation.selection_audit import (
    SelectionResearchProtocol,
    build_selection_research_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_positive_complete_60_date_fixture_can_claim_preliminary_but_not_confirmatory(
    tmp_path: Path,
) -> None:
    paths = _write_sixty_date_fixture(tmp_path, reverse_scores=False)

    evidence = build_selection_research_evidence(_protocol(paths))

    assert evidence.result["claim_status"] == "PRELIMINARY_SIGNAL_EVIDENCE"
    assert evidence.result["valid_date_count"] == 60
    assert evidence.result["observed_mean_rank_ic"] == 1.0
    assert evidence.result["empirical_p_value"] is not None
    assert evidence.result["empirical_p_value"] <= 0.05
    assert evidence.result["multiplicity_status"] == "UNADJUSTED_EXPLORATORY"
    assert "CONFIRMATORY_SIGNAL_EVIDENCE" not in evidence.report


def test_negative_60_date_fixture_reports_no_detectable_signal(tmp_path: Path) -> None:
    paths = _write_sixty_date_fixture(tmp_path, reverse_scores=True)

    evidence = build_selection_research_evidence(_protocol(paths))

    assert evidence.result["claim_status"] == "NO_DETECTABLE_SIGNAL"
    assert evidence.result["valid_date_count"] == 60
    assert evidence.result["observed_mean_rank_ic"] == -1.0
    assert evidence.result["empirical_p_value"] is not None
    assert evidence.result["empirical_p_value"] > 0.05


def _protocol(paths: dict[str, Path | str]) -> SelectionResearchProtocol:
    return SelectionResearchProtocol(
        research_semantics_version="selection-research-v1",
        study_name="sixty-date-claim-fixture",
        trial_family_id="fixture-family",
        predictions_dir=paths["predictions"],
        market_data_path=paths["market"],
        pit_universe_path=paths["pit"],
        expected_scorable_path=paths["expected"],
        calendar_path=paths["calendar"],
        label_horizon=2,
        test_start=str(paths["test_start"]),
        test_end=str(paths["test_end"]),
        data_as_of=str(paths["data_as_of"]),
        top_k=2,
        price_basis="ADJUSTED_RESEARCH",
        price_adjustment_provenance="fixture-adjusted-close-v1",
        null_draws=1000,
        null_seed=17,
        hac_lag=1,
        bootstrap_block_length=2,
        bootstrap_resamples=100,
        bootstrap_seed=19,
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


def _write_sixty_date_fixture(
    tmp_path: Path,
    *,
    reverse_scores: bool,
) -> dict[str, Path | str]:
    dates = pd.date_range("2024-01-02", periods=62, freq="B").strftime("%Y-%m-%d")
    instruments = ("AAA", "BBB", "CCC", "DDD")
    daily_growth = {"AAA": 1.04, "BBB": 1.03, "CCC": 1.02, "DDD": 1.01}
    score_values = {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0}
    if reverse_scores:
        score_values = {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0}

    predictions_dir = tmp_path / "averaged_predictions"
    predictions_dir.mkdir()
    pd.DataFrame(
        [
            {"dt": dt, "kdcode": kdcode, "score": score_values[kdcode]}
            for dt in dates[:60]
            for kdcode in instruments
        ]
    ).to_csv(predictions_dir / "predictions.csv", index=False)

    market_path = tmp_path / "market.csv"
    pd.DataFrame(
        [
            {
                "dt": dt,
                "kdcode": kdcode,
                "close": 100.0 * daily_growth[kdcode] ** date_index,
            }
            for date_index, dt in enumerate(dates)
            for kdcode in instruments
        ]
    ).to_csv(market_path, index=False)

    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame({"dt": dates}).to_csv(calendar_path, index=False)

    pit_path = tmp_path / "pit.csv"
    pd.DataFrame(
        {
            "kdcode": instruments,
            "valid_from": ["2024-01-01"] * len(instruments),
            "valid_to": ["2024-12-31"] * len(instruments),
            "known_from": ["2023-12-29"] * len(instruments),
        }
    ).to_csv(pit_path, index=False)

    expected_path = tmp_path / "expected.csv"
    pd.DataFrame(
        [
            {"dt": dt, "kdcode": kdcode, "expected_scorable": True}
            for dt in dates[:60]
            for kdcode in instruments
        ]
    ).to_csv(expected_path, index=False)

    return {
        "predictions": predictions_dir,
        "market": market_path,
        "pit": pit_path,
        "expected": expected_path,
        "calendar": calendar_path,
        "test_start": dates[0],
        "test_end": dates[59],
        "data_as_of": dates[-1],
    }
