"""Candidate models are scored on a saved run's own axis, then pooled over windows (#384).

``mci_gru.evaluation.candidates`` defines what a candidate model sees;
``scripts/score_candidates.py`` scores one on a saved run through the run's own
evaluation; ``scripts/pool_candidates.py`` joins windows. These tests pin that:

* a candidate never sees a price after the last test date, nor the test labels;
* a result of the wrong shape, or with an infinite score, is refused;
* the scoring script reproduces the run's own IC first, scores the candidate only
  where the run could rank, hands it the run's own training labels, and writes beside
  the run without touching it;
* the pooled report joins windows, refuses a window scored twice or folders that
  scored different runs, and its pooled IC is the mean of the joined series.

The end-to-end tests reuse ``tests/test_score_baselines.py``'s small synthetic run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_score_baselines import _run

from mci_gru.config import ExperimentConfig
from mci_gru.evaluation import candidate_registry
from mci_gru.evaluation.candidates import (
    CandidateContext,
    CandidateResult,
    check_result,
    panel_up_to,
)
from mci_gru.evaluation.dated_ic import dated_ic_frame
from scripts import pool_candidates, rescore_run, score_candidates
from scripts.score_candidates import SUMMARY_FILENAME, CandidateError

# --- What a candidate sees -------------------------------------------------------------


def _toy_context(**overrides) -> CandidateContext:
    panel = pd.DataFrame(
        {
            "kdcode": ["A", "B", "A", "B"],
            "dt": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
            "close": [1.0, 2.0, 1.1, 2.1],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    fields = {
        "config": ExperimentConfig(),
        "panel": panel,
        "kdcodes": ["A", "B"],
        "test_dates": ["2020-01-03"],
        "rankable_mask": None,
        "pit_intervals": None,
        "data_loader": lambda: {"train_dates": ["2020-01-02"], "test_labels": 1, "df": 2},
    }
    return CandidateContext(**{**fields, **overrides})


def test_a_panel_past_the_last_test_date_is_refused():
    with pytest.raises(ValueError, match="runs past the last test date"):
        _toy_context(test_dates=["2020-01-02"])


def test_panel_up_to_drops_every_later_row():
    panel = _toy_context().panel
    kept = panel_up_to(panel, "2020-01-02")
    assert set(kept["dt"]) == {"2020-01-02"}
    assert len(kept) == 2


def test_the_test_labels_are_left_out_of_the_data_and_it_is_built_once():
    calls = []

    def loader():
        calls.append(1)
        return {
            "train_dates": ["2020-01-02"],
            "val_dates": [],
            "test_labels": 1,
            "test_loss_mask": 1,
            "test_tradable_mask": 1,
            "pit_eligibility": 1,
            "pit_breadth": 1,
            "admission": 1,
            "df": 2,
        }

    context = _toy_context(data_loader=loader)
    for hidden in ("test_labels", "test_loss_mask", "pit_eligibility", "pit_breadth", "admission"):
        assert hidden not in context.data
    assert "df" not in context.data
    assert "test_tradable_mask" in context.data
    assert context.train_dates == ["2020-01-02"]
    assert context.data is context.data
    assert len(calls) == 1


def test_a_result_of_the_wrong_shape_or_with_an_infinite_score_is_refused():
    context = _toy_context()
    with pytest.raises(ValueError, match="shape"):
        check_result(CandidateResult(np.zeros((2, 2))), context)
    with pytest.raises(ValueError, match="infinite"):
        check_result(CandidateResult(np.array([[np.inf, 0.0]])), context)
    assert check_result(CandidateResult(np.array([[np.nan, 1.0]])), context).shape == (1, 2)


def test_every_registered_candidate_has_a_unique_name():
    assert {"residual_linear", "residual_reversal_1w"} <= set(candidate_registry.CANDIDATES)
    with pytest.raises(KeyError, match="unknown candidate"):
        candidate_registry.get_candidate("nope")


# --- Scoring a saved run ------------------------------------------------------------------


SEEN: dict[str, object] = {}


def _spy(context: CandidateContext) -> CandidateResult:
    """Scores each name by its last close, and records what it was shown."""
    SEEN["panel_last_date"] = str(context.panel["dt"].max())
    SEEN["data_keys"] = set(context.data)
    SEEN["train_labels"] = np.asarray(context.data["train_labels"])
    SEEN["train_dates"] = context.train_dates
    closes = context.panel.pivot_table(index="dt", columns="kdcode", values="close")
    closes = closes.reindex(index=context.test_dates, columns=context.kdcodes)
    return CandidateResult(closes.to_numpy(dtype=np.float64), {"note": "spy"})


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("candidates") / "pit", use_pit=True)


@pytest.fixture
def spy_registered(monkeypatch):
    monkeypatch.setitem(candidate_registry.CANDIDATES, "spy", _spy)
    SEEN.clear()


def _summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_candidate_is_scored_where_the_run_could_rank_and_sees_nothing_later(
    run, spy_registered, tmp_path
):
    path = score_candidates.score_candidate(run, "spy", out_dir=tmp_path / "spy")
    summary = _summary(path)
    saved = rescore_run.resolve_run_config(run)
    scoring = rescore_run.rebuild_scoring_inputs(saved.config)

    assert summary["controls"]["model_ic_reproduced"]
    assert SEEN["panel_last_date"] == scoring.dates[-1]
    assert "test_labels" not in SEEN["data_keys"]
    assert summary["controls"]["stock_level_inputs"]["normalisation_matches_run"] is True

    scores = pd.read_csv(path.parent / "scores" / "spy.csv", dtype={"dt": str})
    grid = scores.pivot(index="dt", columns="kdcode", values="score")
    grid = grid.reindex(index=scoring.dates, columns=scoring.kdcodes).to_numpy()
    assert np.array_equal(np.isfinite(grid), scoring.rankable_mask & np.isfinite(grid))
    assert not np.isfinite(grid[~scoring.rankable_mask]).any()

    daily = pd.read_csv(path.parent / "daily_ic" / "spy.csv", dtype={"dt": str})
    expected = dated_ic_frame(grid, scoring.labels, scoring.dates)
    np.testing.assert_allclose(daily["ic"], expected["ic"], equal_nan=True)
    ic = summary["arms"]["spy"]["summary"]["metrics"]["avg_ic"]
    assert ic == pytest.approx(float(np.nanmean(expected["ic"])), abs=1e-12)
    assert {row["control"] for row in summary["paired_daily_difference"]} == {"spy"}


def test_the_candidate_gets_the_runs_own_training_labels(run, spy_registered, tmp_path):
    score_candidates.score_candidate(run, "spy", out_dir=tmp_path / "spy")
    saved = rescore_run.resolve_run_config(run)
    scoring = rescore_run.rebuild_scoring_inputs(saved.config)
    data, _ = score_candidates.stock_level_data(
        saved.config, saved.metadata, saved.recorded, scoring
    )
    np.testing.assert_array_equal(SEEN["train_labels"], np.asarray(data["train_labels"]))
    assert SEEN["train_dates"] == [str(d) for d in data["train_dates"]]
    assert SEEN["train_dates"][-1] < saved.config.data.val_start


def test_the_run_is_untouched_and_an_earlier_result_is_kept_unless_forced(run, spy_registered):
    copy = Path(run).parent / "copy_for_force"
    shutil.copytree(run, copy)
    before = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file()}
    path = score_candidates.score_candidate(copy, "spy")
    assert path == copy / "candidates" / "spy" / SUMMARY_FILENAME
    after = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file()}
    assert {k: after[k] for k in before} == before
    with pytest.raises(FileExistsError):
        score_candidates.score_candidate(copy, "spy")
    score_candidates.score_candidate(copy, "spy", force=True)


def test_the_model_arm_cannot_be_named_as_a_candidate(run):
    with pytest.raises(CandidateError, match="run's own predictions"):
        score_candidates.score_candidate(run, "model")


def test_the_residual_linear_model_scores_a_real_run_end_to_end(run, tmp_path):
    path = score_candidates.score_candidate(run, "residual_linear", out_dir=tmp_path / "rl")
    summary = _summary(path)
    controls = summary["controls"]["candidate"]
    assert set(controls["coefficients"]) == set(controls["signals"])
    assert summary["arms"]["residual_linear"]["coverage"]["share_scored"] > 0.9


# --- Pooling windows ----------------------------------------------------------------------


def _folder(root: Path, window: str, arms: dict[str, list[float]], dates: list[str]) -> Path:
    folder = root / f"{window}_{'_'.join(sorted(arms))}"
    (folder / "daily_ic").mkdir(parents=True)
    for arm, values in arms.items():
        pd.DataFrame({"dt": dates, "ic": values, "rank_ic": values}).to_csv(
            folder / "daily_ic" / f"{arm}.csv", index=False
        )
    (folder / SUMMARY_FILENAME).write_text(
        json.dumps({"run": {"window": window, "label_t": 5, "test_dates": dates[:1]}}),
        encoding="utf-8",
    )
    return folder


def test_the_pooled_ic_is_the_mean_of_the_windows_joined(tmp_path):
    w1 = ["2021-01-25", "2021-01-26", "2021-01-27"]
    w2 = ["2022-01-24", "2022-01-25", "2022-01-26"]
    folders = [
        _folder(tmp_path, "W1", {"model": [0.1, 0.0, 0.2], "cand": [0.3, 0.1, 0.2]}, w1),
        _folder(tmp_path, "W2", {"model": [0.0, -0.1, 0.1], "cand": [0.2, 0.0, 0.1]}, w2),
        _folder(tmp_path, "W1", {"model": [0.1, 0.0, 0.2], "base": [0.0, 0.0, 0.3]}, w1),
    ]
    report = pool_candidates.pool(folders)
    arms = {a["arm"]: a for a in report["arms"]}
    assert arms["model"]["ic"]["mean"] == pytest.approx(np.mean([0.1, 0, 0.2, 0, -0.1, 0.1]))
    assert arms["cand"]["ic"]["n_dates"] == 6
    assert arms["cand"]["ic"]["newey_west_lags"] == 3
    assert arms["base"]["windows"] == ["W1"]
    assert arms["cand"]["per_window"]["W2"]["ic"] == pytest.approx(0.1)
    paired = {(r["arm"], r["metric"]): r for r in report["paired_vs_reference"]}
    assert paired[("cand", "ic")]["mean_delta"] == pytest.approx(
        np.mean([0.2, 0.1, 0, 0.2, 0.1, 0])
    )
    assert paired[("base", "ic")]["n_common_dates"] == 3
    assert "| cand |" in pool_candidates.markdown(report)


def test_folders_that_scored_different_runs_are_refused(tmp_path):
    w1 = ["2021-01-25", "2021-01-26"]
    folders = [
        _folder(tmp_path, "W1", {"model": [0.1, 0.0], "cand": [0.3, 0.1]}, w1),
        _folder(tmp_path, "W1", {"model": [0.1, 0.5], "base": [0.0, 0.0]}, w1),
    ]
    with pytest.raises(ValueError, match="did not score the same run"):
        pool_candidates.pool(folders)


def test_an_arm_scored_twice_on_a_window_is_refused(tmp_path):
    w1 = ["2021-01-25", "2021-01-26"]
    a = _folder(tmp_path / "a", "W1", {"model": [0.1, 0.0], "cand": [0.3, 0.1]}, w1)
    b = _folder(tmp_path / "b", "W1", {"model": [0.1, 0.0], "cand": [0.3, 0.1]}, w1)
    with pytest.raises(ValueError, match="scored twice"):
        pool_candidates.pool([a, b])


def test_the_newey_west_t_reproduces_the_protocol_tables_figure(tmp_path):
    """Bartlett weights on label_t - 2 lags, the five-window table's convention."""
    rng = np.random.default_rng(5)
    values = rng.normal(0.01, 0.1, 300)
    dates = list(pd.bdate_range("2021-01-25", periods=300).strftime("%Y-%m-%d"))
    folder = _folder(tmp_path, "W1", {"model": list(values)}, dates)
    report = pool_candidates.pool([folder])
    lags = 3
    x = values - values.mean()
    var = x @ x / len(x)
    for lag in range(1, lags + 1):
        var += 2 * (1 - lag / (lags + 1)) * (x[lag:] @ x[:-lag]) / len(x)
    expected = values.mean() / np.sqrt(var / len(x))
    assert report["arms"][0]["ic"]["t_newey_west"] == pytest.approx(expected, rel=1e-10)


def test_pit_intervals_are_read_as_the_pipeline_reads_them(tmp_path):
    path = tmp_path / "pit.csv"
    pd.DataFrame(
        {
            "kdcode": ["A.N", "B.N"],
            "valid_from": ["1/4/2016", "2/1/2016"],
            "valid_to": ["2019-07-30", ""],
        }
    ).to_csv(path, index=False)
    frame = score_candidates.read_pit_intervals(str(path), "2026-07-31")
    assert frame.to_dict("records") == [
        {"kdcode": "A.N", "valid_from": "2016-01-04", "valid_to": "2019-07-30"},
        {"kdcode": "B.N", "valid_from": "2016-02-01", "valid_to": "2026-07-31"},
    ]
    assert score_candidates.read_pit_intervals(None, "2026-07-31") is None
