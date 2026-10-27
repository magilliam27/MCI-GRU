"""Simple baselines are scored on a saved run's own axis by the run's own evaluation (#360).

``scripts/score_baselines.py`` rebuilds a saved run's labels, dates, names and PIT
tradable mask (as the re-score does, #337), scores reversal, 12-1 momentum and a
ridge on the run's stock-level inputs with ``compute_evaluation_summary``, and
compares the run with each, day by day.

These tests run ``run_experiment.py`` in-process on small synthetic panels (one with a
PIT universe, a missing bar, a joiner and a leaver; one without PIT), each with more
than a year of history so momentum can be scored, and pin that:

* the run's own predictions, scored by the script, give back every stored metric,
  and a run whose IC does not reproduce writes nothing;
* every baseline is scored only where the run could rank, through the same summary,
  and its dated IC agrees with its summary;
* the linear model's rebuilt inputs must carry the run's recorded normalisation, and
  the global columns are the only ones it leaves out;
* results go to a new folder and an earlier one is kept unless forced.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

import run_experiment
from mci_gru.config import ExperimentConfig
from mci_gru.evaluation.dated_ic import dated_ic_frame
from mci_gru.evaluation.metrics import SUMMARY_SCHEMA_VERSION
from mci_gru.features import FeatureEngineer
from mci_gru.training import ensemble
from scripts import rescore_run, score_baselines
from scripts.rescore_run import RescoreError
from scripts.score_baselines import (
    DEFAULT_OUT_DIRNAME,
    SUMMARY_FILENAME,
    BaselineError,
    check_stock_level_inputs,
    stock_level_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STOCKS = ["A.N", "B.N", "C.N", "D.N", "E.N", "F.N"]
MISSING_BAR = ("C.N", "2020-03-10")
BASELINES = ["reversal_1w", "reversal_1m", "momentum_12_1", "linear_ridge"]
OVERRIDES = [
    "features=base",
    "data.source=csv",
    "data.input_package_manifest=null",
    "data.input_package_manifest_sha256=null",
    "data.input_package_root=null",
    "data.train_start=2020-01-01",
    "data.train_end=2020-02-14",
    "data.val_start=2020-02-18",
    "data.val_end=2020-02-28",
    "data.test_start=2020-03-03",
    "data.test_end=2020-03-20",
    "model.his_t=3",
    "model.label_t=3",
    "model.gru_hidden_sizes=[8,4]",
    "model.hidden_size_gat1=8",
    "model.output_gat1=4",
    "model.gat_heads=1",
    "model.hidden_size_gat2=8",
    "model.num_hidden_states=4",
    "model.cross_attn_heads=1",
    "model.use_self_attention=false",
    "training.num_epochs=1",
    "training.num_models=2",
    "training.batch_size=2",
    "training.warmup_steps=0",
    "tracking.enabled=false",
    "evaluation.top_k_values=[2,4]",
    "graph.corr_lookback_days=20",
]


def _write_panel(path: Path) -> None:
    # From 2019-01-01: more than 252 sessions before the test window opens.
    dates = pd.bdate_range("2019-01-01", periods=320).strftime("%Y-%m-%d")
    rng = np.random.default_rng(11)
    rows = []
    for s_idx, kdcode in enumerate(STOCKS):
        price = 100.0 + s_idx
        for d_idx, dt in enumerate(dates):
            price *= 1.0 + rng.normal(0.0, 0.01)
            if (kdcode, dt) == MISSING_BAR:
                continue
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt,
                    "open": price * (1.0 + rng.normal(0.0, 0.002)),
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000.0 + d_idx + 1000.0 * rng.random(),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pit(path: Path) -> None:
    # D.N leaves and E.N joins inside the test window; the rest are open-ended.
    pd.DataFrame(
        [
            {"kdcode": "A.N", "valid_from": "2019-01-01", "valid_to": ""},
            {"kdcode": "B.N", "valid_from": "2019-01-01", "valid_to": ""},
            {"kdcode": "C.N", "valid_from": "2019-01-01", "valid_to": ""},
            {"kdcode": "D.N", "valid_from": "2019-01-01", "valid_to": "2020-03-11"},
            {"kdcode": "E.N", "valid_from": "2020-03-09", "valid_to": ""},
            {"kdcode": "F.N", "valid_from": "2019-01-01", "valid_to": ""},
        ]
    ).to_csv(path, index=False)


@contextmanager
def _restored_process_state(directory: Path):
    root = logging.getLogger()
    handlers, level, cwd = list(root.handlers), root.level, os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


def _unrounded_rows(predictions, kdcode_list, date, prediction_mask=None):
    """``prediction_rows_for_date`` without its 5-decimal rounding (see test_rescore_run)."""
    mask = np.ones(len(kdcode_list), bool) if prediction_mask is None else prediction_mask
    return [
        [kdcode, date, float(score)]
        for kdcode, score, keep in zip(kdcode_list, predictions, mask, strict=True)
        if keep and np.isfinite(score)
    ]


def _run(base: Path, *, use_pit: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    panel = base / "panel.csv"
    _write_panel(panel)
    overrides = [*OVERRIDES, f"data.filename={panel.as_posix()}"]
    if use_pit:
        _write_pit(base / "pit.csv")
        overrides += [
            "data.use_pit_universe=true",
            f"data.pit_universe_csv={(base / 'pit.csv').as_posix()}",
            "data.pit_export_cutoff=2020-03-31",
            "data.pit_min_scoreable_stocks=1",
            "data.pit_breadth_policy=off",
        ]
    else:
        overrides.append("data.use_pit_universe=false")
    with initialize_config_dir(config_dir=str(REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose("config", overrides=overrides)
    out = base / "run"
    out.mkdir()
    patch = pytest.MonkeyPatch()
    patch.setattr(torch.cuda, "is_available", lambda: False)
    patch.setattr(ensemble, "prediction_rows_for_date", _unrounded_rows)
    try:
        with _restored_process_state(out):
            run_experiment.main.__wrapped__(cfg)
    finally:
        patch.undo()
    windows = {p.parent for p in out.rglob("evaluation_summary.json")}
    assert len(windows) == 1
    return windows.pop()


@pytest.fixture(scope="module")
def pit_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("baselines") / "pit", use_pit=True)


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("baselines") / "plain", use_pit=False)


@pytest.fixture
def run_copy(pit_run, tmp_path) -> Path:
    """A private copy of the PIT run; its recorded inputs stay where the run read them."""
    return Path(shutil.copytree(pit_run, tmp_path / "run"))


@pytest.fixture(scope="module")
def pit_result(pit_run, tmp_path_factory) -> Path:
    """One scoring of the PIT run, into a folder of its own; the run is not touched."""
    out = tmp_path_factory.mktemp("pit_result") / DEFAULT_OUT_DIRNAME
    return score_baselines.score_baselines(pit_run, out_dir=out)


def _summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rebuilt_scoring(run: Path):
    saved = rescore_run.resolve_run_config(run)
    return rescore_run.rebuild_scoring_inputs(saved.config)


# --- The run itself comes back -------------------------------------------------------


def test_the_runs_own_predictions_give_back_every_stored_metric(pit_run, pit_result):
    stored = _summary(pit_run / "evaluation_summary.json")["metrics"]
    model = _summary(pit_result)["arms"]["model"]["summary"]["metrics"]

    assert set(model) == set(stored)
    for key, value in stored.items():
        assert model[key] == pytest.approx(value, abs=1e-12, nan_ok=True), key


def test_a_run_whose_ic_does_not_reproduce_writes_nothing(run_copy, monkeypatch):
    real = rescore_run.compute_labels

    def shifted(panel, kdcodes, dates, label_t, **kwargs):
        return np.roll(real(panel, kdcodes, dates, label_t, **kwargs), 1, axis=0)

    monkeypatch.setattr(rescore_run, "compute_labels", shifted)

    with pytest.raises(RescoreError, match="avg_ic"):
        score_baselines.score_baselines(run_copy, include_linear=False)
    assert not (run_copy / DEFAULT_OUT_DIRNAME).exists()
    assert not list(run_copy.glob(f".{DEFAULT_OUT_DIRNAME}*"))


# --- Every baseline goes through the run's evaluation --------------------------------


def test_every_baseline_is_scored_by_the_runs_summary_where_the_run_could_rank(pit_run, pit_result):
    summary = _summary(pit_result)
    scoring = _rebuilt_scoring(pit_run)
    rankable = pd.DataFrame(scoring.rankable_mask, index=scoring.dates, columns=scoring.kdcodes)

    assert list(summary["arms"]) == ["model", *BASELINES]
    for arm in BASELINES:
        entry = summary["arms"][arm]
        assert entry["summary"]["schema_version"] == SUMMARY_SCHEMA_VERSION
        assert entry["summary"]["benchmark"] == "equal_weight_pit_tradable_universe"
        assert entry["coverage"]["scored_cells"] > 0, arm
        scores = pd.read_csv(pit_result.parent / "scores" / f"{arm}.csv", dtype={"dt": str})
        assert len(scores) == entry["coverage"]["scored_cells"]
        assert all(rankable.loc[row.dt, row.kdcode] for row in scores.itertuples()), arm
        # Same universe benchmark as the run: every arm is ranked on the run's mask.
        assert entry["summary"]["metrics"]["universe_abs_period_return"] == pytest.approx(
            summary["arms"]["model"]["summary"]["metrics"]["universe_abs_period_return"],
            abs=1e-15,
        )


def test_each_arms_dated_ic_agrees_with_its_summary(pit_result):
    summary = _summary(pit_result)
    for arm, entry in summary["arms"].items():
        daily = pd.read_csv(pit_result.parent / "daily_ic" / f"{arm}.csv", dtype={"dt": str})
        metrics = entry["summary"]["metrics"]
        assert daily["ic"].mean() == pytest.approx(metrics["avg_ic"], abs=1e-12), arm
        assert daily["rank_ic"].mean() == pytest.approx(metrics["avg_rank_ic"], abs=1e-12), arm


def test_the_reversal_scores_are_minus_the_last_weeks_return_from_the_panel(pit_run, pit_result):
    scoring = _rebuilt_scoring(pit_run)
    closes = scoring.panel.pivot(index="dt", columns="kdcode", values="close")
    sessions = list(closes.index)
    scores = pd.read_csv(pit_result.parent / "scores" / "reversal_1w.csv", dtype={"dt": str})

    for row in scores.itertuples():
        at = sessions.index(row.dt)
        past = closes.loc[sessions[at - 5], row.kdcode]
        assert row.score == pytest.approx(-(closes.loc[row.dt, row.kdcode] / past - 1.0))


def _scores(result: Path, arm: str, scoring) -> np.ndarray:
    frame = pd.read_csv(result.parent / "scores" / f"{arm}.csv", dtype={"dt": str})
    wide = frame.pivot(index="dt", columns="kdcode", values="score")
    return wide.reindex(index=scoring.dates, columns=scoring.kdcodes).to_numpy()


def test_the_run_is_compared_with_each_baseline_on_the_same_names_day_by_day(pit_run, pit_result):
    summary = _summary(pit_result)
    rows = summary["paired_daily_difference"]
    scoring = _rebuilt_scoring(pit_run)
    model = _scores(pit_result, "model", scoring)

    assert [(r["metric"], r["control"]) for r in rows] == [
        (metric, arm) for metric in ("ic", "rank_ic") for arm in BASELINES
    ]
    left_out = {}
    for row in rows:
        control = _scores(pit_result, row["control"], scoring)
        # The run is scored only on the cells the baseline scored.
        matched = dated_ic_frame(
            np.where(np.isfinite(control), model, np.nan), scoring.labels, scoring.dates
        )
        both = matched.merge(
            dated_ic_frame(control, scoring.labels, scoring.dates), on="dt"
        ).dropna()
        delta = both[f"{row['metric']}_x"] - both[f"{row['metric']}_y"]
        assert row["arm"] == "model"
        assert row["n_common_dates"] == len(both)
        assert row["mean_delta"] == pytest.approx(delta.mean(), abs=1e-12)
        assert row["hac_lags"] == 1  # label_t 3: consecutive outcomes share one session
        left_out[row["control"]] = row["model_cells_left_out"]
    # The missing bar leaves C.N unscored by the one-week reversal a week later.
    assert left_out["reversal_1w"] == 1


def test_a_run_without_pit_is_scored_on_every_name_with_a_close(plain_run, tmp_path):
    path = score_baselines.score_baselines(plain_run, out_dir=tmp_path / "out")
    summary = _summary(path)

    for arm in ["model", *BASELINES]:
        entry = summary["arms"][arm]
        assert entry["summary"]["benchmark"] == "equal_weight_rankable_universe"
        assert entry["coverage"]["scored_cells"] > 0, arm


# --- The linear model sees the run's inputs ------------------------------------------


def test_the_linear_models_inputs_carry_the_runs_normalisation(pit_result):
    controls = _summary(pit_result)["controls"]["linear_inputs"]

    assert controls["normalisation_matches_run"] is True
    assert controls["global_columns_left_out"] == []
    assert controls["ridge_alpha"] in controls["ridge_alphas"]
    assert controls["inputs_per_stock"] == 3 * len(controls["stock_level_columns"])
    assert controls["train_rows"] > 0


def test_inputs_not_normalised_as_the_run_was_are_refused(run_copy):
    path = run_copy / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    column = next(iter(metadata["norm_means"]))
    metadata["norm_means"][column] *= 1.0 + 1e-9
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(BaselineError, match=f"norm_means\\[{column}\\]"):
        score_baselines.score_baselines(run_copy)
    assert not (run_copy / DEFAULT_OUT_DIRNAME).exists()


def test_only_the_global_columns_are_left_out_of_the_linear_models_inputs():
    config = ExperimentConfig()
    config.features.include_global_regime = True
    config.features.include_vix = True
    run_columns = FeatureEngineer(config.features).get_feature_columns()
    stock = stock_level_config(config)
    stock_columns = FeatureEngineer(stock.features).get_feature_columns()
    global_columns = set(run_columns) - set(stock_columns)
    metadata = {
        "feature_cols": run_columns,
        "norm_means": dict.fromkeys(run_columns, 0.5),
        "norm_stds": dict.fromkeys(run_columns, 2.0),
    }
    norms = dict.fromkeys(stock_columns, 0.5), dict.fromkeys(stock_columns, 2.0)

    assert global_columns and not stock.features.include_global_regime
    assert stock.data.auxiliary_snapshot_mode == "source"
    controls = check_stock_level_inputs(metadata, stock_columns, global_columns, *norms)
    assert controls["global_columns_left_out"] == [c for c in run_columns if c in global_columns]

    # A stock-level column missing from the rebuild is refused, not taken as global.
    with pytest.raises(BaselineError, match="stock-level columns"):
        check_stock_level_inputs(metadata, stock_columns[1:], global_columns, *norms)


# --- A new folder, never over an old one ---------------------------------------------


def test_results_go_to_a_new_folder_and_the_run_is_untouched(run_copy):
    before = {p.relative_to(run_copy): p.read_bytes() for p in run_copy.rglob("*") if p.is_file()}

    path = score_baselines.score_baselines(run_copy, include_linear=False)

    after = {p.relative_to(run_copy): p.read_bytes() for p in run_copy.rglob("*") if p.is_file()}
    assert path == run_copy / DEFAULT_OUT_DIRNAME / SUMMARY_FILENAME
    assert {k: after[k] for k in before} == before
    assert all(k.parts[0] == DEFAULT_OUT_DIRNAME for k in set(after) - set(before))
    summary = _summary(path)
    assert summary["controls"]["linear_inputs"] is None
    assert "linear_ridge" not in summary["arms"]
    assert len(summary["code"]["git_commit"]) == 40


def test_an_earlier_result_is_kept_unless_forced(run_copy):
    path = score_baselines.score_baselines(run_copy, include_linear=False)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    earlier = path.read_bytes()

    with pytest.raises(FileExistsError):
        score_baselines.score_baselines(run_copy, include_linear=False)
    assert path.read_bytes() == earlier

    score_baselines.score_baselines(run_copy, include_linear=False, force=True)
    assert path.read_bytes() != earlier


def test_the_command_line_scores_a_run_and_prints_each_arm(run_copy, capsys):
    assert score_baselines.main(["--run_dir", str(run_copy), "--no_linear"]) == 0

    out = capsys.readouterr().out
    for arm in ["model", "reversal_1w", "reversal_1m", "momentum_12_1"]:
        assert f"{arm}: IC" in out
    assert "model - momentum_12_1 (rank_ic)" in out


def test_inputs_staged_at_another_path_are_read_from_there(tmp_path):
    """How Colab runs it: the run's data and PIT files copied elsewhere, the originals gone."""
    run = _run(tmp_path / "moved", use_pit=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in ("panel.csv", "pit.csv"):
        shutil.move(tmp_path / "moved" / name, staged / name)

    path = score_baselines.score_baselines(
        run,
        data_file=str(staged / "panel.csv"),
        pit_universe_csv=str(staged / "pit.csv"),
    )

    summary = _summary(path)
    assert summary["controls"]["linear_inputs"]["normalisation_matches_run"] is True
    assert set(summary["inputs"]) == {"data.filename", "data.pit_universe_csv"}


def test_force_replaces_only_an_earlier_baseline_result(run_copy, tmp_path):
    other = tmp_path / "other_work"
    other.mkdir()
    (other / "keep.txt").write_text("not a baseline result", encoding="utf-8")

    with pytest.raises(BaselineError, match="not one"):
        score_baselines.score_baselines(run_copy, out_dir=other, include_linear=False, force=True)
    assert (other / "keep.txt").read_text(encoding="utf-8") == "not a baseline result"

    for target in (run_copy, run_copy.parent):
        with pytest.raises(BaselineError, match="holds the run"):
            score_baselines.score_baselines(
                run_copy, out_dir=target, include_linear=False, force=True
            )
    assert (run_copy / "run_metadata.json").is_file()
