"""A saved run is re-scored from its own inputs, without retraining (#337).

The first run's numbers were computed by evaluation code that has since been fixed.
Its averaged predictions and its data package are enough to score it again, so
``scripts/rescore_run.py`` rebuilds the labels, the PIT tradable mask and the panel
from the run's recorded inputs and writes a new summary beside the old one.

These tests run ``run_experiment.py`` in-process on small synthetic panels (one with a
PIT universe, a missing bar, a joiner, a leaver and a dated cessation; one without
PIT; each with returns and rank labels) and pin that:

* re-scoring reproduces the stored IC, rank IC and every other stored metric, and a
  re-score whose IC does not reproduce writes nothing;
* the result is a new file carrying the code commit and the current schema, the old
  summary and daily IC are untouched, and an earlier re-score is not overwritten
  unless forced;
* an input whose bytes differ from what the run consumed is refused: the data file,
  a saved prediction file, and the cessation file the replay's checks do not cover.
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
from mci_gru.evaluation.metrics import SUMMARY_SCHEMA_VERSION
from mci_gru.evaluation.replay_inputs import ReplayInputError
from mci_gru.training import ensemble
from scripts import rescore_run
from scripts.rescore_run import (
    RESCORED_DAILY_IC_FILENAME,
    RESCORED_SUMMARY_FILENAME,
    RescoreError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STOCKS = ["A.N", "B.N", "C.N", "D.N", "E.N", "F.N"]
MISSING_BAR = ("C.N", "2020-03-10")
OVERRIDES = [
    "features=base",
    "protocol.exploratory=true",
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
    dates = pd.bdate_range("2019-11-01", periods=105).strftime("%Y-%m-%d")
    rng = np.random.default_rng(5)
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
                    "volume": 1_000_000.0 + d_idx,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pit(path: Path) -> None:
    # D.N leaves and E.N joins inside the test window; the rest are open-ended.
    pd.DataFrame(
        [
            {"kdcode": "A.N", "valid_from": "2019-11-01", "valid_to": ""},
            {"kdcode": "B.N", "valid_from": "2019-11-01", "valid_to": ""},
            {"kdcode": "C.N", "valid_from": "2019-11-01", "valid_to": ""},
            {"kdcode": "D.N", "valid_from": "2019-11-01", "valid_to": "2020-03-11"},
            {"kdcode": "E.N", "valid_from": "2020-03-09", "valid_to": ""},
            {"kdcode": "F.N", "valid_from": "2019-11-01", "valid_to": ""},
        ]
    ).to_csv(path, index=False)


def _write_cessation(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "event_id": "F-cessation-1",
                "kdcode": "F.N",
                "effective_at": "2020-03-13T21:00:00Z",
                "known_from": "2020-03-13T20:00:00Z",
                "acquired_at": "2026-10-24T00:00:00Z",
                "evidence": "synthetic fixture",
            }
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
    """``prediction_rows_for_date`` without its 5-decimal rounding."""
    mask = np.ones(len(kdcode_list), bool) if prediction_mask is None else prediction_mask
    return [
        [kdcode, date, float(score)]
        for kdcode, score, keep in zip(kdcode_list, predictions, mask, strict=True)
        if keep and np.isfinite(score)
    ]


def _run(base: Path, *, use_pit: bool, label_type: str = "returns") -> Path:
    """One in-process run, its averaged scores saved at full precision.

    The runner rounds saved scores to 5 decimals. A one-epoch model on six names scores
    some of them nearly alike, so rounding can tie names the run ranked apart and move
    rank IC well past the script's 1e-4 tolerance; whether it does depends on the
    platform's arithmetic (CI's two jobs disagreed with this machine). Saving full
    precision pins the rebuild itself, exactly. The tolerance exists for real runs,
    where an ensemble over ~110 names makes such ties rare; #338 applies it to the
    first run.
    """
    base.mkdir(parents=True, exist_ok=True)
    panel = base / "panel.csv"
    _write_panel(panel)
    overrides = [
        *OVERRIDES,
        f"data.filename={panel.as_posix()}",
        f"training.label_type={label_type}",
    ]
    if use_pit:
        _write_pit(base / "pit.csv")
        _write_cessation(base / "cessation.csv")
        overrides += [
            "data.use_pit_universe=true",
            f"data.pit_universe_csv={(base / 'pit.csv').as_posix()}",
            f"data.pit_cessation_events_csv={(base / 'cessation.csv').as_posix()}",
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
    return out


@pytest.fixture(scope="module")
def pit_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("rescore") / "pit", use_pit=True)


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("rescore") / "plain", use_pit=False)


@pytest.fixture(scope="module")
def pit_rank_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("rescore") / "pit_rank", use_pit=True, label_type="rank")


@pytest.fixture(scope="module")
def plain_rank_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("rescore") / "plain_rank", use_pit=False, label_type="rank")


ALL_RUNS = ["pit_run", "plain_run", "pit_rank_run", "plain_rank_run"]


@pytest.fixture
def run_copy(pit_run, tmp_path) -> Path:
    """A private copy of the PIT run; its recorded inputs stay where the run read them."""
    return Path(shutil.copytree(pit_run, tmp_path / "run"))


def _metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def _rescored_files(run: Path) -> list[Path]:
    return [
        p for p in (run / RESCORED_SUMMARY_FILENAME, run / RESCORED_DAILY_IC_FILENAME) if p.exists()
    ]


# --- The re-score reproduces the run -------------------------------------------------


@pytest.mark.parametrize("fixture", ALL_RUNS)
def test_rescoring_reproduces_the_stored_ic_and_rank_ic(fixture, request, tmp_path):
    run = Path(shutil.copytree(request.getfixturevalue(fixture), tmp_path / "run"))

    path = rescore_run.rescore_run(run)

    stored, rescored = _metrics(run / "evaluation_summary.json"), _metrics(path)
    assert rescored["avg_ic"] == pytest.approx(stored["avg_ic"], abs=1e-12)
    assert rescored["avg_rank_ic"] == pytest.approx(stored["avg_rank_ic"], abs=1e-12)
    assert rescored["test_dates"] == stored["test_dates"]
    assert rescored["ic_valid_dates"] == stored["ic_valid_dates"]


@pytest.mark.parametrize("fixture", ALL_RUNS)
def test_every_stored_metric_comes_back_when_the_code_has_not_changed(fixture, request, tmp_path):
    """The run was scored by this code, so the portfolio numbers must come back too.

    That pins the rebuilt panel, dates and rankable mask, not only the labels.
    """
    run = Path(shutil.copytree(request.getfixturevalue(fixture), tmp_path / "run"))

    stored, rescored = (
        _metrics(run / "evaluation_summary.json"),
        _metrics(rescore_run.rescore_run(run)),
    )

    assert set(rescored) == set(stored)
    for key, value in stored.items():
        assert rescored[key] == pytest.approx(value, abs=1e-12, nan_ok=True), key


def test_the_pit_run_is_scored_on_its_rebuilt_tradable_mask(run_copy):
    """The fixture's leaver, joiner, missing bar and cessation all move the mask."""
    metadata = json.loads((run_copy / "run_metadata.json").read_text())
    config = rescore_run.create_config_from_dict(
        {
            **json.loads((run_copy / "resolved_config.json").read_text()),
            "data": {
                **json.loads((run_copy / "resolved_config.json").read_text())["data"],
                "filename": metadata["data_inputs"]["data.filename"]["resolved_path"],
                "pit_universe_csv": metadata["data_inputs"]["data.pit_universe_csv"][
                    "resolved_path"
                ],
                "pit_cessation_events_csv": metadata["data_inputs"][
                    "data.pit_cessation_events_csv"
                ]["resolved_path"],
            },
        }
    )
    scoring = rescore_run.rebuild_scoring_inputs(config)
    tradable = pd.DataFrame(scoring.rankable_mask, index=scoring.dates, columns=scoring.kdcodes)
    saved = rescore_run.load_saved_predictions(
        run_copy / "averaged_predictions", scoring.kdcodes, scoring.dates
    )

    assert not tradable.loc["2020-03-12", "D.N"]  # left after 2020-03-11
    assert not tradable.loc["2020-03-06", "E.N"]  # joins 2020-03-09
    assert not tradable.loc["2020-03-10", "C.N"]  # no bar that day
    assert not tradable.loc["2020-03-16", "F.N"] and tradable.loc["2020-03-12", "F.N"]
    # The run saved a score exactly where the rebuilt mask allows one.
    np.testing.assert_array_equal(np.isfinite(saved), scoring.rankable_mask)


def test_a_rescore_whose_ic_does_not_reproduce_writes_nothing(run_copy, monkeypatch):
    real = rescore_run.compute_labels

    def shifted(panel, kdcodes, dates, label_t, **kwargs):
        return np.roll(real(panel, kdcodes, dates, label_t, **kwargs), 1, axis=0)

    monkeypatch.setattr(rescore_run, "compute_labels", shifted)

    with pytest.raises(RescoreError, match="avg_ic"):
        rescore_run.rescore_run(run_copy)
    assert _rescored_files(run_copy) == []


# --- A new file beside the old one ---------------------------------------------------


def test_the_rescore_is_a_new_file_and_the_run_is_untouched(run_copy):
    before = {p.relative_to(run_copy): p.read_bytes() for p in run_copy.rglob("*") if p.is_file()}

    path = rescore_run.rescore_run(run_copy)

    after = {p.relative_to(run_copy): p.read_bytes() for p in run_copy.rglob("*") if p.is_file()}
    assert path == run_copy / RESCORED_SUMMARY_FILENAME
    assert set(after) - set(before) == {
        Path(RESCORED_SUMMARY_FILENAME),
        Path(RESCORED_DAILY_IC_FILENAME),
    }
    assert {k: after[k] for k in before} == before
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    record = summary["rescore"]
    commit = record["code"]["git_commit"]
    assert isinstance(commit, str) and len(commit) == 40
    assert record["stored_summary"]["path"] == "evaluation_summary.json"
    assert set(record["inputs"]) == {
        "data.filename",
        "data.pit_universe_csv",
        "data.pit_cessation_events_csv",
    }
    assert record["not_rescored"] == ["linked_score_ties"]


def test_the_dated_ic_means_are_the_rescored_summary_means(run_copy):
    metrics = _metrics(rescore_run.rescore_run(run_copy))
    daily = pd.read_csv(run_copy / RESCORED_DAILY_IC_FILENAME, dtype={"dt": str})

    assert daily["ic"].mean() == pytest.approx(metrics["avg_ic"], abs=1e-12)
    assert daily["rank_ic"].mean() == pytest.approx(metrics["avg_rank_ic"], abs=1e-12)
    assert daily["dt"].tolist() == sorted(
        p.stem for p in (run_copy / "averaged_predictions").glob("*.csv")
    )


def test_an_earlier_rescore_is_kept_unless_forced(run_copy):
    rescore_run.rescore_run(run_copy)
    earlier = {path: path.read_bytes() + b"\n" for path in _rescored_files(run_copy)}
    for path, content in earlier.items():
        path.write_bytes(content)

    with pytest.raises(FileExistsError):
        rescore_run.rescore_run(run_copy)
    assert {path: path.read_bytes() for path in _rescored_files(run_copy)} == earlier

    rescore_run.rescore_run(run_copy, force=True)
    assert all(path.read_bytes() != content for path, content in earlier.items())


# --- Inputs that are not the run's are refused ----------------------------------------


def _altered_copy(source: Path, target: Path) -> Path:
    # A trailing blank line parses to the same frame, so only the digest can tell.
    target.write_bytes(source.read_bytes() + b"\n")
    return target


def test_a_data_file_with_other_bytes_is_refused(run_copy, tmp_path):
    metadata = json.loads((run_copy / "run_metadata.json").read_text())
    source = Path(metadata["data_inputs"]["data.filename"]["resolved_path"])
    altered = _altered_copy(source, tmp_path / "panel.csv")

    with pytest.raises(ReplayInputError, match="data.filename"):
        rescore_run.rescore_run(run_copy, data_file=str(altered))
    assert _rescored_files(run_copy) == []


def test_a_cessation_file_with_other_bytes_is_refused(run_copy, tmp_path):
    metadata = json.loads((run_copy / "run_metadata.json").read_text())
    source = Path(metadata["data_inputs"]["data.pit_cessation_events_csv"]["resolved_path"])
    altered = _altered_copy(source, tmp_path / "cessation.csv")

    with pytest.raises(RescoreError, match="data.pit_cessation_events_csv"):
        rescore_run.rescore_run(run_copy, cessation_events_csv=str(altered))
    assert _rescored_files(run_copy) == []


def test_a_saved_prediction_changed_after_the_run_is_refused(run_copy):
    target = sorted((run_copy / "averaged_predictions").glob("*.csv"))[0]
    frame = pd.read_csv(target)
    frame["score"] = frame["score"][::-1].to_numpy()
    frame.to_csv(target, index=False)

    with pytest.raises(ReplayInputError, match="receipt"):
        rescore_run.rescore_run(run_copy)
    assert _rescored_files(run_copy) == []


def test_the_command_line_rescores_a_run_and_prints_both_ics(run_copy, capsys):
    assert rescore_run.main(["--run_dir", str(run_copy)]) == 0

    out = capsys.readouterr().out
    assert "avg_ic: stored" in out and "avg_rank_ic: stored" in out
    assert (run_copy / RESCORED_SUMMARY_FILENAME).is_file()


def test_a_stored_summary_changed_after_the_run_is_refused(run_copy):
    stored = run_copy / "evaluation_summary.json"
    summary = json.loads(stored.read_text())
    summary["metrics"]["avg_ic"] += 0.01
    stored.write_text(json.dumps(summary))

    with pytest.raises(RescoreError, match="receipt"):
        rescore_run.rescore_run(run_copy)
    assert _rescored_files(run_copy) == []


def test_a_failed_summary_write_leaves_no_dated_ic_behind(run_copy, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rescore_run, "write_json_artifact", fail)

    with pytest.raises(OSError, match="disk full"):
        rescore_run.rescore_run(run_copy)
    assert _rescored_files(run_copy) == []
    assert not list(run_copy.glob(".*pending"))


def test_evaluation_code_from_another_checkout_is_refused(run_copy, monkeypatch, tmp_path):
    monkeypatch.setattr(rescore_run.mci_gru, "__file__", str(tmp_path / "mci_gru" / "__init__.py"))

    with pytest.raises(RescoreError, match="not from"):
        rescore_run.rescore_run(run_copy)
    assert _rescored_files(run_copy) == []
