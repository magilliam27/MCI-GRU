"""Input admission at the confirmed #223 seams: native read, preparation, runner.

Every proof goes through a public boundary with real native parsing: the
``prepare_data`` / ``prepare_data_index_level`` entry points, or the unchanged
``run_experiment.py`` command. Fixtures are synthetic and providers are off.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from mci_gru import pipeline
from mci_gru.config import create_config_from_dict
from mci_gru.data import path_resolver
from mci_gru.data.quality_contract import (
    RUN_FAILURE_SCHEMA,
    AdmissionError,
    AdmissionItem,
    AdmissionLedger,
    Verdict,
)
from mci_gru.features import FeatureEngineer
from mci_gru.pipeline import prepare_data, prepare_data_index_level

REPO_ROOT = Path(__file__).resolve().parents[1]
DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-01", periods=70)]
COLUMNS = "kdcode,dt,open,high,low,close,volume"


def _control_rows() -> dict[tuple[str, str], list[str]]:
    rows = {}
    for stock, s in [("AAA", 0), ("BBB", 20)]:
        for j, dt in enumerate(DATES):
            rows[(stock, dt)] = [
                stock,
                dt,
                f"{100 + s + j}",
                f"{101 + s + j}",
                f"{99 + s + j}",
                f"{100.5 + s + j}",
                f"{1000 + j}",
            ]
    return rows


def _serialise(rows: dict[tuple[str, str], list[str]], extra: list[list[str]] = ()) -> bytes:
    # Reverse date order: in-memory chronological processing must not be
    # mistaken for rewriting the file.
    body = [",".join(row) for _, row in sorted(rows.items(), key=lambda kv: kv[0][1], reverse=True)]
    body += [",".join(row) for row in extra]
    return ("\n".join([COLUMNS, *body]) + "\n").encode()


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _config(filename: str, **data):
    return create_config_from_dict(
        {
            "data": {
                "filename": filename,
                "use_pit_universe": False,
                "train_start": "2020-01-01",
                "train_end": "2020-02-14",
                "val_start": "2020-02-18",
                "val_end": "2020-02-28",
                "test_start": "2020-03-03",
                "test_end": "2020-03-20",
                **data,
            },
            "features": {"include_momentum": False, "include_weekly_momentum": False},
            "model": {"his_t": 3, "label_t": 2},
            "graph": {"use_multi_feature_edges": False},
            "training": {"label_type": "returns"},
            "tracking": {"enabled": False},
        }
    )


@pytest.fixture
def feature_calls(monkeypatch):
    """Count feature work and auxiliary loading; neither may follow a failed verdict."""
    calls = []
    engineer, auxiliary = pipeline.engineer_features, pipeline.load_auxiliary_data

    def spy_engineer(*args, **kwargs):
        calls.append("engineer_features")
        return engineer(*args, **kwargs)

    def spy_auxiliary(*args, **kwargs):
        calls.append("load_auxiliary_data")
        return auxiliary(*args, **kwargs)

    monkeypatch.setattr(pipeline, "engineer_features", spy_engineer)
    monkeypatch.setattr(pipeline, "load_auxiliary_data", spy_auxiliary)
    return calls


def _prepare_fails(config) -> AdmissionError:
    with pytest.raises(AdmissionError) as caught:
        prepare_data(config, FeatureEngineer(config.features))
    return caught.value


def _codes(error: AdmissionError) -> set[str]:
    return {item.reason_code for item in error.failures}


# ── frozen-history predicate ──────────────────────────────────────────────


def test_a_stock_with_all_four_ohlc_histories_constant_stops_preparation(tmp_path, feature_calls):
    rows = _control_rows()
    for dt in DATES:
        rows[("AAA", dt)][2:6] = ["100", "101", "99", "100"]
    content = _serialise(rows)
    source = _write(tmp_path / "panel.csv", content)

    error = _prepare_fails(_config(str(source)))

    assert _codes(error) == {"frozen_ohlc_history"}
    (finding,) = error.failures
    assert finding.verdict is Verdict.INVALID
    assert finding.evidence["count"] == 1
    (stock,) = finding.evidence["stocks"]
    assert stock["kdcode"] == "AAA"
    assert all(
        stock[f"{f}_n"] == 70 and stock[f"{f}_u"] == 1 for f in ("open", "high", "low", "close")
    )
    assert feature_calls == []
    assert source.read_bytes() == content
    # The failure carries the actual read, not a later re-resolution.
    (observation,) = error.input_observations.to_dict()["observations"]
    assert observation["identity"]["sha256"] == hashlib.sha256(content).hexdigest()


def test_the_varying_control_is_admitted_and_returns_prepared_data(tmp_path):
    source = _write(tmp_path / "panel.csv", _serialise(_control_rows()))
    config = _config(str(source))

    data = prepare_data(config, FeatureEngineer(config.features))

    assert data["admission"]["admitted"] is True
    assert not [i for i in data["admission"]["items"] if i["rule"] == "frozen_history"]
    assert data["kdcode_list"] == ["AAA", "BBB"]


def test_one_constant_field_is_reported_but_does_not_invalidate_the_history(tmp_path):
    rows = _control_rows()
    for j, dt in enumerate(DATES):
        rows[("AAA", dt)][2:6] = ["100", f"{102 + j / 100}", f"{99 - j / 100}", f"{101 + j / 100}"]
    source = _write(tmp_path / "panel.csv", _serialise(rows))
    config = _config(str(source))

    data = prepare_data(config, FeatureEngineer(config.features))

    (reported,) = [i for i in data["admission"]["items"] if i["rule"] == "frozen_history"]
    assert reported["verdict"] == "valid"
    assert reported["reason_code"] == "constant_field_reported"
    (stock,) = reported["evidence"]["stocks"]
    assert (stock["kdcode"], stock["open_u"], stock["close_u"]) == ("AAA", 1, 70)


def test_fewer_than_two_observations_is_insufficient_evidence_and_stops(tmp_path, feature_calls):
    rows = _control_rows()
    for dt in DATES[1:]:
        rows[("AAA", dt)][2:6] = ["", "", "", ""]
    source = _write(tmp_path / "panel.csv", _serialise(rows))

    error = _prepare_fails(_config(str(source)))

    (finding,) = error.failures
    assert finding.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert finding.reason_code == "fewer_than_two_observations"
    assert [s["kdcode"] for s in finding.evidence["stocks"]] == ["AAA"]
    # Blank cells are genuine missingness: counted, never failed as malformed.
    assert error.admission["coverage"]["data.filename"]["per_stock"]["AAA"] == {
        "rows_with_missing_price": 69,
        "absent_sessions_within_history": 0,
    }
    assert feature_calls == []


def test_missing_prices_are_counted_per_stock_without_a_budget(tmp_path):
    rows = _control_rows()
    rows[("BBB", DATES[10])][5] = ""
    del rows[("BBB", DATES[20])]
    source = _write(tmp_path / "panel.csv", _serialise(rows))
    config = _config(str(source))

    data = prepare_data(config, FeatureEngineer(config.features))

    coverage = data["admission"]["coverage"]["data.filename"]
    assert coverage["per_stock"] == {
        "BBB": {"rows_with_missing_price": 1, "absent_sessions_within_history": 1}
    }
    assert data["admission"]["admitted"] is True


# ── market structure ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        (None, None, "duplicate_stock_session"),
        (1, "2020-02-30", "invalid_date"),
        (2, "oops", "non_numeric_open"),
        (5, "0", "out_of_range_close"),
        (6, "-1", "out_of_range_volume"),
        (0, " ", "blank_kdcode"),
    ],
)
def test_malformed_rows_stop_preparation_and_are_never_repaired(
    tmp_path, feature_calls, column, value, code
):
    rows = _control_rows()
    first = rows[("AAA", DATES[0])]
    extra = []
    if column is None:
        extra = [list(first)]
    else:
        first[column] = value
    content = _serialise(rows, extra)
    source = _write(tmp_path / "panel.csv", content)

    error = _prepare_fails(_config(str(source)))

    assert code in _codes(error)
    finding = next(i for i in error.failures if i.reason_code == code)
    assert finding.stage == "validate"
    assert finding.evidence["count"] >= 1
    assert feature_calls == []
    assert source.read_bytes() == content


def test_an_unparseable_file_fails_at_parse_with_its_bytes_observed(tmp_path):
    content = f'{COLUMNS}\n"AAA,2020-01-01,100,101,99,100,1000\n'.encode()
    source = _write(tmp_path / "panel.csv", content)

    error = _prepare_fails(_config(str(source)))

    (finding,) = error.failures
    assert (finding.role, finding.stage, finding.reason_code) == (
        "data.filename",
        "parse",
        "parse_failed",
    )
    (observation,) = error.input_observations.to_dict()["observations"]
    assert observation["outcome"] == "error"
    assert observation["identity"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert error.input_observations.uses == ()


# ── selected-file resolution ──────────────────────────────────────────────


def test_a_missing_selected_file_is_not_substituted_by_a_same_named_decoy(
    tmp_path, monkeypatch, feature_calls
):
    root = tmp_path / "project"
    decoy_bytes = _serialise(_control_rows())
    decoy = _write(root / "data" / "raw" / "market" / "panel.csv", decoy_bytes)
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", root)
    selected = tmp_path / "selected" / "panel.csv"

    error = _prepare_fails(_config(str(selected)))

    (finding,) = error.failures
    assert (finding.role, finding.stage, finding.reason_code) == (
        "data.filename",
        "resolve",
        "missing_file",
    )
    assert finding.configured_path == str(selected)
    assert error.input_observations.to_dict()["observations"] == []
    assert decoy.read_bytes() == decoy_bytes
    assert feature_calls == []


def test_a_missing_index_file_is_not_substituted_either(tmp_path, monkeypatch):
    root = tmp_path / "project"
    _write(root / "data" / "raw" / "market" / "index.csv", b"dt,close\n2020-01-01,1\n")
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", root)
    config = _config(str(tmp_path / "unused.csv"), index_filename=str(tmp_path / "x" / "index.csv"))

    with pytest.raises(AdmissionError) as caught:
        prepare_data_index_level(config, FeatureEngineer(config.features))

    (finding,) = caught.value.failures
    assert (finding.role, finding.stage) == ("data.index_filename", "resolve")


# ── unresolved required verdicts ──────────────────────────────────────────


def test_an_unresolved_required_verdict_cannot_return_admitted_data(tmp_path, feature_calls):
    source = _write(tmp_path / "panel.csv", _serialise(_control_rows()))
    config = _config(str(source))
    admission = AdmissionLedger()
    pending = AdmissionItem(
        role="regime",
        rule="historical_validity",
        verdict=Verdict.UNRESOLVED,
        reason_code="verdict_pending",
        reason="A required verdict has not been made",
    )
    admission.record(pending)

    with pytest.raises(AdmissionError) as caught:
        prepare_data(config, FeatureEngineer(config.features), admission=admission)

    assert caught.value.failures == [pending]
    assert feature_calls == []


def test_an_unresolved_optional_item_does_not_block(tmp_path):
    source = _write(tmp_path / "panel.csv", _serialise(_control_rows()))
    config = _config(str(source))
    admission = AdmissionLedger()
    admission.record(
        AdmissionItem(
            role="sector",
            rule="coverage",
            verdict=Verdict.UNRESOLVED,
            reason_code="deferred",
            reason="Disabled role",
            required=False,
        )
    )

    data = prepare_data(config, FeatureEngineer(config.features), admission=admission)

    assert data["admission"]["admitted"] is True


# ── PIT interval structure and breadth ────────────────────────────────────


def _pit_config(tmp_path, pit_rows: list[str], **data):
    source = _write(tmp_path / "panel.csv", _serialise(_control_rows()))
    pit = tmp_path / "pit.csv"
    pit.write_text("kdcode,valid_from,valid_to\n" + "\n".join(pit_rows) + "\n", encoding="utf-8")
    settings = {
        "use_pit_universe": True,
        "pit_universe_csv": str(pit),
        "pit_min_scoreable_stocks": 0,
        **data,
    }
    return _config(str(source), **settings)


def test_a_blank_valid_to_is_membership_through_the_export_cutoff(tmp_path):
    config = _pit_config(
        tmp_path,
        ["AAA,2020-01-01,", "BBB,2020-01-01,2020-03-31"],
        pit_export_cutoff="2020-03-31",
    )

    data = prepare_data(config, FeatureEngineer(config.features))

    assert data["kdcode_list"] == ["AAA", "BBB"]
    assert data["test_active_member_mask"][:, 0].all()


def test_a_blank_valid_to_without_a_declared_cutoff_stops(tmp_path, feature_calls):
    config = _pit_config(tmp_path, ["AAA,2020-01-01,", "BBB,2020-01-01,2020-03-31"])

    error = _prepare_fails(config)

    assert _codes(error) == {"open_interval_without_cutoff"}
    assert error.failures[0].evidence["rows"][0]["kdcode"] == "AAA"
    assert feature_calls == ["load_auxiliary_data"]


@pytest.mark.parametrize(
    ("pit_rows", "code", "named"),
    [
        (["AAA,2020-03-31,2020-01-01", "BBB,2020-01-01,2020-03-31"], "inverted_interval", "AAA"),
        (
            ["AAA,2020-01-01,2020-02-14", "AAA,2020-02-14,2020-03-31", "BBB,2020-01-01,2020-03-31"],
            "overlapping_intervals",
            "AAA",
        ),
        (
            ["AAA,2020-01-01,2020-03-31", "BBB,2020-01-01,2020-03-31", "CCC,2020-02-01,2020-03-31"],
            "pit_names_without_panel_rows",
            "CCC",
        ),
    ],
)
def test_pit_structure_failures_stop_and_name_the_stock(
    tmp_path, feature_calls, pit_rows, code, named
):
    error = _prepare_fails(_pit_config(tmp_path, pit_rows))

    assert _codes(error) == {code}
    evidence = error.failures[0].evidence
    names = evidence.get("kdcodes") or [row["kdcode"] for row in evidence["rows"]]
    assert named in names
    assert "engineer_features" not in feature_calls


def test_adjacent_intervals_for_one_name_are_not_an_overlap(tmp_path):
    config = _pit_config(
        tmp_path,
        ["AAA,2020-01-01,2020-02-13", "AAA,2020-02-14,2020-03-31", "BBB,2020-01-01,2020-03-31"],
    )

    data = prepare_data(config, FeatureEngineer(config.features))

    assert data["kdcode_list"] == ["AAA", "BBB"]


def test_the_session_breadth_floor_stops_through_the_failure_report(tmp_path):
    config = _pit_config(
        tmp_path,
        ["AAA,2020-01-01,2020-03-31", "BBB,2020-01-01,2020-03-31"],
        pit_min_scoreable_stocks=3,
        pit_breadth_policy="error",
    )

    error = _prepare_fails(config)

    assert _codes(error) == {"breadth_below_floor"}
    assert isinstance(error, ValueError)
    assert error.failures[0].evidence["min_scoreable_stocks"] == 3
    assert error.input_observations is not None


# ── composition root: run_experiment.py ───────────────────────────────────

BOOTSTRAP = textwrap.dedent(
    """
    import json, runpy, sys
    from pathlib import Path

    project_root, calls_path, *argv = sys.argv[1:]
    sys.path.insert(0, {repo!r})
    from mci_gru.data import fred_loader, lseg_loader, path_resolver
    import mci_gru.data.data_manager as data_manager
    import mci_gru.models as models
    import mci_gru.training as training

    path_resolver.PROJECT_ROOT = Path(project_root)
    calls = []

    class Guarded(RuntimeError):
        pass

    def guard(name):
        def tripped(*args, **kwargs):
            calls.append(name)
            Path(calls_path).write_text(json.dumps(calls))
            raise Guarded(name)
        return tripped

    fred_loader.FREDLoader.__init__ = guard("provider_setup")
    lseg_loader.LSEGLoader.__init__ = guard("provider_setup")
    data_manager.create_data_loaders = guard("create_data_loaders")
    models.create_model = guard("create_model")
    training.train_multiple_models = guard("train_multiple_models")
    Path(calls_path).write_text(json.dumps(calls))
    sys.argv = [{runner!r}, *argv]
    runpy.run_path({runner!r}, run_name="__main__")
    """
).format(repo=str(REPO_ROOT), runner=str(REPO_ROOT / "run_experiment.py"))


def _run_cli(tmp_path: Path, selected: Path, project_root: Path):
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text(BOOTSTRAP, encoding="utf-8")
    run_dir = tmp_path / "run"
    calls = tmp_path / "calls.json"
    command = [
        sys.executable,
        str(bootstrap),
        str(project_root),
        str(calls),
        "features=base",
        "data.source=csv",
        f"data.filename={selected.as_posix()}",
        "data.use_pit_universe=false",
        "data.train_start=2020-01-01",
        "data.train_end=2020-02-14",
        "data.val_start=2020-02-18",
        "data.val_end=2020-02-28",
        "data.test_start=2020-03-03",
        "data.test_end=2020-03-20",
        "model.his_t=3",
        "model.label_t=2",
        "training.num_epochs=1",
        "training.num_models=1",
        "tracking.enabled=false",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    completed = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=300
    )
    return completed, run_dir, json.loads(calls.read_text())


def _assert_stopped_before_training(completed, run_dir, calls):
    assert completed.returncode != 0
    assert calls == []
    for artifact in (
        "run_metadata.json",
        "graph_data.pt",
        "training_summary.json",
        "admission.json",
        "averaged_predictions",
        "checkpoints",
    ):
        assert not (run_dir / artifact).exists(), artifact
    report = json.loads((run_dir / "run_failure.json").read_text(encoding="utf-8"))
    assert report["schema"] == RUN_FAILURE_SCHEMA
    assert (report["outcome"], report["stage"]) == ("failed", "preparation")
    assert report["resolved_config"]["resolved_config_sha256"]
    return report


@pytest.mark.slow
def test_the_runner_reports_a_missing_selected_file_and_never_reads_the_decoy(tmp_path):
    root = tmp_path / "project"
    decoy_bytes = _serialise(_control_rows())
    decoy = _write(root / "data" / "raw" / "market" / "panel.csv", decoy_bytes)
    selected = tmp_path / "selected" / "panel.csv"

    completed, run_dir, calls = _run_cli(tmp_path, selected, root)

    report = _assert_stopped_before_training(completed, run_dir, calls)
    (failure,) = report["failures"]
    assert (failure["role"], failure["source"], failure["stage"], failure["reason_code"]) == (
        "data.filename",
        "file",
        "resolve",
        "missing_file",
    )
    assert failure["configured_path"] == selected.as_posix()
    assert report["input_observations"]["observations"] == []
    output = completed.stdout + completed.stderr
    assert "role=data.filename source=file stage=resolve reason=missing_file" in output
    assert "run_failure.json" in output
    assert decoy.read_bytes() == decoy_bytes


@pytest.mark.slow
def test_the_runner_reports_malformed_rows_with_the_identity_of_the_read(tmp_path):
    rows = _control_rows()
    content = _serialise(rows, [list(rows[("AAA", DATES[0])])])
    selected = _write(tmp_path / "selected" / "panel.csv", content)

    completed, run_dir, calls = _run_cli(tmp_path, selected, tmp_path / "project")

    report = _assert_stopped_before_training(completed, run_dir, calls)
    (failure,) = report["failures"]
    assert (failure["stage"], failure["reason_code"]) == ("validate", "duplicate_stock_session")
    (observation,) = report["input_observations"]["observations"]
    assert observation["identity"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert "reason=duplicate_stock_session" in completed.stdout + completed.stderr


@pytest.mark.slow
def test_the_runner_control_passes_admission_and_reaches_the_guarded_trainer_path(tmp_path):
    """Proves the guards are live: a valid panel gets past preparation to them."""
    selected = _write(tmp_path / "selected" / "panel.csv", _serialise(_control_rows()))

    completed, run_dir, calls = _run_cli(tmp_path, selected, tmp_path / "project")

    assert completed.returncode != 0
    assert calls == ["create_data_loaders"]
    assert not (run_dir / "run_failure.json").exists()
    admission = json.loads((run_dir / "admission.json").read_text(encoding="utf-8"))
    assert admission["admitted"] is True
