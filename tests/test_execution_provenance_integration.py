"""Execution evidence from ensemble members and from the runner (#144).

Member proof: two tiny synthetic CPU members run through ``train_multiple_models``
against a retained execution-start record. Readback must use only the retained
evidence: planned seeds or a checkpoint filename never prove that a member ran.

Runner proof: ``run_experiment.py`` runs in-process on a synthetic panel. Each
window gets its own immutable start, its metadata references that start, and a
separate terminal receipt binds the start, member events, metadata, checkpoints
and outputs. A run is complete only when a run receipt binds every window the
plan expected; a missing event or receipt never reads as completion.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

import run_experiment
from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from mci_gru.evaluation import execution_provenance
from mci_gru.evaluation.artifacts import canonical_json_bytes
from mci_gru.evaluation.execution_provenance import (
    BACKEND_OBSERVATIONS,
    ExecutionReference,
    MemberEventLog,
    capture_execution_start,
    read_execution_provenance,
    read_execution_run,
    read_member_execution,
    write_execution_plan,
    write_window_receipt,
)
from mci_gru.evaluation.experiment_summary import write_resolved_config
from mci_gru.evaluation.run_input_attachments import AttachmentReference, read_run_inputs
from mci_gru.training import ensemble
from mci_gru.training.ensemble import train_multiple_models
from mci_gru.training.trainer import NoCheckpointSelectedError, Trainer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
N_STOCKS = 4
KDCODES = ["AAA", "BBB", "CCC", "DDD"]
DATES = ["2025-01-10", "2025-01-13"]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N_STOCKS) * 0.01)

    def forward(self, time_series, *args, **kwargs):
        del args, kwargs
        return self.weight.unsqueeze(0).expand(time_series.shape[0], -1)


class NaNModel(TinyModel):
    def __init__(self) -> None:
        super().__init__()
        nn.init.constant_(self.weight, float("nan"))


def _loader():
    labels = [[0.20, -0.10, 0.00, 0.10], [0.10, 0.00, -0.10, 0.20]]
    return [
        (
            torch.zeros((1, N_STOCKS, 3, 2)),
            torch.tensor([labels[i]]),
            torch.zeros((1, N_STOCKS, 2)),
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            N_STOCKS,
            [DATES[i]],
        )
        for i in range(len(DATES))
    ]


def _config(tmp_path: Path, *, num_models: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        seed=73,
        training=TrainingConfig(
            loss_type="mse",
            selection_metric="val_loss",
            num_epochs=2,
            num_models=num_models,
            batch_size=2,
            learning_rate=1e-3,
            lr_scheduler="none",
            use_amp=True,
        ),
        experiment_name="member_execution",
        output_dir=str(tmp_path / "run"),
    )


def _start(tmp_path: Path, config: ExperimentConfig) -> ExecutionReference:
    repo = tmp_path / "repo"
    (repo / "mci_gru").mkdir(parents=True)
    (repo / "run_experiment.py").write_bytes(b"SEED = 73\n")
    (repo / "mci_gru" / "model.py").write_bytes(b"WIDTH = 4\n")
    identity = write_resolved_config(config, tmp_path / "config")
    return capture_execution_start(
        repo,
        tmp_path / "config" / identity["resolved_config_path"],
        resolved_config_sha256=identity["resolved_config_sha256"],
        output_dir=tmp_path / "evidence",
        window_id="0",
    )


def _train(tmp_path, config, reference, *, model_factory=TinyModel):
    return train_multiple_models(
        model_factory=model_factory,
        config=config,
        train_loader=_loader(),
        val_loader=_loader(),
        test_loader=_loader(),
        kdcode_list=KDCODES,
        test_dates=DATES,
        output_path=str(tmp_path / "run"),
        execution=reference,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_members_record_what_actually_ran_and_survive_relocation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    reference = _start(tmp_path, config)
    results, _ = _train(tmp_path, config, reference)

    report = read_member_execution(reference)
    assert report.status == "complete"
    assert report.expected_members == 2
    assert [member["model_id"] for member in report.members] == [0, 1]
    for model_id, member in enumerate(report.members):
        assert member["status"] == "complete", member["problems"]
        assert member["problems"] == []
        assert member["planned_seed"] == member["applied_seed"] == 73 + model_id
        assert member["torch_initial_seed"] == {"status": "observed", "value": 73 + model_id}
        runtime = member["runtime"]
        assert runtime["device"] == {"status": "observed", "value": "cpu"}
        # Requested AMP is recorded separately from what actually ran on CPU.
        assert (runtime["amp_requested"], runtime["amp_effective"]) == (True, False)
        assert runtime["parameter_dtype"] == {"status": "observed", "value": "torch.float32"}
        assert set(runtime["backend"]) == {
            "cuda_available",
            "cudnn_version",
            "cudnn_enabled",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "deterministic_algorithms",
            "float32_matmul_precision",
        }
        checkpoint = tmp_path / "run" / "checkpoints" / f"model_{model_id}_best.pth"
        assert member["checkpoint"]["saved_sha256"] == _sha256(checkpoint)
        assert member["checkpoint"]["loaded_sha256"] == _sha256(checkpoint)
    # #111's optional co-metrics are untouched by recording.
    assert all(result.best_val_loss is not None for result in results)

    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "evidence", moved)
    shutil.rmtree(tmp_path / "evidence")
    shutil.rmtree(tmp_path / "run")
    relocated = ExecutionReference(
        moved / reference.path.relative_to(tmp_path / "evidence"), reference.sha256
    )

    def unavailable(*args, **kwargs):
        raise AssertionError("Readback must not observe the live runtime")

    monkeypatch.setattr(torch, "initial_seed", unavailable)
    monkeypatch.setattr(torch.cuda, "is_available", unavailable)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", unavailable)
    config.seed = 999
    config.training.num_models = 5
    assert read_member_execution(relocated) == report


def test_failure_after_seeding_is_explicit_and_never_complete(tmp_path):
    config = _config(tmp_path)
    reference = _start(tmp_path, config)
    calls = iter([TinyModel, None])

    def factory():
        build = next(calls)
        if build is None:
            raise RuntimeError("fixture: model construction failed")
        return build()

    with pytest.raises(RuntimeError, match="model construction failed"):
        _train(tmp_path, config, reference, model_factory=factory)

    report = read_member_execution(reference)
    assert report.status == "failed"
    assert [member["status"] for member in report.members] == ["complete", "failed"]
    failed = report.members[1]
    assert failed["applied_seed"] == 74
    assert failed["error_type"] == "RuntimeError"
    assert failed["runtime"] is None


def test_a_member_killed_before_finishing_leaves_an_incomplete_attempt(tmp_path):
    config = _config(tmp_path)
    reference = _start(tmp_path, config)
    # The process dies in member two after its checkpoint is saved and loaded:
    # everything but completion is recorded, and nothing may stand in for it.
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(REPOSITORY_ROOT)!r})
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from pathlib import Path
        import test_execution_provenance_integration as fixture
        from mci_gru.evaluation.execution_provenance import ExecutionReference
        from mci_gru.training.trainer import Trainer

        calls = []
        original = Trainer.save_predictions

        def dies_on_member_two(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                os._exit(17)
            return original(self, *args, **kwargs)

        Trainer.save_predictions = dies_on_member_two
        tmp = Path({str(tmp_path)!r})
        reference = ExecutionReference(Path({str(reference.path)!r}), {reference.sha256!r})
        fixture._train(tmp, fixture._config(tmp), reference)
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 17, completed.stderr

    report = read_member_execution(reference)
    assert report.status == "incomplete"
    assert [member["status"] for member in report.members] == ["complete", "incomplete"]
    killed = report.members[1]
    assert killed["problems"] == ["member did not finish"]
    assert killed["checkpoint"]["saved_sha256"] == killed["checkpoint"]["loaded_sha256"]


def test_a_checkpoint_changed_after_saving_is_not_the_selected_evidence(tmp_path, monkeypatch):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    original_load = Trainer.load_best_model

    def replace_checkpoint(self, best_model_path=None):
        torch.save(TinyModel().state_dict(), best_model_path)
        return original_load(self, best_model_path)

    monkeypatch.setattr(Trainer, "load_best_model", replace_checkpoint)
    _train(tmp_path, config, reference)

    member = read_member_execution(reference).members[0]
    assert member["status"] == "incomplete"
    assert member["problems"] == ["loaded checkpoint differs from the last one saved"]


def test_load_evidence_describes_the_bytes_loaded_not_the_file_afterwards(tmp_path, monkeypatch):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    original_load = Trainer.load_best_model

    def load_then_replace(self, best_model_path=None):
        loaded = original_load(self, best_model_path)
        torch.save(TinyModel().state_dict(), best_model_path)
        return loaded

    monkeypatch.setattr(Trainer, "load_best_model", load_then_replace)
    _train(tmp_path, config, reference)

    member = read_member_execution(reference).members[0]
    assert member["status"] == "complete", member["problems"]


def test_training_with_a_config_other_than_the_retained_one_is_not_complete(tmp_path):
    reference = _start(tmp_path, _config(tmp_path, num_models=1))
    drifted = _config(tmp_path, num_models=1)
    drifted.seed = 80
    _train(tmp_path, drifted, reference)

    member = read_member_execution(reference).members[0]
    assert (member["planned_seed"], member["applied_seed"]) == (73, 80)
    assert member["problems"] == ["applied seed differs from the plan"]


def test_an_unfinished_final_event_write_keeps_the_attempt_incomplete(tmp_path):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    _train(tmp_path, config, reference)
    events = reference.path.with_name(reference.path.stem + ".events.jsonl")
    assert read_member_execution(reference).status == "complete"
    with events.open("ab") as handle:
        handle.write(b'{"schema":"mci_gru.execution_member_event.v1","att')
    assert read_member_execution(reference).status == "incomplete"


@pytest.mark.parametrize(
    "forged",
    [
        [(0, "seeded"), (1, "seeded")],
        [(0, "seeded"), (0, "training_started"), (0, "member_completed")],
        [(0, "training_started")],
        [(1, "seeded")],
    ],
)
def test_events_out_of_execution_order_are_rejected_even_when_chained(tmp_path, forged):
    config = _config(tmp_path)
    reference = _start(tmp_path, config)
    events = MemberEventLog(reference)
    payloads = {
        "seeded": {"seed": 73, "torch_initial_seed": {"status": "observed", "value": 73}},
        "training_started": {
            "amp_requested": False,
            "amp_effective": False,
            "device": {"status": "observed", "value": "cpu"},
            "parameter_dtype": {"status": "observed", "value": "torch.float32"},
            "backend": {
                name: {"status": "unknown", "value": None, "reason": "fixture"}
                for name in BACKEND_OBSERVATIONS
            },
        },
        "member_completed": {},
    }
    for model_id, event in forged:
        events.record(model_id, event, payloads[event])
    with pytest.raises(ValueError, match="member events"):
        read_member_execution(reference)


def test_a_checkpoint_left_by_an_earlier_run_is_not_this_attempts_evidence(tmp_path):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    stale = tmp_path / "run" / "checkpoints" / "model_0_best.pth"
    stale.parent.mkdir(parents=True)
    torch.save(TinyModel().state_dict(), stale)

    # A NaN validation loss never improves on the initial best, so nothing is
    # saved and training refuses to fall back to the stale file.
    with pytest.raises(NoCheckpointSelectedError):
        _train(tmp_path, config, reference, model_factory=NaNModel)

    report = read_member_execution(reference)
    member = report.members[0]
    assert report.status == "failed"
    assert member["status"] == "failed"
    assert member["error_type"] == "NoCheckpointSelectedError"
    assert member["checkpoint"] == {"saved_sha256": None, "loaded_sha256": None}


def test_a_checkpoint_saved_without_being_recorded_is_not_complete(tmp_path, monkeypatch):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    original_train = Trainer.train

    def unrecorded_saves(self, *args, execution_callback=None, **kwargs):
        def record(event, data):
            if event != "checkpoint_saved":
                execution_callback(event, data)

        return original_train(self, *args, execution_callback=record, **kwargs)

    monkeypatch.setattr(Trainer, "train", unrecorded_saves)
    _train(tmp_path, config, reference)

    member = read_member_execution(reference).members[0]
    assert member["status"] == "incomplete"
    assert member["checkpoint"]["saved_sha256"] is None
    assert "no checkpoint saved in this attempt" in member["problems"]


def test_a_missing_checkpoint_at_load_is_explicit(tmp_path, monkeypatch):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    original_load = Trainer.load_best_model

    def lose_checkpoint(self, best_model_path=None):
        os.remove(best_model_path)
        return original_load(self, best_model_path)

    monkeypatch.setattr(Trainer, "load_best_model", lose_checkpoint)
    _train(tmp_path, config, reference)

    member = read_member_execution(reference).members[0]
    assert member["status"] == "incomplete"
    assert member["checkpoint"]["loaded_sha256"] is None
    assert "checkpoint missing at load" in member["problems"]


def test_unavailable_runtime_observations_are_explicit_and_block_completion(tmp_path, monkeypatch):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)

    def broken(*args, **kwargs):
        raise RuntimeError("fixture probe failure")

    monkeypatch.setattr(torch, "initial_seed", broken)
    monkeypatch.setattr(torch.backends.cudnn, "version", broken)
    _train(tmp_path, config, reference)

    member = read_member_execution(reference).members[0]
    assert member["torch_initial_seed"] == {
        "status": "unavailable",
        "value": None,
        "reason": "RuntimeError",
    }
    assert member["runtime"]["backend"]["cudnn_version"]["status"] == "unavailable"
    assert member["status"] == "incomplete"
    assert "seed application not observed" in member["problems"]


def test_an_unobserved_device_blocks_completion(tmp_path, monkeypatch):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    # The recorder finds the device through the model's first parameter; hide it
    # from the recorder only, so training itself runs normally.
    monkeypatch.setattr(ensemble, "next", lambda iterator, default: default, raising=False)
    _train(tmp_path, config, reference)

    member = read_member_execution(reference).members[0]
    assert member["runtime"]["device"] == {
        "status": "unavailable",
        "value": None,
        "reason": "AttributeError",
    }
    assert member["status"] == "incomplete"
    assert member["problems"] == ["device not observed"]


def test_a_member_that_never_started_is_counted_against_the_planned_total(tmp_path):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, _config(tmp_path / "planned", num_models=2))
    _train(tmp_path, config, reference)

    report = read_member_execution(reference)
    assert report.expected_members == 2
    assert [member["status"] for member in report.members] == ["complete"]
    assert report.status == "incomplete"


@pytest.mark.parametrize(
    "damage",
    ["edit_event", "drop_middle_event", "reorder", "foreign_attempt", "noncanonical"],
)
def test_altered_member_events_are_rejected(tmp_path, damage):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    _train(tmp_path, config, reference)
    events = reference.path.with_name(reference.path.stem + ".events.jsonl")
    lines = events.read_bytes().splitlines()
    # The chain protects every event but the last; the last is checked on its own.
    if damage == "edit_event":
        entry = json.loads(lines[0])
        entry["data"]["seed"] = 1
        lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    elif damage == "drop_middle_event":
        del lines[1]
    elif damage == "reorder":
        lines[0], lines[1] = lines[1], lines[0]
    elif damage == "foreign_attempt":
        entry = json.loads(lines[-1])
        entry["attempt_id"] = "0" * 32
        lines[-1] = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    elif damage == "noncanonical":
        lines[-1] = lines[-1].replace(b":", b": ", 1)
    events.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ValueError, match="member events"):
        read_member_execution(reference)


def test_recording_refuses_a_second_writer_for_the_same_attempt(tmp_path):
    config = _config(tmp_path, num_models=1)
    reference = _start(tmp_path, config)
    _train(tmp_path, config, reference)
    with pytest.raises(ValueError, match="already"):
        _train(tmp_path, config, reference)


# --- Runner proof ---------------------------------------------------------------

RUNNER_OVERRIDES = [
    "features=base",
    "data.source=csv",
    "data.use_pit_universe=false",
    "data.train_start=2020-01-01",
    "data.train_end=2020-02-14",
    "data.val_start=2020-02-18",
    "data.val_end=2020-02-28",
    "data.test_start=2020-03-03",
    "data.test_end=2020-03-20",
    "model.his_t=3",
    "model.label_t=2",
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
]
TWO_WINDOWS = [
    "data.test_end=2021-06-30",
    "training.walkforward.enabled=true",
    "training.walkforward.window_train_years=1",
    "training.walkforward.window_val_months=1",
    "training.walkforward.test_span_months=1",
    "training.walkforward.step_months=1",
    "training.walkforward.max_windows=2",
]


def _write_panel(path: Path, *, days: int) -> None:
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows = []
    for s_idx, stock in enumerate(["A.N", "B.N", "C.N", "D.N"]):
        price = 100.0 + s_idx
        for d_idx, dt in enumerate(dates):
            open_px = price * (1.0 + 0.002 * np.sin(d_idx / 4.0 + s_idx))
            close_px = open_px * (1.0 + 0.001 * (s_idx + 1) * np.cos(d_idx / 7.0))
            rows.append(
                {
                    "kdcode": stock,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": open_px,
                    "high": max(open_px, close_px) * 1.001,
                    "low": min(open_px, close_px) * 0.999,
                    "close": close_px,
                    "volume": 1_000_000.0 + d_idx,
                }
            )
            price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)


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


def _run(base: Path, overrides: list[str], *, days: int = 60) -> Path:
    """Run the experiment runner in-process; the run's output root is returned."""
    base.mkdir(parents=True, exist_ok=True)
    panel = base / "panel.csv"
    _write_panel(panel, days=days)
    out = base / "out"
    out.mkdir()
    with initialize_config_dir(config_dir=str(REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose(
            "config",
            overrides=[*RUNNER_OVERRIDES, f"data.filename={panel.as_posix()}", *overrides],
        )
    with _restored_process_state(out):
        try:
            run_experiment.main.__wrapped__(cfg)
        except Exception as error:
            raise _RunFailed(out) from error
    return out


class _RunFailed(Exception):
    def __init__(self, out: Path) -> None:
        super().__init__(str(out))
        self.out = out


def _run_failing(base: Path, overrides: list[str], **kwargs) -> tuple[Path, BaseException]:
    with pytest.raises(_RunFailed) as failure:
        _run(base, overrides, **kwargs)
    return failure.value.out, failure.value.__cause__


def _plan(out: Path) -> ExecutionReference:
    (path,) = (out / "execution_provenance").glob("*.plan.json")
    return ExecutionReference(path, _sha256(path))


def _window_files(out: Path, window: dict) -> tuple[Path, Path, Path]:
    evidence = out / window["output_dir"] / "execution_provenance"
    attempt = window["attempt_id"]
    return (
        evidence / f"{attempt}.json",
        evidence / f"{attempt}.events.jsonl",
        evidence / f"{attempt}.receipt.json",
    )


@pytest.fixture(scope="module")
def stock_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("stock_run"), [])


@pytest.fixture(scope="module")
def two_window_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("two_window_run"), TWO_WINDOWS, days=400)


@pytest.fixture
def stock_copy(stock_run, tmp_path) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(stock_run, copy)
    return copy


def test_a_stock_run_links_its_start_metadata_members_and_outputs(stock_run):
    plan = _plan(stock_run)
    run = read_execution_run(plan)
    assert (run.status, run.problems, run.experiment_mode) == ("complete", [], "stock_level")
    assert run.expected_windows == 1
    (window,) = run.windows
    assert (window["walkforward_window"], window["window_id"], window["output_dir"]) == (
        0,
        "0",
        ".",
    )
    assert (window["status"], window["problems"]) == ("complete", [])
    assert window["members"].status == "complete"
    assert window["members"].expected_members == 2

    start_path, events_path, receipt_path = _window_files(stock_run, window)
    start = read_execution_provenance(ExecutionReference(start_path, window["start_sha256"]))
    # Captured after the effective config was serialized, and bound to this run's plan.
    assert start.record["window_id"] == "0"
    assert start.record["plan"] == {"run_id": run.run_id, "sha256": plan.sha256}
    assert start.resolved_config_bytes == (stock_run / "resolved_config.json").read_bytes()
    assert start.record["execution_status"] == "incomplete"

    metadata = json.loads((stock_run / "run_metadata.json").read_text())
    assert metadata["walkforward_window"] == 0
    assert metadata["execution_start"] == {
        "path": f"execution_provenance/{window['attempt_id']}.json",
        "sha256": window["start_sha256"],
    }

    receipt = json.loads(receipt_path.read_bytes())
    assert receipt_path.read_bytes() == canonical_json_bytes(receipt)
    assert window["receipt_sha256"] == _sha256(receipt_path)
    assert receipt["start"] == {"path": start_path.name, "sha256": window["start_sha256"]}
    assert receipt["member_events"] == {"path": events_path.name, "sha256": _sha256(events_path)}
    artifacts = receipt["artifacts"]
    for name in [
        "run_metadata.json",
        "feature_reference.json",
        "graph_data.pt",
        "checkpoints/model_0_best.pth",
        "checkpoints/model_1_best.pth",
        "training_summary.json",
        "evaluation_summary.json",
        "timing_summary.json",
    ]:
        assert artifacts[name] == _sha256(stock_run / name), name
    predictions = sorted((stock_run / "averaged_predictions").glob("*.csv"))
    assert predictions
    for path in predictions:
        assert artifacts[path.relative_to(stock_run).as_posix()] == _sha256(path)
    assert not any(name.startswith("execution_provenance/") for name in artifacts)

    run_receipt = json.loads(
        (stock_run / "execution_provenance" / f"{run.run_id}.run_receipt.json").read_bytes()
    )
    assert run.run_receipt_sha256 == _sha256(
        stock_run / "execution_provenance" / f"{run.run_id}.run_receipt.json"
    )
    assert run_receipt["plan_sha256"] == plan.sha256
    assert run_receipt["windows"] == [
        {
            "walkforward_window": 0,
            "path": f"execution_provenance/{receipt_path.name}",
            "sha256": window["receipt_sha256"],
        }
    ]


def test_an_index_level_run_is_linked_the_same_way(tmp_path):
    base = tmp_path / "index"
    base.mkdir()
    dates = pd.bdate_range("2019-01-01", "2020-03-31")
    closes = 3000.0 * np.cumprod(1.0 + 0.002 * np.sin(np.arange(len(dates)) / 5.0))
    pd.DataFrame({"dt": dates.strftime("%Y-%m-%d"), "close": closes}).to_csv(
        base / "index.csv", index=False
    )
    out = _run(
        base,
        [
            "+data.experiment_mode=index_level",
            # Index-level preparation emits scalar edge weights for its edgeless graph,
            # so the default four-feature edges cannot run there (a separate defect).
            "graph.use_multi_feature_edges=false",
            # A cross-sectional IC needs more than the one index series.
            "training.loss_type=mse",
            "training.selection_metric=val_loss",
            f"+data.index_filename={(base / 'index.csv').as_posix()}",
        ],
    )
    run = read_execution_run(_plan(out))
    assert (run.status, run.experiment_mode) == ("complete", "index_level")
    (window,) = run.windows
    assert (window["status"], window["window_id"], window["problems"]) == ("complete", "0", [])
    assert json.loads((out / "run_metadata.json").read_text())["kdcode_list"] == ["INDEX"]


def _declared_panel(tmp_path: Path) -> tuple[Path, Path, str, list[str]]:
    """Publish the runner's panel as a package first; the runner rewrites identical bytes."""
    base = tmp_path / "declared_run"
    base.mkdir()
    _write_panel(base / "panel.csv", days=60)
    manifest = tmp_path / "declared" / "panel.r1.json"
    manifest.parent.mkdir()
    package = write_input_manifest(
        manifest,
        base,
        package_id="runner-panel",
        package_revision="r1",
        files=[InputFileSpec("panel.csv", "price panel")],
        provenance={
            "source": "synthetic fixture",
            "acquisition_mode": "fixture",
            "acquired_at": "2020-04-01T00:00:00+00:00",
            "producing_command": None,
            "producing_arguments": None,
            "unknowns": ["Fixture packages have no producing command"],
        },
    )
    overrides = [
        f"data.input_package_manifest={manifest.as_posix()}",
        f"data.input_package_manifest_sha256={package.sha256}",
        f"data.input_package_root={base.as_posix()}",
    ]
    return base, manifest, package.sha256, overrides


def test_a_window_retains_its_declared_inputs_and_its_receipt_binds_them(tmp_path):
    base, manifest, package_sha256, overrides = _declared_panel(tmp_path)
    out = _run(base, overrides)
    (window,) = read_execution_run(_plan(out)).windows
    metadata = json.loads((out / "run_metadata.json").read_text())
    attached = metadata["input_attachment"]
    assert attached["path"] == f"input_attachments/{window['attempt_id']}/record.json"
    assert attached["status"] == "complete"
    receipt = json.loads(_window_files(out, window)[2].read_bytes())
    assert receipt["artifacts"][attached["path"]] == attached["sha256"]
    for name in ("input_observations.json", f"manifests/{package_sha256}.json"):
        retained = Path(attached["path"]).parent / name
        assert receipt["artifacts"][retained.as_posix()] == _sha256(out / retained)

    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(base)
    shutil.rmtree(manifest.parent)
    inputs = read_run_inputs(AttachmentReference(moved / attached["path"], attached["sha256"]))
    assert (inputs.status, inputs.problems) == ("complete", [])
    assert inputs.execution == {"status": "unknown", "start": metadata["execution_start"]}
    assert [role["role"] for role in inputs.roles] == ["data.filename"]


class _Fred:
    """A local FRED client: daily series on weekdays, copper monthly."""

    def __init__(self, api_key):
        del api_key

    def get_series(self, series_id, observation_start, observation_end):
        frequency = "MS" if series_id == "PCOPPUSDM" else "B"
        dates = pd.date_range(observation_start, observation_end, freq=frequency)
        return pd.Series(np.linspace(10.0, 100.0, len(dates)), index=dates)


def test_a_capture_run_keeps_its_regime_snapshots_and_its_inputs_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=_Fred))
    base, manifest, _, overrides = _declared_panel(tmp_path)
    out = _run(
        base,
        [
            *overrides,
            "features.include_global_regime=true",
            "data.auxiliary_snapshot_mode=capture",
        ],
    )
    attached = json.loads((out / "run_metadata.json").read_text())["input_attachment"]
    assert attached["status"] == "complete"
    record = json.loads((out / attached["path"]).read_bytes())
    snapshots = {package["manifest_sha256"] for package in record["packages"][1:]}
    retained = sorted((out / "input_snapshots").glob("*/manifest.json"))
    assert len(retained) == len(snapshots) == 6
    assert {_sha256(path) for path in retained} == snapshots

    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(base)
    shutil.rmtree(manifest.parent)
    inputs = read_run_inputs(AttachmentReference(moved / attached["path"], attached["sha256"]))
    assert (inputs.status, inputs.problems) == ("complete", [])
    regime = [role for role in inputs.roles if role["role"].startswith("fred.")]
    assert len(regime) == 6 and all(role["observation_ids"] for role in regime)


def test_an_undeclared_panel_leaves_the_attachment_incomplete_but_the_run_finishes(stock_run):
    attached = json.loads((stock_run / "run_metadata.json").read_text())["input_attachment"]
    inputs = read_run_inputs(AttachmentReference(stock_run / attached["path"], attached["sha256"]))
    assert attached["status"] == inputs.status == "incomplete"
    assert inputs.problems == ["required role data.filename has no declared package"]
    # The data group's package is not attached when no selected file lies in it.
    assert inputs.packages == ()
    assert read_execution_run(_plan(stock_run)).status == "complete"


def test_each_window_gets_its_own_attempt_and_the_run_survives_relocation(two_window_run, tmp_path):
    plan = _plan(two_window_run)
    run = read_execution_run(plan)
    assert (run.status, run.problems, run.expected_windows) == ("complete", [], 2)
    assert [w["walkforward_window"] for w in run.windows] == [0, 1]
    assert [w["window_id"] for w in run.windows] == ["0", "1"]
    assert [w["output_dir"] for w in run.windows] == ["walkforward/w000", "walkforward/w001"]
    assert [w["status"] for w in run.windows] == ["complete", "complete"]
    assert run.windows[0]["attempt_id"] != run.windows[1]["attempt_id"]
    for window in run.windows:
        start_path, _, _ = _window_files(two_window_run, window)
        start = read_execution_provenance(ExecutionReference(start_path, window["start_sha256"]))
        assert start.record["window_id"] == window["window_id"]
        window_dir = two_window_run / window["output_dir"]
        assert start.resolved_config_bytes == (window_dir / "resolved_config.json").read_bytes()
        metadata = json.loads((window_dir / "run_metadata.json").read_text())
        assert metadata["walkforward_window"] == window["walkforward_window"]
        assert metadata["execution_start"]["sha256"] == window["start_sha256"]
    run_receipt = json.loads(
        (two_window_run / "execution_provenance" / f"{run.run_id}.run_receipt.json").read_bytes()
    )
    assert run_receipt["artifacts"] == {
        "walkforward_summary.json": _sha256(two_window_run / "walkforward_summary.json")
    }

    moved = tmp_path / "moved"
    shutil.copytree(two_window_run, moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with _restored_process_state(elsewhere):
        relocated = read_execution_run(
            ExecutionReference(moved / plan.path.relative_to(two_window_run), plan.sha256)
        )
    assert relocated == run


def test_a_preparation_failure_leaves_an_incomplete_attempt_and_no_receipt(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture: preparation failed")

    monkeypatch.setattr(run_experiment, "prepare_data", fail)
    out, error = _run_failing(tmp_path, [])
    assert "preparation failed" in str(error)

    run = read_execution_run(_plan(out))
    assert (run.status, run.run_receipt_sha256) == ("incomplete", None)
    assert "no run receipt" in run.problems
    (window,) = run.windows
    assert window["status"] == "incomplete"
    assert "no terminal receipt" in window["problems"]
    assert window["receipt_sha256"] is None
    # The immutable start was captured before preparation and is still this run's.
    start_path, _, receipt_path = _window_files(out, window)
    start = read_execution_provenance(ExecutionReference(start_path, window["start_sha256"]))
    assert start.record["window_id"] == "0"
    assert not receipt_path.exists()
    assert window["members"].status == "incomplete"
    assert window["members"].members == ()


def test_a_member_failure_marks_the_window_and_run_failed(tmp_path, monkeypatch):
    built = []
    create_model = run_experiment.create_model

    def second_member_fails(*args, **kwargs):
        built.append(1)
        if len(built) == 2:
            raise RuntimeError("fixture: member two failed")
        return create_model(*args, **kwargs)

    monkeypatch.setattr(run_experiment, "create_model", second_member_fails)
    out, _ = _run_failing(tmp_path, [])

    run = read_execution_run(_plan(out))
    assert run.status == "failed"
    (window,) = run.windows
    assert window["status"] == "failed"
    assert [m["status"] for m in window["members"].members] == ["complete", "failed"]
    assert "no terminal receipt" in window["problems"]


def test_an_output_failure_after_every_member_finished_is_not_completion(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fixture: evaluation output failed")

    monkeypatch.setattr(run_experiment, "compute_evaluation_summary", fail)
    out, _ = _run_failing(tmp_path, [])

    run = read_execution_run(_plan(out))
    (window,) = run.windows
    assert window["members"].status == "complete"
    assert (window["status"], run.status) == ("incomplete", "incomplete")
    assert "no terminal receipt" in window["problems"]


@pytest.mark.parametrize("failing_window", [0, 1])
def test_an_expected_window_that_did_not_finish_keeps_the_run_incomplete(
    tmp_path, monkeypatch, failing_window
):
    prepare_data = run_experiment.prepare_data
    calls = []

    def fail_one_window(*args, **kwargs):
        calls.append(1)
        if len(calls) == failing_window + 1:
            raise RuntimeError("fixture: window preparation failed")
        return prepare_data(*args, **kwargs)

    monkeypatch.setattr(run_experiment, "prepare_data", fail_one_window)
    out, _ = _run_failing(tmp_path, TWO_WINDOWS, days=400)

    run = read_execution_run(_plan(out))
    assert (run.status, run.expected_windows) == ("incomplete", 2)
    statuses = [window["status"] for window in run.windows]
    if failing_window == 0:
        assert statuses == ["incomplete", "incomplete"]
        assert "no execution start" in run.windows[1]["problems"]
        assert run.windows[1]["attempt_id"] is None
    else:
        assert statuses == ["complete", "incomplete"]
        assert "no terminal receipt" in run.windows[1]["problems"]
        assert run.windows[1]["attempt_id"] is not None


def _reseal(out: Path, run_id: str, window: dict) -> None:
    """Forge consistent receipts over the current files, as a careless rewrite would."""
    _, events_path, receipt_path = _window_files(out, window)
    receipt = json.loads(receipt_path.read_bytes())
    window_dir = out / window["output_dir"]
    receipt["artifacts"] = {name: _sha256(window_dir / name) for name in receipt["artifacts"]}
    receipt["member_events"]["sha256"] = _sha256(events_path)
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    run_receipt_path = out / "execution_provenance" / f"{run_id}.run_receipt.json"
    run_receipt = json.loads(run_receipt_path.read_bytes())
    run_receipt["windows"][window["walkforward_window"]]["sha256"] = _sha256(receipt_path)
    run_receipt_path.write_bytes(canonical_json_bytes(run_receipt))


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        ("edit_output", "artifact differs: evaluation_summary.json"),
        ("delete_prediction", "artifact missing: averaged_predictions/"),
        ("replace_checkpoint", "artifact differs: checkpoints/model_1_best.pth"),
        ("drop_final_event", "member events differ from the terminal receipt"),
        ("drop_final_event_resealed", "members incomplete"),
        ("delete_window_receipt", "terminal receipt missing"),
        ("edit_window_receipt", "terminal receipt differs from the run receipt"),
        ("metadata_names_another_start", "metadata does not reference this start"),
        ("checkpoint_not_the_loaded_one", "checkpoint differs from the one member 1 loaded"),
    ],
)
def test_damaged_window_evidence_is_never_complete(stock_copy, damage, problem):
    plan = _plan(stock_copy)
    before = read_execution_run(plan)
    (window,) = before.windows
    start_path, events_path, receipt_path = _window_files(stock_copy, window)
    if damage == "edit_output":
        (stock_copy / "evaluation_summary.json").write_text("{}")
    elif damage == "delete_prediction":
        sorted((stock_copy / "averaged_predictions").glob("*.csv"))[0].unlink()
    elif damage == "replace_checkpoint":
        (stock_copy / "checkpoints" / "model_1_best.pth").write_bytes(b"replaced")
    elif damage in {"drop_final_event", "drop_final_event_resealed"}:
        lines = events_path.read_bytes().splitlines(keepends=True)
        events_path.write_bytes(b"".join(lines[:-1]))
        if damage == "drop_final_event_resealed":
            _reseal(stock_copy, before.run_id, window)
    elif damage == "delete_window_receipt":
        receipt_path.unlink()
    elif damage == "edit_window_receipt":
        receipt = json.loads(receipt_path.read_bytes())
        receipt["written_at_utc"] = "2020-01-01T00:00:00+00:00"
        receipt_path.write_bytes(canonical_json_bytes(receipt))
    elif damage == "metadata_names_another_start":
        metadata = json.loads((stock_copy / "run_metadata.json").read_text())
        metadata["execution_start"]["sha256"] = "0" * 64
        (stock_copy / "run_metadata.json").write_text(json.dumps(metadata))
        _reseal(stock_copy, before.run_id, window)
    elif damage == "checkpoint_not_the_loaded_one":
        (stock_copy / "checkpoints" / "model_1_best.pth").write_bytes(b"replaced")
        _reseal(stock_copy, before.run_id, window)

    after = read_execution_run(plan)
    (damaged,) = after.windows
    assert (damaged["status"], after.status) == ("incomplete", "incomplete")
    assert any(entry.startswith(problem) for entry in damaged["problems"]), damaged["problems"]


def _rewrite(path: Path, edit) -> None:
    record = json.loads(path.read_bytes())
    edit(record)
    path.write_bytes(canonical_json_bytes(record))


def test_a_receipt_that_does_not_bind_the_metadata_is_not_complete(stock_copy):
    plan = _plan(stock_copy)
    before = read_execution_run(plan)
    (window,) = before.windows
    _, _, receipt_path = _window_files(stock_copy, window)
    _rewrite(receipt_path, lambda record: record["artifacts"].pop("run_metadata.json"))
    _reseal(stock_copy, before.run_id, window)

    (damaged,) = read_execution_run(plan).windows
    assert damaged["status"] == "incomplete"
    assert damaged["problems"] == ["terminal receipt does not bind the run metadata"]


@pytest.mark.parametrize(
    ("damage", "run_problem", "window_problem"),
    [
        ("unsafe_receipt_path", "run receipt invalid: Unsafe relative path", None),
        ("run_receipt_is_a_directory", "run receipt invalid: not a readable file", None),
        ("receipt_missing_a_field", None, "terminal receipt invalid: 'attempt_id'"),
        ("unanchored_receipt_missing_a_field", "no run receipt", "no terminal receipt"),
        ("directories_named_like_evidence", "no run receipt", "no terminal receipt"),
        ("unreadable_artifact", None, "artifact unreadable: evaluation_summary.json"),
        ("unreadable_receipt", None, "terminal receipt unreadable"),
    ],
)
def test_malformed_or_unreadable_evidence_is_reported_not_raised(
    stock_copy, monkeypatch, damage, run_problem, window_problem
):
    plan = _plan(stock_copy)
    before = read_execution_run(plan)
    (window,) = before.windows
    _, _, receipt_path = _window_files(stock_copy, window)
    evidence = stock_copy / "execution_provenance"
    run_receipt = evidence / f"{before.run_id}.run_receipt.json"
    if damage == "unsafe_receipt_path":
        _rewrite(run_receipt, lambda record: record["windows"][0].update(path="../receipt.json"))
    elif damage == "run_receipt_is_a_directory":
        run_receipt.unlink()
        run_receipt.mkdir()
    elif damage == "receipt_missing_a_field":
        _rewrite(receipt_path, lambda record: record.pop("attempt_id"))
        _reseal(stock_copy, before.run_id, window)
    elif damage == "unanchored_receipt_missing_a_field":
        _rewrite(receipt_path, lambda record: record.pop("plan_sha256"))
        run_receipt.unlink()
    elif damage == "directories_named_like_evidence":
        run_receipt.unlink()
        receipt_path.unlink()
        (evidence / ("e" * 32 + ".receipt.json")).mkdir()
        (evidence / ("e" * 32 + ".json")).mkdir()
    else:
        unreadable = (
            stock_copy / "evaluation_summary.json"
            if damage == "unreadable_artifact"
            else receipt_path
        )
        original = execution_provenance.sha256_file

        def deny(path):
            if Path(path) == unreadable:
                raise PermissionError("fixture: unreadable")
            return original(path)

        monkeypatch.setattr(execution_provenance, "sha256_file", deny)

    run = read_execution_run(plan)
    (damaged,) = run.windows
    assert run.status == "incomplete"
    if run_problem is not None:
        assert any(entry.startswith(run_problem) for entry in run.problems), run.problems
    if window_problem is not None:
        assert damaged["status"] == "incomplete"
        assert any(entry.startswith(window_problem) for entry in damaged["problems"]), damaged[
            "problems"
        ]


def test_a_window_receipt_refuses_to_bind_evidence_or_a_moved_start(stock_copy, tmp_path):
    plan = _plan(stock_copy)
    (window,) = read_execution_run(plan).windows
    start_path, events_path, _ = _window_files(stock_copy, window)
    start = ExecutionReference(start_path, _sha256(start_path))
    with pytest.raises(ValueError, match="cannot bind itself"):
        write_window_receipt(
            start, plan=plan, walkforward_window=0, artifacts=["execution_provenance"]
        )

    moved = tmp_path / "elsewhere" / "execution_provenance"
    moved.mkdir(parents=True)
    shutil.copy2(start_path, moved / start_path.name)
    shutil.copy2(events_path, moved / events_path.name)
    with pytest.raises(ValueError, match="does not belong to this plan window"):
        write_window_receipt(
            ExecutionReference(moved / start_path.name, start.sha256),
            plan=plan,
            walkforward_window=0,
            artifacts=[],
        )


def test_a_run_without_its_run_receipt_is_incomplete(stock_copy):
    plan = _plan(stock_copy)
    run_id = read_execution_run(plan).run_id
    (stock_copy / "execution_provenance" / f"{run_id}.run_receipt.json").unlink()
    run = read_execution_run(plan)
    assert (run.status, run.run_receipt_sha256) == ("incomplete", None)
    assert run.problems == ["no run receipt"]


def test_an_altered_plan_is_rejected(stock_copy):
    plan = _plan(stock_copy)
    plan.path.write_bytes(plan.path.read_bytes().replace(b'"stock_level"', b'"index_level"'))
    with pytest.raises(ValueError, match="digest mismatch"):
        read_execution_run(plan)


def test_a_terminal_receipt_from_another_run_is_not_this_runs(stock_copy):
    plan = _plan(stock_copy)
    run = read_execution_run(plan)
    (window,) = run.windows
    _, _, receipt_path = _window_files(stock_copy, window)
    receipt = json.loads(receipt_path.read_bytes())
    receipt["plan_sha256"] = "0" * 64
    receipt_path.unlink()
    receipt_path.with_name("f" * 32 + ".receipt.json").write_bytes(canonical_json_bytes(receipt))
    (stock_copy / "execution_provenance" / f"{run.run_id}.run_receipt.json").unlink()

    (unbound,) = read_execution_run(plan).windows
    assert unbound["status"] == "incomplete"
    assert "no terminal receipt" in unbound["problems"]
    assert unbound["attempt_id"] == window["attempt_id"]


def test_a_start_from_another_run_in_the_same_directory_is_not_this_runs(stock_copy, tmp_path):
    plan = _plan(stock_copy)
    run = read_execution_run(plan)
    (window,) = run.windows
    _, _, receipt_path = _window_files(stock_copy, window)
    receipt_path.unlink()
    (stock_copy / "execution_provenance" / f"{run.run_id}.run_receipt.json").unlink()
    other_plan = write_execution_plan(
        tmp_path / "other", experiment_mode="stock_level", window_dirs=[tmp_path / "other"]
    )
    config = stock_copy / "resolved_config.json"
    capture_execution_start(
        REPOSITORY_ROOT,
        config,
        resolved_config_sha256=_sha256(config),
        output_dir=stock_copy,
        window_id="0",
        plan=other_plan,
    )

    (unbound,) = read_execution_run(plan).windows
    assert unbound["attempt_id"] == window["attempt_id"]
    assert "more than one execution start" not in unbound["problems"]
    assert "no terminal receipt" in unbound["problems"]
