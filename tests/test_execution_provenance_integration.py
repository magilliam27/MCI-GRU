"""Ensemble members leave retained evidence of what actually ran (#144, member proof).

Two tiny synthetic CPU members run through ``train_multiple_models`` against a
retained execution-start record. Readback must use only the retained evidence:
planned seeds or a checkpoint filename never prove that a member ran.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mci_gru.config import ExperimentConfig, TrainingConfig
from mci_gru.evaluation.execution_provenance import (
    BACKEND_OBSERVATIONS,
    ExecutionReference,
    MemberEventLog,
    capture_execution_start,
    read_member_execution,
)
from mci_gru.evaluation.experiment_summary import write_resolved_config
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
