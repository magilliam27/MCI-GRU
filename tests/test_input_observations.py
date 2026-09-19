"""Native read-to-metadata guards for immutable consumed-input observations."""

import gzip
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from mci_gru.config import DataConfig, ExperimentConfig
from mci_gru.data.data_manager import DataManager
from mci_gru.data.input_observations import InputObservationContext, InputObservationError
from mci_gru.evaluation.experiment_summary import build_run_metadata


def _save(tmp_path, config, observations):
    metadata = build_run_metadata(
        config,
        {
            "input_observations": observations,
            "norm_means": {},
            "norm_stds": {},
            "feature_cols": ["close"],
            "kdcode_list": ["A"],
        },
        walkforward_window=0,
        resolved_config_identity={},
        logger=logging.getLogger(__name__),
    )
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def test_native_reader_rejects_a_sealed_context_before_reading_source(tmp_path, monkeypatch):
    source = tmp_path / "panel.csv"
    source.write_bytes(b"kdcode,dt,close\nA,2020-01-01,1\n")
    manager = DataManager(DataConfig(filename=str(source)))
    manager.load()
    manager.input_observations.freeze()
    real_open = io.open

    def forbid_read(file, *args, **kwargs):
        if Path(file) == source:
            pytest.fail("A sealed context still opened a source")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", forbid_read)
    with pytest.raises(ValueError, match="sealed"):
        manager.load()


def test_invalid_observation_cannot_pollute_the_saved_read_inventory(tmp_path):
    source = tmp_path / "panel.csv"
    source.write_bytes(b"kdcode,dt,close\nA,2020-01-01,1\n")
    config = ExperimentConfig(data=DataConfig(filename=str(source)))
    context = InputObservationContext()
    DataManager(config.data, input_observations=context).load()

    with pytest.raises(ValueError, match="observation"):
        context.record_observation({"outcome": "invented"})

    metadata = _save(tmp_path, config, context.freeze())
    assert len(metadata["input_observations"]["observations"]) == 1


def test_compressed_csv_records_original_bytes_and_parser_options(tmp_path):
    source = tmp_path / "panel.csv.gz"
    content = gzip.compress(b"kdcode,dt,close\nA,2020-01-01,17\n", mtime=0)
    source.write_bytes(content)
    config = ExperimentConfig(data=DataConfig(filename=str(source)))
    manager = DataManager(config.data)
    before = datetime.now(timezone.utc)
    frame = manager.load()
    after = datetime.now(timezone.utc)
    metadata = _save(tmp_path, config, manager.input_observations.freeze())

    assert frame["close"].tolist() == [17]
    assert metadata["data_inputs"]["data.filename"]["sha256"] == hashlib.sha256(content).hexdigest()
    observation = metadata["input_observations"]["observations"][0]
    assert observation["parser"] == {"name": "pandas.read_csv", "options": {"compression": "gzip"}}
    assert observation["source_kind"] == "file"
    assert observation["stage"] == "parse"
    assert observation["outcome"] == "success"
    assert observation["role"] == "data.filename"
    assert observation["configured_path"] == str(source)
    assert before <= datetime.fromisoformat(observation["observed_at"]) <= after


def test_parser_and_saved_identity_share_bytes_even_if_source_changes_during_parse(tmp_path):
    source = tmp_path / "panel.csv"
    original = b"kdcode,dt,close\nA,2020-01-01,17\n"
    source.write_bytes(original)
    config = ExperimentConfig(data=DataConfig(filename=str(source)))
    context = InputObservationContext()

    def parse(content):
        source.write_bytes(b"replacement with different bytes")
        return pd.read_csv(io.BytesIO(content))

    frame = context.read_file(
        source,
        role="data.filename",
        configured_path=str(source),
        parse=parse,
        parser={"name": "pandas.read_csv", "options": {}},
    )
    snapshot = context.freeze()
    metadata = _save(tmp_path, config, snapshot)

    assert frame["close"].tolist() == [17]
    assert (
        metadata["data_inputs"]["data.filename"]["sha256"] == hashlib.sha256(original).hexdigest()
    )
    assert metadata["data_inputs"]["data.filename"]["size_bytes"] == len(original)
    assert snapshot is context.freeze()
    metadata["input_observations"]["observations"][0]["identity"]["sha256"] = "tampered"
    assert (
        _save(tmp_path, config, snapshot)["data_inputs"]["data.filename"]["sha256"]
        == hashlib.sha256(original).hexdigest()
    )


def test_saved_record_preserves_nested_contributions_and_distinct_use_links(tmp_path):
    context = InputObservationContext()
    record = {
        "source_kind": "sdk",
        "stage": "acquisition",
        "outcome": "success",
        "observed_at": "2020-01-01T00:00:00+00:00",
        "provider": {
            "series": "synthetic",
            "request": {"bounds": ["2019", "2020"]},
            "snapshot": {"manifest_bytes_base64": "e30K", "codec": "synthetic.v1"},
        },
    }
    reference = context.record_observation(record)
    context.record_use(reference, "first-role")
    context.record_use(reference, "second-role", "explicit-alias")
    unused = context.record_observation(record)
    record["provider"]["request"]["bounds"][0] = "mutated"
    record["provider"]["snapshot"]["codec"] = "mutated"
    snapshot = context.freeze()
    metadata = _save(tmp_path, ExperimentConfig(), snapshot)

    events = metadata["input_observations"]
    assert events["schema"] == "mci_gru.input_observations.v1"
    assert [event["observation_id"] for event in events["observations"]] == [0, 1]
    assert events["observations"][0]["provider"] == {
        "series": "synthetic",
        "request": {"bounds": ["2019", "2020"]},
        "snapshot": {"manifest_bytes_base64": "e30K", "codec": "synthetic.v1"},
    }
    assert events["uses"] == [
        {"observation_id": 0, "role": "first-role", "configured_path": None},
        {"observation_id": 0, "role": "second-role", "configured_path": "explicit-alias"},
    ]
    assert set(metadata["data_inputs"]) == {"first-role", "second-role"}
    with pytest.raises(ValueError, match="sealed"):
        context.record_use(unused, "late-use")
    with pytest.raises(ValueError, match="sealed"):
        context.record_observation(record)


@pytest.mark.parametrize("outcome", ["empty", "error"])
def test_nonconsumed_and_foreign_observations_cannot_become_saved_uses(tmp_path, outcome):
    record = {
        "source_kind": "sdk",
        "stage": "acquisition",
        "outcome": outcome,
        "observed_at": "2020-01-01T00:00:00+00:00",
    }
    context = InputObservationContext()
    reference = context.record_observation(record)
    foreign = InputObservationContext().record_observation({**record, "outcome": "success"})
    with pytest.raises(ValueError, match="successful"):
        context.record_use(reference, "invalid-use")
    with pytest.raises(ValueError, match="Unknown"):
        context.record_use(foreign, "foreign-use")
    metadata = _save(tmp_path, ExperimentConfig(), context.freeze())
    assert metadata["data_inputs"] == {}
    assert metadata["input_observations"]["observations"][0]["outcome"] == outcome


def test_parser_integrity_failure_is_distinct_from_ordinary_parse_failure(tmp_path):
    source = tmp_path / "panel.csv"
    source.write_bytes(b"exact source bytes")
    context = InputObservationContext()

    def invalid_snapshot(content):
        raise InputObservationError("synthetic integrity failure")

    with pytest.raises(InputObservationError):
        context.read_file(
            source,
            role="data.filename",
            configured_path=str(source),
            parse=invalid_snapshot,
            parser={"name": "synthetic", "options": {}},
        )
    metadata = _save(tmp_path, ExperimentConfig(), context.freeze())
    event = metadata["input_observations"]["observations"][0]
    assert event["stage"] == "integrity"
    assert event["outcome"] == "error"
    assert event["error_code"] == "InputObservationError"
    assert metadata["data_inputs"] == {}


def test_failed_native_read_records_only_known_facts_and_cannot_be_consumed(tmp_path):
    source = tmp_path / "missing.csv"
    context = InputObservationContext()
    with pytest.raises(FileNotFoundError):
        context.read_csv(source, role="optional.csv", configured_path="missing.csv")
    metadata = _save(tmp_path, ExperimentConfig(), context.freeze())
    event = metadata["input_observations"]["observations"][0]
    assert event["stage"] == "read"
    assert event["outcome"] == "error"
    assert event["error_code"] == "FileNotFoundError"
    assert event["configured_path"] == "missing.csv"
    assert event["identity"] == {"resolved_path": str(source)}
    assert metadata["data_inputs"] == {}
    assert not source.exists()
