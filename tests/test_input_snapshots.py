"""Public exact-observation snapshot preservation and offline replay contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mci_gru.data.input_manifest import InputFileSpec, write_input_manifest
from mci_gru.data.input_observations import InputObservationError
from mci_gru.data.input_snapshots import SnapshotReference, load_snapshot, save_snapshot


def test_snapshot_round_trip_keeps_exact_file_bytes_and_original_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"dt,value\r\n2020-01-03,1.25\r\n2020-01-01,\r\n2020-01-03,1.25\r\n"
    original = tmp_path / "original.csv"
    original.write_bytes(content)
    original_stat = original.stat()
    request = {"configured_path": "original.csv", "operation": "read_csv"}
    reference = save_snapshot(
        tmp_path / "retained",
        content,
        package_id="synthetic-vix",
        package_revision="r1",
        role="implicit.vix_csv",
        source="file",
        request=request,
        acquired_at="2020-02-01T12:00:00+00:00",
    )
    real_open = Path.open

    def deny_original(path, *args, **kwargs):
        if path == original:
            raise AssertionError("Replay reopened the original source")
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", deny_original)
        replay = load_snapshot(reference, role="implicit.vix_csv", source="file", request=request)

    assert replay.data == content
    assert replay.provenance == {
        "role": "implicit.vix_csv",
        "source": "file",
        "request": request,
        "acquired_at": "2020-02-01T12:00:00+00:00",
    }
    assert replay.manifest.manifest.package_id == "synthetic-vix"
    assert replay.manifest.manifest.package_revision == "r1"
    assert replay.manifest.sha256 == reference.manifest_sha256
    assert original.read_bytes() == content
    assert original.stat().st_mtime_ns == original_stat.st_mtime_ns


@pytest.mark.parametrize(
    "target", ["manifest.json", "files/observations.bin", "files/snapshot.json"]
)
def test_snapshot_rejects_changed_retained_bytes_without_repair(
    tmp_path: Path, target: str
) -> None:
    reference = save_snapshot(
        tmp_path / "retained",
        b"dt,value\n2020-01-01,3\n",
        package_id="synthetic",
        package_revision="r1",
        role="vix",
        source="file",
        request={"operation": "read_csv"},
        acquired_at="2020-02-01T12:00:00+00:00",
    )
    damaged = tmp_path / "retained" / target
    damaged.write_bytes(damaged.read_bytes() + b" ")
    before = damaged.read_bytes()
    with pytest.raises(InputObservationError) as failure:
        load_snapshot(reference, role="vix", source="file", request={"operation": "read_csv"})
    assert failure.value.facts["stage"] == "integrity"
    assert failure.value.facts["role"] == "vix"
    assert failure.value.facts["expected_sha256"] != failure.value.facts["observed_sha256"]
    assert damaged.read_bytes() == before


@pytest.mark.parametrize("mismatch", ["role", "source", "request"])
def test_snapshot_refuses_a_different_selected_request(tmp_path: Path, mismatch: str) -> None:
    selected = {"role": "regime.yield_10y", "source": "fred", "request": {"series": "DGS10"}}
    reference = save_snapshot(
        tmp_path / "retained",
        b"original",
        package_id="synthetic",
        package_revision="r1",
        **selected,
        acquired_at="2020-02-01T12:00:00+00:00",
    )
    incompatible = {
        **selected,
        mismatch: {"series": "DGS3MO"} if mismatch == "request" else "other",
    }
    with pytest.raises(InputObservationError, match="selection_mismatch"):
        load_snapshot(reference, **incompatible)


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_snapshot_keeps_sdk_observation_types_axes_order_and_missingness(
    tmp_path: Path, kind: str
) -> None:
    dates = pd.DatetimeIndex(
        ["2020-01-03", "2020-01-01", "2020-01-03", None], tz="US/Eastern", name="Date"
    )
    data = pd.Series([1.2345678901234567, np.nan, -0.0, np.inf], index=dates, name="DGS10")
    if kind == "frame":
        data = pd.concat(
            [
                data,
                pd.Series([".", None, pd.NA, "3.1"], index=dates, dtype=object),
                pd.Series([3, None, 1, 2], index=dates, dtype="Int64"),
            ],
            axis=1,
        )
        data.columns = pd.MultiIndex.from_tuples(
            [(".VIX", "CLOSE"), (".VIX", "CLOSE"), (".VIX", "COUNT")],
            names=["Instrument", "Field"],
        )
    reference = save_snapshot(
        tmp_path / "retained",
        data,
        package_id="synthetic",
        package_revision="r1",
        role="sdk",
        source="fred" if kind == "series" else "lseg",
        request={"operation": "history"},
        acquired_at="2020-02-01T12:00:00+00:00",
    )
    replay = load_snapshot(
        reference,
        role="sdk",
        source="fred" if kind == "series" else "lseg",
        request={"operation": "history"},
    )
    if kind == "series":
        pd.testing.assert_series_equal(replay.data, data, check_exact=True)
        assert np.signbit(replay.data.iloc[2])
    else:
        pd.testing.assert_frame_equal(replay.data, data, check_exact=True)
        assert replay.data.iloc[1, 1] is None
        assert replay.data.iloc[2, 1] is pd.NA


@pytest.mark.parametrize("fault", ["version", "codec", "missing_member", "missing_manifest"])
def test_snapshot_rejects_missing_or_incompatible_packages(tmp_path: Path, fault: str) -> None:
    reference = save_snapshot(
        tmp_path / "retained",
        b"original",
        package_id="synthetic",
        package_revision="r1",
        role="vix",
        source="file",
        request={},
        acquired_at="2020-02-01T12:00:00+00:00",
    )
    root = tmp_path / "retained" / "files"
    if fault.startswith("missing"):
        (
            root / "observations.bin" if fault == "missing_member" else reference.manifest_path
        ).unlink()
    else:
        descriptor_path = root / "snapshot.json"
        descriptor = json.loads(descriptor_path.read_bytes())
        descriptor["schema_version" if fault == "version" else "codec"] = 999
        descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
        manifest_path = reference.manifest_path.with_name("incompatible.json")
        manifest = write_input_manifest(
            manifest_path,
            root,
            package_id="synthetic",
            package_revision="r2",
            files=[
                InputFileSpec("observations.bin", "original"),
                InputFileSpec("snapshot.json", "contract"),
            ],
            provenance={
                "source": "file",
                "acquisition_mode": "snapshot",
                "acquired_at": "2020-02-01",
                "producing_command": None,
                "producing_arguments": None,
                "unknowns": ["not known"],
            },
        )
        reference = SnapshotReference(manifest_path, manifest.sha256)
    with pytest.raises(InputObservationError):
        load_snapshot(reference, role="vix", source="file", request={})


def test_snapshot_keeps_literal_sdk_metadata_types(tmp_path: Path) -> None:
    data = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    data.attrs = {1: ("vintage", None), "tags": ["published", 3], "nonfinite": np.nan}
    reference = save_snapshot(
        tmp_path / "retained",
        data,
        package_id="synthetic",
        package_revision="r1",
        role="fred.rate",
        source="fred",
        request={},
        acquired_at="2020-02-01T12:00:00+00:00",
    )
    replay = load_snapshot(reference, role="fred.rate", source="fred", request={})
    assert replay.data.attrs[1] == ("vintage", None)
    assert replay.data.attrs["tags"] == ["published", 3]
    assert np.isnan(replay.data.attrs["nonfinite"])
