"""Guards for the per-input data identity recorded in ``run_metadata.json``.

A run's data identity is every raw input it consumed, not only
``data.filename``. These tests pin the two seams the identity is observed at:
the parsed ``run_metadata.json`` payload, and the pure fingerprint helpers in
``mci_gru.evaluation.experiment_summary`` that produce it.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

from mci_gru.config import create_config_from_dict
from mci_gru.data import path_resolver
from mci_gru.evaluation.experiment_summary import (
    build_run_metadata,
    data_file_fingerprint,
    data_input_identity,
    data_inputs_identity,
)

logger = logging.getLogger(__name__)


def test_data_input_identity_records_resolved_path_and_content_digest(tmp_path):
    payload = b"kdcode,valid_from,valid_to\nAAPL,2020-01-01,2025-12-31\n"
    csv_path = tmp_path / "pit_universe.csv"
    csv_path.write_bytes(payload)

    identity = data_input_identity(str(csv_path), logger)

    assert identity["configured_path"] == str(csv_path)
    assert identity["resolved_path"] == str(csv_path.resolve())
    assert identity["sha256"] == hashlib.sha256(payload).hexdigest()
    assert identity["size_bytes"] == len(payload)
    assert (
        identity["mtime_iso"]
        == datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc).isoformat()
    )


def test_data_input_identity_unresolvable_path_warns_and_nulls_fields(tmp_path, caplog):
    missing = tmp_path / "absent_universe.csv"

    with caplog.at_level(logging.WARNING):
        identity = data_input_identity(str(missing), logger)

    assert identity == {
        "configured_path": str(missing),
        "resolved_path": None,
        "sha256": None,
        "size_bytes": None,
        "mtime_iso": None,
    }
    assert any("absent_universe.csv" in record.getMessage() for record in caplog.records)


def test_data_input_identity_records_the_fallback_file_the_resolver_found(tmp_path, monkeypatch):
    """A basename fallback must be visible: resolved path differs from configured."""
    fallback_dir = tmp_path / "data" / "raw" / "market"
    fallback_dir.mkdir(parents=True)
    payload = b"kdcode,dt,close\nAAPL,2026-01-02,100\n"
    (fallback_dir / "panel.csv").write_bytes(payload)
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path / "data")

    identity = data_input_identity("some/other/place/panel.csv", logger)

    assert identity["configured_path"] == "some/other/place/panel.csv"
    assert identity["resolved_path"] == str((fallback_dir / "panel.csv").resolve())
    assert identity["sha256"] == hashlib.sha256(payload).hexdigest()
    # The legacy cwd-relative helper cannot see this file at all, which is the
    # gap this block exists to close.
    assert data_file_fingerprint("some/other/place/panel.csv", logger)["data_file_sha256"] is None


def _write(path, payload=b"a,b\n1,2\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def test_data_inputs_identity_keys_every_configured_input_by_its_config_field(tmp_path):
    config = create_config_from_dict(
        {
            "data": {
                "filename": _write(tmp_path / "panel.csv"),
                "use_pit_universe": True,
                "pit_universe_csv": _write(tmp_path / "pit.csv"),
            },
            "graph": {
                "use_sector_relation": True,
                "sector_map_csv": _write(tmp_path / "sectors.csv"),
            },
            "features": {
                "include_global_regime": True,
                "regime_inputs_csv": _write(tmp_path / "regime.csv"),
            },
        }
    )

    inputs = data_inputs_identity(config, logger)

    assert sorted(inputs) == [
        "data.filename",
        "data.pit_universe_csv",
        "features.regime_inputs_csv",
        "graph.sector_map_csv",
    ]
    assert inputs["data.pit_universe_csv"]["resolved_path"] == str((tmp_path / "pit.csv").resolve())
    assert all(entry["sha256"] for entry in inputs.values())


def test_data_inputs_identity_omits_inputs_that_are_unset_or_switched_off(tmp_path):
    config = create_config_from_dict(
        {
            "data": {
                "filename": _write(tmp_path / "panel.csv"),
                # PIT universe named but switched off: not read, so not recorded.
                "use_pit_universe": False,
                "pit_universe_csv": _write(tmp_path / "pit.csv"),
            },
            "graph": {"use_sector_relation": False, "sector_map_csv": None},
            "features": {"include_global_regime": True, "regime_inputs_csv": None},
        }
    )

    inputs = data_inputs_identity(config, logger)

    assert list(inputs) == ["data.filename"]


def test_data_inputs_identity_records_the_index_csv_instead_of_the_panel(tmp_path):
    config = create_config_from_dict(
        {
            "data": {
                "experiment_mode": "index_level",
                "filename": _write(tmp_path / "panel.csv"),
                "index_filename": _write(tmp_path / "index.csv"),
            },
        }
    )

    inputs = data_inputs_identity(config, logger)

    assert list(inputs) == ["data.index_filename"]


def test_run_metadata_carries_data_inputs_beside_unchanged_legacy_keys(tmp_path, monkeypatch):
    """The artifact seam: what a window actually writes to run_metadata.json."""
    panel = tmp_path / "panel.csv"
    panel_payload = b"kdcode,dt,close\nAAPL,2026-01-02,100\n"
    panel.write_bytes(panel_payload)
    pit = tmp_path / "pit.csv"
    pit_payload = b"kdcode,valid_from,valid_to\nAAPL,2020-01-01,2026-12-31\n"
    pit.write_bytes(pit_payload)
    monkeypatch.chdir(tmp_path)

    config = create_config_from_dict(
        {
            "data": {
                "filename": "panel.csv",
                "use_pit_universe": True,
                "pit_universe_csv": "pit.csv",
            },
            "model": {"his_t": 5, "label_t": 3},
            "seed": 11,
        }
    )
    data = {
        "norm_means": {"close": 1.0},
        "norm_stds": {"close": 2.0},
        "feature_cols": ["close"],
        "kdcode_list": ["AAPL"],
        "graph_static_valid_from": "2021-01-01",
        "pit_universe_mode": "row_filter",
        "pit_breadth": {"2026-01-02": 1},
    }

    metadata = build_run_metadata(
        config,
        data,
        walkforward_window=2,
        resolved_config_identity={
            "resolved_config_path": "resolved_config.json",
            "resolved_config_sha256": "deadbeef",
        },
        logger=logger,
    )
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    parsed = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Legacy keys are unchanged: the notebook generators read data_file_sha256.
    assert parsed["data_file"] == "panel.csv"
    assert parsed["train_end"] == config.data.train_end
    assert parsed["data_file_sha256"] == hashlib.sha256(panel_payload).hexdigest()
    assert parsed["data_file_size_bytes"] == len(panel_payload)
    assert datetime.fromisoformat(parsed["data_file_mtime_iso"]).tzinfo is not None
    assert parsed["walkforward_window"] == 2
    assert parsed["resolved_config_sha256"] == "deadbeef"

    # And every configured input the window read is now identified.
    assert sorted(parsed["data_inputs"]) == ["data.filename", "data.pit_universe_csv"]
    assert parsed["data_inputs"]["data.pit_universe_csv"] == {
        "configured_path": "pit.csv",
        "resolved_path": str(pit.resolve()),
        "sha256": hashlib.sha256(pit_payload).hexdigest(),
        "size_bytes": len(pit_payload),
        "mtime_iso": parsed["data_inputs"]["data.pit_universe_csv"]["mtime_iso"],
    }
    assert parsed["data_inputs"]["data.filename"]["sha256"] == parsed["data_file_sha256"]


def test_data_inputs_identity_omits_a_sector_map_whose_relation_is_switched_off(tmp_path):
    """Naming a sector map does not mean the run reads it.

    ``mci_gru/pipeline.py`` gates the sector-map read on
    ``use_sector_relation and sector_map_csv``, so a config that names the file
    with the relation off never opens it. Recording it would assert an input the
    run did not consume.
    """
    config = create_config_from_dict(
        {
            "data": {"filename": _write(tmp_path / "panel.csv")},
            "graph": {
                "use_sector_relation": False,
                "sector_map_csv": _write(tmp_path / "sectors.csv"),
            },
        }
    )

    inputs = data_inputs_identity(config, logger)

    assert list(inputs) == ["data.filename"]


def test_legacy_data_file_keys_keep_cwd_semantics_where_the_resolver_disagrees(
    tmp_path, monkeypatch
):
    """The two blocks mean different things, and that is deliberate.

    ``data_file_sha256`` is read by the PIT notebook generators, so it keeps its
    existing cwd-relative meaning. ``data_inputs`` reports what the loaders
    actually open. Where a basename fallback makes those diverge, the legacy key
    stays null and the new block carries the digest.
    """
    fallback_dir = tmp_path / "data" / "raw" / "market"
    fallback_dir.mkdir(parents=True)
    payload = b"kdcode,dt,close\nAAPL,2026-01-02,100\n"
    (fallback_dir / "panel.csv").write_bytes(payload)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(elsewhere)

    config = create_config_from_dict({"data": {"filename": "panel.csv"}})
    data = {
        "norm_means": {"close": 1.0},
        "norm_stds": {"close": 2.0},
        "feature_cols": ["close"],
        "kdcode_list": ["AAPL"],
    }

    metadata = build_run_metadata(
        config,
        data,
        walkforward_window=0,
        resolved_config_identity={},
        logger=logger,
    )

    assert metadata["data_file_sha256"] is None
    assert metadata["data_inputs"]["data.filename"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["data_inputs"]["data.filename"]["resolved_path"] == str(
        (fallback_dir / "panel.csv").resolve()
    )
