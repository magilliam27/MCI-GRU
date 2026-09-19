"""Actual native reads remain identifiable in each saved preparation result."""

import builtins
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mci_gru.config import create_config_from_dict
from mci_gru.data import path_resolver
from mci_gru.data.input_observations import InputObservationError
from mci_gru.evaluation.experiment_summary import (
    build_run_metadata,
    data_file_fingerprint,
)
from mci_gru.features import FeatureEngineer
from mci_gru.pipeline import prepare_data, prepare_data_index_level

logger = logging.getLogger(__name__)


def _native_panel() -> bytes:
    rows = ["kdcode,dt,open,high,low,close,volume,turnover"]
    for day in range(1, 32):
        for stock, offset in [("AAA", 0), ("BBB", 20)]:
            close = 100 + day + offset
            rows.append(
                f"{stock},2020-01-{day:02d},{close},{close + 1},{close - 1},"
                f"{close},1000,{close * 1000}"
            )
    return ("\n".join(rows) + "\n").encode()


def _native_config(filename: str):
    return create_config_from_dict(
        {
            "data": {
                "filename": filename,
                "use_pit_universe": False,
                "train_start": "2020-01-05",
                "train_end": "2020-01-15",
                "val_start": "2020-01-19",
                "val_end": "2020-01-23",
                "test_start": "2020-01-27",
                "test_end": "2020-01-31",
            },
            "features": {"include_momentum": False, "include_weekly_momentum": False},
            "model": {"his_t": 2, "label_t": 2},
            "graph": {"use_multi_feature_edges": False},
            "training": {"label_type": "returns"},
            "tracking": {"enabled": False},
        }
    )


def _saved_metadata(tmp_path, config, data, window=0):
    metadata = build_run_metadata(
        config,
        data,
        walkforward_window=window,
        resolved_config_identity={"resolved_config_sha256": "synthetic-config"},
        logger=logger,
    )
    target = tmp_path / f"run_metadata_{window}.json"
    target.write_text(json.dumps(metadata), encoding="utf-8")
    return json.loads(target.read_text(encoding="utf-8"))


def test_saved_metadata_keeps_consumed_csv_identity_after_source_replacement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "panel.csv"
    original = _native_panel()
    source.write_bytes(original)
    os.utime(source, (1700000000, 1700000000))
    config = _native_config("panel.csv")

    data = prepare_data(config, FeatureEngineer(config.features))
    replacement = b"a different file after preparation\n"
    source.write_bytes(replacement)
    os.utime(source, (1800000000, 1800000000))
    metadata = _saved_metadata(tmp_path, config, data)

    identity = metadata["data_inputs"]["data.filename"]
    assert identity["sha256"] == hashlib.sha256(original).hexdigest()
    assert identity["size_bytes"] == len(original)
    assert identity["mtime_iso"] == "2023-11-14T22:13:20+00:00"
    assert identity["configured_path"] == "panel.csv"
    assert identity["resolved_path"] == str(source)
    assert metadata["data_file_sha256"] == hashlib.sha256(replacement).hexdigest()


def test_index_metadata_records_its_read_and_excludes_unused_stock_pit_and_sector(tmp_path):
    source = tmp_path / "index.csv"
    original = (
        "dt,close\n" + "".join(f"2020-01-{d:02d},{100 + d}\n" for d in range(1, 32))
    ).encode()
    source.write_bytes(original)
    config = _native_config(str(tmp_path / "unused_stock.csv"))
    config.data.index_filename = str(source)
    config.data.use_pit_universe = True
    config.data.pit_universe_csv = str(tmp_path / "unused_pit.csv")
    config.graph.sector_map_csv = str(tmp_path / "unused_sector.csv")
    data = prepare_data_index_level(config, FeatureEngineer(config.features))
    source.write_bytes(b"replacement\n")

    metadata = _saved_metadata(tmp_path, config, data)

    assert set(metadata["data_inputs"]) == {"data.index_filename"}
    assert (
        metadata["data_inputs"]["data.index_filename"]["sha256"]
        == hashlib.sha256(original).hexdigest()
    )
    assert metadata["data_file_sha256"] is None


@pytest.mark.parametrize(
    "mode,normalisation",
    [("row_filter", "zscore"), ("masked_panel", "zscore"), ("masked_panel", "rank_gauss")],
)
def test_preparation_keeps_both_pit_reads_when_the_file_changes(
    tmp_path, monkeypatch, mode, normalisation
):
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    pit_path = tmp_path / "pit.csv"
    original = b"kdcode,valid_from,valid_to\nAAA,2020-01-01,2020-01-31\nBBB,2020-01-01,2020-01-31\n"
    replacement = original.replace(b"2020-01-31", b"2020-02-01")
    pit_path.write_bytes(original)
    config = _native_config(str(source))
    config.data.use_pit_universe = True
    config.data.pit_universe_csv = str(pit_path)
    config.data.pit_universe_mode = mode
    config.data.pit_min_scoreable_stocks = 0
    config.data.normalisation = normalisation
    original_open = io.open
    reads = []

    def replace_before_second_read(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == pit_path and "r" in mode:
            reads.append(str(file))
            if len(reads) == 2:
                pit_path.write_bytes(replacement)
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", replace_before_second_read)
    monkeypatch.setattr(io, "open", replace_before_second_read)
    data = prepare_data(config, FeatureEngineer(config.features))
    metadata = _saved_metadata(tmp_path, config, data)

    assert len(reads) == 2
    events = metadata["input_observations"]
    uses = [use for use in events["uses"] if use["role"] == "data.pit_universe_csv"]
    assert len(uses) == 2
    identities = [events["observations"][use["observation_id"]]["identity"] for use in uses]
    assert [identity["sha256"] for identity in identities] == [
        hashlib.sha256(original).hexdigest(),
        hashlib.sha256(replacement).hexdigest(),
    ]
    assert metadata["data_inputs"]["data.pit_universe_csv"]["observation_ids"] == [
        use["observation_id"] for use in uses
    ]


@pytest.mark.parametrize("zero_edges", [False, True])
def test_sector_metadata_identifies_the_csv_used_by_the_sector_parser(tmp_path, zero_edges):
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    sectors = tmp_path / "sectors.csv"
    original = b'kdcode,sector\r\nAAA,"Tech, communications"\r\nBBB,"Tech, communications"\r\n'
    sectors.write_bytes(original)
    config = _native_config(str(source))
    config.graph.use_sector_relation = True
    config.graph.sector_map_csv = str(sectors)
    config.graph.zero_edges = zero_edges
    data = prepare_data(config, FeatureEngineer(config.features))
    sectors.write_bytes(b"replaced\n")
    metadata = _saved_metadata(tmp_path, config, data)

    assert data["edge_index_sector"].tolist() == [[0, 1], [1, 0]]
    assert (
        metadata["data_inputs"]["graph.sector_map_csv"]["sha256"]
        == hashlib.sha256(original).hexdigest()
    )


def test_implicit_vix_csv_is_observed_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    vix = tmp_path / "vix_data.csv"
    original = ("dt,vix\n" + "".join(f"2020-01-{d:02d},{20 + d}\n" for d in range(1, 32))).encode()
    vix.write_bytes(original)
    config = _native_config(str(source))
    config.features.include_vix = True
    data = prepare_data(config, FeatureEngineer(config.features))
    vix.write_bytes(b"changed after prepare\n")
    metadata = _saved_metadata(tmp_path, config, data)

    identity = metadata["data_inputs"]["implicit.vix_csv"]
    assert identity["configured_path"] == "vix_data.csv"
    assert identity["resolved_path"] == str(vix)
    assert identity["sha256"] == hashlib.sha256(original).hexdigest()


def test_regime_override_identity_precedes_parsing_and_transforms(tmp_path):
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    regime = tmp_path / "regime.csv"
    original = (
        b"dt,regime_market,regime_yield_curve,regime_oil,regime_copper,regime_stock_bond_corr,regime_monetary_policy,regime_volatility\n"
        b"2020-01-01,1,2,3,4,5,6,7\n2020-01-02,2,3,4,5,6,7,8\n"
    )
    regime.write_bytes(original)
    config = _native_config(str(source))
    config.features.include_global_regime = True
    config.features.regime_inputs_csv = str(regime)
    config.features.regime_include_subsequent_returns = False
    data = prepare_data(config, FeatureEngineer(config.features))
    regime.write_bytes(b"changed after prepare\n")

    metadata = _saved_metadata(tmp_path, config, data)

    identity = metadata["data_inputs"]["features.regime_inputs_csv"]
    assert identity["sha256"] == hashlib.sha256(original).hexdigest()
    assert identity["size_bytes"] == len(original)


def test_legacy_data_file_keys_keep_cwd_semantics_where_the_resolver_disagrees(
    tmp_path, monkeypatch
):
    fallback = tmp_path / "data" / "raw" / "market"
    fallback.mkdir(parents=True)
    source = fallback / "panel.csv"
    original = _native_panel()
    source.write_bytes(original)
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    config = _native_config("missing/panel.csv")
    data = prepare_data(config, FeatureEngineer(config.features))
    higher_priority = tmp_path / "missing"
    higher_priority.mkdir()
    (higher_priority / "panel.csv").write_bytes(b"new higher priority file")

    metadata = _saved_metadata(tmp_path, config, data)

    assert metadata["data_file_sha256"] == hashlib.sha256(b"new higher priority file").hexdigest()
    assert (
        metadata["data_inputs"]["data.filename"]["sha256"] == hashlib.sha256(original).hexdigest()
    )
    assert metadata["data_inputs"]["data.filename"]["resolved_path"] == str(source)
    assert metadata["data_inputs"]["data.filename"]["configured_path"] == "missing/panel.csv"
    (higher_priority / "panel.csv").unlink()
    missing_legacy = _saved_metadata(tmp_path, config, data)
    assert missing_legacy["data_file_sha256"] is None
    assert missing_legacy["data_file_size_bytes"] is None
    assert missing_legacy["data_file_mtime_iso"] is None


def test_disabled_inputs_are_absent_and_each_preparation_has_its_own_observations(tmp_path):
    source = tmp_path / "panel.csv"
    original = _native_panel()
    source.write_bytes(original)
    config = _native_config(str(source))
    config.data.pit_universe_csv = str(tmp_path / "unused_pit.csv")
    config.graph.sector_map_csv = str(tmp_path / "unused_sector.csv")
    config.features.regime_inputs_csv = str(tmp_path / "unused_regime.csv")
    (tmp_path / "vix_data.csv").write_bytes(b"unused vix")
    first = prepare_data(config, FeatureEngineer(config.features))
    replacement = original.replace(b"1000,", b"2000,")
    source.write_bytes(replacement)
    second = prepare_data(config, FeatureEngineer(config.features))

    for window, (data, content) in enumerate([(first, original), (second, replacement)]):
        metadata = _saved_metadata(tmp_path, config, data, window)
        assert metadata["walkforward_window"] == window
        assert set(metadata["data_inputs"]) == {"data.filename"}
        assert len(metadata["input_observations"]["observations"]) == 1
        assert (
            metadata["data_inputs"]["data.filename"]["sha256"]
            == hashlib.sha256(content).hexdigest()
        )


def test_legacy_missing_file_still_warns_and_returns_nulls(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        identity = data_file_fingerprint(str(tmp_path / "missing.csv"), logger)
    assert identity == {
        "data_file_sha256": None,
        "data_file_size_bytes": None,
        "data_file_mtime_iso": None,
    }
    assert "skipping sha256" in caplog.text


def test_failed_optional_csv_parse_is_recorded_without_changing_continuation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    vix = tmp_path / "regime.csv"
    vix.write_bytes(b"")
    config = _native_config(str(source))
    config.features.include_global_regime = True
    config.features.regime_inputs_csv = "regime.csv"
    data = prepare_data(config, FeatureEngineer(config.features))

    metadata = _saved_metadata(tmp_path, config, data)

    events = metadata["input_observations"]["observations"]
    failed = [event for event in events if event["outcome"] == "error"]
    assert len(failed) == 1
    assert failed[0]["stage"] == "parse"
    assert failed[0]["error_code"] == "EmptyDataError"
    assert failed[0]["configured_path"] == "regime.csv"
    assert failed[0]["identity"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert "features.regime_inputs_csv" not in metadata["data_inputs"]
    assert vix.read_bytes() == b""


@pytest.mark.parametrize("route", ["vix", "credit", "regime"])
@pytest.mark.parametrize("integrity", [True, False])
def test_preparation_propagates_observation_integrity_errors_but_preserves_ordinary_errors(
    tmp_path, monkeypatch, route, integrity
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    config = _native_config(str(source))
    error_type = InputObservationError if integrity else ValueError
    if route == "credit":
        config.features.include_credit_spread = True
        monkeypatch.setenv("FRED_API_KEY", "synthetic-never-sent")

        class FaultingFred:
            def __init__(self, **kwargs):
                pass

            def get_series(self, *args, **kwargs):
                raise error_type("synthetic input failure")

        monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=FaultingFred))
    else:
        auxiliary = tmp_path / ("vix_data.csv" if route == "vix" else "regime.csv")
        auxiliary.write_bytes(b"present but read will fail")
        if route == "vix":
            config.features.include_vix = True
        else:
            config.features.include_global_regime = True
            config.features.regime_strict = False
            config.features.regime_inputs_csv = str(auxiliary)
        real_open = io.open

        def fail_auxiliary_read(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and Path(file) == auxiliary:
                raise error_type("synthetic input failure")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(io, "open", fail_auxiliary_read)
    if integrity:
        with pytest.raises(InputObservationError, match="synthetic input failure"):
            prepare_data(config, FeatureEngineer(config.features))
    elif route == "vix":
        with pytest.raises(ValueError, match="vix_df not provided"):
            prepare_data(config, FeatureEngineer(config.features))
    else:
        data = prepare_data(config, FeatureEngineer(config.features))
        metadata = _saved_metadata(tmp_path, config, data)
        assert set(metadata["data_inputs"]) == {"data.filename"}


@pytest.mark.parametrize("invalid", ["columns", "date"])
def test_rejected_regime_csv_remains_an_observation_without_a_consumption_link(tmp_path, invalid):
    source = tmp_path / "panel.csv"
    source.write_bytes(_native_panel())
    regime = tmp_path / "regime.csv"
    content = (
        b"dt,regime_market\n2020-01-01,1\n"
        if invalid == "columns"
        else b"dt,regime_market,regime_yield_curve,regime_oil,regime_copper,regime_stock_bond_corr,regime_monetary_policy,regime_volatility\ninvalid-date,1,2,3,4,5,6,7\n"
    )
    regime.write_bytes(content)
    config = _native_config(str(source))
    config.features.include_global_regime = True
    config.features.regime_strict = False
    config.features.regime_inputs_csv = str(regime)
    data = prepare_data(config, FeatureEngineer(config.features))
    metadata = _saved_metadata(tmp_path, config, data)

    assert set(metadata["data_inputs"]) == {"data.filename"}
    events = metadata["input_observations"]["observations"]
    rejected = [event for event in events if event["role"] == "features.regime_inputs_csv"]
    assert len(rejected) == 1
    assert rejected[0]["outcome"] == "error"
    assert rejected[0]["stage"] == "parse"
    assert rejected[0]["identity"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert regime.read_bytes() == content
