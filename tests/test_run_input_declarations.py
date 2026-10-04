"""Runner wiring for saved-run inputs: configured roles bound to declared packages (#208).

Each preparation here is real: synthetic package files are read through the
landed loaders, FRED is a local fake, and the attachment is verified only after
the package, its manifest and every captured snapshot have been removed.
"""

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mci_gru.config import DataConfig, create_config_from_dict
from mci_gru.data.input_manifest import (
    InputFileSpec,
    ManifestDigestMismatchError,
    read_input_manifest,
    write_input_manifest,
)
from mci_gru.data.input_observations import InputObservationContext
from mci_gru.evaluation.run_input_attachments import AttachmentReference, read_run_inputs
from mci_gru.evaluation.run_input_declarations import (
    attach_window_inputs,
    declare_window_inputs,
    keep_captures_with_run,
    required_input_roles,
)
from mci_gru.features import FeatureEngineer
from mci_gru.pipeline import prepare_data, prepare_data_index_level

REPO_ROOT = Path(__file__).resolve().parent.parent
REGIME_ROLES = [
    "fred.regime_market",
    "fred.yield_10y",
    "fred.yield_3m",
    "fred.regime_oil",
    "fred.regime_copper",
    "fred.regime_volatility",
]
STOCKS = ["AAA", "BBB", "CCC", "DDD"]
PROVENANCE = {
    "source": "synthetic fixture",
    "acquisition_mode": "fixture",
    "acquired_at": "2020-04-01T00:00:00+00:00",
    "producing_command": None,
    "producing_arguments": None,
    "unknowns": ["Fixture packages have no producing command"],
}


class Fred:
    """A local FRED client: daily series on weekdays, copper monthly."""

    def __init__(self, api_key):
        del api_key

    def get_series(self, series_id, observation_start, observation_end):
        frequency = "MS" if series_id == "PCOPPUSDM" else "B"
        dates = pd.date_range(observation_start, observation_end, freq=frequency)
        return pd.Series(np.linspace(10.0, 100.0, len(dates)), index=dates)


@dataclass
class Setup:
    root: Path
    package: Path
    manifest: Path
    data: dict


def _write_panel(path: Path) -> None:
    rows = []
    for index, stock in enumerate(STOCKS):
        price = 100.0 + index
        for day, dt in enumerate(pd.bdate_range("2019-06-03", "2020-03-31")):
            close = price * (1.0 + 0.003 * np.sin(day / 5.0 + index))
            rows.append(
                {
                    "kdcode": stock,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "open": price,
                    "high": max(price, close) * 1.001,
                    "low": min(price, close) * 0.999,
                    "close": close,
                    "volume": 1_000_000.0 + day,
                }
            )
            price = close
    pd.DataFrame(rows).to_csv(path, index=False)


def _setup(tmp_path: Path, monkeypatch, *, snapshot_mode: str = "capture") -> Setup:
    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    package = tmp_path / "raw"
    (package / "market").mkdir(parents=True)
    (package / "constituents").mkdir()
    _write_panel(package / "market" / "panel.csv")
    pd.DataFrame({"kdcode": STOCKS, "valid_from": "2019-01-01", "valid_to": "2020-03-31"}).to_csv(
        package / "constituents" / "pit.csv", index=False
    )
    manifest = tmp_path / "declared" / "synthetic.r1.json"
    manifest.parent.mkdir()
    snapshot = write_input_manifest(
        manifest,
        package,
        package_id="synthetic-stock",
        package_revision="r1",
        files=[
            InputFileSpec("market/panel.csv", "price panel"),
            InputFileSpec("constituents/pit.csv", "point-in-time universe"),
        ],
        provenance=PROVENANCE,
    )
    data = {
        "filename": str(package / "market" / "panel.csv"),
        "use_pit_universe": True,
        "pit_universe_csv": str(package / "constituents" / "pit.csv"),
        "pit_min_scoreable_stocks": 1,
        "train_start": "2020-01-02",
        "train_end": "2020-02-14",
        "val_start": "2020-02-18",
        "val_end": "2020-02-28",
        "test_start": "2020-03-03",
        "test_end": "2020-03-20",
        "auxiliary_sources": {"regime": "fred"},
        "auxiliary_snapshot_mode": snapshot_mode,
        "auxiliary_snapshot_directory": str(tmp_path / "snapshots"),
        "input_package_manifest": str(manifest),
        "input_package_manifest_sha256": snapshot.sha256,
        "input_package_root": str(package),
    }
    return Setup(tmp_path, package, manifest, data)


def _config(data: dict, **features):
    return create_config_from_dict(
        {
            "data": data,
            "features": {
                "include_momentum": False,
                "include_weekly_momentum": False,
                "include_global_regime": True,
                **features,
            },
            "model": {"his_t": 3, "label_t": 2},
            "graph": {"use_multi_feature_edges": False},
            "training": {"label_type": "returns"},
            "tracking": {"enabled": False},
        }
    )


def _prepare(config):
    prepare = (
        prepare_data_index_level if config.data.experiment_mode == "index_level" else prepare_data
    )
    return prepare(config, FeatureEngineer(config.features))["input_observations"]


def _attach(setup: Setup, config, observations) -> AttachmentReference:
    window = setup.root / "run" / "window"
    attached = attach_window_inputs(
        config,
        observations,
        window,
        attempt_id="a" * 32,
        execution_start={"path": "execution_provenance/start.json", "sha256": "b" * 64},
        logger=logging.getLogger(__name__),
    )
    assert attached["path"] == f"input_attachments/{'a' * 32}/record.json"
    return AttachmentReference(window / attached["path"], attached["sha256"])


def _relocated(setup: Setup, reference: AttachmentReference) -> AttachmentReference:
    """Move the run away and remove the package, its manifest and every snapshot."""
    moved = setup.root / "elsewhere"
    shutil.move(setup.root / "run", moved)
    for source in (setup.package, setup.manifest.parent, setup.root / "snapshots"):
        if source.exists():
            shutil.rmtree(source)
    window = setup.root / "run" / "window"
    return AttachmentReference(
        moved / "window" / reference.path.relative_to(window), reference.sha256
    )


def _bindings(inputs) -> dict[str, tuple]:
    return {role["role"]: (role["manifest_sha256"], role["path"]) for role in inputs.roles}


def test_a_captured_window_is_complete_after_every_source_is_gone(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    config = _config(setup.data)
    observations = _prepare(config)

    inputs = read_run_inputs(_relocated(setup, _attach(setup, config, observations)))

    assert (inputs.status, inputs.problems) == ("complete", [])
    bindings = _bindings(inputs)
    pinned = setup.data["input_package_manifest_sha256"]
    assert bindings["data.filename"] == (pinned, "market/panel.csv")
    assert bindings["data.pit_universe_csv"] == (pinned, "constituents/pit.csv")
    # Each FRED role binds to the snapshot package that retained its own bytes.
    snapshot_digests = {bindings[role][0] for role in REGIME_ROLES}
    assert len(snapshot_digests) == 6 and pinned not in snapshot_digests
    assert {bindings[role][1] for role in REGIME_ROLES} == {"observations.bin"}
    assert all(role["observation_ids"] for role in inputs.roles)
    assert inputs.execution == {
        "status": "unknown",
        "start": {"path": "execution_provenance/start.json", "sha256": "b" * 64},
    }
    assert inputs.preservation == {"status": "unproven"}


def test_replay_binds_the_snapshots_the_configuration_references(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    _prepare(_config(setup.data))
    captured = sorted((setup.root / "snapshots").glob("*/manifest.json"))
    references = {}
    for manifest in captured:
        snapshot = read_input_manifest(manifest)
        references.setdefault(snapshot.manifest.package_id, []).append(
            {"manifest_path": str(manifest), "manifest_sha256": snapshot.sha256}
        )
    monkeypatch.delenv("FRED_API_KEY")
    monkeypatch.setitem(sys.modules, "fredapi", None)
    replay = _config(
        {
            **setup.data,
            "auxiliary_snapshot_mode": "replay",
            "auxiliary_snapshot_references": references,
        }
    )

    inputs = read_run_inputs(_relocated(setup, _attach(setup, replay, _prepare(replay))))

    assert (inputs.status, inputs.problems) == ("complete", [])
    declared = {package["manifest_sha256"] for package in inputs.packages}
    assert {_bindings(inputs)[role][0] for role in REGIME_ROLES} == {
        item["manifest_sha256"] for values in references.values() for item in values
    }
    assert declared >= {_bindings(inputs)[role][0] for role in REGIME_ROLES}


def test_live_provider_reads_are_linked_but_never_complete(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch, snapshot_mode="source")
    config = _config(setup.data)

    inputs = read_run_inputs(_relocated(setup, _attach(setup, config, _prepare(config))))

    assert inputs.status == "incomplete"
    assert sorted(inputs.problems) == sorted(
        f"required role {role} has no declared package" for role in REGIME_ROLES
    )
    links = {role["role"]: role["observation_ids"] for role in inputs.roles}
    assert all(links[role] for role in REGIME_ROLES)
    assert _bindings(inputs)["data.filename"][1] == "market/panel.csv"


def test_a_selected_file_outside_the_declared_package_has_no_declared_package(
    tmp_path, monkeypatch
):
    setup = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere-pit.csv"
    shutil.copyfile(setup.package / "constituents" / "pit.csv", outside)
    config = _config({**setup.data, "pit_universe_csv": str(outside)})

    inputs = read_run_inputs(_relocated(setup, _attach(setup, config, _prepare(config))))

    assert inputs.status == "incomplete"
    assert inputs.problems == ["required role data.pit_universe_csv has no declared package"]


def test_a_package_file_changed_after_its_manifest_is_not_the_declared_read(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    panel = setup.package / "market" / "panel.csv"
    panel.write_bytes(panel.read_bytes() + b"\n")
    config = _config(setup.data)

    inputs = read_run_inputs(_relocated(setup, _attach(setup, config, _prepare(config))))

    assert inputs.status == "incomplete"
    assert inputs.problems[0].startswith("observed input differs for role data.filename")
    assert "required role data.filename was not consumed from market/panel.csv" in inputs.problems


def test_a_declared_manifest_that_is_not_the_pinned_one_stops_the_run(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    observations = InputObservationContext().freeze()
    edited = {**setup.data, "input_package_manifest_sha256": "0" * 64}
    with pytest.raises(ManifestDigestMismatchError):
        declare_window_inputs(_config(edited), observations)
    setup.manifest.unlink()
    with pytest.raises(FileNotFoundError):
        declare_window_inputs(_config(setup.data), observations)


def test_a_snapshot_manifest_changed_after_capture_is_refused(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    config = _config(setup.data)
    observations = _prepare(config)
    manifest = sorted((setup.root / "snapshots").glob("*/manifest.json"))[0]
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(ManifestDigestMismatchError):
        declare_window_inputs(config, observations)


@pytest.mark.parametrize(
    "case",
    [
        "recipe",
        "credit_and_vix_file",
        "legacy_regime_csv_without_pit",
        "cessation_and_sector_map",
        "index_file",
        "index_fred",
    ],
)
def test_required_roles_are_exactly_the_roles_a_real_preparation_consumes(
    tmp_path, monkeypatch, case
):
    setup = _setup(tmp_path, monkeypatch, snapshot_mode="source")
    data = dict(setup.data)
    features = {}
    if case == "credit_and_vix_file":
        monkeypatch.setattr("mci_gru.data.path_resolver.PROJECT_ROOT", tmp_path)
        vix = tmp_path / "data" / "raw" / "market" / "vix_data.csv"
        vix.parent.mkdir(parents=True)
        dates = pd.bdate_range("2019-06-03", "2020-03-31")
        pd.DataFrame(
            {"dt": dates.strftime("%Y-%m-%d"), "close": np.linspace(15, 25, len(dates))}
        ).to_csv(vix, index=False)
        data["auxiliary_sources"] = {"regime": "fred", "credit": "fred", "vix": "file"}
        features = {"include_vix": True, "include_credit_spread": True}
    if case == "legacy_regime_csv_without_pit":
        data["use_pit_universe"] = False
        regime = tmp_path / "regime.csv"
        dates = pd.bdate_range("2004-01-01", "2020-03-31")
        columns = [
            "regime_market",
            "regime_yield_curve",
            "regime_oil",
            "regime_copper",
            "regime_stock_bond_corr",
            "regime_monetary_policy",
            "regime_volatility",
        ]
        pd.DataFrame(
            {
                "dt": dates.strftime("%Y-%m-%d"),
                **{column: np.linspace(1.0, 2.0, len(dates)) for column in columns},
            }
        ).to_csv(regime, index=False)
        features = {"regime_inputs_csv": str(regime)}
    graph = {}
    if case == "cessation_and_sector_map":
        cessation = tmp_path / "cessation.csv"
        cessation.write_text("event_id,kdcode,effective_at,known_from\n", encoding="utf-8")
        data["pit_cessation_events_csv"] = str(cessation)
        sectors = tmp_path / "sectors.csv"
        pd.DataFrame({"kdcode": STOCKS, "sector": ["Tech", "Tech", "Energy", "Energy"]}).to_csv(
            sectors, index=False
        )
        graph = {"use_sector_relation": True, "sector_map_csv": str(sectors)}
    if case.startswith("index"):
        data["experiment_mode"] = "index_level"
        if case == "index_file":
            index = tmp_path / "index.csv"
            dates = pd.bdate_range("2017-01-02", "2020-03-31")
            pd.DataFrame(
                {"dt": dates.strftime("%Y-%m-%d"), "close": np.linspace(2000, 3000, len(dates))}
            ).to_csv(index, index=False)
            data["index_filename"] = str(index)
        else:
            data["auxiliary_sources"] = {"regime": "fred", "index": "fred"}
    config = _config(data, **features)
    for key, value in graph.items():
        setattr(config.graph, key, value)

    observations = _prepare(config)

    required = [entry.role for entry in required_input_roles(config)]
    assert len(required) == len(set(required))
    assert set(required) == {use.role for use in observations.uses}


def _recipe_config(overrides: list[str]):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))


def test_the_recipe_data_config_binds_both_selected_files_to_the_preserved_package():
    config = _recipe_config(["features.include_global_regime=true"])
    observations = InputObservationContext().freeze()

    declarations = declare_window_inputs(config, observations)

    (package,) = declarations.manifests
    assert package.manifest.package_id == ("sp500_pit_gics_top10_mcap_monthly_20160104_20260731")
    assert package.sha256 == config.data.input_package_manifest_sha256
    files = {record.path for record in package.manifest.files}
    bindings = {binding.role: binding for binding in declarations.required}
    for role in ("data.filename", "data.pit_universe_csv"):
        assert bindings[role].manifest_sha256 == package.sha256
        assert bindings[role].path in files
    assert bindings["data.filename"].path.startswith("market/")
    assert bindings["data.pit_universe_csv"].path.startswith("constituents/")
    # The recipe's six regime inputs are provider reads: declared only once captured.
    assert {role for role, binding in bindings.items() if binding.manifest_sha256 is None} == set(
        REGIME_ROLES
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_package_manifest_sha256": None}, "set together or not at all"),
        ({"input_package_root": None}, "set together or not at all"),
        ({"input_package_manifest_sha256": "A" * 64}, "64 lowercase hex"),
        ({"input_package_manifest_sha256": "a" * 63}, "64 lowercase hex"),
    ],
)
def test_a_partial_or_malformed_package_declaration_is_refused(
    tmp_path, monkeypatch, overrides, message
):
    setup = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        _config({**setup.data, **overrides})


def test_a_capture_with_no_named_folder_keeps_its_snapshots_in_the_run(tmp_path):
    data = DataConfig(auxiliary_snapshot_mode="capture")

    chosen = keep_captures_with_run(data, tmp_path / "run")

    assert chosen == data.auxiliary_snapshot_directory == str(tmp_path / "run" / "input_snapshots")


@pytest.mark.parametrize(
    "data",
    [
        DataConfig(auxiliary_snapshot_mode="capture", auxiliary_snapshot_directory="named"),
        DataConfig(auxiliary_snapshot_mode="source"),
        DataConfig(auxiliary_snapshot_mode="replay"),
    ],
)
def test_a_named_folder_or_another_mode_is_left_as_configured(tmp_path, data):
    before = data.auxiliary_snapshot_directory

    assert keep_captures_with_run(data, tmp_path) is None
    assert data.auxiliary_snapshot_directory == before
