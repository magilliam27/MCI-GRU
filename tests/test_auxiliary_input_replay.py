"""Public auxiliary loader and preparation capture/replay contracts."""

import gzip
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import DataConfig, create_config_from_dict
from mci_gru.data.data_manager import DataManager
from mci_gru.data.fred_loader import FREDLoader
from mci_gru.data.input_observations import InputObservationContext, InputObservationError
from mci_gru.data.input_snapshots import InputSnapshots, load_snapshot
from mci_gru.data.lseg_loader import LSEGLoader
from mci_gru.features import FeatureEngineer
from mci_gru.pipeline import prepare_data, prepare_data_index_level


def test_fred_replay_uses_original_sdk_series_before_fill_and_lag(
    tmp_path: Path, monkeypatch
) -> None:
    raw = pd.Series(
        [5.0, 1.0, np.nan, 3.0, 4.0],
        index=pd.to_datetime(
            ["2020-01-05", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
        ),
        name="observation",
    )
    calls = []

    class Fred:
        def __init__(self, api_key):
            calls.append("setup")

        def get_series(self, series_id, **request):
            calls.append((series_id, request))
            return raw.copy()

    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    context = InputObservationContext()
    snapshots = InputSnapshots(
        mode="capture", directory=tmp_path / "snapshots", input_observations=context
    )
    source = FREDLoader(api_key="test-only", snapshots=snapshots).get_series(
        "DGS10", "2020-01-03", "2020-01-05", "yield_10y"
    )
    expected = pd.DataFrame(
        {"dt": ["2020-01-03", "2020-01-04", "2020-01-05"], "yield_10y": [1.0, 3.0, 4.0]}
    )
    pd.testing.assert_frame_equal(source, expected)
    record = context.freeze().to_dict()["observations"][0]
    retained = load_snapshot(
        next(iter(snapshots.references.values()))[0],
        role=record["role"],
        source="fred",
        request=record["request"],
    )
    pd.testing.assert_series_equal(retained.data, raw, check_exact=True)
    assert record["identity"]["manifest_sha256"] == retained.manifest.sha256
    assert len(context.freeze().uses) == 1
    forbidden = []

    class DeniedFred:
        def __init__(self, *args, **kwargs):
            forbidden.append("setup")
            raise AssertionError("Replay started FRED")

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=DeniedFred))
    replay_context = InputObservationContext()
    replay_store = InputSnapshots(
        mode="replay", references=snapshots.references, input_observations=replay_context
    )
    replay = FREDLoader(snapshots=replay_store).get_series(
        "DGS10", "2020-01-03", "2020-01-05", "yield_10y"
    )
    pd.testing.assert_frame_equal(replay, expected)
    assert forbidden == []
    assert len(calls) == 2
    replay_record = replay_context.freeze().to_dict()["observations"][0]
    assert replay_record["acquired_at"] == record["acquired_at"]
    assert replay_record["identity"] == record["identity"]


def test_credit_replay_keeps_both_original_series_and_the_existing_lag(
    tmp_path: Path, monkeypatch
) -> None:
    raw = {
        "BAMLC0A0CM": pd.Series(
            [1.0, np.nan, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4)
        ),
        "BAMLH0A0HYM2": pd.Series(
            [5.0, 6.0, 7.0, 8.0], index=pd.date_range("2020-01-01", periods=4)
        ),
    }

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, **kwargs):
            return raw[series_id].copy()

    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    store = InputSnapshots(mode="capture", directory=tmp_path / "snapshots")
    source = FREDLoader(api_key="test-only", snapshots=store).get_credit_spreads(
        "2020-01-02", "2020-01-04"
    )
    assert len(store.references) == 2
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    replay_store = InputSnapshots(mode="replay", references=store.references)
    replay = FREDLoader(snapshots=replay_store).get_credit_spreads("2020-01-02", "2020-01-04")
    expected = pd.DataFrame(
        {
            "dt": ["2020-01-02", "2020-01-03", "2020-01-04"],
            "ig_spread": [1.0, 1.0, 3.0],
            "hy_spread": [5.0, 6.0, 7.0],
        }
    )
    pd.testing.assert_frame_equal(source, expected)
    pd.testing.assert_frame_equal(replay, expected)
    assert {use.role for use in replay_store.input_observations.freeze().uses} == {
        "fred.ig_spread",
        "fred.hy_spread",
    }


@pytest.mark.parametrize("method", ["vix", "series"])
def test_lseg_replay_preserves_raw_history_without_opening_a_session(
    tmp_path: Path, monkeypatch, method: str
) -> None:
    raw = pd.DataFrame(
        {"CLOSE": [21.0, 19.0, 20.0]},
        index=pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"], name="Date"),
    )
    calls = []

    def history(**request):
        calls.append(request)
        return raw.copy()

    fake = SimpleNamespace(
        open_session=lambda: calls.append("setup"), close_session=lambda: None, get_history=history
    )
    monkeypatch.setitem(sys.modules, "refinitiv", SimpleNamespace(data=fake))
    monkeypatch.setitem(sys.modules, "refinitiv.data", fake)
    store = InputSnapshots(mode="capture", directory=tmp_path / "snapshots")
    loader = LSEGLoader(snapshots=store)
    loader.connect()
    source = (
        loader.get_vix("2020-01-01", "2020-01-03")
        if method == "vix"
        else loader.get_series(".VIX", "2020-01-01", "2020-01-03", "vix")
    )
    loader.disconnect()
    assert len(store.references) == 1
    record = store.input_observations.freeze().to_dict()["observations"][0]
    retained = load_snapshot(
        next(iter(store.references.values()))[0],
        role=record["role"],
        source="lseg",
        request=record["request"],
    )
    pd.testing.assert_frame_equal(retained.data, raw, check_exact=True)
    forbidden = []

    def deny_session():
        forbidden.append("setup")
        raise AssertionError("Replay opened LSEG session")

    fake.open_session = deny_session
    fake.get_history = lambda **kwargs: forbidden.append("history")
    replay_store = InputSnapshots(mode="replay", references=store.references)
    replay_loader = LSEGLoader(snapshots=replay_store)
    replay_loader.connect()
    replay = (
        replay_loader.get_vix("2020-01-01", "2020-01-03")
        if method == "vix"
        else replay_loader.get_series(".VIX", "2020-01-01", "2020-01-03", "vix")
    )
    replay_loader.disconnect()
    pd.testing.assert_frame_equal(source, replay)
    assert source["vix"].tolist() == ([21.0, 19.0, 20.0] if method == "vix" else [19.0, 20.0, 21.0])
    assert forbidden == []
    assert calls[1]["universe"] == [".VIX"]


@pytest.mark.parametrize("provider", ["fred", "lseg"])
@pytest.mark.parametrize("fault", ["missing_replay", "source_error"])
def test_required_provider_failure_retains_safe_facts_without_fallback_or_use(
    tmp_path: Path, monkeypatch, provider: str, fault: str
) -> None:
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("secret-token=do-not-retain")

    class Fred:
        def __init__(self, api_key):
            calls.append("setup")

        def get_series(self, series_id, **kwargs):
            return fail(series_id=series_id, **kwargs)

    fake = SimpleNamespace(open_session=lambda: calls.append("setup"), get_history=fail)
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    monkeypatch.setitem(sys.modules, "refinitiv.data", fake)
    store = InputSnapshots(
        mode="replay" if fault == "missing_replay" else "capture", directory=tmp_path / "snapshots"
    )
    loader = (
        FREDLoader(api_key="test-only", snapshots=store)
        if provider == "fred"
        else LSEGLoader(snapshots=store)
    )
    if provider == "lseg":
        loader.connect()
    with pytest.raises(InputObservationError) as caught:
        if provider == "fred":
            loader.get_series("DGS10", "2020-01-01", "2020-01-03", "yield_10y")
        else:
            loader.get_vix("2020-01-01", "2020-01-03")
    assert caught.value.facts["source"] == provider
    evidence = store.input_observations.freeze().to_dict()
    assert len(evidence["observations"]) == 1
    assert evidence["observations"][0]["outcome"] == "error"
    assert evidence["uses"] == []
    assert "secret-token" not in str(evidence) + str(caught.value.facts) + str(caught.value)
    assert len(calls) == (0 if fault == "missing_replay" else 2)


def test_data_manager_regime_capture_and_replay_account_for_all_six_sources(
    tmp_path: Path, monkeypatch
) -> None:
    dates = pd.date_range("2017-01-01", periods=1200)
    series_ids = ["DGS10", "DGS3MO", "DCOILWTICO", "VIXCLS", "SP500", "PCOPPUSDM"]
    calls = []

    class Fred:
        def __init__(self, api_key):
            calls.append("setup")

        def get_series(self, series_id, **kwargs):
            calls.append(series_id)
            return pd.Series(
                np.linspace(10.0, 100.0, len(dates)) + series_ids.index(series_id), index=dates
            )

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    config = DataConfig(
        auxiliary_sources={"regime": "fred"},
        auxiliary_snapshot_mode="capture",
        auxiliary_snapshot_directory=str(tmp_path / "snapshots"),
    )
    manager = DataManager(config)
    source = manager.load_regime_inputs()
    recorded = manager.input_observations.freeze().to_dict()
    assert len(recorded["observations"]) == len(recorded["uses"]) == 6
    assert {record["request"]["series_id"] for record in recorded["observations"]} == set(
        series_ids
    )
    config.auxiliary_snapshot_mode = "replay"
    config.auxiliary_snapshot_references = {
        key: [
            {**asdict(reference), "manifest_path": str(reference.manifest_path)}
            for reference in values
        ]
        for key, values in manager.input_snapshots.references.items()
    }
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    replay_manager = DataManager(config)
    replay = replay_manager.load_regime_inputs()
    pd.testing.assert_frame_equal(replay, source, check_exact=True)
    assert len(calls) == 7
    assert len(replay_manager.input_observations.freeze().uses) == 6


def _preparation_config(tmp_path: Path):
    rows = ["kdcode,dt,open,high,low,close,volume"]
    for day in range(1, 32):
        for stock, offset in [("AAA", 0), ("BBB", 20)]:
            value = 100 + offset + day
            rows.append(f"{stock},2020-01-{day:02d},{value},{value + 1},{value - 1},{value},1000")
    panel = tmp_path / "panel.csv"
    panel.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return create_config_from_dict(
        {
            "data": {
                "filename": str(panel),
                "train_start": "2020-01-05",
                "train_end": "2020-01-15",
                "val_start": "2020-01-19",
                "val_end": "2020-01-23",
                "test_start": "2020-01-27",
                "test_end": "2020-01-31",
                "use_pit_universe": False,
            },
            "features": {"include_momentum": False, "include_weekly_momentum": False},
            "model": {"his_t": 2, "label_t": 2},
            "graph": {"use_multi_feature_edges": False},
            "training": {"label_type": "returns"},
            "tracking": {"enabled": False},
        }
    )


@pytest.mark.parametrize("role", ["vix", "credit", "regime", "index"])
def test_preparation_requires_selected_file_inputs_without_implicit_provider_setup(
    tmp_path: Path, monkeypatch, role: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mci_gru.data.path_resolver.PROJECT_ROOT", tmp_path)
    config = _preparation_config(tmp_path)
    if role != "index":
        setattr(
            config.features,
            {
                "vix": "include_vix",
                "credit": "include_credit_spread",
                "regime": "include_global_regime",
            }[role],
            True,
        )
    if role == "regime":
        config.features.regime_inputs_csv = str(tmp_path / "missing-regime.csv")
    calls = []

    def deny(*args, **kwargs):
        calls.append("provider")
        raise AssertionError("File mode started a provider")

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=deny))
    monkeypatch.setitem(sys.modules, "refinitiv.data", SimpleNamespace(open_session=deny))
    original = Path(config.data.filename).read_bytes()
    with pytest.raises(InputObservationError) as caught:
        (prepare_data_index_level if role == "index" else prepare_data)(
            config, FeatureEngineer(config.features)
        )
    failure = caught.value
    assert failure.facts["required"] is True
    assert failure.facts["source"] == "file"
    assert failure.facts["stage"] == "resolve"
    assert failure.facts["reason_code"]
    observations = failure.input_observations
    assert set(observations.data_inputs()) == (set() if role == "index" else {"data.filename"})
    # Resolution can fail before there is an observation to contribute.
    assert "observed_sha256" not in failure.facts
    assert calls == []
    assert Path(config.data.filename).read_bytes() == original


@pytest.mark.parametrize(
    "role,method",
    [("vix", "load_vix"), ("credit", "load_credit_spreads"), ("index", "load_index_series")],
)
def test_data_manager_replays_explicit_auxiliary_sources_with_a_csv_stock_config(
    tmp_path: Path, monkeypatch, role: str, method: str
) -> None:
    config = _preparation_config(tmp_path).data
    config.auxiliary_sources = {role: "lseg" if role == "vix" else "fred"}
    config.auxiliary_snapshot_mode = "capture"
    config.auxiliary_snapshot_directory = str(tmp_path / "snapshots")
    dates = pd.date_range("2020-01-01", periods=31)
    raw = pd.Series(np.arange(31, dtype=float) + 10, index=dates)

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, *args, **kwargs):
            return raw.copy()

    fake = SimpleNamespace(
        open_session=lambda: None,
        close_session=lambda: None,
        get_history=lambda **kwargs: pd.DataFrame({"CLOSE": raw}).rename_axis("Date"),
    )
    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    monkeypatch.setitem(sys.modules, "refinitiv.data", fake)
    manager = DataManager(config)
    source = getattr(manager, method)()
    assert len(manager.input_snapshots.references) == (2 if role == "credit" else 1)
    config.auxiliary_snapshot_mode = "replay"
    config.auxiliary_snapshot_references = {
        key: [
            {**asdict(reference), "manifest_path": str(reference.manifest_path)}
            for reference in values
        ]
        for key, values in manager.input_snapshots.references.items()
    }
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    monkeypatch.setitem(sys.modules, "refinitiv.data", None)
    replay_manager = DataManager(config)
    replay = getattr(replay_manager, method)()
    pd.testing.assert_frame_equal(source, replay, check_exact=True)
    assert len(replay_manager.input_observations.freeze().uses) == (2 if role == "credit" else 1)


@pytest.mark.parametrize("enabled", [True, False])
def test_preparation_returns_complete_replayable_input_identities_for_enabled_roles(
    tmp_path: Path, monkeypatch, enabled: bool
) -> None:
    config = _preparation_config(tmp_path)
    config.features.include_vix = enabled
    config.features.include_credit_spread = enabled
    config.features.include_global_regime = enabled
    config.data.auxiliary_sources = {"vix": "lseg", "credit": "fred", "regime": "fred"}
    config.data.auxiliary_snapshot_mode = "capture"
    config.data.auxiliary_snapshot_directory = str(tmp_path / "snapshots")
    dates = pd.date_range("2010-01-01", "2020-01-31")
    raw = pd.Series(
        100.0 + np.arange(len(dates)) * 0.01 + np.sin(np.arange(len(dates)) / 10), index=dates
    )
    calls = []

    class Fred:
        def __init__(self, api_key):
            calls.append("fred_setup")

        def get_series(self, series_id, **kwargs):
            calls.append(series_id)
            return raw.copy()

    fake = SimpleNamespace(
        open_session=lambda: calls.append("lseg_setup"),
        close_session=lambda: None,
        get_history=lambda **kwargs: pd.DataFrame({"CLOSE": raw}).rename_axis("Date"),
    )
    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    monkeypatch.setitem(sys.modules, "refinitiv.data", fake)
    source = prepare_data(config, FeatureEngineer(config.features))
    observed = source["input_observations"]
    expected_roles = {"data.filename"}
    if enabled:
        expected_roles |= {
            "lseg.vix",
            "fred.ig_spread",
            "fred.hy_spread",
            "fred.yield_10y",
            "fred.yield_3m",
            "fred.regime_oil",
            "fred.regime_volatility",
            "fred.regime_market",
            "fred.regime_copper",
        }
    assert set(observed.data_inputs()) == expected_roles
    references = {}
    for record in observed.to_dict()["observations"]:
        if record["role"] != "data.filename":
            identity = record["identity"]
            references.setdefault(identity["snapshot_key"], []).append(
                {
                    "manifest_path": identity["manifest_path"],
                    "manifest_sha256": identity["manifest_sha256"],
                }
            )
    config.data.auxiliary_snapshot_mode = "replay"
    config.data.auxiliary_snapshot_references = references
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    monkeypatch.setitem(sys.modules, "refinitiv.data", None)
    replay = prepare_data(config, FeatureEngineer(config.features))
    pd.testing.assert_frame_equal(source["df"], replay["df"], check_exact=True)
    assert set(replay["input_observations"].data_inputs()) == expected_roles
    if not enabled:
        assert calls == []
        assert not (tmp_path / "snapshots").exists()


@pytest.mark.parametrize("role", ["vix", "regime"])
def test_data_manager_file_capture_replays_the_original_read_buffer(
    tmp_path: Path, monkeypatch, role: str
) -> None:
    monkeypatch.chdir(tmp_path)
    if role == "vix":
        content = b"dt,vix\r\n2020-01-03,21\r\n2020-01-01,19\r\n2020-01-02,\r\n"
        original = tmp_path / "vix_data.csv"
    else:
        content = gzip.compress(
            b"dt,regime_market,regime_yield_curve,regime_oil,regime_copper,regime_stock_bond_corr,regime_monetary_policy,regime_volatility\n2020-01-03,1,2,3,4,5,6,7\n2020-01-01,8,9,10,11,12,13,14\n2020-01-02,2,3,4,5,6,7,8\n",
            mtime=0,
        )
        original = tmp_path / "regime.csv.gz"
    original.write_bytes(content)
    original_stat = original.stat()
    config = DataConfig(
        auxiliary_snapshot_mode="capture", auxiliary_snapshot_directory=str(tmp_path / "snapshots")
    )
    manager = DataManager(config)
    source = (
        manager.load_vix()
        if role == "vix"
        else manager.load_regime_inputs(regime_inputs_csv=str(original), regime_enforce_lag_days=1)
    )
    evidence = manager.input_observations.freeze().to_dict()
    publications = [event for event in evidence["observations"] if event["stage"] == "retain"]
    assert len(publications) == 1
    assert len([event for event in evidence["observations"] if event["stage"] == "parse"]) == 1
    publication = publications[0]
    used_ids = {use["observation_id"] for use in evidence["uses"]}
    assert publication["observation_id"] in used_ids
    native_parse = next(event for event in evidence["observations"] if event["stage"] == "parse")
    assert native_parse["observation_id"] in used_ids
    assert publication["identity"]["sha256"] == hashlib.sha256(content).hexdigest()
    saved = load_snapshot(
        next(iter(manager.input_snapshots.references.values()))[0],
        role=publication["role"],
        source="file",
        request=publication["request"],
    )
    assert saved.data == content
    assert original.read_bytes() == content
    assert original.stat().st_mtime_ns == original_stat.st_mtime_ns
    config.auxiliary_snapshot_mode = "replay"
    config.auxiliary_snapshot_references = {
        key: [
            {**asdict(reference), "manifest_path": str(reference.manifest_path)}
            for reference in values
        ]
        for key, values in manager.input_snapshots.references.items()
    }
    real_open = Path.open
    forbidden = []

    def deny_original(path, *args, **kwargs):
        if path == original:
            forbidden.append(str(path))
            raise AssertionError("Replay reopened native original")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_original)
    replay_manager = DataManager(config)
    replay = (
        replay_manager.load_vix()
        if role == "vix"
        else replay_manager.load_regime_inputs(
            regime_inputs_csv=str(original), regime_enforce_lag_days=1
        )
    )
    pd.testing.assert_frame_equal(source, replay, check_exact=True)
    assert forbidden == []


def test_preparation_keeps_rejected_file_snapshot_without_consumption(
    tmp_path: Path, monkeypatch
) -> None:
    config = _preparation_config(tmp_path)
    original = tmp_path / "regime.csv"
    original.write_bytes(b"")
    config.features.include_global_regime = True
    config.features.regime_inputs_csv = str(original)
    config.data.auxiliary_snapshot_mode = "capture"
    config.data.auxiliary_snapshot_directory = str(tmp_path / "snapshots")
    with pytest.raises(InputObservationError) as capture_failure:
        prepare_data(config, FeatureEngineer(config.features))
    evidence = capture_failure.value.input_observations.to_dict()
    publication = next(event for event in evidence["observations"] if event["stage"] == "retain")
    assert all(use["role"] == "data.filename" for use in evidence["uses"])
    identity = publication["identity"]
    config.data.auxiliary_snapshot_mode = "replay"
    config.data.auxiliary_snapshot_references = {
        identity["snapshot_key"]: [
            {
                "manifest_path": identity["manifest_path"],
                "manifest_sha256": identity["manifest_sha256"],
            }
        ]
    }
    with pytest.raises(InputObservationError) as replay_failure:
        prepare_data(config, FeatureEngineer(config.features))
    replay_evidence = replay_failure.value.input_observations.to_dict()
    failures = [event for event in replay_evidence["observations"] if event["outcome"] == "error"]
    assert len(failures) == 1
    assert failures[0]["stage"] == "parse"
    assert failures[0]["identity"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert failures[0]["error_code"] == "EmptyDataError"
    assert all(use["role"] == "data.filename" for use in replay_evidence["uses"])
    assert original.read_bytes() == b""


def test_provider_capture_uses_the_retained_observation_if_sdk_buffer_changes(
    tmp_path: Path, monkeypatch
) -> None:
    raw = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, *args, **kwargs):
            return raw

    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    original_write = Path.write_bytes

    def write_then_change_sdk_buffer(path, content):
        written = original_write(path, content)
        if path.name == "observations.bin":
            raw.iloc[0] = 900.0
        return written

    monkeypatch.setattr(Path, "write_bytes", write_then_change_sdk_buffer)
    store = InputSnapshots(mode="capture", directory=tmp_path / "snapshots")
    source = FREDLoader(api_key="test-only", snapshots=store).get_series(
        "DGS10", "2020-01-01", "2020-01-03", "rate", lag_days=0
    )
    assert raw.iloc[0] == 900.0  # The external mutation actually happened.
    assert source["rate"].tolist() == [1.0, 2.0, 3.0]
    replay_store = InputSnapshots(mode="replay", references=store.references)
    replay = FREDLoader(snapshots=replay_store).get_series(
        "DGS10", "2020-01-01", "2020-01-03", "rate", lag_days=0
    )
    pd.testing.assert_frame_equal(source, replay, check_exact=True)


@pytest.mark.parametrize("fault", ["codec", "publication"])
def test_preparation_does_not_reacquire_after_retention_failure(
    tmp_path: Path, monkeypatch, fault: str
) -> None:
    config = _preparation_config(tmp_path)
    config.features.include_global_regime = True
    config.data.auxiliary_sources = {"regime": "fred"}
    config.data.auxiliary_snapshot_mode = "capture"
    config.data.auxiliary_snapshot_directory = str(tmp_path / "snapshots")
    calls = []

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, **kwargs):
            calls.append(series_id)
            return (
                object()
                if fault == "codec"
                else pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
            )

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setenv("MCI_GRU_FRED_MAX_ATTEMPTS", "2")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    real_write = Path.write_bytes

    def fail_publication(path, content):
        if path.name == "observations.bin":
            raise PermissionError("sensitive storage error")
        return real_write(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_publication)
    with pytest.raises(InputObservationError) as caught:
        prepare_data(config, FeatureEngineer(config.features))
    assert calls == ["DGS10"]
    assert caught.value.facts["stage"] == ("validate" if fault == "codec" else "retain")
    assert caught.value.facts["reason_code"] != "acquisition_failed"
    assert set(caught.value.input_observations.data_inputs()) == {"data.filename"}
    assert "sensitive storage error" not in str(caught.value.facts)


def test_preparation_reports_the_current_rejected_series_after_an_earlier_retry(
    tmp_path: Path, monkeypatch
) -> None:
    config = _preparation_config(tmp_path)
    config.features.include_global_regime = True
    config.data.auxiliary_sources = {"regime": "fred"}
    config.data.auxiliary_snapshot_mode = "capture"
    config.data.auxiliary_snapshot_directory = str(tmp_path / "snapshots")
    calls = []

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, **kwargs):
            calls.append(series_id)
            if calls == ["DGS10"]:
                raise TimeoutError("transient synthetic timeout")
            values = (
                [1.0, 2.0, 3.0] if series_id == "DGS10" else ["malformed", "malformed", "malformed"]
            )
            return pd.Series(values, index=pd.date_range("2020-01-01", periods=3))

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setenv("MCI_GRU_FRED_MAX_ATTEMPTS", "2")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    with pytest.raises(InputObservationError) as caught:
        prepare_data(config, FeatureEngineer(config.features))
    assert calls == ["DGS10", "DGS10", "DGS3MO"]
    failure = caught.value
    assert failure.facts["role"] == "fred.yield_3m"
    assert failure.facts["request"]["series_id"] == "DGS3MO"
    assert failure.facts["stage"] == "parse"
    assert failure.facts["identity"]["manifest_sha256"]
    evidence = failure.input_observations.to_dict()
    failed = [event for event in evidence["observations"] if event["outcome"] == "error"]
    assert [event["role"] for event in failed] == ["fred.yield_10y", "fred.yield_3m"]
    assert set(failure.input_observations.data_inputs()) == {"data.filename", "fred.yield_10y"}


@pytest.mark.parametrize("selection", ["regime_lseg", "regime_ric", "vix_fred"])
def test_explicit_unsupported_source_configuration_is_not_ignored(
    tmp_path: Path, monkeypatch, selection: str
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vix_data.csv").write_text("dt,vix\n2020-01-01,20\n", encoding="utf-8")
    calls = []

    def deny(*args, **kwargs):
        calls.append("provider")
        raise AssertionError("Unsupported configuration started a provider")

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=deny))
    selected = {
        "regime_lseg": {"regime": "lseg"},
        "regime_ric": {"regime": "fred"},
        "vix_fred": {"vix": "fred"},
    }[selection]
    manager = DataManager(DataConfig(auxiliary_sources=selected))
    with pytest.raises(InputObservationError, match="unsupported"):
        if selection == "vix_fred":
            manager.load_vix()
        elif selection == "regime_ric":
            manager.load_regime_inputs(lseg_market_ric="explicit-other-ric")
        else:
            manager.load_regime_inputs()
    assert calls == []
    assert manager.input_observations.freeze().uses == ()
