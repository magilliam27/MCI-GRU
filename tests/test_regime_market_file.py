"""#276: the market regime role can come from an S&P 500 index file instead of FRED.

FRED serves only the last ten years of SP500, so the market role started late. With
``regime_market_csv`` set, the role is read from a ``dt,close`` index file (EODHD
GSPC in the recipe), FRED SP500 is not requested, and the #224 rules apply
unchanged. These tests go through public DataManager.load_regime_inputs with a fake
FRED SDK for the other five roles. The file is captured with the run and replays
without the file or the provider.
"""

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mci_gru.config import DataConfig
from mci_gru.data import path_resolver
from mci_gru.data.auxiliary_quality import ADMISSION_RULE, MAX_CARRY_SESSIONS
from mci_gru.data.data_manager import MARKET_FILE_ROLE, DataManager
from mci_gru.data.input_snapshots import InputSnapshotError

DATES = pd.bdate_range("2024-12-02", "2025-03-13")
DAILY = {
    "SP500": (5000.0, 1.0),
    "DGS10": (4.0, 0.001),
    "DGS3MO": (5.0, 0.001),
    "DCOILWTICO": (70.0, 0.01),
    "VIXCLS": (15.0, 0.01),
}


def market(dates: pd.DatetimeIndex = DATES) -> pd.Series:
    first, step = DAILY["SP500"]
    return pd.Series([first + DATES.get_loc(date) * step for date in dates], index=dates)


def fred_series() -> dict[str, pd.Series]:
    series = {
        series_id: pd.Series(
            [first + position * step for position in range(len(DATES))], index=DATES
        ).astype(object)
        for series_id, (first, step) in DAILY.items()
    }
    series["PCOPPUSDM"] = pd.Series(
        [9000.0, 9100.0, 9200.0, 9300.0, 9400.0],
        index=pd.to_datetime(
            ["2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"]
        ),
    ).astype(object)
    return series


def write_market(directory, closes: pd.Series) -> str:
    path = directory / "sp500_index.csv"
    pd.DataFrame({"dt": closes.index.strftime("%Y-%m-%d"), "close": closes.to_numpy()}).to_csv(
        path, index=False
    )
    return str(path)


def config(tmp_path, mode="capture", references=None) -> DataConfig:
    return DataConfig(
        auxiliary_sources={"regime": "fred"},
        train_start="2025-01-06",
        train_end="2025-01-31",
        val_start="2025-02-03",
        val_end="2025-02-14",
        test_start="2025-02-17",
        test_end="2025-03-14",
        auxiliary_snapshot_mode=mode,
        auxiliary_snapshot_directory=str(tmp_path / "snapshots"),
        auxiliary_snapshot_references=references or {},
    )


def install_sdk(monkeypatch, series: dict[str, pd.Series]) -> list[str]:
    requested: list[str] = []

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end):
            requested.append(series_id)
            return series[series_id].copy()

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    return requested


def regime_items(manager: DataManager) -> list[dict]:
    return [item for item in manager.admission.to_dict()["items"] if item["rule"] == ADMISSION_RULE]


def test_the_market_role_comes_from_the_file_and_replays_without_it(tmp_path, monkeypatch) -> None:
    path = write_market(tmp_path, market())
    requested = install_sdk(monkeypatch, fred_series())
    capture = DataManager(config(tmp_path))
    captured = capture.load_regime_inputs(regime_market_csv=path)
    assert sorted(requested) == ["DCOILWTICO", "DGS10", "DGS3MO", "PCOPPUSDM", "VIXCLS"]

    # Same values as the FRED path given the same series: the source changes, not the rules.
    install_sdk(monkeypatch, fred_series())
    fred_only = DataManager(config(tmp_path / "fred")).load_regime_inputs()
    pd.testing.assert_frame_equal(captured, fred_only, check_exact=True)

    items = {item["role"]: item for item in regime_items(capture)}
    assert "fred.regime_market" not in items
    item = items["eodhd.regime_market"]
    assert item["verdict"] == "valid"
    assert item["source"] == "eodhd"
    assert item["evidence"]["series_id"] == "GSPC.INDX"
    assert item["evidence"]["configured_path"] == path
    assert item["evidence"]["first_usable_session"] == "2024-12-03"

    references = {
        key: [{**asdict(entry), "manifest_path": str(entry.manifest_path)} for entry in values]
        for key, values in capture.input_snapshots.references.items()
    }
    (tmp_path / "sp500_index.csv").unlink()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    replayed = DataManager(config(tmp_path, "replay", references)).load_regime_inputs(
        regime_market_csv=path
    )
    pd.testing.assert_frame_equal(replayed, captured, check_exact=True)


def test_file_values_obey_next_session_availability(tmp_path, monkeypatch) -> None:
    install_sdk(monkeypatch, fred_series())
    rows = (
        DataManager(config(tmp_path))
        .load_regime_inputs(regime_market_csv=write_market(tmp_path, market()))
        .set_index("dt")["regime_market"]
    )
    assert pd.isna(rows.loc["2024-12-02"])
    assert rows.loc["2025-01-03"] == market().loc["2025-01-02"]


def edited(set_values=None, drop=()) -> pd.Series:
    closes = market()
    for date, value in (set_values or {}).items():
        closes.loc[pd.Timestamp(date)] = value
    return closes.drop(list(drop))


@pytest.mark.parametrize(
    ("closes", "reason"),
    [
        (edited(set_values={"2025-02-03": 0.0}), "non_positive_value"),
        (edited(drop=DATES[40 : 40 + MAX_CARRY_SESSIONS + 1]), "carry_forward_limit_exceeded"),
    ],
)
def test_a_file_breaking_a_224_rule_stops_the_run_naming_the_file(
    tmp_path, monkeypatch, closes, reason
) -> None:
    install_sdk(monkeypatch, fred_series())
    manager = DataManager(config(tmp_path))
    path = write_market(tmp_path, closes)
    with pytest.raises(InputSnapshotError) as caught:
        manager.load_regime_inputs(regime_market_csv=path)
    facts = caught.value.facts
    assert (facts["role"], facts["source"], facts["stage"]) == (
        "eodhd.regime_market",
        "eodhd",
        "validate",
    )
    assert facts["reason_code"] == reason
    assert facts["configured_path"] == path
    ledger = caught.value.admission
    assert ledger["admitted"] is False
    assert [
        (item["role"], item["reason_code"])
        for item in ledger["items"]
        if item["verdict"] == "invalid"
    ] == [("eodhd.regime_market", reason)]


def test_without_the_file_the_market_role_is_fred(tmp_path, monkeypatch) -> None:
    requested = install_sdk(monkeypatch, fred_series())
    manager = DataManager(config(tmp_path))
    manager.load_regime_inputs()
    assert "SP500" in requested
    assert "fred.regime_market" in {item["role"] for item in regime_items(manager)}


def write_rows(tmp_path, dts: list[str], closes: list[str]) -> str:
    path = tmp_path / "sp500_index.csv"
    path.write_text(
        "dt,close\n" + "".join(f"{dt},{close}\n" for dt, close in zip(dts, closes, strict=True)),
        encoding="utf-8",
    )
    return str(path)


PLAIN_DATES = [f"{date:%Y-%m-%d}" for date in DATES]
CLOSES = [str(value) for value in market()]


@pytest.mark.parametrize(
    ("dts", "closes"),
    [
        ([f"{dt} 16:00:00" for dt in PLAIN_DATES], CLOSES),
        ([f"{dt}T00:00:00-05:00" for dt in PLAIN_DATES], CLOSES),
        (["", *PLAIN_DATES[1:]], CLOSES),
    ],
    ids=["time_of_day", "offset", "blank_dt"],
)
def test_a_malformed_market_date_stops_naming_the_file(tmp_path, monkeypatch, dts, closes) -> None:
    path = write_rows(tmp_path, dts, closes)
    install_sdk(monkeypatch, fred_series())
    with pytest.raises(InputSnapshotError) as caught:
        DataManager(config(tmp_path)).load_regime_inputs(regime_market_csv=path)
    facts = caught.value.facts
    assert (facts["role"], facts["stage"], facts["reason_code"]) == (
        MARKET_FILE_ROLE,
        "parse",
        "parse_failed",
    )
    assert facts["configured_path"] == path


@pytest.mark.parametrize("token", ["NaN", "N/A", "null", "#N/A", "None"])
def test_only_a_blank_close_is_a_gap(tmp_path, monkeypatch, token) -> None:
    install_sdk(monkeypatch, fred_series())
    gap = DataManager(config(tmp_path / "gap"))
    gap.load_regime_inputs(regime_market_csv=write_rows(tmp_path, PLAIN_DATES, ["", *CLOSES[1:]]))
    market_verdict = {item["role"]: item for item in regime_items(gap)}["eodhd.regime_market"]
    assert market_verdict["evidence"]["missing_observations"] == 1

    with pytest.raises(InputSnapshotError) as caught:
        DataManager(config(tmp_path)).load_regime_inputs(
            regime_market_csv=write_rows(tmp_path, PLAIN_DATES, [token, *CLOSES[1:]])
        )
    assert caught.value.facts["role"] == "eodhd.regime_market"
    assert caught.value.facts["reason_code"] == "malformed_value"


def test_a_missing_market_file_is_not_substituted_by_basename(tmp_path, monkeypatch) -> None:
    # A same-named file sits where the resolver's basename fallback would look.
    decoy = tmp_path / "root" / "data" / "raw" / "market"
    decoy.mkdir(parents=True)
    write_market(decoy, market())
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", tmp_path / "root")
    install_sdk(monkeypatch, fred_series())
    missing = str(tmp_path / "absent" / "sp500_index.csv")
    with pytest.raises(InputSnapshotError) as caught:
        DataManager(config(tmp_path)).load_regime_inputs(regime_market_csv=missing)
    assert caught.value.facts["role"] == MARKET_FILE_ROLE
    assert caught.value.facts["reason_code"] == "file_not_found"
    assert caught.value.facts["configured_path"] == missing


def recorded(manager: DataManager) -> tuple[dict[int, dict], list[dict]]:
    """The window's observations by id, and its uses."""
    window = manager.input_snapshots.input_observations.freeze().to_dict()
    return {event["observation_id"]: event for event in window["observations"]}, window["uses"]


@pytest.mark.parametrize("mode", ["source", "capture"])
def test_the_market_verdict_names_the_read_it_was_ruled_on(tmp_path, monkeypatch, mode) -> None:
    install_sdk(monkeypatch, fred_series())
    path = write_market(tmp_path, market())
    manager = DataManager(config(tmp_path, mode))
    manager.load_regime_inputs(regime_market_csv=path)

    verdict = {item["role"]: item for item in regime_items(manager)}["eodhd.regime_market"]
    observation_id = verdict["evidence"]["observation_id"]
    events, uses = recorded(manager)
    read = events[observation_id]
    assert read["role"] == MARKET_FILE_ROLE
    assert read["identity"]["sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert ("manifest_sha256" in read["identity"]) == (mode == "capture")
    assert [use["observation_id"] for use in uses if use["role"] == MARKET_FILE_ROLE] == [
        observation_id
    ]


@pytest.mark.parametrize(
    ("closes", "reason"),
    [
        (edited(set_values={"2025-02-03": 0.0}), "non_positive_value"),
        (None, "parse_failed"),
    ],
)
def test_a_stopped_market_file_is_recorded_as_read_but_never_used(
    tmp_path, monkeypatch, closes, reason
) -> None:
    install_sdk(monkeypatch, fred_series())
    path = (
        write_market(tmp_path, closes)
        if closes is not None
        else write_rows(tmp_path, ["", *PLAIN_DATES[1:]], CLOSES)
    )
    manager = DataManager(config(tmp_path))
    with pytest.raises(InputSnapshotError) as caught:
        manager.load_regime_inputs(regime_market_csv=path)
    assert caught.value.facts["reason_code"] == reason

    events, uses = recorded(manager)
    assert MARKET_FILE_ROLE not in {use["role"] for use in uses}
    (read,) = [
        event
        for event in events.values()
        if event["role"] == MARKET_FILE_ROLE and event["outcome"] == "success"
    ]
    # The rejection is linked to the read it rejected.
    assert [
        event["source_observation_id"]
        for event in events.values()
        if event["role"] == MARKET_FILE_ROLE and event["outcome"] == "error"
    ] == [read["observation_id"]]
