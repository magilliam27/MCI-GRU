"""#276: an S&P 500 index file extends FRED SP500 backwards past its ten-year window.

FRED serves only the last ten years of SP500, so the market role starts late. A
``dt,close`` history of the same index fills the dates before FRED's first
observation. These tests go through public DataManager.load_regime_inputs with a
fake FRED SDK: the fill is captured with the run and replays without the file or
the provider, it must agree with FRED where both exist, and the #224 rules still
apply to the spliced series.
"""

import sys
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from mci_gru.config import DataConfig
from mci_gru.data import path_resolver
from mci_gru.data.auxiliary_quality import ADMISSION_RULE, HISTORY_FILL_TOLERANCE
from mci_gru.data.data_manager import MARKET_HISTORY_ROLE, DataManager
from mci_gru.data.input_snapshots import InputSnapshotError

DATES = pd.bdate_range("2024-12-02", "2025-03-13")
FRED_START = pd.Timestamp("2025-01-02")
DAILY = {
    "SP500": (5000.0, 1.0),
    "DGS10": (4.0, 0.001),
    "DGS3MO": (5.0, 0.001),
    "DCOILWTICO": (70.0, 0.01),
    "VIXCLS": (15.0, 0.01),
}


def market(dates: pd.DatetimeIndex) -> pd.Series:
    first, step = DAILY["SP500"]
    return pd.Series([first + DATES.get_loc(date) * step for date in dates], index=dates)


def fred_series(sp500_from: pd.Timestamp = FRED_START) -> dict[str, pd.Series]:
    series = {
        series_id: pd.Series(
            [first + position * step for position in range(len(DATES))], index=DATES
        ).astype(object)
        for series_id, (first, step) in DAILY.items()
    }
    series["SP500"] = series["SP500"].loc[sp500_from:]
    series["PCOPPUSDM"] = pd.Series(
        [9000.0, 9100.0, 9200.0, 9300.0, 9400.0],
        index=pd.to_datetime(
            ["2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"]
        ),
    ).astype(object)
    return series


def write_history(tmp_path, closes: pd.Series) -> str:
    path = tmp_path / "sp500_index.csv"
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


def install_sdk(monkeypatch, series: dict[str, pd.Series]) -> None:
    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end):
            return series[series_id].copy()

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))


def market_item(manager: DataManager) -> dict:
    items = manager.admission.to_dict()["items"]
    return next(
        item
        for item in items
        if item["rule"] == ADMISSION_RULE and item["role"] == "fred.regime_market"
    )


def test_history_fills_before_fred_and_replays_without_the_file(tmp_path, monkeypatch) -> None:
    history = write_history(tmp_path, market(DATES))
    install_sdk(monkeypatch, fred_series())
    capture = DataManager(config(tmp_path))
    captured = capture.load_regime_inputs(regime_market_history_csv=history)

    install_sdk(monkeypatch, fred_series(sp500_from=DATES[0]))
    full = DataManager(config(tmp_path / "full")).load_regime_inputs()
    pd.testing.assert_series_equal(captured["regime_market"], full["regime_market"])

    fill = market_item(capture)["evidence"]["history_fill"]
    assert fill["role"] == MARKET_HISTORY_ROLE
    assert fill["configured_path"] == history
    assert fill["splice_before"] == "2025-01-02"
    assert fill["fill_first_observation"] == "2024-12-02"
    assert fill["fill_last_observation"] == "2025-01-01"
    assert fill["fill_observations"] == len(DATES[DATES < FRED_START])
    assert fill["overlap_observations"] == len(DATES[DATES >= FRED_START])
    assert fill["overlap_max_relative_difference"] == 0.0
    assert market_item(capture)["evidence"]["first_observation"] == "2024-12-02"

    references = {
        key: [{**asdict(item), "manifest_path": str(item.manifest_path)} for item in values]
        for key, values in capture.input_snapshots.references.items()
    }
    (tmp_path / "sp500_index.csv").unlink()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    replayed = DataManager(config(tmp_path, "replay", references)).load_regime_inputs(
        regime_market_history_csv=history
    )
    pd.testing.assert_frame_equal(replayed, captured, check_exact=True)


def test_fred_values_stand_from_its_first_observation(tmp_path, monkeypatch) -> None:
    closes = market(DATES)
    closes.loc[FRED_START:] *= 1 + HISTORY_FILL_TOLERANCE / 2
    install_sdk(monkeypatch, fred_series())
    filled = DataManager(config(tmp_path)).load_regime_inputs(
        regime_market_history_csv=write_history(tmp_path, closes)
    )
    rows = filled.set_index("dt")["regime_market"]
    # A daily value dated D is used from the next session, so FRED's 2025-01-02 lands on 01-03.
    assert rows.loc["2025-01-02"] == market(DATES).loc["2025-01-01"]
    assert rows.loc["2025-01-03"] == market(DATES).loc["2025-01-02"]


@pytest.mark.parametrize(
    ("closes", "reason"),
    [
        (market(DATES) * 1.01, "history_fill_mismatch"),
        (market(DATES[: DATES.get_loc(FRED_START) + 5]), "history_fill_overlap_too_short"),
    ],
)
def test_history_that_disagrees_or_barely_overlaps_stops_the_run(
    tmp_path, monkeypatch, closes, reason
) -> None:
    install_sdk(monkeypatch, fred_series())
    manager = DataManager(config(tmp_path))
    with pytest.raises(InputSnapshotError) as caught:
        manager.load_regime_inputs(regime_market_history_csv=write_history(tmp_path, closes))
    assert caught.value.facts["role"] == "fred.regime_market"
    assert caught.value.facts["reason_code"] == reason


def test_without_history_the_market_role_is_fred_only(tmp_path, monkeypatch) -> None:
    install_sdk(monkeypatch, fred_series())
    manager = DataManager(config(tmp_path))
    rows = manager.load_regime_inputs().set_index("dt")
    assert rows.loc[:"2025-01-02", "regime_market"].isna().all()
    assert "history_fill" not in market_item(manager)["evidence"]


def test_a_missing_history_file_is_not_substituted_by_basename(tmp_path, monkeypatch) -> None:
    # A same-named file sits where the resolver's basename fallback would look.
    decoy = tmp_path / "root" / "data" / "raw" / "market"
    decoy.mkdir(parents=True)
    write_history(decoy, market(DATES))
    monkeypatch.setattr(path_resolver, "PROJECT_ROOT", tmp_path / "root")
    install_sdk(monkeypatch, fred_series())
    with pytest.raises(FileNotFoundError):
        DataManager(config(tmp_path)).load_regime_inputs(
            regime_market_history_csv=str(tmp_path / "absent" / "sp500_index.csv")
        )
