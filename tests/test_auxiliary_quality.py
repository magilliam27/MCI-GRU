"""#224 regime input rulings, proven through public DataManager.load_regime_inputs.

The fixture dates were fixed on the issue before these tests were written
(https://github.com/magilliam27/MCI-GRU/issues/224#issuecomment-5971699371).
Each load is captured from a fake SDK and then replayed with fredapi removed
and no FRED_API_KEY, so the proof runs with providers and network off.
"""

import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mci_gru.config import DataConfig
from mci_gru.data.auxiliary_quality import MAX_CARRY_SESSIONS
from mci_gru.data.data_manager import DataManager
from mci_gru.data.input_snapshots import InputSnapshotError

DAILY_DATES = pd.bdate_range("2024-12-02", "2025-03-13")
DAILY = {
    # FRED series: (role column, first value, step per session)
    "SP500": ("regime_market", 5000.0, 1.0),
    "DGS10": ("yield_10y", 4.0, 0.001),
    "DGS3MO": ("yield_3m", 5.0, 0.001),
    "DCOILWTICO": ("regime_oil", 70.0, 0.01),
    "VIXCLS": ("regime_volatility", 15.0, 0.01),
}
COPPER_DATES = pd.to_datetime(
    ["2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"]
)
COPPER_VALUES = [9000.0, 9100.0, 9200.0, 9300.0, 9400.0]
YIELD_GAP = pd.bdate_range("2025-01-13", "2025-01-17")


def daily_value(series_id: str, date: str) -> float:
    _, first, step = DAILY[series_id]
    return first + DAILY_DATES.get_loc(pd.Timestamp(date)) * step


def fixture_series(**edits: dict[str, object]) -> dict[str, pd.Series]:
    """The fixed fixture; edits map a series to {date: value or None to drop}."""
    series = {
        series_id: pd.Series(
            [daily_value(series_id, date) for date in DAILY_DATES], index=DAILY_DATES
        ).astype(object)
        for series_id in DAILY
    }
    series["DGS10"].loc[YIELD_GAP] = "."
    series["PCOPPUSDM"] = pd.Series(COPPER_VALUES, index=COPPER_DATES).astype(object)
    for series_id, changes in edits.items():
        for date, value in changes.items():
            if value is None:
                series[series_id] = series[series_id].drop(pd.Timestamp(date))
            else:
                series[series_id].loc[pd.Timestamp(date)] = value
    return series


def fixture_config(tmp_path, mode="capture", references=None) -> DataConfig:
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
    calls: list[str] = []

    class Fred:
        def __init__(self, api_key):
            calls.append("setup")

        def get_series(self, series_id, observation_start, observation_end):
            calls.append(series_id)
            return series[series_id].copy()

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    return calls


def replay(tmp_path, monkeypatch, series: dict[str, pd.Series]):
    """Capture once from the fake SDK, then load again offline; both must agree."""
    install_sdk(monkeypatch, series)
    capture = DataManager(fixture_config(tmp_path))
    captured = capture.load_regime_inputs()
    references = {
        key: [{**asdict(item), "manifest_path": str(item.manifest_path)} for item in values]
        for key, values in capture.input_snapshots.references.items()
    }
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "fredapi", None)
    manager = DataManager(fixture_config(tmp_path, "replay", references))
    out = manager.load_regime_inputs()
    pd.testing.assert_frame_equal(out, captured, check_exact=True)
    return out.set_index("dt"), manager


def ten_year(rows: pd.DataFrame) -> pd.Series:
    return rows["regime_yield_curve"] + rows["regime_monetary_policy"]


def stop(tmp_path, monkeypatch, series) -> dict:
    install_sdk(monkeypatch, series)
    with pytest.raises(InputSnapshotError) as caught:
        DataManager(fixture_config(tmp_path)).load_regime_inputs()
    return caught.value.facts


def test_session_t_sees_only_values_dated_before_t(tmp_path, monkeypatch) -> None:
    rows, _ = replay(tmp_path, monkeypatch, fixture_series())
    proof = pd.bdate_range("2024-12-03", "2025-03-14").strftime("%Y-%m-%d")
    previous = DAILY_DATES.strftime("%Y-%m-%d")
    assert list(proof) == list(previous[1:]) + ["2025-03-14"]
    for series_id in ("SP500", "DCOILWTICO", "VIXCLS", "DGS3MO"):
        column, _, _ = DAILY[series_id]
        column = "regime_monetary_policy" if column == "yield_3m" else column
        expected = [daily_value(series_id, date) for date in previous]
        np.testing.assert_allclose(rows.loc[proof, column].to_numpy(), expected)
    assert rows.loc["2025-01-07", "regime_market"] == daily_value("SP500", "2025-01-06")
    assert rows.loc["2025-01-07", "regime_market"] != daily_value("SP500", "2025-01-07")


def test_a_gap_carries_exactly_five_sessions_and_no_more(tmp_path, monkeypatch) -> None:
    rows, manager = replay(tmp_path, monkeypatch, fixture_series())
    carried = pd.bdate_range("2025-01-13", "2025-01-20").strftime("%Y-%m-%d")
    np.testing.assert_allclose(ten_year(rows.loc[carried]), daily_value("DGS10", "2025-01-10"))
    assert ten_year(rows.loc[["2025-01-21"]]).iloc[0] == pytest.approx(
        daily_value("DGS10", "2025-01-20")
    )
    verdicts = {item["role"]: item for item in manager.regime_input_receipt["verdicts"]}
    assert verdicts["yield_10y"]["max_carry_sessions_seen"] == MAX_CARRY_SESSIONS == 5
    assert verdicts["yield_10y"]["missing_observations"] == len(YIELD_GAP)

    facts = stop(tmp_path, monkeypatch, fixture_series(DGS10={"2025-01-20": "."}))
    assert facts["reason"] == "carry_forward_limit_exceeded"
    assert facts["role"] == "fred.yield_10y"
    assert facts["regime_role_label"] == "10-year yield"
    assert facts["last_observation"] == "2025-01-10"
    assert facts["first_session_over_limit"] == "2025-01-21"
    assert facts["stage"] == "validate"


def test_copper_for_month_m_counts_from_month_m_plus_2(tmp_path, monkeypatch) -> None:
    rows, manager = replay(tmp_path, monkeypatch, fixture_series())
    copper = rows["regime_copper"]
    assert copper.loc["2024-12-02":"2024-12-31"].isna().all()
    for month, value in (("2025-01", 9000.0), ("2025-02", 9100.0), ("2025-03", 9200.0)):
        in_month = copper[copper.index.str.startswith(month)]
        assert len(in_month) >= 10
        assert (in_month == value).all()
    verdicts = {item["role"]: item for item in manager.regime_input_receipt["verdicts"]}
    assert verdicts["regime_copper"]["first_usable_session"] == "2025-01-01"

    facts = stop(tmp_path, monkeypatch, fixture_series(PCOPPUSDM={"2024-12-01": None}))
    assert facts["reason"] == "carry_forward_limit_exceeded"
    assert facts["role"] == "fred.regime_copper"
    assert facts["last_observation"] == "2024-11-01"
    assert facts["first_session_over_limit"] == "2025-02-03"


def test_changing_a_later_value_never_changes_earlier_output(tmp_path, monkeypatch) -> None:
    base, _ = replay(tmp_path / "base", monkeypatch, fixture_series())
    changed, _ = replay(
        tmp_path / "changed", monkeypatch, fixture_series(SP500={"2025-02-03": 9999.0})
    )
    pd.testing.assert_frame_equal(
        changed.loc[:"2025-02-03"], base.loc[:"2025-02-03"], check_exact=True
    )
    assert base.loc["2025-02-04", "regime_market"] == daily_value("SP500", "2025-02-03")
    assert changed.loc["2025-02-04", "regime_market"] == 9999.0


def test_sessions_before_the_first_usable_value_stay_empty_and_are_counted(
    tmp_path, monkeypatch
) -> None:
    rows, manager = replay(tmp_path, monkeypatch, fixture_series())
    leading = rows.loc[:"2024-12-02"]
    assert len(leading) > 1000
    assert leading.isna().all().all()
    assert rows.loc["2024-12-03"].drop(["regime_copper", "regime_stock_bond_corr"]).notna().all()
    receipt = manager.regime_input_receipt
    expected = len(pd.bdate_range(rows.index[0], "2024-12-02"))
    for verdict in receipt["verdicts"]:
        assert verdict["revisions"] == "unchecked"
        assert verdict["rules"]["back_fill"] is False
        if verdict["role"] != "regime_copper":
            assert verdict["first_usable_session"] == "2024-12-03"
            assert verdict["leading_gap_sessions"] == expected
    assert receipt["leading_gap_sessions"]["regime_market"] == expected
    assert receipt["leading_gap_sessions"]["regime_stock_bond_corr"] == len(rows)
    assert len(receipt["verdicts"]) == 6


@pytest.mark.parametrize(
    ("series_id", "value", "reason"),
    [
        ("DCOILWTICO", "n/a", "malformed_value"),
        ("DGS10", float("inf"), "non_finite_value"),
        ("SP500", 0.0, "non_positive_value"),
        ("VIXCLS", -1.0, "non_positive_value"),
    ],
)
def test_a_malformed_or_invalid_value_stops_the_run_naming_role_and_date(
    tmp_path, monkeypatch, series_id, value, reason
) -> None:
    facts = stop(tmp_path, monkeypatch, fixture_series(**{series_id: {"2025-01-08": value}}))
    assert facts["reason"] == reason
    assert facts["role"] == f"fred.{DAILY[series_id][0]}"
    assert facts["dates"] == ["2025-01-08"]


def test_copper_must_be_positive(tmp_path, monkeypatch) -> None:
    facts = stop(tmp_path, monkeypatch, fixture_series(PCOPPUSDM={"2024-12-01": -1.0}))
    assert facts["reason"] == "non_positive_value"
    assert facts["dates"] == ["2024-12-01"]


def test_gaps_negative_oil_and_low_yields_are_valid(tmp_path, monkeypatch) -> None:
    series = fixture_series(
        DCOILWTICO={"2025-01-08": -37.63, "2025-01-09": "  "},
        DGS3MO={"2025-01-08": 0.0, "2025-01-09": -0.01},
    )
    rows, _ = replay(tmp_path, monkeypatch, series)
    assert rows.loc["2025-01-09", "regime_oil"] == -37.63
    assert rows.loc["2025-01-10", "regime_oil"] == -37.63
    assert rows.loc["2025-01-10", "regime_monetary_policy"] == -0.01


def test_a_stop_keeps_its_facts_through_the_preparation_failure(tmp_path, monkeypatch) -> None:
    install_sdk(monkeypatch, fixture_series(DCOILWTICO={"2025-01-08": "n/a"}))
    manager = DataManager(fixture_config(tmp_path))
    with pytest.raises(InputSnapshotError) as caught:
        manager.load_regime_inputs()
    failure = manager.required_input_failure(caught.value, role="regime", source="fred")
    assert failure.facts["reason_code"] == "malformed_value"
    assert failure.facts["dates"] == ["2025-01-08"]
    assert "oil" in failure.facts["detail"]
    observations = failure.input_observations.to_dict()
    oil = [item for item in observations["observations"] if item["role"] == "fred.regime_oil"]
    assert [item["outcome"] for item in oil] == ["success", "error"]
    assert "fred.regime_oil" not in {use["role"] for use in observations["uses"]}


def test_a_source_covering_the_request_leaves_no_leading_gap(tmp_path, monkeypatch) -> None:
    """Copper's request reaches back to month X-2, so its first session is usable."""

    class Fred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end):
            dates = pd.date_range(observation_start, observation_end)
            return pd.Series(np.linspace(1.0, 2.0, len(dates)), index=dates)

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=Fred))
    manager = DataManager(fixture_config(tmp_path))
    out = manager.load_regime_inputs()
    receipt = manager.regime_input_receipt
    assert {item["role"]: item["leading_gap_sessions"] for item in receipt["verdicts"]} == {
        role: 0
        for role in (
            "yield_10y",
            "yield_3m",
            "regime_oil",
            "regime_volatility",
            "regime_market",
            "regime_copper",
        )
    }
    assert out["regime_copper"].notna().all()
