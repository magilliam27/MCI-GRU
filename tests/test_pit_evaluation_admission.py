"""PIT admission must be consistent across every backtest replay path (issue #116).

The benchmark is PIT-filtered on all paths via
``calendar_returns_for_evaluation_window(..., pit_universe_df=...)``, but only the daily
simulator used to accept ``pit_universe_df``: a ``holding_period > 1`` run compared a
full-universe portfolio against a PIT-filtered benchmark and reported the difference as
excess return. These tests pin the refusal guard, the unified dispatch, and the
admission basis of the portfolio, the internal benchmark, and the saved artifacts.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.backtest_engine import (
    DEFAULT_CONFIG,
    admit_pit_candidates,
    calculate_forward_returns,
    dispatch_trading_simulation,
    evaluate,
    load_pit_universe_for_backtest,
    load_predictions,
    load_stock_data,
    save_backtest_results,
    simulate_trading_strategy,
    simulate_trading_strategy_block,
    simulate_trading_strategy_staggered,
)

if TYPE_CHECKING:
    from pathlib import Path

PIT_ACTIVE = ["STK01", "STK02", "STK03", "STK04", "STK05"]
PIT_INACTIVE = "HOTX"
TOP_K = 3
LABEL_T = 5
HOLDING_PERIOD = 5
TEST_START = "2025-02-03"
TEST_END = "2025-03-31"


def _sessions() -> list[str]:
    return pd.bdate_range(TEST_START, TEST_END).strftime("%Y-%m-%d").tolist()


def _price_panel(kdcodes: list[str]) -> pd.DataFrame:
    """Deterministic panel in which the PIT-inactive name has the strongest returns."""
    sessions = _sessions()
    rows = []
    for stock_idx, kdcode in enumerate(kdcodes):
        price = 100.0
        for day_idx, session in enumerate(sessions):
            step = 0.02 if kdcode == PIT_INACTIVE else 0.004 * np.sin(day_idx / 3.0 + stock_idx)
            price *= 1.0 + step
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": session,
                    "open": price,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price * 1.001,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _write_predictions(directory: Path, kdcodes: list[str]) -> Path:
    """One CSV per session; the PIT-inactive name always ranks first."""
    directory.mkdir(parents=True, exist_ok=True)
    for day_idx, session in enumerate(_sessions()):
        rows = [
            {
                "kdcode": kdcode,
                "dt": session,
                "score": 10.0
                if kdcode == PIT_INACTIVE
                else float(np.cos(day_idx / 4.0 + stock_idx)),
            }
            for stock_idx, kdcode in enumerate(kdcodes)
        ]
        pd.DataFrame(rows).to_csv(directory / f"{session}.csv", index=False)
    return directory


@pytest.fixture
def pit_fixture(tmp_path: Path) -> dict[str, Any]:
    """Full universe plus PIT intervals, and a universe the name never belonged to."""
    full_csv = tmp_path / "stock_data.csv"
    _price_panel([*PIT_ACTIVE, PIT_INACTIVE]).to_csv(full_csv, index=False)

    active_only_csv = tmp_path / "stock_data_active_only.csv"
    _price_panel(PIT_ACTIVE).to_csv(active_only_csv, index=False)

    pit_csv = tmp_path / "pit_universe.csv"
    pd.DataFrame(
        [
            {"kdcode": kdcode, "valid_from": TEST_START, "valid_to": "2025-12-31"}
            for kdcode in PIT_ACTIVE
        ]
    ).to_csv(pit_csv, index=False)

    return {
        "full_csv": full_csv,
        "active_only_csv": active_only_csv,
        "pit_csv": pit_csv,
        "full_predictions": _write_predictions(
            tmp_path / "preds_full", [*PIT_ACTIVE, PIT_INACTIVE]
        ),
        "active_predictions": _write_predictions(tmp_path / "preds_active", PIT_ACTIVE),
    }


def _stock_data(csv_path: Path) -> pd.DataFrame:
    return calculate_forward_returns(
        load_stock_data(str(csv_path), TEST_START, TEST_END), label_t=LABEL_T
    )


def _run(
    fixture: dict[str, Any],
    *,
    rebalance_style: str,
    holding_period: int = HOLDING_PERIOD,
    with_pit: bool,
) -> dict[str, Any]:
    pit_universe_df = load_pit_universe_for_backtest(str(fixture["pit_csv"])) if with_pit else None
    return dispatch_trading_simulation(
        predictions_df=load_predictions(str(fixture["full_predictions"])),
        stock_data_df=_stock_data(fixture["full_csv"]),
        top_k=TOP_K,
        label_t=LABEL_T,
        holding_period=holding_period,
        rebalance_style=rebalance_style,
        transaction_costs=None,
        rank_drop_gate=None,
        pit_universe_df=pit_universe_df,
    )


def _held_kdcodes(sim_results: dict[str, Any]) -> set[str]:
    return {str(record["kdcode"]) for record in sim_results["daily_holdings"]}


def _eval_config(**overrides: Any) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "top_k": TOP_K,
            "label_t": LABEL_T,
            "test_start": TEST_START,
            "test_end": TEST_END,
        }
    )
    config.update(overrides)
    return config


# ── D4(a): the dispatch refuses a mixed-basis run ────────────────────────


def _simulator_without_pit_support(**kwargs: Any) -> dict[str, Any]:
    """Stand-in for a simulator that has lost (or never had) PIT admission."""
    _simulator_without_pit_support.calls.append(kwargs)
    return {"dates": [], "portfolio_returns": np.array([])}


_simulator_without_pit_support.calls = []


@pytest.mark.parametrize(
    ("rebalance_style", "target"),
    [
        ("block", "simulate_trading_strategy_block"),
        ("staggered", "simulate_trading_strategy_staggered"),
    ],
)
def test_dispatch_refuses_pit_universe_when_simulator_cannot_admit_it(
    pit_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    rebalance_style: str,
    target: str,
) -> None:
    """A simulator that cannot take the PIT universe must not be run with one."""
    _simulator_without_pit_support.calls = []
    monkeypatch.setattr(
        f"mci_gru.evaluation.backtest_engine.{target}",
        _simulator_without_pit_support,
    )

    with pytest.raises(ValueError) as excinfo:
        _run(pit_fixture, rebalance_style=rebalance_style, with_pit=True)

    message = str(excinfo.value)
    assert "pit_universe_df" in message
    assert f"holding_period={HOLDING_PERIOD}" in message
    assert rebalance_style in message
    assert "pit_universe_csv" in message
    assert _simulator_without_pit_support.calls == []

    # Without a PIT universe there is no mixed basis to refuse: the path still runs.
    _run(pit_fixture, rebalance_style=rebalance_style, with_pit=False)
    assert len(_simulator_without_pit_support.calls) == 1


def test_shipped_simulators_all_accept_the_pit_universe(pit_fixture: dict[str, Any]) -> None:
    """Every dispatchable path admits on the PIT basis, so production never refuses."""
    for holding_period, rebalance_style in [(1, "staggered"), (5, "block"), (5, "staggered")]:
        sim_results = _run(
            pit_fixture,
            rebalance_style=rebalance_style,
            holding_period=holding_period,
            with_pit=True,
        )
        assert len(sim_results["dates"]) > 0


# ── D4(b): one admission basis for portfolio, benchmark, and artifacts ───


@pytest.mark.parametrize("rebalance_style", ["block", "staggered"])
def test_pit_inactive_name_is_never_held(pit_fixture: dict[str, Any], rebalance_style: str) -> None:
    """The top-scoring name is held only when the PIT universe admits it."""
    without_pit = _run(pit_fixture, rebalance_style=rebalance_style, with_pit=False)
    with_pit = _run(pit_fixture, rebalance_style=rebalance_style, with_pit=True)

    assert PIT_INACTIVE in _held_kdcodes(without_pit)
    assert PIT_INACTIVE not in _held_kdcodes(with_pit)
    assert _held_kdcodes(with_pit) <= set(PIT_ACTIVE)


@pytest.mark.parametrize("rebalance_style", ["block", "staggered"])
def test_internal_benchmark_uses_the_pit_filtered_universe(
    pit_fixture: dict[str, Any], rebalance_style: str
) -> None:
    """Each booked benchmark return is the equal-weight mean over PIT-active rows only."""
    stock_data = _stock_data(pit_fixture["full_csv"])
    active_mean = (
        stock_data[stock_data["kdcode"] != PIT_INACTIVE].groupby("dt")["open_to_open_return"].mean()
    )
    full_mean = stock_data.groupby("dt")["open_to_open_return"].mean()

    sim_results = _run(pit_fixture, rebalance_style=rebalance_style, with_pit=True)
    booked = pd.Series(sim_results["benchmark_returns"], index=sim_results["dates"], dtype=float)
    expected = active_mean.reindex(booked.index).fillna(0.0)
    unfiltered = full_mean.reindex(booked.index).fillna(0.0)

    np.testing.assert_allclose(booked.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0.0)
    assert not np.allclose(expected.to_numpy(), unfiltered.to_numpy(), rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("rebalance_style", ["block", "staggered"])
def test_pit_run_matches_a_universe_that_never_contained_the_name(
    pit_fixture: dict[str, Any], rebalance_style: str
) -> None:
    """Public evaluate(): a PIT-restricted run equals a genuinely restricted universe.

    Portfolio and benchmark must share one basis for this to hold, so it fails if
    either side is built on the full universe.
    """
    pit_metrics = evaluate(
        str(pit_fixture["full_predictions"]),
        _eval_config(
            data_file=str(pit_fixture["full_csv"]),
            pit_universe_csv=str(pit_fixture["pit_csv"]),
            holding_period=HOLDING_PERIOD,
            rebalance_style=rebalance_style,
        ),
    )
    restricted_metrics = evaluate(
        str(pit_fixture["active_predictions"]),
        _eval_config(
            data_file=str(pit_fixture["active_only_csv"]),
            pit_universe_csv=None,
            holding_period=HOLDING_PERIOD,
            rebalance_style=rebalance_style,
        ),
    )

    for key in ("ARR", "total_return", "benchmark_return", "excess_return", "IR", "MSE"):
        assert pit_metrics[key] == pytest.approx(restricted_metrics[key], rel=1e-9, abs=1e-15)
    assert pit_metrics["num_trading_days"] == restricted_metrics["num_trading_days"]


def test_saved_artifacts_exclude_pit_inactive_holdings(
    pit_fixture: dict[str, Any], tmp_path: Path
) -> None:
    """The saved holdings artifacts inherit the same PIT admission basis."""
    config = _eval_config(
        data_file=str(pit_fixture["full_csv"]),
        pit_universe_csv=str(pit_fixture["pit_csv"]),
        holding_period=HOLDING_PERIOD,
        rebalance_style="block",
    )
    results = evaluate(str(pit_fixture["full_predictions"]), config)
    sim_results = _run(pit_fixture, rebalance_style="block", with_pit=True)

    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    save_backtest_results(results, str(backtest_dir), sim_results=sim_results, config=config)

    for filename in ("daily_holdings.csv", "holdings_summary.csv", "portfolio_composition.csv"):
        frame = pd.read_csv(backtest_dir / filename)
        held = set(frame["kdcode"].astype(str))
        assert PIT_INACTIVE not in held, filename
        assert held <= set(PIT_ACTIVE), filename

    saved_metrics = json.loads((backtest_dir / "backtest_metrics.json").read_text())
    assert saved_metrics["benchmark_return"] == pytest.approx(results["benchmark_return"])


# ── D5: with no PIT universe, nothing changes ────────────────────────────


@pytest.mark.parametrize(
    ("holding_period", "rebalance_style", "simulator"),
    [
        (1, "staggered", simulate_trading_strategy),
        (5, "block", simulate_trading_strategy_block),
        (5, "staggered", simulate_trading_strategy_staggered),
    ],
)
def test_dispatch_matches_direct_simulator_call_without_pit(
    pit_fixture: dict[str, Any],
    holding_period: int,
    rebalance_style: str,
    simulator: Any,
) -> None:
    """With no PIT universe the unified dispatch reproduces the legacy call exactly."""
    stock_data = _stock_data(pit_fixture["full_csv"])
    predictions_df = load_predictions(str(pit_fixture["full_predictions"]))
    legacy_kwargs: dict[str, Any] = {
        "predictions_df": predictions_df,
        "stock_data_df": stock_data,
        "top_k": TOP_K,
        "label_t": LABEL_T,
        "transaction_costs": None,
        "rank_drop_gate": None,
    }
    if holding_period != 1:
        legacy_kwargs["holding_period"] = holding_period

    expected = simulator(**legacy_kwargs)
    got = dispatch_trading_simulation(
        predictions_df=predictions_df,
        stock_data_df=stock_data,
        top_k=TOP_K,
        label_t=LABEL_T,
        holding_period=holding_period,
        rebalance_style=rebalance_style,
        transaction_costs=None,
        rank_drop_gate=None,
        pit_universe_df=None,
    )

    assert got["dates"] == expected["dates"]
    np.testing.assert_array_equal(got["portfolio_returns"], expected["portfolio_returns"])
    np.testing.assert_array_equal(got["benchmark_returns"], expected["benchmark_returns"])
    assert got["daily_holdings"] == expected["daily_holdings"]
    assert got["trade_records"] == expected["trade_records"]


def test_admit_pit_candidates_is_a_noop_without_a_pit_universe() -> None:
    """The shared admission helper must not perturb non-PIT runs."""
    day_preds = pd.DataFrame(
        [
            {"kdcode": "STK01", "dt": TEST_START, "score": 1.0},
            {"kdcode": PIT_INACTIVE, "dt": TEST_START, "score": 9.0},
        ]
    )

    pd.testing.assert_frame_equal(admit_pit_candidates(day_preds, None, TEST_START), day_preds)

    pit_universe_df = pd.DataFrame(
        [{"kdcode": "STK01", "valid_from": TEST_START, "valid_to": "2025-12-31"}]
    )
    admitted = admit_pit_candidates(day_preds, pit_universe_df, TEST_START)
    assert admitted["kdcode"].tolist() == ["STK01"]
