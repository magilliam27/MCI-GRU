"""PIT admission must be consistent across every backtest replay path (issue #116).

A ``holding_period > 1`` run once compared a full-universe portfolio against a
PIT-filtered benchmark and reported the difference as excess return; a dispatch guard
then refused any simulator that could not admit candidates on the PIT basis. Since
#340 every strategy (the default close-to-close tranches and the open-to-open daily,
staggered and block variants) is one book, ``build_replay_book``, which always takes
the PIT membership, so there is no second path to guard. These tests pin the admission
basis of the portfolio, the internal benchmark and the saved artifacts on each strategy.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from mci_gru.evaluation.backtest_engine import (
    DEFAULT_CONFIG,
    replay,
    save_backtest_results,
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

# (strategy, holding_period, rebalance_style): every way the replay can trade.
STRATEGIES = [
    ("close_tranches", None, None),
    ("open_to_open", 1, None),
    ("open_to_open", HOLDING_PERIOD, "staggered"),
    ("open_to_open", HOLDING_PERIOD, "block"),
]


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


def _eval_config(fixture: dict[str, Any], strategy, *, with_pit: bool) -> dict[str, Any]:
    name, holding_period, rebalance_style = strategy
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "top_k": TOP_K,
            "label_t": LABEL_T,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "strategy": name,
            "holding_period": holding_period,
            "rebalance_style": rebalance_style,
            "data_file": str(fixture["full_csv"] if with_pit else fixture["active_only_csv"]),
            "pit_universe_csv": str(fixture["pit_csv"]) if with_pit else None,
        }
    )
    return config


def _replay(fixture: dict[str, Any], strategy, *, with_pit: bool, restricted: bool = False):
    config = _eval_config(fixture, strategy, with_pit=with_pit)
    if not with_pit and not restricted:
        config["data_file"] = str(fixture["full_csv"])
    predictions = fixture["active_predictions" if restricted else "full_predictions"]
    return replay(str(predictions), config)


def _held(book) -> set[str]:
    return {str(row["kdcode"]) for row in book.holdings}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pit_inactive_name_is_never_held(pit_fixture: dict[str, Any], strategy) -> None:
    """The top-scoring name is held only when the PIT universe admits it."""
    _, without_pit = _replay(pit_fixture, strategy, with_pit=False)
    _, with_pit = _replay(pit_fixture, strategy, with_pit=True)

    assert PIT_INACTIVE in _held(without_pit)
    assert PIT_INACTIVE not in _held(with_pit)
    assert _held(with_pit) <= set(PIT_ACTIVE)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pit_run_matches_a_universe_that_never_contained_the_name(
    pit_fixture: dict[str, Any], strategy
) -> None:
    """A PIT-restricted run equals a genuinely restricted universe, benchmark included.

    Portfolio and benchmark must share one basis for this to hold, so it fails if
    either side is built on the full universe.
    """
    pit_metrics, pit_book = _replay(pit_fixture, strategy, with_pit=True)
    restricted_metrics, _ = _replay(pit_fixture, strategy, with_pit=False, restricted=True)
    _, unfiltered = _replay(pit_fixture, strategy, with_pit=False)

    for key in ("ARR", "total_return", "benchmark_return", "excess_return", "IR"):
        assert pit_metrics[key] == pytest.approx(restricted_metrics[key], rel=1e-9, abs=1e-15)
    assert pit_metrics["num_trading_days"] == restricted_metrics["num_trading_days"]
    # The control: the full universe's benchmark is different, so the match means something.
    assert not np.allclose(pit_book.benchmark_returns, unfiltered.benchmark_returns)


def test_saved_artifacts_exclude_pit_inactive_holdings(
    pit_fixture: dict[str, Any], tmp_path: Path
) -> None:
    """The saved holdings artifacts inherit the same PIT admission basis."""
    strategy = ("open_to_open", HOLDING_PERIOD, "block")
    results, book = _replay(pit_fixture, strategy, with_pit=True)

    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    config = _eval_config(pit_fixture, strategy, with_pit=True)
    save_backtest_results(results, str(backtest_dir), book=book, config=config)

    for filename in ("holdings.csv", "holdings_summary.csv", "trade_journal.csv"):
        frame = pd.read_csv(backtest_dir / filename)
        held = set(frame["kdcode"].astype(str))
        assert PIT_INACTIVE not in held, filename
        assert held <= set(PIT_ACTIVE), filename

    saved_metrics = json.loads((backtest_dir / "backtest_metrics.json").read_text())
    assert saved_metrics["benchmark_return"] == pytest.approx(results["benchmark_return"])
