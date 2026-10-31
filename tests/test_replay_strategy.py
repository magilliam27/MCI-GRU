"""The replay trades the trained return and reports one consistent set of metrics (#340).

The replay used to buy at open D+1 and sell at open D+2, a hold the model never learned;
read returns from per-stock row shifts, so a missing bar stretched one booked "day" over
two sessions (the #225 class); drop a holding with no next price and re-weight the rest;
hold both Alphabet classes and count Alphabet twice in its benchmark; and report ARR,
Sharpe and drawdown on traded days while IR and the benchmark used a padded calendar.
These tests pin the fixes:

* the default book buys at close D+1 and sells at close D+5 in staggered tranches, with
  buy-and-hold weights and costs booked on the session a trade happens;
* returns are read on the panel's session axis, a missing exit books the last close and
  then cash (E3), and the event is recorded;
* at most one share class per issuer is held and the benchmark weights the issuer once;
* every metric and output file reads one daily series, drawdown starts from 1.0, IR is
  annualised, monthly returns compound, the paper's ASR sits beside a standard Sharpe,
  and BHY needs the trial ledger;
* on a synthetic run, the replay's top-k excess equals the in-run summary's (#336, the
  summary #337 re-scores) on the label's legs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

import run_experiment
from mci_gru.data.preprocessing import compute_labels
from mci_gru.evaluation import backtest_engine
from mci_gru.evaluation.replay_book import (
    CostModel,
    ReplayStrategy,
    build_replay_book,
    issuer_keys,
)
from mci_gru.training import ensemble
from scripts import rescore_run

SESSIONS = pd.bdate_range("2025-01-06", periods=16).strftime("%Y-%m-%d").tolist()


def _panel(closes: dict[str, list[float]], opens: dict[str, list[float]] | None = None):
    """A long panel; NaN in a price list means the stock has no bar that session."""
    rows = []
    for kdcode, series in closes.items():
        for t, close in enumerate(series):
            if np.isnan(close):
                continue
            open_ = opens[kdcode][t] if opens else close
            rows.append({"kdcode": kdcode, "dt": SESSIONS[t], "open": open_, "close": close})
    return pd.DataFrame(rows)


def _walk(start: float, steps: list[float]) -> list[float]:
    out = [start]
    for step in steps:
        out.append(out[-1] * (1.0 + step))
    return out


def _scores(order: list[str], kdcodes: list[str], dates: list[str]) -> np.ndarray:
    """The same ranking every date: ``order[0]`` highest."""
    row = np.array([len(order) - order.index(k) if k in order else np.nan for k in kdcodes])
    return np.tile(row, (len(dates), 1)).astype(np.float64)


def _book(panel, scores, dates, kdcodes, **kwargs):
    kwargs.setdefault("strategy", ReplayStrategy.close_tranches(label_t=5))
    kwargs.setdefault("top_k", 1)
    return build_replay_book(panel, scores, dates, kdcodes, **kwargs)


# -- The default trades the training label -------------------------------------------


def test_the_default_book_buys_close_d_plus_1_and_sells_close_d_plus_5() -> None:
    rng = np.random.default_rng(3)
    kdcodes = ["A", "B", "C"]
    closes = {k: _walk(100.0, list(rng.normal(0, 0.02, len(SESSIONS) - 1))) for k in kdcodes}
    panel = _panel(closes)
    dates = SESSIONS[:8]

    book = _book(panel, _scores(["B", "A", "C"], kdcodes, dates), dates, kdcodes)

    labels = compute_labels(panel, kdcodes, dates, 5, fill_missing=False)
    assert book.strategy.tranches == 4
    assert book.signal_dates == dates
    # Training labels are float32.
    np.testing.assert_allclose(book.period_returns, labels[:, kdcodes.index("B")], rtol=1e-6)
    # Every tranche's full-hold growth in the daily P&L is 1 + its booked return.
    b = closes["B"]
    expected = (b[5] / b[1] - 1.0, b[11] / b[7] - 1.0)
    assert book.holdings[0]["stock_return"] == pytest.approx(expected[0])
    assert book.holdings[6]["stock_return"] == pytest.approx(expected[1])


def test_a_missing_bar_never_moves_an_endpoint() -> None:
    """A row shift would read the close one session later for every date before the gap."""
    a = _walk(100.0, [0.01] * 15)
    b = _walk(50.0, [0.03, -0.02, 0.05, 0.01, -0.04, 0.02, 0.03, -0.01, 0.02, 0.0] + [0.01] * 5)
    b[4] = np.nan  # B has no bar on session 4
    panel = _panel({"A": a, "B": b})
    dates = SESSIONS[:3]

    book = _book(panel, _scores(["B", "A"], ["A", "B"], dates), dates, ["A", "B"])

    # Signal on session 0: entry close of session 1, exit close of session 5.
    assert book.period_returns[0] == pytest.approx(b[5] / b[1] - 1.0)
    assert book.period_returns[2] == pytest.approx(b[7] / b[3] - 1.0)

    # The daily P&L reads the same session axis: one tranche holding B from close 1 to
    # close 5 grows by b5/b1, flat through the missing session, then b5/b3 on session 5.
    one = _book(
        panel,
        _scores(["B", "A"], ["A", "B"], dates[:1]),
        dates[:1],
        ["A", "B"],
        strategy=ReplayStrategy.close_tranches(label_t=5, tranches=1),
    )
    assert one.sessions == SESSIONS[1:6]
    assert np.prod(1.0 + one.gross_returns) == pytest.approx(b[5] / b[1])
    assert one.gross_returns[3] == 0.0
    assert one.gross_returns[4] == pytest.approx(b[5] / b[3] - 1.0)


def test_a_missing_exit_books_the_last_close_then_cash_and_is_recorded() -> None:
    """D delists after session 3: its loss is booked, and it is not dropped from the
    tranche with the others re-weighted."""
    up = _walk(100.0, [0.01] * 15)
    falls = [100.0, 100.0, 90.0, 80.0] + [np.nan] * 12
    panel = _panel({"A": up, "D": falls})
    dates = SESSIONS[:1]

    book = _book(panel, _scores(["D", "A"], ["A", "D"], dates), dates, ["A", "D"], top_k=2)

    expected = 0.5 * (up[5] / up[1] - 1.0) + 0.5 * (80.0 / 100.0 - 1.0)
    assert book.period_returns[0] == pytest.approx(expected)
    # The daily P&L agrees: the four-sleeve book holds this tranche in one sleeve.
    assert np.prod(1.0 + book.gross_returns) - 1.0 == pytest.approx(expected / 4.0)
    assert book.events == [
        {
            "signal_date": SESSIONS[0],
            "kdcode": "D",
            "kind": "missing_exit",
            "scheduled_session": SESSIONS[5],
            "booked_session": SESSIONS[3],
        }
    ]


def test_weights_drift_with_prices_inside_a_tranche() -> None:
    """A doubles then doubles again; B is flat. Buy and hold: the second day returns
    (400 + 100) / (200 + 100) - 1 = 2/3, not the daily re-weighted (1.0 + 0) / 2."""
    a = [100.0, 100.0, 200.0, 400.0, 400.0, 400.0] + [400.0] * 10
    b = [100.0] * 16
    panel = _panel({"A": a, "B": b})
    dates = SESSIONS[:1]
    strategy = ReplayStrategy.close_tranches(label_t=5, tranches=1)

    book = _book(
        panel, _scores(["A", "B"], ["A", "B"], dates), dates, ["A", "B"], strategy=strategy, top_k=2
    )

    returns = dict(zip(book.sessions, book.gross_returns, strict=True))
    assert returns[SESSIONS[2]] == pytest.approx(0.5)
    assert returns[SESSIONS[3]] == pytest.approx(2.0 / 3.0)


def test_costs_are_booked_on_the_sessions_trades_happen() -> None:
    rng = np.random.default_rng(11)
    kdcodes = ["A", "B", "C", "D"]
    closes = {k: _walk(100.0, list(rng.normal(0, 0.01, len(SESSIONS) - 1))) for k in kdcodes}
    panel = _panel(closes)
    dates = SESSIONS[:9]
    scores = rng.normal(size=(len(dates), len(kdcodes)))
    strategy = ReplayStrategy.close_tranches(label_t=5, tranches=1)
    costs = CostModel(bid_ask_spread=0.0010, slippage=0.0005)

    book = _book(panel, scores, dates, kdcodes, strategy=strategy, top_k=2, costs=costs)

    trade_sessions = {SESSIONS[1], SESSIONS[5], SESSIONS[9], SESSIONS[13]}
    charged = {s for s, c in zip(book.sessions, book.costs, strict=True) if c > 0}
    assert charged <= trade_sessions and SESSIONS[1] in charged and SESSIONS[13] in charged
    for s, gross, net in zip(book.sessions, book.gross_returns, book.net_returns, strict=True):
        if s not in trade_sessions:
            assert net == pytest.approx(gross, abs=1e-15)
    # The first buy is the whole sleeve at 5 + 5 bps per side.
    assert book.costs[book.sessions.index(SESSIONS[1])] == pytest.approx(0.0010)


def test_a_name_carried_into_the_next_tranche_trades_only_its_drift() -> None:
    """A rises 10% in the first hold, B is flat; both are picked again at the rollover.

    Back to equal weight trades 0.025 of each side per unit invested, so the rollover
    costs c x 0.05 / 1.05 of the book, not a sale and re-purchase of both names."""
    a = [100.0, 100.0, 110.0] + [110.0] * 13
    b = [100.0] * 16
    panel = _panel({"A": a, "B": b})
    dates = SESSIONS[:5]
    c = 0.0010
    costs = CostModel(bid_ask_spread=2 * c, slippage=0.0)

    book = _book(
        panel,
        _scores(["A", "B"], ["A", "B"], dates),
        dates,
        ["A", "B"],
        strategy=ReplayStrategy.close_tranches(label_t=5, tranches=1),
        top_k=2,
        costs=costs,
    )

    charged = dict(zip(book.sessions, book.costs, strict=True))
    assert charged[SESSIONS[1]] == pytest.approx(c)
    assert charged[SESSIONS[5]] == pytest.approx(c * 0.05 / 1.05)


def test_a_missing_exit_is_charged_when_it_is_sold() -> None:
    """B has no close after session 2: it is sold at that close, and charged there."""
    a = [100.0] * 16
    b = [100.0, 100.0, 100.0] + [np.nan] * 13
    panel = _panel({"A": a, "B": b})
    dates = SESSIONS[:1]
    c = 0.0010
    costs = CostModel(bid_ask_spread=2 * c, slippage=0.0)

    book = _book(
        panel,
        _scores(["B", "A"], ["A", "B"], dates),
        dates,
        ["A", "B"],
        strategy=ReplayStrategy.close_tranches(label_t=5, tranches=1),
        top_k=2,
        costs=costs,
    )

    charged = dict(zip(book.sessions, book.costs, strict=True))
    assert charged[SESSIONS[2]] == pytest.approx(c * 0.5)
    assert charged[SESSIONS[3]] == 0.0 and charged[SESSIONS[4]] == 0.0
    assert charged[SESSIONS[5]] == pytest.approx(c * 0.5 / (1.0 - 0.5 * c))


# -- One share class per issuer ------------------------------------------------------


ALPHABET = ["GOOG.OQ", "GOOGL.OQ", "MSFT.OQ", "AAPL.OQ"]


def _alphabet_book(**kwargs):
    rng = np.random.default_rng(7)
    closes = {k: _walk(100.0, list(rng.normal(0, 0.01, len(SESSIONS) - 1))) for k in ALPHABET}
    panel = _panel(closes)
    dates = SESSIONS[:6]
    scores = _scores(["GOOGL.OQ", "GOOG.OQ", "MSFT.OQ", "AAPL.OQ"], ALPHABET, dates)
    return _book(panel, scores, dates, ALPHABET, top_k=2, **kwargs), panel, dates


def test_the_portfolio_holds_one_share_class_per_issuer() -> None:
    book, _, _ = _alphabet_book()

    held = pd.DataFrame(book.holdings)
    assert set(held["kdcode"]) == {"GOOGL.OQ", "MSFT.OQ"}
    assert held.groupby("signal_date")["issuer"].nunique().eq(2).all()


def test_the_benchmark_weights_the_issuer_once() -> None:
    book, panel, dates = _alphabet_book()

    labels = compute_labels(panel, ALPHABET, dates, 5, fill_missing=False)
    alphabet = labels[:, :2].mean(axis=1)
    expected = (alphabet + labels[:, 2] + labels[:, 3]) / 3.0
    np.testing.assert_allclose(
        book.benchmark_period_returns, expected[: len(book.signal_dates)], rtol=1e-6
    )
    assert issuer_keys(ALPHABET) == ["Alphabet", "Alphabet", "MSFT.OQ", "AAPL.OQ"]


# -- The open-to-open variant --------------------------------------------------------


def test_the_open_to_open_variant_buys_open_d_plus_1_and_sells_open_d_plus_2() -> None:
    closes = {"A": _walk(100.0, [0.01] * 15), "B": _walk(100.0, [0.0] * 15)}
    opens = {"A": _walk(99.0, [0.02] * 15), "B": _walk(100.0, [0.0] * 15)}
    panel = _panel(closes, opens)
    dates = SESSIONS[:4]

    book = _book(
        panel,
        _scores(["A", "B"], ["A", "B"], dates),
        dates,
        ["A", "B"],
        strategy=ReplayStrategy.open_to_open(),
    )

    assert book.strategy.name == "open_to_open" and book.strategy.tranches == 1
    o = opens["A"]
    np.testing.assert_allclose(book.period_returns, [o[i + 2] / o[i + 1] - 1 for i in range(4)])
    # A daily return ends on the open it is priced at: the book buys at the open of
    # session 1 and its first held return, open 1 to open 2, ends on session 2.
    returns = dict(zip(book.sessions, book.gross_returns, strict=True))
    assert book.sessions[0] == SESSIONS[1] and returns[SESSIONS[1]] == 0.0
    assert returns[SESSIONS[2]] == pytest.approx(o[2] / o[1] - 1)


def test_the_rank_drop_gate_keeps_a_holding_until_its_rank_falls_far_enough() -> None:
    kdcodes = [f"S{i}" for i in range(6)]
    closes = {k: [100.0] * 16 for k in kdcodes}
    panel = _panel(closes)
    dates = SESSIONS[:3]
    scores = np.array(
        [
            [6, 5, 4, 3, 2, 1],  # S0 rank 1
            [3, 6, 5, 4, 2, 1],  # S0 falls to rank 4: a drop of 3
            [1, 6, 5, 4, 3, 2],  # S0 falls to rank 6: a drop of 2 from rank 4
        ],
        dtype=np.float64,
    )
    strategy = ReplayStrategy.open_to_open()

    kept = _book(panel, scores, dates, kdcodes, strategy=strategy, min_rank_drop=4)
    dropped = _book(panel, scores, dates, kdcodes, strategy=strategy, min_rank_drop=3)

    assert [h["kdcode"] for h in kept.holdings] == ["S0", "S0", "S0"]
    assert [h["kdcode"] for h in dropped.holdings] == ["S0", "S1", "S1"]
    assert dropped.rank_gate["gate_exits"] == 1


def test_the_rank_drop_gate_compares_each_sleeve_with_its_own_last_tranche() -> None:
    """Four sleeves: signal 4 rolls over sleeve 0, opened on signal 0."""
    kdcodes = [f"S{i}" for i in range(6)]
    panel = _panel({k: [100.0] * 16 for k in kdcodes})
    dates = SESSIONS[:5]
    first = [6, 5, 4, 3, 2, 1]  # S0 rank 1
    other = [1, 6, 5, 4, 3, 2]  # S0 rank 6 on the dates in between
    later = [3, 6, 5, 4, 2, 1]  # S0 rank 4 on signal 4: a drop of 3 from sleeve 0's
    scores = np.array([first, other, other, other, later], dtype=np.float64)

    kept = _book(panel, scores, dates, kdcodes, min_rank_drop=4)
    dropped = _book(panel, scores, dates, kdcodes, min_rank_drop=3)

    assert [h["kdcode"] for h in kept.holdings] == ["S0", "S1", "S1", "S1", "S0"]
    assert [h["kdcode"] for h in dropped.holdings] == ["S0", "S1", "S1", "S1", "S1"]


# -- Metrics on one daily series -----------------------------------------------------


def test_drawdown_starts_from_one() -> None:
    assert backtest_engine.calculate_mdd(np.array([-0.10, -0.10, 0.05])) == pytest.approx(-0.19)


def test_information_ratio_is_annualised() -> None:
    rng = np.random.default_rng(0)
    portfolio = rng.normal(0.001, 0.01, 500)
    benchmark = rng.normal(0.0, 0.01, 500)
    excess = portfolio - benchmark

    ir = backtest_engine.calculate_ir(portfolio, benchmark)

    assert ir == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(252))


def test_standard_sharpe_sits_beside_the_papers_asr() -> None:
    returns = np.array([0.03, -0.03] * 50)

    metrics = backtest_engine.portfolio_metrics(returns, np.zeros_like(returns))

    assert metrics["sharpe"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["ASR"] < 0  # paper ASR = geometric ARR / AVoL carries the volatility drag
    assert metrics["ASR"] == pytest.approx(metrics["ARR"] / metrics["AVoL"])


def test_monthly_returns_compound() -> None:
    sessions = pd.bdate_range("2025-03-03", periods=21).strftime("%Y-%m-%d").tolist()

    monthly = backtest_engine.monthly_performance(sessions, np.full(21, 0.01), np.zeros(21))

    assert monthly["portfolio_return"].iloc[0] == pytest.approx(1.01**21 - 1)


def test_holm_for_one_strategy_and_bhy_only_with_the_trial_ledger(tmp_path: Path) -> None:
    single = backtest_engine.multiple_testing(
        sharpe=1.5, num_years=3.0, num_tests=50, method="holm"
    )
    assert single["adjusted_p_value"] == pytest.approx(min(1.0, single["original_p_value"] * 50))

    with pytest.raises(ValueError, match="trial ledger"):
        backtest_engine.multiple_testing(sharpe=1.5, num_years=3.0, num_tests=50, method="bhy")

    ledger = tmp_path / "trials.csv"
    pd.DataFrame({"p_value": [0.5, 0.2, 0.9]}).to_csv(ledger, index=False)
    bhy = backtest_engine.multiple_testing(
        sharpe=1.5, num_years=3.0, num_tests=1, method="bhy", trial_ledger=str(ledger)
    )
    assert bhy["num_tests"] == 4
    # The step-up keeps the top trial at or below its single-step inflation.
    raw = bhy["original_p_value"] * 4 * (1 + 1 / 2 + 1 / 3 + 1 / 4)
    assert bhy["adjusted_p_value"] == pytest.approx(min(1.0, raw))


# -- End to end through the CLI on the golden fixture ---------------------------------


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "backtest_golden"


def _cli(tmp_path: Path, *extra: str) -> Path:
    run = tmp_path / "run"
    shutil.copytree(FIXTURE_ROOT / "run" / "averaged_predictions", run / "averaged_predictions")
    backtest_engine.main(
        [
            "--predictions_dir",
            str(run / "averaged_predictions"),
            "--data_file",
            str(FIXTURE_ROOT / "stock_data.csv"),
            "--test_start",
            "2025-02-03",
            "--test_end",
            "2025-04-30",
            "--top_k",
            "3",
            "--label_t",
            "5",
            "--auto_save",
            "--disable_mlflow_autolink",
            *extra,
        ]
    )
    return run / "backtest"


def test_every_metric_and_file_reads_one_daily_series(tmp_path: Path) -> None:
    out = _cli(tmp_path, "--transaction_costs")

    metrics = json.loads((out / "backtest_metrics.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(out / "daily_returns.csv")
    cumulative = pd.read_csv(out / "cumulative_performance.csv")
    monthly = pd.read_csv(out / "monthly_performance.csv")

    assert len(daily) == metrics["num_trading_days"] == len(cumulative)
    assert daily["date"].iloc[0] == metrics["start_date"]
    assert daily["date"].iloc[-1] == metrics["end_date"]
    r, b = daily["portfolio_return"].to_numpy(), daily["benchmark_return"].to_numpy()
    assert np.prod(1 + r) - 1 == pytest.approx(metrics["total_return"])
    assert np.prod(1 + b) - 1 == pytest.approx(metrics["benchmark_return"])
    assert cumulative["portfolio_value"].iloc[-1] == pytest.approx(1 + metrics["total_return"])
    assert backtest_engine.calculate_mdd(r) == pytest.approx(metrics["MDD"])
    assert cumulative["portfolio_drawdown"].min() == pytest.approx(metrics["MDD"])
    assert np.prod(1 + monthly["portfolio_return"]) - 1 == pytest.approx(metrics["total_return"])
    assert backtest_engine.calculate_ir(r, b) == pytest.approx(metrics["IR"])
    gross = daily["gross_portfolio_return"].to_numpy()
    assert np.prod(1 + gross) - 1 == pytest.approx(metrics["gross_total_return"])
    assert metrics["strategy"] == "close_tranches" and metrics["tranches"] == 4
    assert "MSE" not in metrics and "MAE" not in metrics


def test_outputs_record_input_hashes(tmp_path: Path) -> None:
    out = _cli(tmp_path)

    metrics = json.loads((out / "backtest_metrics.json").read_text(encoding="utf-8"))
    config = json.loads((out / "backtest_config.json").read_text(encoding="utf-8"))
    data_sha = backtest_engine.sha256_file(FIXTURE_ROOT / "stock_data.csv")
    assert metrics["inputs"]["data_file_sha256"] == data_sha
    assert config["inputs"] == metrics["inputs"]
    files = metrics["inputs"]["prediction_files"]
    assert len(files) == len(list((FIXTURE_ROOT / "run" / "averaged_predictions").glob("*.csv")))
    assert all(len(digest) == 64 for digest in files.values())
    # The one-row CSV, the only file an --output run writes, carries them too.
    row = pd.read_csv(out / "backtest_results.csv").iloc[0]
    assert row["data_file_sha256"] == data_sha
    assert (
        isinstance(row["prediction_files_sha256"], str)
        and len(row["prediction_files_sha256"]) == 64
    )


def test_an_issuer_map_joins_share_classes_the_built_in_map_lacks(tmp_path: Path) -> None:
    issuer_map = tmp_path / "issuers.csv"
    pd.DataFrame({"kdcode": ["STK01", "STK02"], "issuer": ["Pair", "Pair"]}).to_csv(
        issuer_map, index=False
    )
    out = _cli(tmp_path, "--issuer_map", str(issuer_map))

    holdings = pd.read_csv(out / "holdings.csv")
    pair = holdings[holdings["kdcode"].isin(["STK01", "STK02"])]
    assert len(pair) > 0
    assert pair.groupby(["signal_date"])["kdcode"].nunique().max() == 1
    assert set(pair["issuer"]) == {"Pair"}
    metrics = json.loads((out / "backtest_metrics.json").read_text(encoding="utf-8"))
    assert metrics["inputs"]["issuer_map_sha256"] == backtest_engine.sha256_file(issuer_map)


def test_bhy_without_a_ledger_is_refused_before_the_replay_runs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _cli(tmp_path, "--adjustment_method", "bhy")
    assert not (tmp_path / "run" / "backtest").exists()


def test_multi_model_is_gone(tmp_path: Path) -> None:
    """It averaged member metrics (not the ensemble the run reports) and found no folders."""
    with pytest.raises(SystemExit):
        backtest_engine.main(["--multi_model", str(tmp_path)])
    assert not hasattr(backtest_engine, "evaluate_multiple_models")


# -- The replay agrees with the in-run summary ---------------------------------------


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_STOCKS = ["A.N", "B.N", "C.N", "D.N", "E.N", "F.N"]
RUN_OVERRIDES = [
    "features=base",
    "protocol.exploratory=true",
    "data.source=csv",
    "data.input_package_manifest=null",
    "data.input_package_manifest_sha256=null",
    "data.input_package_root=null",
    "data.train_start=2020-01-01",
    "data.train_end=2020-02-07",
    "data.val_start=2020-02-14",
    "data.val_end=2020-02-21",
    "data.test_start=2020-03-02",
    "data.test_end=2020-03-20",
    "model.his_t=3",
    "model.label_t=5",
    "model.gru_hidden_sizes=[8,4]",
    "model.hidden_size_gat1=8",
    "model.output_gat1=4",
    "model.gat_heads=1",
    "model.hidden_size_gat2=8",
    "model.num_hidden_states=4",
    "model.cross_attn_heads=1",
    "model.use_self_attention=false",
    "training.num_epochs=1",
    "training.num_models=2",
    "training.batch_size=2",
    "training.warmup_steps=0",
    "tracking.enabled=false",
    "evaluation.top_k_values=[2,3]",
    "graph.corr_lookback_days=20",
]


def _write_run_panel(path: Path) -> None:
    dates = pd.bdate_range("2019-11-01", periods=110).strftime("%Y-%m-%d")
    rng = np.random.default_rng(5)
    rows = []
    for s_idx, kdcode in enumerate(RUN_STOCKS):
        price = 100.0 + s_idx
        for d_idx, dt in enumerate(dates):
            price *= 1.0 + rng.normal(0.0, 0.01)
            if (kdcode, dt) == ("C.N", "2020-03-10"):
                continue  # a missing bar inside the test window
            rows.append(
                {
                    "kdcode": kdcode,
                    "dt": dt,
                    "open": price * (1.0 + rng.normal(0.0, 0.002)),
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000.0 + d_idx,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_run_pit(path: Path) -> None:
    pd.DataFrame(
        [
            {"kdcode": "A.N", "valid_from": "2019-11-01", "valid_to": ""},
            {"kdcode": "B.N", "valid_from": "2019-11-01", "valid_to": ""},
            {"kdcode": "C.N", "valid_from": "2019-11-01", "valid_to": ""},
            {"kdcode": "D.N", "valid_from": "2019-11-01", "valid_to": "2020-03-11"},
            {"kdcode": "E.N", "valid_from": "2020-03-09", "valid_to": ""},
            {"kdcode": "F.N", "valid_from": "2019-11-01", "valid_to": ""},
        ]
    ).to_csv(path, index=False)


@contextmanager
def _restored_process_state(directory: Path):
    root = logging.getLogger()
    handlers, level, cwd = list(root.handlers), root.level, os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("replay_crosscheck")
    _write_run_panel(base / "panel.csv")
    _write_run_pit(base / "pit.csv")
    overrides = [
        *RUN_OVERRIDES,
        f"data.filename={(base / 'panel.csv').as_posix()}",
        "data.use_pit_universe=true",
        f"data.pit_universe_csv={(base / 'pit.csv').as_posix()}",
        "data.pit_export_cutoff=2020-03-31",
        "data.pit_min_scoreable_stocks=1",
        "data.pit_breadth_policy=off",
    ]
    with initialize_config_dir(config_dir=str(REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose("config", overrides=overrides)
    out = base / "run"
    out.mkdir()
    patch = pytest.MonkeyPatch()
    patch.setattr(torch.cuda, "is_available", lambda: False)
    patch.setattr(ensemble, "prediction_rows_for_date", _unrounded_rows)
    try:
        with _restored_process_state(out):
            run_experiment.main.__wrapped__(cfg)
    finally:
        patch.undo()
    rescore_run.rescore_run(out)
    return out


def _unrounded_rows(predictions, kdcode_list, date, prediction_mask=None):
    """``prediction_rows_for_date`` without its 5-decimal rounding.

    The summary ranks the run's own scores; the replay ranks the saved file. Rounding
    can tie two names a one-epoch model scored nearly alike and flip a pick on one
    platform but not another, so the fixture saves full precision (as #337's does).
    """
    mask = np.ones(len(kdcode_list), bool) if prediction_mask is None else prediction_mask
    return [
        [kdcode, date, float(score)]
        for kdcode, score, keep in zip(kdcode_list, predictions, mask, strict=True)
        if keep and np.isfinite(score)
    ]


@pytest.mark.parametrize(
    "summary_file", ["evaluation_summary.json", rescore_run.RESCORED_SUMMARY_FILENAME]
)
@pytest.mark.parametrize("top_k", [2, 3])
def test_the_replay_top_k_excess_equals_the_in_run_summary(
    synthetic_run, tmp_path, top_k, summary_file
):
    """Against the summary the run wrote, and the one #337 re-scores from its inputs."""
    run = Path(shutil.copytree(synthetic_run, tmp_path / "run"))
    summary = json.loads((run / summary_file).read_text(encoding="utf-8"))["metrics"]

    backtest_engine.main(
        ["--run_dir", str(run), "--top_k", str(top_k), "--auto_save", "--disable_mlflow_autolink"]
    )

    replay = json.loads((run / "backtest" / "backtest_metrics.json").read_text(encoding="utf-8"))
    assert replay["tranches"] == 4 and replay["hold_sessions"] == 4
    assert replay["top_k_excess_period_return"] == pytest.approx(
        summary[f"top_{top_k}_excess_period_return"], abs=1e-12
    )
    assert replay["excess_cumulative_return"] == pytest.approx(
        summary[f"top_{top_k}_excess_cumulative_return"], abs=1e-12
    )
    assert replay["total_return"] == pytest.approx(
        summary[f"top_{top_k}_abs_cumulative_return"], abs=1e-12
    )
    assert replay["benchmark_return"] == pytest.approx(
        summary["universe_abs_cumulative_return"], abs=1e-12
    )
    # The replay's series also carries the first entry session (zero gross, its buying
    # cost net), which the summary's book starts after.
    assert replay["num_trading_days"] == summary["book_sessions"] + 1
