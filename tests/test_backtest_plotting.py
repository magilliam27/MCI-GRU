from pathlib import Path

import numpy as np
import pandas as pd

from mci_gru.evaluation import backtest_engine


def test_plot_equity_curve_uses_agg_backend_when_saving(monkeypatch, tmp_path: Path) -> None:
    backend_calls: list[tuple[str, bool | None]] = []
    original_use = backtest_engine.matplotlib.use

    def record_backend_use(backend: str, *args, **kwargs) -> None:
        backend_calls.append((backend, kwargs.get("force")))
        original_use(backend, *args, **kwargs)

    monkeypatch.setattr(backtest_engine.matplotlib, "use", record_backend_use)
    monkeypatch.setattr(backtest_engine, "load_predictions", lambda _path: pd.DataFrame())
    monkeypatch.setattr(
        backtest_engine,
        "calculate_forward_returns",
        lambda stock_data, label_t: stock_data,
    )
    monkeypatch.setattr(backtest_engine, "load_pit_universe_for_backtest", lambda _path: None)
    monkeypatch.setattr(
        backtest_engine,
        "simulate_trading_strategy",
        lambda **_kwargs: {
            "dates": ["2026-01-02", "2026-01-03"],
            "portfolio_returns": np.array([0.01, -0.005]),
            "benchmark_returns": np.array([0.002, 0.003]),
            "transaction_costs_enabled": False,
        },
    )
    monkeypatch.setattr(
        backtest_engine,
        "calendar_returns_for_evaluation_window",
        lambda *_args, **_kwargs: (
            ["2026-01-02", "2026-01-03"],
            np.array([0.01, -0.005]),
            np.array([0.002, 0.003]),
        ),
    )

    output_path = tmp_path / "equity_curve.png"

    backtest_engine.plot_equity_curve(
        predictions_dir=tmp_path,
        stock_data=pd.DataFrame({"dt": pd.to_datetime(["2026-01-02"])}),
        config={
            "label_t": 1,
            "pit_universe_csv": None,
            "transaction_costs": {"enabled": False},
            "rank_drop_gate": {"enabled": False, "min_rank_drop": 10},
            "holding_period": 1,
            "rebalance_style": "staggered",
            "top_k": 1,
        },
        output_path=output_path,
    )

    assert ("Agg", True) in backend_calls
    assert output_path.exists()
