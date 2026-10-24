from pathlib import Path

import numpy as np

from mci_gru.evaluation import backtest_engine
from mci_gru.evaluation.replay_book import ReplayBook, ReplayStrategy


def test_plot_equity_curve_uses_agg_backend_when_saving(monkeypatch, tmp_path: Path) -> None:
    backend_calls: list[tuple[str, bool | None]] = []
    original_use = backtest_engine.matplotlib.use

    def record_backend_use(backend: str, *args, **kwargs) -> None:
        backend_calls.append((backend, kwargs.get("force")))
        original_use(backend, *args, **kwargs)

    monkeypatch.setattr(backtest_engine.matplotlib, "use", record_backend_use)
    empty = np.asarray([], dtype=float)
    book = ReplayBook(
        strategy=ReplayStrategy.open_to_open(),
        top_k=1,
        sessions=["2026-01-02", "2026-01-05"],
        gross_returns=np.array([0.01, -0.005]),
        net_returns=np.array([0.01, -0.005]),
        benchmark_returns=np.array([0.002, 0.003]),
        costs=np.zeros(2),
        turnover=np.zeros(2),
        signal_dates=[],
        period_returns=empty,
        benchmark_period_returns=empty,
    )
    output_path = tmp_path / "equity_curve.png"

    backtest_engine.plot_equity_curve(book, output_path=output_path)

    assert ("Agg", True) in backend_calls
    assert output_path.exists()
