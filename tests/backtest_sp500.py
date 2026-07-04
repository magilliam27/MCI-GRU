"""DEPRECATED shim — the backtest engine moved in WS-C.

The engine now lives in ``mci_gru/evaluation/backtest_engine.py`` and the CLI
entry point is ``scripts/backtest_sp500.py``. This shim re-exports the engine
namespace and keeps ``python tests/backtest_sp500.py`` working for one release
cycle; import from ``mci_gru.evaluation.backtest_engine`` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mci_gru.evaluation.backtest_engine import *  # noqa: F403,E402
from mci_gru.evaluation.backtest_engine import main  # noqa: E402

if __name__ == "__main__":
    main()
