"""DEPRECATED shim — the daily backtest CLI moved in WS-C; engine merged in WS-N.

The daily CLI wrapper lives at ``scripts/backtest_sp500_daily.py`` and delegates
to ``mci_gru/evaluation/backtest_engine.py``. This shim re-exports the engine
namespace and keeps ``python tests/backtest_sp500_daily.py`` working for one
release cycle; invoke ``scripts/backtest_sp500_daily.py`` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.backtest_sp500_daily import *  # noqa: F403,E402
from scripts.backtest_sp500_daily import main  # noqa: E402

if __name__ == "__main__":
    main()
