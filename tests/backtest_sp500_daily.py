"""DEPRECATED shim — the daily backtest CLI moved in WS-C.

The self-contained daily engine now lives at ``scripts/backtest_sp500_daily.py``.
This shim re-exports its namespace and keeps
``python tests/backtest_sp500_daily.py`` working for one release cycle; invoke
``scripts/backtest_sp500_daily.py`` instead.
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
