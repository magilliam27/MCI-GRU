"""Thin CLI wrapper for the full backtest engine.

The engine body lives in ``mci_gru/evaluation/backtest_engine.py`` (WS-C).
``main()`` consumes ``sys.argv`` via argparse, so delegation preserves the
historical command-line behavior of ``tests/backtest_sp500.py`` exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mci_gru.evaluation.backtest_engine import main  # noqa: E402

if __name__ == "__main__":
    main()
