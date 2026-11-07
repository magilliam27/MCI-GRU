"""Thin CLI wrapper for the full backtest engine.

The engine body lives in ``mci_gru/evaluation/backtest_engine.py`` (WS-C).
``main()`` consumes ``sys.argv`` via argparse. Since #340 the default strategy buys at
close D+1 and sells at close D+label_t in staggered tranches (the training label);
``--strategy open_to_open`` (or ``--holding_period``/``--rebalance_style``) trades the
paper's open-to-open hold.
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
