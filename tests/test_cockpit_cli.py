from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_refresh_cockpit_help_lists_local_only_default() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/refresh_cockpit.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--date" in completed.stdout
    assert "--repo-root" in completed.stdout
    assert "--github-sync" in completed.stdout
    assert "disabled by default" in completed.stdout


def test_cockpit_docs_are_linked_from_index() -> None:
    runbook = Path("docs/agents/cockpit/RUNBOOK.md").read_text(encoding="utf-8")
    index = Path("docs/index.md").read_text(encoding="utf-8")

    assert "scripts/refresh_cockpit.py --date 2026-06-20" in runbook
    assert "GitHub sync is disabled by default" in runbook
    assert "agents/workstreams.md" in index
    assert "agents/cockpit/RUNBOOK.md" in index
