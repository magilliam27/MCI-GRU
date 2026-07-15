from __future__ import annotations

import importlib
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace


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
    assert "--auto-decisions" in completed.stdout
    assert "--no-auto-decisions" in completed.stdout
    assert "disabled by default" in completed.stdout


def test_cockpit_docs_are_linked_from_index() -> None:
    runbook = Path("docs/agents/cockpit/RUNBOOK.md").read_text(encoding="utf-8")
    index = Path("docs/index.md").read_text(encoding="utf-8")

    assert "scripts/refresh_cockpit.py --date 2026-06-20" in runbook
    assert "GitHub sync is disabled by default" in runbook
    assert "python scripts/refresh_cockpit.py --date 2026-06-20 --github-sync" in runbook
    assert "gh auth status" in runbook
    assert "codex/cockpit-refresh-YYYYMMDD" in runbook
    assert "close_issue_with_evidence" in runbook
    assert "agents/workstreams.md" in index
    assert "agents/cockpit/RUNBOOK.md" in index


def test_refresh_cockpit_github_sync_uses_live_runner(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("scripts.refresh_cockpit")
    calls: list[tuple[Path, date, bool]] = []

    def fake_live_runner(
        repo_root: Path,
        run_date: date,
        *,
        auto_decisions_enabled: bool = True,
    ):
        calls.append((repo_root, run_date, auto_decisions_enabled))
        return SimpleNamespace(
            register_path=repo_root / "docs" / "agents" / "workstreams.md",
            packet_path=repo_root / "docs" / "agents" / "cockpit" / "2026-06-20.md",
            color=SimpleNamespace(value="green"),
            github=SimpleNamespace(pr_url="https://github.com/magilliam27/MCI-GRU/pull/99"),
        )

    monkeypatch.setattr(module, "run_github_cockpit_refresh", fake_live_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_cockpit.py",
            "--date",
            "2026-06-20",
            "--repo-root",
            str(tmp_path),
            "--github-sync",
        ],
    )

    assert module.main() == 0
    assert calls == [(tmp_path.resolve(), date(2026, 6, 20), True)]


def test_refresh_cockpit_enables_auto_decisions_for_local_runner_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("scripts.refresh_cockpit")
    calls: list[tuple[Path, date, bool]] = []

    def fake_local_runner(
        repo_root: Path,
        run_date: date,
        *,
        auto_decisions_enabled: bool = False,
    ):
        calls.append((repo_root, run_date, auto_decisions_enabled))
        return SimpleNamespace(
            register_path=repo_root / "docs" / "agents" / "workstreams.md",
            packet_path=repo_root / "docs" / "agents" / "cockpit" / "2026-06-20.md",
            color=SimpleNamespace(value="green"),
            github=None,
        )

    monkeypatch.setattr(module, "run_local_cockpit_refresh", fake_local_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_cockpit.py",
            "--date",
            "2026-06-20",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert module.main() == 0
    assert calls == [(tmp_path.resolve(), date(2026, 6, 20), True)]


def test_refresh_cockpit_no_auto_decisions_uses_legacy_local_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("scripts.refresh_cockpit")
    calls: list[bool] = []

    def fake_local_runner(
        repo_root: Path,
        run_date: date,
        *,
        auto_decisions_enabled: bool = True,
    ):
        calls.append(auto_decisions_enabled)
        return SimpleNamespace(
            register_path=repo_root / "docs" / "agents" / "workstreams.md",
            packet_path=repo_root / "docs" / "agents" / "cockpit" / "2026-06-20.md",
            color=SimpleNamespace(value="green"),
            github=None,
        )

    monkeypatch.setattr(module, "run_local_cockpit_refresh", fake_local_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_cockpit.py",
            "--date",
            "2026-06-20",
            "--repo-root",
            str(tmp_path),
            "--no-auto-decisions",
        ],
    )

    assert module.main() == 0
    assert calls == [False]
