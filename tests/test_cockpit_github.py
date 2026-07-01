from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mci_gru.cockpit.github import (
    GitHubSyncDisabled,
    _run_command,
    close_issue_with_evidence,
    cockpit_branch_name,
    create_issue,
    sync_github,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_cockpit_branch_name_is_dated() -> None:
    assert cockpit_branch_name(date(2026, 6, 20)) == "codex/cockpit-refresh-20260620"


def test_sync_github_requires_explicit_enablement() -> None:
    with pytest.raises(GitHubSyncDisabled, match="requires --github-sync"):
        sync_github(enabled=False)


def test_sync_github_creates_branch_pr_issue_and_comment_with_fake_runner(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command.startswith("git switch -C codex/cockpit-refresh-20260620"):
            return ""
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return "M docs/agents/workstreams.md\nM docs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git add docs/agents/workstreams.md"):
            return ""
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command.startswith("git push -u origin codex/cockpit-refresh-20260620"):
            return ""
        if command.startswith("gh pr list"):
            return ""
        if command.startswith("gh pr create"):
            return "https://github.com/magilliam27/MCI-GRU/pull/99"
        if command.startswith("gh issue list"):
            return ""
        if command.startswith("gh issue create"):
            return "https://github.com/magilliam27/MCI-GRU/issues/100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\nready-for-agent\n"
        if command.startswith("gh issue edit 100"):
            return ""
        if command.startswith("gh issue comment 100"):
            return ""
        raise AssertionError(command)

    result = sync_github(
        enabled=True,
        repo_root=tmp_path,
        run_date=date(2026, 6, 20),
        run_color="yellow",
        decision_queue=["Portfolio-IC: choose promotion path"],
        run_command=fake_run,
    )

    assert result.branch == "codex/cockpit-refresh-20260620"
    assert result.pr_url == "https://github.com/magilliam27/MCI-GRU/pull/99"
    assert result.cockpit_issue_number == 100
    assert any(command[:2] == ["git", "add"] for command in commands)
    assert any(command[:3] == ["gh", "pr", "list"] and "--state" in command for command in commands)
    assert any(command[:3] == ["gh", "issue", "comment"] for command in commands)


def test_default_github_runner_applies_safe_directory_to_git_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        seen_commands.append(args)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("mci_gru.cockpit.github.subprocess.run", fake_run)

    _run_command(tmp_path)(["git", "status", "--short"])
    _run_command(tmp_path)(["gh", "auth", "status"])

    assert seen_commands[0] == ["git", "-c", f"safe.directory={tmp_path}", "status", "--short"]
    assert seen_commands[1] == ["gh", "auth", "status"]


def test_create_issue_applies_only_existing_labels() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args[:3] == ["gh", "label", "list"]:
            return "ready-for-agent\nneeds-info\n"
        if args[:3] == ["gh", "issue", "create"]:
            return "https://github.com/magilliam27/MCI-GRU/issues/101"
        raise AssertionError(" ".join(args))

    url = create_issue(
        title="Clear cockpit follow-up",
        body="Evidence-backed next action.",
        labels=["ready-for-agent", "missing-label"],
        run_command=fake_run,
    )

    assert url.endswith("/101")
    create_command = next(
        command for command in commands if command[:3] == ["gh", "issue", "create"]
    )
    assert "ready-for-agent" in create_command
    assert "missing-label" not in create_command


def test_close_issue_requires_evidence() -> None:
    with pytest.raises(ValueError, match="closure evidence"):
        close_issue_with_evidence(issue_number=8, evidence="", run_command=lambda args: "")


def test_close_issue_comments_with_evidence_before_closing() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        return ""

    close_issue_with_evidence(
        issue_number=8,
        evidence="Merged PR #99 resolves this issue.",
        run_command=fake_run,
    )

    assert commands[0][:3] == ["gh", "issue", "comment"]
    assert commands[1][:3] == ["gh", "issue", "close"]
