from __future__ import annotations

import json
import subprocess
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from cockpit.github import (
    GitHubSyncDisabled,
    _run_command,
    close_issue_with_evidence,
    cockpit_branch_name,
    collect_github_evidence,
    create_issue,
    sync_github,
)
from cockpit.models import GitHubEvidence, IssueEvidence, PullRequestEvidence

if TYPE_CHECKING:
    from pathlib import Path


PR_COMMAND = [
    "gh",
    "pr",
    "list",
    "--state",
    "all",
    "--limit",
    "1000",
    "--json",
    "number,headRefName,url,isDraft,state,mergedAt,updatedAt",
]
ISSUE_COMMAND = [
    "gh",
    "issue",
    "list",
    "--state",
    "all",
    "--limit",
    "1000",
    "--json",
    "number,title,url,state,labels,updatedAt",
]


def test_collect_github_evidence_uses_exact_read_only_commands_and_normalizes() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args == PR_COMMAND:
            return json.dumps(
                [
                    {
                        "number": 12,
                        "headRefName": "origin/CODEX/Feature ",
                        "url": " https://github.example/pull/12 ",
                        "isDraft": False,
                        "state": "OPEN",
                        "mergedAt": None,
                        "updatedAt": "2026-07-12T03:04:05Z",
                    }
                ]
            )
        if args == ISSUE_COMMAND:
            return json.dumps(
                [
                    {
                        "number": 34,
                        "title": "  Feature Work  ",
                        "url": " https://github.example/issues/34 ",
                        "state": "OPEN",
                        "labels": [{"name": "Needs-Info"}, {"name": " blocked "}],
                        "updatedAt": "2026-07-11T01:02:03Z",
                    }
                ]
            )
        raise AssertionError(args)

    evidence = collect_github_evidence(run_command=fake_run)

    assert commands == [PR_COMMAND, ISSUE_COMMAND]
    assert evidence == GitHubEvidence(
        pull_requests=(
            PullRequestEvidence(
                number=12,
                head_ref="CODEX/Feature",
                url="https://github.example/pull/12",
                state="open",
                is_draft=False,
                merged_at=None,
                updated_at=date(2026, 7, 12),
            ),
        ),
        issues=(
            IssueEvidence(
                number=34,
                title="Feature Work",
                url="https://github.example/issues/34",
                state="open",
                labels=("blocked", "needs-info"),
                updated_at=date(2026, 7, 11),
            ),
        ),
    )


def test_collect_github_evidence_requests_and_keeps_more_than_default_page() -> None:
    commands: list[list[str]] = []
    pull_requests = [
        {
            "number": number,
            "headRefName": f"codex/feature-{number}",
            "url": f"https://github.example/pull/{number}",
            "isDraft": False,
            "state": "OPEN",
            "mergedAt": None,
            "updatedAt": "2026-07-12T03:04:05Z",
        }
        for number in range(1, 36)
    ]
    issues = [
        {
            "number": number,
            "title": f"Issue {number}",
            "url": f"https://github.example/issues/{number}",
            "state": "OPEN",
            "labels": [],
            "updatedAt": "2026-07-11T01:02:03+00:00",
        }
        for number in range(1, 36)
    ]

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args == PR_COMMAND:
            return json.dumps(pull_requests)
        if args == ISSUE_COMMAND:
            return json.dumps(issues)
        raise AssertionError(args)

    evidence = collect_github_evidence(run_command=fake_run)

    assert evidence is not None
    assert commands == [PR_COMMAND, ISSUE_COMMAND]
    assert len(evidence.pull_requests) == 35
    assert len(evidence.issues) == 35


def test_collect_github_evidence_preserves_head_branch_case() -> None:
    outputs = iter(
        [
            json.dumps(
                [
                    {
                        "number": 1,
                        "headRefName": " origin/CODEX/Feature ",
                        "url": "https://github.example/pull/1",
                        "isDraft": False,
                        "state": "OPEN",
                        "mergedAt": None,
                        "updatedAt": "2026-07-12T03:04:05Z",
                    }
                ]
            ),
            "[]",
        ]
    )

    evidence = collect_github_evidence(run_command=lambda args: next(outputs))

    assert evidence is not None
    assert evidence.pull_requests[0].head_ref == "CODEX/Feature"


def test_collect_github_evidence_distinguishes_confirmed_empty_from_unavailable() -> None:
    assert collect_github_evidence(run_command=lambda args: "[]") == GitHubEvidence()

    def failing_run(args: list[str]) -> str:
        raise subprocess.CalledProcessError(1, args)

    assert collect_github_evidence(run_command=failing_run) is None


def test_collect_github_evidence_returns_none_on_oserror() -> None:
    def unavailable_run(args: list[str]) -> str:
        raise OSError("gh executable unavailable")

    assert collect_github_evidence(run_command=unavailable_run) is None


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["gh"], 30),
        RuntimeError("collector runtime failure"),
    ],
)
def test_collect_github_evidence_returns_none_on_ordinary_exception(
    error: Exception,
) -> None:
    def failing_run(args: list[str]) -> str:
        raise error

    assert collect_github_evidence(run_command=failing_run) is None


def test_collect_github_evidence_rejects_malformed_issue_schema() -> None:
    calls = 0

    def malformed_issue_run(args: list[str]) -> str:
        nonlocal calls
        calls += 1
        return "[]" if calls == 1 else '[{"number": 1}]'

    assert collect_github_evidence(run_command=malformed_issue_run) is None


@pytest.mark.parametrize(
    "bad_output",
    [
        "{not-json",
        "{}",
        '[{"number": true}]',
        '[{"number": 1, "headRefName": "x"}]',
    ],
)
def test_collect_github_evidence_rejects_malformed_json_or_schema(bad_output: str) -> None:
    assert collect_github_evidence(run_command=lambda args: bad_output) is None


@pytest.mark.parametrize("state", ["DRAFT", "UNKNOWN"])
def test_collect_github_evidence_rejects_unknown_pr_state(state: str) -> None:
    pull_request = json.dumps(
        [
            {
                "number": 1,
                "headRefName": "codex/feature",
                "url": "https://github.example/pull/1",
                "isDraft": False,
                "state": state,
                "mergedAt": None,
                "updatedAt": "2026-07-12T03:04:05Z",
            }
        ]
    )
    outputs = iter([pull_request, "[]"])

    assert collect_github_evidence(run_command=lambda args: next(outputs)) is None


def test_collect_github_evidence_rejects_unknown_issue_state() -> None:
    outputs = iter(
        [
            "[]",
            json.dumps(
                [
                    {
                        "number": 1,
                        "title": "Feature",
                        "url": "https://github.example/issues/1",
                        "state": "MERGED",
                        "labels": [],
                        "updatedAt": "2026-07-12T03:04:05Z",
                    }
                ]
            ),
        ]
    )

    assert collect_github_evidence(run_command=lambda args: next(outputs)) is None


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-12",
        "2026-07-12garbage",
        "2026-07-12T03:04:05Zgarbage",
        "not-a-timestamp",
    ],
)
def test_collect_github_evidence_rejects_malformed_full_timestamp(
    timestamp: str,
) -> None:
    pull_request = json.dumps(
        [
            {
                "number": 1,
                "headRefName": "codex/feature",
                "url": "https://github.example/pull/1",
                "isDraft": False,
                "state": "OPEN",
                "mergedAt": None,
                "updatedAt": timestamp,
            }
        ]
    )
    outputs = iter([pull_request, "[]"])

    assert collect_github_evidence(run_command=lambda args: next(outputs)) is None


def test_collect_github_evidence_accepts_z_and_offset_timestamps() -> None:
    outputs = iter(
        [
            json.dumps(
                [
                    {
                        "number": 1,
                        "headRefName": "codex/feature",
                        "url": "https://github.example/pull/1",
                        "isDraft": False,
                        "state": "MERGED",
                        "mergedAt": "2026-07-10T23:59:59-04:00",
                        "updatedAt": "2026-07-12T03:04:05Z",
                    }
                ]
            ),
            json.dumps(
                [
                    {
                        "number": 2,
                        "title": "Feature",
                        "url": "https://github.example/issues/2",
                        "state": "CLOSED",
                        "labels": [],
                        "updatedAt": "2026-07-11T01:02:03+05:30",
                    }
                ]
            ),
        ]
    )

    evidence = collect_github_evidence(run_command=lambda args: next(outputs))

    assert evidence is not None
    assert evidence.pull_requests[0].merged_at == date(2026, 7, 10)
    assert evidence.pull_requests[0].updated_at == date(2026, 7, 12)
    assert evidence.issues[0].updated_at == date(2026, 7, 11)


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

    monkeypatch.setattr("cockpit.github.subprocess.run", fake_run)

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
