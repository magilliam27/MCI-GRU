from __future__ import annotations

import json
import subprocess
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from cockpit.decisions import DECISION_REGISTRY_PATH
from cockpit.github import (
    GitHubSyncDisabled,
    _apply_existing_labels,
    _ensure_issue_digest_comment,
    _ensure_pr,
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


def _existing_cockpit_pr_payload() -> str:
    return json.dumps(
        [
            {
                "number": 99,
                "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                "title": "Cockpit refresh: 2026-06-20",
                "baseRefName": "main",
                "headRefName": "codex/cockpit-refresh-20260620",
                "state": "OPEN",
            }
        ]
    )


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


def test_sync_github_requires_the_prepared_dated_branch(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args == ["gh", "auth", "status"]:
            return "Logged in"
        if args == ["git", "branch", "--show-current"]:
            return "main\n"
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="dated cockpit branch"):
        sync_github(
            enabled=True,
            repo_root=tmp_path,
            run_date=date(2026, 6, 20),
            run_command=fake_run,
        )

    assert commands == [
        ["gh", "auth", "status"],
        ["git", "branch", "--show-current"],
    ]


def test_sync_github_rejects_unrelated_existing_pr_file_before_push(tmp_path: Path) -> None:
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    head_oid = "a" * 40
    fetched = ""
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        nonlocal fetched
        commands.append(args)
        if args == ["gh", "auth", "status"]:
            return "Logged in"
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "fetch", "origin", "main"]:
            fetched = "main"
            return ""
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return f"{head_oid}\trefs/heads/{branch}\n"
        if args == ["git", "fetch", "origin", branch]:
            fetched = "branch"
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return (base_oid if fetched == "main" else head_oid) + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return head_oid + "\n"
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return "M\x00docs/agents/workstreams.md\x00"
        if args[:3] == ["gh", "pr", "list"]:
            return _existing_cockpit_pr_payload()
        if args[:3] == ["gh", "pr", "view"]:
            return json.dumps(
                {
                    "title": "Cockpit refresh: 2026-06-20",
                    "headRefName": branch,
                    "headRefOid": head_oid,
                    "baseRefName": "main",
                    "baseRefOid": base_oid,
                    "headRepositoryOwner": {"login": "magilliam27"},
                    "isCrossRepository": False,
                    "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                    "state": "OPEN",
                }
            )
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/pulls/99/files"]:
            return json.dumps([[{"filename": "user-work.txt"}]])
        if args[:3] == ["git", "status", "--short"]:
            return ""
        if args == ["git", "push", "-u", "origin", branch]:
            return ""
        if args[:3] == ["gh", "issue", "list"]:
            return "100"
        if args[:3] == ["gh", "label", "list"]:
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return "[[]]"
        if args[:3] == ["gh", "issue", "comment"]:
            return ""
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="unexpected PR path: user-work.txt"):
        sync_github(
            enabled=True,
            repo_root=tmp_path,
            run_date=date(2026, 6, 20),
            run_command=fake_run,
            producer_base_oid=base_oid,
            producer_remote_head_oid=head_oid,
        )

    assert ["git", "push", "-u", "origin", branch] not in commands


def test_sync_github_reuses_curator_registry_pr_without_commit_churn(tmp_path: Path) -> None:
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    head_oid = "a" * 40
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args == ["gh", "auth", "status"]:
            return "Logged in"
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return head_oid + "\n"
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return f"M\x00{DECISION_REGISTRY_PATH}\x00"
        if args[:3] == ["gh", "pr", "list"]:
            return _existing_cockpit_pr_payload()
        if args[:3] == ["gh", "pr", "view"]:
            return json.dumps(
                {
                    "title": "Cockpit refresh: 2026-06-20",
                    "headRefName": branch,
                    "headRefOid": head_oid,
                    "baseRefName": "main",
                    "baseRefOid": base_oid,
                    "headRepositoryOwner": {"login": "magilliam27"},
                    "isCrossRepository": False,
                    "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                    "state": "OPEN",
                }
            )
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/pulls/99/files"]:
            return json.dumps([[{"filename": DECISION_REGISTRY_PATH}]])
        if args[:3] == ["git", "status", "--short"]:
            return ""
        if args == ["git", "push", "-u", "origin", branch]:
            return ""
        if args[:3] == ["gh", "issue", "list"]:
            return "100"
        if args[:3] == ["gh", "label", "list"]:
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return "[[]]"
        if args[:3] == ["gh", "issue", "comment"]:
            return ""
        raise AssertionError(" ".join(args))

    result = sync_github(
        enabled=True,
        repo_root=tmp_path,
        run_date=date(2026, 6, 20),
        run_command=fake_run,
        producer_base_oid=base_oid,
        producer_remote_head_oid=head_oid,
    )

    assert result.pr_url == "https://github.com/magilliam27/MCI-GRU/pull/99"
    assert "no cockpit file changes to commit" in result.actions_skipped
    assert not any(command[:2] in (["git", "add"], ["git", "commit"]) for command in commands)


def test_sync_github_creates_branch_pr_issue_and_comment_with_fake_runner(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    issue_labels: set[str] = set()
    pr_labels: set[str] = set()

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command == "git branch --show-current":
            return "codex/cockpit-refresh-20260620\n"
        if command.startswith("git switch -C codex/cockpit-refresh-20260620"):
            return ""
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return "M docs/agents/workstreams.md\nM docs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git add docs/agents/workstreams.md"):
            return ""
        if command == "git diff --cached --name-only":
            return "docs/agents/workstreams.md\ndocs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command.startswith("git push -u origin codex/cockpit-refresh-20260620"):
            return ""
        if command.startswith("gh pr list"):
            return "[]"
        if command.startswith("gh pr create"):
            return "https://github.com/magilliam27/MCI-GRU/pull/99"
        if command.startswith("gh issue list"):
            return ""
        if command.startswith("gh issue create"):
            return "https://github.com/magilliam27/MCI-GRU/issues/100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\nready-for-agent\n"
        if command.startswith("gh issue view 100"):
            return "\n".join(sorted(issue_labels))
        if command.startswith("gh issue edit 100"):
            issue_labels.update(args[-1].split(","))
            return ""
        if command.startswith("gh pr view 99"):
            return "\n".join(sorted(pr_labels))
        if command.startswith("gh pr edit 99"):
            pr_labels.update(args[-1].split(","))
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return "[[]]"
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
    status_command = next(
        command for command in commands if command[:3] == ["git", "status", "--short"]
    )
    add_command = next(command for command in commands if command[:2] == ["git", "add"])
    for command in (status_command, add_command):
        assert "docs/agents/cockpit/auto-decisions.json" in command
        assert DECISION_REGISTRY_PATH in command
        assert "docs/agents/cockpit/override-receipts.json" in command
    assert any(command[:3] == ["gh", "pr", "list"] and "--state" in command for command in commands)
    assert any(command[:3] == ["gh", "issue", "comment"] for command in commands)


def test_sync_github_labels_dated_pr_idempotently_and_reads_back(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    digest_bodies: list[str] = []
    pr_labels = {"codex"}
    digest_comment: dict[str, object] | None = None

    def fake_run(args: list[str]) -> str:
        nonlocal digest_comment
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command == "git branch --show-current":
            return "codex/cockpit-refresh-20260620\n"
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return ""
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            return ""
        if command.startswith("gh pr list"):
            return _existing_cockpit_pr_payload()
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\ncodex\ncodex-automation-v2\n"
        if command.startswith("gh issue view 100"):
            return "cockpit-reviewed\n"
        if command.startswith("gh pr view 99"):
            return "\n".join(sorted(pr_labels))
        if command.startswith("gh pr edit 99"):
            pr_labels.update(args[-1].split(","))
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return json.dumps([[digest_comment]] if digest_comment is not None else [[]])
        if args[:3] == ["gh", "issue", "comment"]:
            digest_bodies.append(args[-1])
            digest_comment = {
                "id": 501,
                "body": args[-1],
                "user": {"login": "magilliam27"},
                "author_association": "OWNER",
            }
            return ""
        if args[:4] == ["gh", "api", "--method", "PATCH"]:
            body = args[-1].removeprefix("body=")
            digest_bodies.append(body)
            assert digest_comment is not None
            digest_comment["body"] = body
            return ""
        raise AssertionError(command)

    results = [
        sync_github(
            enabled=True,
            repo_root=tmp_path,
            run_date=date(2026, 6, 20),
            run_command=fake_run,
        )
        for _ in range(2)
    ]

    pr_edits = [command for command in commands if command[:3] == ["gh", "pr", "edit"]]
    assert pr_edits == [
        [
            "gh",
            "pr",
            "edit",
            "99",
            "--repo",
            "magilliam27/MCI-GRU",
            "--add-label",
            "cockpit-reviewed",
        ]
    ]
    assert pr_labels == {"cockpit-reviewed", "codex"}
    assert not any(command[:3] == ["gh", "label", "create"] for command in commands)
    assert sum(command[:3] == ["gh", "pr", "view"] for command in commands) == 3
    assert any(
        "applied labels to dated cockpit PR #99" in item for item in results[0].actions_taken
    )
    assert any(
        "labels already present on dated cockpit PR #99" in item
        for item in results[1].actions_skipped
    )
    first_receipt = results[0].dated_pr_labels
    assert first_receipt is not None
    assert first_receipt.applied == ("cockpit-reviewed",)
    assert first_receipt.already_present == ("codex",)
    assert first_receipt.skipped_missing == ("codex-automation",)
    assert first_receipt.verified_present == ("cockpit-reviewed", "codex")
    assert "Dated PR label receipt:" in digest_bodies[0]
    assert "- applied: cockpit-reviewed" in digest_bodies[0]
    assert "- already-present: codex" in digest_bodies[0]
    assert "- skipped-missing: codex-automation" in digest_bodies[0]
    assert "- verified-after-readback: cockpit-reviewed, codex" in digest_bodies[0]
    second_receipt = results[1].dated_pr_labels
    assert second_receipt is not None
    assert second_receipt.applied == ()
    assert second_receipt.already_present == ("cockpit-reviewed", "codex")
    assert digest_comment is not None
    persistent_body = str(digest_comment["body"])
    assert "- applied: cockpit-reviewed" in persistent_body
    assert "- already-present: codex" in persistent_body
    assert "- applied: none" not in persistent_body
    assert not any(command[:4] == ["gh", "api", "--method", "PATCH"] for command in commands)


def test_sync_github_same_date_skips_identical_issue_digest_comment(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    comments: list[dict[str, object]] = []
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    head_oid = "a" * 40
    fetched = ""

    def fake_run(args: list[str]) -> str:
        nonlocal fetched
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command == "git branch --show-current":
            return branch + "\n"
        if args == ["git", "fetch", "origin", "main"]:
            fetched = "main"
            return ""
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return f"{head_oid}\trefs/heads/{branch}\n"
        if args == ["git", "fetch", "origin", branch]:
            fetched = "branch"
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return (base_oid if fetched == "main" else head_oid) + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return head_oid + "\n"
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return ""
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return ""
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            return ""
        if command.startswith("gh pr list"):
            return _existing_cockpit_pr_payload()
        if args[:3] == ["gh", "pr", "view"]:
            return json.dumps(
                {
                    "title": "Cockpit refresh: 2026-06-20",
                    "headRefName": branch,
                    "headRefOid": head_oid,
                    "baseRefName": "main",
                    "baseRefOid": base_oid,
                    "headRepositoryOwner": {"login": "magilliam27"},
                    "isCrossRepository": False,
                    "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                    "state": "OPEN",
                }
            )
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return json.dumps([comments])
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/pulls/99/files"]:
            return "[[]]"
        if args[:3] == ["gh", "issue", "comment"]:
            comments.append(
                {
                    "id": 501,
                    "body": args[-1],
                    "user": {"login": "github-actions[bot]"},
                    "author_association": "NONE",
                }
            )
            return ""
        raise AssertionError(command)

    results = []
    for _ in range(2):
        results.append(
            sync_github(
                enabled=True,
                repo_root=tmp_path,
                run_date=date(2026, 6, 20),
                run_color="yellow",
                decision_queue=["Portfolio-IC: choose promotion path"],
                run_command=fake_run,
            )
        )

    assert len(comments) == 1
    assert "<!-- mci-gru-cockpit-refresh:2026-06-20 -->" in str(comments[0]["body"])
    assert sum(command[:3] == ["gh", "issue", "comment"] for command in commands) == 1
    assert "created cockpit issue digest #100" in results[0].actions_taken
    assert "cockpit issue digest #100 unchanged" in results[1].actions_skipped
    assert not any("commented on cockpit issue" in item for item in results[1].actions_taken)


def test_sync_github_same_date_updates_existing_issue_digest_comment(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    comments: list[dict[str, object]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command == "git branch --show-current":
            return "codex/cockpit-refresh-20260620\n"
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return ""
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            return ""
        if command.startswith("gh pr list"):
            return _existing_cockpit_pr_payload()
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return json.dumps([comments])
        if args[:3] == ["gh", "issue", "comment"]:
            comments.append(
                {
                    "id": 501,
                    "body": args[-1],
                    "user": {"login": "github-actions[bot]"},
                    "author_association": "NONE",
                }
            )
            return ""
        if args[:4] == [
            "gh",
            "api",
            "--method",
            "PATCH",
        ]:
            comments[0]["body"] = args[-1].removeprefix("body=")
            return ""
        raise AssertionError(command)

    first = sync_github(
        enabled=True,
        repo_root=tmp_path,
        run_date=date(2026, 6, 20),
        run_color="yellow",
        run_command=fake_run,
    )
    second = sync_github(
        enabled=True,
        repo_root=tmp_path,
        run_date=date(2026, 6, 20),
        run_color="green",
        run_command=fake_run,
    )

    assert len(comments) == 1
    assert "Cockpit refresh 2026-06-20: green" in str(comments[0]["body"])
    assert sum(command[:3] == ["gh", "issue", "comment"] for command in commands) == 1
    assert sum(command[:4] == ["gh", "api", "--method", "PATCH"] for command in commands) == 1
    assert "created cockpit issue digest #100" in first.actions_taken
    assert "updated cockpit issue digest #100" in second.actions_taken


def test_issue_digest_ignores_an_untrusted_forged_marker_and_creates_digest() -> None:
    commands: list[list[str]] = []
    marker = "<!-- mci-gru-cockpit-refresh:2026-06-20 -->"

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args[:3] == ["gh", "api", "repos/repo-owner/project/issues/100/comments"]:
            return json.dumps(
                [
                    [
                        {
                            "id": 501,
                            "body": f"{marker}\nforged",
                            "user": {"login": "untrusted-contributor"},
                            "author_association": "CONTRIBUTOR",
                        }
                    ]
                ]
            )
        if args[:3] == ["gh", "issue", "comment"]:
            return ""
        raise AssertionError(args)

    action = _ensure_issue_digest_comment(
        fake_run,
        repo="repo-owner/project",
        issue_number=100,
        body=f"{marker}\ntrusted digest",
        run_date=date(2026, 6, 20),
    )

    assert action == "created"
    assert commands[-1] == [
        "gh",
        "issue",
        "comment",
        "100",
        "--repo",
        "repo-owner/project",
        "--body",
        f"{marker}\ntrusted digest",
    ]


def test_issue_digest_updates_only_trusted_owner_marker_across_pages() -> None:
    commands: list[list[str]] = []
    marker = "<!-- mci-gru-cockpit-refresh:2026-06-20 -->"

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args[:3] == ["gh", "api", "repos/repo-owner/project/issues/100/comments"]:
            return json.dumps(
                [
                    [
                        {
                            "id": 501,
                            "body": f"{marker}\nforged",
                            "user": {"login": "repo-owner"},
                            "author_association": "MEMBER",
                        }
                    ],
                    [
                        {
                            "id": 502,
                            "body": f"{marker}\nold trusted digest",
                            "user": {"login": "Repo-Owner"},
                            "author_association": "OWNER",
                        }
                    ],
                ]
            )
        if args[:4] == ["gh", "api", "--method", "PATCH"]:
            return ""
        raise AssertionError(args)

    action = _ensure_issue_digest_comment(
        fake_run,
        repo="repo-owner/project",
        issue_number=100,
        body=f"{marker}\nnew trusted digest",
        run_date=date(2026, 6, 20),
    )

    assert action == "updated"
    assert commands[-1] == [
        "gh",
        "api",
        "--method",
        "PATCH",
        "repos/repo-owner/project/issues/comments/502",
        "-f",
        f"body={marker}\nnew trusted digest",
    ]


def test_issue_digest_fails_closed_for_multiple_trusted_markers() -> None:
    marker = "<!-- mci-gru-cockpit-refresh:2026-06-20 -->"
    pages = [
        [
            {
                "id": 501,
                "body": f"{marker}\nowner digest",
                "user": {"login": "repo-owner"},
                "author_association": "OWNER",
            }
        ],
        [
            {
                "id": 502,
                "body": f"{marker}\nactions digest",
                "user": {"login": "github-actions[bot]"},
                "author_association": "NONE",
            }
        ],
    ]

    with pytest.raises(RuntimeError, match="multiple dated digest comments"):
        _ensure_issue_digest_comment(
            lambda args: json.dumps(pages),
            repo="repo-owner/project",
            issue_number=100,
            body=f"{marker}\nnew trusted digest",
            run_date=date(2026, 6, 20),
        )


@pytest.mark.parametrize(
    ("title", "base_ref", "state"),
    [
        ("Wrong title", "main", "OPEN"),
        (" Cockpit refresh: 2026-06-20", "main", "OPEN"),
        ("Cockpit refresh: 2026-06-20 ", "main", "OPEN"),
        ("Cockpit refresh: 2026-06-20", "release", "OPEN"),
        ("Cockpit refresh: 2026-06-20", " main", "OPEN"),
        ("Cockpit refresh: 2026-06-20", "main ", "OPEN"),
        ("Cockpit refresh: 2026-06-20", "main", "CLOSED"),
    ],
)
def test_ensure_pr_rejects_existing_pr_with_wrong_contract(
    title: str,
    base_ref: str,
    state: str,
) -> None:
    payload = json.dumps(
        [
            {
                "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                "title": title,
                "baseRefName": base_ref,
                "headRefName": "codex/cockpit-refresh-20260620",
                "state": state,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="existing dated cockpit PR"):
        _ensure_pr(
            lambda args: payload,
            "magilliam27/MCI-GRU",
            "codex/cockpit-refresh-20260620",
            date(2026, 6, 20),
        )


@pytest.mark.parametrize(
    "head_ref",
    [" codex/cockpit-refresh-20260620", "codex/cockpit-refresh-20260620 "],
)
def test_ensure_pr_rejects_existing_pr_with_padded_head_ref(head_ref: str) -> None:
    payload = json.dumps(
        [
            {
                "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                "title": "Cockpit refresh: 2026-06-20",
                "baseRefName": "main",
                "headRefName": head_ref,
                "state": "OPEN",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="existing dated cockpit PR"):
        _ensure_pr(
            lambda args: payload,
            "magilliam27/MCI-GRU",
            "codex/cockpit-refresh-20260620",
            date(2026, 6, 20),
        )


def test_apply_existing_labels_is_idempotent_for_already_applied_label() -> None:
    commands: list[list[str]] = []
    issue_labels: set[str] = set()

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args[:3] == ["gh", "label", "list"]:
            return "cockpit-reviewed\n"
        if args[:3] == ["gh", "issue", "view"]:
            return "\n".join(sorted(issue_labels))
        if args[:3] == ["gh", "issue", "edit"]:
            issue_labels.add("cockpit-reviewed")
            return ""
        raise AssertionError(" ".join(args))

    for _ in range(2):
        _apply_existing_labels(
            fake_run,
            "magilliam27/MCI-GRU",
            100,
            ["cockpit-reviewed"],
            [],
            [],
        )

    assert sum(command[:3] == ["gh", "issue", "edit"] for command in commands) == 1


def test_apply_existing_pr_labels_fails_closed_when_readback_is_missing() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args[:3] == ["gh", "label", "list"]:
            return "cockpit-reviewed\n"
        if args[:3] == ["gh", "pr", "view"]:
            return ""
        if args[:3] == ["gh", "pr", "edit"]:
            return ""
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="Label readback failed for dated cockpit PR #99"):
        _apply_existing_labels(
            fake_run,
            "magilliam27/MCI-GRU",
            99,
            ["cockpit-reviewed"],
            [],
            [],
            resource="pr",
            target_name="dated cockpit PR",
        )

    assert sum(command[:3] == ["gh", "pr", "view"] for command in commands) == 2


def test_sync_github_refuses_an_unrelated_staged_path(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command == "git branch --show-current":
            return "codex/cockpit-refresh-20260620\n"
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return "M docs/agents/workstreams.md\n"
        if command.startswith("git add docs/agents/workstreams.md"):
            return ""
        if command == "git diff --cached --name-only":
            return "docs/agents/workstreams.md\nuser-work.txt\n"
        if command.startswith("git commit"):
            pytest.fail("sync must validate the complete staged index before committing")
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="unexpected staged path: user-work.txt"):
        sync_github(
            enabled=True,
            repo_root=tmp_path,
            run_date=date(2026, 6, 20),
            run_command=fake_run,
        )

    assert ["git", "diff", "--cached", "--name-only"] in commands


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
