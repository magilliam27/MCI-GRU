from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cockpit.git import with_safe_directory
from cockpit.models import GitHubEvidence, IssueEvidence, PullRequestEvidence

if TYPE_CHECKING:
    from collections.abc import Callable

    CommandRunner = Callable[[list[str]], str]


_PR_LIST_COMMAND = [
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
_ISSUE_LIST_COMMAND = [
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


class GitHubSyncDisabled(RuntimeError):
    """Raised when a caller tries to sync GitHub without explicit enablement."""


@dataclass(frozen=True)
class GitHubSyncResult:
    branch: str
    pr_url: str
    cockpit_issue_number: int
    cockpit_issue_url: str
    actions_taken: list[str] = field(default_factory=list)
    actions_skipped: list[str] = field(default_factory=list)


def collect_github_evidence(
    *,
    repo_root: Path | None = None,
    run_command: CommandRunner | None = None,
) -> GitHubEvidence | None:
    """Collect normalized read-only PR and issue evidence; return None on any gap."""
    if run_command is None:
        runner = _run_command(repo_root or Path.cwd())
    else:
        runner = run_command
    try:
        pull_requests = _parse_pull_requests(runner(_PR_LIST_COMMAND))
        issues = _parse_issues(runner(_ISSUE_LIST_COMMAND))
    except Exception:
        return None
    return GitHubEvidence(pull_requests=pull_requests, issues=issues)


def cockpit_branch_name(run_date: date) -> str:
    return f"codex/cockpit-refresh-{run_date:%Y%m%d}"


def sync_github(
    *,
    enabled: bool,
    repo_root: Path | None = None,
    run_date: date | None = None,
    run_color: str = "",
    decision_queue: list[str] | None = None,
    run_command: CommandRunner | None = None,
    repo: str = "magilliam27/MCI-GRU",
) -> GitHubSyncResult:
    if not enabled:
        raise GitHubSyncDisabled("GitHub cockpit sync requires --github-sync.")
    if repo_root is None or run_date is None:
        raise ValueError("GitHub cockpit sync requires repo_root and run_date.")
    decisions = decision_queue or []
    runner = run_command or _run_command(repo_root)
    branch = cockpit_branch_name(run_date)
    runner(["gh", "auth", "status"])
    runner(["git", "switch", "-C", branch])
    paths = [
        "docs/agents/workstreams.md",
        f"docs/agents/cockpit/{run_date.isoformat()}.md",
        "docs/agents/cockpit/RUNBOOK.md",
    ]
    actions_taken: list[str] = []
    actions_skipped: list[str] = []
    if runner(["git", "status", "--short", "--", *paths]).strip():
        runner(["git", "add", *paths])
        runner(["git", "commit", "-m", f"Refresh cockpit status for {run_date.isoformat()}"])
        actions_taken.append("committed cockpit files")
    else:
        actions_skipped.append("no cockpit file changes to commit")
    runner(["git", "push", "-u", "origin", branch])
    pr_url = _ensure_pr(runner, repo, branch, run_date)
    issue_number, issue_url = _ensure_cockpit_issue(runner, repo)
    _apply_existing_labels(
        runner,
        repo,
        issue_number,
        ["cockpit-reviewed"],
        actions_taken,
        actions_skipped,
    )
    runner(
        [
            "gh",
            "issue",
            "comment",
            str(issue_number),
            "--repo",
            repo,
            "--body",
            _issue_comment(run_date, run_color, pr_url, decisions),
        ]
    )
    actions_taken.append(f"commented on cockpit issue #{issue_number}")
    return GitHubSyncResult(
        branch=branch,
        pr_url=pr_url,
        cockpit_issue_number=issue_number,
        cockpit_issue_url=issue_url,
        actions_taken=actions_taken,
        actions_skipped=actions_skipped,
    )


def create_issue(
    *,
    title: str,
    body: str,
    labels: list[str],
    run_command: CommandRunner,
    repo: str = "magilliam27/MCI-GRU",
) -> str:
    existing_labels = set(
        _split_lines(
            run_command(
                ["gh", "label", "list", "--repo", repo, "--json", "name", "--jq", ".[].name"]
            )
        )
    )
    labels_to_apply = [label for label in labels if label in existing_labels]
    command = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
    if labels_to_apply:
        command.extend(["--label", ",".join(labels_to_apply)])
    return run_command(command).strip()


def close_issue_with_evidence(
    *,
    issue_number: int,
    evidence: str,
    run_command: CommandRunner,
    repo: str = "magilliam27/MCI-GRU",
) -> None:
    if not evidence.strip():
        raise ValueError("Issue closure requires closure evidence.")
    run_command(["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", evidence])
    run_command(["gh", "issue", "close", str(issue_number), "--repo", repo])


def _ensure_pr(runner: CommandRunner, repo: str, branch: str, run_date: date) -> str:
    existing = runner(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--head",
            branch,
            "--state",
            "all",
            "--json",
            "url",
            "--jq",
            ".[0].url",
        ]
    ).strip()
    if existing:
        return existing
    return runner(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            f"Cockpit refresh: {run_date.isoformat()}",
            "--body",
            f"Automated cockpit refresh for {run_date.isoformat()}.",
        ]
    ).strip()


def _ensure_cockpit_issue(runner: CommandRunner, repo: str) -> tuple[int, str]:
    existing = runner(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--search",
            "MCI-GRU Cockpit in:title",
            "--json",
            "number,url",
            "--jq",
            ".[0].number",
        ]
    ).strip()
    if existing:
        number = int(existing)
        return number, f"https://github.com/{repo}/issues/{number}"
    url = runner(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            "MCI-GRU Cockpit",
            "--body",
            "Daily cockpit review surface for MCI-GRU.",
        ]
    ).strip()
    return _parse_issue_url_number(url), url


def _apply_existing_labels(
    runner: CommandRunner,
    repo: str,
    issue_number: int,
    labels: list[str],
    actions_taken: list[str],
    actions_skipped: list[str],
) -> None:
    existing = set(
        _split_lines(
            runner(["gh", "label", "list", "--repo", repo, "--json", "name", "--jq", ".[].name"])
        )
    )
    labels_to_apply = [label for label in labels if label in existing]
    missing = [label for label in labels if label not in existing]
    if labels_to_apply:
        runner(
            [
                "gh",
                "issue",
                "edit",
                str(issue_number),
                "--repo",
                repo,
                "--add-label",
                ",".join(labels_to_apply),
            ]
        )
        actions_taken.append(
            f"applied labels to cockpit issue #{issue_number}: {', '.join(labels_to_apply)}"
        )
    if missing:
        actions_skipped.append(f"missing labels: {', '.join(missing)}")


def _issue_comment(run_date: date, run_color: str, pr_url: str, decision_queue: list[str]) -> str:
    lines = [
        f"Cockpit refresh {run_date.isoformat()}: {run_color}",
        "",
        f"PR: {pr_url}",
    ]
    if decision_queue:
        lines.extend(["", "Decision queue:"])
        lines.extend(f"- {decision}" for decision in decision_queue)
    return "\n".join(lines)


def _parse_issue_url_number(url: str) -> int:
    return int(url.rstrip("/").split("/")[-1])


def _split_lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _parse_pull_requests(output: str) -> tuple[PullRequestEvidence, ...]:
    values = _json_list(output)
    parsed: list[PullRequestEvidence] = []
    required = {
        "number",
        "headRefName",
        "url",
        "isDraft",
        "state",
        "mergedAt",
        "updatedAt",
    }
    for value in values:
        item = _json_object(value)
        if not required.issubset(item):
            raise ValueError("malformed pull request evidence")
        number = _number(item["number"])
        is_draft = item["isDraft"]
        if not isinstance(is_draft, bool):
            raise ValueError("malformed pull request draft state")
        state = _state(item["state"], {"open", "closed", "merged"})
        parsed.append(
            PullRequestEvidence(
                number=number,
                head_ref=_normalize_branch(_text(item["headRefName"])),
                url=_text(item["url"]),
                state=state,
                is_draft=is_draft,
                merged_at=_optional_date(item["mergedAt"]),
                updated_at=_required_date(item["updatedAt"]),
            )
        )
    return tuple(sorted(parsed, key=lambda item: (item.number, item.head_ref, item.url)))


def _parse_issues(output: str) -> tuple[IssueEvidence, ...]:
    values = _json_list(output)
    parsed: list[IssueEvidence] = []
    required = {"number", "title", "url", "state", "labels", "updatedAt"}
    for value in values:
        item = _json_object(value)
        if not required.issubset(item):
            raise ValueError("malformed issue evidence")
        raw_labels = item["labels"]
        if not isinstance(raw_labels, list):
            raise ValueError("malformed issue labels")
        state = _state(item["state"], {"open", "closed"})
        labels = tuple(
            sorted({_text(_json_object(label).get("name")).lower() for label in raw_labels})
        )
        parsed.append(
            IssueEvidence(
                number=_number(item["number"]),
                title=_text(item["title"]),
                url=_text(item["url"]),
                state=state,
                labels=labels,
                updated_at=_required_date(item["updatedAt"]),
            )
        )
    return tuple(sorted(parsed, key=lambda item: (item.number, item.title, item.url)))


def _json_list(output: str) -> list[object]:
    value = json.loads(output)
    if not isinstance(value, list):
        raise ValueError("expected JSON list")
    return value


def _json_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected JSON object")
    return value


def _number(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("expected positive integer")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-empty text")
    return value.strip()


def _state(value: object, allowed: set[str]) -> str:
    state = _text(value).lower()
    if state not in allowed:
        raise ValueError("unexpected GitHub state")
    return state


def _required_date(value: object) -> date:
    raw_timestamp = _text(value)
    normalized_timestamp = (
        f"{raw_timestamp[:-1]}+00:00" if raw_timestamp.endswith("Z") else raw_timestamp
    )
    timestamp = datetime.fromisoformat(normalized_timestamp)
    if timestamp.tzinfo is None:
        raise ValueError("expected timezone-aware ISO timestamp")
    return timestamp.date()


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    return _required_date(value)


def _normalize_branch(value: str) -> str:
    return (
        value.removeprefix("refs/remotes/origin/")
        .removeprefix("remotes/origin/")
        .removeprefix("origin/")
        .strip()
    )


def _run_command(repo_root: Path) -> CommandRunner:
    def run(args: list[str]) -> str:
        completed = subprocess.run(
            with_safe_directory(args, repo_root),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    return run
