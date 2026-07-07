from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cockpit.git import with_safe_directory

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date
    from pathlib import Path

    CommandRunner = Callable[[list[str]], str]


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
