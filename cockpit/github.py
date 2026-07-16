from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cockpit.decisions import DECISION_REGISTRY_PATH
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
_DATED_COCKPIT_PR_LABELS = ("cockpit-reviewed", "codex", "codex-automation")


class GitHubSyncDisabled(RuntimeError):
    """Raised when a caller tries to sync GitHub without explicit enablement."""


@dataclass(frozen=True)
class LabelSyncReceipt:
    applied: tuple[str, ...] = ()
    already_present: tuple[str, ...] = ()
    skipped_missing: tuple[str, ...] = ()
    verified_present: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitHubSyncResult:
    branch: str
    pr_url: str
    cockpit_issue_number: int
    cockpit_issue_url: str
    actions_taken: list[str] = field(default_factory=list)
    actions_skipped: list[str] = field(default_factory=list)
    dated_pr_labels: LabelSyncReceipt | None = None


@dataclass(frozen=True)
class PullRequestComment:
    """Stable GitHub comment evidence consumed by the cockpit curator."""

    comment_id: str
    url: str
    author_login: str
    author_association: str
    body: str
    created_at: datetime


@dataclass(frozen=True)
class CommentAuthorization:
    """Fail-closed owner authorization with retained GitHub evidence."""

    comment_id: str
    comment_url: str
    author_login: str
    author_association: str
    repository_permission: str | None
    authorized: bool
    reason: str


@dataclass(frozen=True)
class CockpitPullRequestTarget:
    """Validated metadata for the PR branch the curator may update."""

    pr_number: int
    head_ref: str
    head_oid: str
    base_oid: str
    head_repository_owner: str
    url: str
    state: str
    is_cross_repository: bool
    title: str = ""
    base_ref: str = "main"


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


def collect_pull_request_comments(
    *,
    pr_number: int,
    repo: str = "magilliam27/MCI-GRU",
    repo_root: Path | None = None,
    run_command: CommandRunner | None = None,
) -> tuple[PullRequestComment, ...] | None:
    """Collect paginated PR comments; return None when GitHub evidence is unavailable."""
    if pr_number <= 0:
        raise ValueError("Pull request number must be positive.")
    runner = run_command or _run_command(repo_root or Path.cwd())
    command = [
        "gh",
        "api",
        f"repos/{repo}/issues/{pr_number}/comments",
        "--paginate",
        "--slurp",
    ]
    try:
        pages = json.loads(runner(command))
        if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
            raise ValueError("expected paginated JSON lists")
        comments = [_parse_pull_request_comment(value) for page in pages for value in page]
    except Exception:
        return None
    return tuple(sorted(comments, key=lambda comment: (int(comment.comment_id), comment.url)))


def collect_pull_request_paths(
    *,
    pr_number: int,
    repo: str = "magilliam27/MCI-GRU",
    repo_root: Path | None = None,
    run_command: CommandRunner | None = None,
) -> tuple[str, ...] | None:
    """Collect every changed PR filename for the curator's fail-closed allowlist."""
    if pr_number <= 0:
        raise ValueError("Pull request number must be positive.")
    runner = run_command or _run_command(repo_root or Path.cwd())
    command = [
        "gh",
        "api",
        f"repos/{repo}/pulls/{pr_number}/files",
        "--paginate",
        "--slurp",
    ]
    try:
        pages = json.loads(runner(command))
        if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
            raise ValueError("expected paginated JSON lists")
        paths: set[str] = set()
        for page in pages:
            for value in page:
                item = _json_object(value)
                paths.add(_text(item.get("filename")))
                if "previous_filename" in item:
                    paths.add(_text(item["previous_filename"]))
    except Exception:
        return None
    return tuple(sorted(paths))


def collect_cockpit_pr_target(
    *,
    pr_number: int,
    repo: str = "magilliam27/MCI-GRU",
    repo_root: Path | None = None,
    run_command: CommandRunner | None = None,
) -> CockpitPullRequestTarget | None:
    """Collect the PR target metadata required by the cockpit curator."""
    if pr_number <= 0:
        raise ValueError("Pull request number must be positive.")
    runner = run_command or _run_command(repo_root or Path.cwd())
    command = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "title,headRefName,headRefOid,baseRefName,baseRefOid,headRepositoryOwner,isCrossRepository,url,state",
    ]
    try:
        item = _json_object(json.loads(runner(command)))
        required = {
            "title",
            "headRefName",
            "headRefOid",
            "baseRefName",
            "baseRefOid",
            "headRepositoryOwner",
            "isCrossRepository",
            "url",
            "state",
        }
        if not required.issubset(item):
            raise ValueError("malformed pull request target evidence")
        owner = _json_object(item["headRepositoryOwner"])
        is_cross_repository = item["isCrossRepository"]
        if not isinstance(is_cross_repository, bool):
            raise ValueError("malformed cross-repository state")
        target = CockpitPullRequestTarget(
            pr_number=pr_number,
            head_ref=_raw_text(item["headRefName"]),
            head_oid=_oid(item["headRefOid"]),
            base_oid=_oid(item["baseRefOid"]),
            head_repository_owner=_text(owner.get("login")),
            url=_text(item["url"]),
            state=_state(item["state"], {"open", "closed", "merged"}),
            is_cross_repository=is_cross_repository,
            title=_raw_text(item["title"]),
            base_ref=_raw_text(item["baseRefName"]),
        )
        repo_owner, separator, repo_name = repo.partition("/")
        branch_date = target.head_ref.removeprefix("codex/cockpit-refresh-")
        is_cockpit_branch = (
            target.head_ref == f"codex/cockpit-refresh-{branch_date}"
            and len(branch_date) == 8
            and branch_date.isdigit()
        )
        branch_run_date = date(
            int(branch_date[:4]),
            int(branch_date[4:6]),
            int(branch_date[6:]),
        )
        if (
            not separator
            or not repo_owner.strip()
            or not repo_name.strip()
            or target.head_repository_owner.casefold() != repo_owner.strip().casefold()
            or target.is_cross_repository
            or not is_cockpit_branch
            or target.state != "open"
            or target.base_ref != "main"
            or target.title != f"Cockpit refresh: {branch_run_date.isoformat()}"
        ):
            return None
        return target
    except Exception:
        return None


def authorize_owner_comment(
    comment: PullRequestComment,
    *,
    repo: str = "magilliam27/MCI-GRU",
    repo_root: Path | None = None,
    run_command: CommandRunner | None = None,
) -> CommentAuthorization:
    """Authorize trusted API evidence for the repository owner's own comment."""
    owner, separator, name = repo.partition("/")
    if not separator or not owner.strip() or not name.strip():
        raise ValueError("Repository must use the owner/name form.")
    association = comment.author_association.lower()
    is_owner = comment.author_login.casefold() == owner.strip().casefold()
    authorized = is_owner and association == "owner"
    if authorized:
        reason = "verified repository owner comment association"
    else:
        reason = "only the repository owner may apply cockpit overrides"
    return CommentAuthorization(
        comment_id=comment.comment_id,
        comment_url=comment.url,
        author_login=comment.author_login,
        author_association=association,
        repository_permission=None,
        authorized=authorized,
        reason=reason,
    )


def post_pull_request_response(
    target: CockpitPullRequestTarget,
    *,
    body: str,
    run_command: CommandRunner,
    repo: str = "magilliam27/MCI-GRU",
) -> str:
    """Post a curator response to a validated open PR target."""
    response_body = body.strip()
    if not response_body:
        raise ValueError("Pull request response body must not be empty.")
    if target.state != "open":
        raise ValueError("Cockpit curator responses require an open pull request.")
    return run_command(
        [
            "gh",
            "pr",
            "comment",
            str(target.pr_number),
            "--repo",
            repo,
            "--body",
            response_body,
        ]
    ).strip()


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
    producer_base_oid: str | None = None,
    producer_remote_head_oid: str | None = None,
) -> GitHubSyncResult:
    if not enabled:
        raise GitHubSyncDisabled("GitHub cockpit sync requires --github-sync.")
    if repo_root is None or run_date is None:
        raise ValueError("GitHub cockpit sync requires repo_root and run_date.")
    decisions = decision_queue or []
    runner = run_command or _run_command(repo_root)
    branch = cockpit_branch_name(run_date)
    runner(["gh", "auth", "status"])
    current_branch = runner(["git", "branch", "--show-current"]).strip()
    if current_branch != branch:
        raise RuntimeError(
            f"GitHub cockpit sync requires the prepared dated cockpit branch {branch}."
        )
    paths = [
        "docs/agents/workstreams.md",
        f"docs/agents/cockpit/{run_date.isoformat()}.md",
        "docs/agents/cockpit/auto-decisions.json",
        DECISION_REGISTRY_PATH,
        "docs/agents/cockpit/override-receipts.json",
        "docs/agents/cockpit/RUNBOOK.md",
    ]
    actions_taken: list[str] = []
    actions_skipped: list[str] = []
    if producer_base_oid is not None:
        _preflight_producer_branch(
            runner,
            repo=repo,
            branch=branch,
            run_date=run_date,
            allowed_paths=set(paths),
            base_oid=producer_base_oid,
            remote_head_oid=producer_remote_head_oid,
        )
    if runner(["git", "status", "--short", "--", *paths]).strip():
        runner(["git", "add", *paths])
        staged_paths = {
            line.strip()
            for line in runner(["git", "diff", "--cached", "--name-only"]).splitlines()
            if line.strip()
        }
        unexpected = staged_paths - set(paths)
        if unexpected:
            raise RuntimeError(
                "GitHub cockpit sync found an unexpected staged path: "
                + ", ".join(sorted(unexpected))
            )
        runner(["git", "commit", "-m", f"Refresh cockpit status for {run_date.isoformat()}"])
        actions_taken.append("committed cockpit files")
    else:
        actions_skipped.append("no cockpit file changes to commit")
    if producer_base_oid is not None:
        local_head_oid = _oid(runner(["git", "rev-parse", "HEAD"]))
        _require_allowed_git_diff(runner, producer_base_oid, local_head_oid, set(paths))
    runner(["git", "push", "-u", "origin", branch])
    pr_url = _ensure_pr(runner, repo, branch, run_date)
    pr_number = _parse_issue_url_number(pr_url)
    issue_number, issue_url = _ensure_cockpit_issue(runner, repo)
    _apply_existing_labels(
        runner,
        repo,
        issue_number,
        ["cockpit-reviewed"],
        actions_taken,
        actions_skipped,
    )
    pr_actions_taken: list[str] = []
    pr_actions_skipped: list[str] = []
    pr_label_receipt = _apply_existing_labels(
        runner,
        repo,
        pr_number,
        list(_DATED_COCKPIT_PR_LABELS),
        pr_actions_taken,
        pr_actions_skipped,
        resource="pr",
        target_name="dated cockpit PR",
    )
    actions_taken.extend(pr_actions_taken)
    actions_skipped.extend(pr_actions_skipped)
    digest_action = _ensure_issue_digest_comment(
        runner,
        repo=repo,
        issue_number=issue_number,
        body=_issue_comment(
            run_date,
            run_color,
            pr_url,
            decisions,
            label_receipt=pr_label_receipt,
        ),
        run_date=run_date,
    )
    if digest_action == "unchanged":
        actions_skipped.append(f"cockpit issue digest #{issue_number} unchanged")
    else:
        actions_taken.append(f"{digest_action} cockpit issue digest #{issue_number}")
    return GitHubSyncResult(
        branch=branch,
        pr_url=pr_url,
        cockpit_issue_number=issue_number,
        cockpit_issue_url=issue_url,
        actions_taken=actions_taken,
        actions_skipped=actions_skipped,
        dated_pr_labels=pr_label_receipt,
    )


def _preflight_producer_branch(
    runner: CommandRunner,
    *,
    repo: str,
    branch: str,
    run_date: date,
    allowed_paths: set[str],
    base_oid: str,
    remote_head_oid: str | None,
) -> None:
    base_oid = _oid(base_oid)
    if remote_head_oid is not None:
        remote_head_oid = _oid(remote_head_oid)
    local_head_oid = _oid(runner(["git", "rev-parse", "HEAD"]))
    expected_head_oid = remote_head_oid or base_oid
    if local_head_oid != expected_head_oid:
        raise RuntimeError("Local dated cockpit branch does not match validated fetched evidence.")

    compared_head_oid = remote_head_oid or base_oid
    branch_paths = _require_allowed_git_diff(
        runner,
        base_oid,
        compared_head_oid,
        allowed_paths,
    )
    existing = _existing_prs(runner, repo, branch)
    if not existing:
        return
    if remote_head_oid is None:
        raise RuntimeError("Existing dated cockpit PR has no matching remote branch.")
    item = existing[0]
    try:
        pr_number = _number(item.get("number"))
    except Exception as exc:
        raise RuntimeError(
            "The existing dated cockpit PR has incomplete identity evidence."
        ) from exc
    target = collect_cockpit_pr_target(pr_number=pr_number, repo=repo, run_command=runner)
    if (
        target is None
        or target.head_ref != branch
        or target.title != f"Cockpit refresh: {run_date.isoformat()}"
        or target.base_ref != "main"
        or target.state != "open"
        or target.base_oid != base_oid
        or target.head_oid != remote_head_oid
    ):
        raise RuntimeError("The existing dated cockpit PR does not match fetched branch evidence.")
    api_paths = collect_pull_request_paths(
        pr_number=pr_number,
        repo=repo,
        run_command=runner,
    )
    if api_paths is None:
        raise RuntimeError("The existing dated cockpit PR has incomplete file evidence.")
    unexpected = set(api_paths) - allowed_paths
    if unexpected:
        raise RuntimeError(
            "GitHub cockpit sync found an unexpected PR path: " + ", ".join(sorted(unexpected))
        )
    if set(api_paths) != branch_paths:
        raise RuntimeError(
            "The existing dated cockpit PR file evidence does not match the fetched branch diff."
        )


def _existing_prs(runner: CommandRunner, repo: str, branch: str) -> list[dict[str, object]]:
    payload = runner(
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
            "number,url,title,baseRefName,headRefName,state",
        ]
    )
    existing = [_json_object(value) for value in _json_list(payload)]
    if len(existing) > 1:
        raise RuntimeError("Multiple PRs exist for the dated cockpit branch.")
    return existing


def _require_allowed_git_diff(
    runner: CommandRunner,
    base_oid: str,
    head_oid: str,
    allowed_paths: set[str],
) -> set[str]:
    output = runner(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            f"{base_oid}...{head_oid}",
            "--",
        ]
    )
    fields = output.split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    paths: set[str] = set()
    index = 0
    while index < len(fields):
        status_parts = fields[index].split("\t")
        index += 1
        status = status_parts[0]
        if not status or status[0] not in "ACDMRTUXB":
            raise RuntimeError("GitHub cockpit sync received malformed branch diff evidence.")
        expected_paths = 2 if status[0] in "RC" else 1
        values = status_parts[1:]
        while len(values) < expected_paths and index < len(fields):
            values.append(fields[index])
            index += 1
        if len(values) != expected_paths or any(not value for value in values):
            raise RuntimeError("GitHub cockpit sync received malformed branch diff evidence.")
        paths.update(values)
    unexpected = paths - allowed_paths
    if unexpected:
        raise RuntimeError(
            "GitHub cockpit sync found an unexpected branch path: " + ", ".join(sorted(unexpected))
        )
    return paths


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
                [
                    "gh",
                    "label",
                    "list",
                    "--repo",
                    repo,
                    "--limit",
                    "1000",
                    "--json",
                    "name",
                    "--jq",
                    ".[].name",
                ]
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
    existing_payload = runner(
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
            "url,title,baseRefName,headRefName,state",
        ]
    )
    existing = _json_list(existing_payload)
    if len(existing) > 1:
        raise RuntimeError("Multiple PRs exist for the dated cockpit branch.")
    if existing:
        item = _json_object(existing[0])
        expected_title = f"Cockpit refresh: {run_date.isoformat()}"
        if (
            _raw_text(item.get("title")) != expected_title
            or _raw_text(item.get("baseRefName")) != "main"
            or _raw_text(item.get("headRefName")) != branch
            or _state(item.get("state"), {"open", "closed", "merged"}) != "open"
        ):
            raise RuntimeError("The existing dated cockpit PR does not match its contract.")
        return _text(item.get("url"))
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
    *,
    resource: str = "issue",
    target_name: str = "cockpit issue",
) -> LabelSyncReceipt:
    if resource not in {"issue", "pr"}:
        raise ValueError("Label target resource must be 'issue' or 'pr'.")
    existing = set(
        _split_lines(
            runner(
                [
                    "gh",
                    "label",
                    "list",
                    "--repo",
                    repo,
                    "--limit",
                    "1000",
                    "--json",
                    "name",
                    "--jq",
                    ".[].name",
                ]
            )
        )
    )
    eligible_labels = [label for label in labels if label in existing]
    view_command = [
        "gh",
        resource,
        "view",
        str(issue_number),
        "--repo",
        repo,
        "--json",
        "labels",
        "--jq",
        ".labels[].name",
    ]
    current = set(_split_lines(runner(view_command))) if eligible_labels else set()
    labels_to_apply = [label for label in labels if label in existing and label not in current]
    labels_already_applied = [label for label in labels if label in existing and label in current]
    missing = [label for label in labels if label not in existing]
    readback = current
    if labels_to_apply:
        runner(
            [
                "gh",
                resource,
                "edit",
                str(issue_number),
                "--repo",
                repo,
                "--add-label",
                ",".join(labels_to_apply),
            ]
        )
        if resource == "pr":
            readback = set(_split_lines(runner(view_command)))
            absent_after_edit = [label for label in eligible_labels if label not in readback]
            if absent_after_edit:
                raise RuntimeError(
                    f"Label readback failed for {target_name} #{issue_number}: "
                    + ", ".join(absent_after_edit)
                )
        else:
            readback = current | set(labels_to_apply)
        actions_taken.append(
            f"applied labels to {target_name} #{issue_number}: {', '.join(labels_to_apply)}"
        )
    if missing:
        actions_skipped.append(
            f"skipped missing labels for {target_name} #{issue_number}: {', '.join(missing)}"
        )
    if labels_already_applied:
        actions_skipped.append(
            f"labels already present on {target_name} #{issue_number}: "
            + ", ".join(labels_already_applied)
        )
    return LabelSyncReceipt(
        applied=tuple(labels_to_apply),
        already_present=tuple(labels_already_applied),
        skipped_missing=tuple(missing),
        verified_present=tuple(label for label in labels if label in readback),
    )


def _issue_comment(
    run_date: date,
    run_color: str,
    pr_url: str,
    decision_queue: list[str],
    *,
    label_receipt: LabelSyncReceipt | None = None,
) -> str:
    lines = [
        f"<!-- mci-gru-cockpit-refresh:{run_date.isoformat()} -->",
        f"Cockpit refresh {run_date.isoformat()}: {run_color}",
        "",
        f"PR: {pr_url}",
    ]
    if decision_queue:
        lines.extend(["", "Decision queue:"])
        lines.extend(f"- {decision}" for decision in decision_queue)
    if label_receipt is not None:
        lines.extend(
            [
                "",
                "Dated PR label receipt:",
                f"- applied: {_label_receipt_value(label_receipt.applied)}",
                (f"- already-present: {_label_receipt_value(label_receipt.already_present)}"),
                (f"- skipped-missing: {_label_receipt_value(label_receipt.skipped_missing)}"),
                (
                    "- verified-after-readback: "
                    f"{_label_receipt_value(label_receipt.verified_present)}"
                ),
            ]
        )
    return "\n".join(lines)


def _label_receipt_value(labels: tuple[str, ...]) -> str:
    return ", ".join(labels) if labels else "none"


def _ensure_issue_digest_comment(
    runner: CommandRunner,
    *,
    repo: str,
    issue_number: int,
    body: str,
    run_date: date,
) -> str:
    marker = f"<!-- mci-gru-cockpit-refresh:{run_date.isoformat()} -->"
    raw_pages = json.loads(
        runner(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/{issue_number}/comments",
                "--paginate",
                "--slurp",
            ]
        )
    )
    if not isinstance(raw_pages, list) or not all(isinstance(page, list) for page in raw_pages):
        raise RuntimeError("Cockpit issue comment evidence is incomplete.")
    matches: list[tuple[int, str]] = []
    for page in raw_pages:
        for raw_comment in page:
            comment = _json_object(raw_comment)
            comment_body = _text(comment.get("body"))
            if marker in comment_body and _trusted_digest_author(comment, repo):
                matches.append((_number(comment.get("id")), comment_body))
    if len(matches) > 1:
        raise RuntimeError("Cockpit issue has multiple dated digest comments.")
    if not matches:
        runner(
            [
                "gh",
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                repo,
                "--body",
                body,
            ]
        )
        return "created"
    comment_id, existing_body = matches[0]
    body = _preserve_label_receipt_history(existing_body, body)
    if existing_body == body:
        return "unchanged"
    runner(
        [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{repo}/issues/comments/{comment_id}",
            "-f",
            f"body={body}",
        ]
    )
    return "updated"


def _preserve_label_receipt_history(existing_body: str, current_body: str) -> str:
    previous = _extract_label_receipt(existing_body)
    current = _extract_label_receipt(current_body)
    if previous is None or current is None:
        return current_body
    ever_applied = set(previous.applied) | set(current.applied)
    applied = tuple(label for label in _DATED_COCKPIT_PR_LABELS if label in ever_applied)
    already_present = tuple(
        label
        for label in _DATED_COCKPIT_PR_LABELS
        if label not in ever_applied and label in current.already_present
    )
    skipped_missing = tuple(
        label
        for label in _DATED_COCKPIT_PR_LABELS
        if label not in ever_applied
        and label not in already_present
        and label in current.skipped_missing
    )
    durable = LabelSyncReceipt(
        applied=applied,
        already_present=already_present,
        skipped_missing=skipped_missing,
        verified_present=current.verified_present,
    )
    return _replace_label_receipt(current_body, durable)


def _extract_label_receipt(body: str) -> LabelSyncReceipt | None:
    lines = body.splitlines()
    try:
        start = lines.index("Dated PR label receipt:")
    except ValueError:
        return None
    expected = (
        "- applied: ",
        "- already-present: ",
        "- skipped-missing: ",
        "- verified-after-readback: ",
    )
    values: list[tuple[str, ...]] = []
    for offset, prefix in enumerate(expected, start=1):
        if start + offset >= len(lines) or not lines[start + offset].startswith(prefix):
            raise RuntimeError("Cockpit issue label receipt is malformed.")
        raw = lines[start + offset].removeprefix(prefix)
        values.append(() if raw == "none" else tuple(item.strip() for item in raw.split(",")))
    return LabelSyncReceipt(*values)


def _replace_label_receipt(body: str, receipt: LabelSyncReceipt) -> str:
    lines = body.splitlines()
    try:
        start = lines.index("Dated PR label receipt:")
    except ValueError:
        return body
    replacement = [
        "Dated PR label receipt:",
        f"- applied: {_label_receipt_value(receipt.applied)}",
        f"- already-present: {_label_receipt_value(receipt.already_present)}",
        f"- skipped-missing: {_label_receipt_value(receipt.skipped_missing)}",
        f"- verified-after-readback: {_label_receipt_value(receipt.verified_present)}",
    ]
    return "\n".join([*lines[:start], *replacement, *lines[start + 5 :]])


def _trusted_digest_author(comment: dict[str, object], repo: str) -> bool:
    author = _json_object(comment.get("user"))
    login = _text(author.get("login"))
    association = _text(comment.get("author_association")).lower()
    repository_owner = _text(repo).split("/", 1)[0]
    return (
        login.casefold() == repository_owner.casefold() and association == "owner"
    ) or login.casefold() == "github-actions[bot]"


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


def _parse_pull_request_comment(value: object) -> PullRequestComment:
    item = _json_object(value)
    required = {"id", "html_url", "user", "author_association", "body", "created_at"}
    if not required.issubset(item):
        raise ValueError("malformed pull request comment evidence")
    body = item["body"]
    if not isinstance(body, str):
        raise ValueError("malformed pull request comment body")
    author = _json_object(item["user"])
    comment_id = _number(item["id"])
    return PullRequestComment(
        comment_id=str(comment_id),
        url=_text(item["html_url"]),
        author_login=_text(author.get("login")),
        author_association=_text(item["author_association"]).lower(),
        body=body,
        created_at=_required_timestamp(item["created_at"]),
    )


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


def _raw_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-empty text")
    return value


def _oid(value: object) -> str:
    oid = _text(value).lower()
    if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", oid) is None:
        raise ValueError("expected full Git object ID")
    return oid


def _state(value: object, allowed: set[str]) -> str:
    state = _text(value).lower()
    if state not in allowed:
        raise ValueError("unexpected GitHub state")
    return state


def _required_date(value: object) -> date:
    return _required_timestamp(value).date()


def _required_timestamp(value: object) -> datetime:
    raw_timestamp = _text(value)
    normalized_timestamp = (
        f"{raw_timestamp[:-1]}+00:00" if raw_timestamp.endswith("Z") else raw_timestamp
    )
    timestamp = datetime.fromisoformat(normalized_timestamp)
    if timestamp.tzinfo is None:
        raise ValueError("expected timezone-aware ISO timestamp")
    return timestamp


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
