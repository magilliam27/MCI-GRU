from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from cockpit.decisions import DECISION_REGISTRY_PATH
from cockpit.git import with_safe_directory
from cockpit.models import BranchEvidence, GitTopologySnapshot, WorktreeEvidence

RunCommand = Callable[[list[str]], str]


@dataclass(frozen=True)
class LocalEvidence:
    repo_root: Path
    required_docs: dict[str, bool]
    recent_handoffs: list[str]
    dirty_paths: list[str]
    branches: list[str]
    worktrees: str
    recent_commits: str
    git_topology: GitTopologySnapshot
    recent_branches: list[tuple[str, date]] = field(default_factory=list)
    branch_commit_dates: dict[str, date] = field(default_factory=dict)


REQUIRED_DOCS = [
    "AGENTS.md",
    "docs/agents/domain.md",
    "docs/agents/issue-tracker.md",
    "docs/agents/triage-labels.md",
    DECISION_REGISTRY_PATH,
    "docs/index.md",
    "docs/research/README.md",
]


def collect_local_evidence(repo_root: Path, run_command: RunCommand | None = None) -> LocalEvidence:
    runner = run_command or (lambda args: _run_git(args, cwd=repo_root))
    required_docs = {path: (repo_root / path).exists() for path in REQUIRED_DOCS}
    status_branch = runner(["git", "status", "--short", "--branch"])
    branches = _split_lines(runner(["git", "branch", "--format=%(refname:short)"]))
    worktrees = runner(["git", "worktree", "list", "--porcelain"]).strip()
    topology = _build_git_topology(
        repo_root=repo_root,
        status_branch=status_branch,
        branches=branches,
        unmerged_branches=_parse_branch_list(
            runner(["git", "branch", "--all", "--no-merged", "origin/main"])
        ),
        origin_main_divergence=_parse_divergence(
            runner(["git", "rev-list", "--left-right", "--count", "origin/main...HEAD"])
        ),
        worktrees=worktrees,
        run_command=runner,
    )
    branch_date_output = runner(
        [
            "git",
            "for-each-ref",
            "--sort=-committerdate",
            "--format=%(committerdate:short)%09%(refname)",
            "refs/heads",
            "refs/remotes/origin",
        ]
    )
    recent_branches, branch_commit_dates = _parse_branch_commit_dates(branch_date_output)
    return LocalEvidence(
        repo_root=repo_root,
        required_docs=required_docs,
        recent_handoffs=_recent_handoffs(repo_root),
        dirty_paths=_parse_dirty_paths(status_branch),
        branches=branches,
        worktrees=worktrees,
        recent_commits=runner(["git", "log", "-5", "--oneline"]).strip(),
        git_topology=topology,
        recent_branches=recent_branches,
        branch_commit_dates=branch_commit_dates,
    )


def _parse_branch_commit_dates(
    output: str,
) -> tuple[list[tuple[str, date]], dict[str, date]]:
    """Parse local and origin committer dates with local duplicate precedence.

    Full refnames are preferred, but historical short-ref fixtures remain valid.
    Malformed lines and origin/HEAD are skipped.
    """
    local_dates: dict[str, date] = {}
    remote_dates: dict[str, date] = {}
    local_order: list[str] = []
    for line in output.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        parts = cleaned.split(maxsplit=1)
        if len(parts) != 2:
            continue
        raw_date, name = parts[0], parts[1].strip()
        if not name:
            continue
        try:
            committer_date = date.fromisoformat(raw_date)
        except ValueError:
            continue
        is_remote = name.startswith(("refs/remotes/origin/", "remotes/origin/", "origin/"))
        normalized = (
            name.removeprefix("refs/heads/")
            .removeprefix("refs/remotes/origin/")
            .removeprefix("remotes/origin/")
            .removeprefix("origin/")
            .strip()
        )
        if not normalized or normalized == "HEAD":
            continue
        if is_remote:
            remote_dates.setdefault(normalized, committer_date)
            continue
        if normalized not in local_dates:
            local_order.append(normalized)
        local_dates.setdefault(normalized, committer_date)
    combined = {**remote_dates, **local_dates}
    recent_branches = [(name, local_dates[name]) for name in local_order]
    return recent_branches, dict(sorted(combined.items()))


def _recent_handoffs(repo_root: Path) -> list[str]:
    handoff_dir = repo_root / "docs" / "handoffs"
    if not handoff_dir.exists():
        return []
    files = sorted(handoff_dir.glob("*.md"), key=lambda path: path.name, reverse=True)
    return [path.relative_to(repo_root).as_posix() for path in files[:10]]


def _parse_dirty_paths(status_output: str) -> list[str]:
    paths: list[str] = []
    for line in status_output.splitlines():
        if not line.strip():
            continue
        if line.startswith("##"):
            continue
        paths.append(line[3:].strip())
    return paths


def _split_lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _build_git_topology(
    *,
    repo_root: Path,
    status_branch: str,
    branches: list[str],
    unmerged_branches: list[BranchEvidence],
    origin_main_divergence: tuple[int, int],
    worktrees: str,
    run_command: RunCommand,
) -> GitTopologySnapshot:
    parsed_worktrees = _collect_worktree_statuses(_parse_worktrees(worktrees), run_command)
    current_branch = _parse_current_branch(status_branch)
    if _is_detached_branch_name(current_branch):
        current_branch = _detached_current_branch(repo_root, parsed_worktrees)
    current_path = _normalize_path(repo_root)
    control_plane_worktree: WorktreeEvidence | None = None
    for index, worktree in enumerate(parsed_worktrees):
        if _normalize_path(Path(worktree.path)) != current_path:
            continue
        control_plane_worktree = WorktreeEvidence(
            path=worktree.path,
            head=worktree.head,
            branch=worktree.branch,
            detached=worktree.detached,
            status_header=worktree.status_header,
            dirty_paths=worktree.dirty_paths,
            status_error=worktree.status_error,
            origin_main_ahead=origin_main_divergence[0],
            origin_main_behind=origin_main_divergence[1],
        )
        parsed_worktrees[index] = control_plane_worktree
        break
    if control_plane_worktree is None:
        fallback = parsed_worktrees[0] if parsed_worktrees else None
        control_plane_worktree = WorktreeEvidence(
            path=fallback.path if fallback is not None else repo_root.as_posix(),
            head=fallback.head if fallback is not None else "unknown",
            branch=fallback.branch if fallback is not None else current_branch,
            detached=(
                fallback.detached
                if fallback is not None
                else _is_detached_branch_name(current_branch)
            ),
            status_header=(
                fallback.status_header
                if fallback is not None
                else _first_status_header(status_branch)
            ),
            dirty_paths=(
                fallback.dirty_paths if fallback is not None else _parse_dirty_paths(status_branch)
            ),
            status_error=fallback.status_error if fallback is not None else "",
            origin_main_ahead=origin_main_divergence[0],
            origin_main_behind=origin_main_divergence[1],
        )
        if fallback is not None:
            parsed_worktrees[0] = control_plane_worktree
    return GitTopologySnapshot(
        current_branch=current_branch,
        status_header=_first_status_header(status_branch),
        origin_main_ahead=origin_main_divergence[0],
        origin_main_behind=origin_main_divergence[1],
        branches=branches,
        unmerged_branches=[branch.name for branch in unmerged_branches],
        unmerged_branch_details=unmerged_branches,
        worktrees=parsed_worktrees,
        control_plane_worktree=control_plane_worktree,
        primary_worktree=parsed_worktrees[0] if parsed_worktrees else control_plane_worktree,
    )


def _parse_current_branch(status_output: str) -> str:
    header = _first_status_header(status_output)
    if not header:
        return "unknown"
    branch = header.removeprefix("## ").split("...")[0].strip()
    return branch or "unknown"


def _is_detached_branch_name(branch: str) -> bool:
    return branch == "HEAD" or branch.startswith("HEAD ")


def _detached_current_branch(repo_root: Path, worktrees: list[WorktreeEvidence]) -> str:
    current_path = _normalize_path(repo_root)
    for worktree in worktrees:
        if worktree.detached and _normalize_path(Path(worktree.path)) == current_path:
            return worktree.branch
    for worktree in worktrees:
        if worktree.detached:
            return worktree.branch
    return "detached"


def _normalize_path(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def _first_status_header(status_output: str) -> str:
    for line in status_output.splitlines():
        if line.startswith("##"):
            return line.strip()
    return ""


def _parse_branch_list(output: str) -> list[BranchEvidence]:
    branches: dict[str, BranchEvidence] = {}
    for line in output.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = cleaned.removeprefix("*").removeprefix("+").strip()
        if cleaned.startswith("(HEAD detached"):
            continue
        if cleaned.startswith("remotes/origin/HEAD"):
            continue
        remote = cleaned.startswith("remotes/origin/")
        name = cleaned.removeprefix("remotes/origin/").strip()
        if not name:
            continue
        existing = branches.get(name, BranchEvidence(name=name))
        branches[name] = BranchEvidence(
            name=name,
            local=existing.local or not remote,
            remote=existing.remote or remote,
        )
    return list(branches.values())


def _parse_divergence(output: str) -> tuple[int, int]:
    parts = output.split()
    if len(parts) < 2:
        return (0, 0)
    try:
        return (int(parts[1]), int(parts[0]))
    except ValueError:
        return (0, 0)


def _parse_worktrees(output: str) -> list[WorktreeEvidence]:
    worktrees: list[WorktreeEvidence] = []
    current: dict[str, str | bool] = {}
    for line in [*output.splitlines(), ""]:
        if not line.strip():
            if current:
                worktrees.append(_worktree_from_block(current))
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line.removeprefix("worktree ").strip()
        elif line.startswith("HEAD "):
            current["head"] = line.removeprefix("HEAD ").strip()
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch refs/heads/").strip()
            current["detached"] = False
        elif line.strip() == "detached":
            current["detached"] = True
    return worktrees


def _worktree_from_block(block: dict[str, str | bool]) -> WorktreeEvidence:
    head = str(block.get("head", "unknown"))
    detached = bool(block.get("detached", False))
    path = str(block.get("path", ""))
    detached_branch = _detached_surface_id(path, head) if detached else "unknown"
    branch = str(block.get("branch", detached_branch))
    return WorktreeEvidence(
        path=path,
        head=head,
        branch=branch,
        detached=detached,
        status_header="",
    )


def _detached_surface_id(path: str, head: str) -> str:
    normalized_path = _normalize_path(Path(path)) if path else "<missing-worktree-path>"
    path_digest = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:10]
    return f"detached@{head[:7]}-{path_digest}"


def _collect_worktree_statuses(
    worktrees: list[WorktreeEvidence],
    run_command: RunCommand,
) -> list[WorktreeEvidence]:
    with_status: list[WorktreeEvidence] = []
    for worktree in worktrees:
        try:
            status = run_command(
                [
                    "git",
                    "-c",
                    f"safe.directory={worktree.path}",
                    "-C",
                    worktree.path,
                    "status",
                    "--porcelain=v1",
                    "-b",
                    "--untracked-files=all",
                ]
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            with_status.append(
                WorktreeEvidence(
                    path=worktree.path,
                    head=worktree.head,
                    branch=worktree.branch,
                    detached=worktree.detached,
                    status_header=worktree.status_header,
                    dirty_paths=worktree.dirty_paths,
                    status_error=str(exc),
                )
            )
            continue
        status_divergence = _status_origin_main_divergence(status)
        origin_main_ahead, origin_main_behind = _worktree_origin_main_divergence(
            worktree,
            run_command,
            fallback=status_divergence,
        )
        with_status.append(
            WorktreeEvidence(
                path=worktree.path,
                head=worktree.head,
                branch=worktree.branch,
                detached=worktree.detached,
                status_header=_first_status_header(status),
                dirty_paths=_parse_dirty_paths(status),
                origin_main_ahead=origin_main_ahead,
                origin_main_behind=origin_main_behind,
            )
        )
    return with_status


def _worktree_origin_main_divergence(
    worktree: WorktreeEvidence,
    run_command: RunCommand,
    *,
    fallback: tuple[int | None, int | None],
) -> tuple[int | None, int | None]:
    try:
        output = run_command(
            [
                "git",
                "-c",
                f"safe.directory={worktree.path}",
                "-C",
                worktree.path,
                "rev-list",
                "--left-right",
                "--count",
                "origin/main...HEAD",
            ]
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    parsed = _parse_optional_divergence(output)
    return parsed if parsed != (None, None) else fallback


def _parse_optional_divergence(output: str) -> tuple[int | None, int | None]:
    parts = output.split()
    if len(parts) < 2:
        return (None, None)
    try:
        return (int(parts[1]), int(parts[0]))
    except ValueError:
        return (None, None)


def _status_origin_main_divergence(status_output: str) -> tuple[int | None, int | None]:
    header = _first_status_header(status_output)
    if "...origin/main" not in header or "[gone]" in header:
        return (None, None)
    ahead_match = re.search(r"\bahead (\d+)\b", header)
    behind_match = re.search(r"\bbehind (\d+)\b", header)
    return (
        int(ahead_match.group(1)) if ahead_match else 0,
        int(behind_match.group(1)) if behind_match else 0,
    )


def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        with_safe_directory(args, cwd),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
