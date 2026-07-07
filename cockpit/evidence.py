from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


REQUIRED_DOCS = [
    "AGENTS.md",
    "docs/agents/domain.md",
    "docs/agents/issue-tracker.md",
    "docs/agents/triage-labels.md",
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
    return LocalEvidence(
        repo_root=repo_root,
        required_docs=required_docs,
        recent_handoffs=_recent_handoffs(repo_root),
        dirty_paths=_parse_dirty_paths(status_branch),
        branches=branches,
        worktrees=worktrees,
        recent_commits=runner(["git", "log", "-5", "--oneline"]).strip(),
        git_topology=topology,
    )


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
    return GitTopologySnapshot(
        current_branch=current_branch,
        status_header=_first_status_header(status_branch),
        origin_main_ahead=origin_main_divergence[0],
        origin_main_behind=origin_main_divergence[1],
        branches=branches,
        unmerged_branches=[branch.name for branch in unmerged_branches],
        unmerged_branch_details=unmerged_branches,
        worktrees=parsed_worktrees,
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
    branch = str(block.get("branch", f"detached@{head[:7]}" if detached else "unknown"))
    return WorktreeEvidence(
        path=str(block.get("path", "")),
        head=head,
        branch=branch,
        detached=detached,
        status_header="",
    )


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
        with_status.append(
            WorktreeEvidence(
                path=worktree.path,
                head=worktree.head,
                branch=worktree.branch,
                detached=worktree.detached,
                status_header=_first_status_header(status),
                dirty_paths=_parse_dirty_paths(status),
            )
        )
    return with_status


def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        with_safe_directory(args, cwd),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
