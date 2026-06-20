from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
    return LocalEvidence(
        repo_root=repo_root,
        required_docs=required_docs,
        recent_handoffs=_recent_handoffs(repo_root),
        dirty_paths=_parse_dirty_paths(runner(["git", "status", "--short"])),
        branches=_split_lines(runner(["git", "branch", "--format=%(refname:short)"])),
        worktrees=runner(["git", "worktree", "list", "--porcelain"]).strip(),
        recent_commits=runner(["git", "log", "-5", "--oneline"]).strip(),
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
        paths.append(line[3:].strip())
    return paths


def _split_lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout
