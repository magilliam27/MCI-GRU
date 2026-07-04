from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cockpit._compat import StrEnum

if TYPE_CHECKING:
    from datetime import date


class WorkstreamStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    PARKED = "parked"
    LOCAL_ONLY = "local-only"
    READY_FOR_AGENT = "ready-for-agent"
    NEEDS_USER_DECISION = "needs-user-decision"
    DONE = "done"
    ARCHIVE = "archive"
    STALE = "stale"


class RunColor(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class Workstream:
    name: str
    status: WorkstreamStatus
    tracker: str
    continuation: str
    source_of_truth: str
    latest_artifact: str
    last_verification: str
    blocked_on: str
    next_action: str
    owner: str
    last_reviewed: date


@dataclass(frozen=True)
class Decision:
    workstream: str
    question: str
    options: str


@dataclass(frozen=True)
class GitHubAction:
    action: str
    target: str
    reason: str


@dataclass(frozen=True)
class BranchEvidence:
    name: str
    local: bool = False
    remote: bool = False

    @property
    def provenance_label(self) -> str:
        if self.local and self.remote:
            return "local+remote"
        if self.remote:
            return "remote-only"
        return "local"

    @property
    def display_name(self) -> str:
        if self.remote and not self.local:
            return f"origin/{self.name}"
        return self.name


@dataclass(frozen=True)
class WorktreeEvidence:
    path: str
    head: str
    branch: str
    detached: bool
    status_header: str
    dirty_paths: list[str] = field(default_factory=list)
    status_error: str = ""

    @property
    def is_dirty(self) -> bool:
        return bool(self.dirty_paths or self.status_error)


@dataclass(frozen=True)
class GitTopologySnapshot:
    current_branch: str
    status_header: str
    origin_main_ahead: int
    origin_main_behind: int
    branches: list[str] = field(default_factory=list)
    unmerged_branches: list[str] = field(default_factory=list)
    unmerged_branch_details: list[BranchEvidence] = field(default_factory=list)
    worktrees: list[WorktreeEvidence] = field(default_factory=list)

    @property
    def detached_worktrees(self) -> list[WorktreeEvidence]:
        return [worktree for worktree in self.worktrees if worktree.detached]

    @property
    def dirty_worktrees(self) -> list[WorktreeEvidence]:
        return [worktree for worktree in self.worktrees if worktree.is_dirty]

    @property
    def has_attention_items(self) -> bool:
        return bool(
            self.origin_main_ahead
            or self.origin_main_behind
            or self.unmerged_branches
            or self.detached_worktrees
            or self.dirty_worktrees
        )


@dataclass(frozen=True)
class CockpitReport:
    run_date: date
    color: RunColor
    executive_summary: str
    decisions: list[Decision] = field(default_factory=list)
    active_workstreams: list[Workstream] = field(default_factory=list)
    blocked_workstreams: list[Workstream] = field(default_factory=list)
    local_only_work: list[Workstream] = field(default_factory=list)
    stale_or_archive_candidates: list[Workstream] = field(default_factory=list)
    github_actions_taken: list[GitHubAction] = field(default_factory=list)
    github_actions_skipped: list[GitHubAction] = field(default_factory=list)
    git_tree_impact: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
