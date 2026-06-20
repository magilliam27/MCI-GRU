from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

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
    verification_notes: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
