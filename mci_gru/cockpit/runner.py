from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mci_gru.cockpit.evidence import collect_local_evidence
from mci_gru.cockpit.github import GitHubSyncResult, cockpit_branch_name, sync_github
from mci_gru.cockpit.models import (
    CockpitReport,
    Decision,
    GitHubAction,
    RunColor,
    Workstream,
    WorkstreamStatus,
)
from mci_gru.cockpit.render import render_cockpit_packet, render_workstream_register

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from mci_gru.cockpit.evidence import RunCommand


@dataclass(frozen=True)
class CockpitRunResult:
    register_path: Path
    packet_path: Path
    color: RunColor
    dirty_paths: list[str]
    report: CockpitReport
    github: GitHubSyncResult | None = None


INITIAL_WORKSTREAMS = [
    (
        "LambdaRankIC",
        WorkstreamStatus.ACTIVE,
        "Review lower-pair screen and loss optimization follow-ups.",
    ),
    (
        "Portfolio-IC",
        WorkstreamStatus.NEEDS_USER_DECISION,
        "Decide whether to promote, park, or rerun current evidence.",
    ),
    (
        "Issue #8 volatility targeting",
        WorkstreamStatus.PARKED,
        "Resume from the issue workflow when user prioritizes it.",
    ),
    (
        "Colab operations",
        WorkstreamStatus.READY_FOR_AGENT,
        "Use the Chrome-control runbook for the next live Colab smoke.",
    ),
    (
        "Regime CSV contract",
        WorkstreamStatus.PARKED,
        "Keep no-lookahead contract tests as the source of truth.",
    ),
    (
        "LSEG access",
        WorkstreamStatus.BLOCKED,
        "Refresh access probe only when data access is needed.",
    ),
    (
        "Daily bug scans",
        WorkstreamStatus.READY_FOR_AGENT,
        "Collapse repeated no-op scans unless a distinct regression appears.",
    ),
    (
        "Docs and research evidence",
        WorkstreamStatus.ACTIVE,
        "Keep docs/research/README.md as the evidence map.",
    ),
    (
        "Git and worktree hygiene",
        WorkstreamStatus.ACTIVE,
        "Track detached or local-only continuation surfaces explicitly.",
    ),
]


def run_local_cockpit_refresh(
    repo_root: Path,
    run_date: date,
    run_command: RunCommand | None = None,
) -> CockpitRunResult:
    evidence = collect_local_evidence(repo_root, run_command=run_command)
    workstreams = _seed_workstreams(run_date)
    color = RunColor.RED if evidence.dirty_paths else RunColor.GREEN
    report = CockpitReport(
        run_date=run_date,
        color=color,
        executive_summary=_executive_summary(evidence.dirty_paths),
        decisions=[] if color == RunColor.GREEN else [_dirty_state_decision()],
        active_workstreams=[row for row in workstreams if row.status == WorkstreamStatus.ACTIVE],
        blocked_workstreams=[row for row in workstreams if row.status == WorkstreamStatus.BLOCKED],
        local_only_work=[row for row in workstreams if row.status == WorkstreamStatus.LOCAL_ONLY],
        stale_or_archive_candidates=[
            row
            for row in workstreams
            if row.status in {WorkstreamStatus.PARKED, WorkstreamStatus.STALE}
        ],
        github_actions_skipped=[
            GitHubAction(
                action="sync",
                target="GitHub Cockpit issue and dated PR",
                reason="Local-only cockpit refresh; GitHub mutation disabled.",
            )
        ],
        verification_notes=_verification_notes(evidence.dirty_paths),
        evidence_gaps=_evidence_gaps(evidence.required_docs),
    )
    register_path = repo_root / "docs" / "agents" / "workstreams.md"
    packet_path = repo_root / "docs" / "agents" / "cockpit" / f"{run_date.isoformat()}.md"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    register_path.write_text(render_workstream_register(workstreams, run_date), encoding="utf-8")
    packet_path.write_text(render_cockpit_packet(report), encoding="utf-8")
    return CockpitRunResult(
        register_path=register_path,
        packet_path=packet_path,
        color=color,
        dirty_paths=evidence.dirty_paths,
        report=report,
    )


def run_github_cockpit_refresh(
    repo_root: Path,
    run_date: date,
    run_command: RunCommand | None = None,
) -> CockpitRunResult:
    runner = run_command or _run_command(repo_root)
    runner(["git", "switch", "-C", cockpit_branch_name(run_date)])
    result = run_local_cockpit_refresh(repo_root, run_date, run_command=runner)
    github = sync_github(
        enabled=True,
        repo_root=repo_root,
        run_date=run_date,
        run_color=result.color.value,
        decision_queue=_decision_queue(result.report),
        run_command=runner,
    )
    return CockpitRunResult(
        register_path=result.register_path,
        packet_path=result.packet_path,
        color=result.color,
        dirty_paths=result.dirty_paths,
        report=result.report,
        github=github,
    )


def _seed_workstreams(run_date: date) -> list[Workstream]:
    return [
        Workstream(
            name=name,
            status=status,
            tracker="" if name != "Issue #8 volatility targeting" else "GitHub issue #8",
            continuation="See latest branch, worktree, issue, or handoff evidence.",
            source_of_truth="AGENTS.md; docs/agents/domain.md; docs/research/README.md",
            latest_artifact="Initial cockpit seed from approved design.",
            last_verification="Local cockpit generation.",
            blocked_on="",
            next_action=next_action,
            owner="Codex" if status != WorkstreamStatus.NEEDS_USER_DECISION else "User",
            last_reviewed=run_date,
        )
        for name, status, next_action in INITIAL_WORKSTREAMS
    ]


def _executive_summary(dirty_paths: list[str]) -> str:
    if not dirty_paths:
        return "Local cockpit artifacts generated from seeded workstreams; GitHub sync skipped."
    return "Local cockpit artifacts generated, but dirty paths need review before any commit."


def _dirty_state_decision() -> Decision:
    return Decision(
        workstream="Git and worktree hygiene",
        question="Decide whether dirty non-cockpit paths should be parked, committed separately, or ignored.",
        options="Park, separate commit, or leave untouched.",
    )


def _verification_notes(dirty_paths: list[str]) -> list[str]:
    notes = ["Generated local Markdown artifacts only."]
    if dirty_paths:
        notes.append(f"Dirty paths before cockpit write: {', '.join(dirty_paths)}")
    return notes


def _evidence_gaps(required_docs: dict[str, bool]) -> list[str]:
    gaps = [f"Missing required doc: {path}" for path, exists in required_docs.items() if not exists]
    gaps.append("No live GitHub issue or PR scan in local-only mode.")
    return gaps


def _decision_queue(report: CockpitReport) -> list[str]:
    return [f"{decision.workstream}: {decision.question}" for decision in report.decisions]


def _run_command(repo_root: Path) -> RunCommand:
    def run(args: list[str]) -> str:
        completed = subprocess.run(args, cwd=repo_root, check=True, capture_output=True, text=True)
        return completed.stdout

    return run
