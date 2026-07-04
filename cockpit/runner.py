from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from cockpit.evidence import collect_local_evidence
from cockpit.git import with_safe_directory
from cockpit.github import GitHubSyncResult, cockpit_branch_name, sync_github
from cockpit.models import (
    BranchEvidence,
    CockpitReport,
    Decision,
    GitHubAction,
    GitTopologySnapshot,
    RunColor,
    Workstream,
    WorkstreamStatus,
    WorktreeEvidence,
)
from cockpit.render import render_cockpit_packet, render_workstream_register

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from cockpit.evidence import LocalEvidence, RunCommand


@dataclass(frozen=True)
class CockpitRunResult:
    register_path: Path
    packet_path: Path
    color: RunColor
    dirty_paths: list[str]
    report: CockpitReport
    github: GitHubSyncResult | None = None


@dataclass(frozen=True)
class WorkstreamSeed:
    name: str
    status: WorkstreamStatus
    next_action: str
    tracker: str = ""
    branch_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopologySurface:
    branch: str
    label: str
    provenance: str


INITIAL_WORKSTREAMS = [
    WorkstreamSeed(
        name="LambdaRankIC",
        status=WorkstreamStatus.ACTIVE,
        next_action="Review lower-pair screen and loss optimization follow-ups.",
        branch_terms=("lambdarank", "lambda-rank", "top10"),
    ),
    WorkstreamSeed(
        name="Portfolio-IC",
        status=WorkstreamStatus.NEEDS_USER_DECISION,
        next_action="Decide whether to promote, park, or rerun current evidence.",
        branch_terms=("portfolio", "portfolio-ic"),
    ),
    WorkstreamSeed(
        name="Issue #8 volatility targeting",
        status=WorkstreamStatus.PARKED,
        next_action="Resume from the issue workflow when user prioritizes it.",
        tracker="GitHub issue #8",
        branch_terms=("issue8", "volatility", "vol-"),
    ),
    WorkstreamSeed(
        name="Colab operations",
        status=WorkstreamStatus.READY_FOR_AGENT,
        next_action="Use the Chrome-control runbook for the next live Colab smoke.",
        branch_terms=("colab",),
    ),
    WorkstreamSeed(
        name="Regime CSV contract",
        status=WorkstreamStatus.PARKED,
        next_action="Keep no-lookahead contract tests as the source of truth.",
        branch_terms=("regime", "csv"),
    ),
    WorkstreamSeed(
        name="LSEG access",
        status=WorkstreamStatus.BLOCKED,
        next_action="Refresh access probe only when data access is needed.",
        branch_terms=("lseg",),
    ),
    WorkstreamSeed(
        name="Daily bug scans",
        status=WorkstreamStatus.READY_FOR_AGENT,
        next_action="Collapse repeated no-op scans unless a distinct regression appears.",
        branch_terms=("daily-bug", "bug-scan"),
    ),
    WorkstreamSeed(
        name="Docs and research evidence",
        status=WorkstreamStatus.ACTIVE,
        next_action="Keep docs/research/README.md as the evidence map.",
        branch_terms=("evidence", "research", "docs"),
    ),
    WorkstreamSeed(
        name="Git and worktree hygiene",
        status=WorkstreamStatus.ACTIVE,
        next_action="Review branch/worktree attention items before continuing implementation work.",
        branch_terms=("cockpit", "hygiene", "ruff-format"),
    ),
]


def run_local_cockpit_refresh(
    repo_root: Path,
    run_date: date,
    run_command: RunCommand | None = None,
    *,
    github_sync_enabled: bool = False,
    git_snapshot_timing: str = "at cockpit evidence collection",
) -> CockpitRunResult:
    evidence = collect_local_evidence(repo_root, run_command=run_command)
    workstreams = _resolve_workstreams(evidence, run_date)
    color = _run_color(evidence, workstreams)
    report = CockpitReport(
        run_date=run_date,
        color=color,
        executive_summary=_executive_summary(evidence, workstreams),
        decisions=_decisions(evidence, color, workstreams),
        active_workstreams=[row for row in workstreams if row.status == WorkstreamStatus.ACTIVE],
        blocked_workstreams=[row for row in workstreams if row.status == WorkstreamStatus.BLOCKED],
        local_only_work=[row for row in workstreams if row.status == WorkstreamStatus.LOCAL_ONLY],
        stale_or_archive_candidates=[
            row
            for row in workstreams
            if row.status in {WorkstreamStatus.PARKED, WorkstreamStatus.STALE}
        ],
        github_actions_skipped=_github_actions_skipped(github_sync_enabled),
        git_tree_impact=_git_tree_impact(evidence.git_topology, git_snapshot_timing),
        verification_notes=_verification_notes(evidence, git_snapshot_timing),
        evidence_gaps=_evidence_gaps(evidence.required_docs, github_sync_enabled),
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
    result = run_local_cockpit_refresh(
        repo_root,
        run_date,
        run_command=runner,
        github_sync_enabled=True,
        git_snapshot_timing="before GitHub sync commits/pushes",
    )
    github = sync_github(
        enabled=True,
        repo_root=repo_root,
        run_date=run_date,
        run_color=result.color.value,
        decision_queue=_decision_queue(result.report),
        run_command=runner,
    )
    report = _report_with_github_result(result.report, github)
    result.packet_path.write_text(render_cockpit_packet(report), encoding="utf-8")
    _commit_synced_packet(run_date, runner)
    return CockpitRunResult(
        register_path=result.register_path,
        packet_path=result.packet_path,
        color=result.color,
        dirty_paths=result.dirty_paths,
        report=report,
        github=github,
    )


def _resolve_workstreams(evidence: LocalEvidence, run_date: date) -> list[Workstream]:
    topology = evidence.git_topology
    surfaces = _topology_surfaces(topology)
    live_topology = bool(surfaces) or topology.has_attention_items
    rows: list[Workstream] = []
    claimed: set[str] = set()
    hygiene_seed: WorkstreamSeed | None = None
    for seed in INITIAL_WORKSTREAMS:
        if seed.name == "Git and worktree hygiene":
            hygiene_seed = seed
            continue
        matches = _matching_surface_entries(seed.branch_terms, surfaces)
        if matches:
            rows.append(_resolve_workstream(seed, evidence, run_date, matches))
            claimed.update(surface.branch for surface in matches)
        elif not live_topology:
            rows.append(_resolve_workstream(seed, evidence, run_date, matches))
    rows.extend(
        _topology_surface_workstream(surface, evidence, run_date)
        for surface in surfaces
        if surface.branch not in claimed
    )
    if hygiene_seed is not None:
        rows.append(_git_hygiene_workstream(hygiene_seed, topology, run_date))
    return rows


def _resolve_workstream(
    seed: WorkstreamSeed,
    evidence: LocalEvidence,
    run_date: date,
    surfaces: list[TopologySurface],
) -> Workstream:
    status = seed.status
    blocked_on = ""
    continuation = "No matching branch/worktree in this snapshot; continue from tracker/docs before starting new work."
    next_action = seed.next_action
    if len(surfaces) == 1:
        continuation = surfaces[0].label
    elif len(surfaces) > 1:
        status = WorkstreamStatus.NEEDS_USER_DECISION
        continuation = "needs-user-decision"
        blocked_on = "Competing surfaces: " + "; ".join(surface.label for surface in surfaces)
        next_action = "Choose the canonical continuation surface before continuing this workstream."
    return Workstream(
        name=seed.name,
        status=status,
        tracker=seed.tracker,
        continuation=continuation,
        source_of_truth="AGENTS.md; docs/agents/domain.md; docs/research/README.md",
        latest_artifact=_latest_artifact(evidence),
        last_verification=_git_verification_label(),
        blocked_on=blocked_on,
        next_action=next_action,
        owner="User" if status == WorkstreamStatus.NEEDS_USER_DECISION else "Codex",
        last_reviewed=run_date,
    )


def _topology_surface_workstream(
    surface: TopologySurface,
    evidence: LocalEvidence,
    run_date: date,
) -> Workstream:
    return Workstream(
        name=f"Git surface: {surface.branch}",
        status=WorkstreamStatus.NEEDS_USER_DECISION,
        tracker="",
        continuation=surface.label,
        source_of_truth="git branch --all --no-merged origin/main; git worktree list --porcelain",
        latest_artifact=_git_topology_summary(evidence.git_topology),
        last_verification=_git_verification_label(),
        blocked_on="",
        next_action="Classify this live git surface into a workstream, park it, merge it, or close it.",
        owner="User",
        last_reviewed=run_date,
    )


def _git_hygiene_workstream(
    seed: WorkstreamSeed, topology: GitTopologySnapshot, run_date: date
) -> Workstream:
    dirty = topology.dirty_worktrees
    detached = topology.detached_worktrees
    blocked_parts = []
    if dirty:
        blocked_parts.append(
            "Dirty worktrees: " + ", ".join(_worktree_label(worktree) for worktree in dirty)
        )
    if detached:
        blocked_parts.append(
            "Detached worktrees: " + ", ".join(_worktree_label(worktree) for worktree in detached)
        )
    return Workstream(
        name=seed.name,
        status=seed.status,
        tracker=seed.tracker,
        continuation=(
            f"Current branch `{topology.current_branch}`; "
            f"{len(topology.worktrees)} worktree(s); {len(topology.unmerged_branches)} unmerged branch(es)."
        ),
        source_of_truth=(
            "git status --short --branch; git worktree list --porcelain; "
            "git branch --all --no-merged origin/main; "
            "git rev-list --left-right --count origin/main...HEAD"
        ),
        latest_artifact=_git_topology_summary(topology),
        last_verification=_git_verification_label(),
        blocked_on="; ".join(blocked_parts),
        next_action=seed.next_action,
        owner="Codex",
        last_reviewed=run_date,
    )


def _topology_surfaces(topology: GitTopologySnapshot) -> list[TopologySurface]:
    surfaces: list[TopologySurface] = []
    seen: set[str] = set()
    details_by_name = {branch.name: branch for branch in topology.unmerged_branch_details}
    worktrees_by_branch: dict[str, list[WorktreeEvidence]] = {}
    for worktree in topology.worktrees:
        worktrees_by_branch.setdefault(worktree.branch, []).append(worktree)
    candidate_branches = [
        branch
        for branch in topology.branches
        if _is_live_branch(branch, topology.current_branch, details_by_name)
    ]
    if _is_live_branch(topology.current_branch, topology.current_branch, details_by_name):
        candidate_branches.insert(0, topology.current_branch)
    for branch in candidate_branches:
        if branch in seen:
            continue
        surfaces.append(
            _surface_from_branch(
                branch,
                details_by_name.get(branch),
                worktrees_by_branch.get(branch, []),
            )
        )
        seen.add(branch)
    for worktree in topology.worktrees:
        if worktree.branch in seen or not _is_live_worktree(worktree):
            continue
        surfaces.append(_surface_from_worktree(worktree, details_by_name.get(worktree.branch)))
        seen.add(worktree.branch)
    for branch in topology.unmerged_branch_details:
        if branch.name in seen:
            continue
        surfaces.append(
            _surface_from_branch(branch.name, branch, worktrees_by_branch.get(branch.name, []))
        )
        seen.add(branch.name)
    return surfaces


def _is_live_branch(
    branch: str,
    current_branch: str,
    unmerged: dict[str, BranchEvidence],
) -> bool:
    if branch in {"main", "master", "HEAD", "unknown"}:
        return False
    if branch.startswith("HEAD ") or branch.startswith("detached@"):
        return False
    return branch == current_branch or branch in unmerged or branch.startswith("codex/")


def _is_live_worktree(worktree: WorktreeEvidence) -> bool:
    return worktree.branch not in {"main", "master", "HEAD", "unknown"} or worktree.is_dirty


def _surface_from_branch(
    branch: str,
    detail: BranchEvidence | None,
    worktrees: list[WorktreeEvidence],
) -> TopologySurface:
    provenance = detail.provenance_label if detail else "local"
    label = _branch_label(detail or BranchEvidence(name=branch, local=True))
    worktree_labels = [_worktree_path_label(worktree) for worktree in worktrees]
    if worktree_labels:
        label = f"{label} @ {', '.join(worktree_labels)}"
    return TopologySurface(branch=branch, label=label, provenance=provenance)


def _surface_from_worktree(
    worktree: WorktreeEvidence,
    detail: BranchEvidence | None,
) -> TopologySurface:
    branch = worktree.branch
    if worktree.detached:
        return TopologySurface(
            branch=branch,
            label=f"`{branch}` @ {_worktree_path_label(worktree)} (detached)",
            provenance="detached",
        )
    provenance = detail.provenance_label if detail else "worktree"
    label = _branch_label(detail or BranchEvidence(name=branch, local=True))
    return TopologySurface(
        branch=branch,
        label=f"{label} @ {_worktree_path_label(worktree)}",
        provenance=provenance,
    )


def _matching_surface_entries(
    terms: tuple[str, ...],
    surfaces: list[TopologySurface],
) -> list[TopologySurface]:
    if not terms:
        return []
    matches: list[TopologySurface] = []
    seen: set[str] = set()
    for surface in surfaces:
        if not _matches_terms(surface.branch, terms) or surface.branch in seen:
            continue
        matches.append(surface)
        seen.add(surface.branch)
    return matches


def _matches_terms(value: str, terms: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in terms)


def _worktree_path_label(worktree: WorktreeEvidence) -> str:
    return f"`{worktree.path}`"


def _branch_label(branch: BranchEvidence) -> str:
    return f"`{branch.display_name}` ({branch.provenance_label})"


def _worktree_label(worktree: WorktreeEvidence) -> str:
    return f"{worktree.branch} at {worktree.path}"


def _latest_artifact(evidence: LocalEvidence) -> str:
    if evidence.recent_handoffs:
        return evidence.recent_handoffs[0]
    return _git_topology_summary(evidence.git_topology)


def _git_verification_label() -> str:
    return (
        "git status --short --branch; git worktree list --porcelain; "
        "git branch --all --no-merged origin/main; "
        "git rev-list --left-right --count origin/main...HEAD"
    )


def _git_topology_summary(topology: GitTopologySnapshot) -> str:
    return (
        f"Git topology snapshot: {len(topology.unmerged_branches)} unmerged branch(es), "
        f"{len(topology.worktrees)} worktree(s), {len(topology.detached_worktrees)} detached, "
        f"{len(topology.dirty_worktrees)} dirty."
    )


def _run_color(evidence: LocalEvidence, workstreams: list[Workstream]) -> RunColor:
    if evidence.dirty_paths or _missing_required_docs(evidence.required_docs):
        return RunColor.RED
    if evidence.git_topology.has_attention_items or _classification_surface_count(workstreams):
        return RunColor.YELLOW
    return RunColor.GREEN


def _executive_summary(evidence: LocalEvidence, workstreams: list[Workstream]) -> str:
    topology = evidence.git_topology
    if evidence.dirty_paths:
        return "Cockpit generated, but the current checkout has dirty paths that need review before any commit."
    missing_docs = _missing_required_doc_paths(evidence.required_docs)
    if missing_docs:
        return "Cockpit generated, but required docs are missing: " + ", ".join(missing_docs)
    live_surface_count = _classification_surface_count(workstreams)
    if topology.has_attention_items or live_surface_count:
        live_surface_clause = (
            f", {live_surface_count} live git surface(s) needing classification"
            if live_surface_count
            else ""
        )
        return (
            "Cockpit generated with git topology attention items: "
            f"{len(topology.unmerged_branches)} unmerged branch(es), "
            f"{len(topology.detached_worktrees)} detached worktree(s), "
            f"{len(topology.dirty_worktrees)} dirty worktree(s)"
            f"{live_surface_clause}."
        )
    return "Cockpit generated from current git topology; no branch/worktree attention items found."


def _decisions(
    evidence: LocalEvidence,
    color: RunColor,
    workstreams: list[Workstream],
) -> list[Decision]:
    decisions: list[Decision] = []
    if evidence.dirty_paths:
        decisions.append(_dirty_state_decision())
    topology = evidence.git_topology
    if color != RunColor.GREEN and (
        topology.has_attention_items or _classification_surface_count(workstreams)
    ):
        decisions.append(
            Decision(
                workstream="Git and worktree hygiene",
                question="Review branch/worktree attention items before choosing the next continuation surface.",
                options="Continue current branch, park dirty work, close stale surfaces, or choose a canonical branch.",
            )
        )
    return decisions


def _dirty_state_decision() -> Decision:
    return Decision(
        workstream="Git and worktree hygiene",
        question="Decide whether dirty non-cockpit paths should be parked, committed separately, or ignored.",
        options="Park, separate commit, or leave untouched.",
    )


def _classification_surface_count(workstreams: list[Workstream]) -> int:
    return sum(1 for workstream in workstreams if workstream.name.startswith("Git surface: "))


def _verification_notes(evidence: LocalEvidence, snapshot_timing: str) -> list[str]:
    notes = [_git_verification_label(), f"Git topology snapshot timing: {snapshot_timing}"]
    if evidence.dirty_paths:
        notes.append(f"Dirty paths before cockpit write: {', '.join(evidence.dirty_paths)}")
    notes.append(_git_topology_summary(evidence.git_topology))
    return notes


def _evidence_gaps(required_docs: dict[str, bool], github_sync_enabled: bool) -> list[str]:
    gaps = [f"Missing required doc: {path}" for path, exists in required_docs.items() if not exists]
    if not github_sync_enabled:
        gaps.append("No live GitHub issue or PR scan in local-only mode.")
    return gaps


def _missing_required_docs(required_docs: dict[str, bool]) -> bool:
    return any(not exists for exists in required_docs.values())


def _missing_required_doc_paths(required_docs: dict[str, bool]) -> list[str]:
    return [path for path, exists in required_docs.items() if not exists]


def _github_actions_skipped(github_sync_enabled: bool) -> list[GitHubAction]:
    if github_sync_enabled:
        return []
    return [
        GitHubAction(
            action="sync",
            target="GitHub Cockpit issue and dated PR",
            reason="Local-only cockpit refresh; GitHub mutation disabled.",
        )
    ]


def _git_tree_impact(topology: GitTopologySnapshot, snapshot_timing: str) -> list[str]:
    return [
        f"Git topology snapshot timing: {snapshot_timing}",
        f"Current branch: `{topology.current_branch}`",
        f"origin/main divergence: {topology.origin_main_ahead} ahead / {topology.origin_main_behind} behind",
        f"Unmerged branches: {len(topology.unmerged_branches)}",
        (
            f"Worktrees: {len(topology.worktrees)} total, "
            f"{len(topology.detached_worktrees)} detached, {len(topology.dirty_worktrees)} dirty"
        ),
        _format_branch_detail_list("Unmerged branch names", topology),
        _format_worktree_list("Dirty worktrees", topology.dirty_worktrees),
        _format_worktree_list("Detached worktrees", topology.detached_worktrees),
    ]


def _format_branch_detail_list(label: str, topology: GitTopologySnapshot) -> str:
    if topology.unmerged_branch_details:
        return f"{label}: " + ", ".join(
            _branch_label(branch) for branch in topology.unmerged_branch_details
        )
    if not topology.unmerged_branches:
        return f"{label}: none"
    return f"{label}: " + ", ".join(f"`{branch}`" for branch in topology.unmerged_branches)


def _format_worktree_list(label: str, worktrees: list[WorktreeEvidence]) -> str:
    if not worktrees:
        return f"{label}: none"
    return f"{label}: " + "; ".join(
        f"`{worktree.branch}` at `{worktree.path}`" for worktree in worktrees
    )


def _report_with_github_result(report: CockpitReport, github: GitHubSyncResult) -> CockpitReport:
    return replace(
        report,
        github_actions_taken=[
            GitHubAction(
                action="sync",
                target="GitHub Cockpit issue and dated PR",
                reason=action,
            )
            for action in github.actions_taken
        ],
        github_actions_skipped=[
            GitHubAction(
                action="skip",
                target="GitHub Cockpit issue and dated PR",
                reason=action,
            )
            for action in github.actions_skipped
        ],
    )


def _commit_synced_packet(run_date: date, runner: RunCommand) -> None:
    packet_path = f"docs/agents/cockpit/{run_date.isoformat()}.md"
    if not runner(["git", "status", "--short", "--", packet_path]).strip():
        return
    runner(["git", "add", packet_path])
    runner(["git", "commit", "-m", f"Record cockpit GitHub sync for {run_date.isoformat()}"])
    runner(["git", "push", "-u", "origin", cockpit_branch_name(run_date)])


def _decision_queue(report: CockpitReport) -> list[str]:
    return [f"{decision.workstream}: {decision.question}" for decision in report.decisions]


def _run_command(repo_root: Path) -> RunCommand:
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
