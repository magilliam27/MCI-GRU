from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING, Protocol

from cockpit.decisions import (
    DECISION_REGISTRY_PATH,
    DecisionRegistry,
    SurfaceDecision,
    SurfaceDisposition,
    WorkstreamDecision,
    load_decision_registry,
    read_registry_aliases,
    read_registry_workstream_names,
)
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
    from collections.abc import Sequence
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


class WorkstreamSource(Protocol):
    """Propose candidate workstream seeds from the evidence a source can observe.

    Sources only *propose* identities; the recorded decision registry disposes of
    status, canonical surface, next action, and last-reviewed date through
    ``_resolve_workstream``. A source never gets the last word on disposition.
    """

    def provide(self, evidence: LocalEvidence, run_date: date) -> list[WorkstreamSeed]: ...


RESERVED_SURFACE_PREFIX = "Git surface: "

# How many days of recent git activity (branch committer date / handoff filename
# date) GitActivitySource considers when deriving workstream seeds.
GIT_ACTIVITY_LOOKBACK_DAYS = 14

# Process-meta phrases that must never become fake workstreams. They are split
# into individual tokens; any token appearing here is dropped before naming, and
# a topic whose surviving tokens are empty contributes no seed. This stays a code
# constant (not part of the JSON contract) because it is mechanical extraction
# noise, not user-facing curation.
GIT_ACTIVITY_STOPWORD_PHRASES = (
    "cockpit-refresh",
    "salvage",
    "pr-repair",
    "worktree-snapshot",
    "registry-closeout",
    "decision-closeout",
    "ci-repair",
)
GIT_ACTIVITY_STOPWORDS = frozenset(
    token for phrase in GIT_ACTIVITY_STOPWORD_PHRASES for token in phrase.split("-")
)

# Integration branches are not topic branches: deriving a workstream from them
# would create a permanent bogus row (e.g. "Main") that can never be retired.
GIT_ACTIVITY_EXCLUDED_BRANCHES = frozenset({"main", "master", "head", "(no branch)"})

# Trailing ``-YYYYMMDD`` or ``-YYYY-MM-DD`` date segment on a branch name.
_BRANCH_DATE_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")
# Trailing short-hex-hash segment (4+ hex chars containing at least one digit,
# e.g. ``-040a`` or ``-8937``). Requiring a digit avoids stripping real words that
# happen to be all-hex letters (``beef``, ``cafe``).
_BRANCH_HASH_SUFFIX = re.compile(r"-(?=[0-9a-f]*[0-9])[0-9a-f]{4,}$")
# ``YYYY-MM-DD-<slug>.md`` handoff filename shape.
_HANDOFF_FILENAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.md$")

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
]


GIT_HYGIENE_SEED = WorkstreamSeed(
    name="Git and worktree hygiene",
    status=WorkstreamStatus.ACTIVE,
    next_action="Review branch/worktree attention items before continuing implementation work.",
    branch_terms=("cockpit", "hygiene", "ruff-format"),
)


@dataclass(frozen=True)
class StaticWorkstreamSource:
    """Propose the hardcoded seed list in its declared order.

    The hygiene seed is intentionally excluded: it is not owned by any source and
    is appended last inside ``_resolve_workstreams``.
    """

    seeds: tuple[WorkstreamSeed, ...] = tuple(INITIAL_WORKSTREAMS)

    def provide(self, evidence: LocalEvidence, run_date: date) -> list[WorkstreamSeed]:
        return list(self.seeds)


@dataclass(frozen=True)
class RegistryWorkstreamSource:
    """Propose one seed per workstream declared in the decision registry.

    The seed fields are neutral placeholders: the registry's recorded status and
    next action override them during ``_resolve_workstream``, so these defaults
    only matter for a registry-declared workstream that never resolves against a
    live surface or recorded decision. ``names`` lets the caller inject the
    registry keys it has already read so the file is not parsed a second time;
    when it is ``None`` the source reads the registry itself.
    """

    names: tuple[str, ...] | None = None

    def provide(self, evidence: LocalEvidence, run_date: date) -> list[WorkstreamSeed]:
        names = self.names
        if names is None:
            names = tuple(sorted(read_registry_workstream_names(evidence.repo_root)))
        return [
            WorkstreamSeed(
                name=name,
                status=WorkstreamStatus.ACTIVE,
                next_action="Continue per the recorded registry decision.",
                branch_terms=(),
            )
            for name in names
        ]


@dataclass(frozen=True)
class GitActivitySource:
    """Derive workstream seeds from recent branch names and handoff filenames.

    Branch committer dates come from ``evidence.recent_branches`` and handoff dates
    from the ``YYYY-MM-DD`` filename prefix in ``evidence.recent_handoffs``. Only
    activity within ``lookback_days`` of ``run_date`` is considered. Topic tokens
    are extracted from each name, stopword tokens are dropped, and the surviving
    tokens are resolved through the registry alias map (token or full-slug -> the
    canonical workstream name). Unaliased topics become a title-cased seed so
    genuinely new work auto-appears. Every emitted seed is ``ACTIVE`` (never
    ``NEEDS_USER_DECISION``) and carries its surviving tokens as ``branch_terms``
    so live-topology suppression does not hide the row.

    ``aliases`` lets callers inject the alias map for testing; when ``None`` the
    source reads it defensively from the registry via ``read_registry_aliases``.
    """

    aliases: dict[str, str] | None = None
    lookback_days: int = GIT_ACTIVITY_LOOKBACK_DAYS

    def provide(self, evidence: LocalEvidence, run_date: date) -> list[WorkstreamSeed]:
        aliases = self.aliases
        if aliases is None:
            aliases = read_registry_aliases(evidence.repo_root)
        token_lists: list[list[str]] = []
        for name, committer_date in evidence.recent_branches:
            if name.strip().lower() in GIT_ACTIVITY_EXCLUDED_BRANCHES:
                continue
            if self._within_lookback(committer_date, run_date):
                token_lists.append(_branch_topic_tokens(name))
        for handoff in evidence.recent_handoffs:
            handoff_date, tokens = _handoff_topic_tokens(handoff)
            if handoff_date is not None and self._within_lookback(handoff_date, run_date):
                token_lists.append(tokens)
        return _seeds_from_token_lists(token_lists, aliases)

    def _within_lookback(self, activity_date: date, run_date: date) -> bool:
        delta = (run_date - activity_date).days
        return 0 <= delta <= self.lookback_days


def _branch_topic_tokens(branch: str) -> list[str]:
    """Extract topic tokens from a branch name.

    Strips a leading ``codex/`` or ``cursor/`` prefix and any trailing date or
    short-hash segment, then splits the remainder on hyphens into lowercase
    tokens.
    """
    slug = branch.strip()
    for prefix in ("codex/", "cursor/"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
            break
    slug = slug.lower()
    slug = _BRANCH_DATE_SUFFIX.sub("", slug)
    slug = _BRANCH_HASH_SUFFIX.sub("", slug)
    return [token for token in slug.split("-") if token]


def _handoff_topic_tokens(handoff: str) -> tuple[date | None, list[str]]:
    """Extract the date and topic tokens from a handoff path or filename.

    Returns ``(None, [])`` when the filename does not match the
    ``YYYY-MM-DD-<slug>.md`` shape or the date prefix is unparseable.
    """
    filename = handoff.rsplit("/", 1)[-1]
    match = _HANDOFF_FILENAME.match(filename)
    if match is None:
        return None, []
    try:
        handoff_date = date.fromisoformat(match.group(1))
    except ValueError:
        return None, []
    tokens = [token for token in match.group(2).lower().split("-") if token]
    return handoff_date, tokens


def _seeds_from_token_lists(
    token_lists: list[list[str]],
    aliases: dict[str, str],
) -> list[WorkstreamSeed]:
    """Turn extracted token lists into deterministic, collapsed ACTIVE seeds.

    Stopword tokens are removed; a topic whose surviving tokens are empty is
    skipped. Surviving tokens resolve through ``aliases`` to a canonical name, or
    become a title-cased join otherwise. Topics that map to the same name collapse
    into one seed whose ``branch_terms`` is the sorted union of their tokens.
    """
    terms_by_name: dict[str, set[str]] = {}
    for tokens in token_lists:
        survivors = [token for token in tokens if token not in GIT_ACTIVITY_STOPWORDS]
        if not survivors:
            continue
        name = _resolve_seed_name(survivors, aliases)
        terms_by_name.setdefault(name, set()).update(survivors)
    seeds: list[WorkstreamSeed] = []
    for name in sorted(terms_by_name):
        seeds.append(
            WorkstreamSeed(
                name=name,
                status=WorkstreamStatus.ACTIVE,
                next_action="Confirm this git-derived topic is a real workstream or retire it.",
                branch_terms=tuple(sorted(terms_by_name[name])),
            )
        )
    return seeds


def _resolve_seed_name(survivors: list[str], aliases: dict[str, str]) -> str:
    """Resolve surviving tokens to a canonical alias name or a title-cased join.

    A full-slug alias match (the hyphen-joined surviving tokens) is preferred,
    then the first surviving token that is an alias key, then a title-cased,
    space-joined fallback that lets new topics auto-appear.
    """
    joined = "-".join(survivors)
    if joined in aliases:
        return aliases[joined]
    for token in survivors:
        if token in aliases:
            return aliases[token]
    return " ".join(token.title() for token in survivors)


def merge_workstream_sources(
    sources: Sequence[WorkstreamSource],
    evidence: LocalEvidence,
    run_date: date,
) -> list[WorkstreamSeed]:
    """Combine source proposals into a deterministic, deduped seed list.

    Each source's ``provide`` is called in list order and the first seed to claim
    a name wins, so earlier sources take precedence. Two name classes are dropped
    rather than raised on, because a daily automated refresh must not crash over a
    misbehaving source:

    * the reserved ``"Git surface: "`` prefix, which run color and the decision
      queue use to identify unclassified topology surfaces; and
    * the static ``"Git and worktree hygiene"`` seed, which is appended last
      inside ``_resolve_workstreams`` and is not owned by any source.

    First-seen order is preserved, so a static source keeps its declared order and
    later sources contribute only their new names in the order they propose them.
    """
    merged: list[WorkstreamSeed] = []
    seen: set[str] = set()
    for source in sources:
        for seed in source.provide(evidence, run_date):
            if seed.name.startswith(RESERVED_SURFACE_PREFIX):
                continue
            if seed.name == GIT_HYGIENE_SEED.name:
                continue
            if seed.name in seen:
                continue
            seen.add(seed.name)
            merged.append(seed)
    return merged


def run_local_cockpit_refresh(
    repo_root: Path,
    run_date: date,
    run_command: RunCommand | None = None,
    *,
    github_sync_enabled: bool = False,
    git_snapshot_timing: str = "at cockpit evidence collection",
    sources: Sequence[WorkstreamSource] | None = None,
) -> CockpitRunResult:
    evidence = collect_local_evidence(repo_root, run_command=run_command)
    registry_names = read_registry_workstream_names(repo_root)
    if sources is None:
        sources = (
            StaticWorkstreamSource(),
            RegistryWorkstreamSource(names=tuple(sorted(registry_names))),
            GitActivitySource(),
        )
    seeds = [*merge_workstream_sources(sources, evidence, run_date), GIT_HYGIENE_SEED]
    registry = load_decision_registry(
        repo_root,
        known_workstreams={seed.name for seed in seeds} | registry_names,
    )
    workstreams = _resolve_workstreams(evidence, run_date, registry, seeds)
    color = _run_color(evidence, workstreams)
    report = CockpitReport(
        run_date=run_date,
        color=color,
        executive_summary=_executive_summary(evidence, workstreams),
        decisions=_decisions(evidence, color, workstreams),
        decision_workstreams=_decision_workstreams(workstreams),
        active_workstreams=[row for row in workstreams if row.status == WorkstreamStatus.ACTIVE],
        blocked_workstreams=[row for row in workstreams if row.status == WorkstreamStatus.BLOCKED],
        local_only_work=[row for row in workstreams if row.status == WorkstreamStatus.LOCAL_ONLY],
        stale_or_archive_candidates=[
            row
            for row in workstreams
            if row.status
            in {WorkstreamStatus.PARKED, WorkstreamStatus.STALE, WorkstreamStatus.ARCHIVE}
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


def _resolve_workstreams(
    evidence: LocalEvidence,
    run_date: date,
    registry: DecisionRegistry,
    seeds: list[WorkstreamSeed],
) -> list[Workstream]:
    topology = evidence.git_topology
    surfaces = _topology_surfaces(topology)
    live_topology = bool(surfaces) or topology.has_attention_items
    rows: list[Workstream] = []
    claimed: set[str] = set()
    hygiene_seed: WorkstreamSeed | None = None
    for seed in seeds:
        if seed.name == "Git and worktree hygiene":
            hygiene_seed = seed
            continue
        matches = _workstream_surfaces(seed, surfaces, registry)
        decision = registry.workstreams.get(seed.name)
        if matches or decision is not None:
            rows.append(
                _resolve_workstream(
                    seed,
                    evidence,
                    run_date,
                    matches,
                    decision=decision,
                    registry=registry,
                )
            )
            claimed.update(surface.branch for surface in matches)
        elif not live_topology:
            rows.append(
                _resolve_workstream(
                    seed,
                    evidence,
                    run_date,
                    matches,
                    decision=None,
                    registry=registry,
                )
            )
    for surface in surfaces:
        surface_decision = registry.surfaces.get(surface.branch)
        if surface.branch not in claimed or (
            surface_decision is not None
            and surface_decision.disposition != SurfaceDisposition.CANONICAL
        ):
            rows.append(
                _topology_surface_workstream(
                    surface,
                    evidence,
                    run_date,
                    decision=surface_decision,
                )
            )
    if hygiene_seed is not None:
        rows.append(_git_hygiene_workstream(hygiene_seed, topology, run_date))
    return rows


def _resolve_workstream(
    seed: WorkstreamSeed,
    evidence: LocalEvidence,
    run_date: date,
    surfaces: list[TopologySurface],
    *,
    decision: WorkstreamDecision | None,
    registry: DecisionRegistry,
) -> Workstream:
    status = seed.status
    blocked_on = ""
    continuation = "No matching branch/worktree in this snapshot; continue from tracker/docs before starting new work."
    next_action = seed.next_action
    last_reviewed = run_date
    source_of_truth = "AGENTS.md; docs/agents/domain.md; docs/research/README.md"
    if decision is not None:
        unreviewed = [
            surface for surface in surfaces if not registry.is_reviewed(seed.name, surface.branch)
        ]
        status = decision.status
        continuation = decision.canonical_surface
        next_action = decision.next_action
        last_reviewed = decision.last_reviewed
        source_of_truth = f"{DECISION_REGISTRY_PATH}; {source_of_truth}"
        if unreviewed:
            status = WorkstreamStatus.NEEDS_USER_DECISION
            blocked_on = (
                f"New unreviewed surfaces since the {decision.last_reviewed.isoformat()} decision: "
                + "; ".join(surface.label for surface in unreviewed)
            )
            next_action = "Review only the new surface(s) against the recorded canonical decision."
            last_reviewed = run_date
    elif len(surfaces) == 1:
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
        source_of_truth=source_of_truth,
        latest_artifact=_latest_artifact(evidence),
        last_verification=_git_verification_label(),
        blocked_on=blocked_on,
        next_action=next_action,
        owner="User" if status == WorkstreamStatus.NEEDS_USER_DECISION else "Codex",
        last_reviewed=last_reviewed,
    )


def _topology_surface_workstream(
    surface: TopologySurface,
    evidence: LocalEvidence,
    run_date: date,
    *,
    decision: SurfaceDecision | None,
) -> Workstream:
    if decision is not None:
        status = _surface_status(decision.disposition)
        return Workstream(
            name=f"Git surface: {surface.branch}",
            status=status,
            tracker="",
            continuation=surface.label,
            source_of_truth=DECISION_REGISTRY_PATH,
            latest_artifact=decision.reason,
            last_verification=_git_verification_label(),
            blocked_on="",
            next_action=decision.next_action,
            owner=(
                "User" if status in {WorkstreamStatus.ARCHIVE, WorkstreamStatus.STALE} else "Codex"
            ),
            last_reviewed=decision.last_reviewed,
        )
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


def _surface_status(disposition: SurfaceDisposition) -> WorkstreamStatus:
    return {
        SurfaceDisposition.CANONICAL: WorkstreamStatus.ACTIVE,
        SurfaceDisposition.PARKED: WorkstreamStatus.PARKED,
        SurfaceDisposition.ARCHIVE: WorkstreamStatus.ARCHIVE,
        SurfaceDisposition.STALE: WorkstreamStatus.STALE,
    }[disposition]


def _workstream_surfaces(
    seed: WorkstreamSeed,
    surfaces: list[TopologySurface],
    registry: DecisionRegistry,
) -> list[TopologySurface]:
    heuristic_matches = _matching_surface_entries(seed.branch_terms, surfaces)
    matched = {surface.branch: surface for surface in heuristic_matches}
    for surface in surfaces:
        decision = registry.surfaces.get(surface.branch)
        if decision is not None and seed.name in decision.workstreams:
            matched[surface.branch] = surface
    return list(matched.values())


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
    decisions.extend(
        _workstream_decision(workstream) for workstream in _decision_workstreams(workstreams)
    )
    topology = evidence.git_topology
    unresolved_surfaces = _classification_surface_count(workstreams)
    if color != RunColor.GREEN and (
        topology.origin_main_ahead
        or topology.origin_main_behind
        or topology.dirty_worktrees
        or unresolved_surfaces
    ):
        decisions.append(
            Decision(
                workstream="Git and worktree hygiene",
                question="Review branch/worktree attention items before choosing the next continuation surface.",
                options="Continue current branch, park dirty work, close stale surfaces, or choose a canonical branch.",
            )
        )
    return decisions


def _decision_workstreams(workstreams: list[Workstream]) -> list[Workstream]:
    return [
        workstream
        for workstream in workstreams
        if workstream.status == WorkstreamStatus.NEEDS_USER_DECISION
        and not workstream.name.startswith("Git surface: ")
    ]


def _workstream_decision(workstream: Workstream) -> Decision:
    return Decision(
        workstream=workstream.name,
        question=workstream.next_action,
        options="Continue, park, archive, or choose a canonical branch/worktree.",
    )


def _dirty_state_decision() -> Decision:
    return Decision(
        workstream="Git and worktree hygiene",
        question="Decide whether dirty non-cockpit paths should be parked, committed separately, or ignored.",
        options="Park, separate commit, or leave untouched.",
    )


def _classification_surface_count(workstreams: list[Workstream]) -> int:
    return sum(
        1
        for workstream in workstreams
        if workstream.name.startswith("Git surface: ")
        and workstream.status == WorkstreamStatus.NEEDS_USER_DECISION
    )


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
