from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING, Literal, Protocol

from cockpit.decisions import DecisionRegistry
from cockpit.models import (
    AutoDecisionChange,
    AutoDecisionSet,
    AutoDisposition,
    AutoWorkstreamDecision,
    Confidence,
    GitHubEvidence,
    GitTopologySnapshot,
    IssueEvidence,
    PullRequestEvidence,
    SurfaceDisposition,
    WorkstreamStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

AUTO_DECISIONS_PATH = "docs/agents/cockpit/auto-decisions.json"
AUTO_FORMAT_VERSION = 2
SUPPORTED_AUTO_FORMAT_VERSIONS = frozenset({1, 2})
DEFAULT_STALE_LOOKBACK_DAYS = 30
_HYGIENE_WORKSTREAM = "Git and worktree hygiene"
_UNKNOWN_EVIDENCE_DATE = date(1970, 1, 1)
_REMOVED_GENERATED_VALUE = "no generated decision"
_GENERIC_ISSUE_BRANCH_TERMS = frozenset(
    {
        "bug",
        "code",
        "csv",
        "data",
        "docs",
        "documentation",
        "evidence",
        "feature",
        "fix",
        "issue",
        "model",
        "research",
        "test",
        "tests",
        "update",
        "work",
    }
)
_BRANCH_DATE_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")
_BRANCH_HASH_SUFFIX = re.compile(r"-(?=[0-9a-f]*[0-9])[0-9a-f]{4,}$")


class PolicySurface(Protocol):
    branch: str
    provenance: str


class PolicyWorkstream(Protocol):
    name: str
    branch_terms: tuple[str, ...]


def compute_auto_decisions(
    *,
    surfaces: Sequence[PolicySurface],
    workstreams: Sequence[PolicyWorkstream],
    topology: GitTopologySnapshot,
    registry: DecisionRegistry,
    aliases: Mapping[str, str],
    implied_aliases: Mapping[str, str] | None = None,
    run_date: date,
    recent_branches: Sequence[tuple[str, date]],
    branch_commit_dates: Mapping[str, date] | None = None,
    github_evidence: GitHubEvidence | None = None,
    stale_lookback_days: int = DEFAULT_STALE_LOOKBACK_DAYS,
) -> AutoDecisionSet:
    """Compute deterministic decisions from local and optional GitHub evidence."""
    if stale_lookback_days < 0:
        raise ValueError("stale_lookback_days must be non-negative")

    phase2_enabled = branch_commit_dates is not None
    activity_dates = (
        dict(branch_commit_dates)
        if branch_commit_dates is not None
        else {branch: activity_date for branch, activity_date in recent_branches}
    )
    workstreams_by_name = {workstream.name: workstream for workstream in workstreams}
    surface_associations = {
        surface.branch: _surface_association(
            surface.branch,
            workstreams,
            workstreams_by_name,
            registry,
            aliases,
            implied_aliases or {},
        )
        for surface in surfaces
    }
    associations = {
        branch: association.workstreams for branch, association in surface_associations.items()
    }
    unmerged = set(topology.unmerged_branches)
    uncertain_worktrees = {
        worktree.branch: (
            "Attached worktree status unavailable; merged state cannot be trusted."
            if worktree.status_error
            else "Attached worktree is dirty; merged state cannot be trusted."
        )
        for worktree in topology.worktrees
        if not worktree.detached and worktree.is_dirty
    }
    auto_surfaces = _surface_decisions(
        surfaces=surfaces,
        associations=associations,
        unmerged=unmerged,
        real_branches=set(topology.branches) | unmerged,
        uncertain_worktrees=uncertain_worktrees,
        activity_dates=activity_dates,
        registry=registry,
        run_date=run_date,
        github_evidence=github_evidence,
        stale_lookback_days=stale_lookback_days,
        phase2_enabled=phase2_enabled,
    )
    auto_surfaces = {
        branch: _audit_surface_association(
            decision,
            surface_associations[branch],
        )
        for branch, decision in auto_surfaces.items()
    }
    auto_workstreams = _workstream_decisions(
        workstreams=workstreams,
        associations=associations,
        auto_surfaces=auto_surfaces,
        activity_dates=activity_dates,
        registry=registry,
        run_date=run_date,
        stale_lookback_days=stale_lookback_days,
        github_evidence=github_evidence,
        aliases=aliases,
        phase2_enabled=phase2_enabled,
    )
    auto_workstreams = _audit_workstream_associations(
        auto_workstreams,
        surface_associations,
        activity_dates,
    )
    return AutoDecisionSet(surfaces=auto_surfaces, workstreams=auto_workstreams)


def write_auto_decisions(repo_root: Path, decisions: AutoDecisionSet) -> Path:
    """Write generated decisions with stable ordering and no wall-clock timestamp."""
    path = repo_root / AUTO_DECISIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": AUTO_FORMAT_VERSION,
        "surfaces": {
            branch: _surface_payload(decision)
            for branch, decision in sorted(decisions.surfaces.items())
        },
        "workstreams": {
            name: _workstream_payload(decision)
            for name, decision in sorted(decisions.workstreams.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_auto_decisions(repo_root: Path) -> AutoDecisionSet:
    """Read generated decisions, treating only an absent artifact as an empty baseline."""
    path = repo_root / AUTO_DECISIONS_PATH
    if not path.exists():
        return AutoDecisionSet()
    return parse_auto_decisions_text(path.read_text(encoding="utf-8"))


def parse_auto_decisions_text(payload: str) -> AutoDecisionSet:
    """Parse present generated-decision evidence, rejecting any corrupt payload."""
    try:
        return _parse_auto_decisions(json.loads(payload))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid generated auto-decision payload") from exc


def compare_auto_decisions(
    previous: AutoDecisionSet,
    current: AutoDecisionSet,
    registry: DecisionRegistry,
    *,
    previous_registry: DecisionRegistry | None = None,
) -> list[AutoDecisionChange]:
    """Describe deterministic user-visible changes between generated decision sets."""
    baseline_registry = previous_registry or DecisionRegistry()
    changes = [
        AutoDecisionChange(
            kind="surface",
            target=branch,
            change="added",
            after=decision.disposition.value,
        )
        for branch, decision in sorted(current.surfaces.items())
        if branch not in previous.surfaces
    ]
    changes.extend(
        AutoDecisionChange(
            kind="surface",
            target=branch,
            change="removed",
            before=decision.disposition.value,
        )
        for branch, decision in sorted(previous.surfaces.items())
        if branch not in current.surfaces
    )
    changes.extend(
        AutoDecisionChange(
            kind="surface",
            target=branch,
            change="choice",
            before=previous.surfaces[branch].disposition.value,
            after=decision.disposition.value,
        )
        for branch, decision in sorted(current.surfaces.items())
        if branch in previous.surfaces
        and previous.surfaces[branch].disposition != decision.disposition
    )
    changes.extend(
        AutoDecisionChange(
            kind="surface",
            target=branch,
            change="confidence",
            before=previous.surfaces[branch].confidence.value,
            after=decision.confidence.value,
        )
        for branch, decision in sorted(current.surfaces.items())
        if branch in previous.surfaces
        and previous.surfaces[branch].confidence != decision.confidence
    )
    changes.extend(
        AutoDecisionChange(
            kind="surface",
            target=branch,
            change="confidence",
            before=decision.confidence.value,
            after=_REMOVED_GENERATED_VALUE,
        )
        for branch, decision in sorted(previous.surfaces.items())
        if branch not in current.surfaces
    )
    changes.extend(_surface_metadata_changes(previous, current))
    changes.extend(_surface_override_changes(previous, current, baseline_registry, registry))
    changes.extend(
        AutoDecisionChange(
            kind="workstream",
            target=name,
            change="added",
            after=decision.status.value,
        )
        for name, decision in sorted(current.workstreams.items())
        if name not in previous.workstreams
    )
    changes.extend(
        AutoDecisionChange(
            kind="workstream",
            target=name,
            change="removed",
            before=decision.status.value,
        )
        for name, decision in sorted(previous.workstreams.items())
        if name not in current.workstreams
    )
    changes.extend(
        AutoDecisionChange(
            kind="workstream",
            target=name,
            change="choice",
            before=previous.workstreams[name].status.value,
            after=decision.status.value,
        )
        for name, decision in sorted(current.workstreams.items())
        if name in previous.workstreams and previous.workstreams[name].status != decision.status
    )
    changes.extend(
        AutoDecisionChange(
            kind="workstream",
            target=name,
            change="confidence",
            before=previous.workstreams[name].confidence.value,
            after=decision.confidence.value,
        )
        for name, decision in sorted(current.workstreams.items())
        if name in previous.workstreams
        and previous.workstreams[name].confidence != decision.confidence
    )
    changes.extend(
        AutoDecisionChange(
            kind="workstream",
            target=name,
            change="confidence",
            before=decision.confidence.value,
            after=_REMOVED_GENERATED_VALUE,
        )
        for name, decision in sorted(previous.workstreams.items())
        if name not in current.workstreams
    )
    changes.extend(_workstream_metadata_changes(previous, current))
    changes.extend(_workstream_override_changes(previous, current, baseline_registry, registry))
    return changes


def _surface_metadata_changes(
    previous: AutoDecisionSet,
    current: AutoDecisionSet,
) -> list[AutoDecisionChange]:
    fields = (
        ("workstreams", lambda decision: _sequence_value(decision.workstreams)),
        ("rule", lambda decision: decision.rule),
        ("evidence", lambda decision: decision.evidence),
        ("alternatives", lambda decision: _sequence_value(decision.alternatives)),
        ("last_reviewed", lambda decision: decision.last_reviewed.isoformat()),
        ("association_basis", lambda decision: decision.association_basis),
    )
    changes: list[AutoDecisionChange] = []
    for branch, prior in sorted(previous.surfaces.items()):
        decision = current.surfaces.get(branch)
        for field, value in fields:
            before = value(prior)
            after = value(decision) if decision is not None else _REMOVED_GENERATED_VALUE
            if before != after:
                changes.append(
                    AutoDecisionChange(
                        kind="surface",
                        target=branch,
                        change="metadata",
                        field=field,
                        before=before,
                        after=after,
                    )
                )
    return changes


def _sequence_value(values: tuple[str, ...]) -> str:
    return "; ".join(values) if values else "none"


def _workstream_metadata_changes(
    previous: AutoDecisionSet,
    current: AutoDecisionSet,
) -> list[AutoDecisionChange]:
    fields = (
        ("canonical_surface", lambda decision: decision.canonical_surface or "none"),
        ("rule", lambda decision: decision.rule),
        ("evidence", lambda decision: decision.evidence),
        ("alternatives", lambda decision: _sequence_value(decision.alternatives)),
        ("last_reviewed", lambda decision: decision.last_reviewed.isoformat()),
    )
    changes: list[AutoDecisionChange] = []
    for name, prior in sorted(previous.workstreams.items()):
        decision = current.workstreams.get(name)
        for field, value in fields:
            before = value(prior)
            after = value(decision) if decision is not None else _REMOVED_GENERATED_VALUE
            if before != after:
                changes.append(
                    AutoDecisionChange(
                        kind="workstream",
                        target=name,
                        change="metadata",
                        field=field,
                        before=before,
                        after=after,
                    )
                )
    return changes


def _surface_override_changes(
    previous: AutoDecisionSet,
    current: AutoDecisionSet,
    previous_registry: DecisionRegistry,
    current_registry: DecisionRegistry,
) -> list[AutoDecisionChange]:
    changes: list[AutoDecisionChange] = []
    targets = sorted(previous_registry.surfaces.keys() | current_registry.surfaces.keys())
    for branch in targets:
        previous_override = previous_registry.surfaces.get(branch)
        current_override = current_registry.surfaces.get(branch)
        previous_choice = previous_override.disposition.value if previous_override else None
        current_choice = current_override.disposition.value if current_override else None
        if previous_override == current_override:
            continue
        previous_generated = previous.surfaces.get(branch)
        current_generated = current.surfaces.get(branch)
        before = previous_choice or (
            current_generated.disposition.value
            if current_generated is not None
            else previous_generated.disposition.value
            if previous_generated is not None
            else "none"
        )
        after = current_choice or (
            current_generated.disposition.value if current_generated is not None else "none"
        )
        changes.append(
            AutoDecisionChange(
                kind="surface",
                target=branch,
                change=_override_lifecycle_change(
                    previous_override is not None,
                    current_override is not None,
                ),
                before=before,
                after=after,
            )
        )
    return changes


def _workstream_override_changes(
    previous: AutoDecisionSet,
    current: AutoDecisionSet,
    previous_registry: DecisionRegistry,
    current_registry: DecisionRegistry,
) -> list[AutoDecisionChange]:
    changes: list[AutoDecisionChange] = []
    targets = sorted(previous_registry.workstreams.keys() | current_registry.workstreams.keys())
    for name in targets:
        previous_override = previous_registry.workstreams.get(name)
        current_override = current_registry.workstreams.get(name)
        previous_choice = previous_override.status.value if previous_override else None
        current_choice = current_override.status.value if current_override else None
        if previous_override == current_override:
            continue
        previous_generated = previous.workstreams.get(name)
        current_generated = current.workstreams.get(name)
        before = previous_choice or (
            current_generated.status.value
            if current_generated is not None
            else previous_generated.status.value
            if previous_generated is not None
            else "none"
        )
        after = current_choice or (
            current_generated.status.value if current_generated is not None else "none"
        )
        changes.append(
            AutoDecisionChange(
                kind="workstream",
                target=name,
                change=_override_lifecycle_change(
                    previous_override is not None,
                    current_override is not None,
                ),
                before=before,
                after=after,
            )
        )
    return changes


def _override_lifecycle_change(
    had_override: bool,
    has_override: bool,
) -> Literal["override-added", "override-changed", "override-cleared"]:
    if had_override and has_override:
        return "override-changed"
    if had_override:
        return "override-cleared"
    return "override-added"


@dataclass(frozen=True)
class _SurfaceAssociation:
    workstreams: tuple[str, ...]
    basis: str
    tied_workstreams: tuple[str, ...] = ()


def _audit_surface_association(
    decision: AutoDisposition,
    association: _SurfaceAssociation,
) -> AutoDisposition:
    audited = replace(decision, association_basis=association.basis)
    if not association.tied_workstreams:
        return audited
    primary = association.workstreams[0]
    tied = tuple(name for name in association.tied_workstreams if name != primary)
    tie_evidence = (
        "Workstream association ambiguity: equal best branch-term matches "
        f"{', '.join(association.tied_workstreams)}; {primary} is the deterministic primary "
        "and all tied workstreams are retained."
    )
    association_alternatives = tuple(
        f"workstream association: {name} is tied with deterministic primary {primary}"
        for name in tied
    )
    return replace(
        audited,
        confidence=Confidence.LOW,
        evidence=f"{audited.evidence} {tie_evidence}",
        alternatives=tuple(dict.fromkeys((*audited.alternatives, *association_alternatives))),
    )


def _audit_workstream_associations(
    decisions: dict[str, AutoWorkstreamDecision],
    associations: Mapping[str, _SurfaceAssociation],
    activity_dates: Mapping[str, date],
) -> dict[str, AutoWorkstreamDecision]:
    audited = dict(decisions)
    for branch, association in sorted(associations.items()):
        if not association.tied_workstreams:
            continue
        primary = association.workstreams[0]
        tie_evidence = (
            "Workstream association ambiguity: equal best branch-term matches "
            f"{', '.join(association.tied_workstreams)}; {primary} is the deterministic primary "
            "and all tied workstreams are retained."
        )
        for name in association.tied_workstreams:
            if name == primary:
                continue
            alternative = f"active: associate {branch} with {name} instead of {primary}"
            current = audited.get(name)
            if current is None or current.rule == "no-current-evidence-stale":
                audited[name] = AutoWorkstreamDecision(
                    status=WorkstreamStatus.STALE,
                    canonical_surface="",
                    rule="association-tie-unselected",
                    evidence=(
                        f"{tie_evidence} Surface {branch} was assigned to {primary}; "
                        f"no selected surface remains for {name}."
                    ),
                    confidence=Confidence.LOW,
                    alternatives=(alternative,),
                    last_reviewed=_surface_metadata_date(branch, activity_dates),
                )
                continue
            audited[name] = replace(
                current,
                confidence=Confidence.LOW,
                evidence=(
                    f"{current.evidence} {tie_evidence} Surface {branch} was assigned to "
                    f"{primary} instead of {name}."
                ),
                alternatives=tuple(dict.fromkeys((*current.alternatives, alternative))),
                last_reviewed=max(
                    current.last_reviewed,
                    _surface_metadata_date(branch, activity_dates),
                ),
            )
    return dict(sorted(audited.items()))


def _surface_association(
    branch: str,
    workstreams: Sequence[PolicyWorkstream],
    workstreams_by_name: Mapping[str, PolicyWorkstream],
    registry: DecisionRegistry,
    aliases: Mapping[str, str],
    implied_aliases: Mapping[str, str],
) -> _SurfaceAssociation:
    tokens = branch_topic_tokens(branch)
    joined = "-".join(tokens)
    alias_key = (
        joined if joined in aliases else next((token for token in tokens if token in aliases), None)
    )
    alias_target = aliases.get(alias_key) if alias_key is not None else None
    if alias_target in workstreams_by_name:
        explicit_alias_key = (
            joined
            if joined in registry.aliases
            else next((token for token in tokens if token in registry.aliases), None)
        )
        explicitly_grounded = (
            explicit_alias_key is not None and registry.aliases[explicit_alias_key] == alias_target
        )
        basis = (
            "implied-alias"
            if alias_key in implied_aliases and not explicitly_grounded
            else "explicit-alias"
        )
        return _SurfaceAssociation((alias_target,), basis)

    candidates: list[tuple[int, str]] = []
    for workstream in workstreams:
        if workstream.name == _HYGIENE_WORKSTREAM:
            continue
        specificity = max(
            (_term_specificity(term, tokens, joined) for term in workstream.branch_terms),
            default=0,
        )
        if specificity:
            candidates.append((specificity, workstream.name))
    if candidates:
        best_specificity = max(specificity for specificity, _ in candidates)
        matches = tuple(
            sorted({name for specificity, name in candidates if specificity == best_specificity})
        )
        selected = matches[0]
        basis = getattr(workstreams_by_name[selected], "association_basis", "branch-term")
        return _SurfaceAssociation(
            (selected,),
            basis,
            tied_workstreams=matches if len(matches) > 1 else (),
        )
    return _SurfaceAssociation((_HYGIENE_WORKSTREAM,), "unclassified")


def _surface_decisions(
    *,
    surfaces: Sequence[PolicySurface],
    associations: Mapping[str, tuple[str, ...]],
    unmerged: set[str],
    real_branches: set[str],
    uncertain_worktrees: Mapping[str, str],
    activity_dates: Mapping[str, date],
    registry: DecisionRegistry,
    run_date: date,
    github_evidence: GitHubEvidence | None,
    stale_lookback_days: int,
    phase2_enabled: bool,
) -> dict[str, AutoDisposition]:
    decisions: dict[str, AutoDisposition] = {}
    live_by_workstream: dict[str, list[str]] = {}
    competition_uncertainties: dict[str, tuple[str, ...]] = {}
    non_branch_uncertainties: dict[str, str] = {}
    selected_open_pr_rejections: dict[str, PullRequestEvidence] = {}
    open_prs = _open_prs_by_branch(github_evidence)
    for surface in surfaces:
        branch = surface.branch
        workstream_names = associations[branch]
        activity_date = activity_dates.get(branch)
        metadata_date = _surface_metadata_date(branch, activity_dates)
        if branch not in real_branches:
            non_branch_uncertainties[branch] = (
                f"Surface provenance {surface.provenance} is not a normalized branch ref; "
                "merged state cannot be inferred."
            )
        if branch not in unmerged and branch in uncertain_worktrees:
            for workstream_name in workstream_names:
                live_by_workstream.setdefault(workstream_name, []).append(branch)
            continue
        if branch not in unmerged and branch in real_branches:
            decisions[branch] = AutoDisposition(
                workstreams=workstream_names,
                disposition=SurfaceDisposition.ARCHIVE,
                rule="merged-into-main",
                evidence=(
                    "Branch absent from unmerged topology; merged into origin/main"
                    + _tip_date_clause(activity_date)
                    + "."
                ),
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=metadata_date,
            )
            continue
        if branch in non_branch_uncertainties:
            for workstream_name in workstream_names:
                live_by_workstream.setdefault(workstream_name, []).append(branch)
            continue
        if (
            phase2_enabled
            and branch not in open_prs
            and activity_date is not None
            and activity_date <= run_date
            and (run_date - activity_date).days > stale_lookback_days
        ):
            online = github_evidence is not None
            evidence = (
                f"Tip committer date {activity_date.isoformat()} is older than "
                f"{stale_lookback_days} days; "
                + (
                    "GitHub evidence confirmed no open PR heads this branch."
                    if online
                    else "PR state unknown (GitHub evidence unavailable)."
                )
            )
            alternatives = (
                ()
                if online
                else ("canonical: an unknown open PR may supersede this stale classification",)
            )
            decisions[branch] = AutoDisposition(
                workstreams=workstream_names,
                disposition=SurfaceDisposition.STALE,
                rule="stale-no-recent-commits",
                evidence=evidence,
                confidence=Confidence.HIGH if online else Confidence.MEDIUM,
                alternatives=alternatives,
                last_reviewed=metadata_date,
            )
            continue
        for workstream_name in workstream_names:
            live_by_workstream.setdefault(workstream_name, []).append(branch)

    proposals: dict[str, list[tuple[SurfaceDisposition, str, Confidence, tuple[str, ...]]]] = {}
    for _workstream_name, branches in sorted(live_by_workstream.items()):
        unique_branches = sorted(set(branches))
        uncertain_branches = [branch for branch in unique_branches if branch in uncertain_worktrees]
        uncertainty_alternative = (
            "archive: resolve uncertain worktree state for " + ", ".join(uncertain_branches)
            if uncertain_branches
            else ""
        )
        non_branch_branches = [
            branch for branch in unique_branches if branch in non_branch_uncertainties
        ]
        uncertainty_evidence = tuple(
            uncertain_worktrees[branch] for branch in uncertain_branches
        ) + tuple(non_branch_uncertainties[branch] for branch in non_branch_branches)
        for branch in unique_branches:
            competition_uncertainties[branch] = uncertainty_evidence
        open_branches = [branch for branch in unique_branches if branch in open_prs]
        if open_branches:
            selected = _select_open_pr_surface(open_branches, open_prs)
            conflicting = len(unique_branches) > 1
            for branch in unique_branches:
                if branch == selected:
                    alternatives = tuple(
                        candidate for candidate in unique_branches if candidate != branch
                    )
                    proposals.setdefault(branch, []).append(
                        (
                            SurfaceDisposition.CANONICAL,
                            "open-pr-canonical",
                            Confidence.LOW if conflicting else Confidence.HIGH,
                            alternatives,
                        )
                    )
                else:
                    selected_open_pr_rejections[branch] = open_prs[selected]
                    proposals.setdefault(branch, []).append(
                        (
                            SurfaceDisposition.STALE,
                            "open-pr-canonical",
                            Confidence.LOW,
                            (selected,),
                        )
                    )
            continue
        if len(unique_branches) == 1:
            branch = unique_branches[0]
            uncertain = branch in uncertain_worktrees
            non_branch = branch in non_branch_uncertainties
            proposals.setdefault(branch, []).append(
                (
                    SurfaceDisposition.CANONICAL,
                    "non-branch-surface"
                    if non_branch
                    else "worktree-state-uncertain"
                    if uncertain
                    else "unique-live-surface",
                    Confidence.LOW if uncertain or non_branch else Confidence.HIGH,
                    (("archive: resolve non-branch provenance and merged state before cleanup"),)
                    if non_branch
                    else (uncertainty_alternative,)
                    if uncertain
                    else (),
                )
            )
            continue
        selected = _select_newest_surface(unique_branches, activity_dates)
        rule = "newest-live-surface"
        for branch in unique_branches:
            if branch == selected:
                alternatives_list = [
                    candidate for candidate in unique_branches if candidate != branch
                ]
                if uncertainty_alternative:
                    alternatives_list.append(uncertainty_alternative)
                alternatives = tuple(alternatives_list)
                disposition = SurfaceDisposition.CANONICAL
            else:
                alternatives = (selected,) + (
                    (uncertainty_alternative,) if uncertainty_alternative else ()
                )
                disposition = SurfaceDisposition.STALE
            proposals.setdefault(branch, []).append(
                (
                    disposition,
                    rule,
                    Confidence.LOW,
                    alternatives,
                )
            )

    for surface in surfaces:
        branch = surface.branch
        if branch in decisions:
            continue
        branch_proposals = proposals.get(branch)
        if not branch_proposals:
            continue
        disposition = (
            SurfaceDisposition.CANONICAL
            if any(item[0] == SurfaceDisposition.CANONICAL for item in branch_proposals)
            else SurfaceDisposition.STALE
        )
        confidence = (
            Confidence.LOW
            if any(item[2] == Confidence.LOW for item in branch_proposals)
            else Confidence.HIGH
        )
        rule = (
            "open-pr-canonical"
            if any(item[1] == "open-pr-canonical" for item in branch_proposals)
            else "newest-live-surface"
            if any(item[1] == "newest-live-surface" for item in branch_proposals)
            else "worktree-state-uncertain"
            if any(item[1] == "worktree-state-uncertain" for item in branch_proposals)
            else "non-branch-surface"
            if any(item[1] == "non-branch-surface" for item in branch_proposals)
            else "unique-live-surface"
        )
        alternatives = tuple(
            sorted({alternative for item in branch_proposals for alternative in item[3]})
        )
        activity_date = activity_dates.get(branch)
        names = associations[branch]
        if rule == "worktree-state-uncertain":
            evidence = uncertain_worktrees[branch]
        elif rule == "non-branch-surface":
            evidence = non_branch_uncertainties[branch]
        elif rule == "open-pr-canonical" and branch in selected_open_pr_rejections:
            pull_request = selected_open_pr_rejections[branch]
            evidence = (
                f"Competing surface rejected because PR #{pull_request.number} "
                f"({pull_request.url}) is the deterministic open-PR continuation for "
                f"{', '.join(names)}" + _tip_date_clause(activity_date) + "."
            )
        elif rule == "open-pr-canonical" and branch in open_prs:
            pull_request = open_prs[branch]
            evidence = (
                f"Open PR #{pull_request.number} ({pull_request.url}) heads branch; "
                "canonical continuation is that PR."
            )
            if confidence == Confidence.LOW:
                evidence += " Competing live surfaces require deterministic selection."
        elif rule == "open-pr-canonical":
            evidence = (
                f"Competing surface rejected because open PR continuation exists for "
                f"{', '.join(names)}" + _tip_date_clause(activity_date) + "."
            )
        elif rule == "unique-live-surface":
            evidence = (
                f"Only live surface for {', '.join(names)}; branch remains unmerged"
                + _tip_date_clause(activity_date)
                + "."
            )
        elif disposition == SurfaceDisposition.CANONICAL:
            evidence = (
                f"Newest live surface for {', '.join(names)} selected by committer date "
                "with lexical branch-name tie-break" + _tip_date_clause(activity_date) + "."
            )
        else:
            evidence = (
                f"Competing live surface for {', '.join(names)} rejected by newest-surface rule"
                + _tip_date_clause(activity_date)
                + "."
            )
        uncertainties = competition_uncertainties.get(branch, ())
        if uncertainties and rule not in {"non-branch-surface", "worktree-state-uncertain"}:
            evidence += " Competition uncertainty: " + " ".join(uncertainties)
        decisions[branch] = AutoDisposition(
            workstreams=names,
            disposition=disposition,
            rule=rule,
            evidence=evidence,
            confidence=confidence,
            alternatives=alternatives,
            last_reviewed=_surface_metadata_date(branch, activity_dates),
        )
    return dict(sorted(decisions.items()))


def _workstream_decisions(
    *,
    workstreams: Sequence[PolicyWorkstream],
    associations: Mapping[str, tuple[str, ...]],
    auto_surfaces: Mapping[str, AutoDisposition],
    activity_dates: Mapping[str, date],
    registry: DecisionRegistry,
    run_date: date,
    stale_lookback_days: int,
    github_evidence: GitHubEvidence | None,
    aliases: Mapping[str, str],
    phase2_enabled: bool,
) -> dict[str, AutoWorkstreamDecision]:
    decisions: dict[str, AutoWorkstreamDecision] = {}
    issue_associations = _associate_issues(github_evidence, workstreams, aliases)
    for workstream in sorted(workstreams, key=lambda item: item.name):
        if workstream.name == _HYGIENE_WORKSTREAM:
            continue
        known = sorted(branch for branch, names in associations.items() if workstream.name in names)
        if phase2_enabled:
            phase2_decision = _phase2_workstream_decision(
                workstream=workstream,
                known=known,
                auto_surfaces=auto_surfaces,
                activity_dates=activity_dates,
                registry=registry,
                run_date=run_date,
                stale_lookback_days=stale_lookback_days,
                github_evidence=github_evidence,
                issue_associations=issue_associations,
            )
            decisions[workstream.name] = phase2_decision or _no_current_evidence_decision()
            continue
        if not known:
            decisions[workstream.name] = _no_current_evidence_decision()
            continue
        generated_dispositions = {branch: auto_surfaces[branch].disposition for branch in known}
        candidate_live = [
            branch
            for branch in known
            if generated_dispositions[branch] == SurfaceDisposition.CANONICAL
            or auto_surfaces[branch].rule == "newest-live-surface"
        ]
        live = [
            branch
            for branch in known
            if generated_dispositions[branch] == SurfaceDisposition.CANONICAL
        ]
        canonical = _select_newest_surface(live, activity_dates) if live else ""
        metadata_dates = [_surface_metadata_date(branch, activity_dates) for branch in known]
        metadata_date = (
            _surface_metadata_date(canonical, activity_dates)
            if canonical
            else max(metadata_dates, default=_UNKNOWN_EVIDENCE_DATE)
        )

        all_complete = not live and all(
            disposition == SurfaceDisposition.ARCHIVE
            for disposition in generated_dispositions.values()
        )
        if all_complete:
            decisions[workstream.name] = AutoWorkstreamDecision(
                status=WorkstreamStatus.DONE,
                canonical_surface="",
                rule="all-surfaces-complete",
                evidence="All known surfaces are merged or archived; no live surface remains.",
                confidence=Confidence.MEDIUM,
                alternatives=(),
                last_reviewed=metadata_date,
            )
            continue
        if not live:
            known_activity_dates = sorted(
                {activity_dates[branch] for branch in known if branch in activity_dates}
            )
            latest_activity_date = max(known_activity_dates, default=None)
            activity_clause = (
                " Observed activity dates: "
                + ", ".join(item.isoformat() for item in known_activity_dates)
                + "."
                if known_activity_dates
                else ""
            )
            proven_stale = (
                latest_activity_date is not None
                and (run_date - latest_activity_date).days > stale_lookback_days
            )
            if latest_activity_date is None:
                evidence = (
                    "No effective canonical surface remains and activity dates are unavailable; "
                    "explicit stale or parked surface dispositions prevent an active claim."
                )
                alternatives = (
                    "active: reconsider only after a canonical surface and valid activity date exist",
                )
            elif proven_stale:
                evidence = (
                    "No effective canonical surface remains; latest known commit activity date "
                    f"{latest_activity_date.isoformat()} is older than "
                    f"{stale_lookback_days} days." + activity_clause
                )
                alternatives = ()
            else:
                evidence = (
                    "No effective canonical surface remains; latest known commit activity date "
                    f"{latest_activity_date.isoformat()} conflicts with explicit stale or parked "
                    "surface dispositions." + activity_clause
                )
                alternatives = (
                    "active: recent or future activity conflicts with noncanonical dispositions",
                )
            decisions[workstream.name] = AutoWorkstreamDecision(
                status=WorkstreamStatus.STALE,
                canonical_surface="",
                rule="no-live-surfaces-stale",
                evidence=evidence,
                confidence=Confidence.MEDIUM if proven_stale else Confidence.LOW,
                alternatives=alternatives,
                last_reviewed=metadata_date,
            )
            continue

        canonical_date = activity_dates.get(canonical)
        generated_alternatives = (
            (
                f"{auto_surfaces[canonical].disposition.value}: generated surface rule "
                f"{auto_surfaces[canonical].rule}",
            )
            if canonical and auto_surfaces[canonical].disposition != SurfaceDisposition.CANONICAL
            else ()
        )
        uncertainty_evidence, uncertainty_alternatives = _canonical_uncertainty_context(
            canonical,
            auto_surfaces,
        )
        contextual_alternatives = generated_alternatives + uncertainty_alternatives
        conflicting = len(candidate_live) > 1
        if canonical_date is None:
            status = WorkstreamStatus.ACTIVE
            rule = "activity-date-unknown"
            evidence = (
                f"Canonical surface {canonical or 'unknown'} has unavailable tip committer date; "
                "staleness cannot be determined."
            )
            alternatives = contextual_alternatives + (
                "stale: classify only after a valid activity date is available",
            )
            confidence = Confidence.LOW
        elif canonical_date > run_date:
            status = WorkstreamStatus.ACTIVE
            rule = "activity-date-future"
            evidence = (
                f"Canonical surface {canonical} has future tip committer date "
                f"{canonical_date.isoformat()}; staleness cannot be determined."
            )
            alternatives = contextual_alternatives + (
                "stale: classify only after correcting or reaching the future activity date",
            )
            confidence = Confidence.LOW
        elif (run_date - canonical_date).days <= stale_lookback_days:
            status = WorkstreamStatus.ACTIVE
            rule = "recent-activity-active"
            evidence = (
                f"Canonical surface {canonical} has activity within {stale_lookback_days} days; "
                f"tip committer date {canonical_date.isoformat()}."
            )
            alternatives = contextual_alternatives + (
                ("stale: competing live surfaces may supersede the canonical choice",)
                if conflicting
                else ()
            )
            confidence = (
                Confidence.LOW
                if conflicting
                or contextual_alternatives
                or auto_surfaces[canonical].confidence == Confidence.LOW
                else Confidence.HIGH
            )
        else:
            status = WorkstreamStatus.STALE
            rule = "inactive-stale"
            date_label = canonical_date.isoformat() if canonical_date is not None else "unknown"
            evidence = (
                f"Canonical surface {canonical or 'unknown'} has no activity within "
                f"{stale_lookback_days} days; tip committer date {date_label}."
            )
            alternatives = contextual_alternatives + (
                ("active: competing live surfaces provide conflicting evidence",)
                if conflicting
                else ()
            )
            confidence = (
                Confidence.LOW
                if conflicting
                or contextual_alternatives
                or auto_surfaces[canonical].confidence == Confidence.LOW
                else Confidence.MEDIUM
            )
        evidence += uncertainty_evidence
        decisions[workstream.name] = AutoWorkstreamDecision(
            status=status,
            canonical_surface=canonical,
            rule=rule,
            evidence=evidence,
            confidence=confidence,
            alternatives=alternatives,
            last_reviewed=metadata_date,
        )
    return decisions


def _no_current_evidence_decision() -> AutoWorkstreamDecision:
    return AutoWorkstreamDecision(
        status=WorkstreamStatus.STALE,
        canonical_surface="",
        rule="no-current-evidence-stale",
        evidence=(
            "No associated surface, pull request, or issue evidence is available; "
            "stale is the conservative terminal status."
        ),
        confidence=Confidence.LOW,
        alternatives=("active: require an associated current surface, open PR, or active issue",),
        last_reviewed=_UNKNOWN_EVIDENCE_DATE,
    )


def _phase2_workstream_decision(
    *,
    workstream: PolicyWorkstream,
    known: list[str],
    auto_surfaces: Mapping[str, AutoDisposition],
    activity_dates: Mapping[str, date],
    registry: DecisionRegistry,
    run_date: date,
    stale_lookback_days: int,
    github_evidence: GitHubEvidence | None,
    issue_associations: Mapping[
        str,
        tuple[tuple[IssueEvidence, ...], tuple[str, ...]],
    ],
) -> AutoWorkstreamDecision | None:
    issues, association_ambiguities = issue_associations.get(workstream.name, ((), ()))
    matching_pull_requests = (
        tuple(pr for pr in github_evidence.pull_requests if pr.head_ref in known)
        if github_evidence is not None
        else ()
    )
    if not known and not issues and not matching_pull_requests:
        return None

    generated_dispositions = {branch: auto_surfaces[branch].disposition for branch in known}
    live = [
        branch for branch in known if generated_dispositions[branch] == SurfaceDisposition.CANONICAL
    ]
    live_pull_requests = tuple(pr for pr in matching_pull_requests if pr.head_ref in live)
    open_prs = tuple(pr for pr in live_pull_requests if pr.state == "open")
    open_pr_branches = sorted({pr.head_ref for pr in open_prs})
    if open_pr_branches:
        canonical = _select_open_pr_surface(
            open_pr_branches,
            {pr.head_ref: pr for pr in open_prs},
        )
    elif live:
        canonical = _select_newest_surface(live, activity_dates)
    else:
        canonical = ""

    canonical_date = activity_dates.get(canonical)
    recent_canonical = (
        canonical_date is not None
        and canonical_date <= run_date
        and (run_date - canonical_date).days <= stale_lookback_days
    )
    open_issues = tuple(issue for issue in issues if issue.state == "open")
    blocked_issues = tuple(
        issue for issue in open_issues if {"blocked", "needs-info"}.intersection(issue.labels)
    )
    ready_issues = tuple(issue for issue in open_issues if "ready-for-agent" in issue.labels)
    ordinary_open_issues = tuple(
        issue
        for issue in open_issues
        if not {"blocked", "needs-info", "ready-for-agent"}.intersection(issue.labels)
    )
    merged_prs = tuple(
        pr for pr in matching_pull_requests if pr.state == "merged" or pr.merged_at is not None
    )
    closed_issues = tuple(issue for issue in issues if issue.state == "closed")

    signals: list[tuple[WorkstreamStatus, str, str]] = []
    fallback_alternatives: tuple[str, ...] = ()
    ordinary_issue_active = bool(ordinary_open_issues and not blocked_issues and not ready_issues)
    if open_prs or recent_canonical or ordinary_issue_active:
        evidence_parts: list[str] = []
        if open_prs:
            evidence_parts.append("linked open " + _pr_references(open_prs))
        if recent_canonical and canonical_date is not None:
            evidence_parts.append(
                f"canonical surface {canonical} activity {canonical_date.isoformat()}"
            )
        if ordinary_issue_active:
            evidence_parts.append("matching open " + _issue_references(ordinary_open_issues))
        signals.append(
            (
                WorkstreamStatus.ACTIVE,
                (
                    "open-pr-active"
                    if open_prs
                    else "recent-activity-active"
                    if recent_canonical
                    else "open-issue-active"
                ),
                "; ".join(evidence_parts) + ".",
            )
        )
    if blocked_issues:
        signals.append(
            (
                WorkstreamStatus.BLOCKED,
                "blocked-issue",
                "Matching open blocked/needs-info " + _issue_references(blocked_issues) + ".",
            )
        )
    if ready_issues and not (open_prs or recent_canonical):
        signals.append(
            (
                WorkstreamStatus.READY_FOR_AGENT,
                "ready-issue",
                "Matching open ready-for-agent " + _issue_references(ready_issues) + ".",
            )
        )
    if not live and not open_prs and not open_issues and (merged_prs or closed_issues):
        completion = []
        if merged_prs:
            completion.append("merged " + _pr_references(merged_prs))
        if closed_issues:
            completion.append("closed matching " + _issue_references(closed_issues))
        signals.append(
            (
                WorkstreamStatus.DONE,
                "github-complete",
                "; ".join(completion) + "; no live surface remains.",
            )
        )

    valid_activity = [
        activity_dates[branch]
        for branch in known
        if branch in activity_dates and activity_dates[branch] <= run_date
    ]
    latest_activity = max(valid_activity, default=None)
    proven_old = (
        latest_activity is not None and (run_date - latest_activity).days > stale_lookback_days
    )
    all_complete = (
        bool(known)
        and not live
        and all(
            disposition == SurfaceDisposition.ARCHIVE
            for disposition in generated_dispositions.values()
        )
    )
    completion_conflict = bool(
        ordinary_issue_active and (all_complete or merged_prs or closed_issues)
    )
    if not signals and all_complete and not open_issues:
        signals.append(
            (
                WorkstreamStatus.DONE,
                "all-surfaces-complete",
                "All known surfaces are merged or archived; no live surface remains.",
            )
        )
    stale_allowed = not open_prs and (github_evidence is None or not open_issues)
    if proven_old and stale_allowed and not all_complete:
        online = github_evidence is not None
        signals.append(
            (
                WorkstreamStatus.STALE,
                "inactive-stale",
                (
                    f"Latest branch activity {latest_activity.isoformat()} is older than "
                    f"{stale_lookback_days} days; "
                    + (
                        "GitHub confirms no open PR or active matching issue."
                        if online
                        else "GitHub PR/issue state unknown (evidence unavailable)."
                    )
                ),
            )
        )

    if not signals and known and not live:
        if latest_activity is None:
            evidence = (
                "No effective canonical surface remains and branch activity is unavailable; "
                "explicit parked or stale surface disposition prevents an active claim."
            )
            fallback_alternatives = (
                "active: missing commit activity prevents confirming whether the "
                "noncanonical surface is current",
            )
        else:
            evidence = (
                "No effective canonical surface remains; latest known commit activity "
                f"{latest_activity.isoformat()} conflicts with explicit parked or stale "
                "surface disposition."
            )
            fallback_alternatives = (
                f"active: recent commit activity {latest_activity.isoformat()} conflicts "
                "with explicit noncanonical surface disposition",
            )
        signals.append(
            (
                WorkstreamStatus.STALE,
                "no-live-surfaces-stale",
                evidence,
            )
        )

    if not signals:
        if live:
            if canonical_date is None:
                signals.append(
                    (
                        WorkstreamStatus.ACTIVE,
                        "activity-date-unknown",
                        f"Canonical surface {canonical} has unavailable tip committer date; "
                        "staleness cannot be determined.",
                    )
                )
            elif canonical_date > run_date:
                signals.append(
                    (
                        WorkstreamStatus.ACTIVE,
                        "activity-date-future",
                        f"Canonical surface {canonical} has future tip committer date "
                        f"{canonical_date.isoformat()}; staleness cannot be determined.",
                    )
                )
        if not signals:
            return None

    precedence = {
        WorkstreamStatus.ACTIVE: 0,
        WorkstreamStatus.BLOCKED: 1,
        WorkstreamStatus.READY_FOR_AGENT: 2,
        WorkstreamStatus.DONE: 3,
        WorkstreamStatus.STALE: 4,
    }
    chosen = min(signals, key=lambda item: precedence[item[0]])
    rejected = [signal for signal in signals if signal is not chosen]
    alternatives = tuple(f"{status.value}: {evidence}" for status, _, evidence in rejected)
    alternatives += fallback_alternatives
    if completion_conflict:
        open_numbers = ", ".join(
            f"#{issue.number}"
            for issue in sorted(ordinary_open_issues, key=lambda item: item.number)
        )
        if all_complete:
            alternatives += (
                "done: all known surfaces are merged or archived, but matching open issue "
                f"{open_numbers} prevents completion",
            )
        else:
            completion_parts = []
            if merged_prs:
                completion_parts.append("merged " + _pr_references(merged_prs))
            if closed_issues:
                completion_parts.append("closed matching " + _issue_references(closed_issues))
            alternatives += (
                "done: "
                + "; ".join(completion_parts)
                + f", but matching open issue {open_numbers} prevents completion",
            )
    alternatives += association_ambiguities
    conflicting_surfaces = len(set(live) | set(open_pr_branches)) > 1
    canonical_surface_low = bool(
        canonical and auto_surfaces[canonical].confidence == Confidence.LOW
    )
    low_confidence = (
        bool(rejected)
        or bool(association_ambiguities)
        or conflicting_surfaces
        or canonical_surface_low
        or completion_conflict
    )
    if chosen[1] == "no-live-surfaces-stale":
        confidence = Confidence.LOW
    elif chosen[0] == WorkstreamStatus.STALE and github_evidence is None:
        confidence = Confidence.MEDIUM
        alternatives += (
            "active: unavailable GitHub evidence may contain an open PR or active issue",
        )
    elif chosen[1] in {"activity-date-unknown", "activity-date-future"}:
        confidence = Confidence.LOW
        alternatives += ("stale: requires a valid non-future branch activity date",)
    elif chosen[1] == "all-surfaces-complete":
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW if low_confidence else Confidence.HIGH

    evidence = chosen[2]
    if canonical_surface_low:
        canonical_decision = auto_surfaces[canonical]
        evidence += (
            f" Canonical surface uncertainty ({canonical_decision.rule}): "
            f"{canonical_decision.evidence}"
        )
        alternatives += canonical_decision.alternatives
    if association_ambiguities:
        evidence += " Issue title association is ambiguous across exact workstream terms."
    metadata_dates = [
        *(_surface_metadata_date(branch, activity_dates) for branch in known),
        *(issue.updated_at for issue in issues),
        *(pr.updated_at for pr in matching_pull_requests),
    ]
    return AutoWorkstreamDecision(
        status=chosen[0],
        canonical_surface=canonical,
        rule=chosen[1],
        evidence=evidence,
        confidence=confidence,
        alternatives=alternatives,
        last_reviewed=max(metadata_dates, default=_UNKNOWN_EVIDENCE_DATE),
    )


def _associate_issues(
    github_evidence: GitHubEvidence | None,
    workstreams: Sequence[PolicyWorkstream],
    aliases: Mapping[str, str],
) -> dict[str, tuple[tuple[IssueEvidence, ...], tuple[str, ...]]]:
    if github_evidence is None:
        return {}
    issues_by_workstream: dict[str, list[IssueEvidence]] = {}
    ambiguities_by_workstream: dict[str, set[str]] = {}
    for issue in github_evidence.issues:
        title_tokens = _normalized_tokens(issue.title)
        matches: list[str] = []
        for workstream in workstreams:
            if workstream.name == _HYGIENE_WORKSTREAM:
                continue
            workstream_name = _normalized_tokens(workstream.name)
            alias_sequences = [
                _normalized_tokens(alias)
                for alias, target in aliases.items()
                if target == workstream.name
            ]
            branch_term_sequences = [
                sequence
                for term in workstream.branch_terms
                if (sequence := _normalized_tokens(term)) and _specific_issue_branch_term(sequence)
            ]
            exact_name = bool(workstream_name) and title_tokens == workstream_name
            explicit_alias = any(
                sequence and _contains_token_sequence(title_tokens, sequence)
                for sequence in alias_sequences
            )
            specific_branch_term = any(
                _contains_token_sequence(title_tokens, sequence)
                for sequence in branch_term_sequences
            )
            if exact_name or explicit_alias or specific_branch_term:
                matches.append(workstream.name)
        ordered_matches = sorted(set(matches))
        if len(ordered_matches) > 1:
            ambiguity = (
                f"association ambiguity: issue #{issue.number} ({issue.url}) "
                f"matches workstreams {', '.join(ordered_matches)}"
            )
            for name in ordered_matches:
                ambiguities_by_workstream.setdefault(name, set()).add(ambiguity)
        for name in ordered_matches:
            issues_by_workstream.setdefault(name, []).append(issue)
    return {
        name: (
            tuple(sorted(issues, key=lambda item: (item.number, item.url))),
            tuple(sorted(ambiguities_by_workstream.get(name, set()))),
        )
        for name, issues in issues_by_workstream.items()
    }


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.lower()))


def _contains_token_sequence(
    tokens: tuple[str, ...],
    sequence: tuple[str, ...],
) -> bool:
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence for index in range(len(tokens) - width + 1)
    )


def _specific_issue_branch_term(sequence: tuple[str, ...]) -> bool:
    return any(token not in _GENERIC_ISSUE_BRANCH_TERMS for token in sequence)


def _pr_references(pull_requests: Sequence[PullRequestEvidence]) -> str:
    return ", ".join(
        f"PR #{pr.number} ({pr.url})"
        for pr in sorted(pull_requests, key=lambda item: (item.number, item.url))
    )


def _issue_references(issues: Sequence[IssueEvidence]) -> str:
    return ", ".join(
        f"issue #{issue.number} ({issue.url})"
        for issue in sorted(issues, key=lambda item: (item.number, item.url))
    )


def _canonical_uncertainty_context(
    canonical: str,
    auto_surfaces: Mapping[str, AutoDisposition],
) -> tuple[str, tuple[str, ...]]:
    if not canonical:
        return ("", ())
    surface = auto_surfaces[canonical]
    if surface.confidence != Confidence.LOW:
        return ("", ())
    return (
        f" Canonical surface uncertainty ({surface.rule}): {surface.evidence}",
        surface.alternatives,
    )


def _surface_metadata_date(
    branch: str,
    activity_dates: Mapping[str, date],
) -> date:
    return activity_dates.get(branch, _UNKNOWN_EVIDENCE_DATE)


def _select_newest_surface(
    branches: Sequence[str],
    activity_dates: Mapping[str, date],
) -> str:
    newest_date = max(
        (activity_dates.get(branch, date.min) for branch in branches), default=date.min
    )
    return min(branch for branch in branches if activity_dates.get(branch, date.min) == newest_date)


def _open_prs_by_branch(
    github_evidence: GitHubEvidence | None,
) -> dict[str, PullRequestEvidence]:
    if github_evidence is None:
        return {}
    open_prs: dict[str, PullRequestEvidence] = {}
    for pull_request in github_evidence.pull_requests:
        if pull_request.state != "open":
            continue
        existing = open_prs.get(pull_request.head_ref)
        if existing is None or (pull_request.updated_at, -pull_request.number) > (
            existing.updated_at,
            -existing.number,
        ):
            open_prs[pull_request.head_ref] = pull_request
    return open_prs


def _select_open_pr_surface(
    branches: Sequence[str],
    open_prs: Mapping[str, PullRequestEvidence],
) -> str:
    newest = max(open_prs[branch].updated_at for branch in branches)
    return min(branch for branch in branches if open_prs[branch].updated_at == newest)


def branch_topic_tokens(branch: str) -> list[str]:
    """Extract normalized branch-topic tokens shared by discovery and policy."""
    slug = branch.strip()
    for prefix in ("codex/", "cursor/"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
            break
    slug = _BRANCH_DATE_SUFFIX.sub("", slug.lower())
    slug = _BRANCH_HASH_SUFFIX.sub("", slug)
    return [token for token in slug.split("-") if token]


def _term_specificity(term: str, tokens: list[str], full_slug: str) -> int:
    term_tokens = [token for token in term.strip().lower().split("-") if token]
    if not term_tokens:
        return 0
    normalized = "-".join(term_tokens)
    if normalized == full_slug:
        return 1000 + len(term_tokens)
    width = len(term_tokens)
    if any(
        tokens[index : index + width] == term_tokens for index in range(len(tokens) - width + 1)
    ):
        return width
    return 0


def _tip_date_clause(activity_date: date | None) -> str:
    if activity_date is None:
        return "; tip committer date unavailable"
    return f"; tip committer date {activity_date.isoformat()}"


def _surface_payload(decision: AutoDisposition) -> dict[str, object]:
    return {
        "workstreams": list(decision.workstreams),
        "disposition": decision.disposition.value,
        "rule": decision.rule,
        "evidence": decision.evidence,
        "confidence": decision.confidence.value,
        "association_basis": decision.association_basis,
        "alternatives": list(decision.alternatives),
        "last_reviewed": decision.last_reviewed.isoformat(),
    }


def _workstream_payload(decision: AutoWorkstreamDecision) -> dict[str, object]:
    return {
        "status": decision.status.value,
        "canonical_surface": decision.canonical_surface,
        "rule": decision.rule,
        "evidence": decision.evidence,
        "confidence": decision.confidence.value,
        "alternatives": list(decision.alternatives),
        "last_reviewed": decision.last_reviewed.isoformat(),
    }


def _parse_auto_decisions(raw: object) -> AutoDecisionSet:
    root = _object(raw)
    if set(root) != {"format_version", "surfaces", "workstreams"}:
        raise ValueError("invalid auto decision root")
    format_version = root["format_version"]
    if (
        not isinstance(format_version, int)
        or isinstance(format_version, bool)
        or format_version not in SUPPORTED_AUTO_FORMAT_VERSIONS
    ):
        raise ValueError("unsupported auto decision format")
    surface_entries = _object(root["surfaces"])
    workstream_entries = _object(root["workstreams"])
    surfaces = {
        _identifier(branch, "surface key"): _parse_surface(
            value,
            format_version=format_version,
        )
        for branch, value in surface_entries.items()
    }
    workstreams = {
        _identifier(name, "workstream key"): _parse_workstream(value)
        for name, value in workstream_entries.items()
    }
    return AutoDecisionSet(surfaces=surfaces, workstreams=workstreams)


def _parse_surface(raw: object, *, format_version: int) -> AutoDisposition:
    item = _object(raw)
    base_fields = {
        "workstreams",
        "disposition",
        "rule",
        "evidence",
        "confidence",
        "alternatives",
        "last_reviewed",
    }
    allowed_fields = base_fields | {"association_basis"}
    if set(item) - allowed_fields or not base_fields.issubset(item):
        raise ValueError("invalid surface decision")
    if format_version >= 2 and "association_basis" not in item:
        raise ValueError("invalid surface decision")
    workstreams = _strings(item["workstreams"], allow_empty=False, unique=True)
    return AutoDisposition(
        workstreams=workstreams,
        disposition=SurfaceDisposition(_text(item["disposition"])),
        rule=_text(item["rule"]),
        evidence=_text(item["evidence"]),
        confidence=Confidence(_text(item["confidence"])),
        association_basis=(
            _text(item["association_basis"]) if "association_basis" in item else "unclassified"
        ),
        alternatives=_strings(item["alternatives"], allow_empty=True),
        last_reviewed=date.fromisoformat(_text(item["last_reviewed"])),
    )


def _parse_workstream(raw: object) -> AutoWorkstreamDecision:
    item = _object(raw)
    required = {
        "status",
        "canonical_surface",
        "rule",
        "evidence",
        "confidence",
        "alternatives",
        "last_reviewed",
    }
    if set(item) != required:
        raise ValueError("invalid workstream decision")
    canonical_surface = _optional_identifier(item["canonical_surface"], "canonical surface")
    return AutoWorkstreamDecision(
        status=WorkstreamStatus(_text(item["status"])),
        canonical_surface=canonical_surface,
        rule=_text(item["rule"]),
        evidence=_text(item["evidence"]),
        confidence=Confidence(_text(item["confidence"])),
        alternatives=_strings(item["alternatives"], allow_empty=True),
        last_reviewed=date.fromisoformat(_text(item["last_reviewed"])),
    )


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected object")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-empty text")
    return value.strip()


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"invalid {label}")
    return value


def _optional_identifier(value: object, label: str) -> str:
    if value == "":
        return ""
    return _identifier(value, label)


def _strings(
    value: object,
    *,
    allow_empty: bool,
    unique: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("expected string list")
    if not allow_empty and not value:
        raise ValueError("expected non-empty string list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("expected string list")
    normalized = tuple(item.strip() for item in value)
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError("expected unique string list")
    return normalized
