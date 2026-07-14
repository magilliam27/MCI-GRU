from __future__ import annotations

import json
from datetime import date

import pytest

from cockpit.decisions import (
    DecisionRegistry,
    SurfaceDecision,
    SurfaceDisposition,
    overlay_auto_decisions,
)
from cockpit.models import (
    Confidence,
    GitHubEvidence,
    GitTopologySnapshot,
    IssueEvidence,
    PullRequestEvidence,
    WorkstreamStatus,
    WorktreeEvidence,
)
from cockpit.policy import (
    AUTO_DECISIONS_PATH,
    compute_auto_decisions,
    read_auto_decisions,
    write_auto_decisions,
)
from cockpit.runner import TopologySurface, WorkstreamSeed, _workstream_surfaces


def test_compute_auto_decisions_applies_surface_precedence_and_status_rules() -> None:
    surfaces = [
        _surface("codex/merged-lambdarank"),
        _surface("codex/live-lambdarank"),
        _surface("codex/old-portfolio"),
    ]
    workstreams = [
        _seed("LambdaRankIC", "lambdarank"),
        _seed("Portfolio-IC", "portfolio"),
    ]
    topology = _topology(
        unmerged=["codex/live-lambdarank", "codex/old-portfolio"],
        branches=[surface.branch for surface in surfaces],
    )

    decisions = compute_auto_decisions(
        surfaces=surfaces,
        workstreams=workstreams,
        topology=topology,
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[
            ("codex/merged-lambdarank", date(2026, 7, 8)),
            ("codex/live-lambdarank", date(2026, 7, 11)),
            ("codex/old-portfolio", date(2026, 5, 1)),
        ],
    )

    merged = decisions.surfaces["codex/merged-lambdarank"]
    assert merged.disposition == SurfaceDisposition.ARCHIVE
    assert merged.rule == "merged-into-main"
    assert merged.confidence == Confidence.HIGH
    assert merged.association_basis == "branch-term"
    assert merged.last_reviewed == date(2026, 7, 8)

    live = decisions.surfaces["codex/live-lambdarank"]
    assert live.disposition == SurfaceDisposition.CANONICAL
    assert live.rule == "unique-live-surface"
    assert decisions.workstreams["LambdaRankIC"].status == WorkstreamStatus.ACTIVE
    assert decisions.workstreams["LambdaRankIC"].confidence == Confidence.HIGH

    stale = decisions.workstreams["Portfolio-IC"]
    assert stale.status == WorkstreamStatus.STALE
    assert stale.rule == "inactive-stale"
    assert stale.confidence == Confidence.MEDIUM


def test_compute_auto_decisions_marks_completed_workstream_done() -> None:
    decisions = compute_auto_decisions(
        surfaces=[_surface("codex/merged-lambdarank")],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[], branches=["codex/merged-lambdarank"]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[("codex/merged-lambdarank", date(2026, 7, 8))],
    )

    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.DONE
    assert workstream.rule == "all-surfaces-complete"
    assert workstream.confidence == Confidence.MEDIUM
    assert workstream.canonical_surface == ""


def test_compute_auto_decisions_records_explicit_and_implied_alias_basis() -> None:
    explicit_branch = "codex/direct-followup"
    implied_branch = "codex/learned-followup"
    registry = DecisionRegistry(aliases={"direct": "Canonical stream"})

    decisions = compute_auto_decisions(
        surfaces=[_surface(explicit_branch), _surface(implied_branch)],
        workstreams=[_seed("Canonical stream")],
        topology=_topology(unmerged=[explicit_branch, implied_branch]),
        registry=registry,
        aliases={"direct": "Canonical stream", "learned": "Canonical stream"},
        implied_aliases={"learned": "Canonical stream"},
        run_date=date(2026, 7, 12),
        recent_branches=[
            (explicit_branch, date(2026, 7, 11)),
            (implied_branch, date(2026, 7, 10)),
        ],
    )

    assert decisions.surfaces[explicit_branch].association_basis == "explicit-alias"
    assert decisions.surfaces[implied_branch].association_basis == "implied-alias"


def test_custom_workstream_seed_association_basis_fails_closed() -> None:
    branch = "codex/custom-source-topic"
    seed = WorkstreamSeed(
        name="Custom source",
        status=WorkstreamStatus.ACTIVE,
        next_action="Continue.",
        branch_terms=("custom",),
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[seed],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].association_basis == "unclassified"


def test_linked_pull_request_grounds_title_fallback_surface_association() -> None:
    branch = "codex/novel-stream"
    seed = WorkstreamSeed(
        name="Novel stream",
        status=WorkstreamStatus.ACTIVE,
        next_action="Continue.",
        branch_terms=("novel",),
        association_basis="title-case-fallback",
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[seed],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
        github_evidence=GitHubEvidence(pull_requests=(_pr(80, branch, state="open"),)),
    )

    assert decisions.surfaces[branch].association_basis == "linked-pr"


def test_unambiguous_linked_issue_grounds_title_fallback_surface_association() -> None:
    branch = "codex/novel-stream"
    seed = WorkstreamSeed(
        name="Novel stream",
        status=WorkstreamStatus.ACTIVE,
        next_action="Continue.",
        branch_terms=("novel",),
        association_basis="title-case-fallback",
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[seed],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
        github_evidence=GitHubEvidence(
            issues=(_issue(81, "Novel stream", "open", ("enhancement",)),)
        ),
    )

    assert decisions.surfaces[branch].association_basis == "linked-issue"


def test_dirty_attached_worktree_is_not_archived_as_merged() -> None:
    branch = "codex/lambdarank-dirty"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(
            unmerged=[],
            branches=[branch],
            worktrees=[
                WorktreeEvidence(
                    path="C:/repo-dirty",
                    head="abc1234",
                    branch=branch,
                    detached=False,
                    status_header=f"## {branch}",
                    dirty_paths=["cockpit/policy.py"],
                )
            ],
        ),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    surface = decisions.surfaces[branch]
    assert surface.disposition != SurfaceDisposition.ARCHIVE
    assert surface.rule == "worktree-state-uncertain"
    assert surface.confidence == Confidence.LOW
    assert "dirty" in surface.evidence.lower()
    assert any("archive" in alternative for alternative in surface.alternatives)
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.LOW
    assert "worktree-state-uncertain" in workstream.evidence
    assert "dirty" in workstream.evidence.lower()
    assert any("archive" in alternative for alternative in workstream.alternatives)


def test_status_error_attached_worktree_is_not_archived_as_merged() -> None:
    branch = "codex/lambdarank-status-error"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(
            unmerged=[],
            branches=[branch],
            worktrees=[
                WorktreeEvidence(
                    path="C:/repo-unreadable",
                    head="def5678",
                    branch=branch,
                    detached=False,
                    status_header="",
                    status_error="git status failed",
                )
            ],
        ),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    surface = decisions.surfaces[branch]
    assert surface.disposition != SurfaceDisposition.ARCHIVE
    assert surface.rule == "worktree-state-uncertain"
    assert surface.confidence == Confidence.LOW
    assert "status unavailable" in surface.evidence.lower()
    assert any("archive" in alternative for alternative in surface.alternatives)
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.LOW
    assert "worktree-state-uncertain" in workstream.evidence
    assert "status unavailable" in workstream.evidence.lower()
    assert any("archive" in alternative for alternative in workstream.alternatives)


@pytest.mark.parametrize(
    ("dirty_paths", "status_error", "evidence_text"),
    [
        (["cockpit/policy.py"], "", "dirty"),
        ([], "git status failed", "status unavailable"),
    ],
)
def test_old_uncertain_canonical_surface_keeps_workstream_confidence_low(
    dirty_paths: list[str],
    status_error: str,
    evidence_text: str,
) -> None:
    branch = f"codex/lambdarank-old-{evidence_text.replace(' ', '-')}"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(
            unmerged=[],
            branches=[branch],
            worktrees=[
                WorktreeEvidence(
                    path="C:/repo-uncertain",
                    head="abc1234",
                    branch=branch,
                    detached=False,
                    status_header=f"## {branch}",
                    dirty_paths=dirty_paths,
                    status_error=status_error,
                )
            ],
        ),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 5, 1))],
    )

    surface = decisions.surfaces[branch]
    assert surface.disposition == SurfaceDisposition.CANONICAL
    assert surface.rule == "worktree-state-uncertain"
    assert surface.confidence == Confidence.LOW
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.STALE
    assert workstream.confidence == Confidence.LOW
    assert evidence_text in workstream.evidence.lower()
    assert any("archive" in alternative for alternative in workstream.alternatives)


def test_uncertain_worktree_competes_with_ordinary_live_surface() -> None:
    dirty = "codex/lambdarank-dirty-newer"
    ordinary = "codex/lambdarank-ordinary"
    decisions = compute_auto_decisions(
        surfaces=[_surface(dirty), _surface(ordinary)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(
            unmerged=[ordinary],
            branches=[dirty, ordinary],
            worktrees=[
                WorktreeEvidence(
                    path="C:/repo-dirty",
                    head="abc1234",
                    branch=dirty,
                    detached=False,
                    status_header=f"## {dirty}",
                    dirty_paths=["cockpit/policy.py"],
                )
            ],
        ),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[
            (dirty, date(2026, 7, 12)),
            (ordinary, date(2026, 7, 11)),
        ],
    )

    canonical = [
        branch
        for branch, decision in decisions.surfaces.items()
        if decision.disposition == SurfaceDisposition.CANONICAL
    ]
    assert canonical == [dirty]
    assert decisions.surfaces[dirty].confidence == Confidence.LOW
    assert "dirty" in decisions.surfaces[dirty].evidence.lower()
    assert ordinary in decisions.surfaces[dirty].alternatives
    assert decisions.surfaces[ordinary].disposition == SurfaceDisposition.STALE
    assert decisions.surfaces[ordinary].confidence == Confidence.LOW
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == dirty
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.LOW
    assert "dirty" in workstream.evidence.lower()
    assert any("archive" in alternative for alternative in workstream.alternatives)
    assert any("competing" in alternative for alternative in workstream.alternatives)


def test_explicit_archived_surface_does_not_keep_generated_workstream_active() -> None:
    branch = "codex/lambdarank-retired"
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.ARCHIVE,
                reason="Explicitly retired.",
                next_action="Keep archived.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].disposition == SurfaceDisposition.ARCHIVE
    assert decisions.workstreams["LambdaRankIC"].status == WorkstreamStatus.DONE
    assert decisions.workstreams["LambdaRankIC"].canonical_surface == ""


@pytest.mark.parametrize(
    "overridden_disposition",
    [
        SurfaceDisposition.ARCHIVE,
        SurfaceDisposition.STALE,
        SurfaceDisposition.PARKED,
    ],
)
def test_explicit_noncanonical_surface_is_excluded_from_live_competition(
    overridden_disposition: SurfaceDisposition,
) -> None:
    valid = "codex/lambdarank-valid"
    excluded = "codex/lambdarank-excluded"
    registry = DecisionRegistry(
        surfaces={
            excluded: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=overridden_disposition,
                reason="Explicitly excluded from canonical competition.",
                next_action="Keep excluded.",
                last_reviewed=date(2026, 7, 12),
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(valid), _surface(excluded)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[valid, excluded]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[
            (valid, date(2026, 7, 11)),
            (excluded, date(2026, 7, 12)),
        ],
    )

    assert decisions.surfaces[excluded].disposition == overridden_disposition
    assert decisions.surfaces[valid].disposition == SurfaceDisposition.CANONICAL
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == valid
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.HIGH


@pytest.mark.parametrize(
    "surface_disposition",
    [SurfaceDisposition.STALE, SurfaceDisposition.PARKED],
)
@pytest.mark.parametrize(
    ("activity_date", "expected_confidence"),
    [
        (date(2026, 5, 1), Confidence.MEDIUM),
        (None, Confidence.LOW),
    ],
)
def test_nonarchive_surfaces_without_canonical_produce_stale_workstream(
    surface_disposition: SurfaceDisposition,
    activity_date: date | None,
    expected_confidence: Confidence,
) -> None:
    branch = f"codex/lambdarank-{surface_disposition.value}-only"
    explicit_review_date = date(2026, 7, 11)
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=surface_disposition,
                reason="Explicitly removed from live continuation.",
                next_action="Keep noncanonical.",
                last_reviewed=explicit_review_date,
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[] if activity_date is None else [(branch, activity_date)],
    )

    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == ""
    assert workstream.status == WorkstreamStatus.STALE
    assert workstream.rule == "no-live-surfaces-stale"
    assert workstream.confidence == expected_confidence
    assert workstream.last_reviewed == explicit_review_date
    assert explicit_review_date.isoformat() not in workstream.evidence
    if activity_date is None:
        assert "unavailable" in workstream.evidence.lower()
        assert "conflict" not in workstream.evidence.lower()
    else:
        assert activity_date.isoformat() in workstream.evidence
        assert "older than 30 days" in workstream.evidence
        assert "conflict" not in workstream.evidence.lower()


@pytest.mark.parametrize(
    "surface_disposition",
    [SurfaceDisposition.STALE, SurfaceDisposition.PARKED],
)
@pytest.mark.parametrize(
    ("activity_date", "expected_confidence"),
    [
        (date(2026, 7, 10), Confidence.LOW),
        (None, Confidence.LOW),
        (date(2026, 5, 1), Confidence.MEDIUM),
    ],
)
def test_phase2_noncanonical_only_surface_always_gets_stale_workstream(
    surface_disposition: SurfaceDisposition,
    activity_date: date | None,
    expected_confidence: Confidence,
) -> None:
    branch = f"codex/lambdarank-{surface_disposition.value}-phase2"
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=surface_disposition,
                reason="Explicitly removed from live continuation.",
                next_action="Keep noncanonical.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )
    branch_dates = {} if activity_date is None else {branch: activity_date}

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates=branch_dates,
        github_evidence=None,
    )

    assert decisions.surfaces[branch].disposition == surface_disposition
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.STALE
    assert workstream.canonical_surface == ""
    assert workstream.confidence == expected_confidence
    assert workstream.status != WorkstreamStatus.NEEDS_USER_DECISION
    if activity_date is None or activity_date > date(2026, 6, 12):
        assert any(alternative.startswith("active:") for alternative in workstream.alternatives)


def test_competing_surfaces_choose_newest_and_lexical_tie_without_blocking() -> None:
    surfaces = [
        _surface("codex/lambdarank-alpha"),
        _surface("codex/lambdarank-zeta"),
    ]
    activity = [
        ("codex/lambdarank-alpha", date(2026, 7, 10)),
        ("codex/lambdarank-zeta", date(2026, 7, 10)),
    ]

    decisions = compute_auto_decisions(
        surfaces=surfaces,
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[surface.branch for surface in surfaces]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=activity,
    )

    chosen = decisions.surfaces["codex/lambdarank-alpha"]
    rejected = decisions.surfaces["codex/lambdarank-zeta"]
    assert chosen.disposition == SurfaceDisposition.CANONICAL
    assert chosen.rule == "newest-live-surface"
    assert chosen.confidence == Confidence.LOW
    assert chosen.alternatives == ("codex/lambdarank-zeta",)
    assert rejected.disposition == SurfaceDisposition.STALE
    assert rejected.confidence == Confidence.LOW
    assert decisions.workstreams["LambdaRankIC"].canonical_surface == "codex/lambdarank-alpha"
    assert decisions.workstreams["LambdaRankIC"].status == WorkstreamStatus.ACTIVE
    assert decisions.workstreams["LambdaRankIC"].confidence == Confidence.LOW
    assert decisions.low_confidence_decisions


def test_explicit_canonical_surface_controls_effective_selection() -> None:
    overridden = "codex/lambdarank-override"
    newer = "codex/lambdarank-newer"
    registry = DecisionRegistry(
        surfaces={
            overridden: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Explicit canonical continuation.",
                next_action="Continue here.",
                last_reviewed=date(2026, 7, 12),
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(overridden), _surface(newer)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[overridden, newer]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[
            (overridden, date(2026, 7, 10)),
            (newer, date(2026, 7, 11)),
        ],
    )
    effective = overlay_auto_decisions(registry, decisions)

    assert decisions.surfaces[overridden].disposition == SurfaceDisposition.CANONICAL
    assert decisions.surfaces[newer].disposition == SurfaceDisposition.STALE
    assert f"newest-by-activity: {newer}" in decisions.surfaces[overridden].alternatives
    assert decisions.workstreams["LambdaRankIC"].canonical_surface == overridden
    canonical = [
        branch
        for branch, decision in effective.surfaces.items()
        if decision.disposition == SurfaceDisposition.CANONICAL
    ]
    assert canonical == [overridden]


def test_explicit_canonical_merged_branch_prevents_workstream_completion() -> None:
    branch = "codex/lambdarank-merged-but-canonical"
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Explicit continuation remains canonical.",
                next_action="Continue from this surface.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[], branches=[branch]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].disposition == SurfaceDisposition.ARCHIVE
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == branch
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert any("archive" in alternative for alternative in workstream.alternatives)


def test_explicit_canonical_nonbranch_surface_prevents_workstream_completion() -> None:
    branch = "detached@a2684d2"
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Detached investigation is explicit continuation.",
                next_action="Continue from detached worktree.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )
    surface = TopologySurface(
        branch=branch,
        label=f"`{branch}` @ `C:/repo` (detached)",
        provenance="detached",
    )

    decisions = compute_auto_decisions(
        surfaces=[surface],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].rule == "non-branch-surface"
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == branch
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.LOW


def test_unknown_surface_falls_back_to_git_hygiene() -> None:
    decisions = compute_auto_decisions(
        surfaces=[_surface("codex/unmapped-topic")],
        workstreams=[
            _seed("LambdaRankIC", "lambdarank"),
            _seed("Git and worktree hygiene", "cockpit"),
        ],
        topology=_topology(unmerged=["codex/unmapped-topic"]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[("codex/unmapped-topic", date(2026, 7, 11))],
    )

    assert decisions.surfaces["codex/unmapped-topic"].workstreams == ("Git and worktree hygiene",)


def test_detached_surface_does_not_infer_merged_branch() -> None:
    surface = TopologySurface(
        branch="detached@a2684d2",
        label="`detached@a2684d2` @ `C:/repo` (detached)",
        provenance="detached",
    )
    registry = DecisionRegistry(
        surfaces={
            surface.branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Detached worktree is explicit continuation.",
                next_action="Continue investigation.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[surface],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
    )

    detached = decisions.surfaces[surface.branch]
    assert detached.disposition != SurfaceDisposition.ARCHIVE
    assert detached.rule == "non-branch-surface"
    assert detached.confidence == Confidence.LOW
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == surface.branch
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.LOW


def test_worktree_only_surface_does_not_infer_merged_branch() -> None:
    surface = TopologySurface(
        branch="codex/worktree-only",
        label="`codex/worktree-only` @ `C:/other-worktree`",
        provenance="worktree",
    )
    registry = DecisionRegistry(
        surfaces={
            surface.branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Worktree-only surface is explicit continuation.",
                next_action="Continue from worktree.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )

    decisions = compute_auto_decisions(
        surfaces=[surface],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
    )

    worktree_only = decisions.surfaces[surface.branch]
    assert worktree_only.disposition != SurfaceDisposition.ARCHIVE
    assert worktree_only.rule == "non-branch-surface"
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.canonical_surface == surface.branch
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.confidence == Confidence.LOW


def test_unknown_merged_branch_archives_without_generating_hygiene_status() -> None:
    branch = "codex/unmapped-merged"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("Git and worktree hygiene", "cockpit")],
        topology=_topology(unmerged=[], branches=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 10))],
    )

    assert decisions.surfaces[branch].disposition == SurfaceDisposition.ARCHIVE
    assert decisions.surfaces[branch].workstreams == ("Git and worktree hygiene",)
    assert "Git and worktree hygiene" not in decisions.workstreams


def test_surface_matching_rejects_partial_token_substrings() -> None:
    branch = "codex/research-notes"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[
            _seed("Search work", "search"),
            _seed("Git and worktree hygiene", "cockpit"),
        ],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].workstreams == ("Git and worktree hygiene",)
    assert "Search work" not in decisions.workstreams


def test_surface_matching_selects_one_most_specific_primary_workstream() -> None:
    branch = "codex/regime-csv-no-backfill"
    regime = _seed("Regime", "regime")
    regime_csv = _seed("Regime CSV", "regime-csv")
    surface = _surface(branch)
    decisions = compute_auto_decisions(
        surfaces=[surface],
        workstreams=[
            regime,
            regime_csv,
            _seed("Git and worktree hygiene", "cockpit"),
        ],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].workstreams == ("Regime CSV",)
    assert set(decisions.workstreams) == {"Regime CSV"}
    effective = overlay_auto_decisions(DecisionRegistry(), decisions)
    assert (
        _workstream_surfaces(
            regime,
            [surface],
            effective,
            strict_surface_assignments=True,
        )
        == []
    )
    assert _workstream_surfaces(
        regime_csv,
        [surface],
        effective,
        strict_surface_assignments=True,
    ) == [surface]


def test_overlapping_terms_do_not_make_loser_canonical_in_second_workstream() -> None:
    older = "codex/shared-older"
    newer = "codex/shared-newer"
    decisions = compute_auto_decisions(
        surfaces=[_surface(older), _surface(newer)],
        workstreams=[
            _seed("Alpha", "shared"),
            _seed("Beta", "shared"),
        ],
        topology=_topology(unmerged=[older, newer]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[
            (older, date(2026, 7, 10)),
            (newer, date(2026, 7, 11)),
        ],
    )

    assert decisions.surfaces[older].workstreams == ("Alpha",)
    assert decisions.surfaces[older].disposition == SurfaceDisposition.STALE
    assert decisions.surfaces[newer].workstreams == ("Alpha",)
    assert decisions.surfaces[newer].disposition == SurfaceDisposition.CANONICAL
    assert "Beta" not in decisions.workstreams


def test_explicit_surface_mapping_preserves_genuine_multi_workstream_assignment() -> None:
    branch = "codex/shared-explicit"
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("Alpha", "Beta"),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Shared integration surface.",
                next_action="Continue shared validation.",
                last_reviewed=date(2026, 7, 11),
            )
        }
    )
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[
            _seed("Alpha", "shared"),
            _seed("Beta", "shared"),
        ],
        topology=_topology(unmerged=[branch]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )

    assert decisions.surfaces[branch].workstreams == ("Alpha", "Beta")
    assert set(decisions.workstreams) == {"Alpha", "Beta"}


def test_open_pr_is_canonical_but_merged_topology_still_archives() -> None:
    live = "codex/lambdarank-live"
    merged = "codex/lambdarank-merged"
    github = GitHubEvidence(
        pull_requests=(
            _pr(10, live, state="open"),
            _pr(11, merged, state="open"),
        )
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(live), _surface(merged)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[live], branches=[live, merged]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={live: date(2026, 5, 1), merged: date(2026, 5, 1)},
        github_evidence=github,
    )

    assert decisions.surfaces[live].disposition == SurfaceDisposition.CANONICAL
    assert decisions.surfaces[live].rule == "open-pr-canonical"
    assert decisions.surfaces[live].confidence == Confidence.HIGH
    assert "#10" in decisions.surfaces[live].evidence
    assert "https://github.example/pull/10" in decisions.surfaces[live].evidence
    assert decisions.surfaces[merged].disposition == SurfaceDisposition.ARCHIVE
    assert decisions.surfaces[merged].rule == "merged-into-main"


def test_stale_open_pr_cannot_reactivate_merged_surface_workstream() -> None:
    branch = "codex/lambdarank-merged"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[], branches=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 5, 1)},
        github_evidence=GitHubEvidence(pull_requests=(_pr(12, branch, state="open"),)),
    )

    surface = decisions.surfaces[branch]
    assert surface.disposition == SurfaceDisposition.ARCHIVE
    assert surface.confidence == Confidence.HIGH
    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.DONE
    assert workstream.canonical_surface == ""


def test_stale_surface_confidence_tracks_online_and_offline_pr_evidence() -> None:
    branch = "codex/lambdarank-old"
    inputs = {
        "surfaces": [_surface(branch)],
        "workstreams": [_seed("LambdaRankIC", "lambdarank")],
        "topology": _topology(unmerged=[branch]),
        "registry": DecisionRegistry(),
        "aliases": {},
        "run_date": date(2026, 7, 12),
        "recent_branches": [],
        "branch_commit_dates": {branch: date(2026, 5, 1)},
    }

    online = compute_auto_decisions(github_evidence=GitHubEvidence(), **inputs)
    offline = compute_auto_decisions(github_evidence=None, **inputs)

    assert online.surfaces[branch].disposition == SurfaceDisposition.STALE
    assert online.surfaces[branch].confidence == Confidence.HIGH
    assert "confirmed no open PR" in online.surfaces[branch].evidence
    assert offline.surfaces[branch].disposition == SurfaceDisposition.STALE
    assert offline.surfaces[branch].confidence == Confidence.MEDIUM
    assert "PR state unknown" in offline.surfaces[branch].evidence
    assert any("open PR" in item for item in offline.surfaces[branch].alternatives)
    assert online.workstreams["LambdaRankIC"].confidence == Confidence.HIGH
    assert offline.workstreams["LambdaRankIC"].confidence == Confidence.MEDIUM


@pytest.mark.parametrize("branch_date", [None, date(2026, 7, 20)])
def test_missing_or_future_branch_date_never_causes_stale_by_age(
    branch_date: date | None,
) -> None:
    branch = "codex/lambdarank-undated"
    branch_dates = {} if branch_date is None else {branch: branch_date}
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates=branch_dates,
        github_evidence=GitHubEvidence(),
    )

    assert decisions.surfaces[branch].disposition == SurfaceDisposition.CANONICAL
    assert decisions.workstreams["LambdaRankIC"].status == WorkstreamStatus.ACTIVE


def test_open_pr_prevents_stale_surface_and_makes_workstream_active() -> None:
    branch = "codex/lambdarank-old-open-pr"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 5, 1)},
        github_evidence=GitHubEvidence(pull_requests=(_pr(20, branch, state="open"),)),
    )

    assert decisions.surfaces[branch].disposition == SurfaceDisposition.CANONICAL
    assert decisions.workstreams["LambdaRankIC"].status == WorkstreamStatus.ACTIVE
    assert decisions.workstreams["LambdaRankIC"].rule == "open-pr-active"
    assert decisions.workstreams["LambdaRankIC"].confidence == Confidence.HIGH


def test_competing_open_prs_choose_one_deterministic_canonical_surface() -> None:
    older = "codex/lambdarank-pr-older"
    newer = "codex/lambdarank-pr-newer"
    github = GitHubEvidence(
        pull_requests=(
            PullRequestEvidence(
                number=21,
                head_ref=older,
                url="https://github.example/pull/21",
                state="open",
                is_draft=False,
                merged_at=None,
                updated_at=date(2026, 7, 10),
            ),
            PullRequestEvidence(
                number=22,
                head_ref=newer,
                url="https://github.example/pull/22",
                state="open",
                is_draft=False,
                merged_at=None,
                updated_at=date(2026, 7, 11),
            ),
        )
    )

    decisions = compute_auto_decisions(
        surfaces=[_surface(older), _surface(newer)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[older, newer]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={
            older: date(2026, 7, 10),
            newer: date(2026, 7, 11),
        },
        github_evidence=github,
    )

    assert decisions.surfaces[newer].disposition == SurfaceDisposition.CANONICAL
    assert decisions.surfaces[older].disposition == SurfaceDisposition.STALE
    assert decisions.surfaces[newer].confidence == Confidence.LOW
    assert decisions.surfaces[older].confidence == Confidence.LOW
    assert "PR #22" in decisions.surfaces[older].evidence
    assert "https://github.example/pull/22" in decisions.surfaces[older].evidence
    assert decisions.workstreams["LambdaRankIC"].canonical_surface == newer
    assert decisions.workstreams["LambdaRankIC"].confidence == Confidence.LOW


@pytest.mark.parametrize(
    ("number", "labels", "expected_status", "expected_rule"),
    [
        (30, ("blocked",), WorkstreamStatus.BLOCKED, "blocked-issue"),
        (
            31,
            ("needs-info",),
            WorkstreamStatus.BLOCKED,
            "blocked-issue",
        ),
        (
            32,
            ("ready-for-agent",),
            WorkstreamStatus.READY_FOR_AGENT,
            "ready-issue",
        ),
    ],
)
def test_issue_labels_drive_blocked_and_ready_statuses(
    number: int,
    labels: tuple[str, ...],
    expected_status: WorkstreamStatus,
    expected_rule: str,
) -> None:
    issue = _issue(number, "LambdaRankIC", "open", labels)
    decisions = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(issues=(issue,)),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == expected_status
    assert decision.rule == expected_rule
    assert decision.confidence == Confidence.HIGH
    assert f"#{issue.number}" in decision.evidence
    assert issue.url in decision.evidence


def test_merged_pr_and_closed_issue_support_done_without_live_surface() -> None:
    branch = "codex/lambdarank-merged"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[], branches=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 7, 1)},
        github_evidence=GitHubEvidence(
            pull_requests=(_pr(40, branch, state="merged", merged_at=date(2026, 7, 2)),),
            issues=(_issue(41, "LambdaRankIC", "closed", ()),),
        ),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.DONE
    assert decision.rule == "github-complete"
    assert decision.confidence == Confidence.HIGH
    assert "#40" in decision.evidence
    assert "#41" in decision.evidence


def test_closed_issue_cannot_mark_done_while_matching_unlabeled_issue_is_open() -> None:
    decisions = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(
            issues=(
                _issue(42, "LambdaRankIC", "closed", ()),
                _issue(43, "LambdaRankIC", "open", ()),
            )
        ),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.ACTIVE
    assert decision.confidence == Confidence.LOW
    assert "issue #43" in decision.evidence
    assert any(alternative.startswith("done:") for alternative in decision.alternatives)


def test_archived_surface_with_unlabeled_open_issue_is_active_low() -> None:
    branch = "codex/lambdarank-merged"
    issue = _issue(44, "LambdaRankIC", "open", ())
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[], branches=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 7, 1)},
        github_evidence=GitHubEvidence(issues=(issue,)),
    )

    assert decisions.surfaces[branch].disposition == SurfaceDisposition.ARCHIVE
    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.ACTIVE
    assert decision.rule == "open-issue-active"
    assert decision.confidence == Confidence.LOW
    assert f"issue #{issue.number}" in decision.evidence
    assert issue.url in decision.evidence
    assert decision.alternatives == (
        "done: all known surfaces are merged or archived, but matching open issue "
        "#44 prevents completion",
    )


def test_no_surface_with_unlabeled_open_issue_is_active_high() -> None:
    issue = _issue(46, "LambdaRankIC", "open", ())
    decisions = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(issues=(issue,)),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.ACTIVE
    assert decision.rule == "open-issue-active"
    assert decision.confidence == Confidence.HIGH
    assert f"issue #{issue.number}" in decision.evidence
    assert issue.url in decision.evidence
    assert decision.alternatives == ()


def test_no_surface_with_neutral_labeled_open_issue_is_active_high() -> None:
    issue = _issue(47, "LambdaRankIC", "open", ("enhancement",))
    decisions = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(issues=(issue,)),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.ACTIVE
    assert decision.rule == "open-issue-active"
    assert decision.confidence == Confidence.HIGH
    assert f"issue #{issue.number}" in decision.evidence
    assert issue.url in decision.evidence


def test_archived_surface_with_closed_only_issue_is_done_high() -> None:
    branch = "codex/lambdarank-merged"
    issue = _issue(45, "LambdaRankIC", "closed", ())
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[], branches=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 7, 1)},
        github_evidence=GitHubEvidence(issues=(issue,)),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.DONE
    assert decision.rule == "github-complete"
    assert decision.confidence == Confidence.HIGH
    assert f"issue #{issue.number}" in decision.evidence
    assert issue.url in decision.evidence


def test_auto_policy_emits_one_terminal_decision_per_known_non_hygiene_seed() -> None:
    recent_parked = "codex/recent-parked"
    missing_stale = "codex/missing-stale"
    registry = DecisionRegistry(
        surfaces={
            recent_parked: SurfaceDecision(
                workstreams=("Recent parked",),
                disposition=SurfaceDisposition.PARKED,
                reason="Explicit parked surface.",
                next_action="Keep parked.",
                last_reviewed=date(2026, 7, 11),
            ),
            missing_stale: SurfaceDecision(
                workstreams=("Missing stale",),
                disposition=SurfaceDisposition.STALE,
                reason="Explicit stale surface.",
                next_action="Keep stale.",
                last_reviewed=date(2026, 7, 11),
            ),
        }
    )
    workstreams = [
        _seed("Issue active", "issue-active"),
        _seed("Recent parked", "recent-parked"),
        _seed("Missing stale", "missing-stale"),
        _seed("Git and worktree hygiene", "hygiene"),
    ]

    decisions = compute_auto_decisions(
        surfaces=[_surface(recent_parked), _surface(missing_stale)],
        workstreams=workstreams,
        topology=_topology(unmerged=[recent_parked, missing_stale]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={recent_parked: date(2026, 7, 10)},
        github_evidence=GitHubEvidence(
            issues=(_issue(48, "Issue active", "open", ("enhancement",)),)
        ),
    )

    assert set(decisions.workstreams) == {
        "Issue active",
        "Recent parked",
        "Missing stale",
    }
    assert all(
        decision.status != WorkstreamStatus.NEEDS_USER_DECISION
        for decision in decisions.workstreams.values()
    )


def test_conflicting_open_pr_and_blocked_issue_choose_active_with_alternative() -> None:
    branch = "codex/lambdarank-conflict"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 7, 10)},
        github_evidence=GitHubEvidence(
            pull_requests=(_pr(50, branch, state="open"),),
            issues=(_issue(51, "LambdaRankIC", "open", ("blocked",)),),
        ),
    )

    decision = decisions.workstreams["LambdaRankIC"]
    assert decision.status == WorkstreamStatus.ACTIVE
    assert decision.confidence == Confidence.LOW
    assert any(
        "blocked" in alternative and "#51" in alternative for alternative in decision.alternatives
    )


def test_issue_title_matching_rejects_substrings_and_marks_ambiguity_low() -> None:
    substring = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed("Search", "search")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(issues=(_issue(60, "Research", "open", ("blocked",)),)),
    )
    assert "Search" not in substring.workstreams

    ambiguous = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed("Alpha", "shared"), _seed("Beta", "shared")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(issues=(_issue(61, "Shared", "open", ("blocked",)),)),
    )
    assert ambiguous.workstreams["Alpha"].status == WorkstreamStatus.BLOCKED
    assert ambiguous.workstreams["Beta"].status == WorkstreamStatus.BLOCKED
    assert ambiguous.workstreams["Alpha"].confidence == Confidence.LOW
    assert "ambiguous" in ambiguous.workstreams["Alpha"].evidence.lower()
    ambiguity = (
        "association ambiguity: issue #61 (https://github.example/issues/61) "
        "matches workstreams Alpha, Beta",
    )
    assert ambiguous.workstreams["Alpha"].alternatives == ambiguity
    assert ambiguous.workstreams["Beta"].alternatives == ambiguity


@pytest.mark.parametrize(
    ("name", "term", "title"),
    [
        ("Regime CSV contract", "csv", "Export CSV diagnostics"),
        ("Docs evidence", "docs", "Update docs for another subsystem"),
        ("Research evidence", "research", "Research unrelated vendor options"),
    ],
)
def test_generic_singleton_branch_terms_do_not_associate_unrelated_issues(
    name: str,
    term: str,
    title: str,
) -> None:
    decisions = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed(name, term)],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(issues=(_issue(62, title, "open", ("blocked",)),)),
    )

    assert name not in decisions.workstreams


def test_issue_title_matching_accepts_full_workstream_slug_and_explicit_alias() -> None:
    name = "Regime CSV contract"
    decisions = compute_auto_decisions(
        surfaces=[],
        workstreams=[_seed(name, "csv")],
        topology=_topology(unmerged=[]),
        registry=DecisionRegistry(),
        aliases={"regime-contract": name},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={},
        github_evidence=GitHubEvidence(
            issues=(
                _issue(63, "Regime-CSV-contract", "open", ("blocked",)),
                _issue(64, "Follow up regime-contract", "open", ("needs-info",)),
            )
        ),
    )

    decision = decisions.workstreams[name]
    assert decision.status == WorkstreamStatus.BLOCKED
    assert "#63" in decision.evidence
    assert "#64" in decision.evidence
    assert decision.confidence == Confidence.HIGH


def test_mixed_case_pr_head_matches_same_case_local_branch_exactly() -> None:
    branch = "codex/FeatureWork"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("Feature work", "featurework")],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 5, 1)},
        github_evidence=GitHubEvidence(pull_requests=(_pr(65, branch, state="open"),)),
    )

    assert decisions.surfaces[branch].rule == "open-pr-canonical"
    assert decisions.workstreams["Feature work"].status == WorkstreamStatus.ACTIVE
    assert "#65" in decisions.workstreams["Feature work"].evidence


def test_explicit_surface_override_still_wins_over_open_pr_generated_choice() -> None:
    branch = "codex/lambdarank-overridden"
    registry = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.STALE,
                reason="User override.",
                next_action="Keep stale.",
                last_reviewed=date(2026, 7, 12),
            )
        }
    )
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=registry,
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[],
        branch_commit_dates={branch: date(2026, 5, 1)},
        github_evidence=GitHubEvidence(pull_requests=(_pr(70, branch, state="open"),)),
    )

    effective = overlay_auto_decisions(registry, decisions)
    assert decisions.surfaces[branch].association_basis == "explicit-surface"
    assert effective.surfaces[branch].disposition == SurfaceDisposition.STALE


def test_auto_decision_file_schema_is_sorted_and_rerun_is_byte_identical(tmp_path) -> None:
    decisions = compute_auto_decisions(
        surfaces=[
            _surface("codex/zeta-lambdarank"),
            _surface("codex/alpha-portfolio"),
        ],
        workstreams=[
            _seed("LambdaRankIC", "lambdarank"),
            _seed("Portfolio-IC", "portfolio"),
        ],
        topology=_topology(unmerged=["codex/zeta-lambdarank", "codex/alpha-portfolio"]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[
            ("codex/zeta-lambdarank", date(2026, 7, 11)),
            ("codex/alpha-portfolio", date(2026, 7, 10)),
        ],
    )

    path = write_auto_decisions(tmp_path, decisions)
    first = path.read_bytes()
    write_auto_decisions(tmp_path, decisions)
    second = path.read_bytes()
    payload = json.loads(first)

    assert path == tmp_path / AUTO_DECISIONS_PATH
    assert first == second
    assert first.endswith(b"\n")
    assert payload["format_version"] == 2
    assert list(payload["surfaces"]) == sorted(payload["surfaces"])
    assert list(payload["workstreams"]) == sorted(payload["workstreams"])
    assert "run_timestamp" not in first.decode()
    assert payload["surfaces"]["codex/zeta-lambdarank"] == {
        "workstreams": ["LambdaRankIC"],
        "disposition": "canonical",
        "rule": "unique-live-surface",
        "evidence": (
            "Only live surface for LambdaRankIC; branch remains unmerged; "
            "tip committer date 2026-07-11."
        ),
        "confidence": "high",
        "association_basis": "branch-term",
        "alternatives": [],
        "last_reviewed": "2026-07-11",
    }
    assert read_auto_decisions(tmp_path) == decisions


def test_auto_decision_reader_accepts_v1_surface_without_association_basis(tmp_path) -> None:
    path = tmp_path / AUTO_DECISIONS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "surfaces": {
                    "codex/legacy-grounding": {
                        "workstreams": ["Legacy stream"],
                        "disposition": "canonical",
                        "rule": "unique-live-surface",
                        "evidence": "Legacy Phase 1 decision.",
                        "confidence": "high",
                        "alternatives": [],
                        "last_reviewed": "2026-07-12",
                    }
                },
                "workstreams": {},
            }
        ),
        encoding="utf-8",
    )

    decisions = read_auto_decisions(tmp_path)

    assert decisions.surfaces["codex/legacy-grounding"].association_basis == "unclassified"


def test_missing_activity_date_does_not_churn_metadata_across_run_dates(tmp_path) -> None:
    branch = "codex/lambdarank-no-local-date"
    inputs = {
        "surfaces": [_surface(branch)],
        "workstreams": [_seed("LambdaRankIC", "lambdarank")],
        "topology": _topology(unmerged=[branch]),
        "registry": DecisionRegistry(),
        "aliases": {},
        "recent_branches": [],
    }

    first = compute_auto_decisions(run_date=date(2026, 7, 12), **inputs)
    first_path = write_auto_decisions(tmp_path, first)
    first_bytes = first_path.read_bytes()
    second = compute_auto_decisions(run_date=date(2026, 7, 20), **inputs)
    second_path = write_auto_decisions(tmp_path, second)

    workstream = first.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.rule == "activity-date-unknown"
    assert workstream.confidence == Confidence.LOW
    assert "unavailable" in workstream.evidence.lower()
    assert any("stale" in alternative for alternative in workstream.alternatives)
    assert first == second
    assert first_bytes == second_path.read_bytes()


def test_future_activity_date_does_not_claim_workstream_stale() -> None:
    branch = "codex/lambdarank-future-date"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 20))],
    )

    workstream = decisions.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.rule == "activity-date-future"
    assert workstream.confidence == Confidence.LOW
    assert "future" in workstream.evidence.lower()
    assert any("stale" in alternative for alternative in workstream.alternatives)


def test_auto_decision_reader_treats_missing_and_malformed_files_as_empty(tmp_path) -> None:
    assert read_auto_decisions(tmp_path).surfaces == {}
    assert read_auto_decisions(tmp_path).workstreams == {}

    path = tmp_path / AUTO_DECISIONS_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert read_auto_decisions(tmp_path).surfaces == {}

    path.write_text(json.dumps({"format_version": 2, "surfaces": {}, "workstreams": {}}))
    assert read_auto_decisions(tmp_path).workstreams == {}


@pytest.mark.parametrize("boolean_version", [True, False])
def test_auto_decision_reader_rejects_boolean_format_version(
    tmp_path,
    boolean_version: bool,
) -> None:
    branch = "codex/lambdarank-format"
    decisions = compute_auto_decisions(
        surfaces=[_surface(branch)],
        workstreams=[_seed("LambdaRankIC", "lambdarank")],
        topology=_topology(unmerged=[branch]),
        registry=DecisionRegistry(),
        aliases={},
        run_date=date(2026, 7, 12),
        recent_branches=[(branch, date(2026, 7, 11))],
    )
    path = write_auto_decisions(tmp_path, decisions)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format_version"] = boolean_version
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = read_auto_decisions(tmp_path)
    assert parsed.surfaces == {}
    assert parsed.workstreams == {}


def _surface(branch: str) -> TopologySurface:
    return TopologySurface(branch=branch, label=f"`{branch}` (local)", provenance="local")


def _seed(name: str, *terms: str) -> WorkstreamSeed:
    return WorkstreamSeed(
        name=name,
        status=WorkstreamStatus.ACTIVE,
        next_action="Continue.",
        branch_terms=terms,
        association_basis="branch-term",
    )


def _topology(
    *,
    unmerged: list[str],
    branches: list[str] | None = None,
    worktrees: list[WorktreeEvidence] | None = None,
) -> GitTopologySnapshot:
    return GitTopologySnapshot(
        current_branch="main",
        status_header="## main",
        origin_main_ahead=0,
        origin_main_behind=0,
        branches=["main", *(branches if branches is not None else unmerged)],
        unmerged_branches=unmerged,
        worktrees=worktrees or [],
    )


def _pr(
    number: int,
    branch: str,
    *,
    state: str,
    merged_at: date | None = None,
) -> PullRequestEvidence:
    return PullRequestEvidence(
        number=number,
        head_ref=branch,
        url=f"https://github.example/pull/{number}",
        state=state,
        is_draft=False,
        merged_at=merged_at,
        updated_at=date(2026, 7, 11),
    )


def _issue(
    number: int,
    title: str,
    state: str,
    labels: tuple[str, ...],
) -> IssueEvidence:
    return IssueEvidence(
        number=number,
        title=title,
        url=f"https://github.example/issues/{number}",
        state=state,
        labels=labels,
        updated_at=date(2026, 7, 11),
    )
