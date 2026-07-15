from __future__ import annotations

import json
from datetime import date

import pytest

from cockpit.decisions import (
    DECISION_REGISTRY_PATH,
    DecisionRegistry,
    SurfaceDecision,
    SurfaceDisposition,
    WorkstreamDecision,
    load_decision_registry,
    overlay_auto_decisions,
    read_registry_aliases,
    read_registry_workstream_names,
)
from cockpit.models import (
    AutoDecisionSet,
    AutoDisposition,
    AutoWorkstreamDecision,
    Confidence,
    WorkstreamStatus,
)


def test_load_decision_registry_parses_versioned_contract(tmp_path) -> None:
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "workstreams": {
                "LambdaRankIC": {
                    "status": "active",
                    "canonical_surface": "PR #65 / codex/canonical-lambdarank",
                    "reason": "Recovery guardrails are the reviewed continuation.",
                    "next_action": "Fix lint before runtime work.",
                    "last_reviewed": "2026-07-09",
                }
            },
            "surfaces": {
                "codex/canonical-lambdarank": {
                    "workstreams": ["LambdaRankIC"],
                    "disposition": "canonical",
                    "reason": "Reviewed canonical branch.",
                    "next_action": "Continue through PR #65.",
                    "last_reviewed": "2026-07-09",
                },
                "codex/old-lambdarank": {
                    "workstreams": ["LambdaRankIC"],
                    "disposition": "archive",
                    "reason": "Superseded by the recovery branch.",
                    "next_action": "Remove only after cleanup approval.",
                    "last_reviewed": "2026-07-09",
                },
            },
        },
    )

    registry = load_decision_registry(tmp_path, known_workstreams={"LambdaRankIC"})

    workstream = registry.workstreams["LambdaRankIC"]
    assert workstream.status == WorkstreamStatus.ACTIVE
    assert workstream.canonical_surface == "PR #65 / codex/canonical-lambdarank"
    assert workstream.last_reviewed == date(2026, 7, 9)
    assert registry.is_reviewed("LambdaRankIC", "codex/canonical-lambdarank")
    assert registry.surfaces["codex/old-lambdarank"].disposition == SurfaceDisposition.ARCHIVE
    assert workstream.provenance == "override"
    assert registry.surfaces["codex/old-lambdarank"].provenance == "override"


def test_overlay_auto_decisions_applies_explicit_overrides_last() -> None:
    auto = AutoDecisionSet(
        surfaces={
            "codex/lambdarank": AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                rule="unique-live-surface",
                evidence="Only live surface.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 10),
            )
        },
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface="codex/lambdarank",
                rule="recent-activity-active",
                evidence="Recent activity.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 10),
            )
        },
    )
    overrides = DecisionRegistry(
        surfaces={
            "codex/lambdarank": SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.PARKED,
                reason="Pause explicitly.",
                next_action="Wait.",
                last_reviewed=date(2026, 7, 11),
            )
        },
        workstreams={
            "LambdaRankIC": WorkstreamDecision(
                status=WorkstreamStatus.PARKED,
                canonical_surface="codex/lambdarank",
                reason="Pause explicitly.",
                next_action="Wait.",
                last_reviewed=date(2026, 7, 11),
            )
        },
        aliases={"lambda": "LambdaRankIC"},
    )

    effective = overlay_auto_decisions(overrides, auto)

    assert effective.surfaces["codex/lambdarank"].disposition == SurfaceDisposition.PARKED
    assert effective.surfaces["codex/lambdarank"].provenance == "override"
    assert effective.workstreams["LambdaRankIC"].status == WorkstreamStatus.PARKED
    assert effective.workstreams["LambdaRankIC"].provenance == "override"
    assert effective.aliases == overrides.aliases


def test_overlay_explicit_canonical_surface_reconciles_effective_workstream_pointer() -> None:
    generated_branch = "codex/lambdarank-generated"
    explicit_branch = "codex/lambdarank-explicit"
    auto = AutoDecisionSet(
        surfaces={
            generated_branch: AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                rule="newest-live-surface",
                evidence="Generated policy selected the newest branch.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 13),
            )
        },
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface=generated_branch,
                rule="recent-activity-active",
                evidence="Generated policy selected the newest branch.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 13),
            )
        },
    )
    overrides = DecisionRegistry(
        surfaces={
            explicit_branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Reviewed continuation wins.",
                next_action="Continue here.",
                last_reviewed=date(2026, 7, 14),
            )
        }
    )

    effective = overlay_auto_decisions(overrides, auto)

    assert auto.workstreams["LambdaRankIC"].canonical_surface == generated_branch
    assert effective.workstreams["LambdaRankIC"].canonical_surface == explicit_branch
    assert effective.surfaces[generated_branch].disposition == SurfaceDisposition.STALE
    assert effective.surfaces[explicit_branch].disposition == SurfaceDisposition.CANONICAL


@pytest.mark.parametrize(
    "disposition",
    [SurfaceDisposition.PARKED, SurfaceDisposition.STALE],
)
def test_overlay_rejects_noncanonical_override_of_sole_generated_canonical(
    disposition: SurfaceDisposition,
) -> None:
    branch = "codex/lambdarank-generated"
    auto = AutoDecisionSet(
        surfaces={
            branch: AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                rule="unique-live-surface",
                evidence="Only live continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface=branch,
                rule="recent-activity-active",
                evidence="Only live continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
    )
    overrides = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=disposition,
                reason="Pause this surface.",
                next_action="Wait.",
                last_reviewed=date(2026, 7, 14),
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="No effective canonical surface for continuing workstream LambdaRankIC",
    ):
        overlay_auto_decisions(overrides, auto)


def test_overlay_allows_noncanonical_override_with_alternate_explicit_canonical() -> None:
    generated_branch = "codex/lambdarank-generated"
    explicit_branch = "codex/lambdarank-explicit"
    auto = AutoDecisionSet(
        surfaces={
            generated_branch: AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                rule="unique-live-surface",
                evidence="Generated continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface=generated_branch,
                rule="recent-activity-active",
                evidence="Generated continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
    )
    overrides = DecisionRegistry(
        surfaces={
            generated_branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.PARKED,
                reason="Pause the generated surface.",
                next_action="Use the reviewed continuation.",
                last_reviewed=date(2026, 7, 14),
            ),
            explicit_branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Reviewed continuation.",
                next_action="Continue here.",
                last_reviewed=date(2026, 7, 14),
            ),
        }
    )

    effective = overlay_auto_decisions(overrides, auto)

    assert effective.surfaces[generated_branch].disposition == SurfaceDisposition.PARKED
    assert effective.surfaces[explicit_branch].disposition == SurfaceDisposition.CANONICAL
    assert effective.workstreams["LambdaRankIC"].canonical_surface == explicit_branch


def test_overlay_allows_authoritative_explicit_workstream_route_outside_topology() -> None:
    workstream = "Colab operations"
    generated_branch = "codex/colab-gpu-utilization-hardening-20260620"
    explicit_route = "origin/main plus docs/workflows/COLAB_CHROME_CONTROL_GUIDE.md"
    auto = AutoDecisionSet(
        surfaces={
            generated_branch: AutoDisposition(
                workstreams=(workstream,),
                disposition=SurfaceDisposition.CANONICAL,
                rule="recent-activity-active",
                evidence="Generated branch continuation.",
                confidence=Confidence.LOW,
                alternatives=(),
                last_reviewed=date(2026, 6, 21),
            )
        },
        workstreams={
            workstream: AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface=generated_branch,
                rule="recent-activity-active",
                evidence="Generated branch continuation.",
                confidence=Confidence.LOW,
                alternatives=(),
                last_reviewed=date(2026, 6, 21),
            )
        },
    )
    explicit_workstream = WorkstreamDecision(
        status=WorkstreamStatus.READY_FOR_AGENT,
        canonical_surface=explicit_route,
        reason="The Chrome-control runbook is the canonical operational surface.",
        next_action="Use the runbook for the next approved Colab operation.",
        last_reviewed=date(2026, 7, 9),
    )
    registry = DecisionRegistry(
        workstreams={workstream: explicit_workstream},
        surfaces={
            generated_branch: SurfaceDecision(
                workstreams=(workstream,),
                disposition=SurfaceDisposition.ARCHIVE,
                reason="The runbook supersedes this historical branch.",
                next_action="Remove only during approved cleanup.",
                last_reviewed=date(2026, 7, 9),
            )
        },
    )

    effective = overlay_auto_decisions(registry, auto)

    assert effective.workstreams[workstream] == explicit_workstream
    assert effective.surfaces[generated_branch].disposition == SurfaceDisposition.ARCHIVE
    assert auto.workstreams[workstream].canonical_surface == generated_branch


def test_overlay_preserves_authoritative_external_route_with_generated_canonical() -> None:
    workstream = "Issue recovery"
    generated_branch = "codex/live"
    explicit_route = "issue #99 outside topology"
    auto_workstream = AutoWorkstreamDecision(
        status=WorkstreamStatus.ACTIVE,
        canonical_surface=generated_branch,
        rule="recent-activity-active",
        evidence="Generated branch continuation.",
        confidence=Confidence.HIGH,
        alternatives=(),
        last_reviewed=date(2026, 7, 14),
    )
    auto_surface = AutoDisposition(
        workstreams=(workstream,),
        disposition=SurfaceDisposition.CANONICAL,
        rule="unique-live-surface",
        evidence="Generated branch continuation.",
        confidence=Confidence.HIGH,
        alternatives=(),
        last_reviewed=date(2026, 7, 14),
    )
    auto = AutoDecisionSet(
        surfaces={generated_branch: auto_surface},
        workstreams={workstream: auto_workstream},
    )
    explicit_workstream = WorkstreamDecision(
        status=WorkstreamStatus.READY_FOR_AGENT,
        canonical_surface=explicit_route,
        reason="The issue is the reviewed continuation route.",
        next_action="Continue from the issue.",
        last_reviewed=date(2026, 7, 14),
    )
    registry = DecisionRegistry(workstreams={workstream: explicit_workstream})

    effective = overlay_auto_decisions(registry, auto)

    assert effective.workstreams[workstream].canonical_surface == explicit_route
    assert effective.surfaces[generated_branch].disposition == SurfaceDisposition.CANONICAL
    assert registry.workstreams[workstream] == explicit_workstream
    assert auto.workstreams[workstream] == auto_workstream
    assert auto.surfaces[generated_branch] == auto_surface


def test_overlay_rejects_multiple_explicit_canonical_surfaces_for_one_workstream() -> None:
    overrides = DecisionRegistry(
        surfaces={
            branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Explicit canonical continuation.",
                next_action="Continue here.",
                last_reviewed=date(2026, 7, 14),
            )
            for branch in ("codex/lambdarank-first", "codex/lambdarank-second")
        }
    )

    with pytest.raises(
        ValueError,
        match="Multiple explicit canonical surfaces for workstream LambdaRankIC",
    ):
        overlay_auto_decisions(overrides, AutoDecisionSet())


def test_overlay_reconciles_an_explicit_workstream_pointer_without_mutating_override() -> None:
    generated_branch = "codex/lambdarank-generated"
    explicit_branch = "codex/lambdarank-explicit"
    explicit_workstream = WorkstreamDecision(
        status=WorkstreamStatus.ACTIVE,
        canonical_surface=generated_branch,
        reason="Keep the workstream active.",
        next_action="Continue.",
        last_reviewed=date(2026, 7, 13),
    )
    auto = AutoDecisionSet(
        surfaces={
            generated_branch: AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.PARKED,
                rule="competing-live-surfaces",
                evidence="Generated topology keeps this branch parked.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface=generated_branch,
                rule="recent-activity-active",
                evidence="Generated continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
    )
    overrides = DecisionRegistry(
        workstreams={"LambdaRankIC": explicit_workstream},
        surfaces={
            explicit_branch: SurfaceDecision(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                reason="Reviewed continuation wins.",
                next_action="Continue here.",
                last_reviewed=date(2026, 7, 14),
            )
        },
    )

    effective = overlay_auto_decisions(overrides, auto)

    assert effective.workstreams["LambdaRankIC"].canonical_surface == explicit_branch
    assert effective.workstreams["LambdaRankIC"].provenance == "override"
    assert overrides.workstreams["LambdaRankIC"].canonical_surface == generated_branch


def test_overlay_cleared_surface_override_restores_explicit_workstream_pointer() -> None:
    stored_branch = "codex/lambdarank-stored"
    explicit_workstream = WorkstreamDecision(
        status=WorkstreamStatus.ACTIVE,
        canonical_surface=stored_branch,
        reason="Stored workstream continuation.",
        next_action="Continue.",
        last_reviewed=date(2026, 7, 13),
    )
    auto = AutoDecisionSet(
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface="codex/lambdarank-generated",
                rule="recent-activity-active",
                evidence="Generated continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        }
    )

    effective = overlay_auto_decisions(
        DecisionRegistry(workstreams={"LambdaRankIC": explicit_workstream}),
        auto,
    )

    assert effective.workstreams["LambdaRankIC"].canonical_surface == stored_branch
    assert explicit_workstream.canonical_surface == stored_branch


def test_overlay_preserves_explicit_route_outside_generated_topology() -> None:
    generated_branch = "codex/lambdarank-generated"
    stored_branch = "codex/lambdarank-explicit"
    explicit_workstream = WorkstreamDecision(
        status=WorkstreamStatus.ACTIVE,
        canonical_surface=stored_branch,
        reason="Stored workstream continuation.",
        next_action="Continue.",
        last_reviewed=date(2026, 7, 13),
    )
    auto = AutoDecisionSet(
        surfaces={
            generated_branch: AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                rule="unique-live-surface",
                evidence="Generated continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
        workstreams={
            "LambdaRankIC": AutoWorkstreamDecision(
                status=WorkstreamStatus.ACTIVE,
                canonical_surface=generated_branch,
                rule="recent-activity-active",
                evidence="Generated continuation.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 14),
            )
        },
    )

    effective = overlay_auto_decisions(
        DecisionRegistry(workstreams={"LambdaRankIC": explicit_workstream}),
        auto,
    )

    assert effective.workstreams["LambdaRankIC"].canonical_surface == stored_branch
    assert effective.surfaces[generated_branch].disposition == SurfaceDisposition.CANONICAL
    assert auto.surfaces[generated_branch].disposition == SurfaceDisposition.CANONICAL
    assert auto.workstreams["LambdaRankIC"].canonical_surface == generated_branch
    assert explicit_workstream.canonical_surface == stored_branch


def test_overlay_auto_decisions_marks_generated_entries_reviewed() -> None:
    auto = AutoDecisionSet(
        surfaces={
            "codex/lambdarank": AutoDisposition(
                workstreams=("LambdaRankIC",),
                disposition=SurfaceDisposition.CANONICAL,
                rule="unique-live-surface",
                evidence="Only live surface.",
                confidence=Confidence.HIGH,
                alternatives=(),
                last_reviewed=date(2026, 7, 10),
            )
        }
    )

    effective = overlay_auto_decisions(DecisionRegistry(), auto)

    assert effective.is_reviewed("LambdaRankIC", "codex/lambdarank")
    assert effective.surfaces["codex/lambdarank"].provenance == "auto"


def test_load_decision_registry_parses_v2_with_aliases(tmp_path) -> None:
    _write_registry(
        tmp_path,
        {
            "format_version": 2,
            "workstream_aliases": {
                "lambdarank": "LambdaRankIC",
                "top10": "LambdaRankIC",
                "portfolio-ic": "Portfolio-IC",
            },
            "workstreams": {},
            "surfaces": {},
        },
    )

    registry = load_decision_registry(tmp_path, known_workstreams={"LambdaRankIC"})

    assert registry.aliases == {
        "lambdarank": "LambdaRankIC",
        "top10": "LambdaRankIC",
        "portfolio-ic": "Portfolio-IC",
    }


def test_load_decision_registry_parses_v2_without_aliases(tmp_path) -> None:
    _write_registry(tmp_path, {"format_version": 2, "workstreams": {}, "surfaces": {}})

    registry = load_decision_registry(tmp_path, known_workstreams={"LambdaRankIC"})

    assert registry.aliases == {}


def test_load_decision_registry_still_parses_v1(tmp_path) -> None:
    _write_registry(tmp_path, {"format_version": 1, "workstreams": {}, "surfaces": {}})

    registry = load_decision_registry(tmp_path, known_workstreams={"LambdaRankIC"})

    assert registry.aliases == {}


def test_read_registry_aliases_returns_map_for_valid_file(tmp_path) -> None:
    _write_registry(
        tmp_path,
        {
            "format_version": 2,
            "workstream_aliases": {"lambdarank": "LambdaRankIC", "vol": "Issue #8"},
            "workstreams": {},
            "surfaces": {},
        },
    )

    assert read_registry_aliases(tmp_path) == {"lambdarank": "LambdaRankIC", "vol": "Issue #8"}


def test_read_registry_aliases_returns_empty_when_absent_or_broken(tmp_path) -> None:
    _write_registry(tmp_path, {"format_version": 2, "workstreams": {}, "surfaces": {}})
    assert read_registry_aliases(tmp_path) == {}

    _write_registry(
        tmp_path,
        {
            "format_version": 2,
            "workstream_aliases": ["lambdarank"],
            "workstreams": {},
            "surfaces": {},
        },
    )
    assert read_registry_aliases(tmp_path) == {}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"format_version": 3, "workstreams": {}, "surfaces": {}},
            "Unsupported cockpit decision registry format_version: 3",
        ),
        (
            {
                "format_version": 2,
                "workstreams": {},
                "surfaces": {},
                "workstream_aliases": ["lambdarank"],
            },
            "workstream_aliases must be a JSON object",
        ),
        (
            {
                "format_version": 2,
                "workstreams": {},
                "surfaces": {},
                "workstream_aliases": {"lambdarank": ""},
            },
            "workstream_aliases.lambdarank must be a non-empty string",
        ),
        *(
            (
                {
                    "format_version": 2,
                    "workstreams": {
                        "LambdaRankIC": {
                            "status": "ready-for-agent",
                            "canonical_surface": invalid_canonical,
                            "reason": "Explicit route.",
                            "next_action": "Continue.",
                            "last_reviewed": "2026-07-14",
                        }
                    },
                    "surfaces": {},
                },
                "workstreams.LambdaRankIC.canonical_surface must be a non-empty string",
            )
            for invalid_canonical in ("", None)
        ),
        (
            {
                "format_version": 2,
                "workstreams": {},
                "surfaces": {},
                "workstream_aliases": {"lambdarank": 5},
            },
            "workstream_aliases.lambdarank must be a non-empty string",
        ),
        (
            {
                "format_version": 1,
                "workstreams": {
                    "Unknown stream": {
                        "status": "active",
                        "canonical_surface": "origin/main",
                        "reason": "Typo should not be ignored.",
                        "next_action": "Fix the registry.",
                        "last_reviewed": "2026-07-09",
                    }
                },
                "surfaces": {},
            },
            "Unknown workstream in cockpit decision registry: Unknown stream",
        ),
        (
            {
                "format_version": 1,
                "workstreams": {},
                "surfaces": {
                    "origin/codex/not-normalized": {
                        "workstreams": ["LambdaRankIC"],
                        "disposition": "archive",
                        "reason": "Remote prefix is ambiguous.",
                        "next_action": "Normalize the key.",
                        "last_reviewed": "2026-07-09",
                    }
                },
            },
            "Surface keys must use normalized branch names",
        ),
    ],
)
def test_load_decision_registry_rejects_invalid_contract(tmp_path, payload, message) -> None:
    _write_registry(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_decision_registry(tmp_path, known_workstreams={"LambdaRankIC"})


def test_load_decision_registry_returns_empty_when_required_file_is_missing(tmp_path) -> None:
    registry = load_decision_registry(tmp_path, known_workstreams={"LambdaRankIC"})

    assert registry.workstreams == {}
    assert registry.surfaces == {}


def test_read_registry_workstream_names_returns_keys_for_valid_file(tmp_path) -> None:
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "workstreams": {
                "LambdaRankIC": {
                    "status": "active",
                    "canonical_surface": "origin/main",
                    "reason": "Reviewed continuation.",
                    "next_action": "Continue.",
                    "last_reviewed": "2026-07-09",
                },
                "Harness rollout": {
                    "status": "active",
                    "canonical_surface": "origin/main",
                    "reason": "New workstream.",
                    "next_action": "Continue.",
                    "last_reviewed": "2026-07-09",
                },
            },
            "surfaces": {},
        },
    )

    assert read_registry_workstream_names(tmp_path) == {"LambdaRankIC", "Harness rollout"}


def test_read_registry_workstream_names_returns_empty_for_missing_file(tmp_path) -> None:
    assert read_registry_workstream_names(tmp_path) == set()


def test_read_registry_workstream_names_returns_empty_for_invalid_json(tmp_path) -> None:
    path = tmp_path / DECISION_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    assert read_registry_workstream_names(tmp_path) == set()


def test_read_registry_workstream_names_returns_empty_when_workstreams_absent(tmp_path) -> None:
    _write_registry(tmp_path, {"format_version": 1, "surfaces": {}})

    assert read_registry_workstream_names(tmp_path) == set()


def test_read_registry_workstream_names_returns_empty_when_workstreams_not_object(tmp_path) -> None:
    _write_registry(
        tmp_path,
        {"format_version": 1, "workstreams": ["LambdaRankIC"], "surfaces": {}},
    )

    assert read_registry_workstream_names(tmp_path) == set()


def _write_registry(repo, payload: dict[str, object]) -> None:
    path = repo / DECISION_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
