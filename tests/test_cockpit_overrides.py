import json
from datetime import date

import pytest

from cockpit.models import SurfaceDisposition, WorkstreamStatus
from cockpit.overrides import (
    ClearSurfaceOverrideCommand,
    ClearWorkstreamOverrideCommand,
    SurfaceOverrideCommand,
    WorkstreamOverrideCommand,
    apply_override_command,
    parse_override_command,
)


def test_parse_workstream_status_override() -> None:
    command = parse_override_command(
        '/cockpit override workstream "LambdaRankIC" status parked '
        'reason "Pause until data contract review completes."',
        known_workstreams={"LambdaRankIC"},
        known_branches=set(),
    )

    assert command == WorkstreamOverrideCommand(
        workstream="LambdaRankIC",
        status=WorkstreamStatus.PARKED,
        reason="Pause until data contract review completes.",
    )


def test_parse_surface_disposition_override() -> None:
    command = parse_override_command(
        '/cockpit override surface "codex/example-branch" disposition archive '
        'workstream "LambdaRankIC" reason "Superseded by PR #90."',
        known_workstreams={"LambdaRankIC"},
        known_branches={"codex/example-branch"},
    )

    assert command == SurfaceOverrideCommand(
        branch="codex/example-branch",
        disposition=SurfaceDisposition.ARCHIVE,
        workstream="LambdaRankIC",
        reason="Superseded by PR #90.",
    )


def test_parse_clear_override_commands() -> None:
    workstream = parse_override_command(
        '/cockpit clear-override workstream "LambdaRankIC"',
        known_workstreams={"LambdaRankIC"},
        known_branches={"codex/example-branch"},
    )
    surface = parse_override_command(
        '/cockpit clear-override surface "codex/example-branch"',
        known_workstreams={"LambdaRankIC"},
        known_branches={"codex/example-branch"},
    )

    assert workstream == ClearWorkstreamOverrideCommand(workstream="LambdaRankIC")
    assert surface == ClearSurfaceOverrideCommand(branch="codex/example-branch")


def test_parse_rejects_blank_reason() -> None:
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        parse_override_command(
            '/cockpit override workstream "LambdaRankIC" status parked reason "   "',
            known_workstreams={"LambdaRankIC"},
            known_branches=set(),
        )
    with pytest.raises(ValueError, match="Malformed cockpit override command"):
        parse_override_command(
            '/cockpit override workstream "LambdaRankIC" status parked reason ""',
            known_workstreams={"LambdaRankIC"},
            known_branches=set(),
        )


@pytest.mark.parametrize(
    "text",
    [
        '/cockpit override workstream "LambdaRankIC" status parked reason " Pause."',
        '/cockpit override workstream "LambdaRankIC" status parked reason "Pause. "',
        '/cockpit override surface "codex/example-branch" disposition archive '
        'workstream "LambdaRankIC" reason " Superseded."',
        '/cockpit override surface "codex/example-branch" disposition archive '
        'workstream "LambdaRankIC" reason "Superseded. "',
    ],
)
def test_parse_rejects_reason_with_outer_whitespace(text: str) -> None:
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        parse_override_command(
            text,
            known_workstreams={"LambdaRankIC"},
            known_branches={"codex/example-branch"},
        )


def test_apply_workstream_override_mutates_only_targeted_entry(tmp_path) -> None:
    registry_path = tmp_path / "workstream-decisions.json"
    registry_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "workstream_aliases": {"lambda": "LambdaRankIC"},
                "workstreams": {
                    "LambdaRankIC": {
                        "status": "active",
                        "canonical_surface": "codex/lambda-old",
                        "reason": "Old reason.",
                        "next_action": "Old action.",
                        "last_reviewed": "2026-07-12",
                    },
                    "Portfolio-IC": {
                        "status": "parked",
                        "canonical_surface": "codex/portfolio",
                        "reason": "Leave untouched.",
                        "next_action": "Leave untouched.",
                        "last_reviewed": "2026-07-11",
                    },
                },
                "surfaces": {},
            }
        ),
        encoding="utf-8",
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(registry))
    command = parse_override_command(
        '/cockpit override workstream "LambdaRankIC" status parked '
        'reason "Pause until data contract review completes."',
        known_workstreams={"LambdaRankIC", "Portfolio-IC"},
        known_branches={"codex/lambda-current", "codex/portfolio"},
    )

    result = apply_override_command(
        registry,
        command,
        command_id="comment-42",
        applied_on=date(2026, 7, 13),
        canonical_surfaces={"LambdaRankIC": "codex/lambda-current"},
    )

    assert result.applied is True
    assert result.processed_command_ids == ("comment-42",)
    assert result.registry["workstreams"]["LambdaRankIC"] == {
        "status": "parked",
        "canonical_surface": "codex/lambda-old",
        "reason": "Pause until data contract review completes.",
        "next_action": "Pause until data contract review completes.",
        "last_reviewed": "2026-07-13",
    }
    assert result.registry["workstreams"]["Portfolio-IC"] == original["workstreams"]["Portfolio-IC"]
    assert result.registry["workstream_aliases"] == original["workstream_aliases"]
    assert registry == original
    json.dumps(result.registry, sort_keys=True)


def test_apply_override_persists_accepted_reason_text_exactly() -> None:
    reason = "Pause  until data contract review completes."
    command = parse_override_command(
        f'/cockpit override workstream "LambdaRankIC" status parked reason "{reason}"',
        known_workstreams={"LambdaRankIC"},
        known_branches=set(),
    )

    result = apply_override_command(
        {"format_version": 2, "workstreams": {}, "surfaces": {}},
        command,
        command_id="comment-exact-reason",
        applied_on=date(2026, 7, 13),
        canonical_surfaces={"LambdaRankIC": "codex/lambda-current"},
    )

    assert command.reason == reason
    assert result.registry["workstreams"]["LambdaRankIC"]["reason"] == reason
    assert result.registry["workstreams"]["LambdaRankIC"]["next_action"] == reason


def test_apply_surface_override_mutates_only_targeted_entry() -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {"lambda": "LambdaRankIC"},
        "workstreams": {
            "LambdaRankIC": {
                "status": "active",
                "canonical_surface": "codex/lambda-current",
                "reason": "Existing workstream override.",
                "next_action": "Existing action.",
                "last_reviewed": "2026-07-12",
            }
        },
        "surfaces": {
            "codex/other": {
                "workstreams": ["Portfolio-IC"],
                "disposition": "parked",
                "reason": "Leave untouched.",
                "next_action": "Leave untouched.",
                "last_reviewed": "2026-07-11",
            }
        },
    }
    original = json.loads(json.dumps(registry))
    command = parse_override_command(
        '/cockpit override surface "codex/example-branch" disposition archive '
        'workstream "LambdaRankIC" reason "Superseded by PR #90."',
        known_workstreams={"LambdaRankIC", "Portfolio-IC"},
        known_branches={"codex/example-branch", "codex/other"},
    )

    result = apply_override_command(
        registry,
        command,
        command_id="comment-10",
        applied_on=date(2026, 7, 13),
        canonical_surfaces={},
        processed_command_ids={"comment-9"},
    )

    assert result.applied is True
    assert result.processed_command_ids == ("comment-10", "comment-9")
    assert result.registry["surfaces"]["codex/example-branch"] == {
        "workstreams": ["LambdaRankIC"],
        "disposition": "archive",
        "reason": "Superseded by PR #90.",
        "next_action": "Superseded by PR #90.",
        "last_reviewed": "2026-07-13",
    }
    assert result.registry["surfaces"]["codex/other"] == original["surfaces"]["codex/other"]
    assert result.registry["workstreams"] == original["workstreams"]
    assert registry == original


def test_apply_canonical_surface_override_rejects_second_explicit_canonical() -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {},
        "workstreams": {},
        "surfaces": {
            "codex/lambda-current": {
                "workstreams": ["LambdaRankIC"],
                "disposition": "canonical",
                "reason": "Reviewed continuation.",
                "next_action": "Continue here.",
                "last_reviewed": "2026-07-13",
            }
        },
    }
    original = json.loads(json.dumps(registry))
    command = parse_override_command(
        '/cockpit override surface "codex/lambda-new" disposition canonical '
        'workstream "LambdaRankIC" reason "Use the new continuation."',
        known_workstreams={"LambdaRankIC"},
        known_branches={"codex/lambda-current", "codex/lambda-new"},
    )

    with pytest.raises(
        ValueError,
        match=(
            "Workstream LambdaRankIC already has explicit canonical surface codex/lambda-current"
        ),
    ):
        apply_override_command(
            registry,
            command,
            command_id="comment-second-canonical",
            applied_on=date(2026, 7, 14),
            canonical_surfaces={"LambdaRankIC": "codex/lambda-current"},
        )

    assert registry == original


def test_apply_canonical_surface_override_can_replace_its_target_entry() -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {},
        "workstreams": {},
        "surfaces": {
            "codex/lambda-current": {
                "workstreams": ["LambdaRankIC"],
                "disposition": "canonical",
                "reason": "Old reason.",
                "next_action": "Old action.",
                "last_reviewed": "2026-07-13",
            },
            "codex/portfolio": {
                "workstreams": ["Portfolio-IC"],
                "disposition": "parked",
                "reason": "Leave untouched.",
                "next_action": "Wait.",
                "last_reviewed": "2026-07-13",
            },
        },
    }
    original = json.loads(json.dumps(registry))
    command = SurfaceOverrideCommand(
        branch="codex/lambda-current",
        disposition=SurfaceDisposition.CANONICAL,
        workstream="LambdaRankIC",
        reason="Updated reviewed continuation.",
    )

    result = apply_override_command(
        registry,
        command,
        command_id="comment-replace-canonical",
        applied_on=date(2026, 7, 14),
        canonical_surfaces={"LambdaRankIC": "codex/lambda-current"},
    )

    assert result.registry["surfaces"]["codex/lambda-current"] == {
        "workstreams": ["LambdaRankIC"],
        "disposition": "canonical",
        "reason": "Updated reviewed continuation.",
        "next_action": "Updated reviewed continuation.",
        "last_reviewed": "2026-07-14",
    }
    assert result.registry["surfaces"]["codex/portfolio"] == original["surfaces"]["codex/portfolio"]
    assert registry == original


def test_clear_then_apply_canonical_surface_replacement_changes_one_entry_at_a_time() -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {},
        "workstreams": {},
        "surfaces": {
            "codex/lambda-old": {
                "workstreams": ["LambdaRankIC"],
                "disposition": "canonical",
                "reason": "Old continuation.",
                "next_action": "Continue here.",
                "last_reviewed": "2026-07-13",
            },
            "codex/portfolio": {
                "workstreams": ["Portfolio-IC"],
                "disposition": "parked",
                "reason": "Leave untouched.",
                "next_action": "Wait.",
                "last_reviewed": "2026-07-13",
            },
        },
    }
    original = json.loads(json.dumps(registry))
    clear = ClearSurfaceOverrideCommand(branch="codex/lambda-old")

    cleared = apply_override_command(
        registry,
        clear,
        command_id="comment-clear-canonical",
        applied_on=date(2026, 7, 14),
        canonical_surfaces={},
    )

    assert set(cleared.registry["surfaces"]) == {"codex/portfolio"}
    assert (
        cleared.registry["surfaces"]["codex/portfolio"] == original["surfaces"]["codex/portfolio"]
    )

    replacement = SurfaceOverrideCommand(
        branch="codex/lambda-new",
        disposition=SurfaceDisposition.CANONICAL,
        workstream="LambdaRankIC",
        reason="New reviewed continuation.",
    )
    replaced = apply_override_command(
        cleared.registry,
        replacement,
        command_id="comment-add-canonical",
        applied_on=date(2026, 7, 14),
        canonical_surfaces={},
        processed_command_ids=cleared.processed_command_ids,
    )

    assert set(replaced.registry["surfaces"]) == {"codex/lambda-new", "codex/portfolio"}
    assert replaced.registry["surfaces"]["codex/lambda-new"]["disposition"] == "canonical"
    assert (
        replaced.registry["surfaces"]["codex/portfolio"] == original["surfaces"]["codex/portfolio"]
    )
    assert registry == original


def test_apply_clear_commands_remove_only_the_targeted_override() -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {"lambda": "LambdaRankIC"},
        "workstreams": {
            "LambdaRankIC": {"sentinel": "remove"},
            "Portfolio-IC": {"sentinel": "keep"},
        },
        "surfaces": {
            "codex/old-registry-only": {"sentinel": "remove"},
            "codex/other": {"sentinel": "keep"},
        },
    }
    clear_workstream = parse_override_command(
        '/cockpit clear-override workstream "LambdaRankIC"',
        known_workstreams={"LambdaRankIC", "Portfolio-IC"},
        known_branches={"codex/old-registry-only", "codex/other"},
    )
    clear_surface = parse_override_command(
        '/cockpit clear-override surface "codex/old-registry-only"',
        known_workstreams={"LambdaRankIC", "Portfolio-IC"},
        known_branches={"codex/old-registry-only", "codex/other"},
    )

    workstream_result = apply_override_command(
        registry,
        clear_workstream,
        command_id="comment-20",
        applied_on=date(2026, 7, 13),
        canonical_surfaces={},
    )
    surface_result = apply_override_command(
        registry,
        clear_surface,
        command_id="comment-21",
        applied_on=date(2026, 7, 13),
        canonical_surfaces={},
    )

    assert workstream_result.registry["workstreams"] == {"Portfolio-IC": {"sentinel": "keep"}}
    assert workstream_result.registry["surfaces"] == registry["surfaces"]
    assert surface_result.registry["surfaces"] == {"codex/other": {"sentinel": "keep"}}
    assert surface_result.registry["workstreams"] == registry["workstreams"]
    assert registry["workstreams"]["LambdaRankIC"] == {"sentinel": "remove"}
    assert registry["surfaces"]["codex/old-registry-only"] == {"sentinel": "remove"}


@pytest.mark.parametrize(
    ("command_text", "expected_message"),
    [
        (
            '/cockpit clear-override workstream "LambdaRankIC"',
            "No explicit workstream override exists",
        ),
        (
            '/cockpit clear-override surface "codex/example-branch"',
            "No explicit surface override exists",
        ),
    ],
)
def test_clear_override_rejects_target_without_explicit_override(
    command_text: str,
    expected_message: str,
) -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {},
        "workstreams": {},
        "surfaces": {},
    }
    command = parse_override_command(
        command_text,
        known_workstreams={"LambdaRankIC"},
        known_branches={"codex/example-branch"},
    )

    with pytest.raises(ValueError, match=expected_message):
        apply_override_command(
            registry,
            command,
            command_id="comment-noop-clear",
            applied_on=date(2026, 7, 13),
            canonical_surfaces={},
        )


def test_duplicate_command_id_is_a_deterministic_no_op() -> None:
    registry = {
        "format_version": 2,
        "workstream_aliases": {},
        "workstreams": {"LambdaRankIC": {"sentinel": "unchanged"}},
        "surfaces": {},
    }
    command = parse_override_command(
        '/cockpit override workstream "LambdaRankIC" status parked reason "Pause."',
        known_workstreams={"LambdaRankIC"},
        known_branches=set(),
    )

    result = apply_override_command(
        registry,
        command,
        command_id="comment-42",
        applied_on=date(2026, 7, 13),
        canonical_surfaces={},
        processed_command_ids=["z-comment", "comment-42", "a-comment"],
    )

    assert result.applied is False
    assert result.registry == registry
    assert result.registry is not registry
    assert result.processed_command_ids == ("a-comment", "comment-42", "z-comment")


def test_apply_rejects_blank_command_id() -> None:
    command = ClearWorkstreamOverrideCommand(workstream="LambdaRankIC")

    with pytest.raises(ValueError, match="command_id must be a non-empty string"):
        apply_override_command(
            {"format_version": 2, "workstreams": {}, "surfaces": {}},
            command,
            command_id="   ",
            applied_on=date(2026, 7, 13),
            canonical_surfaces={},
        )


def test_parse_rejects_remote_prefixed_branch_even_when_listed_as_known() -> None:
    with pytest.raises(ValueError, match="normalized branch name"):
        parse_override_command(
            '/cockpit clear-override surface "origin/codex/example-branch"',
            known_workstreams={"LambdaRankIC"},
            known_branches={"origin/codex/example-branch"},
        )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '/cockpit override workstream "Unknown" status parked reason "Pause."',
            "Unknown workstream: Unknown",
        ),
        (
            '/cockpit override workstream "LambdaRankIC" status invented reason "Pause."',
            "Unknown workstream status: invented",
        ),
        (
            '/cockpit override workstream "LambdaRankIC" status needs-user-decision '
            'reason "Pause."',
            "cannot be needs-user-decision",
        ),
        (
            '/cockpit override surface "codex/unknown" disposition archive '
            'workstream "LambdaRankIC" reason "Superseded."',
            "Unknown branch: codex/unknown",
        ),
        (
            '/cockpit override surface "codex/example-branch" disposition invented '
            'workstream "LambdaRankIC" reason "Superseded."',
            "Unknown surface disposition: invented",
        ),
        (
            '/cockpit override surface "codex/example-branch" disposition archive '
            'workstream "Unknown" reason "Superseded."',
            "Unknown workstream: Unknown",
        ),
        (
            '/cockpit override workstream LambdaRankIC status parked reason "Pause."',
            "Malformed cockpit override command",
        ),
        (
            '/cockpit override workstream "LambdaRankIC" status parked',
            "Malformed cockpit override command",
        ),
    ],
)
def test_parse_rejects_invalid_commands(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_override_command(
            text,
            known_workstreams={"LambdaRankIC"},
            known_branches={"codex/example-branch"},
        )


def test_apply_workstream_override_requires_caller_supplied_canonical_surface() -> None:
    command = WorkstreamOverrideCommand(
        workstream="LambdaRankIC",
        status=WorkstreamStatus.ACTIVE,
        reason="Resume.",
    )

    with pytest.raises(ValueError, match="Missing canonical surface for workstream"):
        apply_override_command(
            {"format_version": 2, "workstreams": {}, "surfaces": {}},
            command,
            command_id="comment-50",
            applied_on=date(2026, 7, 13),
            canonical_surfaces={},
        )


def test_apply_workstream_override_rejects_blank_canonical_surface() -> None:
    command = WorkstreamOverrideCommand(
        workstream="LambdaRankIC",
        status=WorkstreamStatus.ACTIVE,
        reason="Resume.",
    )

    with pytest.raises(ValueError, match="Missing canonical surface for workstream"):
        apply_override_command(
            {"format_version": 2, "workstreams": {}, "surfaces": {}},
            command,
            command_id="comment-51",
            applied_on=date(2026, 7, 13),
            canonical_surfaces={"LambdaRankIC": "   "},
        )
