from __future__ import annotations

import json
from datetime import date

import pytest

from cockpit.decisions import (
    DECISION_REGISTRY_PATH,
    SurfaceDisposition,
    load_decision_registry,
)
from cockpit.models import WorkstreamStatus


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"format_version": 2, "workstreams": {}, "surfaces": {}},
            "Unsupported cockpit decision registry format_version: 2",
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
                    "codex/unknown-stream": {
                        "workstreams": ["Unknown stream"],
                        "disposition": "archive",
                        "reason": "A surface cannot declare a new workstream.",
                        "next_action": "Declare the workstream first.",
                        "last_reviewed": "2026-07-09",
                    }
                },
            },
            "Unknown workstream in surfaces.codex/unknown-stream.workstreams: Unknown stream",
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


def _write_registry(repo, payload: dict[str, object]) -> None:
    path = repo / DECISION_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
