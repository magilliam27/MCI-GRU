from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cockpit.github import (
    CockpitPullRequestTarget,
    CommentAuthorization,
    PullRequestComment,
    authorize_owner_comment,
    collect_cockpit_pr_target,
    collect_pull_request_comments,
    collect_pull_request_paths,
    post_pull_request_response,
)
from cockpit.overrides import parse_override_command

HEAD_OID = "a" * 40
BASE_OID = "b" * 40


def _comment(
    *,
    author_login: str = "repo-owner",
    author_association: str = "owner",
) -> PullRequestComment:
    return PullRequestComment(
        comment_id="101",
        url="https://github.example/pull/99#issuecomment-101",
        author_login=author_login,
        author_association=author_association,
        body='/cockpit override workstream "Alpha" status parked',
        created_at=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
    )


def test_collect_pull_request_comments_returns_stable_paginated_evidence() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        return json.dumps(
            [
                [
                    {
                        "id": 202,
                        "html_url": "https://github.example/pull/99#issuecomment-202",
                        "user": {"login": "Maintainer-One"},
                        "author_association": "MEMBER",
                        "body": '  /cockpit clear-override workstream "Alpha"  ',
                        "created_at": "2026-07-13T12:30:00Z",
                    }
                ],
                [
                    {
                        "id": 101,
                        "html_url": "https://github.example/pull/99#issuecomment-101",
                        "user": {"login": "Repo-Owner"},
                        "author_association": "OWNER",
                        "body": '/cockpit override workstream "Alpha" status parked',
                        "created_at": "2026-07-13T12:00:00+00:00",
                    }
                ],
            ]
        )

    comments = collect_pull_request_comments(
        pr_number=99,
        repo="repo-owner/project",
        run_command=fake_run,
    )

    assert commands == [
        [
            "gh",
            "api",
            "repos/repo-owner/project/issues/99/comments",
            "--paginate",
            "--slurp",
        ]
    ]
    assert comments == (
        PullRequestComment(
            comment_id="101",
            url="https://github.example/pull/99#issuecomment-101",
            author_login="Repo-Owner",
            author_association="owner",
            body='/cockpit override workstream "Alpha" status parked',
            created_at=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
        ),
        PullRequestComment(
            comment_id="202",
            url="https://github.example/pull/99#issuecomment-202",
            author_login="Maintainer-One",
            author_association="member",
            body='  /cockpit clear-override workstream "Alpha"  ',
            created_at=datetime(2026, 7, 13, 12, 30, tzinfo=timezone.utc),
        ),
    )


def test_collected_command_whitespace_is_rejected_by_selection_or_full_match() -> None:
    raw_bodies = (
        ' /cockpit clear-override workstream "Alpha"',
        '/cockpit clear-override workstream "Alpha" ',
    )
    comments = collect_pull_request_comments(
        pr_number=99,
        repo="repo-owner/project",
        run_command=lambda args: json.dumps(
            [
                [
                    {
                        "id": comment_id,
                        "html_url": (f"https://github.example/pull/99#issuecomment-{comment_id}"),
                        "user": {"login": "Repo-Owner"},
                        "author_association": "OWNER",
                        "body": body,
                        "created_at": "2026-07-13T12:00:00Z",
                    }
                    for comment_id, body in enumerate(raw_bodies, start=101)
                ]
            ]
        ),
    )

    assert comments is not None
    selected = tuple(comment for comment in comments if comment.body.startswith("/cockpit "))
    assert tuple(comment.comment_id for comment in selected) == ("102",)
    with pytest.raises(ValueError, match="Malformed cockpit override command"):
        parse_override_command(
            selected[0].body,
            known_workstreams={"Alpha"},
            known_branches=set(),
        )


def test_collect_pull_request_paths_returns_sorted_paginated_filenames() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        return json.dumps(
            [
                [{"filename": "docs/agents/workstreams.md"}],
                [
                    {
                        "filename": "docs/agents/cockpit/2026-07-13.md",
                        "previous_filename": "scripts/untrusted-refresh.py",
                    },
                    {"filename": "docs/agents/workstreams.md"},
                ],
            ]
        )

    paths = collect_pull_request_paths(
        pr_number=99,
        repo="repo-owner/project",
        run_command=fake_run,
    )

    assert commands == [
        [
            "gh",
            "api",
            "repos/repo-owner/project/pulls/99/files",
            "--paginate",
            "--slurp",
        ]
    ]
    assert paths == (
        "docs/agents/cockpit/2026-07-13.md",
        "docs/agents/workstreams.md",
        "scripts/untrusted-refresh.py",
    )


def test_authorize_owner_comment_uses_trusted_owner_association_without_admin_api() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        raise AssertionError(args)

    authorization = authorize_owner_comment(
        _comment(),
        repo="repo-owner/project",
        run_command=fake_run,
    )

    assert commands == []
    assert authorization == CommentAuthorization(
        comment_id="101",
        comment_url="https://github.example/pull/99#issuecomment-101",
        author_login="repo-owner",
        author_association="owner",
        repository_permission=None,
        authorized=True,
        reason="verified repository owner comment association",
    )


@pytest.mark.parametrize("association", ["member", "owner"])
def test_authorize_owner_comment_rejects_non_owner_collaborators(association: str) -> None:
    authorization = authorize_owner_comment(
        _comment(author_login="trusted-collaborator", author_association=association),
        repo="repo-owner/project",
        run_command=lambda args: (_ for _ in ()).throw(AssertionError(args)),
    )

    assert authorization.authorized is False
    assert authorization.repository_permission is None
    assert authorization.reason == "only the repository owner may apply cockpit overrides"


def test_authorize_owner_comment_rejects_owner_login_without_owner_association() -> None:
    authorization = authorize_owner_comment(
        _comment(author_association="member"),
        repo="repo-owner/project",
        run_command=lambda args: (_ for _ in ()).throw(AssertionError(args)),
    )

    assert authorization.authorized is False
    assert authorization.repository_permission is None
    assert authorization.reason == "only the repository owner may apply cockpit overrides"


def test_collect_cockpit_pr_target_returns_same_branch_metadata() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        return json.dumps(
            {
                "title": "Cockpit refresh: 2026-07-13",
                "headRefName": "codex/cockpit-refresh-20260713",
                "headRefOid": HEAD_OID,
                "baseRefName": "main",
                "baseRefOid": BASE_OID,
                "headRepositoryOwner": {"login": "Repo-Owner"},
                "isCrossRepository": False,
                "url": "https://github.example/repo-owner/project/pull/99",
                "state": "OPEN",
            }
        )

    target = collect_cockpit_pr_target(
        pr_number=99,
        repo="repo-owner/project",
        run_command=fake_run,
    )

    assert commands == [
        [
            "gh",
            "pr",
            "view",
            "99",
            "--repo",
            "repo-owner/project",
            "--json",
            "title,headRefName,headRefOid,baseRefName,baseRefOid,headRepositoryOwner,isCrossRepository,url,state",
        ]
    ]
    assert target == CockpitPullRequestTarget(
        pr_number=99,
        head_ref="codex/cockpit-refresh-20260713",
        head_oid=HEAD_OID,
        base_oid=BASE_OID,
        head_repository_owner="Repo-Owner",
        url="https://github.example/repo-owner/project/pull/99",
        state="open",
        is_cross_repository=False,
        title="Cockpit refresh: 2026-07-13",
        base_ref="main",
    )


@pytest.mark.parametrize(
    ("title", "base_ref"),
    [
        ("Ordinary feature", "main"),
        ("Cockpit refresh: 2026-07-12", "main"),
        (" Cockpit refresh: 2026-07-13", "main"),
        ("Cockpit refresh: 2026-07-13 ", "main"),
        ("Cockpit refresh: 2026-07-13", "release"),
        ("Cockpit refresh: 2026-07-13", " main"),
        ("Cockpit refresh: 2026-07-13", "main "),
    ],
)
def test_collect_cockpit_pr_target_rejects_wrong_title_date_or_base(
    title: str,
    base_ref: str,
) -> None:
    output = json.dumps(
        {
            "title": title,
            "headRefName": "codex/cockpit-refresh-20260713",
            "headRefOid": HEAD_OID,
            "baseRefName": base_ref,
            "baseRefOid": BASE_OID,
            "headRepositoryOwner": {"login": "repo-owner"},
            "isCrossRepository": False,
            "url": "https://github.example/repo-owner/project/pull/99",
            "state": "OPEN",
        }
    )

    assert (
        collect_cockpit_pr_target(
            pr_number=99,
            repo="repo-owner/project",
            run_command=lambda args: output,
        )
        is None
    )


@pytest.mark.parametrize(
    ("head_ref", "head_owner", "is_cross_repository"),
    [
        ("codex/cockpit-refresh-20260713", "fork-owner", False),
        ("codex/ordinary-feature-20260713", "repo-owner", False),
        (" codex/cockpit-refresh-20260713", "repo-owner", False),
        ("codex/cockpit-refresh-20260713 ", "repo-owner", False),
        ("codex/cockpit-refresh-20260713", "repo-owner", True),
    ],
)
def test_collect_cockpit_pr_target_fails_closed_for_untrusted_head(
    head_ref: str,
    head_owner: str,
    is_cross_repository: bool,
) -> None:
    output = json.dumps(
        {
            "title": "Cockpit refresh: 2026-07-13",
            "headRefName": head_ref,
            "headRefOid": HEAD_OID,
            "baseRefName": "main",
            "baseRefOid": BASE_OID,
            "headRepositoryOwner": {"login": head_owner},
            "isCrossRepository": is_cross_repository,
            "url": "https://github.example/repo-owner/project/pull/99",
            "state": "OPEN",
        }
    )

    target = collect_cockpit_pr_target(
        pr_number=99,
        repo="repo-owner/project",
        run_command=lambda args: output,
    )

    assert target is None


def test_collect_cockpit_pr_target_fails_closed_when_pr_is_not_open() -> None:
    output = json.dumps(
        {
            "title": "Cockpit refresh: 2026-07-13",
            "headRefName": "codex/cockpit-refresh-20260713",
            "headRefOid": HEAD_OID,
            "baseRefName": "main",
            "baseRefOid": BASE_OID,
            "headRepositoryOwner": {"login": "repo-owner"},
            "isCrossRepository": False,
            "url": "https://github.example/repo-owner/project/pull/99",
            "state": "CLOSED",
        }
    )

    target = collect_cockpit_pr_target(
        pr_number=99,
        repo="repo-owner/project",
        run_command=lambda args: output,
    )

    assert target is None


def test_post_pull_request_response_uses_injected_boundary() -> None:
    commands: list[list[str]] = []
    target = CockpitPullRequestTarget(
        pr_number=99,
        head_ref="codex/cockpit-refresh-20260713",
        head_oid=HEAD_OID,
        base_oid=BASE_OID,
        head_repository_owner="repo-owner",
        url="https://github.example/repo-owner/project/pull/99",
        state="open",
        is_cross_repository=False,
    )
    response_body = (
        "Cockpit override rejected for comment 101: only the repository owner is authorized."
    )

    response_url = post_pull_request_response(
        target,
        body=response_body,
        repo="repo-owner/project",
        run_command=lambda args: (
            commands.append(args)
            or "https://github.example/repo-owner/project/pull/99#issuecomment-303\n"
        ),
    )

    assert commands == [
        [
            "gh",
            "pr",
            "comment",
            "99",
            "--repo",
            "repo-owner/project",
            "--body",
            response_body,
        ]
    ]
    assert response_url == "https://github.example/repo-owner/project/pull/99#issuecomment-303"
