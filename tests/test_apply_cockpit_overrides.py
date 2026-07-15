from __future__ import annotations

import json
import subprocess
from datetime import date
from typing import TYPE_CHECKING

import pytest

from cockpit.runner import run_local_cockpit_refresh
from scripts.apply_cockpit_overrides import run_cockpit_override_curator

if TYPE_CHECKING:
    from pathlib import Path

REPO = "magilliam27/MCI-GRU"
COMMENT_ID = "9001"
COMMENT_BODY = (
    '/cockpit override workstream "LambdaRankIC" status parked reason "Pause pending data review."'
)
HEAD_OID = "a" * 40
BASE_OID = "b" * 40
PUSHED_OID = "c" * 40


class FakeRunner:
    def __init__(
        self,
        *,
        body: str = COMMENT_BODY,
        author_login: str = "magilliam27",
        author_association: str = "OWNER",
        dirty: bool = False,
        staged_paths: tuple[str, ...] | None = None,
        pr_paths: tuple[str, ...] | None = None,
        pr_file_items: tuple[dict[str, str], ...] | None = None,
        fetched_oid: str = HEAD_OID,
        fetched_base_oid: str = BASE_OID,
        local_diff_output: str | None = None,
        current_branch: str = "main",
        local_target_exists: bool = False,
        local_head_oid: str = HEAD_OID,
        committed_head_oid: str = PUSHED_OID,
        fail_push: bool = False,
        fail_response_once: bool = False,
    ) -> None:
        self.commands: list[list[str]] = []
        self.body = body
        self.author_login = author_login
        self.author_association = author_association
        self.dirty = dirty
        default_paths = (
            "docs/agents/cockpit/workstream-decisions.json",
            "docs/agents/cockpit/override-receipts.json",
            "docs/agents/cockpit/auto-decisions.json",
            "docs/agents/workstreams.md",
            "docs/agents/cockpit/2026-07-13.md",
        )
        self.staged_paths = staged_paths or default_paths
        self.pr_paths = pr_paths or default_paths
        self.pr_file_items = pr_file_items
        self.fetched_oid = fetched_oid
        self.fetched_base_oid = fetched_base_oid
        self.current_branch = current_branch
        self.local_target_exists = local_target_exists
        self.local_head_oid = local_head_oid
        self.committed_head_oid = committed_head_oid
        self.fail_push = fail_push
        self.fail_response_once = fail_response_once
        self.target_head_oid = HEAD_OID
        self.last_fetch_ref = ""
        self.response_comments: list[dict[str, object]] = []
        self.commit_messages: dict[str, str] = {}
        self.local_diff_output = local_diff_output or "".join(
            f"M\0{path}\0" for path in default_paths
        )

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        if command[:3] == ["git", "fetch", "origin"]:
            self.last_fetch_ref = command[3]
            return ""
        if command == ["git", "status", "--porcelain=v1"]:
            return " M unrelated.txt\n" if self.dirty else ""
        if command == ["git", "diff", "--cached", "--name-only"]:
            return "\n".join(self.staged_paths) + "\n"
        if command == ["git", "rev-parse", "FETCH_HEAD"]:
            oid = self.fetched_base_oid if self.last_fetch_ref == "main" else self.fetched_oid
            return oid + "\n"
        if command == ["git", "branch", "--show-current"]:
            return self.current_branch + "\n"
        if command == ["git", "branch", "--list", "codex/cockpit-refresh-20260713"]:
            return "  codex/cockpit-refresh-20260713\n" if self.local_target_exists else ""
        if command == ["git", "rev-parse", "HEAD"]:
            return self.local_head_oid + "\n"
        if command[:2] == ["git", "commit"]:
            self.local_head_oid = self.committed_head_oid
            message_parts = [
                command[index + 1] for index, value in enumerate(command) if value == "-m"
            ]
            self.commit_messages[self.local_head_oid] = "\n\n".join(message_parts)
            return ""
        if command[:3] == ["git", "push", "origin"] and self.fail_push:
            raise RuntimeError("push failed")
        if command[:3] == ["git", "push", "origin"]:
            self.fetched_oid = self.local_head_oid
            self.target_head_oid = self.local_head_oid
            return ""
        if command == [
            "git",
            "switch",
            "--create",
            "codex/cockpit-refresh-20260713",
            "FETCH_HEAD",
        ]:
            self.current_branch = "codex/cockpit-refresh-20260713"
            self.local_target_exists = True
            self.local_head_oid = self.fetched_oid
            return ""
        if command == ["git", "switch", "codex/cockpit-refresh-20260713"]:
            self.current_branch = "codex/cockpit-refresh-20260713"
            return ""
        if command[:4] == ["git", "diff", "--name-status", "-z"]:
            return self.local_diff_output
        if command == ["git", "log", "--format=%H%x09%s", "HEAD", "--"]:
            return "".join(
                f"{oid}\t{message.splitlines()[0]}\n"
                for oid, message in self.commit_messages.items()
            )
        if command[:4] == ["git", "show", "--no-patch", "--format=%B"]:
            return self.commit_messages.get(command[4], "") + "\n"
        if command[:3] == ["gh", "pr", "view"]:
            return json.dumps(
                {
                    "title": "Cockpit refresh: 2026-07-13",
                    "headRefName": "codex/cockpit-refresh-20260713",
                    "headRefOid": self.target_head_oid,
                    "baseRefName": "main",
                    "baseRefOid": BASE_OID,
                    "headRepositoryOwner": {"login": "magilliam27"},
                    "isCrossRepository": False,
                    "url": "https://github.com/magilliam27/MCI-GRU/pull/88",
                    "state": "OPEN",
                }
            )
        if (
            command[:2] == ["gh", "api"]
            and command[2].endswith("/pulls/88/files")
            and command[-2:] == ["--paginate", "--slurp"]
        ):
            items = self.pr_file_items or tuple({"filename": path} for path in self.pr_paths)
            return json.dumps([items])
        if command[:2] == ["gh", "api"] and command[-2:] == ["--paginate", "--slurp"]:
            return json.dumps(
                [
                    [
                        {
                            "id": 9001,
                            "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-9001",
                            "body": self.body,
                            "created_at": "2026-07-13T12:00:00Z",
                            "author_association": self.author_association,
                            "user": {"login": self.author_login},
                        },
                        *self.response_comments,
                    ]
                ]
            )
        if command[:3] == ["gh", "pr", "comment"]:
            if self.fail_response_once:
                self.fail_response_once = False
                raise RuntimeError("response post failed")
            self.response_comments.append(
                {
                    "id": 9002 + len(self.response_comments),
                    "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-response",
                    "body": command[-1],
                    "created_at": "2026-07-13T12:01:00Z",
                    "author_association": "NONE",
                    "user": {"login": "github-actions[bot]"},
                }
            )
            return "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-9002\n"
        return ""


def _write_inputs(repo_root: Path) -> None:
    cockpit_dir = repo_root / "docs" / "agents" / "cockpit"
    cockpit_dir.mkdir(parents=True)
    registry = {
        "format_version": 2,
        "workstream_aliases": {"lambda": "LambdaRankIC"},
        "workstreams": {
            "LambdaRankIC": {
                "status": "active",
                "canonical_surface": "origin/main",
                "reason": "Existing reason.",
                "next_action": "Existing next action.",
                "last_reviewed": "2026-07-12",
            }
        },
        "surfaces": {},
    }
    auto = {
        "format_version": 2,
        "surfaces": {},
        "workstreams": {
            "LambdaRankIC": {
                "status": "active",
                "canonical_surface": "codex/generated-choice",
                "rule": "open-pr",
                "evidence": "PR #88 is open.",
                "confidence": "high",
                "alternatives": [],
                "last_reviewed": "2026-07-13",
            }
        },
    }
    (cockpit_dir / "workstream-decisions.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    (cockpit_dir / "auto-decisions.json").write_text(
        json.dumps(auto, indent=2) + "\n", encoding="utf-8"
    )


def _add_generated_canonical_surface(repo_root: Path) -> None:
    auto_path = repo_root / "docs" / "agents" / "cockpit" / "auto-decisions.json"
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    auto["surfaces"]["codex/generated-choice"] = {
        "workstreams": ["LambdaRankIC"],
        "disposition": "canonical",
        "rule": "unique-live-surface",
        "evidence": "Only live continuation.",
        "confidence": "high",
        "association_basis": "branch-term",
        "alternatives": [],
        "last_reviewed": "2026-07-13",
    }
    auto_path.write_text(json.dumps(auto, indent=2) + "\n", encoding="utf-8")


def _noop_refresh(repo_root: Path, run_date: date) -> None:
    del repo_root, run_date


def _record_processed_commit(runner: FakeRunner) -> None:
    subject = f"Apply cockpit override from comment {COMMENT_ID}"
    comment_url = f"https://github.com/{REPO}/pull/88#issuecomment-{COMMENT_ID}"
    runner.commit_messages[PUSHED_OID] = f"{subject}\n\n{comment_url}\n\n{COMMENT_BODY}"
    runner.target_head_oid = PUSHED_OID
    runner.fetched_oid = PUSHED_OID
    runner.local_head_oid = PUSHED_OID


def test_owner_comment_applies_exact_override_and_updates_same_pr(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner()
    refresh_calls: list[tuple[Path, date]] = []

    def trusted_refresh(repo_root: Path, run_date: date) -> None:
        refresh_calls.append((repo_root, run_date))

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=trusted_refresh,
    )

    assert result.applied_comment_ids == (COMMENT_ID,)
    assert result.rejected_comment_ids == ()
    assert refresh_calls == [(tmp_path, date(2026, 7, 13))]
    assert not any("scripts/refresh_cockpit.py" in command for command in runner.commands)
    registry = json.loads(
        (tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["workstream_aliases"] == {"lambda": "LambdaRankIC"}
    assert registry["workstreams"]["LambdaRankIC"] == {
        "status": "parked",
        "canonical_surface": "origin/main",
        "reason": "Pause pending data review.",
        "next_action": "Pause pending data review.",
        "last_reviewed": "2026-07-13",
    }
    receipts = json.loads(
        (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipts == {"format_version": 1, "processed_comment_ids": [COMMENT_ID]}
    assert [
        "git",
        "switch",
        "--create",
        "codex/cockpit-refresh-20260713",
        "FETCH_HEAD",
    ] in runner.commands
    assert not any(command[:3] == ["git", "switch", "--detach"] for command in runner.commands)
    assert not any(command[:3] == ["git", "switch", "-C"] for command in runner.commands)
    commit = next(command for command in runner.commands if command[:2] == ["git", "commit"])
    assert COMMENT_BODY in commit
    assert "pull/88#issuecomment-9001" in " ".join(commit)
    assert [
        "git",
        "push",
        "origin",
        "HEAD:codex/cockpit-refresh-20260713",
    ] in runner.commands
    response = next(
        command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    )
    assert "Applied cockpit override" in response[-1]
    assert COMMENT_BODY in response[-1]
    assert PUSHED_OID in response[-1]
    assert f"https://github.com/{REPO}/commit/{PUSHED_OID}" in response[-1]
    push_index = runner.commands.index(
        ["git", "push", "origin", "HEAD:codex/cockpit-refresh-20260713"]
    )
    response_index = runner.commands.index(response)
    assert push_index < response_index


def test_curator_rejects_noncanonical_override_of_sole_generated_canonical(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _add_generated_canonical_surface(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    del registry["workstreams"]["LambdaRankIC"]
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    original_registry = registry_path.read_bytes()
    runner = FakeRunner(
        body=(
            '/cockpit override surface "codex/generated-choice" disposition parked '
            'workstream "LambdaRankIC" reason "Pause the only continuation."'
        )
    )

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == ()
    assert result.rejected_comment_ids == (COMMENT_ID,)
    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    response = next(
        command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    )
    assert "No effective canonical surface" in response[-1]


def test_curator_allows_noncanonical_override_with_alternate_explicit_canonical(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _add_generated_canonical_surface(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["surfaces"]["codex/reviewed-choice"] = {
        "workstreams": ["LambdaRankIC"],
        "disposition": "canonical",
        "reason": "Reviewed continuation.",
        "next_action": "Continue here.",
        "last_reviewed": "2026-07-13",
    }
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    original = json.loads(json.dumps(registry))
    runner = FakeRunner(
        body=(
            '/cockpit override surface "codex/generated-choice" disposition stale '
            'workstream "LambdaRankIC" reason "Use the reviewed continuation."'
        )
    )

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == (COMMENT_ID,)
    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    assert updated["surfaces"]["codex/generated-choice"]["disposition"] == "stale"
    assert (
        updated["surfaces"]["codex/reviewed-choice"]
        == original["surfaces"]["codex/reviewed-choice"]
    )
    assert updated["workstreams"] == original["workstreams"]


def test_untrusted_preseeded_marker_does_not_suppress_applied_acknowledgement(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner()
    marker = f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->"
    runner.response_comments.append(
        {
            "id": 9002,
            "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-spoof",
            "body": (
                f"{marker}\nApplied cockpit override from "
                f"https://github.com/{REPO}/pull/88#issuecomment-{COMMENT_ID} and refreshed "
                "the generated cockpit artifacts on `codex/cockpit-refresh-20260713`.\n\n"
                "Accepted command:\n"
                f"```text\n{COMMENT_BODY}\n```\n\n"
                f"Pushed commit: [`{PUSHED_OID}`](https://github.com/{REPO}/commit/"
                f"{PUSHED_OID})"
            ),
            "created_at": "2026-07-13T11:59:00Z",
            "author_association": "NONE",
            "user": {"login": "attacker"},
        }
    )

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == (COMMENT_ID,)
    responses = [command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]]
    assert len(responses) == 1
    assert responses[0][-1].startswith(marker)
    assert runner.response_comments[-1]["user"] == {"login": "github-actions[bot]"}


def test_mismatched_trusted_marker_fails_before_override_commit_or_push(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner()
    runner.response_comments.append(
        {
            "id": 9002,
            "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-response",
            "body": (
                f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->\n"
                "Applied cockpit override with evidence that predates the commit."
            ),
            "created_at": "2026-07-13T11:59:00Z",
            "author_association": "NONE",
            "user": {"login": "github-actions[bot]"},
        }
    )

    with pytest.raises(RuntimeError, match="trusted curator response does not match"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    assert not any(command[:2] == ["git", "push"] for command in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "comment"] for command in runner.commands)


def test_curator_does_not_acknowledge_when_same_branch_push_fails(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(fail_push=True)

    with pytest.raises(RuntimeError, match="push failed"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert [
        "git",
        "push",
        "origin",
        "HEAD:codex/cockpit-refresh-20260713",
    ] in runner.commands
    assert not any(command[:3] == ["gh", "pr", "comment"] for command in runner.commands)


def test_retry_recovers_pushed_commit_when_initial_response_post_failed(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(fail_response_once=True)

    with pytest.raises(RuntimeError, match="response post failed"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    retry = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert retry.applied_comment_ids == ()
    assert retry.skipped_comment_ids == (COMMENT_ID,)
    assert len([command for command in runner.commands if command[:2] == ["git", "commit"]]) == 1
    assert len([command for command in runner.commands if command[:2] == ["git", "push"]]) == 1
    responses = [
        command[-1] for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    ]
    assert len(responses) == 2
    assert len(runner.response_comments) == 1
    recovered_response = responses[-1]
    assert f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->" in recovered_response
    assert f"```text\n{COMMENT_BODY}\n```" in recovered_response
    assert f"https://github.com/{REPO}/commit/{PUSHED_OID}" in recovered_response
    assert recovered_response.count(PUSHED_OID) == 2


def test_curator_rejects_invalid_post_commit_oid_before_push_or_ack(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(committed_head_oid="not-a-full-commit-oid")

    with pytest.raises(RuntimeError, match="validate the committed HEAD OID"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "push"] for command in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "comment"] for command in runner.commands)


def test_processed_comment_recovers_commit_ack_without_another_commit(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    receipt_path = tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": [COMMENT_ID]}, indent=2) + "\n",
        encoding="utf-8",
    )
    original_registry = (
        tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    ).read_bytes()
    runner = FakeRunner()
    _record_processed_commit(runner)

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == ()
    assert result.rejected_comment_ids == ()
    assert result.skipped_comment_ids == (COMMENT_ID,)
    assert (
        tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    ).read_bytes() == original_registry
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    response = next(
        command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    )
    assert f"```text\n{COMMENT_BODY}\n```" in response[-1]
    assert f"https://github.com/{REPO}/commit/{PUSHED_OID}" in response[-1]


def test_untrusted_marker_does_not_suppress_processed_commit_recovery(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    receipt_path = tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": [COMMENT_ID]}, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = FakeRunner()
    _record_processed_commit(runner)
    runner.response_comments.append(
        {
            "id": 9002,
            "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-spoof",
            "body": f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->\nspoof",
            "created_at": "2026-07-13T12:01:00Z",
            "author_association": "NONE",
            "user": {"login": "attacker"},
        }
    )

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.skipped_comment_ids == (COMMENT_ID,)
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    assert not any(command[:2] == ["git", "push"] for command in runner.commands)
    response = next(
        command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    )
    assert response[-1].startswith(f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->\n")
    assert f"https://github.com/{REPO}/commit/{PUSHED_OID}" in response[-1]


def test_processed_comment_fails_closed_for_mismatched_trusted_marker(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    receipt_path = tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": [COMMENT_ID]}, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = FakeRunner()
    _record_processed_commit(runner)
    runner.response_comments.append(
        {
            "id": 9002,
            "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-response",
            "body": (
                f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->\n"
                "Applied cockpit override with unverified evidence."
            ),
            "created_at": "2026-07-13T12:01:00Z",
            "author_association": "NONE",
            "user": {"login": "github-actions[bot]"},
        }
    )

    with pytest.raises(RuntimeError, match="trusted curator response does not match"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "comment"] for command in runner.commands)


def test_processed_comment_fails_closed_for_duplicate_trusted_markers(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    receipt_path = tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": [COMMENT_ID]}, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = FakeRunner()
    _record_processed_commit(runner)
    marker = f"<!-- mci-gru-cockpit-response:{COMMENT_ID} -->"
    exact_body = (
        f"{marker}\nApplied cockpit override from "
        f"https://github.com/{REPO}/pull/88#issuecomment-{COMMENT_ID} and refreshed the "
        "generated cockpit artifacts on `codex/cockpit-refresh-20260713`.\n\n"
        "Accepted command:\n"
        f"```text\n{COMMENT_BODY}\n```\n\n"
        f"Pushed commit: [`{PUSHED_OID}`](https://github.com/{REPO}/commit/{PUSHED_OID})"
    )
    runner.response_comments.extend(
        [
            {
                "id": 9002,
                "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-9002",
                "body": exact_body,
                "created_at": "2026-07-13T12:01:00Z",
                "author_association": "NONE",
                "user": {"login": "github-actions[bot]"},
            },
            {
                "id": 9003,
                "html_url": "https://github.com/magilliam27/MCI-GRU/pull/88#issuecomment-9003",
                "body": exact_body,
                "created_at": "2026-07-13T12:02:00Z",
                "author_association": "OWNER",
                "user": {"login": "magilliam27"},
            },
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate trusted curator responses"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "comment"] for command in runner.commands)


def test_processed_receipt_without_matching_commit_fails_closed(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    receipt_path = tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": [COMMENT_ID]}, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = FakeRunner()

    with pytest.raises(RuntimeError, match="uniquely recover"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    assert not any(command[:2] == ["git", "push"] for command in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "comment"] for command in runner.commands)


def test_nightly_reconciliation_does_not_repeat_duplicate_acknowledgement(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    receipt_path = tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": [COMMENT_ID]}, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = FakeRunner()
    _record_processed_commit(runner)

    for _ in range(2):
        result = run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )
        assert result.skipped_comment_ids == (COMMENT_ID,)

    responses = [command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]]
    assert len(responses) == 1
    assert "<!-- mci-gru-cockpit-response:9001 -->" in responses[0][-1]


@pytest.mark.parametrize(
    ("runner", "expected_reason"),
    [
        (
            FakeRunner(
                author_login="maintainer",
                author_association="MEMBER",
            ),
            "only the repository owner",
        ),
        (
            FakeRunner(body='/cockpit override workstream "LambdaRankIC" status parked'),
            "Malformed cockpit override command",
        ),
    ],
)
def test_rejected_comments_receive_response_without_registry_mutation(
    tmp_path: Path,
    runner: FakeRunner,
    expected_reason: str,
) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    original_registry = registry_path.read_bytes()

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == ()
    assert result.rejected_comment_ids == (COMMENT_ID,)
    assert result.skipped_comment_ids == ()
    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    response = next(
        command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    )
    assert expected_reason in response[-1]


def test_clear_generated_only_target_is_rejected_without_receipt_or_commit(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["workstreams"] = {}
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    original_registry = registry_path.read_bytes()
    runner = FakeRunner(body='/cockpit clear-override workstream "LambdaRankIC"')

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.rejected_comment_ids == (COMMENT_ID,)
    assert result.applied_comment_ids == ()
    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)
    response = next(
        command for command in runner.commands if command[:3] == ["gh", "pr", "comment"]
    )
    assert "No explicit workstream override exists" in response[-1]


def test_curator_rejects_dirty_checkout_before_fetch_or_switch(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(dirty=True)

    with pytest.raises(RuntimeError, match="clean disposable checkout"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "fetch"] for command in runner.commands)
    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)


def test_curator_rolls_back_its_files_when_refresh_fails(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    original_registry = registry_path.read_bytes()
    runner = FakeRunner()

    def failing_refresh(repo_root: Path, run_date: date) -> None:
        del repo_root, run_date
        raise RuntimeError("refresh failed")

    with pytest.raises(RuntimeError, match="refresh failed"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=failing_refresh,
        )

    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)


def test_curator_rejects_staged_paths_outside_allowlist_and_rolls_back(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    original_registry = registry_path.read_bytes()
    runner = FakeRunner(
        staged_paths=("docs/agents/cockpit/workstream-decisions.json", "secrets.txt")
    )

    with pytest.raises(RuntimeError, match="unexpected staged path"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)


def test_curator_rejects_pr_paths_outside_generated_artifact_allowlist(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(pr_paths=("docs/agents/workstreams.md", "scripts/refresh_cockpit.py"))

    with pytest.raises(RuntimeError, match="unexpected pull-request path"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "fetch"] for command in runner.commands)
    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)


def test_curator_rejects_api_path_set_missing_fetched_path_before_switch(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    original_registry = registry_path.read_bytes()
    runner = FakeRunner(
        pr_paths=(
            "docs/agents/workstreams.md",
            "docs/agents/cockpit/2026-07-13.md",
        ),
        local_diff_output=(
            "M\0docs/agents/workstreams.md\0"
            "M\0docs/agents/cockpit/2026-07-13.md\0"
            "M\0docs/agents/cockpit/auto-decisions.json\0"
        ),
    )

    with pytest.raises(RuntimeError, match="path evidence does not match exactly"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)


def test_curator_rejects_api_path_set_with_extra_path_before_switch(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    original_registry = registry_path.read_bytes()
    runner = FakeRunner(
        pr_paths=(
            "docs/agents/workstreams.md",
            "docs/agents/cockpit/2026-07-13.md",
            "docs/agents/cockpit/auto-decisions.json",
        ),
        local_diff_output=("M\0docs/agents/cockpit/2026-07-13.md\0M\0docs/agents/workstreams.md\0"),
    )

    with pytest.raises(
        RuntimeError,
        match="missing from fetched diff: docs/agents/cockpit/auto-decisions.json",
    ):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)


def test_curator_accepts_equal_rename_path_sets_in_different_order(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(
        pr_file_items=(
            {
                "filename": "docs/agents/workstreams.md",
                "previous_filename": "docs/agents/cockpit/2026-07-13.md",
            },
        ),
        local_diff_output=("R100\0docs/agents/cockpit/2026-07-13.md\0docs/agents/workstreams.md\0"),
    )

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == (COMMENT_ID,)
    assert any(command[:2] == ["git", "commit"] for command in runner.commands)


def test_curator_rejects_mismatched_rename_path_sets_before_switch(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    registry_path = tmp_path / "docs" / "agents" / "cockpit" / "workstream-decisions.json"
    original_registry = registry_path.read_bytes()
    runner = FakeRunner(
        pr_file_items=(
            {
                "filename": "docs/agents/workstreams.md",
                "previous_filename": "docs/agents/cockpit/2026-07-13.md",
            },
        ),
        local_diff_output=(
            "R100\0docs/agents/cockpit/auto-decisions.json\0docs/agents/workstreams.md\0"
        ),
    )

    with pytest.raises(RuntimeError, match="path evidence does not match exactly"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert registry_path.read_bytes() == original_registry
    assert not (tmp_path / "docs" / "agents" / "cockpit" / "override-receipts.json").exists()
    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)
    assert not any(command[:2] == ["git", "commit"] for command in runner.commands)


def test_curator_rejects_explicit_date_that_differs_from_validated_branch_date(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner()

    with pytest.raises(ValueError, match="does not match validated cockpit PR date"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 12),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "fetch"] for command in runner.commands)


def test_curator_rejects_fetched_base_that_changed_after_pr_validation(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(fetched_base_oid="c" * 40)

    with pytest.raises(RuntimeError, match="fetched base does not match"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert ["git", "fetch", "origin", "main"] in runner.commands
    assert ["git", "fetch", "origin", "codex/cockpit-refresh-20260713"] not in runner.commands


def test_curator_reuses_current_validated_branch_at_exact_head(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(
        current_branch="codex/cockpit-refresh-20260713",
        local_target_exists=True,
    )

    result = run_cockpit_override_curator(
        repo_root=tmp_path,
        run_date=date(2026, 7, 13),
        pr_number=88,
        comment_id=COMMENT_ID,
        repo=REPO,
        run_command=runner,
        refresh_cockpit=_noop_refresh,
    )

    assert result.applied_comment_ids == (COMMENT_ID,)
    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)


def test_curator_rejects_fetched_head_that_changed_after_pr_validation(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(fetched_oid="c" * 40)

    with pytest.raises(RuntimeError, match="fetched head does not match"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)


def test_curator_rejects_renamed_source_path_in_fetched_commit(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    runner = FakeRunner(
        pr_paths=("docs/agents/workstreams.md",),
        local_diff_output=("R100\0scripts/untrusted-refresh.py\0docs/agents/workstreams.md\0"),
    )

    with pytest.raises(RuntimeError, match="unexpected fetched pull-request path"):
        run_cockpit_override_curator(
            repo_root=tmp_path,
            run_date=date(2026, 7, 13),
            pr_number=88,
            comment_id=COMMENT_ID,
            repo=REPO,
            run_command=runner,
            refresh_cockpit=_noop_refresh,
        )

    assert not any(command[:2] == ["git", "switch"] for command in runner.commands)


def test_named_curator_branch_refresh_is_stable_across_head_sha_change(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")
    _git(repo_root, "config", "user.name", "Cockpit Test")
    _git(repo_root, "config", "user.email", "cockpit-test@example.invalid")
    _write_inputs(repo_root)
    _write_required_docs(repo_root)
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "Create cockpit baseline")
    _git(repo_root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo_root, "switch", "--create", "codex/cockpit-refresh-20260713")
    receipt_path = repo_root / "docs" / "agents" / "cockpit" / "override-receipts.json"
    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", str(receipt_path.relative_to(repo_root)))
    _git(repo_root, "commit", "-m", "Start named cockpit branch")

    run_date = date(2026, 7, 13)
    for message in ("Generate projected cockpit artifacts", "Stabilize committed comparison"):
        run_local_cockpit_refresh(
            repo_root=repo_root,
            run_date=run_date,
            github_evidence=None,
            projected_commits=1,
        )
        _git(repo_root, "add", "docs/agents/workstreams.md", "docs/agents/cockpit")
        _git(repo_root, "commit", "-m", message)
    run_local_cockpit_refresh(repo_root=repo_root, run_date=run_date, github_evidence=None)
    assert _git(repo_root, "status", "--porcelain") == ""

    receipt_path.write_text(
        json.dumps({"format_version": 1, "processed_comment_ids": ["sha-change"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    run_local_cockpit_refresh(
        repo_root=repo_root,
        run_date=run_date,
        github_evidence=None,
        projected_commits=1,
        ignored_dirty_paths=("docs/agents/cockpit/override-receipts.json",),
    )
    stable_bytes = {
        path: (repo_root / path).read_bytes()
        for path in (
            "docs/agents/workstreams.md",
            "docs/agents/cockpit/2026-07-13.md",
            "docs/agents/cockpit/auto-decisions.json",
        )
    }
    assert b"detached@" not in stable_bytes["docs/agents/cockpit/2026-07-13.md"]

    head_before = _git(repo_root, "rev-parse", "HEAD")
    _git(repo_root, "add", "docs/agents/workstreams.md", "docs/agents/cockpit")
    _git(repo_root, "commit", "-m", "Advance named branch SHA")
    assert _git(repo_root, "rev-parse", "HEAD") != head_before

    run_local_cockpit_refresh(repo_root=repo_root, run_date=run_date, github_evidence=None)

    assert {path: (repo_root / path).read_bytes() for path in stable_bytes} == stable_bytes
    assert _git(repo_root, "status", "--porcelain") == ""


def _write_required_docs(repo_root: Path) -> None:
    required = {
        "AGENTS.md": "# Agents\n",
        "docs/agents/domain.md": "# Domain\n",
        "docs/agents/issue-tracker.md": "# Issues\n",
        "docs/agents/triage-labels.md": "# Labels\n",
        "docs/research/README.md": "# Research\n",
        "docs/index.md": "# Index\n",
    }
    for relative, content in required.items():
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
