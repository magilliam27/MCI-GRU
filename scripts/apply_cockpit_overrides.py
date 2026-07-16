from __future__ import annotations

import argparse
import json
import re
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from cockpit.decisions import (
    DECISION_REGISTRY_PATH,
    overlay_auto_decisions,
    parse_decision_registry_text,
)
from cockpit.github import (
    authorize_owner_comment,
    cockpit_branch_name,
    collect_cockpit_pr_target,
    collect_pull_request_comments,
    collect_pull_request_paths,
    post_pull_request_response,
)
from cockpit.overrides import apply_override_command, parse_override_command
from cockpit.policy import AUTO_DECISIONS_PATH, read_auto_decisions
from cockpit.runner import run_local_cockpit_refresh

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cockpit.github import CockpitPullRequestTarget, PullRequestComment

    CommandRunner = Callable[[list[str]], str]
    CuratorRefresh = Callable[[Path, date], None]


RECEIPTS_PATH = "docs/agents/cockpit/override-receipts.json"
RECEIPTS_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CuratorRunResult:
    applied_comment_ids: tuple[str, ...] = ()
    rejected_comment_ids: tuple[str, ...] = ()
    skipped_comment_ids: tuple[str, ...] = ()


def run_cockpit_override_curator(
    *,
    repo_root: Path,
    pr_number: int,
    repo: str = "magilliam27/MCI-GRU",
    run_date: date | None = None,
    comment_id: str | None = None,
    run_command: CommandRunner | None = None,
    refresh_cockpit: CuratorRefresh | None = None,
) -> CuratorRunResult:
    """Apply new authenticated override comments to one validated cockpit PR."""
    runner = run_command or _run_command(repo_root)
    target = collect_cockpit_pr_target(
        pr_number=pr_number,
        repo=repo,
        repo_root=repo_root,
        run_command=runner,
    )
    if target is None or target.state != "open":
        raise RuntimeError("Cockpit curator requires a validated open same-repo cockpit PR.")
    branch_date = _run_date_from_branch(target.head_ref)
    if run_date is not None and run_date != branch_date:
        raise ValueError("Requested run date does not match validated cockpit PR date.")
    effective_date = branch_date
    allowed_paths = _curator_paths(effective_date)
    pull_request_paths = collect_pull_request_paths(
        pr_number=pr_number,
        repo=repo,
        repo_root=repo_root,
        run_command=runner,
    )
    if not pull_request_paths:
        raise RuntimeError("Cockpit curator requires complete pull-request path evidence.")
    pull_request_path_set = set(pull_request_paths)
    unexpected_paths = pull_request_path_set - set(allowed_paths)
    if unexpected_paths:
        raise RuntimeError(
            "Cockpit curator found an unexpected pull-request path: "
            + ", ".join(sorted(unexpected_paths))
        )
    _require_clean_checkout(runner)
    runner(["git", "fetch", "origin", target.base_ref])
    fetched_base_oid = runner(["git", "rev-parse", "FETCH_HEAD"]).strip().lower()
    if fetched_base_oid != target.base_oid:
        raise RuntimeError("Cockpit curator fetched base does not match validated PR base.")
    runner(["git", "fetch", "origin", target.head_ref])
    fetched_oid = runner(["git", "rev-parse", "FETCH_HEAD"]).strip().lower()
    if fetched_oid != target.head_oid:
        raise RuntimeError("Cockpit curator fetched head does not match validated PR head.")
    fetched_paths = _fetched_pull_request_paths(runner, target.base_oid)
    unexpected_fetched_paths = fetched_paths - set(allowed_paths)
    if unexpected_fetched_paths:
        raise RuntimeError(
            "Cockpit curator found an unexpected fetched pull-request path: "
            + ", ".join(sorted(unexpected_fetched_paths))
        )
    if pull_request_path_set != fetched_paths:
        missing_from_api = sorted(fetched_paths - pull_request_path_set)
        missing_from_fetched_diff = sorted(pull_request_path_set - fetched_paths)
        details = []
        if missing_from_api:
            details.append("missing from API: " + ", ".join(missing_from_api))
        if missing_from_fetched_diff:
            details.append("missing from fetched diff: " + ", ".join(missing_from_fetched_diff))
        raise RuntimeError(
            "Cockpit curator pull-request path evidence does not match exactly; "
            + "; ".join(details)
        )
    _switch_to_validated_head(runner, target.head_ref, target.head_oid)
    _require_clean_checkout(runner)

    comments = collect_pull_request_comments(
        pr_number=pr_number,
        repo=repo,
        repo_root=repo_root,
        run_command=runner,
    )
    if comments is None:
        raise RuntimeError("GitHub pull-request comment evidence is unavailable.")
    selected = tuple(
        comment
        for comment in comments
        if comment.body.startswith("/cockpit ")
        and (comment_id is None or comment.comment_id == comment_id)
    )
    if comment_id is not None and not selected:
        raise ValueError(f"Cockpit comment ID was not found: {comment_id}")

    registry_path = repo_root / DECISION_REGISTRY_PATH
    receipts_path = repo_root / RECEIPTS_PATH
    trusted_refresh = refresh_cockpit or _trusted_refresh
    registry = _read_json_object(registry_path, "cockpit decision registry")
    processed_ids = _read_receipts(receipts_path)
    auto = read_auto_decisions(repo_root)
    known_workstreams, known_branches, canonical_surfaces = _decision_context(registry, auto)
    applied: list[str] = []
    rejected: list[str] = []
    skipped: list[str] = []

    for comment in selected:
        if comment.comment_id in processed_ids:
            skipped.append(comment.comment_id)
            pushed_commit_oid = _recover_processed_commit(
                runner,
                comment_id=comment.comment_id,
                comment_url=comment.url,
                command_text=comment.body,
            )
            _post_curator_response_once(
                target,
                comments=comments,
                comment_id=comment.comment_id,
                repo=repo,
                run_command=runner,
                body=_applied_response_body(
                    repo=repo,
                    target_branch=target.head_ref,
                    comment_url=comment.url,
                    command_text=comment.body,
                    pushed_commit_oid=pushed_commit_oid,
                ),
            )
            continue
        authorization = authorize_owner_comment(
            comment,
            repo=repo,
            repo_root=repo_root,
            run_command=runner,
        )
        if not authorization.authorized:
            rejected.append(comment.comment_id)
            _post_curator_response_once(
                target,
                comments=comments,
                comment_id=comment.comment_id,
                repo=repo,
                run_command=runner,
                body=(
                    f"Rejected cockpit command from `{comment.author_login}`: "
                    f"{authorization.reason}. No registry change was made."
                ),
            )
            continue
        try:
            command = parse_override_command(
                comment.body,
                known_workstreams=known_workstreams,
                known_branches=known_branches,
            )
            application = apply_override_command(
                registry,
                command,
                command_id=comment.comment_id,
                applied_on=effective_date,
                canonical_surfaces=canonical_surfaces,
                processed_command_ids=processed_ids,
            )
            prospective_registry = parse_decision_registry_text(
                json.dumps(application.registry),
                known_workstreams=known_workstreams,
            )
            overlay_auto_decisions(prospective_registry, auto)
        except (TypeError, ValueError) as exc:
            rejected.append(comment.comment_id)
            _post_curator_response_once(
                target,
                comments=comments,
                comment_id=comment.comment_id,
                repo=repo,
                run_command=runner,
                body=f"Rejected cockpit command: {exc}. No registry change was made.",
            )
            continue
        if not application.applied:
            skipped.append(comment.comment_id)
            continue
        trusted_response = _trusted_curator_response(
            comments,
            comment.comment_id,
            repo=repo,
        )
        if trusted_response is not None:
            raise RuntimeError("Cockpit trusted curator response does not match expected evidence.")
        registry = application.registry
        processed_ids = application.processed_command_ids
        pushed_commit_oid = _refresh_and_commit(
            repo_root=repo_root,
            runner=runner,
            run_date=effective_date,
            target_branch=target.head_ref,
            comment_id=comment.comment_id,
            comment_url=comment.url,
            command_text=comment.body,
            registry=registry,
            processed_ids=processed_ids,
            refresh_cockpit=trusted_refresh,
        )
        _post_curator_response_once(
            target,
            comments=comments,
            comment_id=comment.comment_id,
            repo=repo,
            run_command=runner,
            body=_applied_response_body(
                repo=repo,
                target_branch=target.head_ref,
                comment_url=comment.url,
                command_text=comment.body,
                pushed_commit_oid=pushed_commit_oid,
            ),
        )
        applied.append(comment.comment_id)

    return CuratorRunResult(
        applied_comment_ids=tuple(applied),
        rejected_comment_ids=tuple(rejected),
        skipped_comment_ids=tuple(skipped),
    )


def _post_curator_response_once(
    target: CockpitPullRequestTarget,
    *,
    comments: tuple[PullRequestComment, ...],
    comment_id: str,
    body: str,
    run_command: CommandRunner,
    repo: str,
) -> None:
    marker = _curator_response_marker(comment_id)
    expected_body = f"{marker}\n{body}"
    trusted_response = _trusted_curator_response(
        comments,
        comment_id,
        repo=repo,
    )
    if trusted_response is not None:
        if trusted_response.body != expected_body:
            raise RuntimeError("Cockpit trusted curator response does not match expected evidence.")
        return
    post_pull_request_response(
        target,
        body=expected_body,
        run_command=run_command,
        repo=repo,
    )


def _curator_response_marker(comment_id: str) -> str:
    return f"<!-- mci-gru-cockpit-response:{comment_id} -->"


def _trusted_curator_response(
    comments: tuple[PullRequestComment, ...], comment_id: str, *, repo: str
) -> PullRequestComment | None:
    responses = _trusted_curator_responses(comments, comment_id, repo=repo)
    if len(responses) > 1:
        raise RuntimeError("Cockpit curator found duplicate trusted curator responses.")
    return responses[0] if responses else None


def _trusted_curator_responses(
    comments: tuple[PullRequestComment, ...], comment_id: str, *, repo: str
) -> tuple[PullRequestComment, ...]:
    marker = _curator_response_marker(comment_id)
    owner = repo.partition("/")[0].strip().casefold()
    return tuple(
        comment
        for comment in comments
        if marker in comment.body
        and (
            comment.author_login.casefold() == "github-actions[bot]"
            or (
                comment.author_login.casefold() == owner
                and comment.author_association.casefold() == "owner"
            )
        )
    )


def _applied_response_body(
    *,
    repo: str,
    target_branch: str,
    comment_url: str,
    command_text: str,
    pushed_commit_oid: str,
) -> str:
    return (
        f"Applied cockpit override from {comment_url} and refreshed the "
        f"generated cockpit artifacts on `{target_branch}`.\n\n"
        "Accepted command:\n"
        f"```text\n{command_text}\n```\n\n"
        f"Pushed commit: [`{pushed_commit_oid}`](https://github.com/{repo}/commit/"
        f"{pushed_commit_oid})"
    )


def _recover_processed_commit(
    runner: CommandRunner,
    *,
    comment_id: str,
    comment_url: str,
    command_text: str,
) -> str:
    subject = f"Apply cockpit override from comment {comment_id}"
    candidates: list[str] = []
    for line in runner(["git", "log", "--format=%H%x09%s", "HEAD", "--"]).splitlines():
        oid, separator, commit_subject = line.partition("\t")
        if separator and commit_subject == subject:
            candidates.append(oid.lower())
    if len(candidates) != 1:
        raise RuntimeError(
            "Cockpit curator could not uniquely recover the processed override commit."
        )
    commit_oid = candidates[0]
    if re.fullmatch(r"[0-9a-f]{40}", commit_oid) is None:
        raise RuntimeError("Cockpit curator recovered an invalid override commit OID.")
    message = runner(["git", "show", "--no-patch", "--format=%B", commit_oid]).rstrip("\r\n")
    expected_message = f"{subject}\n\n{comment_url}\n\n{command_text}"
    if message != expected_message:
        raise RuntimeError(
            "Cockpit curator recovered override commit evidence that does not match the comment."
        )
    return commit_oid


def _decision_context(
    registry: Mapping[str, object], auto
) -> tuple[set[str], set[str], dict[str, str]]:
    raw_workstreams = registry.get("workstreams")
    raw_surfaces = registry.get("surfaces")
    if not isinstance(raw_workstreams, dict) or not isinstance(raw_surfaces, dict):
        raise ValueError("Cockpit decision registry requires workstreams and surfaces objects.")
    known_workstreams = {*raw_workstreams, *auto.workstreams}
    known_branches = {*raw_surfaces, *auto.surfaces}
    canonical_surfaces: dict[str, str] = {}
    for name, value in raw_workstreams.items():
        if not isinstance(value, dict):
            continue
        canonical = value.get("canonical_surface")
        if isinstance(canonical, str) and canonical.strip():
            canonical_surfaces[name] = canonical
    for name, decision in auto.workstreams.items():
        canonical_surfaces.setdefault(name, decision.canonical_surface)
    return known_workstreams, known_branches, canonical_surfaces


def _refresh_and_commit(
    *,
    repo_root: Path,
    runner: CommandRunner,
    run_date: date,
    target_branch: str,
    comment_id: str,
    comment_url: str,
    command_text: str,
    registry: Mapping[str, object],
    processed_ids: tuple[str, ...],
    refresh_cockpit: CuratorRefresh,
) -> str:
    paths = _curator_paths(run_date)
    snapshot = _snapshot_files(repo_root, paths)
    committed = False
    try:
        _write_json(repo_root / DECISION_REGISTRY_PATH, registry)
        _write_receipts(repo_root / RECEIPTS_PATH, processed_ids)
        refresh_cockpit(repo_root, run_date)
        runner(["git", "add", "--", *paths])
        staged_paths = {
            line.strip()
            for line in runner(["git", "diff", "--cached", "--name-only"]).splitlines()
            if line.strip()
        }
        unexpected = staged_paths - set(paths)
        if unexpected:
            raise RuntimeError(
                "Cockpit curator found an unexpected staged path: " + ", ".join(sorted(unexpected))
            )
        if not staged_paths:
            raise RuntimeError("Cockpit curator produced no staged changes.")
        runner(
            [
                "git",
                "commit",
                "-m",
                f"Apply cockpit override from comment {comment_id}",
                "-m",
                comment_url,
                "-m",
                command_text,
            ]
        )
        committed = True
        commit_oid = runner(["git", "rev-parse", "HEAD"]).strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", commit_oid) is None:
            raise RuntimeError("Cockpit curator could not validate the committed HEAD OID.")
        runner(["git", "push", "origin", f"HEAD:{target_branch}"])
        return commit_oid
    except Exception:
        if not committed:
            _restore_snapshot(repo_root, paths, snapshot, runner)
        raise


def _curator_paths(run_date: date) -> tuple[str, ...]:
    return (
        DECISION_REGISTRY_PATH,
        RECEIPTS_PATH,
        AUTO_DECISIONS_PATH,
        "docs/agents/workstreams.md",
        f"docs/agents/cockpit/{run_date.isoformat()}.md",
    )


def _fetched_pull_request_paths(runner: CommandRunner, base_oid: str) -> set[str]:
    output = runner(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            f"{base_oid}...FETCH_HEAD",
            "--",
        ]
    )
    fields = output.split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    paths: set[str] = set()
    index = 0
    while index < len(fields):
        status_parts = fields[index].split("\t")
        index += 1
        status = status_parts[0]
        if not status or status[0] not in "ACDMRTUXB":
            raise RuntimeError("Cockpit curator received malformed fetched path evidence.")
        expected_paths = 2 if status[0] in "RC" else 1
        path_values = status_parts[1:]
        while len(path_values) < expected_paths and index < len(fields):
            path_values.append(fields[index])
            index += 1
        if len(path_values) != expected_paths or any(not path for path in path_values):
            raise RuntimeError("Cockpit curator received malformed fetched path evidence.")
        paths.update(path_values)
    if not paths:
        raise RuntimeError("Cockpit curator found no paths in the validated fetched PR diff.")
    return paths


def _trusted_refresh(repo_root: Path, run_date: date) -> None:
    """Refresh via modules imported from the trusted startup checkout."""
    run_local_cockpit_refresh(
        repo_root=repo_root,
        run_date=run_date,
        ignored_dirty_paths=(DECISION_REGISTRY_PATH, RECEIPTS_PATH),
        automation_branch=cockpit_branch_name(run_date),
    )


def _require_clean_checkout(runner: CommandRunner) -> None:
    if runner(["git", "status", "--porcelain=v1"]).strip():
        raise RuntimeError("Cockpit curator requires a clean disposable checkout.")


def _switch_to_validated_head(runner: CommandRunner, branch: str, head_oid: str) -> None:
    current = runner(["git", "branch", "--show-current"]).strip()
    if current != branch:
        if runner(["git", "branch", "--list", branch]).strip():
            runner(["git", "switch", branch])
        else:
            runner(["git", "switch", "--create", branch, "FETCH_HEAD"])
    local_oid = runner(["git", "rev-parse", "HEAD"]).strip().lower()
    if local_oid != head_oid:
        raise RuntimeError("Cockpit curator local branch does not match validated PR head.")


def _snapshot_files(repo_root: Path, paths: tuple[str, ...]) -> dict[str, bytes | None]:
    return {
        relative: path.read_bytes() if (path := repo_root / relative).exists() else None
        for relative in paths
    }


def _restore_snapshot(
    repo_root: Path,
    paths: tuple[str, ...],
    snapshot: Mapping[str, bytes | None],
    runner: CommandRunner,
) -> None:
    with suppress(Exception):
        runner(["git", "restore", "--staged", "--", *paths])
    for relative in paths:
        path = repo_root / relative
        original = snapshot[relative]
        if original is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)


def _read_receipts(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    payload = _read_json_object(path, "cockpit override receipts")
    if set(payload) != {"format_version", "processed_comment_ids"}:
        raise ValueError("Invalid cockpit override receipt contract.")
    if payload["format_version"] != RECEIPTS_FORMAT_VERSION:
        raise ValueError("Unsupported cockpit override receipt format_version.")
    raw_ids = payload["processed_comment_ids"]
    if not isinstance(raw_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw_ids
    ):
        raise ValueError("processed_comment_ids must be a list of non-empty strings.")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("processed_comment_ids must be unique.")
    return tuple(sorted(raw_ids))


def _write_receipts(path: Path, processed_ids: tuple[str, ...]) -> None:
    _write_json(
        path,
        {
            "format_version": RECEIPTS_FORMAT_VERSION,
            "processed_comment_ids": list(sorted(processed_ids)),
        },
    )


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_date_from_branch(branch: str) -> date:
    raw = branch.removeprefix("codex/cockpit-refresh-")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"Cockpit branch does not contain a valid run date: {branch}")
    return date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}")


def _run_command(repo_root: Path) -> CommandRunner:
    def run(command: list[str]) -> str:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply authenticated cockpit PR overrides.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repo", default="magilliam27/MCI-GRU")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--comment-id")
    parser.add_argument("--date", type=date.fromisoformat)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_cockpit_override_curator(
        repo_root=args.repo_root.resolve(),
        run_date=args.date,
        pr_number=args.pr_number,
        comment_id=args.comment_id,
        repo=args.repo,
    )
    print(
        "Cockpit curator: "
        f"applied={len(result.applied_comment_ids)} "
        f"rejected={len(result.rejected_comment_ids)} "
        f"skipped={len(result.skipped_comment_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
