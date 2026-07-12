from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from cockpit.decisions import DECISION_REGISTRY_PATH
from cockpit.evidence import collect_local_evidence
from cockpit.models import WorkstreamStatus
from cockpit.runner import (
    GitActivitySource,
    StaticWorkstreamSource,
    WorkstreamSeed,
    _branch_topic_tokens,
    _run_command,
    merge_workstream_sources,
    run_github_cockpit_refresh,
    run_local_cockpit_refresh,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FixedSource:
    """Test double that proposes a fixed seed list regardless of evidence."""

    def __init__(self, seeds: list[WorkstreamSeed]) -> None:
        self._seeds = list(seeds)

    def provide(self, evidence: object, run_date: date) -> list[WorkstreamSeed]:
        return list(self._seeds)


def test_collect_local_evidence_records_dirty_paths_and_required_docs(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    (repo / "docs" / "agents").mkdir(parents=True)
    (repo / "docs" / "agents" / "domain.md").write_text("# Domain\n", encoding="utf-8")
    (repo / "docs" / "research").mkdir(parents=True)
    (repo / "docs" / "research" / "README.md").write_text("# Research\n", encoding="utf-8")
    registry_path = repo / DECISION_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(_empty_registry(), encoding="utf-8")
    (repo / "docs" / "handoffs").mkdir(parents=True)
    (repo / "docs" / "handoffs" / "2026-06-01-note.md").write_text(
        "# Handoff\n",
        encoding="utf-8",
    )

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return " M docs/agents/domain.md\n?? scratch.txt\n"
        if command == "git status --short --branch":
            return "## codex/example\n M docs/agents/domain.md\n?? scratch.txt\n"
        if command == "git branch --format=%(refname:short)":
            return "codex/example\nmain\n"
        if command == "git branch --all --no-merged origin/main":
            return "  codex/example\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "1\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc123\nbranch refs/heads/codex/example\n"
        if command == "git log -5 --oneline":
            return "abc123 Add cockpit design\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## codex/example\n"
        raise AssertionError(command)

    evidence = collect_local_evidence(repo, run_command=fake_run)

    assert evidence.required_docs["AGENTS.md"] is True
    assert evidence.required_docs["docs/agents/domain.md"] is True
    assert evidence.required_docs["docs/research/README.md"] is True
    assert evidence.required_docs[DECISION_REGISTRY_PATH] is True
    assert evidence.recent_handoffs == ["docs/handoffs/2026-06-01-note.md"]
    assert evidence.dirty_paths == ["docs/agents/domain.md", "scratch.txt"]
    assert evidence.branches == ["codex/example", "main"]
    assert "abc123 Add cockpit design" in evidence.recent_commits


def test_collect_local_evidence_builds_git_topology_snapshot(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return "?? scratch.txt\n"
        if command == "git status --short --branch":
            return "## codex/top10-lambdarank-screen-20260625...origin/codex/top10-lambdarank-screen-20260625\n?? scratch.txt\n"
        if command == "git branch --format=%(refname:short)":
            return (
                "codex/top10-lambdarank-screen-20260625\ncodex/portfolio-ic-hybrid-testing\nmain\n"
            )
        if command == "git branch --all --no-merged origin/main":
            return (
                "  codex/top10-lambdarank-screen-20260625\n"
                "  remotes/origin/codex/top10-lambdarank-screen-20260625\n"
                "+ codex/portfolio-ic-hybrid-testing\n"
            )
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "2\t8\n"
        if command == "git worktree list --porcelain":
            return (
                "worktree C:/repo\n"
                "HEAD 5274d21\n"
                "branch refs/heads/codex/top10-lambdarank-screen-20260625\n"
                "\n"
                "worktree C:/repo/.codex/worktrees/detached/MCI-GRU\n"
                "HEAD a2684d2\n"
                "detached\n"
            )
        if command == "git log -5 --oneline":
            return "5274d21 Promote research maps and agent process docs\n"
        if (
            args[:2] == ["git", "-c"]
            and args[3] == "-C"
            and args[5:10]
            == [
                "status",
                "--porcelain=v1",
                "-b",
                "--untracked-files=all",
            ]
        ):
            if args[4] == "C:/repo":
                return "## codex/top10-lambdarank-screen-20260625...origin/codex/top10-lambdarank-screen-20260625\n?? scratch.txt\n"
            if args[4] == "C:/repo/.codex/worktrees/detached/MCI-GRU":
                return "## HEAD (no branch)\n M tests/test_training_efficiency_config.py\n"
        raise AssertionError(command)

    evidence = collect_local_evidence(repo, run_command=fake_run)

    assert evidence.git_topology.current_branch == "codex/top10-lambdarank-screen-20260625"
    assert evidence.git_topology.origin_main_ahead == 8
    assert evidence.git_topology.origin_main_behind == 2
    assert evidence.git_topology.unmerged_branches == [
        "codex/top10-lambdarank-screen-20260625",
        "codex/portfolio-ic-hybrid-testing",
    ]
    assert [
        branch.provenance_label for branch in evidence.git_topology.unmerged_branch_details
    ] == [
        "local+remote",
        "local",
    ]
    assert [worktree.branch for worktree in evidence.git_topology.dirty_worktrees] == [
        "codex/top10-lambdarank-screen-20260625",
        "detached@a2684d2",
    ]
    assert [worktree.path for worktree in evidence.git_topology.detached_worktrees] == [
        "C:/repo/.codex/worktrees/detached/MCI-GRU"
    ]


def test_default_cockpit_runners_apply_safe_directory_to_git_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    seen_evidence_commands: list[list[str]] = []

    def fake_evidence_run(
        args: list[str],
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        seen_evidence_commands.append(args)
        command = " ".join(_strip_safe_directory_args(args))
        if command.startswith("git for-each-ref"):
            return SimpleNamespace(stdout="")
        if command == "git status --short --branch":
            return SimpleNamespace(stdout="## main\n")
        if command == "git branch --format=%(refname:short)":
            return SimpleNamespace(stdout="main\n")
        if command == "git branch --all --no-merged origin/main":
            return SimpleNamespace(stdout="")
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return SimpleNamespace(stdout="0\t0\n")
        if command == "git worktree list --porcelain":
            return SimpleNamespace(
                stdout=f"worktree {repo.as_posix()}\nHEAD 60e3d96\nbranch refs/heads/main\n"
            )
        if command == f"git -C {repo.as_posix()} status --porcelain=v1 -b --untracked-files=all":
            return SimpleNamespace(stdout="## main\n")
        if command == "git log -5 --oneline":
            return SimpleNamespace(stdout="60e3d96 Add local MCI-GRU cockpit runner\n")
        raise AssertionError(command)

    monkeypatch.setattr("cockpit.evidence.subprocess.run", fake_evidence_run)

    collect_local_evidence(repo)

    primary_git_commands = [
        command
        for command in seen_evidence_commands
        if command[:1] == ["git"] and "-C" not in command
    ]
    assert primary_git_commands
    assert all(command[1:3] == ["-c", f"safe.directory={repo}"] for command in primary_git_commands)

    seen_runner_commands: list[list[str]] = []

    def fake_runner_run(
        args: list[str],
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        seen_runner_commands.append(args)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("cockpit.runner.subprocess.run", fake_runner_run)
    _run_command(repo)(["git", "status", "--short"])

    assert seen_runner_commands == [["git", "-c", f"safe.directory={repo}", "status", "--short"]]


def test_run_local_cockpit_refresh_writes_register_and_packet(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## codex/italy-work-snapshot-20260605\n"
        if command == "git branch --format=%(refname:short)":
            return "codex/italy-work-snapshot-20260605\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD a1b5de5\n"
        if command == "git log -5 --oneline":
            return "a1b5de5 Add MCI-GRU cockpit agent design\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## codex/italy-work-snapshot-20260605\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 20),
        run_command=fake_run,
    )

    assert result.register_path == repo / "docs" / "agents" / "workstreams.md"
    assert result.packet_path == repo / "docs" / "agents" / "cockpit" / "2026-06-20.md"
    assert result.register_path.exists()
    assert result.packet_path.exists()
    register = result.register_path.read_text(encoding="utf-8")
    assert "Git surface: codex/italy-work-snapshot-20260605" in register
    assert "LambdaRankIC" not in register
    assert "Git and worktree hygiene" in register
    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "yellow"
    assert "**Run color:** yellow" in packet
    assert "Cockpit generated with git topology attention items:" in packet


def test_run_local_cockpit_refresh_surfaces_git_topology_without_placeholders(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## codex/top10-lambdarank-screen-20260625...origin/codex/top10-lambdarank-screen-20260625\n"
        if command == "git branch --format=%(refname:short)":
            return (
                "codex/top10-lambdarank-screen-20260625\ncodex/portfolio-ic-hybrid-testing\nmain\n"
            )
        if command == "git branch --all --no-merged origin/main":
            return (
                "  codex/top10-lambdarank-screen-20260625\n"
                "  remotes/origin/codex/lambdarankic-1024-all-years-20260625\n"
                "+ codex/portfolio-ic-hybrid-testing\n"
            )
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "2\t8\n"
        if command == "git worktree list --porcelain":
            return (
                "worktree C:/repo\n"
                "HEAD 5274d21\n"
                "branch refs/heads/codex/top10-lambdarank-screen-20260625\n"
                "\n"
                "worktree C:/repo/.codex/worktrees/portfolio-ic-hybrid-testing/MCI-GRU\n"
                "HEAD 0581cf1\n"
                "branch refs/heads/codex/portfolio-ic-hybrid-testing\n"
            )
        if command == "git log -5 --oneline":
            return "5274d21 Promote research maps and agent process docs\n"
        if (
            args[:2] == ["git", "-c"]
            and args[3] == "-C"
            and args[5:10]
            == [
                "status",
                "--porcelain=v1",
                "-b",
                "--untracked-files=all",
            ]
        ):
            return f"## {args[4]}\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 28),
        run_command=fake_run,
    )

    register = result.register_path.read_text(encoding="utf-8")
    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "yellow"
    assert "See latest branch, worktree, issue, or handoff evidence." not in register
    assert "Initial cockpit seed from approved design." not in register
    assert "Regime CSV contract" not in register
    assert "codex/top10-lambdarank-screen-20260625" in register
    assert "`origin/codex/lambdarankic-1024-all-years-20260625` (remote-only)" in register
    assert "codex/portfolio-ic-hybrid-testing" in register
    assert "## Git Tree Impact" in packet
    assert "Current branch: `codex/top10-lambdarank-screen-20260625`" in packet
    assert "origin/main divergence: 8 ahead / 2 behind" in packet
    assert "Unmerged branches: 3" in packet
    assert (
        "Unmerged branch names: `codex/top10-lambdarank-screen-20260625` (local), "
        "`origin/codex/lambdarankic-1024-all-years-20260625` (remote-only), "
        "`codex/portfolio-ic-hybrid-testing` (local)"
    ) in packet
    assert "Worktrees: 2 total, 0 detached, 0 dirty" in packet
    assert "git rev-list --left-right --count origin/main...HEAD" in packet
    assert "## Workstreams Needing Decisions" in packet
    assert (
        "- **LambdaRankIC** (needs-user-decision): Choose the canonical continuation "
        "surface before continuing this workstream."
    ) in packet
    assert (
        "- **Portfolio-IC** (needs-user-decision): Decide whether to promote, park, "
        "or rerun current evidence."
    ) in packet
    assert (
        "- **LambdaRankIC:** Choose the canonical continuation surface before continuing this workstream."
        in packet
    )
    assert (
        "- **Portfolio-IC:** Decide whether to promote, park, or rerun current evidence." in packet
    )


def test_decision_registry_keeps_reviewed_surfaces_resolved(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={
            "LambdaRankIC": {
                "status": "active",
                "canonical_surface": "PR #65 / codex/canonical-lambdarank",
                "reason": "Recovery guardrails are the reviewed continuation.",
                "next_action": "Fix PR #65 lint before runtime work.",
                "last_reviewed": "2026-07-09",
            },
            "Daily bug scans": {
                "status": "ready-for-agent",
                "canonical_surface": "PR #67 / codex/backtest-plot-test",
                "reason": "The focused regression PR is the active scan result.",
                "next_action": "Format the test and rerun CI.",
                "last_reviewed": "2026-07-09",
            },
        },
        surfaces={
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
            "codex/backtest-plot-test": {
                "workstreams": ["Daily bug scans"],
                "disposition": "canonical",
                "reason": "Explicit assignment must beat branch-name heuristics.",
                "next_action": "Continue through PR #67.",
                "last_reviewed": "2026-07-09",
            },
        },
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner(
            [
                "codex/canonical-lambdarank",
                "codex/old-lambdarank",
                "codex/backtest-plot-test",
            ]
        ),
    )

    by_name = {row.name: row for row in result.report.active_workstreams}
    register = result.register_path.read_text(encoding="utf-8")
    assert by_name["LambdaRankIC"].continuation == "PR #65 / codex/canonical-lambdarank"
    assert by_name["LambdaRankIC"].last_reviewed == date(2026, 7, 9)
    assert "workstream-decisions.json" in by_name["LambdaRankIC"].source_of_truth
    assert (
        "| Daily bug scans | ready-for-agent |  | PR #67 / codex/backtest-plot-test |" in register
    )
    assert not any(row.name == "LambdaRankIC" for row in result.report.decision_workstreams)
    assert not any(
        decision.workstream == "Git and worktree hygiene" for decision in result.report.decisions
    )
    assert "live git surface(s) needing classification" not in result.report.executive_summary
    assert any(
        row.name == "Git surface: codex/old-lambdarank"
        for row in result.report.stale_or_archive_candidates
    )


def test_decision_registry_reopens_only_for_new_unreviewed_surface(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={
            "LambdaRankIC": {
                "status": "active",
                "canonical_surface": "PR #65 / codex/canonical-lambdarank",
                "reason": "Recovery guardrails are the reviewed continuation.",
                "next_action": "Fix PR #65 lint before runtime work.",
                "last_reviewed": "2026-07-09",
            }
        },
        surfaces={
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
                "reason": "Already reviewed and superseded.",
                "next_action": "Remove only after cleanup approval.",
                "last_reviewed": "2026-07-09",
            },
        },
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner(
            [
                "codex/canonical-lambdarank",
                "codex/old-lambdarank",
                "codex/lambdarank-new-experiment",
            ]
        ),
    )

    decision = next(row for row in result.report.decision_workstreams if row.name == "LambdaRankIC")
    assert decision.continuation == "PR #65 / codex/canonical-lambdarank"
    assert decision.blocked_on == (
        "New unreviewed surfaces since the 2026-07-09 decision: "
        "`codex/lambdarank-new-experiment` (local)"
    )
    assert decision.next_action == (
        "Review only the new surface(s) against the recorded canonical decision."
    )


def test_run_local_cockpit_refresh_surfaces_unmatched_real_git_topology(
    tmp_path: Path,
) -> None:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    worktree = tmp_path / "unmapped-worktree"
    _git(["init", "--bare", str(origin)], cwd=tmp_path)
    _git(["init", str(repo)], cwd=tmp_path)
    _git(["config", "user.email", "codex@example.com"], cwd=repo)
    _git(["config", "user.name", "Codex"], cwd=repo)
    _repo_with_required_docs(repo)
    _git(["add", "."], cwd=repo)
    _git(["commit", "-m", "initial docs"], cwd=repo)
    _git(["branch", "-M", "main"], cwd=repo)
    _git(["remote", "add", "origin", str(origin)], cwd=repo)
    _git(["push", "-u", "origin", "main"], cwd=repo)
    _git(["switch", "-c", "codex/unmapped-local"], cwd=repo)
    (repo / "local.txt").write_text("local branch\n", encoding="utf-8")
    _git(["add", "local.txt"], cwd=repo)
    _git(["commit", "-m", "local branch work"], cwd=repo)
    _git(["switch", "main"], cwd=repo)
    _git(["worktree", "add", "-b", "codex/unmapped-worktree", str(worktree)], cwd=repo)
    (worktree / "worktree.txt").write_text("worktree branch\n", encoding="utf-8")
    _git(["add", "worktree.txt"], cwd=worktree)
    _git(["commit", "-m", "worktree branch work"], cwd=worktree)
    _git(["switch", "-c", "codex/unmapped-remote"], cwd=repo)
    (repo / "remote.txt").write_text("remote branch\n", encoding="utf-8")
    _git(["add", "remote.txt"], cwd=repo)
    _git(["commit", "-m", "remote branch work"], cwd=repo)
    _git(["push", "origin", "HEAD:refs/heads/codex/unmapped-remote"], cwd=repo)
    _git(["switch", "main"], cwd=repo)
    _git(["branch", "-D", "codex/unmapped-remote"], cwd=repo)
    _git(["fetch", "--prune", "origin"], cwd=repo)

    result = run_local_cockpit_refresh(repo, date(2026, 6, 28))

    register = result.register_path.read_text(encoding="utf-8")
    packet = result.packet_path.read_text(encoding="utf-8")
    assert "Git surface: codex/unmapped-local" in register
    assert "Git surface: codex/unmapped-worktree" in register
    assert "Git surface: codex/unmapped-remote" in register
    assert "`origin/codex/unmapped-remote` (remote-only)" in register
    assert "`codex/unmapped-local` (local)" in packet
    assert "`codex/unmapped-worktree` (local)" in packet
    assert "`origin/codex/unmapped-remote` (remote-only)" in packet


def test_run_local_cockpit_refresh_labels_detached_current_checkout(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## HEAD (no branch)\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n"
        if command == "git branch --all --no-merged origin/main":
            return "* (HEAD detached at a2684d2)\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD a2684d2\ndetached\n"
        if command == "git log -5 --oneline":
            return "a2684d2 Merge latest main\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## HEAD (no branch)\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 28),
        run_command=fake_run,
    )

    register = result.register_path.read_text(encoding="utf-8")
    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "yellow"
    assert "Git surface: HEAD (no branch)" not in register
    assert "`HEAD (no branch)` (local)" not in register
    assert "Git surface: (HEAD detached at a2684d2)" not in register
    assert "`(HEAD detached at a2684d2)` (local)" not in register
    assert "Git surface: detached@a2684d2" in register
    assert "`detached@a2684d2` @ `C:/repo` (detached)" in register
    assert "Current branch: `detached@a2684d2`" in packet
    assert "Detached worktrees: `detached@a2684d2` at `C:/repo`" in packet


def test_run_local_cockpit_refresh_uses_repo_path_for_detached_current_checkout(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    other = tmp_path / "other-detached"

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## HEAD (no branch)\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return (
                f"worktree {other.as_posix()}\n"
                "HEAD 1111111\n"
                "detached\n"
                "\n"
                f"worktree {repo.as_posix()}\n"
                "HEAD 2222222\n"
                "detached\n"
            )
        if command == "git log -5 --oneline":
            return "2222222 Current detached checkout\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## HEAD (no branch)\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 28),
        run_command=fake_run,
    )

    register = result.register_path.read_text(encoding="utf-8")
    packet = result.packet_path.read_text(encoding="utf-8")
    assert "Current branch: `detached@2222222`" in packet
    assert "Git surface: detached@2222222" in register
    assert "Current branch: `detached@1111111`" not in packet
    assert "Git surface: detached@1111111" in register


def test_run_local_cockpit_refresh_suppresses_seeds_for_main_divergence(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## main...origin/main [ahead 1]\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t1\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc1234\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "abc1234 Local main commit\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main...origin/main [ahead 1]\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 28),
        run_command=fake_run,
    )

    register = result.register_path.read_text(encoding="utf-8")
    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "yellow"
    assert "LambdaRankIC" not in register
    assert "Regime CSV contract" not in register
    assert "Git and worktree hygiene" in register
    assert "origin/main divergence: 1 ahead / 0 behind" in packet
    assert "live git surface(s) needing classification" not in packet


def test_run_local_cockpit_refresh_reports_dirty_paths(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return " M docs/agents/domain.md\n?? scratch.txt\n"
        if command == "git status --short --branch":
            return (
                "## codex/italy-work-snapshot-20260605\n M docs/agents/domain.md\n?? scratch.txt\n"
            )
        if command == "git branch --format=%(refname:short)":
            return "codex/italy-work-snapshot-20260605\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD a1b5de5\n"
        if command == "git log -5 --oneline":
            return "a1b5de5 Add MCI-GRU cockpit agent design\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## codex/italy-work-snapshot-20260605\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 20),
        run_command=fake_run,
    )

    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.dirty_paths == ["docs/agents/domain.md", "scratch.txt"]
    assert "**Run color:** red" in packet
    assert "Dirty paths before cockpit write: docs/agents/domain.md, scratch.txt" in packet


def test_run_local_cockpit_refresh_explains_missing_required_docs(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    (repo / "docs" / "research" / "README.md").unlink()

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## main\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD 60e3d96\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "60e3d96 Add local MCI-GRU cockpit runner\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 20),
        run_command=fake_run,
    )

    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "red"
    assert "Cockpit generated, but required docs are missing:" in packet
    assert "docs/research/README.md" in packet


def test_run_local_cockpit_refresh_reports_missing_decision_registry(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    (repo / DECISION_REGISTRY_PATH).unlink()

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner([]),
    )

    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "red"
    assert f"Missing required doc: {DECISION_REGISTRY_PATH}" in packet


def test_run_github_cockpit_refresh_switches_branch_and_syncs(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git switch -C codex/cockpit-refresh-20260620":
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## main\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD 60e3d96\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "60e3d96 Add local MCI-GRU cockpit runner\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main\n"
        if command == "gh auth status":
            return "Logged in"
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return "M docs/agents/workstreams.md\nM docs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git add docs/agents/workstreams.md"):
            return ""
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            return ""
        if command.startswith("gh pr list"):
            return "https://github.com/magilliam27/MCI-GRU/pull/99"
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\n"
        if command.startswith("gh issue edit 100"):
            return ""
        if command.startswith("gh issue comment 100"):
            return ""
        if command == "git status --short -- docs/agents/cockpit/2026-06-20.md":
            return ""
        raise AssertionError(command)

    result = run_github_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 20),
        run_command=fake_run,
    )

    assert commands[0] == ["git", "switch", "-C", "codex/cockpit-refresh-20260620"]
    assert result.github is not None
    assert result.github.pr_url == "https://github.com/magilliam27/MCI-GRU/pull/99"
    assert result.register_path.exists()
    assert any(command[:3] == ["gh", "issue", "comment"] for command in commands)


def test_run_github_cockpit_refresh_rewrites_packet_with_actions_taken(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git switch -C codex/cockpit-refresh-20260620":
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## codex/cockpit-refresh-20260620\n"
        if command == "git branch --format=%(refname:short)":
            return "codex/cockpit-refresh-20260620\nmain\n"
        if command == "git branch --all --no-merged origin/main":
            return "  codex/cockpit-refresh-20260620\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "1\t0\n"
        if command == "git worktree list --porcelain":
            return (
                "worktree C:/repo\nHEAD 60e3d96\nbranch refs/heads/codex/cockpit-refresh-20260620\n"
            )
        if command == "git log -5 --oneline":
            return "60e3d96 Add local MCI-GRU cockpit runner\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## codex/cockpit-refresh-20260620\n"
        if command == "gh auth status":
            return "Logged in"
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return "M docs/agents/workstreams.md\nM docs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git add docs/agents/workstreams.md"):
            return ""
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            return ""
        if command.startswith("gh pr list"):
            return "https://github.com/magilliam27/MCI-GRU/pull/99"
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\n"
        if command.startswith("gh issue edit 100"):
            return ""
        if command.startswith("gh issue comment 100"):
            return ""
        if command == "git status --short -- docs/agents/cockpit/2026-06-20.md":
            return "M docs/agents/cockpit/2026-06-20.md\n"
        if command == "git add docs/agents/cockpit/2026-06-20.md":
            return ""
        if command.startswith("git commit -m Record cockpit GitHub sync for 2026-06-20"):
            return (
                "[codex/cockpit-refresh-20260620 def456] Record cockpit GitHub sync for 2026-06-20"
            )
        raise AssertionError(command)

    result = run_github_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 6, 20),
        run_command=fake_run,
    )

    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.github is not None
    assert "GitHub Actions Taken" in packet
    assert "Git topology snapshot timing: before GitHub sync commits/pushes" in packet
    assert "committed cockpit files" in packet
    assert "commented on cockpit issue #100" in packet
    assert "GitHub sync skipped" not in packet
    assert "GitHub mutation disabled" not in packet
    comment_index = next(
        index for index, command in enumerate(commands) if command[:3] == ["gh", "issue", "comment"]
    )
    final_commit_index = next(
        index
        for index, command in enumerate(commands)
        if command[:3] == ["git", "commit", "-m"]
        and command[3] == "Record cockpit GitHub sync for 2026-06-20"
    )
    final_push_indices = [
        index
        for index, command in enumerate(commands)
        if command == ["git", "push", "-u", "origin", "codex/cockpit-refresh-20260620"]
    ]
    assert final_commit_index > comment_index
    assert final_push_indices[-1] > final_commit_index


def test_run_local_cockpit_refresh_admits_registry_only_workstream(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={
            "Harness rollout": {
                "status": "ready-for-agent",
                "canonical_surface": "origin/main",
                "reason": "Registry declares a workstream that exists nowhere in code.",
                "next_action": "Continue the harness rollout.",
                "last_reviewed": "2026-07-09",
            }
        },
        surfaces={},
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner([]),
    )

    register = result.register_path.read_text(encoding="utf-8")
    assert "Harness rollout" in register
    assert "| Harness rollout | ready-for-agent |" in register


def test_run_local_cockpit_refresh_admits_surface_for_registry_only_workstream(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={
            "Harness rollout": {
                "status": "ready-for-agent",
                "canonical_surface": "codex/harness-rollout",
                "reason": "Registry declares a workstream that exists nowhere in code.",
                "next_action": "Continue the harness rollout.",
                "last_reviewed": "2026-07-09",
            }
        },
        surfaces={
            "codex/harness-rollout": {
                "workstreams": ["Harness rollout"],
                "disposition": "canonical",
                "reason": "Reviewed canonical branch for the registry-only workstream.",
                "next_action": "Continue through the harness rollout branch.",
                "last_reviewed": "2026-07-09",
            }
        },
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner(["codex/harness-rollout"]),
    )

    register = result.register_path.read_text(encoding="utf-8")
    assert "Harness rollout" in register


def test_run_local_cockpit_refresh_rejects_unknown_workstream_reference(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={},
        surfaces={
            "codex/mystery": {
                "workstreams": ["Totally unknown workstream"],
                "disposition": "canonical",
                "reason": "References a name in neither seeds nor registry keys.",
                "next_action": "Fix the registry.",
                "last_reviewed": "2026-07-09",
            }
        },
    )

    with pytest.raises(ValueError, match="Totally unknown workstream"):
        run_local_cockpit_refresh(
            repo_root=repo,
            run_date=date(2026, 7, 10),
            run_command=_fake_topology_runner(["codex/mystery"]),
        )


def test_merge_workstream_sources_dedupes_with_earlier_source_winning() -> None:
    first = _FixedSource(
        [
            WorkstreamSeed(
                name="Shared",
                status=WorkstreamStatus.ACTIVE,
                next_action="First source wins.",
                tracker="first-tracker",
                branch_terms=("first",),
            )
        ]
    )
    second = _FixedSource(
        [
            WorkstreamSeed(
                name="Shared",
                status=WorkstreamStatus.PARKED,
                next_action="Second source loses.",
            ),
            WorkstreamSeed(
                name="Second only",
                status=WorkstreamStatus.ACTIVE,
                next_action="Unique to the second source.",
            ),
        ]
    )

    merged = merge_workstream_sources((first, second), evidence=None, run_date=date(2026, 7, 11))

    assert [seed.name for seed in merged] == ["Shared", "Second only"]
    shared = next(seed for seed in merged if seed.name == "Shared")
    assert shared.status == WorkstreamStatus.ACTIVE
    assert shared.next_action == "First source wins."
    assert shared.tracker == "first-tracker"
    assert shared.branch_terms == ("first",)


def test_merge_workstream_sources_is_deterministic_across_calls() -> None:
    sources = (
        StaticWorkstreamSource(),
        _FixedSource(
            [
                WorkstreamSeed(name="Zeta", status=WorkstreamStatus.ACTIVE, next_action="z"),
                WorkstreamSeed(name="Alpha", status=WorkstreamStatus.ACTIVE, next_action="a"),
            ]
        ),
    )

    first = merge_workstream_sources(sources, evidence=None, run_date=date(2026, 7, 11))
    second = merge_workstream_sources(sources, evidence=None, run_date=date(2026, 7, 11))

    assert first == second
    assert [seed.name for seed in first] == [seed.name for seed in second]


def test_merge_workstream_sources_drops_reserved_surface_prefix() -> None:
    source = _FixedSource(
        [
            WorkstreamSeed(
                name="Git surface: codex/should-drop",
                status=WorkstreamStatus.ACTIVE,
                next_action="Reserved prefix; must be dropped.",
            ),
            WorkstreamSeed(
                name="Kept workstream",
                status=WorkstreamStatus.ACTIVE,
                next_action="Survives the merge.",
            ),
        ]
    )

    merged = merge_workstream_sources((source,), evidence=None, run_date=date(2026, 7, 11))

    assert [seed.name for seed in merged] == ["Kept workstream"]


def test_run_local_cockpit_refresh_drops_reserved_prefix_seed_without_crashing(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    reserved_source = _FixedSource(
        [
            WorkstreamSeed(
                name="Git surface: codex/should-drop",
                status=WorkstreamStatus.ACTIVE,
                next_action="A buggy source must not crash the daily refresh.",
            )
        ]
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner([]),
        sources=(reserved_source,),
    )

    register = result.register_path.read_text(encoding="utf-8")
    assert result.register_path.exists()
    assert "Git surface: codex/should-drop" not in register
    assert "Git and worktree hygiene" in register


def test_run_local_cockpit_refresh_drops_source_hygiene_but_keeps_static_row(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    hygiene_source = _FixedSource(
        [
            WorkstreamSeed(
                name="Git and worktree hygiene",
                status=WorkstreamStatus.PARKED,
                next_action="Bogus hygiene emitted by a source; must be dropped.",
            )
        ]
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner([]),
        sources=(hygiene_source,),
    )

    register = result.register_path.read_text(encoding="utf-8")
    hygiene_rows = [
        line for line in register.splitlines() if line.startswith("| Git and worktree hygiene |")
    ]
    assert len(hygiene_rows) == 1
    assert "Bogus hygiene emitted by a source; must be dropped." not in register


def test_registry_workstream_without_surface_renders_under_live_topology(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={
            "Harness rollout": {
                "status": "parked",
                "canonical_surface": "origin/main",
                "reason": "Registry declares a workstream with no matching live surface.",
                "next_action": "Continue the harness rollout when reprioritized.",
                "last_reviewed": "2026-07-09",
            }
        },
        surfaces={},
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner(["codex/unrelated-live-branch"]),
    )

    register = result.register_path.read_text(encoding="utf-8")
    assert "Git surface: codex/unrelated-live-branch" in register
    assert "| Harness rollout | parked |" in register


_GIT_ACTIVITY_RUN_DATE = date(2026, 7, 11)


def _git_activity_evidence(
    *,
    recent_branches: list[tuple[str, date]] | None = None,
    recent_handoffs: list[str] | None = None,
) -> SimpleNamespace:
    """Duck-typed evidence exposing only what GitActivitySource reads."""
    return SimpleNamespace(
        recent_branches=recent_branches or [],
        recent_handoffs=recent_handoffs or [],
        repo_root=None,
    )


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        ("codex/top10-lambdarank-screen-20260625", ["top10", "lambdarank", "screen"]),
        ("codex/portfolio-ic-hybrid-testing", ["portfolio", "ic", "hybrid", "testing"]),
        (
            "cursor/intermediate-speed-dynamic-momentum-8937",
            ["intermediate", "speed", "dynamic", "momentum"],
        ),
        ("codex/regime-csv-no-backfill-2026-07-09", ["regime", "csv", "no", "backfill"]),
        ("codex/cockpit-refresh-20260710", ["cockpit", "refresh"]),
    ],
)
def test_branch_topic_tokens_strips_prefix_date_and_hash(branch: str, expected: list[str]) -> None:
    assert _branch_topic_tokens(branch) == expected


def test_git_activity_source_emits_active_titlecased_seed() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[("codex/paper-trade-frozen-graph-20260709", date(2026, 7, 9))]
    )

    seeds = GitActivitySource(aliases={}).provide(evidence, _GIT_ACTIVITY_RUN_DATE)

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.name == "Paper Trade Frozen Graph"
    assert seed.status == WorkstreamStatus.ACTIVE
    assert seed.tracker == ""
    assert seed.branch_terms == ("frozen", "graph", "paper", "trade")


def test_git_activity_source_skips_stopword_only_branch() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[("codex/cockpit-refresh-20260710", date(2026, 7, 10))]
    )

    assert GitActivitySource(aliases={}).provide(evidence, _GIT_ACTIVITY_RUN_DATE) == []


def test_git_activity_source_drops_stopword_token_but_keeps_topic() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[("codex/salvage-pr50-research-docs-20260707", date(2026, 7, 10))]
    )

    seeds = GitActivitySource(aliases={}).provide(evidence, _GIT_ACTIVITY_RUN_DATE)

    assert [seed.name for seed in seeds] == ["Pr50 Research Docs"]
    assert "salvage" not in seeds[0].branch_terms


def test_git_activity_source_resolves_alias_to_canonical_name() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[("codex/top10-lambdarank-screen-20260625", date(2026, 7, 10))]
    )

    seeds = GitActivitySource(aliases={"lambdarank": "LambdaRankIC"}).provide(
        evidence, _GIT_ACTIVITY_RUN_DATE
    )

    assert [seed.name for seed in seeds] == ["LambdaRankIC"]
    assert seeds[0].branch_terms == ("lambdarank", "screen", "top10")


def test_git_activity_source_applies_committer_date_lookback() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[
            ("codex/recent-topic", _GIT_ACTIVITY_RUN_DATE - timedelta(days=5)),
            ("codex/stale-topic", _GIT_ACTIVITY_RUN_DATE - timedelta(days=20)),
        ]
    )

    seeds = GitActivitySource(aliases={}).provide(evidence, _GIT_ACTIVITY_RUN_DATE)

    assert [seed.name for seed in seeds] == ["Recent Topic"]


def test_git_activity_source_collapses_and_is_deterministic() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[
            ("codex/lambdarank-recovery-20260709", date(2026, 7, 9)),
            ("codex/top10-screen-20260710", date(2026, 7, 10)),
            ("codex/zephyr-experiment", date(2026, 7, 11)),
        ]
    )
    source = GitActivitySource(aliases={"lambdarank": "LambdaRankIC", "top10": "LambdaRankIC"})

    first = source.provide(evidence, _GIT_ACTIVITY_RUN_DATE)
    second = source.provide(evidence, _GIT_ACTIVITY_RUN_DATE)

    assert first == second
    assert [seed.name for seed in first] == ["LambdaRankIC", "Zephyr Experiment"]
    collapsed = first[0]
    assert collapsed.branch_terms == ("lambdarank", "recovery", "screen", "top10")


def test_git_activity_source_derives_seed_from_recent_handoff() -> None:
    evidence = _git_activity_evidence(
        recent_handoffs=["docs/handoffs/2026-07-09-portfolio-rebuild-audit.md"]
    )

    seeds = GitActivitySource(aliases={}).provide(evidence, _GIT_ACTIVITY_RUN_DATE)

    assert [seed.name for seed in seeds] == ["Portfolio Rebuild Audit"]


def test_run_local_cockpit_refresh_adds_git_derived_workstream(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return "2026-07-10 codex/paper-trade-frozen-graph-20260709\n"
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc1234\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "abc1234 Current main\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main...origin/main\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 11),
        run_command=fake_run,
    )

    register = result.register_path.read_text(encoding="utf-8")
    assert "Paper Trade Frozen Graph" in register


def _repo_with_required_docs(repo: Path) -> Path:
    (repo / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    (repo / "docs" / "agents").mkdir(parents=True)
    (repo / "docs" / "agents" / "domain.md").write_text("# Domain\n", encoding="utf-8")
    (repo / "docs" / "agents" / "issue-tracker.md").write_text("# Issues\n", encoding="utf-8")
    (repo / "docs" / "agents" / "triage-labels.md").write_text("# Labels\n", encoding="utf-8")
    (repo / "docs" / "research").mkdir(parents=True)
    (repo / "docs" / "research" / "README.md").write_text("# Research\n", encoding="utf-8")
    registry_path = repo / DECISION_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(_empty_registry(), encoding="utf-8")
    (repo / "docs" / "index.md").write_text("# Index\n", encoding="utf-8")
    return repo


def _write_decision_registry(
    repo: Path,
    *,
    workstreams: dict[str, object],
    surfaces: dict[str, object],
) -> None:
    (repo / DECISION_REGISTRY_PATH).write_text(
        json.dumps(
            {
                "format_version": 1,
                "workstreams": workstreams,
                "surfaces": surfaces,
            }
        ),
        encoding="utf-8",
    )


def _empty_registry() -> str:
    return json.dumps({"format_version": 1, "workstreams": {}, "surfaces": {}})


def _fake_topology_runner(branches: list[str]):
    branch_lines = "".join(f"  {branch}\n" for branch in branches)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return "main\n" + "\n".join(branches) + "\n"
        if command == "git branch --all --no-merged origin/main":
            return branch_lines
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc1234\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "abc1234 Current main\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main...origin/main\n"
        raise AssertionError(command)

    return fake_run


def _strip_safe_directory_args(args: list[str]) -> list[str]:
    if (
        len(args) >= 3
        and args[0] == "git"
        and args[1] == "-c"
        and args[2].startswith("safe.directory=")
    ):
        return ["git", *args[3:]]
    return args


def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
