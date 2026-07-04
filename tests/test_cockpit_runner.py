from __future__ import annotations

import subprocess
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING

from cockpit.evidence import collect_local_evidence
from cockpit.runner import (
    _run_command,
    run_github_cockpit_refresh,
    run_local_cockpit_refresh,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_collect_local_evidence_records_dirty_paths_and_required_docs(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    (repo / "docs" / "agents").mkdir(parents=True)
    (repo / "docs" / "agents" / "domain.md").write_text("# Domain\n", encoding="utf-8")
    (repo / "docs" / "research").mkdir(parents=True)
    (repo / "docs" / "research" / "README.md").write_text("# Research\n", encoding="utf-8")
    (repo / "docs" / "handoffs").mkdir(parents=True)
    (repo / "docs" / "handoffs" / "2026-06-01-note.md").write_text(
        "# Handoff\n",
        encoding="utf-8",
    )

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
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
    assert evidence.recent_handoffs == ["docs/handoffs/2026-06-01-note.md"]
    assert evidence.dirty_paths == ["docs/agents/domain.md", "scratch.txt"]
    assert evidence.branches == ["codex/example", "main"]
    assert "abc123 Add cockpit design" in evidence.recent_commits


def test_collect_local_evidence_builds_git_topology_snapshot(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
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


def test_run_github_cockpit_refresh_switches_branch_and_syncs(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
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


def _repo_with_required_docs(repo: Path) -> Path:
    (repo / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    (repo / "docs" / "agents").mkdir(parents=True)
    (repo / "docs" / "agents" / "domain.md").write_text("# Domain\n", encoding="utf-8")
    (repo / "docs" / "agents" / "issue-tracker.md").write_text("# Issues\n", encoding="utf-8")
    (repo / "docs" / "agents" / "triage-labels.md").write_text("# Labels\n", encoding="utf-8")
    (repo / "docs" / "research").mkdir(parents=True)
    (repo / "docs" / "research" / "README.md").write_text("# Research\n", encoding="utf-8")
    (repo / "docs" / "index.md").write_text("# Index\n", encoding="utf-8")
    return repo


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
