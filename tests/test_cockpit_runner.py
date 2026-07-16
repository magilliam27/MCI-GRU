from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import cockpit.runner as cockpit_runner_module
from cockpit.decisions import DECISION_REGISTRY_PATH, SurfaceDisposition
from cockpit.evidence import LocalEvidence, collect_local_evidence
from cockpit.models import (
    AutoDecisionChange,
    AutoDecisionSet,
    AutoDisposition,
    Confidence,
    GitHubEvidence,
    GitTopologySnapshot,
    PullRequestEvidence,
    WorkstreamStatus,
    WorktreeEvidence,
)
from cockpit.policy import AUTO_DECISIONS_PATH, write_auto_decisions
from cockpit.runner import (
    GitActivitySource,
    StaticWorkstreamSource,
    WorkstreamSeed,
    _branch_topic_tokens,
    _run_command,
    _switch_to_cockpit_branch,
    _without_current_worktree_dirty_paths,
    implied_aliases,
    merge_workstream_sources,
    run_github_cockpit_refresh,
    run_local_cockpit_refresh,
)


def test_ignored_owned_paths_update_control_plane_checkout_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "control-plane"
    control_plane = WorktreeEvidence(
        path=str(repo.resolve()),
        head="abc123",
        branch="codex/cockpit-refresh",
        detached=False,
        status_header="## codex/cockpit-refresh",
        dirty_paths=["docs/agents/workstreams.md", "src/user_change.py"],
    )
    canonical = WorktreeEvidence(
        path="C:/canonical",
        head="def456",
        branch="main",
        detached=False,
        status_header="## main",
        dirty_paths=["src/canonical_change.py"],
    )
    topology = GitTopologySnapshot(
        current_branch=control_plane.branch,
        status_header=control_plane.status_header,
        origin_main_ahead=0,
        origin_main_behind=0,
        worktrees=[canonical, control_plane],
        control_plane_worktree=control_plane,
        primary_worktree=canonical,
    )
    evidence = LocalEvidence(
        repo_root=repo,
        required_docs={},
        recent_handoffs=[],
        dirty_paths=list(control_plane.dirty_paths),
        branches=["main", control_plane.branch],
        worktrees="",
        recent_commits="",
        git_topology=topology,
    )

    filtered = _without_current_worktree_dirty_paths(
        evidence,
        repo,
        ["docs/agents/workstreams.md"],
    )

    assert filtered.dirty_paths == ["src/user_change.py"]
    assert filtered.git_topology.worktrees[1].dirty_paths == ["src/user_change.py"]
    assert filtered.git_topology.control_plane_worktree is not None
    assert filtered.git_topology.control_plane_worktree.dirty_paths == ["src/user_change.py"]
    assert filtered.git_topology.primary_worktree == canonical


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


def test_collect_local_evidence_normalizes_branch_dates_with_local_precedence(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    ref_commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            ref_commands.append(args)
            return (
                "2026-07-01\trefs/remotes/origin/codex/shared\n"
                "2026-07-03\trefs/heads/codex/shared\n"
                "2026-06-20\trefs/heads/codex/shared\n"
                "2026-06-01\trefs/remotes/origin/codex/remote-only\n"
                "2026-07-04\trefs/remotes/origin/HEAD\n"
                "bad-date\trefs/heads/codex/bad\n"
            )
        if command == "git status --short --branch":
            return "## main\n"
        if command == "git branch --format=%(refname:short)":
            return "main\ncodex/shared\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return ""
        if command == "git log -5 --oneline":
            return ""
        raise AssertionError(command)

    evidence = collect_local_evidence(repo, run_command=fake_run)

    assert ref_commands == [
        [
            "git",
            "for-each-ref",
            "--sort=-committerdate",
            "--format=%(committerdate:short)%09%(refname)",
            "refs/heads",
            "refs/remotes/origin",
        ]
    ]
    assert evidence.branch_commit_dates == {
        "codex/remote-only": date(2026, 6, 1),
        "codex/shared": date(2026, 7, 3),
    }
    assert evidence.recent_branches == [("codex/shared", date(2026, 7, 3))]


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
        if args[:2] == ["git", "-c"] and args[-4:] == [
            "rev-list",
            "--left-right",
            "--count",
            "origin/main...HEAD",
        ]:
            return "2\t8\n" if args[4] == "C:/repo" else "0\t0\n"
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
    dirty_worktree_branches = [
        worktree.branch for worktree in evidence.git_topology.dirty_worktrees
    ]
    assert dirty_worktree_branches[0] == "codex/top10-lambdarank-screen-20260625"
    assert dirty_worktree_branches[1].startswith("detached@a2684d2-")
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
        if command == (
            f"git -C {repo.as_posix()} rev-list --left-right --count origin/main...HEAD"
        ):
            return SimpleNamespace(stdout="0\t0\n")
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
        auto_decisions_enabled=False,
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


def test_run_local_cockpit_refresh_flag_off_preserves_output_parity(tmp_path: Path) -> None:
    (tmp_path / "default").mkdir()
    (tmp_path / "explicit").mkdir()
    default_repo = _repo_with_required_docs(tmp_path / "default")
    explicit_repo = _repo_with_required_docs(tmp_path / "explicit")
    run_date = date(2026, 7, 12)

    default = run_local_cockpit_refresh(
        default_repo,
        run_date,
        run_command=_fake_topology_runner(["codex/lambdarank-live"]),
        auto_decisions_enabled=False,
    )
    explicit = run_local_cockpit_refresh(
        explicit_repo,
        run_date,
        run_command=_fake_topology_runner(["codex/lambdarank-live"]),
        auto_decisions_enabled=False,
    )

    assert default.register_path.read_bytes() == explicit.register_path.read_bytes()
    assert default.packet_path.read_bytes() == explicit.packet_path.read_bytes()
    # Text reads normalize platform newlines before checking the frozen content snapshots.
    assert (
        hashlib.sha256(default.register_path.read_text(encoding="utf-8").encode()).hexdigest()
        == "f3525d8ddc817444cd23874d924aa06fb29f744146d421a0ef513f0bc5eead66"
    )
    assert (
        hashlib.sha256(default.packet_path.read_text(encoding="utf-8").encode()).hexdigest()
        == "a005ad3baa795a4fc8b62700fd6a26bcbc3945bcfa4e9474dfba81d9ac41b1c1"
    )
    assert not (default_repo / AUTO_DECISIONS_PATH).exists()
    assert not (explicit_repo / AUTO_DECISIONS_PATH).exists()


def test_flag_off_preserves_registry_plus_heuristic_surface_association(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-bug-scan"
    _write_decision_registry(
        repo,
        workstreams={},
        surfaces={
            branch: {
                "workstreams": ["Daily bug scans"],
                "disposition": "canonical",
                "reason": "Explicit mapping coexists with legacy heuristic matching.",
                "next_action": "Continue the bug scan.",
                "last_reviewed": "2026-07-11",
            }
        },
    )

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_fake_topology_runner([branch]),
        auto_decisions_enabled=False,
    )

    rows = [
        tuple(cell.strip() for cell in line.split("|")[1:-1])
        for line in result.register_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("| ")
        and not line.startswith("| Workstream ")
        and not line.startswith("| --- ")
    ]
    assert [(row[0], row[1]) for row in rows] == [
        ("LambdaRankIC", "active"),
        ("Daily bug scans", "ready-for-agent"),
        ("Git and worktree hygiene", "active"),
    ]
    assert rows[0][3] == f"`{branch}` (local)"
    assert rows[1][3] == f"`{branch}` (local)"
    assert not (repo / AUTO_DECISIONS_PATH).exists()


def test_run_local_cockpit_refresh_applies_generated_decisions_by_default(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/merged-lambdarank"

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return f"2026-07-08 {branch}\n"
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{branch}\n"
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
        repo,
        date(2026, 7, 12),
        run_command=fake_run,
        github_evidence=GitHubEvidence(),
    )

    register = result.register_path.read_text(encoding="utf-8")
    packet = result.packet_path.read_text(encoding="utf-8")
    assert (repo / AUTO_DECISIONS_PATH).exists()
    assert f"| Git surface: {branch} | archive |" in register
    assert f"| Git surface: {branch} | needs-user-decision |" not in register
    assert result.report.auto_dispositions[branch].rule == "merged-into-main"
    assert "## Auto-Dispositions Applied" in packet
    assert f"**{branch}** → archive" in packet


def test_run_local_cockpit_refresh_compares_before_overwriting_prior_auto_file(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-live"
    write_auto_decisions(
        repo,
        AutoDecisionSet(
            surfaces={
                branch: AutoDisposition(
                    workstreams=("LambdaRankIC",),
                    disposition=SurfaceDisposition.PARKED,
                    rule="prior-rule",
                    evidence="Prior generated result.",
                    confidence=Confidence.HIGH,
                    alternatives=(),
                    last_reviewed=date(2026, 7, 12),
                )
            }
        ),
    )
    committed_payload = (repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8")
    topology_runner = _fake_topology_runner([branch])

    def committed_baseline_runner(args: list[str]) -> str:
        if args == ["git", "show", f"HEAD:{AUTO_DECISIONS_PATH}"]:
            return committed_payload
        return topology_runner(args)

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 13),
        run_command=committed_baseline_runner,
        github_evidence=GitHubEvidence(),
    )

    change = AutoDecisionChange(
        kind="surface",
        target=branch,
        change="choice",
        before="parked",
        after="canonical",
    )
    assert change in result.report.decision_changes
    first_packet = result.packet_path.read_bytes()
    assert "choice changed parked → canonical" in first_packet.decode("utf-8")

    unchanged = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 13),
        run_command=committed_baseline_runner,
        github_evidence=GitHubEvidence(),
    )

    assert change in unchanged.report.decision_changes
    assert unchanged.packet_path.read_bytes() == first_packet


def test_run_local_cockpit_refresh_reports_new_override_against_committed_registry(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-live"
    topology_runner = _fake_topology_runner([branch])
    run_date = date(2026, 7, 13)

    run_local_cockpit_refresh(
        repo,
        run_date,
        run_command=topology_runner,
        github_evidence=GitHubEvidence(),
    )
    committed_auto = (repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8")
    committed_registry = (repo / DECISION_REGISTRY_PATH).read_text(encoding="utf-8")
    generated_status = json.loads(committed_auto)["workstreams"]["LambdaRankIC"]["status"]
    assert generated_status == "active"

    _write_decision_registry(
        repo,
        workstreams={
            "LambdaRankIC": {
                "status": "parked",
                "canonical_surface": branch,
                "reason": "Explicit PR comment correction.",
                "next_action": "Pause.",
                "last_reviewed": run_date.isoformat(),
            }
        },
        surfaces={},
    )

    def committed_baseline_runner(args: list[str]) -> str:
        if args == ["git", "show", f"HEAD:{AUTO_DECISIONS_PATH}"]:
            return committed_auto
        if args == ["git", "show", f"HEAD:{DECISION_REGISTRY_PATH}"]:
            return committed_registry
        return topology_runner(args)

    result = run_local_cockpit_refresh(
        repo,
        run_date,
        run_command=committed_baseline_runner,
        github_evidence=GitHubEvidence(),
    )

    assert (
        AutoDecisionChange(
            kind="workstream",
            target="LambdaRankIC",
            change="override-added",
            before="active",
            after="parked",
        )
        in result.report.decision_changes
    )
    assert "override added; generated active → explicit parked" in result.packet_path.read_text(
        encoding="utf-8"
    )


def test_run_local_cockpit_refresh_reports_cleared_historical_overrides_absent_now(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/retired-workstream"
    committed_registry = json.dumps(
        {
            "format_version": 1,
            "workstreams": {
                "Retired workstream": {
                    "status": "parked",
                    "canonical_surface": branch,
                    "reason": "The historical override parked this workstream.",
                    "next_action": "Keep it parked.",
                    "last_reviewed": "2026-07-12",
                }
            },
            "surfaces": {
                branch: {
                    "workstreams": ["Retired workstream"],
                    "disposition": "archive",
                    "reason": "The historical override archived this branch.",
                    "next_action": "Retain the archived evidence.",
                    "last_reviewed": "2026-07-12",
                }
            },
        }
    )
    topology_runner = _fake_topology_runner([])

    def committed_baseline_runner(args: list[str]) -> str:
        if args == ["git", "show", f"HEAD:{DECISION_REGISTRY_PATH}"]:
            return committed_registry
        return topology_runner(args)

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 13),
        run_command=committed_baseline_runner,
        sources=(),
        github_evidence=GitHubEvidence(),
    )

    assert (
        AutoDecisionChange(
            kind="workstream",
            target="Retired workstream",
            change="override-cleared",
            before="parked",
            after="none",
        )
        in result.report.decision_changes
    )
    assert (
        AutoDecisionChange(
            kind="surface",
            target=branch,
            change="override-cleared",
            before="archive",
            after="none",
        )
        in result.report.decision_changes
    )


def test_run_local_cockpit_refresh_ignores_malformed_historical_registry(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    committed_registry = json.dumps(
        {
            "format_version": 1,
            "workstreams": {
                "Retired workstream": {
                    "status": "not-a-valid-status",
                    "canonical_surface": "codex/retired-workstream",
                    "reason": "Malformed committed evidence.",
                    "next_action": "Do not trust this snapshot.",
                    "last_reviewed": "2026-07-12",
                }
            },
            "surfaces": {},
        }
    )
    topology_runner = _fake_topology_runner([])

    def committed_baseline_runner(args: list[str]) -> str:
        if args == ["git", "show", f"HEAD:{DECISION_REGISTRY_PATH}"]:
            return committed_registry
        return topology_runner(args)

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 13),
        run_command=committed_baseline_runner,
        sources=(),
        github_evidence=GitHubEvidence(),
    )

    assert all(change.target != "Retired workstream" for change in result.report.decision_changes)


def test_auto_decisions_disabled_never_calls_github_collector(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    calls = 0

    def forbidden_collector() -> GitHubEvidence | None:
        nonlocal calls
        calls += 1
        raise AssertionError("GitHub collector must stay disabled")

    run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_fake_topology_runner(["codex/lambdarank-live"]),
        auto_decisions_enabled=False,
        github_evidence_collector=forbidden_collector,
    )

    assert calls == 0


def test_auto_decisions_enabled_offline_records_degraded_github_gap(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-old"

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_dated_topology_runner(branch, date(2026, 5, 1)),
        auto_decisions_enabled=True,
        github_evidence_collector=lambda: None,
    )

    surface = result.report.auto_dispositions[branch]
    packet = result.packet_path.read_text(encoding="utf-8")
    assert surface.disposition == "stale"
    assert surface.confidence == "medium"
    assert any("GitHub" in gap and "unavailable" in gap for gap in result.report.evidence_gaps)
    assert "GitHub PR and issue evidence unavailable" in packet
    assert "open-pr-canonical" in packet
    assert "stale" in packet


def test_auto_decisions_enabled_survives_collector_oserror(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-old"

    def unavailable_collector() -> GitHubEvidence | None:
        raise OSError("gh unavailable")

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_dated_topology_runner(branch, date(2026, 5, 1)),
        auto_decisions_enabled=True,
        github_evidence_collector=unavailable_collector,
    )

    surface = result.report.auto_dispositions[branch]
    assert surface.disposition == "stale"
    assert surface.confidence == "medium"
    assert any(
        "GitHub PR and issue evidence unavailable" in gap for gap in result.report.evidence_gaps
    )


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["gh"], 30),
        RuntimeError("collector runtime failure"),
    ],
)
def test_auto_decisions_enabled_survives_any_ordinary_collector_exception(
    tmp_path: Path,
    error: Exception,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-old"

    def unavailable_collector() -> GitHubEvidence | None:
        raise error

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_dated_topology_runner(branch, date(2026, 5, 1)),
        auto_decisions_enabled=True,
        github_evidence_collector=unavailable_collector,
    )

    assert result.report.auto_dispositions[branch].confidence == "medium"
    assert any(
        "GitHub PR and issue evidence unavailable" in gap for gap in result.report.evidence_gaps
    )


def test_omitted_github_evidence_calls_injected_collector(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-old"
    calls = 0

    def confirmed_empty_collector() -> GitHubEvidence | None:
        nonlocal calls
        calls += 1
        return GitHubEvidence()

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_dated_topology_runner(branch, date(2026, 5, 1)),
        auto_decisions_enabled=True,
        github_evidence_collector=confirmed_empty_collector,
    )

    assert calls == 1
    assert result.report.auto_dispositions[branch].confidence == "high"
    assert not any(
        "GitHub PR and issue evidence unavailable" in gap for gap in result.report.evidence_gaps
    )
    assert "No live GitHub issue or PR scan in local-only mode." not in result.report.evidence_gaps


def test_explicit_none_github_evidence_skips_collector_and_stays_offline(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-old"
    calls = 0

    def forbidden_collector() -> GitHubEvidence | None:
        nonlocal calls
        calls += 1
        raise AssertionError("explicit None must prevent collection")

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_dated_topology_runner(branch, date(2026, 5, 1)),
        auto_decisions_enabled=True,
        github_evidence=None,
        github_evidence_collector=forbidden_collector,
    )

    assert calls == 0
    assert result.report.auto_dispositions[branch].confidence == "medium"
    assert any(
        "GitHub PR and issue evidence unavailable" in gap for gap in result.report.evidence_gaps
    )


def test_injected_online_github_evidence_changes_generated_output(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/lambdarank-old"
    github = GitHubEvidence(
        pull_requests=(
            PullRequestEvidence(
                number=81,
                head_ref=branch,
                url="https://github.example/pull/81",
                state="open",
                is_draft=False,
                merged_at=None,
                updated_at=date(2026, 7, 11),
            ),
        )
    )

    result = run_local_cockpit_refresh(
        repo,
        date(2026, 7, 12),
        run_command=_dated_topology_runner(branch, date(2026, 5, 1)),
        auto_decisions_enabled=True,
        github_evidence=github,
    )

    assert result.report.auto_dispositions[branch].rule == "open-pr-canonical"
    assert result.report.auto_workstream_decisions["LambdaRankIC"].status == "active"
    assert "PR #81" in result.packet_path.read_text(encoding="utf-8")
    assert not any(
        "GitHub PR and issue evidence unavailable" in gap for gap in result.report.evidence_gaps
    )


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
        if args[:2] == ["git", "-c"] and args[-4:] == [
            "rev-list",
            "--left-right",
            "--count",
            "origin/main...HEAD",
        ]:
            return "2\t8\n" if args[4] == "C:/repo" else "0\t1\n"
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
        auto_decisions_enabled=False,
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
    assert (
        "Control-plane checkout: `C:/repo` on "
        "`codex/top10-lambdarank-screen-20260625`; origin/main divergence: "
        "8 ahead / 2 behind; dirty: no"
    ) in packet
    assert (
        "Canonical active checkout (primary worktree): `C:/repo` on "
        "`codex/top10-lambdarank-screen-20260625`; origin/main divergence: "
        "8 ahead / 2 behind; dirty: no"
    ) in packet
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


def test_cockpit_packet_distinguishes_control_plane_and_canonical_divergence(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/cockpit-refresh-20260715"
    canonical_path = "C:/Users/magil/MCI-GRU"
    control_path = repo.as_posix()

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short --branch":
            return f"## {branch}...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{branch}\n"
        if command == "git branch --all --no-merged origin/main":
            return f"  {branch}\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return (
                f"worktree {canonical_path}\n"
                "HEAD d6b0f60\n"
                "branch refs/heads/codex/canonical-feature\n"
                "\n"
                f"worktree {control_path}\n"
                "HEAD 074474d\n"
                f"branch refs/heads/{branch}\n"
            )
        if command == "git log -5 --oneline":
            return "074474d Cockpit refresh\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            if args[-4:] == [
                "rev-list",
                "--left-right",
                "--count",
                "origin/main...HEAD",
            ]:
                return "5\t1\n" if args[4] == canonical_path else "0\t0\n"
            if args[4] == canonical_path:
                return "## codex/canonical-feature...origin/codex/canonical-feature\n"
            if args[4] == control_path:
                return f"## {branch}...origin/main\n"
        raise AssertionError(command)

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 15),
        run_command=fake_run,
        automation_branch=branch,
        comparison_ref="origin/main",
        github_evidence=None,
    )

    packet = result.packet_path.read_text(encoding="utf-8")
    assert result.color.value == "yellow"
    assert (
        f"Control-plane checkout: `{control_path}` on `{branch}`; "
        "origin/main divergence: 0 ahead / 0 behind; dirty: no"
    ) in packet
    assert (
        f"Canonical active checkout (primary worktree): `{canonical_path}` on "
        "`codex/canonical-feature`; "
        "origin/main divergence: 1 ahead / 5 behind; dirty: no"
    ) in packet


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
    assert by_name["LambdaRankIC"].continuation == "codex/canonical-lambdarank"
    assert by_name["LambdaRankIC"].last_reviewed == date(2026, 7, 9)
    assert "workstream-decisions.json" in by_name["LambdaRankIC"].source_of_truth
    assert "| Daily bug scans | ready-for-agent |  | codex/backtest-plot-test |" in register
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
        auto_decisions_enabled=False,
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
    auto_payload = json.loads((repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8"))
    detached_id = next(
        surface_id
        for surface_id in auto_payload["surfaces"]
        if surface_id.startswith("detached@a2684d2-")
    )
    assert result.color.value == "yellow"
    assert "Git surface: HEAD (no branch)" not in register
    assert "`HEAD (no branch)` (local)" not in register
    assert "Git surface: (HEAD detached at a2684d2)" not in register
    assert "`(HEAD detached at a2684d2)` (local)" not in register
    assert f"Git surface: {detached_id}" in register
    assert f"`{detached_id}` @ `C:/repo` (detached)" in register
    assert f"Control-plane checkout: `C:/repo` on `{detached_id}`" in packet
    assert f"Detached worktrees: `{detached_id}` at `C:/repo`" in packet


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
    auto_payload = json.loads((repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8"))
    current_id = next(
        surface_id
        for surface_id in auto_payload["surfaces"]
        if surface_id.startswith("detached@2222222-")
    )
    other_id = next(
        surface_id
        for surface_id in auto_payload["surfaces"]
        if surface_id.startswith("detached@1111111-")
    )
    control_line = next(line for line in packet.splitlines() if "Control-plane checkout" in line)
    assert f"on `{current_id}`" in control_line
    assert f"Git surface: {current_id}" in register
    assert f"on `{other_id}`" not in control_line
    assert f"Git surface: {other_id}" in register


def test_detached_worktrees_have_stable_path_scoped_surfaces_and_decisions(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    worktrees = ["C:/detached/alpha", "C:/detached/beta", "C:/detached/gamma"]

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
            return (
                "worktree C:/detached/alpha\n"
                f"HEAD {'a' * 40}\n"
                "detached\n\n"
                "worktree C:/detached/beta\n"
                f"HEAD {'a' * 40}\n"
                "detached\n\n"
                "worktree C:/detached/gamma\n"
                f"HEAD {'b' * 40}\n"
                "detached\n"
            )
        if command == "git log -5 --oneline":
            return ""
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## HEAD (no branch)\n"
        raise AssertionError(command)

    first = run_local_cockpit_refresh(repo, date(2026, 7, 13), run_command=fake_run)
    first_register = first.register_path.read_bytes()
    first_auto = (repo / AUTO_DECISIONS_PATH).read_bytes()
    payload = json.loads(first_auto)
    detached_ids = sorted(
        surface_id for surface_id in payload["surfaces"] if surface_id.startswith("detached@")
    )

    assert len(detached_ids) == 3
    assert len(set(detached_ids)) == 3
    assert sum(surface_id.startswith("detached@aaaaaaa-") for surface_id in detached_ids) == 2
    assert sum(surface_id.startswith("detached@bbbbbbb-") for surface_id in detached_ids) == 1
    register = first_register.decode("utf-8")
    for path in worktrees:
        assert path in register
    for surface_id in detached_ids:
        assert f"Git surface: {surface_id}" in register

    second = run_local_cockpit_refresh(repo, date(2026, 7, 13), run_command=fake_run)
    assert second.register_path.read_bytes() == first_register
    assert (repo / AUTO_DECISIONS_PATH).read_bytes() == first_auto


def test_attached_same_branch_worktrees_have_independent_stable_surfaces(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    worktree_paths = {
        "shared_alpha": "C:/attached/shared-alpha",
        "shared_beta": "C:/attached/shared-beta",
        "ordinary": "C:/attached/ordinary",
        "detached": "C:/attached/detached",
    }

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## main\n"
        if command == "git branch --format=%(refname:short)":
            return "main\ncodex/shared\ncodex/ordinary\n"
        if command == "git branch --all --no-merged origin/main":
            return "  codex/shared\n  codex/ordinary\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return (
                "worktree C:/attached/main\n"
                f"HEAD {'0' * 40}\n"
                "branch refs/heads/main\n\n"
                f"worktree {worktree_paths['shared_beta']}\n"
                f"HEAD {'a' * 40}\n"
                "branch refs/heads/codex/shared\n\n"
                f"worktree {worktree_paths['ordinary']}\n"
                f"HEAD {'b' * 40}\n"
                "branch refs/heads/codex/ordinary\n\n"
                f"worktree {worktree_paths['shared_alpha']}\n"
                f"HEAD {'a' * 40}\n"
                "branch refs/heads/codex/shared\n\n"
                f"worktree {worktree_paths['detached']}\n"
                f"HEAD {'c' * 40}\n"
                "detached\n"
            )
        if command == "git log -5 --oneline":
            return ""
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            branch_by_path = {
                "C:/attached/main": "main",
                worktree_paths["shared_alpha"]: "codex/shared",
                worktree_paths["shared_beta"]: "codex/shared",
                worktree_paths["ordinary"]: "codex/ordinary",
            }
            branch = branch_by_path.get(args[4])
            return f"## {branch}\n" if branch is not None else "## HEAD (no branch)\n"
        raise AssertionError(command)

    first = run_local_cockpit_refresh(repo, date(2026, 7, 13), run_command=fake_run)
    first_register = first.register_path.read_bytes()
    first_auto = (repo / AUTO_DECISIONS_PATH).read_bytes()
    payload = json.loads(first_auto)
    surface_ids = set(payload["surfaces"])
    shared_collision_ids = {
        surface_id for surface_id in surface_ids if surface_id.startswith("worktree:codex/shared@")
    }
    detached_ids = {
        surface_id for surface_id in surface_ids if surface_id.startswith("detached@ccccccc-")
    }

    assert "codex/shared" in surface_ids
    assert "codex/ordinary" in surface_ids
    assert len(shared_collision_ids) == 1
    assert len(detached_ids) == 1
    assert len(surface_ids) == 4
    register = first_register.decode("utf-8")
    register_rows = register.splitlines()
    shared_row = next(row for row in register_rows if "Git surface: codex/shared" in row)
    collision_id = next(iter(shared_collision_ids))
    collision_row = next(row for row in register_rows if f"Git surface: {collision_id}" in row)
    ordinary_row = next(row for row in register_rows if "Git surface: codex/ordinary" in row)
    detached_id = next(iter(detached_ids))
    detached_row = next(row for row in register_rows if f"Git surface: {detached_id}" in row)
    assert worktree_paths["shared_alpha"] in shared_row
    assert worktree_paths["shared_beta"] not in shared_row
    assert worktree_paths["shared_beta"] in collision_row
    assert worktree_paths["shared_alpha"] not in collision_row
    assert worktree_paths["ordinary"] in ordinary_row
    assert worktree_paths["detached"] in detached_row
    for surface_id in surface_ids:
        assert f"Git surface: {surface_id}" in register

    second = run_local_cockpit_refresh(repo, date(2026, 7, 13), run_command=fake_run)
    assert second.register_path.read_bytes() == first_register
    assert (repo / AUTO_DECISIONS_PATH).read_bytes() == first_auto


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
        auto_decisions_enabled=False,
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


def test_run_github_cockpit_refresh_uses_preprovisioned_worktree_and_syncs(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    commands: list[list[str]] = []
    current_branch = "codex/cockpit-refresh-20260620"
    pushed = False
    pr_labels: set[str] = set()

    def fake_run(args: list[str]) -> str:
        nonlocal current_branch, pushed
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        command = " ".join(args)
        if command == "git status --porcelain=v1":
            return ""
        if command == "git branch --show-current":
            return current_branch + "\n"
        if command == "git ls-remote --heads origin codex/cockpit-refresh-20260620":
            return ""
        if command == "git fetch origin main":
            return ""
        if command in {"git rev-parse FETCH_HEAD", "git rev-parse HEAD"}:
            return "a" * 40 + "\n"
        if command.startswith("git diff --name-status -z --find-renames"):
            return ""
        if command.startswith("git for-each-ref"):
            return ""
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return "## codex/cockpit-refresh-20260620\n"
        if command == "git branch --format=%(refname:short)":
            return "main\ncodex/cockpit-refresh-20260620\n"
        if command == "git branch --all --no-merged origin/main":
            return ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return (
                "worktree C:/repo\nHEAD 60e3d96\nbranch refs/heads/codex/cockpit-refresh-20260620\n"
            )
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
        if command == "git diff --cached --name-only":
            return "docs/agents/workstreams.md\ndocs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            pushed = True
            return ""
        if command.startswith("gh pr list"):
            if not pushed:
                return "[]"
            return json.dumps(
                [
                    {
                        "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                        "title": "Cockpit refresh: 2026-06-20",
                        "baseRefName": "main",
                        "headRefName": "codex/cockpit-refresh-20260620",
                        "state": "OPEN",
                    }
                ]
            )
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\n"
        if command.startswith("gh issue view 100"):
            return ""
        if command.startswith("gh issue edit 100"):
            return ""
        if command.startswith("gh pr view 99"):
            return "\n".join(sorted(pr_labels))
        if command.startswith("gh pr edit 99"):
            pr_labels.update(args[-1].split(","))
            return ""
        if command == ("gh api repos/magilliam27/MCI-GRU/issues/100/comments --paginate --slurp"):
            return "[[]]"
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

    assert commands[:10] == [
        ["git", "status", "--porcelain=v1"],
        ["git", "branch", "--show-current"],
        ["git", "rev-parse", "--absolute-git-dir"],
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        ["git", "fetch", "origin", "main"],
        ["git", "rev-parse", "FETCH_HEAD"],
        ["git", "ls-remote", "--heads", "origin", "codex/cockpit-refresh-20260620"],
        ["git", "rev-parse", "HEAD"],
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            f"{'a' * 40}...{'a' * 40}",
            "--",
        ],
        ["git", "status", "--porcelain=v1"],
    ]
    assert result.github is not None
    assert result.github.pr_url == "https://github.com/magilliam27/MCI-GRU/pull/99"
    assert result.register_path.exists()
    assert any(command[:3] == ["gh", "issue", "comment"] for command in commands)


def test_run_github_cockpit_refresh_refuses_a_dirty_checkout_before_switching(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        if args == ["git", "status", "--porcelain=v1"]:
            return "M user-work.txt\n"
        if args == ["git", "switch", "-C", "codex/cockpit-refresh-20260620"]:
            return ""
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="clean checkout"):
        run_github_cockpit_refresh(
            repo_root=repo,
            run_date=date(2026, 6, 20),
            run_command=fake_run,
        )

    assert commands == [["git", "status", "--porcelain=v1"]]


def test_producer_reuses_fetched_remote_dated_branch_at_exact_oid() -> None:
    branch = "codex/cockpit-refresh-20260620"
    head_oid = "a" * 40
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return f"{head_oid}\trefs/heads/{branch}\n"
        if args == ["git", "fetch", "origin", "main"]:
            return ""
        if args == ["git", "fetch", "origin", branch]:
            return ""
        if args in (["git", "rev-parse", "FETCH_HEAD"], ["git", "rev-parse", "HEAD"]):
            return head_oid + "\n"
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return ""
        raise AssertionError(" ".join(args))

    _switch_to_cockpit_branch(fake_run, branch)

    assert ["git", "fetch", "origin", branch] in commands
    assert not any(command[:2] == ["git", "switch"] for command in commands)
    assert not any("-C" in command or "reset" in command for command in commands)


def test_producer_reuses_remote_branch_containing_curator_registry_commit() -> None:
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    head_oid = "a" * 40
    fetched = ""

    def fake_run(args: list[str]) -> str:
        nonlocal fetched
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "fetch", "origin", "main"]:
            fetched = "main"
            return ""
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return f"{head_oid}\trefs/heads/{branch}\n"
        if args == ["git", "fetch", "origin", branch]:
            fetched = "branch"
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return (base_oid if fetched == "main" else head_oid) + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return head_oid + "\n"
        if args == [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            f"{base_oid}...{head_oid}",
            "--",
        ]:
            return f"M\x00{DECISION_REGISTRY_PATH}\x00"
        raise AssertionError(" ".join(args))

    assert _switch_to_cockpit_branch(fake_run, branch) == (base_oid, head_oid)


def test_producer_rejects_unrelated_path_already_on_remote_dated_branch() -> None:
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    head_oid = "a" * 40
    fetched = ""
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        nonlocal fetched
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "fetch", "origin", "main"]:
            fetched = "main"
            return ""
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return f"{head_oid}\trefs/heads/{branch}\n"
        if args == ["git", "fetch", "origin", branch]:
            fetched = "branch"
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return (base_oid if fetched == "main" else head_oid) + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return head_oid + "\n"
        if args == [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            f"{base_oid}...{head_oid}",
            "--",
        ]:
            return "A\x00user-work.txt\x00"
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="unexpected path: user-work.txt"):
        _switch_to_cockpit_branch(fake_run, branch)

    assert not any("-C" in command or "reset" in command for command in commands)


def test_producer_refuses_clean_primary_main_without_switching() -> None:
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    commands: list[list[str]] = []
    current = "main"

    def fake_run(args: list[str]) -> str:
        nonlocal current
        commands.append(args)
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return current + "\n"
        if args == ["git", "fetch", "origin", "main"]:
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return base_oid + "\n"
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return ""
        if args == ["git", "branch", "--list", branch]:
            return ""
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return ""
        if args == ["git", "switch", "--create", branch, "FETCH_HEAD"]:
            current = branch
            return ""
        if args == ["git", "rev-parse", "HEAD"]:
            return base_oid + "\n"
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="pre-provisioned disposable linked worktree"):
        _switch_to_cockpit_branch(fake_run, branch)

    assert commands == [
        ["git", "status", "--porcelain=v1"],
        ["git", "branch", "--show-current"],
    ]
    assert current == "main"


def test_producer_refuses_primary_checkout_even_on_dated_branch() -> None:
    branch = "codex/cockpit-refresh-20260620"
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "rev-parse", "--absolute-git-dir"]:
            return "C:/repo/.git\n"
        if args == ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]:
            return "C:/repo/.git\n"
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="pre-provisioned disposable linked worktree"):
        _switch_to_cockpit_branch(fake_run, branch)

    assert not any(command[:2] == ["git", "switch"] for command in commands)


def test_producer_accepts_preprovisioned_unpublished_branch_at_fetched_main() -> None:
    branch = "codex/cockpit-refresh-20260620"
    base_oid = "b" * 40
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return ""
        if args == ["git", "fetch", "origin", "main"]:
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return base_oid + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return base_oid + "\n"
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return ""
        raise AssertionError(" ".join(args))

    _switch_to_cockpit_branch(fake_run, branch)

    assert ["git", "fetch", "origin", "main"] in commands
    assert not any(command[:2] == ["git", "switch"] for command in commands)
    assert not any("-C" in command or "reset" in command for command in commands)


def test_producer_rejects_local_dated_branch_that_differs_from_remote() -> None:
    branch = "codex/cockpit-refresh-20260620"
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        if args == ["git", "status", "--porcelain=v1"]:
            return ""
        if args == ["git", "branch", "--show-current"]:
            return branch + "\n"
        if args == ["git", "ls-remote", "--heads", "origin", branch]:
            return f"{'a' * 40}\trefs/heads/{branch}\n"
        if args == ["git", "fetch", "origin", "main"]:
            return ""
        if args == ["git", "fetch", "origin", branch]:
            return ""
        if args == ["git", "rev-parse", "FETCH_HEAD"]:
            return "a" * 40 + "\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return "c" * 40 + "\n"
        if args[:4] == ["git", "diff", "--name-status", "-z"]:
            return ""
        raise AssertionError(" ".join(args))

    with pytest.raises(RuntimeError, match="does not match fetched remote head"):
        _switch_to_cockpit_branch(fake_run, branch)

    assert not any("-C" in command or "reset" in command for command in commands)


def test_run_github_cockpit_refresh_commits_final_packet_once(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    commands: list[list[str]] = []
    diff_calls = 0
    pr_checks = 0
    pr_labels: set[str] = set()

    def fake_run(args: list[str]) -> str:
        nonlocal diff_calls, pr_checks
        commands.append(args)
        disposable_metadata = _disposable_worktree_metadata(args)
        if disposable_metadata is not None:
            return disposable_metadata
        command = " ".join(args)
        if command == "git status --porcelain=v1":
            return ""
        if command == "git branch --show-current":
            return "codex/cockpit-refresh-20260620\n"
        if command == "git ls-remote --heads origin codex/cockpit-refresh-20260620":
            return f"{'a' * 40}\trefs/heads/codex/cockpit-refresh-20260620\n"
        if command == "git fetch origin codex/cockpit-refresh-20260620":
            return ""
        if command == "git fetch origin main":
            return ""
        if command in {"git rev-parse FETCH_HEAD", "git rev-parse HEAD"}:
            return "a" * 40 + "\n"
        if command.startswith("git diff --name-status -z --find-renames"):
            return ""
        if command.startswith("git for-each-ref"):
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
        if command == "git diff --cached --name-only":
            diff_calls += 1
            if diff_calls == 1:
                return "docs/agents/workstreams.md\ndocs/agents/cockpit/2026-06-20.md\n"
            return "docs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command == "git push -u origin codex/cockpit-refresh-20260620":
            return ""
        if command.startswith("gh pr list"):
            pr_checks += 1
            if pr_checks == 1:
                return "[]"
            return json.dumps(
                [
                    {
                        "url": "https://github.com/magilliam27/MCI-GRU/pull/99",
                        "title": "Cockpit refresh: 2026-06-20",
                        "baseRefName": "main",
                        "headRefName": "codex/cockpit-refresh-20260620",
                        "state": "OPEN",
                    }
                ]
            )
        if command.startswith("gh issue list"):
            return "100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\n"
        if command.startswith("gh issue view 100"):
            return ""
        if command.startswith("gh issue edit 100"):
            return ""
        if command.startswith("gh pr view 99"):
            return "\n".join(sorted(pr_labels))
        if command.startswith("gh pr edit 99"):
            pr_labels.update(args[-1].split(","))
            return ""
        if command == ("gh api repos/magilliam27/MCI-GRU/issues/100/comments --paginate --slurp"):
            return "[[]]"
        if command == ("gh api repos/magilliam27/MCI-GRU/issues/100/comments --paginate --slurp"):
            return "[[]]"
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
    assert "ensure the generated cockpit artifact set is committed" in packet
    assert "ensure one dated cockpit issue digest" in packet
    assert "GitHub sync skipped" not in packet
    assert "GitHub mutation disabled" not in packet
    commits = [command for command in commands if command[:2] == ["git", "commit"]]
    assert len(commits) == 1
    assert commits[0][3] == "Refresh cockpit status for 2026-06-20"
    assert diff_calls == 1


def test_local_refresh_excludes_automation_branch_self_churn_from_generated_artifacts(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    branch = "codex/cockpit-refresh-20260620"
    ahead = 0

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return f"2026-06-20\trefs/heads/{branch}\n"
        if command == "git status --short":
            return ""
        if command == "git status --short --branch":
            return f"## {branch}\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{branch}\n"
        if command == "git branch --all --no-merged origin/main":
            return f"  {branch}\n" if ahead else ""
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return f"0\t{ahead}\n"
        if command == "git worktree list --porcelain":
            return f"worktree C:/automation\nHEAD abc1234\nbranch refs/heads/{branch}\n"
        if command == "git log -5 --oneline":
            return "abc1234 Automation branch\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return f"## {branch}\n"
        if command == f"git show origin/main:{AUTO_DECISIONS_PATH}":
            return '{"format_version":2,"surfaces":{},"workstreams":{}}'
        if command == f"git show origin/main:{DECISION_REGISTRY_PATH}":
            return _empty_registry()
        raise AssertionError(command)

    kwargs = {
        "repo_root": repo,
        "run_date": date(2026, 6, 20),
        "run_command": fake_run,
        "github_evidence": None,
        "automation_branch": branch,
        "comparison_ref": "origin/main",
    }
    first = run_local_cockpit_refresh(**kwargs)
    first_packet = first.packet_path.read_bytes()
    first_auto = (repo / AUTO_DECISIONS_PATH).read_bytes()
    first_register = first.register_path.read_bytes()

    ahead = 2
    second = run_local_cockpit_refresh(**kwargs)

    assert second.packet_path.read_bytes() == first_packet
    assert (repo / AUTO_DECISIONS_PATH).read_bytes() == first_auto
    assert second.register_path.read_bytes() == first_register


def test_producer_restores_owned_files_when_sync_fails_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    topology_runner = _fake_topology_runner([])

    def fake_run(args: list[str]) -> str:
        if args == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if args[:4] == ["git", "restore", "--staged", "--"]:
            return ""
        return topology_runner(args)

    monkeypatch.setattr(
        cockpit_runner_module,
        "_switch_to_cockpit_branch",
        lambda runner, branch: ("a" * 40, None),
    )

    def fail_sync(**kwargs):
        del kwargs
        raise RuntimeError("sync failed before commit")

    monkeypatch.setattr(cockpit_runner_module, "sync_github", fail_sync)

    with pytest.raises(RuntimeError, match="sync failed before commit"):
        run_github_cockpit_refresh(
            repo_root=repo,
            run_date=date(2026, 6, 20),
            run_command=fake_run,
        )

    for relative in (
        "docs/agents/workstreams.md",
        "docs/agents/cockpit/2026-06-20.md",
        AUTO_DECISIONS_PATH,
    ):
        assert not (repo / relative).exists()


def test_github_refresh_same_date_is_commit_and_comment_idempotent(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(["init", "--bare"], cwd=origin)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-b", "main"], cwd=repo)
    _git(["config", "user.name", "Cockpit Test"], cwd=repo)
    _git(["config", "user.email", "cockpit-test@example.invalid"], cwd=repo)
    _repo_with_required_docs(repo)
    cockpit_dir = repo / "docs" / "agents" / "cockpit"
    (cockpit_dir / "RUNBOOK.md").write_text("# Cockpit runbook\n", encoding="utf-8")
    (cockpit_dir / "override-receipts.json").write_text(
        '{"format_version":1,"processed_comment_ids":[]}\n',
        encoding="utf-8",
    )
    _git(["add", "."], cwd=repo)
    _git(["commit", "-m", "Create cockpit baseline"], cwd=repo)
    _git(["remote", "add", "origin", str(origin)], cwd=repo)
    _git(["push", "-u", "origin", "main"], cwd=repo)
    source_head = _git(["rev-parse", "HEAD"], cwd=repo)
    execution = tmp_path / "cockpit-execution"
    _git(
        [
            "worktree",
            "add",
            "-b",
            "codex/cockpit-refresh-20260620",
            str(execution),
            "main",
        ],
        cwd=repo,
    )

    pr_exists = False
    comments: list[dict[str, object]] = []

    def runner(args: list[str]) -> str:
        nonlocal pr_exists
        if args[0] == "git":
            completed = subprocess.run(
                args,
                cwd=execution,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout
        if args == ["gh", "auth", "status"]:
            return "Logged in\n"
        if args[:3] == ["gh", "pr", "list"]:
            if "--head" in args:
                if not pr_exists:
                    return "[]"
                return json.dumps(
                    [
                        {
                            "number": 99,
                            "url": "https://github.example/pull/99",
                            "title": "Cockpit refresh: 2026-06-20",
                            "baseRefName": "main",
                            "headRefName": "codex/cockpit-refresh-20260620",
                            "state": "OPEN",
                        }
                    ]
                )
            if not pr_exists:
                return "[]"
            return json.dumps(
                [
                    {
                        "number": 99,
                        "headRefName": "codex/cockpit-refresh-20260620",
                        "url": "https://github.example/pull/99",
                        "isDraft": False,
                        "state": "OPEN",
                        "mergedAt": None,
                        "updatedAt": "2026-06-20T12:00:00Z",
                    }
                ]
            )
        if args[:3] == ["gh", "pr", "view"]:
            return json.dumps(
                {
                    "title": "Cockpit refresh: 2026-06-20",
                    "headRefName": "codex/cockpit-refresh-20260620",
                    "headRefOid": _git(["rev-parse", "HEAD"], cwd=execution),
                    "baseRefName": "main",
                    "baseRefOid": _git(["rev-parse", "main"], cwd=execution),
                    "headRepositoryOwner": {"login": "magilliam27"},
                    "isCrossRepository": False,
                    "url": "https://github.example/pull/99",
                    "state": "OPEN",
                }
            )
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/pulls/99/files"]:
            paths = _git(["diff", "--name-only", "main...HEAD"], cwd=execution).splitlines()
            return json.dumps([[{"filename": path} for path in paths]])
        if args[:3] == ["gh", "pr", "create"]:
            pr_exists = True
            return "https://github.example/pull/99\n"
        if args[:3] == ["gh", "issue", "list"]:
            return "100\n" if "--search" in args else "[]"
        if args[:3] == ["gh", "label", "list"]:
            return ""
        if args[:3] == ["gh", "api", "repos/magilliam27/MCI-GRU/issues/100/comments"]:
            return json.dumps([comments])
        if args[:3] == ["gh", "issue", "comment"]:
            comments.append(
                {
                    "id": 501,
                    "body": args[-1],
                    "user": {"login": "magilliam27"},
                    "author_association": "OWNER",
                }
            )
            return ""
        raise AssertionError(" ".join(args))

    first = run_github_cockpit_refresh(
        repo_root=execution,
        run_date=date(2026, 6, 20),
        run_command=runner,
    )
    first_head = _git(["rev-parse", "HEAD"], cwd=execution)
    artifact_bytes = {
        path: (execution / path).read_bytes()
        for path in (
            "docs/agents/workstreams.md",
            "docs/agents/cockpit/2026-06-20.md",
            AUTO_DECISIONS_PATH,
        )
    }

    second = run_github_cockpit_refresh(
        repo_root=execution,
        run_date=date(2026, 6, 20),
        run_command=runner,
    )

    assert _git(["branch", "--show-current"], cwd=repo).strip() == "main"
    assert _git(["rev-parse", "HEAD"], cwd=repo) == source_head
    assert _git(["status", "--porcelain"], cwd=repo) == ""
    assert _git(["rev-parse", "HEAD"], cwd=execution) == first_head
    assert _git(["status", "--porcelain"], cwd=execution) == ""
    assert len(comments) == 1
    assert second.packet_path.read_bytes() == artifact_bytes["docs/agents/cockpit/2026-06-20.md"]
    assert second.register_path.read_bytes() == artifact_bytes["docs/agents/workstreams.md"]
    assert (execution / AUTO_DECISIONS_PATH).read_bytes() == artifact_bytes[AUTO_DECISIONS_PATH]
    assert first.github is not None and second.github is not None


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


def test_cockpit_packet_groups_canonical_parked_and_cleanup_queues(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    _write_decision_registry(
        repo,
        workstreams={
            "LambdaRankIC": {
                "status": "active",
                "canonical_surface": "origin/main after merged recovery",
                "reason": "Reviewed active baseline.",
                "next_action": "Continue approved diagnostics.",
                "last_reviewed": "2026-07-10",
            },
            "Issue #8 volatility targeting": {
                "status": "ready-for-agent",
                "canonical_surface": "GitHub issue #8 plus current origin/main",
                "reason": "Reviewed continuation is ready.",
                "next_action": "Inspect existing Drive outputs first.",
                "last_reviewed": "2026-07-09",
            },
            "Colab operations": {
                "status": "ready-for-agent",
                "canonical_surface": "origin/main plus the Chrome-control runbook",
                "reason": "The canonical runbook is ready.",
                "next_action": "Use the canonical runbook.",
                "last_reviewed": "2026-07-09",
            },
            "Portfolio-IC": {
                "status": "parked",
                "canonical_surface": "origin/main pure IC baseline",
                "reason": "Reviewed non-default evidence is parked.",
                "next_action": "Keep parked.",
                "last_reviewed": "2026-07-09",
            },
            "Historical cleanup": {
                "status": "archive",
                "canonical_surface": "merged historical branch",
                "reason": "Historical surface is archive-only.",
                "next_action": "Remove only with cleanup approval.",
                "last_reviewed": "2026-07-09",
            },
        },
        surfaces={},
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 10),
        run_command=_fake_topology_runner([]),
        github_evidence=None,
    )

    packet = result.packet_path.read_text(encoding="utf-8")
    canonical = packet.split("## Canonical / Ready Queue", 1)[1].split("## ", 1)[0]
    parked = packet.split("## Parked Queue", 1)[1].split("## ", 1)[0]
    cleanup = packet.split("## Archive / Cleanup Candidates", 1)[1].split("## ", 1)[0]

    assert "LambdaRankIC" in canonical
    assert "Issue #8 volatility targeting" in canonical
    assert "Colab operations" in canonical
    assert "Git and worktree hygiene" not in canonical
    assert "Portfolio-IC" in parked
    assert "Historical cleanup" in cleanup
    assert "Portfolio-IC" not in cleanup
    assert "Issue #8 volatility targeting" not in parked + cleanup


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


@pytest.mark.parametrize(
    "branches",
    [pytest.param([], id="empty-topology"), pytest.param(["codex/unrelated-live"], id="unrelated")],
)
def test_evidenceless_seed_remains_stale_across_topology(
    tmp_path: Path,
    branches: list[str],
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    source = _FixedSource(
        [
            WorkstreamSeed(
                name="Evidence gap",
                status=WorkstreamStatus.NEEDS_USER_DECISION,
                next_action="Legacy unresolved seed action.",
                branch_terms=("evidence-gap",),
            )
        ]
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 12),
        run_command=_fake_topology_runner(branches),
        sources=(source,),
        github_evidence=GitHubEvidence(),
    )

    register = result.register_path.read_text(encoding="utf-8")
    payload = json.loads((repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8"))
    assert "| Evidence gap | stale |" in register
    assert "| Evidence gap | needs-user-decision |" not in register
    assert payload["workstreams"]["Evidence gap"]["status"] == "stale"
    assert payload["workstreams"]["Evidence gap"]["rule"] == "no-current-evidence-stale"


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


def _generated_surface_for_alias(
    workstream: str,
    *,
    confidence: Confidence = Confidence.HIGH,
    association_basis: str = "branch-term",
) -> AutoDisposition:
    return AutoDisposition(
        workstreams=(workstream,),
        disposition=SurfaceDisposition.CANONICAL,
        rule="unique-live-surface",
        evidence="Independently generated classification.",
        confidence=confidence,
        alternatives=(),
        last_reviewed=date(2026, 7, 12),
        association_basis=association_basis,
    )


def test_implied_aliases_derives_slug_and_tokens_from_generated_surface() -> None:
    aliases = implied_aliases(
        {"codex/portfolio-ic-hybrid-testing-20260712": _generated_surface_for_alias("Portfolio-IC")}
    )

    assert aliases == {
        "hybrid": "Portfolio-IC",
        "ic": "Portfolio-IC",
        "portfolio": "Portfolio-IC",
        "portfolio-ic-hybrid-testing": "Portfolio-IC",
        "testing": "Portfolio-IC",
    }


def test_implied_aliases_drops_collisions_without_losing_unique_aliases() -> None:
    surfaces = {
        branch: _generated_surface_for_alias(workstream)
        for branch, workstream in {
            "codex/alpha-shared-20260712": "Alpha",
            "codex/beta-shared-20260712": "Beta",
        }.items()
    }

    aliases = implied_aliases(surfaces)

    assert "shared" not in aliases
    assert aliases == {
        "alpha": "Alpha",
        "alpha-shared": "Alpha",
        "beta": "Beta",
        "beta-shared": "Beta",
    }


def test_implied_aliases_accepts_only_independently_grounded_high_confidence_auto() -> None:
    surfaces = {
        f"codex/{slug}": _generated_surface_for_alias(
            workstream,
            confidence=confidence,
            association_basis=basis,
        )
        for slug, workstream, confidence, basis in (
            ("direct-learning", "Direct", Confidence.HIGH, "branch-term"),
            ("low-learning", "Low", Confidence.LOW, "branch-term"),
            ("alias-loop", "Loop", Confidence.HIGH, "implied-alias"),
            ("fallback-learning", "Fallback", Confidence.HIGH, "title-case-fallback"),
            ("legacy-pr-link", "Legacy PR", Confidence.HIGH, "linked-pr"),
            ("legacy-issue-link", "Legacy issue", Confidence.HIGH, "linked-issue"),
        )
    }

    aliases = implied_aliases(surfaces)

    assert aliases == {
        "direct": "Direct",
        "direct-learning": "Direct",
        "learning": "Direct",
    }


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
    assert seed.association_basis == "title-case-fallback"


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


def test_git_activity_source_excludes_integration_branches() -> None:
    evidence = _git_activity_evidence(
        recent_branches=[
            ("main", _GIT_ACTIVITY_RUN_DATE),
            ("master", _GIT_ACTIVITY_RUN_DATE),
            ("codex/real-topic", _GIT_ACTIVITY_RUN_DATE),
        ]
    )

    seeds = GitActivitySource(aliases={}).provide(evidence, _GIT_ACTIVITY_RUN_DATE)

    assert [seed.name for seed in seeds] == ["Real Topic"]


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


def test_run_local_cockpit_refresh_uses_reviewed_surface_alias_for_new_branch(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    reviewed = "codex/portfolio-ic-hybrid-testing-20260712"
    discovered = "codex/hybrid-testing-v2-20260713"
    _write_decision_registry(
        repo,
        workstreams={
            "Portfolio-IC": {
                "status": "active",
                "canonical_surface": reviewed,
                "reason": "Reviewed workstream.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            }
        },
        surfaces={
            reviewed: {
                "workstreams": ["Portfolio-IC"],
                "disposition": "canonical",
                "reason": "Reviewed classification.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            }
        },
    )

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return f"2026-07-12\trefs/heads/{reviewed}\n2026-07-13\trefs/heads/{discovered}\n"
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{reviewed}\n{discovered}\n"
        if command == "git branch --all --no-merged origin/main":
            return f"  {reviewed}\n  {discovered}\n"
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
        run_date=date(2026, 7, 13),
        run_command=fake_run,
        auto_decisions_enabled=True,
        github_evidence=None,
    )

    register = result.register_path.read_text(encoding="utf-8")
    auto_path = repo / AUTO_DECISIONS_PATH
    first_bytes = auto_path.read_bytes()
    payload = json.loads(first_bytes)
    assert "Portfolio-IC" in register
    assert "Hybrid Testing V2" not in register
    assert payload["surfaces"][discovered]["workstreams"] == ["Portfolio-IC"]
    assert payload["surfaces"][discovered]["association_basis"] == "implied-alias"

    run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 13),
        run_command=fake_run,
        auto_decisions_enabled=True,
        github_evidence=None,
    )
    assert auto_path.read_bytes() == first_bytes


def test_explicit_surface_assignment_does_not_teach_generated_aliases(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    reviewed = "codex/mystery-shared-20260712"
    discovered = "codex/shared-followup-20260713"
    _write_decision_registry(
        repo,
        workstreams={
            "Portfolio-IC": {
                "status": "active",
                "canonical_surface": reviewed,
                "reason": "Explicit human workstream override.",
                "next_action": "Continue through the effective overlay.",
                "last_reviewed": "2026-07-12",
            }
        },
        surfaces={
            reviewed: {
                "workstreams": ["Portfolio-IC"],
                "disposition": "canonical",
                "reason": "Explicit human association.",
                "next_action": "Apply only in the effective overlay.",
                "last_reviewed": "2026-07-12",
            }
        },
    )

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return f"2026-07-12\trefs/heads/{reviewed}\n2026-07-13\trefs/heads/{discovered}\n"
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{reviewed}\n{discovered}\n"
        if command == "git branch --all --no-merged origin/main":
            return f"  {reviewed}\n  {discovered}\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc1234\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "abc1234 Current main\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main...origin/main\n"
        raise AssertionError(command)

    run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 13),
        run_command=fake_run,
        github_evidence=None,
    )

    payload = json.loads((repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8"))
    assert payload["surfaces"][reviewed]["workstreams"] != ["Portfolio-IC"]
    assert payload["surfaces"][discovered]["workstreams"] != ["Portfolio-IC"]


def test_run_local_cockpit_refresh_auto_alias_learning_reaches_fixed_point_in_one_run(
    tmp_path: Path,
) -> None:
    repo = _repo_with_required_docs(tmp_path)
    teacher = "codex/lambdarank-recovery-20260712"
    learned = "codex/recovery-followup-20260713"

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return f"2026-07-12\trefs/heads/{teacher}\n2026-07-13\trefs/heads/{learned}\n"
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{teacher}\n{learned}\n"
        if command == "git branch --all --no-merged origin/main":
            return f"  {teacher}\n  {learned}\n"
        if command == "git rev-list --left-right --count origin/main...HEAD":
            return "0\t0\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc1234\nbranch refs/heads/main\n"
        if command == "git log -5 --oneline":
            return "abc1234 Current main\n"
        if args[:2] == ["git", "-c"] and args[3] == "-C":
            return "## main...origin/main\n"
        raise AssertionError(command)

    kwargs = {
        "repo_root": repo,
        "run_date": date(2026, 7, 13),
        "run_command": fake_run,
        "auto_decisions_enabled": True,
        "github_evidence": GitHubEvidence(),
    }
    run_local_cockpit_refresh(**kwargs)
    auto_path = repo / AUTO_DECISIONS_PATH
    first_bytes = auto_path.read_bytes()

    run_local_cockpit_refresh(**kwargs)

    assert auto_path.read_bytes() == first_bytes


def test_run_local_cockpit_refresh_explicit_alias_beats_implied_alias(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    reviewed = "codex/portfolio-ic-hybrid-testing-20260712"
    discovered = "codex/hybrid-testing-v2-20260713"
    _write_decision_registry(
        repo,
        workstreams={
            "Portfolio-IC": {
                "status": "active",
                "canonical_surface": reviewed,
                "reason": "Reviewed workstream.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            },
            "Other stream": {
                "status": "active",
                "canonical_surface": discovered,
                "reason": "Explicit alias target.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            },
        },
        surfaces={
            reviewed: {
                "workstreams": ["Portfolio-IC"],
                "disposition": "canonical",
                "reason": "Reviewed classification.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            }
        },
        aliases={"hybrid": "Other stream"},
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 13),
        run_command=_dated_topology_runner(discovered, date(2026, 7, 13)),
        auto_decisions_enabled=True,
        github_evidence=None,
    )

    register = result.register_path.read_text(encoding="utf-8")
    payload = json.loads((repo / AUTO_DECISIONS_PATH).read_text(encoding="utf-8"))
    assert "Other stream" in register
    assert "Hybrid Testing V2" not in register
    assert payload["surfaces"][discovered]["workstreams"] == ["Other stream"]
    assert payload["surfaces"][discovered]["association_basis"] == "explicit-alias"


def test_run_local_cockpit_refresh_flag_off_ignores_implied_aliases(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)
    reviewed = "codex/portfolio-ic-hybrid-testing-20260712"
    discovered = "codex/hybrid-testing-v2-20260713"
    _write_decision_registry(
        repo,
        workstreams={
            "Portfolio-IC": {
                "status": "active",
                "canonical_surface": reviewed,
                "reason": "Reviewed workstream.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            }
        },
        surfaces={
            reviewed: {
                "workstreams": ["Portfolio-IC"],
                "disposition": "canonical",
                "reason": "Reviewed classification.",
                "next_action": "Continue.",
                "last_reviewed": "2026-07-12",
            }
        },
    )

    result = run_local_cockpit_refresh(
        repo_root=repo,
        run_date=date(2026, 7, 13),
        run_command=_dated_topology_runner(discovered, date(2026, 7, 13)),
        auto_decisions_enabled=False,
    )

    register = result.register_path.read_text(encoding="utf-8")
    assert "Hybrid Testing V2" in register
    assert not (repo / AUTO_DECISIONS_PATH).exists()


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
    aliases: dict[str, str] | None = None,
) -> None:
    payload = {
        "format_version": 2 if aliases is not None else 1,
        "workstreams": workstreams,
        "surfaces": surfaces,
    }
    if aliases is not None:
        payload["workstream_aliases"] = aliases
    (repo / DECISION_REGISTRY_PATH).write_text(
        json.dumps(payload),
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


def _dated_topology_runner(branch: str, branch_date: date):
    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command.startswith("git for-each-ref"):
            return f"{branch_date.isoformat()}\trefs/heads/{branch}\n"
        if command == "git status --short --branch":
            return "## main...origin/main\n"
        if command == "git branch --format=%(refname:short)":
            return f"main\n{branch}\n"
        if command == "git branch --all --no-merged origin/main":
            return f"  {branch}\n"
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


def _disposable_worktree_metadata(args: list[str]) -> str | None:
    if args == ["git", "rev-parse", "--absolute-git-dir"]:
        return "C:/repo/.git/worktrees/cockpit-refresh\n"
    if args == ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]:
        return "C:/repo/.git\n"
    return None
