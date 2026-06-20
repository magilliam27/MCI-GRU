from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from mci_gru.cockpit.evidence import collect_local_evidence
from mci_gru.cockpit.runner import run_local_cockpit_refresh

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
        if command == "git branch --format=%(refname:short)":
            return "codex/example\nmain\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD abc123\nbranch refs/heads/codex/example\n"
        if command == "git log -5 --oneline":
            return "abc123 Add cockpit design\n"
        raise AssertionError(command)

    evidence = collect_local_evidence(repo, run_command=fake_run)

    assert evidence.required_docs["AGENTS.md"] is True
    assert evidence.required_docs["docs/agents/domain.md"] is True
    assert evidence.required_docs["docs/research/README.md"] is True
    assert evidence.recent_handoffs == ["docs/handoffs/2026-06-01-note.md"]
    assert evidence.dirty_paths == ["docs/agents/domain.md", "scratch.txt"]
    assert evidence.branches == ["codex/example", "main"]
    assert "abc123 Add cockpit design" in evidence.recent_commits


def test_run_local_cockpit_refresh_writes_register_and_packet(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command == "git status --short":
            return ""
        if command == "git branch --format=%(refname:short)":
            return "codex/italy-work-snapshot-20260605\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD a1b5de5\n"
        if command == "git log -5 --oneline":
            return "a1b5de5 Add MCI-GRU cockpit agent design\n"
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
    assert "LambdaRankIC" in result.register_path.read_text(encoding="utf-8")
    assert "Git and worktree hygiene" in result.register_path.read_text(encoding="utf-8")
    assert "**Run color:** green" in result.packet_path.read_text(encoding="utf-8")


def test_run_local_cockpit_refresh_reports_dirty_paths(tmp_path: Path) -> None:
    repo = _repo_with_required_docs(tmp_path)

    def fake_run(args: list[str]) -> str:
        command = " ".join(args)
        if command == "git status --short":
            return " M docs/agents/domain.md\n?? scratch.txt\n"
        if command == "git branch --format=%(refname:short)":
            return "codex/italy-work-snapshot-20260605\n"
        if command == "git worktree list --porcelain":
            return "worktree C:/repo\nHEAD a1b5de5\n"
        if command == "git log -5 --oneline":
            return "a1b5de5 Add MCI-GRU cockpit agent design\n"
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
