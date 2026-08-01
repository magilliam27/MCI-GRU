"""Pin the Claude Code harness so it cannot be severed silently.

Claude Code loads ``CLAUDE.md`` and has no ``AGENTS.md`` fallback. If the import
or the settings file is removed, a session simply starts uninformed: there is no
error, no failing command, and no visible symptom until an agent acts without
repository policy. These tests make that failure loud.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_claude_md_imports_agents_md():
    claude_md = REPO_ROOT / "CLAUDE.md"
    assert claude_md.exists(), (
        "CLAUDE.md is missing. Claude Code cannot see AGENTS.md without it, so "
        "every session would start blind to repository policy."
    )

    text = claude_md.read_text(encoding="utf-8")
    assert "@AGENTS.md" in text, (
        "CLAUDE.md must import AGENTS.md via '@AGENTS.md'. Claude Code has no "
        "AGENTS.md fallback, so removing the import silently detaches the "
        "harness from the repository entrypoint."
    )


def test_claude_settings_select_the_pocock_skill_set():
    settings_path = REPO_ROOT / ".claude" / "settings.json"
    assert settings_path.exists(), (
        ".claude/settings.json is missing. Without it the repository does not "
        "select a skill set and sessions fall back to whatever is enabled at "
        "user scope."
    )

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    enabled = settings["enabledPlugins"]

    assert enabled["mattpocock-skills@mattpocock"] is True, (
        "The Matt Pocock skill set must be enabled for this repository so that "
        "every harness produces identically shaped tickets, specs, and evidence."
    )
    assert enabled["superpowers@claude-plugins-official"] is False, (
        "superpowers must stay disabled here. It ships a SessionStart hook that "
        "directs agents to its own overlapping skills, which would diverge "
        "Claude sessions from the workflow the rest of the repository runs."
    )


def test_gitignore_covers_session_worktrees():
    """Session worktrees must not be able to dirty the protected checkout.

    Claude Code resolves a repository root from ``--git-common-dir``, which is
    the protected checkout for every linked worktree, so a session-created
    worktree lands under ``<protected checkout>/.claude/worktrees/`` regardless
    of where the session started. That directory is a full checkout. Unless the
    path is ignored it appears as an untracked entry and moves the 40-entry
    dirty fingerprint that every session is required to verify.

    This was previously suppressed only by a ``.git/info/exclude`` entry, which
    is untracked, machine-local, and inherited by no clone.
    """
    entries = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".claude/worktrees/" in entries, (
        "'.claude/worktrees/' must stay in the tracked .gitignore. Without it a "
        "session-created worktree shows as untracked in the protected checkout "
        "and breaks the dirty-count fingerprint, with no error to signal why."
    )
