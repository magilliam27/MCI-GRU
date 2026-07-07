# MCI-GRU Cockpit GitHub Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live `gh`-backed cockpit sync so `scripts/refresh_cockpit.py --github-sync` can create the dated cockpit branch/PR, maintain the cockpit issue, comment with each run, and perform evidence-gated issue actions.

**Architecture:** Keep all GitHub mutation in `cockpit.github` behind explicit `enabled=True`. The CLI delegates to a runner-level orchestration function that switches to the dated cockpit branch, refreshes local artifacts, then asks a command-injected gateway to run `git` and `gh` commands. Tests use fake command runners only; no network or live GitHub calls happen in pytest.

**Tech Stack:** Python standard library, `git` CLI, `gh` CLI, dataclasses, pytest, ruff.

---

## File Structure

- Modify `cockpit/github.py`: replace the disabled stub with command-injected live sync helpers.
- Modify `cockpit/runner.py`: add `run_github_cockpit_refresh(...)` orchestration and include `CockpitReport` on `CockpitRunResult`.
- Modify `scripts/refresh_cockpit.py`: wire `--github-sync` to live sync.
- Modify `tests/test_cockpit_github.py`: add fake-runner tests for branch, PR, issue, comment, label, create-issue, and close-issue command plans.
- Modify `tests/test_cockpit_runner.py`: test runner-level live orchestration with fake commands.
- Modify `tests/test_cockpit_cli.py`: test `--github-sync` no longer exits with the disabled-stub message when command runner is not invoked by `--help`.
- Modify `docs/agents/cockpit/RUNBOOK.md`: document live sync, labels, recovery, and the exact command.

## Guardrails

- Live mutation only runs when `--github-sync` reaches `sync_github(..., enabled=True)`.
- All command execution goes through an injectable runner in tests.
- `git add` stages only:
  - `docs/agents/workstreams.md`
  - `docs/agents/cockpit/YYYY-MM-DD.md`
  - `docs/agents/cockpit/RUNBOOK.md`
- `sync_github(enabled=False, ...)` raises `GitHubSyncDisabled`.
- Labels are applied only if `gh label list` shows they already exist.
- Issue closure requires non-empty evidence text and writes an evidence comment before `gh issue close`.

---

### Task 1: Gateway Types And Existing Guard Compatibility

**Files:**
- Modify: `cockpit/github.py`
- Modify: `tests/test_cockpit_github.py`

- [ ] **Step 1: Add failing tests for branch names and disabled guard**

Add:

```python
from datetime import date

from cockpit.github import cockpit_branch_name


def test_cockpit_branch_name_is_dated() -> None:
    assert cockpit_branch_name(date(2026, 6, 20)) == "codex/cockpit-refresh-20260620"
```

Run:

```bash
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_cockpit_github.py -v --basetemp .tmp_pytest\pytest
```

Expected: FAIL with missing `cockpit_branch_name`.

- [ ] **Step 2: Implement branch naming and command result dataclass**

Add to `cockpit/github.py`:

```python
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

CommandRunner = Callable[[list[str]], str]


@dataclass(frozen=True)
class GitHubSyncResult:
    branch: str
    pr_url: str
    cockpit_issue_number: int
    cockpit_issue_url: str
    actions_taken: list[str] = field(default_factory=list)
    actions_skipped: list[str] = field(default_factory=list)


def cockpit_branch_name(run_date: date) -> str:
    return f"codex/cockpit-refresh-{run_date:%Y%m%d}"
```

- [ ] **Step 3: Run focused test**

Run the same pytest command. Expected: PASS for the new branch-name test and the existing disabled guard test.

---

### Task 2: Live Sync Command Plan

**Files:**
- Modify: `cockpit/github.py`
- Modify: `tests/test_cockpit_github.py`

- [ ] **Step 1: Add failing fake-runner test for PR and cockpit issue sync**

Add:

```python
from pathlib import Path

from cockpit.github import sync_github


def test_sync_github_creates_branch_pr_issue_and_comment_with_fake_runner(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        command = " ".join(args)
        if command == "gh auth status":
            return "Logged in"
        if command.startswith("git switch -C codex/cockpit-refresh-20260620"):
            return ""
        if command.startswith("git status --short -- docs/agents/workstreams.md"):
            return "M docs/agents/workstreams.md\nM docs/agents/cockpit/2026-06-20.md\n"
        if command.startswith("git add docs/agents/workstreams.md"):
            return ""
        if command.startswith("git commit -m Refresh cockpit status for 2026-06-20"):
            return "[codex/cockpit-refresh-20260620 abc123] Refresh cockpit status for 2026-06-20"
        if command.startswith("git push -u origin codex/cockpit-refresh-20260620"):
            return ""
        if command.startswith("gh pr list"):
            return ""
        if command.startswith("gh pr create"):
            return "https://github.com/magilliam27/MCI-GRU/pull/99"
        if command.startswith("gh issue list"):
            return ""
        if command.startswith("gh issue create"):
            return "https://github.com/magilliam27/MCI-GRU/issues/100"
        if command.startswith("gh label list"):
            return "cockpit-reviewed\nready-for-agent\n"
        if command.startswith("gh issue edit 100"):
            return ""
        if command.startswith("gh issue comment 100"):
            return ""
        raise AssertionError(command)

    result = sync_github(
        enabled=True,
        repo_root=tmp_path,
        run_date=date(2026, 6, 20),
        run_color="yellow",
        decision_queue=["Portfolio-IC: choose promotion path"],
        run_command=fake_run,
    )

    assert result.branch == "codex/cockpit-refresh-20260620"
    assert result.pr_url == "https://github.com/magilliam27/MCI-GRU/pull/99"
    assert result.cockpit_issue_number == 100
    assert any(command[:2] == ["git", "add"] for command in commands)
    assert any(command[:3] == ["gh", "issue", "comment"] for command in commands)
```

Run:

```bash
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_cockpit_github.py::test_sync_github_creates_branch_pr_issue_and_comment_with_fake_runner -v --basetemp .tmp_pytest\pytest
```

Expected: FAIL because `sync_github` does not accept the live-sync parameters.

- [ ] **Step 2: Implement live sync**

Implement:

```python
def sync_github(
    *,
    enabled: bool,
    repo_root: Path,
    run_date: date,
    run_color: str,
    decision_queue: list[str],
    run_command: CommandRunner | None = None,
    repo: str = "magilliam27/MCI-GRU",
) -> GitHubSyncResult:
    if not enabled:
        raise GitHubSyncDisabled("GitHub cockpit sync requires --github-sync.")
    runner = run_command or _run_command(repo_root)
    branch = cockpit_branch_name(run_date)
    runner(["gh", "auth", "status"])
    runner(["git", "switch", "-C", branch])
    paths = [
        "docs/agents/workstreams.md",
        f"docs/agents/cockpit/{run_date.isoformat()}.md",
        "docs/agents/cockpit/RUNBOOK.md",
    ]
    actions_taken: list[str] = []
    actions_skipped: list[str] = []
    if runner(["git", "status", "--short", "--", *paths]).strip():
        runner(["git", "add", *paths])
        runner(["git", "commit", "-m", f"Refresh cockpit status for {run_date.isoformat()}"])
        actions_taken.append("committed cockpit files")
    else:
        actions_skipped.append("no cockpit file changes to commit")
    runner(["git", "push", "-u", "origin", branch])
    pr_url = _ensure_pr(runner, repo, branch, run_date)
    issue_number, issue_url = _ensure_cockpit_issue(runner, repo)
    _apply_existing_labels(runner, repo, issue_number, ["cockpit-reviewed"], actions_taken, actions_skipped)
    runner(["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", _issue_comment(run_date, run_color, pr_url, decision_queue)])
    actions_taken.append(f"commented on cockpit issue #{issue_number}")
    return GitHubSyncResult(branch, pr_url, issue_number, issue_url, actions_taken, actions_skipped)
```

Use helper functions for `_ensure_pr`, `_ensure_cockpit_issue`, `_apply_existing_labels`, `_issue_comment`, `_parse_issue_url_number`, and `_run_command`.

- [ ] **Step 3: Run fake-runner sync test**

Run the single test. Expected: PASS.

---

### Task 3: Evidence-Gated Issue Autonomy Helpers

**Files:**
- Modify: `cockpit/github.py`
- Modify: `tests/test_cockpit_github.py`

- [ ] **Step 1: Add failing tests for issue creation and closure evidence**

Add:

```python
from cockpit.github import close_issue_with_evidence, create_issue


def test_create_issue_applies_only_existing_labels() -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str]) -> str:
        commands.append(args)
        if args[:3] == ["gh", "label", "list"]:
            return "ready-for-agent\nneeds-info\n"
        if args[:3] == ["gh", "issue", "create"]:
            return "https://github.com/magilliam27/MCI-GRU/issues/101"
        raise AssertionError(" ".join(args))

    url = create_issue(
        title="Clear cockpit follow-up",
        body="Evidence-backed next action.",
        labels=["ready-for-agent", "missing-label"],
        run_command=fake_run,
    )

    assert url.endswith("/101")
    create_command = next(command for command in commands if command[:3] == ["gh", "issue", "create"])
    assert "ready-for-agent" in create_command
    assert "missing-label" not in create_command


def test_close_issue_requires_evidence() -> None:
    with pytest.raises(ValueError, match="closure evidence"):
        close_issue_with_evidence(issue_number=8, evidence="", run_command=lambda args: "")
```

- [ ] **Step 2: Implement helper functions**

Add `create_issue(...)` and `close_issue_with_evidence(...)`. `close_issue_with_evidence` must comment before closing:

```python
runner(["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", evidence])
runner(["gh", "issue", "close", str(issue_number), "--repo", repo])
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_cockpit_github.py -v --basetemp .tmp_pytest\pytest
```

Expected: PASS.

---

### Task 4: Runner And CLI Live Mode

**Files:**
- Modify: `cockpit/runner.py`
- Modify: `scripts/refresh_cockpit.py`
- Modify: `tests/test_cockpit_runner.py`
- Modify: `tests/test_cockpit_cli.py`

- [ ] **Step 1: Add failing runner orchestration test**

Add a test that calls `run_github_cockpit_refresh(...)` with a fake runner and asserts that:

- branch switch happens before generated file writes
- `sync_github` comments on the cockpit issue
- returned result includes a `github` result

- [ ] **Step 2: Implement `run_github_cockpit_refresh`**

Add:

```python
def run_github_cockpit_refresh(repo_root: Path, run_date: date, run_command: RunCommand | None = None) -> CockpitRunResult:
    branch = cockpit_branch_name(run_date)
    runner = run_command or (lambda args: _run_git(args, repo_root))
    runner(["git", "switch", "-C", branch])
    result = run_local_cockpit_refresh(repo_root, run_date, run_command=runner)
    decision_queue = [decision.question for decision in result.report.decisions]
    github = sync_github(
        enabled=True,
        repo_root=repo_root,
        run_date=run_date,
        run_color=result.color.value,
        decision_queue=decision_queue,
        run_command=runner,
    )
    return replace(result, github=github)
```

Add `report: CockpitReport` and `github: GitHubSyncResult | None` to `CockpitRunResult`.

- [ ] **Step 3: Wire CLI**

Change `scripts/refresh_cockpit.py` so:

```python
if args.github_sync:
    result = run_github_cockpit_refresh(...)
else:
    result = run_local_cockpit_refresh(...)
```

- [ ] **Step 4: Run CLI and runner tests**

Run:

```bash
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_cockpit_runner.py tests/test_cockpit_cli.py -v --basetemp .tmp_pytest\pytest
```

Expected: PASS.

---

### Task 5: Runbook Update And Verification

**Files:**
- Modify: `docs/agents/cockpit/RUNBOOK.md`
- Modify: `tests/test_cockpit_cli.py`

- [ ] **Step 1: Add failing docs test**

Assert the runbook includes:

- `python scripts/refresh_cockpit.py --date 2026-06-20 --github-sync`
- `gh auth status`
- `codex/cockpit-refresh-YYYYMMDD`
- `close_issue_with_evidence`

- [ ] **Step 2: Update runbook**

Document live mode, recovery commands, and what labels are safe.

- [ ] **Step 3: Run final focused tests and ruff**

Run:

```bash
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe -m pytest tests/test_cockpit_render.py tests/test_cockpit_runner.py tests/test_cockpit_cli.py tests/test_cockpit_github.py -v --basetemp .tmp_pytest\pytest
C:\Users\magil\MCI-GRU\.venv\Scripts\ruff.exe check cockpit scripts/refresh_cockpit.py tests/test_cockpit_render.py tests/test_cockpit_runner.py tests/test_cockpit_cli.py tests/test_cockpit_github.py
```

Expected: PASS and `All checks passed!`.

---

## Self-Review

**Spec coverage:**
- Dated branch/PR: Tasks 2 and 4.
- Cockpit issue creation/comment: Task 2.
- Safe labels: Tasks 2 and 3.
- Evidence-gated issue creation/closure: Task 3.
- CLI live mode: Task 4.
- Runbook/recovery docs: Task 5.

**Placeholder scan:** No open-ended implementation placeholders remain; the only deferred behavior is broader evidence inference, which is outside this live-sync plumbing slice.

**Type consistency:** `CommandRunner`, `GitHubSyncResult`, `CockpitRunResult.github`, and `run_github_cockpit_refresh` are named consistently across tasks.
