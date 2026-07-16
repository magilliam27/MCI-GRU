from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "cockpit-overrides.yml"
RUNBOOK_PATH = REPO_ROOT / "docs" / "agents" / "cockpit" / "RUNBOOK.md"


def test_cockpit_override_workflow_routes_structured_pr_comments() -> None:
    source = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "issue_comment:" in source
    assert "types: [created]" in source
    assert "github.event.issue.pull_request" in source
    assert "startsWith(github.event.comment.body, '/cockpit ')" in source
    assert "contents: write" in source
    assert "pull-requests: write" in source
    assert "issues: write" in source
    assert 'git config user.name "github-actions[bot]"' in source
    assert 'git config user.email "41898282+github-actions[bot]@users.noreply.github.com"' in source
    assert source.index("git config user.name") < source.index("scripts/apply_cockpit_overrides.py")
    assert "scripts/apply_cockpit_overrides.py" in source
    assert "--pr-number ${{ github.event.issue.number }}" in source
    assert "--comment-id ${{ github.event.comment.id }}" in source


def test_cockpit_runbook_documents_default_policy_and_curator_contract() -> None:
    source = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert "Auto-decisions are enabled by default" in source
    assert "--no-auto-decisions" in source
    assert "/cockpit override workstream" in source
    assert "/cockpit override surface" in source
    assert "/cockpit clear-override workstream" in source
    assert "/cockpit clear-override surface" in source
    assert "repository owner" in source
    assert "override-receipts.json" in source
    assert "nightly reconciliation" in source
    assert "never deletes branches" in source
