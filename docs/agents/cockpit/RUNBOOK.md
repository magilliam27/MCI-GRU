# MCI-GRU Cockpit Runbook

The cockpit refresh writes the repo-local operating picture:

- `docs/agents/workstreams.md`
- `docs/agents/cockpit/YYYY-MM-DD.md`

Manual local refresh:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20
```

GitHub sync is disabled by default. Live sync requires `--github-sync`:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20 --github-sync
```

Before running live sync, confirm GitHub CLI authentication:

```bash
gh auth status
```

Live sync uses a dated branch named `codex/cockpit-refresh-YYYYMMDD`, commits
only cockpit files, pushes the branch, opens or reuses `Cockpit refresh:
YYYY-MM-DD`, creates or reuses the `MCI-GRU Cockpit` issue, and comments there
with the run color, PR link, and decision queue.

Labels are applied only when they already exist in GitHub. Missing labels are
reported as skipped actions instead of being created with near-duplicate names.
Safe labels include `cockpit-reviewed`, `ready-for-agent`, `needs-info`, and
`needs-triage` when present.

Issue closure must go through `close_issue_with_evidence`, which comments with
the closure evidence before closing the issue. Do not close ambiguous research,
experiment, or data-access issues without clear evidence such as a merged PR,
existing code/tests/docs, a linked duplicate, or an explicit user decision.

Before committing cockpit output, inspect:

```bash
git status --short
git diff -- docs/agents/workstreams.md docs/agents/cockpit/
```

Only cockpit files should be staged for a cockpit refresh.

If live sync goes sideways, inspect and recover with:

```bash
git status --short
git branch --show-current
gh pr list --head codex/cockpit-refresh-YYYYMMDD
gh issue list --search "MCI-GRU Cockpit in:title"
```
