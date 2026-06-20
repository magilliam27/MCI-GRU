# MCI-GRU Cockpit Runbook

The cockpit refresh writes the repo-local operating picture:

- `docs/agents/workstreams.md`
- `docs/agents/cockpit/YYYY-MM-DD.md`

Manual local refresh:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20
```

GitHub sync is disabled by default. Until the guarded GitHub gateway is
implemented and reviewed, the runner does not push branches, open PRs, close
issues, label issues, or comment on the cockpit issue.

Before committing cockpit output, inspect:

```bash
git status --short
git diff -- docs/agents/workstreams.md docs/agents/cockpit/
```

Only cockpit files should be staged for a cockpit refresh.
