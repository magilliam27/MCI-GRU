# Domain And Source-Of-Truth Policy

MCI-GRU is a single-context repository. The repo-wide glossary lives in
`CONTEXT.md`; there is no `CONTEXT-MAP.md` today.

## Source-Of-Truth Hierarchy

When sources disagree, prefer them in this order:

1. Current code and tests for implemented behavior.
2. Repository invariants in `AGENTS.md`.
3. Canonical docs such as `docs/ARCHITECTURE.md`,
   `docs/CONFIGURATION_GUIDE.md`, `docs/DEFAULT_EXPERIMENT_RECIPE.md`,
   `docs/TESTING_GUIDE.md`, and focused data contracts.
4. Current research evidence listed from `docs/research/README.md`.
5. Handoffs and historical references.

If a historical plan, handoff, or notebook contradicts current code or canonical
invariants, treat it as source-of-truth drift. Report the drift, and only fix
the stale prose when it directly affects navigation, active agent behavior, or
the task at hand.

## Agent Working Rules

- Keep `AGENTS.md` short; put durable agent policy in this directory.
- Use `docs/index.md` for the doc map and `docs/research/README.md` for
  research evidence status.
- Treat handoffs as operational continuity, not research evidence, unless a
  current report cites a handoff as provenance.
- Treat generated notebooks as launchers for specific experiments. Generator
  scripts are support and repair aids unless a current workflow or test
  explicitly promotes them as the source of truth.
