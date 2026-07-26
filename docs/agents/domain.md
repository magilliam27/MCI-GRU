# Domain And Source-Of-Truth Policy

MCI-GRU is a single-context repository. The repo-wide glossary lives in
root `CONTEXT.md`; there is no `CONTEXT-MAP.md` today. Repository-wide
architecture decisions live under `docs/adr/` when they are created.

## Before Exploring

- Read root `CONTEXT.md` before naming or changing domain concepts.
- Read ADRs under `docs/adr/` that touch the area about to be explored or
  changed.
- If either path does not exist, proceed silently. Do not create it just to
  complete setup or suggest speculative documentation upfront.

The `/domain-modeling` workflow creates or updates domain documentation lazily
when terminology or a durable decision is actually resolved.

## Source-Of-Truth Hierarchy

When sources disagree, prefer them in this order:

1. Current code and tests for implemented behavior.
2. Repository invariants in `AGENTS.md`.
3. Canonical docs such as `docs/agents/guide.md`, `docs/ARCHITECTURE.md`,
   `docs/CONFIGURATION_GUIDE.md`, `docs/DEFAULT_EXPERIMENT_RECIPE.md`,
   `docs/TESTING_GUIDE.md`, and focused data contracts.
4. Current research evidence listed from `docs/research/README.md`.
5. Handoffs and historical references.

If a historical plan, handoff, or notebook contradicts current code or canonical
invariants, treat it as source-of-truth drift. Report the drift, and only fix
the stale prose when it directly affects navigation, active agent behavior, or
the task at hand.

## Use the Glossary's Vocabulary

When output names a domain concept in an issue title, implementation plan,
refactor proposal, hypothesis, test, or documentation change, use the term as
defined in `CONTEXT.md`. Do not drift to a synonym that the glossary explicitly
marks to avoid.

If the needed concept is missing, reconsider whether the proposed language
belongs to the project. If the gap is real, record it for domain modelling
rather than silently inventing a competing term.

## Flag ADR Conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly
instead of silently overriding the decision. Identify the ADR and explain why
reopening it may be warranted.

## Agent Working Rules

- Keep `AGENTS.md` short; put durable agent policy in this directory.
- Use `guide.md` for current-state repository navigation and engineering
  constraints. Use `target-architecture.md` only for explicit, human-reviewed
  future-state decisions; it is never evidence of current behaviour.
- Use `docs/index.md` for the doc map and `docs/research/README.md` for
  research evidence status.
- Treat handoffs as operational continuity, not research evidence, unless a
  current report cites a handoff as provenance.
- Treat generated notebooks as launchers for specific experiments. Generator
  scripts are support and repair aids unless a current workflow or test
  explicitly promotes them as the source of truth.

## File Layout

The single-context layout is:

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/        # created lazily when a durable decision is resolved
└── mci_gru/
```

Do not create `docs/adr/` solely for repository setup.
