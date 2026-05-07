---
name: research-paper-to-mci-gru
description: Use when translating an academic finance paper, local PDF, extracted paper text, factor paper, anomaly paper, asset-pricing paper, portfolio paper, or risk-premium paper into an MCI-GRU-specific implementation brief and GitHub-ready issue drafts.
---

# Research Paper to MCI-GRU

Translate finance research into an MCI-GRU-aware Research-to-Implementation Brief. Produce a brief plus issue drafts; do not implement code or create GitHub issues unless the user explicitly asks after reviewing the drafts.

## Intake

If the user provides a local PDF path, run:

```bash
python skills/research-paper-to-mci-gru/scripts/extract_paper_text.py <paper.pdf> -o <paper-intake.md>
```

Use `--json` only when structured extraction is more useful than Markdown. If `pypdf` is unavailable, ask for pasted paper text or install/use an environment with `pypdf`; do not summarize from filename or title alone.

Before mapping the paper, read the repo anchors needed for the target surfaces:

- Always read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/CONFIGURATION_GUIDE.md`.
- For features, read `mci_gru/features/registry.py`.
- For graph ideas, read `mci_gru/graph/builder.py` and dynamic-graph notes in `AGENTS.md`.
- For evaluation/backtests, read `mci_gru/evaluation/` and `docs/BACKTEST_FAIRNESS_AUDIT.md`.
- For paper-trade ideas, read `paper_trade/scripts/infer.py` and `paper_trade/scripts/portfolio.py`.

Read `references/mci-gru-surfaces.md` for the surface taxonomy and invariant checklist. Read `references/conditional-skewness-example.md` when you need a calibrated example of higher-moment/coskewness translation.

## Workflow

1. Identify one to three Research Mechanisms. Separate each mechanism from empirical choices such as estimation windows, breakpoints, weighting, filters, robustness settings, and sample definitions.
2. Apply the Data Readiness Gate. Classify every required input as already available, derivable, external dependency, or unavailable. If required data is external or unavailable, block feature, graph, model, evaluation, and paper-trade slices until a data/provenance slice exists.
3. Rank MCI-GRU landing zones. For each mechanism, name primary, secondary, and rejected surfaces with repo evidence as `path: concept/function/config behavior`.
4. Give a Feasibility Opinion for every proposed slice: effort (`easy win`, `medium`, `long-term`), confidence (`high`, `medium`, `low`), one-sentence rationale, and main blocker (`data`, `code complexity`, `no-lookahead risk`, `validation cost`, or `production readiness`).
5. Draft at most four GitHub-ready Implementation Slices by default. Use categories from the taxonomy in `references/mci-gru-surfaces.md`.
6. Emit ADR candidates only when the choice is hard to reverse, surprising without context, and has real alternatives with trade-offs.

## Brief Format

Use exactly these top-level sections:

```md
# Research-to-Implementation Brief: <paper title>

## Intake
## Mechanisms
## Data Readiness Gate
## MCI-GRU Landing Zone Ranking
## Invariant Check
## Feasibility Opinion
## GitHub-Ready Slices
## ADR Candidates
## Rejected Ideas
## Open Questions
```

Each GitHub-ready slice must include:

- Category and title
- Problem
- Proposed scope
- Acceptance criteria
- Suggested tests
- Out of scope
- Feasibility Opinion

## Guardrails

- Draft issues by default; create issues only on explicit user request.
- Block missing-data ideas rather than inventing proxies.
- Treat paper-trade as rejected or long-term until offline validation exists.
- Do not treat research artifacts as source of truth when they conflict with no-lookahead, label embargo, dynamic graph timing, or backtest fairness.
- Do not produce a generic literature summary. Every claim in the mapping must connect to MCI-GRU surfaces, data, invariants, or issue drafts.
