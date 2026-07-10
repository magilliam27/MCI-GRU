# MCI-GRU Context

This context defines the repo-wide language agents should use when navigating
MCI-GRU research, experiments, implementation work, and documentation status.
The original research-translation language is still part of this glossary.

## Language

**Canonical Doc**:
A maintained document that agents can use as a current guide to repo behavior,
workflow, or invariants, while still checking current code when behavior matters.
_Avoid_: Any markdown file, historical note, stale plan

**Historical Reference**:
A retained document, plan, notebook, or tool note that can explain past intent
but must not override current code, canonical docs, or active evidence.
_Avoid_: Current acceptance criteria, live roadmap by default

**Source-of-Truth Drift**:
A disagreement between current code or invariants and older docs, notebooks,
handoffs, or plans. Agents should report this drift and only fix stale prose
when it directly affects navigation or active work.
_Avoid_: Silent doc cleanup, treating old plans as current requirements

**Current Research Evidence**:
A result report, audit, or evaluation summary that still informs active model,
data, validation, or experiment decisions.
_Avoid_: Any dated result file, raw artifact dump

**Superseded Research Evidence**:
A valid historical result report whose conclusion or recommendation has been
replaced by newer evidence.
_Avoid_: Invalid result, deleted evidence

**Research Archive**:
The repo location for superseded research summaries. Bulky artifacts, raw run
outputs, and checkpoints should remain in Drive or external storage and be cited
from the summary.
_Avoid_: Results folder, checkpoint storage

**Handoff**:
An operational continuity note that helps another agent resume work, including
state, blockers, commands, and next steps. A handoff is not research evidence
unless a current report cites it as provenance.
_Avoid_: Result report, canonical doc

**Workstream Decision Registry**:
The versioned cockpit input that preserves reviewed workstream statuses,
canonical continuation surfaces, rationales, next actions, and git-surface
dispositions across generated daily refreshes.
_Avoid_: Manual edit to the generated workstream register, branch-name ignore list

**Canonical Continuation Surface**:
The reviewed issue, PR, branch, worktree, or `main` plus canonical doc from which
a workstream should continue. Historical matching branches do not compete with
it after they are recorded in the Workstream Decision Registry.
_Avoid_: Latest branch by name, arbitrary matching worktree

**Reviewed Git Surface**:
A normalized branch or detached-worktree label whose relationship to one or
more workstreams and whose canonical, parked, archive, or stale disposition is
recorded in the Workstream Decision Registry.
_Avoid_: Silently ignored branch, deleted branch

**Unreviewed Git Surface**:
A live topology surface that matches a workstream but is absent from that
workstream's registry assignments. It reopens only the affected workstream for
review without invalidating unrelated recorded decisions.
_Avoid_: Every historical branch, automatically canonical branch

**Research Mechanism**:
A paper's durable economic or statistical idea that may be transferable into MCI-GRU, limited to at most three per paper.
_Avoid_: Result, technique, implementation detail

**Empirical Choice**:
A paper-specific estimation, filtering, ranking, weighting, or robustness decision used to test a research mechanism.
_Avoid_: Core idea, model contribution

**Landing Zone Ranking**:
An ordered recommendation of MCI-GRU surfaces for a research mechanism, including primary, secondary, and rejected targets.
_Avoid_: Neutral mapping list, architecture tour

**Feasibility Opinion**:
A concrete judgment about whether an implementation slice is an easy win, medium effort, or long-term bet based on data, code, invariant, validation, and production risks.
_Avoid_: Vague priority, interesting idea

**Data Readiness Gate**:
A blocking check that classifies required inputs as already available, derivable, external dependency, or unavailable before implementation slices proceed.
_Avoid_: Proxy by default, assume data exists

**Paper Intake Helper**:
A small script bundled with the research translation skill that extracts text and basic metadata from a local academic PDF without summarizing it.
_Avoid_: Paper summarizer, implementation planner

**Paper Intake Artifact**:
A Markdown-first extraction artifact containing source details, inferred metadata, warnings, and full paper text, with optional JSON output for structured use.
_Avoid_: Raw dump only, hidden preprocessing

**Versioned Skill Artifact**:
A reusable Codex skill stored under the repository root `skills/` directory so it can be reviewed, committed, pushed, and installed separately.
_Avoid_: Local-only skill, hidden prompt

**MCI-GRU-Aware Translation**:
A research translation workflow grounded in this repo's architecture, configuration, evaluation, graph, feature, and paper-trade constraints rather than generic paper summarization.
_Avoid_: Generic finance-paper summary

**Research-to-Implementation Brief**:
A paper-grounded planning artifact that explains a finance paper's mechanism, maps it to MCI-GRU surfaces, checks project invariants, and proposes GitHub-ready implementation slices.
_Avoid_: Paper summary, generic literature review

**Implementation Slice**:
A bounded GitHub-ready work item derived from a research paper, classified by target surface such as feature, experiment, notebook, architecture, evaluation, or ADR.
_Avoid_: Direct implementation, loose idea

**Long-History Preset**:
A Hydra experiment preset that changes `model.his_t` to test longer temporal context while preserving the chosen recipe semantics.
_Avoid_: Foundation model experiment, architecture replacement

**Mechanics Smoke**:
A deliberately cheap run that proves wiring, config composition, data alignment, and artifact creation without serving as model-performance evidence.
_Avoid_: Confirmation run, evidence run

**Gated Long-Window Candidate**:
A high-cost history length, such as `his_t=252`, documented for later evaluation after shorter long-history presets pass memory and runtime checks.
_Avoid_: Default preset, mandatory sweep member

**Anchored Historical Snapshot Universe**:
A non-PIT market CSV built from an older S&P 500 snapshot universe, such as `sp500_2019_universe_data_through_2026.csv`, that may be used for mechanics validation but not headline performance evidence.
_Avoid_: PIT data, clean universe, tainted universe data, current-universe panel

**Colab Evaluation Notebook**:
A generated notebook launcher that runs a resumable explicit experiment matrix on Colab, stages Drive data into local `/content`, and exports compact manifests, results, logs, and summaries back to Drive.
_Avoid_: Hand-edited notebook, local-only runner

**Slice Category**:
One of data, feature, graph, model, training/evaluation, config/experiment, notebook, paper-trade, or ADR.
_Avoid_: Miscellaneous, uncategorized task

**Volatility-Targeting Feature Family**:
Stock-level model input features derived from volatility-targeting research,
such as exponentially weighted volatility, clipped inverse-volatility exposure
proxies, volatility persistence, leverage-effect context, and
momentum-volatility interactions. These features let the model observe
volatility-scaling mechanisms without changing portfolio sizing. The first
implementation should use Harvey-style ex ante timing: volatility-targeting
signals are deliberately lagged so they are known before the modeled forward
return, with tests proving future rows cannot affect earlier feature values.
Chapter-aligned defaults are a 10% annual target volatility and EWM daily-return
volatility estimates, with 20/60/90 trading-day half-life variants when the
family is enabled. The first target-vol exposure proxy should be a clipped
multiplier, `target_vol / estimated_annual_vol`, bounded to `[0.25, 4.0]`;
this is an input-stability guardrail, not a sigma interval.
_Avoid_: Portfolio volatility targeting, exposure cap, rank-gate rule

**Portfolio Volatility Targeting**:
An execution or portfolio-construction rule that scales actual portfolio
notional exposure toward a volatility target. This belongs in evaluation or
paper-trade surfaces, not in issue #8's first feature-family implementation.
_Avoid_: Feature column, model input, issue #8 first slice

**Portfolio-IC Hybrid Loss**:
A training/evaluation loss that keeps cross-sectional IC as the anchor while
adding a differentiable soft top-k forward-return utility term. The first
implementation optimizes same-date PIT-valid 5-day forward-return labels through
a per-date soft top-k surrogate. It is not direct open-to-open PnL, not
transaction-cost-aware sequential training, and not a full end-to-end Markowitz
optimizer.
_Avoid_: Full portfolio optimizer, Sharpe-loss trainer, turnover-aware trainer

**ADR Candidate**:
A possible architecture decision note emitted only when a paper-derived choice is hard to reverse, surprising without context, and involves real trade-offs.
_Avoid_: Default ADR, design note for every idea

**Calibration Example**:
A compact worked example bundled with the skill to show the expected translation style without becoming a full paper summary.
_Avoid_: Full reproduction, hidden acceptance criteria

**Repo Evidence**:
A path plus the relevant concept, function, or config behavior used to justify a paper-to-MCI-GRU mapping, with exact line numbers only when useful.
_Avoid_: Unsupported mapping, brittle citation-only proof

**Issue Draft**:
A GitHub-ready Markdown work item that is produced by default but is not created on GitHub unless the user explicitly requests issue creation.
_Avoid_: Auto-created issue, hidden tracker mutation

## Relationships

- A **Canonical Doc** should reflect current repo behavior, but current code and
  the invariants in `AGENTS.md` still win when implementation behavior matters.
- A **Historical Reference** may explain past intent, but contradictions with
  current code are **Source-of-Truth Drift**.
- **Current Research Evidence** can support active decisions until it is
  replaced by stronger or newer evidence.
- **Superseded Research Evidence** belongs in the **Research Archive** after a
  report-by-report review.
- A **Handoff** preserves operational continuity and can provide provenance, but
  it is not research evidence by default.
- A **Workstream Decision Registry** preserves a **Canonical Continuation
  Surface** across generated cockpit refreshes.
- A **Reviewed Git Surface** cannot reopen its recorded workstream merely by
  remaining in git topology; an **Unreviewed Git Surface** can.
- A paper can contribute one to three **Research Mechanisms**.
- A **Research Mechanism** can have many **Empirical Choices**.
- A **Research-to-Implementation Brief** uses stable sections: Intake, Mechanisms, Data Readiness Gate, Landing Zone Ranking, Invariant Check, Feasibility Opinion, GitHub-Ready Slices, ADR Candidates, Rejected Ideas, and Open Questions.
- A **Research-to-Implementation Brief** cites **Repo Evidence** for major MCI-GRU mappings.
- A **Research Mechanism** gets a **Landing Zone Ranking** before implementation slices are proposed.
- A **Research Mechanism** passes the **Data Readiness Gate** before feature, graph, model, paper-trade, or evaluation slices proceed.
- A **Paper Intake Helper** produces a **Paper Intake Artifact** that can feed a **Research-to-Implementation Brief**, but it does not interpret or rank the paper.
- A **Versioned Skill Artifact** can be uploaded to GitHub from the repo and installed into Codex separately.
- A **Research-to-Implementation Brief** is an **MCI-GRU-Aware Translation**, not a generic summary.
- A **Calibration Example** may be loaded when the agent needs an example of the expected brief style.
- A **Research-to-Implementation Brief** proposes at most four **Implementation Slices** as **Issue Drafts** by default and assigns each one **Slice Category**.
- An **ADR Candidate** is proposed only when all ADR conditions are met.
- An **Implementation Slice** includes a **Feasibility Opinion** with effort, confidence, rationale, and main blocker.
- A **Research-to-Implementation Brief** contains one or more **Implementation Slices**.
- An **Implementation Slice** can become a GitHub issue after review, but the brief does not authorize direct code changes by itself.
- A **Long-History Preset** can be checked by a **Mechanics Smoke**, but only a full confirmation run can support model-performance claims.
- A **Gated Long-Window Candidate** should not become a first-pass **Long-History Preset** until cheaper presets establish acceptable memory and runtime behavior.
- A **Long-History Preset** should preserve the selected frozen recipe semantics and vary temporal history length as the intended experimental factor.
- A first-pass **Long-History Preset** should not add temporal-encoder comparisons; encoder variants belong in a separate follow-up slice after the history-length path is proven.
- An **Anchored Historical Snapshot Universe** can support a **Mechanics Smoke**, but full long-history evaluation must use true PIT masked-panel data.
- Full long-history performance evidence should aggregate PIT masked-panel evaluation across 2022, 2023, 2024, and 2025 rather than rely on a single test year.
- Full long-history PIT evaluation should be launched through a **Colab Evaluation Notebook** generated from a repo script.
- A long-history **Colab Evaluation Notebook** should include the `his_t=10` frozen-default baseline alongside `21`, `63`, and `126`, with `252` gated manually.
- Long-history decision tables should show per-year rows and grouped `his_t` aggregates; a decision score may sort candidates but cannot replace the underlying IC, Sharpe, return, drawdown, turnover, and failure-rate evidence.
- A **Portfolio-IC Hybrid Loss** can be checked by a **Mechanics Smoke**, but
  performance judgment requires PIT portfolio evidence, including Sharpe,
  drawdown, turnover, and transaction-cost-sensitive comparisons after training.

## Example Dialogue

> **Dev:** "Should the skewness paper go straight into a feature branch?"
> **Domain expert:** "No. First write a **Research-to-Implementation Brief** and split it into **Implementation Slices** we can review as issues."

> **Dev:** "The one-epoch **Mechanics Smoke** passed for a 21-day **Long-History Preset**. Can we call long history better?"
> **Domain expert:** "No. The smoke proves wiring; performance claims require a confirmation run."

## Flagged Ambiguities

- "Paper summary" is too broad for this workflow. The resolved term is **Research-to-Implementation Brief**, which must map the paper to MCI-GRU and its invariants.
- "Core mechanism" should not imply exactly one idea. The resolved term is **Research Mechanism**, with a cap of three per paper.
- "Smoke" was ambiguous between mechanics validation and reduced-budget performance evidence. The resolved term is **Mechanics Smoke** for cheap wiring checks only.
- "Long-history experiment" can be ambiguous between a broad config drift and a controlled ablation. Resolved: use **Long-History Preset** for recipe-preserving `model.his_t` changes.
- "Tainted universe data" and "current-universe panel" are imprecise for the non-PIT temporal CSVs. The resolved term is **Anchored Historical Snapshot Universe**, and it must not be used for headline performance evidence.
