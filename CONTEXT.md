# MCI-GRU Research Translation

This context defines the project language for turning finance research papers into MCI-GRU implementation work without skipping architectural review.

## Language

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
